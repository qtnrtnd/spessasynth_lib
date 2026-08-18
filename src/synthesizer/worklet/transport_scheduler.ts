// In-worklet transport scheduler with sample-accurate sub-block dispatch (openDAW pattern).
//
// Per render quantum (128 samples), the scheduler:
//   1. Gathers MIDI events (noteOn / noteOff / loopEnd) whose pulse falls in this block.
//   2. Sorts them by (sampleOffset, eventTypeRank) so noteOffs always come before noteOns
//      At the same pulse — eliminates the FP-ordering hazard that caused back-to-back
//      Same-pitch notes to silence each other.
//   3. Walks the event list, calling synth.processSplit() for the audio range BETWEEN
//      Events, and synth.processMessage(..., { time: 0 }) (which bypasses the eventQueue
//      And invokes processMessageInternal directly) AT each event boundary.
//
// All scheduling is in pulse space (integers), so consecutive events at the same musical
// Position have literally identical sample offsets — no rounding ambiguity. The walker's
// Deterministic sort then guarantees the release-before-attack order.
//
// `overlayNotes` (engine:previewOverlay) rides the same per-quantum gathering and sample-
// Accurate dispatch as `notes`, for a one-shot preview bar overlaid on the live composition —
// The one difference is it is never re-populated on a loop wrap, so it plays exactly once.

import { MIDIMessageTypes, type SpessaSynthProcessor } from "spessasynth_core";

// PPQN math (inlined; matches client's engine/ppqn/PPQN.ts)
const Quarter = 960;
const Bar = Quarter << 2;
const samplesToPulses = (n: number, bpm: number, sr: number) =>
    ((n / sr) * bpm * Quarter) / 60;
// The bar already underway, not the next one — a preview overlay joins in
// Progress instead of waiting up to a full bar to start (see
// "engine:previewOverlay" below). handleCommand and renderBlock run on the
// Same audio-thread tick, so there's no skew between the position read here
// And the p0 renderBlock uses next.
const barStartPulse = (pos: number): number => Math.floor(pos / Bar) * Bar;

// SAB layout (matches client's engine/sync/EngineStateSchema.ts)
const STATE_FLAG_OFFSET = 0;
const POSITION_OFFSET = 8;
const TIMESTAMP_OFFSET = 16;
const IS_PLAYING_OFFSET = 24;
const BPM_OFFSET = 28;
const LOOP_COUNT_OFFSET = 32;
const STATE_READ = 1;
const STATE_WRITING = 2;
const STATE_WRITTEN = 3;

const RENDER_QUANTUM = 128;
const noteOnByte = (channel: number) =>
    MIDIMessageTypes.noteOn | (channel & 0x0f);
const noteOffByte = (channel: number) =>
    MIDIMessageTypes.noteOff | (channel & 0x0f);

// Message types (sent by EngineHost on the main thread)
export interface WorkletNote {
    id: number;
    pulse: number;
    durationPulses: number;
    midi: number;
    vel: number;
    channel: number;
}

// A one-shot preview hit (drumkit audition, previewer.ts), relative to the
// Pattern's own start — the worklet anchors it to a real pulse itself
// (barStartPulse against its own live `position`), never a main-thread one.
// A hit whose offset already lies behind the current position when the
// Command lands is dropped, not replayed late — the bar joins in progress.
export interface OverlayHit {
    offsetPulses: number;
    durationPulses: number;
    midi: number;
    vel: number;
    channel: number;
}

export type EngineCommand =
    | { type: "engine:play"; fromPulse?: number }
    | { type: "engine:pause"; rewindToPulse?: number }
    | { type: "engine:stop" }
    | { type: "engine:seek"; pulse: number }
    | { type: "engine:updateNotes"; notes: WorkletNote[] }
    | { type: "engine:updateBpm"; bpm: number }
    | {
          type: "engine:updateLoop";
          loopStart: number;
          loopEnd: number;
      }
    | { type: "engine:previewOverlay"; hits: OverlayHit[] };

// Event type ranks for sort. Lower rank = processed first at equal sampleOffset.
const TYPE_NOTE_OFF = 0;
const TYPE_LOOP_END = 1;
const TYPE_NOTE_ON = 2;

interface ScheduledEvent {
    sampleOffset: number;
    typeRank: number;
    // For note events:
    midi: number;
    vel: number;
    channel: number;
}

// An OverlayHit anchored to a real pulse — same shape, no `id`: overlay hits
// are never diffed against a previous list, only ever replaced wholesale.
interface OverlayNote {
    pulse: number;
    durationPulses: number;
    midi: number;
    vel: number;
    channel: number;
}

export class TransportScheduler {
    private bpm = 120;
    private position = 0;
    private isPlaying = false;
    private loopStart = 0;
    private loopEnd = Bar;
    private notes: WorkletNote[] = [];
    // Notes that were removed from `notes` while their voice was still playing.
    // They're kept around so their original noteOff fires at the planned end pulse —
    // The user's mental model: deleting a held note should let it ring to its end,
    // Not silence it abruptly. Cleared on pause/stop/seek/loop-wrap.
    private phantomNotes: WorkletNote[] = [];
    // One-shot preview hits (previewer.ts's drumkit audition), overlaid on top
    // of the live composition. Never re-populated on loop wrap, unlike `notes`
    // — each entry is spliced out the moment its noteOff fires, so a bar plays
    // exactly once. See clearOverlay.
    private overlayNotes: OverlayNote[] = [];
    private loopCount = 0;
    private readonly flag: Uint8Array;
    private readonly view: DataView;
    private readonly oneOutput: boolean;
    // Reusable scratch buffer to avoid per-quantum allocation.
    private readonly events: ScheduledEvent[] = [];

    public constructor(sab: SharedArrayBuffer, oneOutput: boolean) {
        this.flag = new Uint8Array(sab);
        this.view = new DataView(sab);
        this.oneOutput = oneOutput;
        Atomics.store(this.flag, STATE_FLAG_OFFSET, STATE_READ);
    }

    /** Force-releases and drops every pending overlay hit. A noteOff for a hit
     *  that never actually sounded is a harmless no-op (same treatment
     *  phantomNotes gets on wrap) — cheaper than tracking which ones fired. */
    private clearOverlay(synth: SpessaSynthProcessor): void {
        for (const n of this.overlayNotes) {
            synth.processMessage([noteOffByte(n.channel), n.midi], 0, {
                time: 0
            });
        }
        this.overlayNotes.length = 0;
    }

    public handleCommand(
        cmd: EngineCommand,
        synth: SpessaSynthProcessor
    ): void {
        switch (cmd.type) {
            case "engine:play": {
                if (cmd.fromPulse !== undefined) this.position = cmd.fromPulse;
                // A position left outside the loop by an edit made at rest
                // (see the isPlaying guard in "engine:updateLoop" below) would
                // otherwise run forward forever once playback starts, since the
                // wrap check further down only fires while already inside the
                // loop — so reclaim it into the loop right here, at the moment
                // playback actually begins.
                if (this.position < this.loopStart || this.position >= this.loopEnd) {
                    this.position = this.loopStart;
                }
                this.isPlaying = true;
                // Re-trigger notes whose pulse range straddles the resume position
                // So a paused-mid-sustain resumes audibly instead of silent until the
                // Next noteOn. Their noteOffs will be picked up naturally when
                // (note.pulse + duration) falls into a future block's [p0, p1) range.
                for (const n of this.notes) {
                    const noteEnd = n.pulse + n.durationPulses;
                    if (n.pulse < this.position && noteEnd > this.position) {
                        synth.processMessage(
                            [noteOnByte(n.channel), n.midi, n.vel],
                            0,
                            { time: 0 }
                        );
                    }
                }
                return;
            }
            case "engine:pause": {
                // Rewind to the audible position so the displayed playhead doesn't jump
                // Forward when the host switches off interpolation.
                if (cmd.rewindToPulse !== undefined)
                    this.position = cmd.rewindToPulse;
                this.isPlaying = false;
                this.phantomNotes.length = 0;
                this.clearOverlay(synth);
                synth.stopAllChannels(false);
                return;
            }
            case "engine:stop": {
                this.isPlaying = false;
                this.position = this.loopStart;
                this.loopCount = 0;
                this.phantomNotes.length = 0;
                this.clearOverlay(synth);
                synth.stopAllChannels(false);
                return;
            }
            case "engine:seek": {
                // Seek is a teleport — kill voices abruptly so the old position's audio
                // Doesn't bleed into the new position.
                this.position = cmd.pulse;
                this.phantomNotes.length = 0;
                this.clearOverlay(synth);
                synth.stopAllChannels(true);
                return;
            }
            case "engine:updateNotes": {
                // While stopped/paused, nothing is sounding — just swap the list.
                if (!this.isPlaying) {
                    this.notes = cmd.notes;
                    return;
                }
                // While playing, diff old vs new note lists by id so we can detect
                // True removals vs additions (modifications keep the same id).
                //
                //   - Removed AND currently held: park as a "phantom" so its planned
                //     NoteOff still fires at the original end pulse — i.e. the
                //     Deleted note rings to its natural end instead of cutting off.
                //   - Added AND currently held: fire noteOn immediately so the user
                //     Hears the just-drawn note right away.
                const pos = this.position;
                const oldById = new Map<number, WorkletNote>();
                for (const n of this.notes) oldById.set(n.id, n);
                const newIds = new Set<number>();
                for (const n of cmd.notes) newIds.add(n.id);

                for (const [id, n] of oldById) {
                    if (newIds.has(id)) continue;
                    if (n.pulse <= pos && n.pulse + n.durationPulses > pos) {
                        this.phantomNotes.push(n);
                    }
                }

                this.notes = cmd.notes;

                for (const n of cmd.notes) {
                    if (oldById.has(n.id)) continue;
                    if (n.pulse <= pos && n.pulse + n.durationPulses > pos) {
                        synth.processMessage(
                            [noteOnByte(n.channel), n.midi, n.vel],
                            0,
                            { time: 0 }
                        );
                    }
                }
                return;
            }
            case "engine:updateBpm": {
                this.bpm = cmd.bpm;
                return;
            }
            case "engine:updateLoop": {
                this.loopStart = cmd.loopStart;
                this.loopEnd = cmd.loopEnd;
                // Only reclaim the transport into the (possibly shrunk) loop
                // while it's actually advancing. At rest (paused/stopped) the
                // composer is just editing notes — the displayed playhead must
                // stay exactly where they left it, even past the new loop end;
                // "engine:play" reclaims it if needed once playback resumes.
                if (this.isPlaying && this.position >= this.loopEnd) {
                    this.position = this.loopStart;
                    this.loopCount++;
                    this.phantomNotes.length = 0;
                    synth.stopAllChannels(true);
                }
                return;
            }
            case "engine:previewOverlay": {
                // A fresh preview always replaces, never stacks with, the last
                // one — cancel whatever is still pending first.
                this.clearOverlay(synth);
                // Dropped, not queued: if playback isn't actually running when
                // This lands (a main-thread/worklet race around a pause), "the
                // Bar underway" means nothing, and installing stale hits would
                // Leak into a later, unrelated resume.
                if (!this.isPlaying || cmd.hits.length === 0) return;
                // Anchor to the bar already underway, not the next one, and
                // Join it in progress: a hit whose offset is already behind
                // The current position is dropped rather than replayed late,
                // So triggering mid-bar plays only what is still ahead.
                const anchor = barStartPulse(this.position);
                const elapsed = this.position - anchor;
                this.overlayNotes = cmd.hits
                    .filter((h) => h.offsetPulses >= elapsed)
                    .map((h) => ({
                        pulse: anchor + h.offsetPulses,
                        durationPulses: h.durationPulses,
                        midi: h.midi,
                        vel: h.vel,
                        channel: h.channel
                    }));
                return;
            }
        }
    }

    /**
     * Drives the audio render for one quantum, splitting it into sub-blocks at event
     * boundaries (sample-accurate dispatch). Always writes RENDER_QUANTUM samples total.
     */
    public renderBlock(
        currentTime: number,
        sampleRate: number,
        synth: SpessaSynthProcessor,
        outputs: Float32Array[][]
    ): void {
        if (!this.isPlaying) {
            this.renderSub(synth, outputs, 0, RENDER_QUANTUM);
            this.publishState(currentTime, sampleRate);
            return;
        }

        let p0 = this.position;
        let cursorSample = 0;

        // Outer loop handles loop wraps: each iteration renders a contiguous pulse range
        // [p0, segmentEnd) within the same iteration. If we wrap, we re-enter to render
        // The remainder of the quantum starting at loopStart.
        while (cursorSample < RENDER_QUANTUM) {
            const remainingSamples = RENDER_QUANTUM - cursorSample;
            const remainingPulses = samplesToPulses(
                remainingSamples,
                this.bpm,
                sampleRate
            );
            const segmentP1 = p0 + remainingPulses;
            const wrapsInSegment =
                segmentP1 > this.loopEnd && this.loopEnd > p0;
            const segmentEnd = wrapsInSegment ? this.loopEnd : segmentP1;
            const endOfSegmentSample = cursorSample + remainingSamples;

            // Gather events whose pulse falls in [p0, segmentEnd) and compute their
            // Sample offset relative to the quantum start.
            this.events.length = 0;
            const pulseSpan = segmentEnd - p0;
            const sampleSpan = remainingSamples;
            const cursorAtSegmentStart = cursorSample;
            const pulseToSampleInSegment = (pulse: number): number => {
                if (pulseSpan <= 0) return cursorAtSegmentStart;
                const local = (pulse - p0) / pulseSpan; // 0..1 within this segment
                return cursorAtSegmentStart + Math.round(local * sampleSpan);
            };

            for (const n of this.notes) {
                // NoteOn interval: [p0, segmentEnd). A note starts in exactly one segment
                // (the one whose [p0, segmentEnd) contains its pulse).
                if (n.pulse >= p0 && n.pulse < segmentEnd) {
                    this.events.push({
                        sampleOffset: pulseToSampleInSegment(n.pulse),
                        typeRank: TYPE_NOTE_ON,
                        midi: n.midi,
                        vel: n.vel,
                        channel: n.channel
                    });
                }
                // NoteOff interval: (p0, segmentEnd]. Inclusive on the upper end so a
                // Note ending exactly at loopEnd is released *before* the wrap event
                // (rather than slipping through the boundary and lingering forever).
                const noteEnd = n.pulse + n.durationPulses;
                if (noteEnd > p0 && noteEnd <= segmentEnd) {
                    this.events.push({
                        sampleOffset: pulseToSampleInSegment(noteEnd),
                        typeRank: TYPE_NOTE_OFF,
                        midi: n.midi,
                        vel: 0,
                        channel: n.channel
                    });
                }
            }
            // Phantom noteOffs: deleted-while-playing notes whose voice is still alive.
            // Same interval rule as live noteOffs. Iterate in reverse so splice() is safe.
            for (let i = this.phantomNotes.length - 1; i >= 0; i--) {
                const p = this.phantomNotes[i];
                const phantomEnd = p.pulse + p.durationPulses;
                if (phantomEnd > p0 && phantomEnd <= segmentEnd) {
                    this.events.push({
                        sampleOffset: pulseToSampleInSegment(phantomEnd),
                        typeRank: TYPE_NOTE_OFF,
                        midi: p.midi,
                        vel: 0,
                        channel: p.channel
                    });
                    this.phantomNotes.splice(i, 1);
                }
            }
            // Preview overlay: one-shot hits, not part of the loop — same
            // interval rule as `notes`, but spliced out the moment their
            // noteOff fires so they never repeat on the next pass.
            for (let i = this.overlayNotes.length - 1; i >= 0; i--) {
                const n = this.overlayNotes[i];
                if (n.pulse >= p0 && n.pulse < segmentEnd) {
                    this.events.push({
                        sampleOffset: pulseToSampleInSegment(n.pulse),
                        typeRank: TYPE_NOTE_ON,
                        midi: n.midi,
                        vel: n.vel,
                        channel: n.channel
                    });
                }
                const overlayEnd = n.pulse + n.durationPulses;
                if (overlayEnd > p0 && overlayEnd <= segmentEnd) {
                    this.events.push({
                        sampleOffset: pulseToSampleInSegment(overlayEnd),
                        typeRank: TYPE_NOTE_OFF,
                        midi: n.midi,
                        vel: 0,
                        channel: n.channel
                    });
                    this.overlayNotes.splice(i, 1);
                }
            }
            if (wrapsInSegment) {
                this.events.push({
                    sampleOffset: pulseToSampleInSegment(this.loopEnd),
                    typeRank: TYPE_LOOP_END,
                    midi: 0,
                    vel: 0,
                    channel: 0
                });
            }

            // Sort: by sampleOffset, then by typeRank (noteOff < loopEnd < noteOn).
            this.events.sort(
                (a, b) =>
                    a.sampleOffset - b.sampleOffset || a.typeRank - b.typeRank
            );

            // Walk events, rendering sub-blocks and firing events.
            for (const e of this.events) {
                const clampedOffset = Math.min(
                    Math.max(e.sampleOffset, cursorSample),
                    endOfSegmentSample
                );
                if (clampedOffset > cursorSample) {
                    this.renderSub(
                        synth,
                        outputs,
                        cursorSample,
                        clampedOffset - cursorSample
                    );
                    cursorSample = clampedOffset;
                }
                if (e.typeRank === TYPE_NOTE_ON) {
                    synth.processMessage(
                        [noteOnByte(e.channel), e.midi, e.vel],
                        0,
                        {
                            time: 0
                        }
                    );
                } else if (e.typeRank === TYPE_NOTE_OFF) {
                    synth.processMessage([noteOffByte(e.channel), e.midi], 0, {
                        time: 0
                    });
                }
                // LOOP_END is handled by exiting the inner walk; the outer while
                // Loop will re-enter for the wrapped remainder.
                if (e.typeRank === TYPE_LOOP_END) break;
            }

            // Render any remaining audio of this segment AFTER the last event.
            if (cursorSample < endOfSegmentSample) {
                this.renderSub(
                    synth,
                    outputs,
                    cursorSample,
                    endOfSegmentSample - cursorSample
                );
                cursorSample = endOfSegmentSample;
            }

            // Advance position. If wrapping, reset to loopStart for the next iteration.
            if (wrapsInSegment) {
                p0 = this.loopStart;
                this.loopCount++;
                // No stopAllChannels — preserve "no kill on wrap" semantics so notes
                // Crossing the loop boundary release naturally.
                // Any phantom whose planned end was *past* loopEnd didn't fire above;
                // Release it now so it doesn't leak into the next iteration.
                for (const p of this.phantomNotes) {
                    synth.processMessage([noteOffByte(p.channel), p.midi], 0, {
                        time: 0
                    });
                }
                this.phantomNotes.length = 0;
                // A preview bar shorter than the loop it was overlaid on would
                // already be spliced out above; one that outlasts the loop's
                // wrap is the accepted residual (soundfonts.md §7.4) — force
                // it off rather than let it hang or bleed into the next pass.
                this.clearOverlay(synth);
            } else {
                p0 = segmentEnd;
            }
        }

        this.position = p0;
        this.publishState(currentTime, sampleRate);
    }

    /**
     * Render `samples` audio samples starting at `startIndex` in the output buffers,
     * driving SpessaSynth's `processSplit` for the active output mode.
     */
    private renderSub(
        synth: SpessaSynthProcessor,
        outputs: Float32Array[][],
        startIndex: number,
        samples: number
    ): void {
        if (samples <= 0) return;
        if (this.oneOutput) {
            const out = outputs[0];
            // Build a fresh channelMap each call — these are just references, cheap.
            const channelMap: Float32Array[][] = [];
            for (let i = 0; i < 32; i += 2) {
                channelMap.push([out[i], out[i + 1]]);
            }
            synth.processSplit(channelMap, out[0], out[0], startIndex, samples);
        } else {
            synth.processSplit(
                outputs.slice(1),
                outputs[0][0],
                outputs[0][1],
                startIndex,
                samples
            );
        }
    }

    private publishState(currentTime: number, sampleRate: number): void {
        // Lock-free SAB writer using atomic CAS — mirrors client's SyncStream.createWriter.
        // Skip silently if the reader is mid-read; the next quantum publishes a fresher value.
        if (
            Atomics.compareExchange(
                this.flag,
                STATE_FLAG_OFFSET,
                STATE_READ,
                STATE_WRITING
            ) !== STATE_READ
        ) {
            return;
        }
        this.view.setFloat64(POSITION_OFFSET, this.position, true);
        this.view.setFloat64(
            TIMESTAMP_OFFSET,
            currentTime + RENDER_QUANTUM / sampleRate,
            true
        );
        this.view.setUint32(IS_PLAYING_OFFSET, this.isPlaying ? 1 : 0, true);
        this.view.setFloat32(BPM_OFFSET, this.bpm, true);
        this.view.setUint32(LOOP_COUNT_OFFSET, this.loopCount, true);
        Atomics.store(this.flag, STATE_FLAG_OFFSET, STATE_WRITTEN);
    }
}
