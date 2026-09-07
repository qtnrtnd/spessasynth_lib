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
const pulsesToSamples = (p: number, bpm: number, sr: number) =>
    ((p * 60) / (bpm * Quarter)) * sr;
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
// 4 × Uint32 = one bit per MIDI pitch: the pitches this transport is holding
// Open right now. Published rather than re-derived on the main thread, which
// Cannot know what was attacked — only what the note data would imply.
const SOUNDING_MASK_OFFSET = 36;
const SOUNDING_MASK_WORDS = 4;
const STATE_READ = 1;
const STATE_WRITING = 2;
const STATE_WRITTEN = 3;

const RENDER_QUANTUM = 128;
const noteOnByte = (channel: number) =>
    MIDIMessageTypes.noteOn | (channel & 0x0f);
const noteOffByte = (channel: number) =>
    MIDIMessageTypes.noteOff | (channel & 0x0f);

/**
 * Transport state as published on the port, for a host that could not allocate
 * a SAB. Mirrors the SAB layout field for field, so both paths hand the host
 * the same six values — see publishStateOverPort for why they arrive sparsely.
 */
export interface EngineStateMessage {
    type: "engine:state";
    position: number;
    playbackTimestamp: number;
    isPlaying: boolean;
    bpm: number;
    loopCount: number;
    soundingMask: number[];
}

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
    // The `notes` entry this event belongs to, or null when it belongs to no
    // Live note (a phantom's release, an overlay hit, the loop end). Only a live
    // Note's attack and release move `openNotes`.
    noteId: number | null;
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
    // The `notes` entries whose voice this transport has actually attacked and
    // Not yet released, as id → the pitch it was attacked on. Deliberately NOT
    // Inferred from "the playhead lies inside the note": that reads the note as
    // It is NOW, and an edit is exactly the moment the two stop agreeing — a
    // Note edited under a live drag would be read as sounding on every frame of
    // It, and each frame would park one more phantom releasing the same key.
    // What was attacked is a fact, so it is recorded rather than deduced. It is
    // Also what the keyboard lane lights up (see publishState), so the fact is
    // Owed to the eye as much as to the scheduler.
    //
    // A zero-length note (a drum one-shot) never enters: it is dispatched and
    // Gone, owes no note-off, and has no sustain for an edit to strand or for a
    // Key to show.
    private readonly openNotes = new Map<number, number>();
    // One-shot preview hits (previewer.ts's drumkit audition), overlaid on top
    // of the live composition. Never re-populated on loop wrap, unlike `notes`
    // — each entry is spliced out the moment its noteOff fires, so a bar plays
    // exactly once. See clearOverlay.
    private overlayNotes: OverlayNote[] = [];
    private loopCount = 0;
    private readonly flag: Uint8Array | null;
    private readonly view: DataView | null;
    private readonly port: MessagePort | null;
    private readonly oneOutput: boolean;
    // Reusable scratch buffers to avoid per-quantum allocation.
    private readonly events: ScheduledEvent[] = [];
    private readonly soundingMask = new Uint32Array(SOUNDING_MASK_WORDS);
    /** Last values PUBLISHED on the port path, to decide whether this quantum
     *  has anything new to say. Never read on the SAB path. */
    private lastSent = {
        isPlaying: false,
        bpm: 0,
        loopCount: -1,
        position: -1,
        mask: new Uint32Array(SOUNDING_MASK_WORDS)
    };

    /**
     * Two ways to publish, and only one of them is available at a time.
     *
     * With a `sab`, state goes out every quantum through an atomic mailbox the
     * host polls on rAF — cheapest possible, but it costs the document its
     * cross-origin isolation, which is what blocks embedding a YouTube frame.
     *
     * Without one, state goes out over the worklet `port`, and then it MUST NOT
     * go out every quantum: 375 messages a second onto the main thread's task
     * queue is exactly what the Web Audio guidance says not to do. It doesn't
     * need to. The host does not sample `position`, it EXTRAPOLATES it from an
     * anchor (`position` + `playbackTimestamp` + `bpm`) against
     * `AudioContext.currentTime`, so a stale anchor still yields the exact
     * position. Only the values it cannot derive have to be pushed, and each of
     * those changes on a discrete event — see publishState.
     */
    public constructor(
        sab: SharedArrayBuffer | null,
        port: MessagePort | null,
        oneOutput: boolean
    ) {
        this.flag = sab ? new Uint8Array(sab) : null;
        this.view = sab ? new DataView(sab) : null;
        this.port = sab ? null : port;
        this.oneOutput = oneOutput;
        if (this.flag) Atomics.store(this.flag, STATE_FLAG_OFFSET, STATE_READ);
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

    /** Releases and drops every phantom parked on one (channel, pitch) — see
     *  the call site, where an attack on that same pair takes the key over. */
    private releasePhantomsOn(
        channel: number,
        midi: number,
        synth: SpessaSynthProcessor
    ): void {
        for (let i = this.phantomNotes.length - 1; i >= 0; i--) {
            const p = this.phantomNotes[i];
            if (p.channel !== channel || p.midi !== midi) continue;
            synth.processMessage([noteOffByte(p.channel), p.midi], 0, {
                time: 0
            });
            this.phantomNotes.splice(i, 1);
        }
    }

    /** Ids of the notes the playhead is standing INSIDE — strictly inside: one
     *  starting exactly on the position is left to renderBlock, whose noteOn
     *  interval is [p0, segmentEnd) and would otherwise attack it twice. */
    private straddlingIds(): number[] {
        const out: number[] = [];
        for (const n of this.notes) {
            if (
                n.pulse < this.position &&
                n.pulse + n.durationPulses > this.position
            ) {
                out.push(n.id);
            }
        }
        return out;
    }

    /** Attacks every note the playhead stands inside — how a position that did
     *  not arrive by playing (a resume, a seek) still sounds like a playhead
     *  standing there, rather than staying silent until the next note begins. */
    private attackStraddling(synth: SpessaSynthProcessor): void {
        for (const n of this.notes) {
            if (
                n.pulse < this.position &&
                n.pulse + n.durationPulses > this.position
            ) {
                synth.processMessage(
                    [noteOnByte(n.channel), n.midi, n.vel],
                    0,
                    { time: 0 }
                );
                if (n.durationPulses > 0) this.openNotes.set(n.id, n.midi);
            }
        }
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
                if (
                    this.position < this.loopStart ||
                    this.position >= this.loopEnd
                ) {
                    this.position = this.loopStart;
                }
                this.isPlaying = true;
                // Re-trigger notes whose pulse range straddles the resume position
                // So a paused-mid-sustain resumes audibly instead of silent until the
                // Next noteOn. Their noteOffs will be picked up naturally when
                // (note.pulse + duration) falls into a future block's [p0, p1) range.
                this.attackStraddling(synth);
                return;
            }
            case "engine:pause": {
                // Rewind to the audible position so the displayed playhead doesn't jump
                // Forward when the host switches off interpolation.
                if (cmd.rewindToPulse !== undefined)
                    this.position = cmd.rewindToPulse;
                this.isPlaying = false;
                this.phantomNotes.length = 0;
                this.openNotes.clear();
                this.clearOverlay(synth);
                synth.stopAllChannels(false);
                return;
            }
            case "engine:stop": {
                this.isPlaying = false;
                this.position = this.loopStart;
                this.loopCount = 0;
                this.phantomNotes.length = 0;
                this.openNotes.clear();
                this.clearOverlay(synth);
                synth.stopAllChannels(false);
                return;
            }
            case "engine:seek": {
                // Seek is a teleport — kill voices abruptly so the old position's audio
                // Doesn't bleed into the new position.
                this.position = cmd.pulse;
                if (!this.isPlaying) {
                    this.phantomNotes.length = 0;
                    this.openNotes.clear();
                    this.clearOverlay(synth);
                    synth.stopAllChannels(true);
                    return;
                }
                // Landing while playing has to SOUND like a playhead standing
                // There, which means attacking the notes it landed inside even
                // Though they began before it — the same welcome a resume gives
                // A sustain it lands in the middle of.
                //
                // But the ruler is DRAGGED, so this arrives once a frame, and a
                // Teleport-then-attack repeated at that rate turns a long note
                // Being scrubbed over into a machine gun. So the landing is
                // Diffed first: as long as the playhead stays inside the same
                // Notes, the voices it is already holding are the right ones and
                // Nothing is re-voiced — only when the set changes is the old
                // One cut and the new one attacked.
                const straddling = this.straddlingIds();
                let sameVoices = straddling.length === this.openNotes.size;
                if (sameVoices) {
                    for (const id of straddling) {
                        if (!this.openNotes.has(id)) {
                            sameVoices = false;
                            break;
                        }
                    }
                }
                // Nothing left over from the old position either: a phantom or a
                // Preview overlay is anchored to absolute pulses the seek may
                // Have just jumped over, and letting one ring on is the very
                // Bleed a teleport exists to prevent. One frame of a drag pays
                // For it, then the shortcut applies again.
                if (
                    sameVoices &&
                    this.phantomNotes.length === 0 &&
                    this.overlayNotes.length === 0
                ) {
                    return;
                }
                this.phantomNotes.length = 0;
                this.openNotes.clear();
                this.clearOverlay(synth);
                synth.stopAllChannels(true);
                this.attackStraddling(synth);
                return;
            }
            case "engine:updateNotes": {
                // While stopped/paused, nothing is sounding — just swap the list.
                if (!this.isPlaying) {
                    this.notes = cmd.notes;
                    return;
                }
                // While playing, diff old vs new note lists by id. The rule the
                // Diff exists to keep: **an edit never reaches the rendition
                // Already under way**. A note whose voice is open at the moment
                // Of the edit plays out as it was written then — and only that
                // Rendition is spared. The edited note is live for everything
                // Else, including the rest of this very pass: push it ahead of
                // The playhead and the playhead will attack it there, drag still
                // Held, mouse still down. That is the reference behaviour (FL
                // Studio's), and it is also what keeps the edit's own release
                // Reachable.
                //
                //   - Removed while its voice is open: park it as a "phantom" so
                //     Its planned noteOff still fires at the original end pulse —
                //     A deleted note rings to its natural end instead of cutting
                //     Off.
                //   - Edited while its voice is open: the same phantom, and for
                //     A sharper reason. The open voice belongs to the note as it
                //     WAS, and nothing in the new list would ever close it:
                //     RenderBlock derives the release from the edited note's
                //     End, which an edit can put behind the playhead, where no
                //     Future block will ever look. That is the note that plays
                //     For ever.
                //   - Added while the playhead stands inside it: noteOn
                //     Immediately, so a note just drawn under the playhead is
                //     Heard right away. Only an addition gets this: an edited
                //     Note dragged under the playhead is a rendition that has
                //     Already begun without it, and it waits for the next pass.
                const pos = this.position;
                // Velocity is left out on purpose: it cannot strand an open
                // Voice (the release is unchanged), so a note whose velocity
                // Alone moved needs no phantom.
                const sameSpan = (a: WorkletNote, b: WorkletNote): boolean =>
                    a.pulse === b.pulse &&
                    a.durationPulses === b.durationPulses &&
                    a.midi === b.midi &&
                    a.channel === b.channel;
                // Hands the open voice over to the phantom list: from here on
                // The note is no longer sounding *as this id*, which is what
                // Keeps a drag that edits it again on the next frame from
                // Parking a second phantom over the same voice.
                const strand = (old: WorkletNote): void => {
                    this.phantomNotes.push(old);
                    this.openNotes.delete(old.id);
                };

                const oldById = new Map<number, WorkletNote>();
                for (const n of this.notes) oldById.set(n.id, n);
                this.notes = cmd.notes;

                for (const n of cmd.notes) {
                    const old = oldById.get(n.id);
                    oldById.delete(n.id); // What stays behind was removed
                    if (old === undefined) {
                        if (
                            n.pulse <= pos &&
                            n.pulse + n.durationPulses > pos
                        ) {
                            synth.processMessage(
                                [noteOnByte(n.channel), n.midi, n.vel],
                                0,
                                { time: 0 }
                            );
                            if (n.durationPulses > 0)
                                this.openNotes.set(n.id, n.midi);
                        }
                        continue;
                    }
                    if (sameSpan(old, n)) continue; // Untouched by this push
                    if (this.openNotes.has(n.id)) strand(old);
                }

                for (const old of oldById.values()) {
                    if (this.openNotes.has(old.id)) strand(old);
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
                    this.openNotes.clear();
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
            // How many samples this segment actually spans — NOT the rest of the
            // quantum. A wrapping segment ends at loopEnd, somewhere inside the
            // block, and claiming the whole remainder for it was two bugs in one:
            // the outer loop could never re-enter (cursorSample reached
            // RENDER_QUANTUM), so the pulses past loopEnd were discarded and every
            // wrap cost the transport whatever was left of the quantum — 2 beats
            // at 128 BPM is 351.5625 quanta at 48 kHz, so the loop ran 352 of them
            // and took 938.67 ms instead of 937.5, every single pass, always the
            // same way. A composer's monitor therefore walked away from the sync
            // grid at 1.17 ms a loop for the length of a turn. And the events of a
            // wrapping segment were spread over the full block besides
            // (`sampleSpan`), so their offsets were stretched to match.
            //
            // At least one sample when wrapping, so a remainder too small to round
            // to one still makes progress and the outer loop cannot spin.
            const segmentSamples = wrapsInSegment
                ? Math.max(
                      1,
                      Math.min(
                          remainingSamples,
                          Math.round(
                              pulsesToSamples(
                                  this.loopEnd - p0,
                                  this.bpm,
                                  sampleRate
                              )
                          )
                      )
                  )
                : remainingSamples;
            const endOfSegmentSample = cursorSample + segmentSamples;

            // Gather events whose pulse falls in [p0, segmentEnd) and compute their
            // Sample offset relative to the quantum start.
            this.events.length = 0;
            const pulseSpan = segmentEnd - p0;
            const sampleSpan = segmentSamples;
            const cursorAtSegmentStart = cursorSample;
            const pulseToSampleInSegment = (pulse: number): number => {
                if (pulseSpan <= 0) return cursorAtSegmentStart;
                const local = (pulse - p0) / pulseSpan; // 0..1 within this segment
                return cursorAtSegmentStart + Math.round(local * sampleSpan);
            };

            for (const n of this.notes) {
                // A zero-length note is a one-shot: dispatched, never held, so it
                // Is not tracked as an open voice (see openNotes). Passing null
                // Here is what keeps it out, on its attack and its release alike.
                const heldId = n.durationPulses > 0 ? n.id : null;
                // NoteOn interval: [p0, segmentEnd). A note starts in exactly one segment
                // (the one whose [p0, segmentEnd) contains its pulse).
                if (n.pulse >= p0 && n.pulse < segmentEnd) {
                    this.events.push({
                        sampleOffset: pulseToSampleInSegment(n.pulse),
                        typeRank: TYPE_NOTE_ON,
                        midi: n.midi,
                        vel: n.vel,
                        channel: n.channel,
                        noteId: heldId
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
                        channel: n.channel,
                        noteId: heldId
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
                        channel: p.channel,
                        noteId: null
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
                        channel: n.channel,
                        noteId: null
                    });
                }
                const overlayEnd = n.pulse + n.durationPulses;
                if (overlayEnd > p0 && overlayEnd <= segmentEnd) {
                    this.events.push({
                        sampleOffset: pulseToSampleInSegment(overlayEnd),
                        typeRank: TYPE_NOTE_OFF,
                        midi: n.midi,
                        vel: 0,
                        channel: n.channel,
                        noteId: null
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
                    channel: 0,
                    noteId: null
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
                    // A phantom still holding this very key would, at its own
                    // Planned end, release the attack about to happen instead of
                    // The voice it was parked for — a noteOff addresses a
                    // (channel, pitch) pair, not a voice. So the attack takes
                    // The key over: the phantom is released right here, one
                    // Instant before, and dropped. That is the note pushed ahead
                    // Of the playhead and caught up with mid-drag — the old
                    // Rendition ends exactly where the new one begins.
                    this.releasePhantomsOn(e.channel, e.midi, synth);
                    synth.processMessage(
                        [noteOnByte(e.channel), e.midi, e.vel],
                        0,
                        {
                            time: 0
                        }
                    );
                    if (e.noteId !== null) this.openNotes.set(e.noteId, e.midi);
                } else if (e.typeRank === TYPE_NOTE_OFF) {
                    synth.processMessage([noteOffByte(e.channel), e.midi], 0, {
                        time: 0
                    });
                    if (e.noteId !== null) this.openNotes.delete(e.noteId);
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
        const timestamp = currentTime + RENDER_QUANTUM / sampleRate;
        if (!this.view || !this.flag) {
            this.publishStateOverPort(timestamp);
            return;
        }
        /* Lock-free SAB writer using atomic CAS. This is the only writer there
           is: the host side owns the reader alone (`sync-stream.ts`), since
           publishing happens from inside this render loop. Skip silently if the
           reader is mid-read; the next quantum publishes a fresher value. */
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
        this.view.setFloat64(TIMESTAMP_OFFSET, timestamp, true);
        this.view.setUint32(IS_PLAYING_OFFSET, this.isPlaying ? 1 : 0, true);
        this.view.setFloat32(BPM_OFFSET, this.bpm, true);
        this.view.setUint32(LOOP_COUNT_OFFSET, this.loopCount, true);
        this.computeSoundingMask();
        for (let w = 0; w < SOUNDING_MASK_WORDS; w++) {
            this.view.setUint32(
                SOUNDING_MASK_OFFSET + w * 4,
                this.soundingMask[w],
                true
            );
        }
        Atomics.store(this.flag, STATE_FLAG_OFFSET, STATE_WRITTEN);
    }

    /**
     * The no-SAB path: post only what the host cannot work out for itself.
     *
     * While the transport runs, `position` is NOT news — the host extrapolates
     * it from the last anchor and the audio clock, exactly as it does on the
     * SAB path, so re-sending it every quantum would buy nothing and cost 375
     * messages a second. What the host cannot derive is a discrete change:
     * play/pause, a tempo change, a loop wrap (which is also what re-anchors
     * the extrapolation, so drift never accumulates past one loop), and the
     * sounding mask, whose whole point is that it cannot be re-derived from the
     * note data (see computeSoundingMask). Position is added to the test only
     * while stopped, where nothing else would report a seek.
     */
    private publishStateOverPort(timestamp: number): void {
        if (!this.port) return;
        this.computeSoundingMask();

        const last = this.lastSent;
        let maskChanged = false;
        for (let w = 0; w < SOUNDING_MASK_WORDS; w++) {
            if (last.mask[w] !== this.soundingMask[w]) {
                maskChanged = true;
                break;
            }
        }
        const changed =
            maskChanged ||
            last.isPlaying !== this.isPlaying ||
            last.bpm !== this.bpm ||
            last.loopCount !== this.loopCount ||
            (!this.isPlaying && last.position !== this.position);
        if (!changed) return;

        last.isPlaying = this.isPlaying;
        last.bpm = this.bpm;
        last.loopCount = this.loopCount;
        last.position = this.position;
        last.mask.set(this.soundingMask);

        this.port.postMessage({
            type: "engine:state",
            position: this.position,
            playbackTimestamp: timestamp,
            isPlaying: this.isPlaying,
            bpm: this.bpm,
            loopCount: this.loopCount,
            /* A plain array: the receiving end reads six numbers at rAF, and a
               transferable would have to be handed back to be reused. */
            soundingMask: Array.from(this.soundingMask)
        });
    }

    /**
     * Fills `soundingMask` with the pitches currently being held, one bit each
     * — what the keyboard lane lights up. Both sources count, because both are
     * audible: the notes whose voice is open, and the phantoms, which are open
     * voices too — a rendition spared by an edit is still sounding, and the key
     * it holds is still down.
     *
     * Built from the ids rather than from the note data, for the same reason the
     * scheduler is: a note dragged behind an oncoming playhead SPANS that
     * playhead without ever having been attacked, and the eye would be told a
     * key is down that no voice ever played. That is also why the host can never
     * reconstruct this mask itself, on either publish path.
     */
    private computeSoundingMask(): void {
        for (let w = 0; w < SOUNDING_MASK_WORDS; w++) this.soundingMask[w] = 0;
        for (const midi of this.openNotes.values()) {
            this.soundingMask[midi >> 5] |= 1 << (midi & 31);
        }
        for (const p of this.phantomNotes) {
            this.soundingMask[p.midi >> 5] |= 1 << (p.midi & 31);
        }
    }
}
