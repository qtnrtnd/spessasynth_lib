import { SpessaLog } from "spessasynth_core";
import { ConsoleColors } from "./utils/other.ts";
import { WORKLET_PROCESSOR_NAME } from "./synthesizer/worklet/worklet_processor_name.ts";
import type { PassedProcessorParameters } from "./synthesizer/types.ts";
import { WorkletSynthesizerCore } from "./synthesizer/worklet/worklet_synthesizer_core.ts";
import {
    type EngineCommand,
    TransportScheduler
} from "./synthesizer/worklet/transport_scheduler.ts";

interface EngineProcessorOptions {
    engineSab?: SharedArrayBuffer;
}

class WorkletSynthesizerProcessor extends AudioWorkletProcessor {
    private readonly core: WorkletSynthesizerCore;
    private readonly scheduler: TransportScheduler | null;

    public constructor(options: {
        processorOptions: PassedProcessorParameters & EngineProcessorOptions;
    }) {
        super();
        this.core = new WorkletSynthesizerCore(
            sampleRate, // AudioWorkletGlobalScope
            currentTime, // AudioWorkletGlobalScope, sync with audioContext time
            this.port,
            options.processorOptions
        );

        /* The in-worklet scheduler always runs: it IS the transport, and it
           lives on the same audio thread as the synth core → note dispatch
           happens via direct processMessage / processSplit calls, with zero
           postMessage hops between scheduler and synthesis, and sample-accurate
           sub-block dispatch.

           The SAB is only how it PUBLISHES state back. When the host could not
           allocate one — any document that is not cross-origin isolated — the
           scheduler posts on this port instead. Scheduling, timing and dispatch
           are identical either way, and the host extrapolates the playhead
           between anchors, so the sparser stream is not felt. This used to
           leave `scheduler` null, which silently disabled the transport
           entirely. */
        const sab = options.processorOptions.engineSab;
        this.scheduler = new TransportScheduler(
            sab ?? null,
            this.port,
            options.processorOptions.oneOutput
        );
        this.core.onPreMessage = (data: unknown): boolean => {
            if (
                typeof data === "object" &&
                data !== null &&
                typeof (data as { type?: unknown }).type === "string" &&
                (data as { type: string }).type.startsWith("engine:")
            ) {
                this.scheduler!.handleCommand(
                    data as EngineCommand,
                    this.core.synthesizer
                );
                return true;
            }
            return false;
        };
    }

    // Don't bind, do it like this for it to work with Chrome 109
    public process(inputs: Float32Array[][], outputs: Float32Array[][]) {
        if (this.scheduler) {
            if (!this.core.isAlive) return false;
            this.core.tickSequencers();
            this.scheduler.renderBlock(
                currentTime,
                sampleRate,
                this.core.synthesizer,
                outputs
            );
            this.core.afterRender();
            return true;
        }
        return this.core.process(inputs, outputs);
    }
}

registerProcessor(WORKLET_PROCESSOR_NAME, WorkletSynthesizerProcessor);
SpessaLog.info(
    "%cProcessor successfully registered!",
    ConsoleColors.recognized
);
