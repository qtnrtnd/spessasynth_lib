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

        // If the host allocated a SAB for transport state, run the in-worklet scheduler.
        // Same audio thread as the synth core → note dispatch happens via direct
        // ProcessMessage / processSplit calls, with zero postMessage hops between
        // Scheduler and synthesis, and sample-accurate sub-block dispatch.
        const sab = options.processorOptions.engineSab;
        if (sab !== undefined) {
            this.scheduler = new TransportScheduler(
                sab,
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
        } else {
            this.scheduler = null;
        }
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
