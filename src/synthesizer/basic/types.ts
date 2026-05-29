import type { PassedProcessorParameters } from "../types";

export interface SynthConfig {
    /**
     * If the synth should use one output with 32 channels (2 audio channels for each midi channel).
     */
    oneOutput: boolean;

    /**
     * Custom audio node creation functions for Web Audio wrappers, such as standardized-audio-context.
     * Pass undefined to use the Web Audio API.
     */
    audioNodeCreators?: AudioNodeCreators;

    /**
     * If the event system should be enabled. This can only be set once.
     */
    eventsEnabled: boolean;

    /**
     * Optional SharedArrayBuffer for the in-worklet transport scheduler. When set,
     * the worklet runs an in-thread scheduler that dispatches MIDI events directly
     * into the synth core (no postMessage hops between scheduling and synthesis).
     * The buffer carries transport state back to the main thread via lock-free
     * SyncStream protocol. Required SAB byte length: 64.
     */
    engineSab?: SharedArrayBuffer;
}

export interface AudioNodeCreators {
    /**
     * A custom creator for an AudioWorkletNode.
     * @param context
     * @param workletName
     * @param options
     */
    worklet: (
        context: BaseAudioContext,
        workletName: string,
        options?: AudioWorkletNodeOptions & {
            processorOptions: PassedProcessorParameters;
        }
    ) => AudioWorkletNode;
}
