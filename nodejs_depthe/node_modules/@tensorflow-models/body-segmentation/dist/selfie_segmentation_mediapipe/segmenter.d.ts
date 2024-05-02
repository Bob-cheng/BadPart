import { BodySegmenter } from '../body_segmenter';
import { MediaPipeSelfieSegmentationMediaPipeModelConfig } from './types';
/**
 * Loads the MediaPipe solution.
 *
 * @param modelConfig An object that contains parameters for
 * the MediaPipeSelfieSegmentation loading process. Please find more details of
 * each parameters in the documentation of the
 * `MediaPipeSelfieSegmentationMediaPipeModelConfig` interface.
 */
export declare function load(modelConfig: MediaPipeSelfieSegmentationMediaPipeModelConfig): Promise<BodySegmenter>;
