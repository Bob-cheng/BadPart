/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import * as tf from '@tensorflow/tfjs-core';
/**
 * Takes the sigmoid of the segmentation output, and generates a segmentation
 * mask with a 1 or 0 at each pixel where there is a person or not a person. The
 * segmentation threshold determines the threshold of a score for a pixel for it
 * to be considered part of a person.
 * @param segmentScores A 3d-tensor of the sigmoid of the segmentation output.
 * @param segmentationThreshold The minimum that segmentation values must have
 * to be considered part of the person.  Affects the generation of the
 * segmentation mask and the clipping of the colored part image.
 *
 * @returns A segmentation mask with a 1 or 0 at each pixel where there is a
 * person or not a person.
 */
export declare function toMaskTensor(segmentScores: tf.Tensor2D, threshold: number): tf.Tensor2D;
/**
 * Takes the sigmoid of the person and part map output, and returns a 2d tensor
 * of an image with the corresponding value at each pixel corresponding to the
 * part with the highest value. These part ids are clipped by the segmentation
 * mask. Wherever the a pixel is clipped by the segmentation mask, its value
 * will set to -1, indicating that there is no part in that pixel.
 * @param segmentScores A 3d-tensor of the sigmoid of the segmentation output.
 * @param partHeatmapScores A 3d-tensor of the sigmoid of the part heatmap
 * output. The third dimension corresponds to the part.
 *
 * @returns A 2d tensor of an image with the corresponding value at each pixel
 * corresponding to the part with the highest value. These part ids are clipped
 * by the segmentation mask.  It will have values of -1 for pixels that are
 * outside of the body and do not have a corresponding part.
 */
export declare function decodePartSegmentation(segmentationMask: tf.Tensor2D, partHeatmapScores: tf.Tensor3D): tf.Tensor2D;
export declare function decodeOnlyPartSegmentation(partHeatmapScores: tf.Tensor3D): tf.Tensor2D;
