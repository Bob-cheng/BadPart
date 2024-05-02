/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
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
 *
 * =============================================================================
 */
import * as tf from '@tensorflow/tfjs-core';
import { BodyPixInput, BodyPixOutputStride, Padding } from './types';
import { Pose, TensorBuffer3D } from './types';
import { BodyPixInternalResolution } from './types';
export declare function getInputSize(input: BodyPixInput): [number, number];
export declare function toValidInputResolution(inputResolution: number, outputStride: BodyPixOutputStride): number;
export declare function toInputResolutionHeightAndWidth(internalResolution: BodyPixInternalResolution, outputStride: BodyPixOutputStride, [inputHeight, inputWidth]: [number, number]): [number, number];
export declare function toInputTensor(input: BodyPixInput): tf.Tensor3D;
export declare function resizeAndPadTo(imageTensor: tf.Tensor3D, [targetH, targetW]: [number, number], flipHorizontal?: boolean): {
    resizedAndPadded: tf.Tensor3D;
    paddedBy: [[number, number], [number, number]];
};
export declare function scaleAndCropToInputTensorShape(tensor: tf.Tensor3D, [inputTensorHeight, inputTensorWidth]: [number, number], [resizedAndPaddedHeight, resizedAndPaddedWidth]: [number, number], [[padT, padB], [padL, padR]]: [[number, number], [number, number]], applySigmoidActivation?: boolean): tf.Tensor3D;
export declare function removePaddingAndResizeBack(resizedAndPadded: tf.Tensor3D, [originalHeight, originalWidth]: [number, number], [[padT, padB], [padL, padR]]: [[number, number], [number, number]]): tf.Tensor3D;
export declare function resize2d(tensor: tf.Tensor2D, resolution: [number, number], nearestNeighbor?: boolean): tf.Tensor2D;
export declare function padAndResizeTo(input: BodyPixInput, [targetH, targetW]: [number, number]): {
    resized: tf.Tensor3D;
    padding: Padding;
};
export declare function toTensorBuffers3D(tensors: tf.Tensor3D[]): Promise<TensorBuffer3D[]>;
export declare function scalePose(pose: Pose, scaleY: number, scaleX: number, offsetY?: number, offsetX?: number): Pose;
export declare function scalePoses(poses: Pose[], scaleY: number, scaleX: number, offsetY?: number, offsetX?: number): Pose[];
export declare function flipPoseHorizontal(pose: Pose, imageWidth: number): Pose;
export declare function flipPosesHorizontal(poses: Pose[], imageWidth: number): Pose[];
export declare function scaleAndFlipPoses(poses: Pose[], [height, width]: [number, number], [inputResolutionHeight, inputResolutionWidth]: [number, number], padding: Padding, flipHorizontal: boolean): Pose[];
