/**
 * @license
 * Copyright 2019 Google Inc. All Rights Reserved.
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
import { Padding, PartSegmentation, PersonSegmentation, Pose } from '../types';
export declare function toPersonKSegmentation(segmentation: tf.Tensor2D, k: number): tf.Tensor2D;
export declare function toPersonKPartSegmentation(segmentation: tf.Tensor2D, bodyParts: tf.Tensor2D, k: number): tf.Tensor2D;
export declare function decodePersonInstanceMasks(segmentation: tf.Tensor2D, longOffsets: tf.Tensor3D, poses: Pose[], height: number, width: number, stride: number, [inHeight, inWidth]: [number, number], padding: Padding, minPoseScore?: number, refineSteps?: number, minKeypointScore?: number, maxNumPeople?: number): Promise<PersonSegmentation[]>;
export declare function decodePersonInstancePartMasks(segmentation: tf.Tensor2D, longOffsets: tf.Tensor3D, partSegmentation: tf.Tensor2D, poses: Pose[], height: number, width: number, stride: number, [inHeight, inWidth]: [number, number], padding: Padding, minPoseScore?: number, refineSteps?: number, minKeypointScore?: number, maxNumPeople?: number): Promise<PartSegmentation[]>;
