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
import { Padding, Part, TensorBuffer3D, Vector2D } from '../types';
export declare function getScale([height, width]: [number, number], [inputResolutionY, inputResolutionX]: [number, number], padding: Padding): [number, number];
export declare function getOffsetPoint(y: number, x: number, keypoint: number, offsets: TensorBuffer3D): Vector2D;
export declare function getImageCoords(part: Part, outputStride: number, offsets: TensorBuffer3D): Vector2D;
export declare function fillArray<T>(element: T, size: number): T[];
export declare function clamp(a: number, min: number, max: number): number;
export declare function squaredDistance(y1: number, x1: number, y2: number, x2: number): number;
export declare function addVectors(a: Vector2D, b: Vector2D): Vector2D;
export declare function clampVector(a: Vector2D, min: number, max: number): Vector2D;
