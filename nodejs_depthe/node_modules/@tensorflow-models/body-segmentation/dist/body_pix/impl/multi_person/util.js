"use strict";
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
Object.defineProperty(exports, "__esModule", { value: true });
exports.clampVector = exports.addVectors = exports.squaredDistance = exports.clamp = exports.fillArray = exports.getImageCoords = exports.getOffsetPoint = exports.getScale = void 0;
var keypoints_1 = require("../keypoints");
function getScale(_a, _b, padding) {
    var height = _a[0], width = _a[1];
    var inputResolutionY = _b[0], inputResolutionX = _b[1];
    var padT = padding.top, padB = padding.bottom, padL = padding.left, padR = padding.right;
    var scaleY = inputResolutionY / (padT + padB + height);
    var scaleX = inputResolutionX / (padL + padR + width);
    return [scaleX, scaleY];
}
exports.getScale = getScale;
function getOffsetPoint(y, x, keypoint, offsets) {
    return {
        y: offsets.get(y, x, keypoint),
        x: offsets.get(y, x, keypoint + keypoints_1.NUM_KEYPOINTS)
    };
}
exports.getOffsetPoint = getOffsetPoint;
function getImageCoords(part, outputStride, offsets) {
    var heatmapY = part.heatmapY, heatmapX = part.heatmapX, keypoint = part.id;
    var _a = getOffsetPoint(heatmapY, heatmapX, keypoint, offsets), y = _a.y, x = _a.x;
    return {
        x: part.heatmapX * outputStride + x,
        y: part.heatmapY * outputStride + y
    };
}
exports.getImageCoords = getImageCoords;
function fillArray(element, size) {
    var result = new Array(size);
    for (var i = 0; i < size; i++) {
        result[i] = element;
    }
    return result;
}
exports.fillArray = fillArray;
function clamp(a, min, max) {
    if (a < min) {
        return min;
    }
    if (a > max) {
        return max;
    }
    return a;
}
exports.clamp = clamp;
function squaredDistance(y1, x1, y2, x2) {
    var dy = y2 - y1;
    var dx = x2 - x1;
    return dy * dy + dx * dx;
}
exports.squaredDistance = squaredDistance;
function addVectors(a, b) {
    return { x: a.x + b.x, y: a.y + b.y };
}
exports.addVectors = addVectors;
function clampVector(a, min, max) {
    return { y: clamp(a.y, min, max), x: clamp(a.x, min, max) };
}
exports.clampVector = clampVector;
//# sourceMappingURL=util.js.map