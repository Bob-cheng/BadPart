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
exports.argmax2d = void 0;
var tf = require("@tensorflow/tfjs-core");
function mod(a, b) {
    return tf.tidy(function () {
        var floored = tf.div(a, tf.scalar(b, 'int32'));
        return tf.sub(a, tf.mul(floored, tf.scalar(b, 'int32')));
    });
}
function argmax2d(inputs) {
    var _a = inputs.shape, height = _a[0], width = _a[1], depth = _a[2];
    return tf.tidy(function () {
        var reshaped = tf.reshape(inputs, [height * width, depth]);
        var coords = tf.argMax(reshaped, 0);
        var yCoords = tf.expandDims(tf.div(coords, tf.scalar(width, 'int32')), 1);
        var xCoords = tf.expandDims(mod(coords, width), 1);
        return tf.concat([yCoords, xCoords], 1);
    });
}
exports.argmax2d = argmax2d;
//# sourceMappingURL=argmax2d.js.map