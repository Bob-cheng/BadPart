"use strict";
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
Object.defineProperty(exports, "__esModule", { value: true });
exports.BaseModel = void 0;
var tf = require("@tensorflow/tfjs-core");
/**
 * BodyPix supports using various convolution neural network models
 * (e.g. ResNet and MobileNetV1) as its underlying base model.
 * The following BaseModel interface defines a unified interface for
 * creating such BodyPix base models. Currently both MobileNet (in
 * ./mobilenet.ts) and ResNet (in ./resnet.ts) implements the BaseModel
 * interface. New base models that conform to the BaseModel interface can be
 * added to BodyPix.
 */
var BaseModel = /** @class */ (function () {
    function BaseModel(model, outputStride) {
        this.model = model;
        this.outputStride = outputStride;
        var inputShape = this.model.inputs[0].shape;
        tf.util.assert((inputShape[1] === -1) && (inputShape[2] === -1), function () { return "Input shape [" + inputShape[1] + ", " + inputShape[2] + "] " +
            "must both be equal to or -1"; });
    }
    /**
     * Predicts intermediate Tensor representations.
     *
     * @param input The input RGB image of the base model.
     * A Tensor of shape: [`inputResolution`, `inputResolution`, 3].
     *
     * @return A dictionary of base model's intermediate predictions.
     * The returned dictionary should contains the following elements:
     * - heatmapScores: A Tensor3D that represents the keypoint heatmap scores.
     * - offsets: A Tensor3D that represents the offsets.
     * - displacementFwd: A Tensor3D that represents the forward displacement.
     * - displacementBwd: A Tensor3D that represents the backward displacement.
     * - segmentation: A Tensor3D that represents the segmentation of all
     * people.
     * - longOffsets: A Tensor3D that represents the long offsets used for
     * instance grouping.
     * - partHeatmaps: A Tensor3D that represents the body part segmentation.
     */
    BaseModel.prototype.predict = function (input) {
        var _this = this;
        return tf.tidy(function () {
            var asFloat = _this.preprocessInput(tf.cast(input, 'float32'));
            var asBatch = tf.expandDims(asFloat, 0);
            var results = _this.model.predict(asBatch);
            var results3d = results.map(function (y) { return tf.squeeze(y, [0]); });
            var namedResults = _this.nameOutputResults(results3d);
            return {
                heatmapScores: tf.sigmoid(namedResults.heatmap),
                offsets: namedResults.offsets,
                displacementFwd: namedResults.displacementFwd,
                displacementBwd: namedResults.displacementBwd,
                segmentation: namedResults.segmentation,
                partHeatmaps: namedResults.partHeatmaps,
                longOffsets: namedResults.longOffsets,
                partOffsets: namedResults.partOffsets
            };
        });
    };
    /**
     * Releases the CPU and GPU memory allocated by the model.
     */
    BaseModel.prototype.dispose = function () {
        this.model.dispose();
    };
    return BaseModel;
}());
exports.BaseModel = BaseModel;
//# sourceMappingURL=base_model.js.map