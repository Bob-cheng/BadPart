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
import { BaseModel } from './base_model';
import { BodyPixArchitecture, BodyPixInput, BodyPixInternalResolution, BodyPixMultiplier, BodyPixOutputStride, BodyPixQuantBytes, Padding } from './types';
import { PartSegmentation, PersonSegmentation, SemanticPartSegmentation, SemanticPersonSegmentation } from './types';
/**
 * BodyPix model loading is configurable using the following config dictionary.
 *
 * `architecture`: BodyPixArchitecture. It determines which BodyPix architecture
 * to load. The supported architectures are: MobileNetV1 and ResNet50.
 *
 * `outputStride`: Specifies the output stride of the BodyPix model.
 * The smaller the value, the larger the output resolution, and more accurate
 * the model at the cost of speed. Set this to a larger value to increase speed
 * at the cost of accuracy. Stride 32 is supported for ResNet and
 * stride 8,16,32 are supported for various MobileNetV1 models.
 *
 * `multiplier`: An optional number with values: 1.01, 1.0, 0.75, or
 * 0.50. The value is used only by MobileNet architecture. It is the float
 * multiplier for the depth (number of channels) for all convolution ops.
 * The larger the value, the larger the size of the layers, and more accurate
 * the model at the cost of speed. Set this to a smaller value to increase speed
 * at the cost of accuracy.
 *
 * `modelUrl`: An optional string that specifies custom url of the model. This
 * is useful for area/countries that don't have access to the model hosted on
 * GCP.
 *
 * `quantBytes`: An optional number with values: 1, 2, or 4.  This parameter
 * affects weight quantization in the models. The available options are
 * 1 byte, 2 bytes, and 4 bytes. The higher the value, the larger the model size
 * and thus the longer the loading time, the lower the value, the shorter the
 * loading time but lower the accuracy.
 */
export interface ModelConfig {
    architecture: BodyPixArchitecture;
    outputStride: BodyPixOutputStride;
    multiplier?: BodyPixMultiplier;
    modelUrl?: string;
    quantBytes?: BodyPixQuantBytes;
}
/**
 * BodyPix inference is configurable using the following config dictionary.
 *
 * `flipHorizontal`: If the left-right keypoint of poses/part segmentation
 * should be flipped/mirrored horizontally. This should be set to true for
 * videos where the video is by default flipped horizontally (i.e. a webcam),
 * and you want the person & body part segmentation to be returned in the proper
 * orientation.
 *
 * `internalResolution`: Defaults to 'medium'. The internal resolution
 * percentage that the input is resized to before inference. The larger the
 * internalResolution the more accurate the model at the cost of slower
 * prediction times. Available values are 'low', 'medium', 'high', 'full', or a
 * percentage value between 0 and 1. The values 'low', 'medium', 'high', and
 * 'full' map to 0.25, 0.5, 0.75, and 1.0 correspondingly.
 *
 * `segmentationThreshold`: The minimum that segmentation values must
 * have to be considered part of the person. Affects the generation of the
 * segmentation mask. More specifically, it is the threshold used to binarize
 * the intermediate person segmentation probability. The probability of each
 * pixel belongs to a person is in range [0, 1]. If the probability is greater
 * than the `segmentationThreshold`, it will be set to 1 otherwise 0.
 *
 */
export interface InferenceConfig {
    flipHorizontal?: boolean;
    internalResolution?: BodyPixInternalResolution;
    segmentationThreshold?: number;
}
/**
 * Person Inference Config
 *
 * `maxDetections`: Defaults to 10. Maximum number of person pose detections per
 * image.
 *
 * `scoreThreshold`: Defaults to 0.4. Only return person pose that have root
 * part score greater or equal to this value.
 *
 * `nmsRadius`: Defaults to 20. Non-maximum suppression part distance in pixels.
 * It needs to be strictly positive. Two pose keypoints suppress each other if
 * they are less than `nmsRadius` pixels away.
 */
export interface PersonInferenceConfig extends InferenceConfig {
    maxDetections?: number;
    scoreThreshold?: number;
    nmsRadius?: number;
}
/**
 * Multiple Person Instance Inference Config
 *
 * `maxDetections`: Defaults to 10. Maximum number of returned instance
 * segmentation and pose detections per image.
 *
 * `scoreThreshold`: Defaults to 0.4. Only returns and uses person
 * poses for instance segmentation assignment when the pose has root part score
 * greater or equal to this value.
 *
 * `nmsRadius`: Defaults to 20. Non-maximum suppression part distance in pixels.
 * It needs to be strictly positive. Two parts suppress each other if they are
 * less than `nmsRadius` pixels away.
 *
 * `minKeypointScore`: Default to 0.3. Keypoints above the score are used
 * for matching and assigning segmentation mask to each person.
 *
 * `refineSteps`: Default to 10. The number of refinement steps used when
 * assigning the instance segmentation. It needs to be strictly positive. The
 * larger the higher the accuracy and slower the inference.
 *
 */
export interface MultiPersonInstanceInferenceConfig extends InferenceConfig {
    maxDetections?: number;
    scoreThreshold?: number;
    nmsRadius?: number;
    minKeypointScore?: number;
    refineSteps?: number;
}
export declare const PERSON_INFERENCE_CONFIG: PersonInferenceConfig;
export declare const MULTI_PERSON_INSTANCE_INFERENCE_CONFIG: MultiPersonInstanceInferenceConfig;
export declare class BodyPix {
    baseModel: BaseModel;
    constructor(net: BaseModel);
    private predictForPersonSegmentation;
    private predictForPersonSegmentationAndPart;
    private predictForMultiPersonInstanceSegmentationAndPart;
    /**
     * Given an image with people, returns a dictionary of all intermediate
     * tensors including: 1) a binary array with 1 for the pixels that are part of
     * the person, and 0 otherwise, 2) heatmapScores, 3) offsets, and 4) paddings.
     *
     * @param input ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement)
     * The input image to feed through the network.
     *
     * @param internalResolution Defaults to 'medium'. The internal resolution
     * that the input is resized to before inference. The larger the
     * internalResolution the more accurate the model at the cost of slower
     * prediction times. Available values are 'low', 'medium', 'high', 'full', or
     * a percentage value between 0 and 1. The values 'low', 'medium', 'high', and
     * 'full' map to 0.25, 0.5, 0.75, and 1.0 correspondingly.
     *
     * @param segmentationThreshold The minimum that segmentation values must have
     * to be considered part of the person. Affects the generation of the
     * segmentation mask.
     *
     * @return A dictionary containing `segmentation`, `heatmapScores`, `offsets`,
     * and `padding`:
     * - `segmentation`: A 2d Tensor with 1 for the pixels that are part of the
     * person, and 0 otherwise. The width and height correspond to the same
     * dimensions of the input image.
     * - `heatmapScores`: A 3d Tensor of the keypoint heatmaps used by
     * pose estimation decoding.
     * - `offsets`: A 3d Tensor of the keypoint offsets used by pose
     * estimation decoding.
     * - `displacementFwd`: A 3d Tensor of the keypoint forward displacement used
     * by pose estimation decoding.
     * - `displacementBwd`: A 3d Tensor of the keypoint backward displacement used
     * by pose estimation decoding.
     * - `padding`: The padding (unit pixels) being applied to the input image
     * before it is fed into the model.
     */
    segmentPersonActivation(input: BodyPixInput, internalResolution: BodyPixInternalResolution, segmentationThreshold?: number): {
        segmentation: tf.Tensor2D;
        heatmapScores: tf.Tensor3D;
        offsets: tf.Tensor3D;
        displacementFwd: tf.Tensor3D;
        displacementBwd: tf.Tensor3D;
        padding: Padding;
        internalResolutionHeightAndWidth: [number, number];
    };
    /**
     * Given an image with many people, returns a PersonSegmentation dictionary
     * that contains the segmentation mask for all people and a single pose.
     *
     * Note: The segmentation mask returned by this method covers all people but
     * the pose works well for one person. If you want to estimate instance-level
     * multiple person segmentation & pose for each person, use
     * `segmentMultiPerson` instead.
     *
     * @param input ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement)
     * The input image to feed through the network.
     *
     * @param config PersonInferenceConfig object that contains
     * parameters for the BodyPix inference using person decoding.
     *
     * @return A SemanticPersonSegmentation dictionary that contains height,
     * width, the flattened binary segmentation mask and the poses for all people.
     * The width and height correspond to the same dimensions of the input image.
     * - `height`: The height of the segmentation data in pixel unit.
     * - `width`: The width of the segmentation data in pixel unit.
     * - `data`: The flattened Uint8Array of segmentation data. 1 means the pixel
     * belongs to a person and 0 means the pixel doesn't belong to a person. The
     * size of the array is equal to `height` x `width` in row-major order.
     * - `allPoses`: The 2d poses of all people.
     */
    segmentPerson(input: BodyPixInput, config?: PersonInferenceConfig): Promise<SemanticPersonSegmentation>;
    /**
     * Given an image with multiple people, returns an *array* of
     * PersonSegmentation object. Each element in the array corresponding to one
     * of the people in the input image. In other words, it predicts
     * instance-level multiple person segmentation & pose for each person.
     *
     * The model does standard ImageNet pre-processing before inferring through
     * the model. The image pixels should have values [0-255].
     *
     * @param input
     * ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement) The input
     * image to feed through the network.
     *
     * @param config MultiPersonInferenceConfig object that contains
     * parameters for the BodyPix inference using multi-person decoding.
     *
     * @return An array of PersonSegmentation object, each containing a width,
     * height, a binary array (1 for the pixels that are part of the
     * person, and 0 otherwise) and 2D pose. The array size corresponds to the
     * number of pixels in the image. The width and height correspond to the
     * dimensions of the image the binary array is shaped to, which are the same
     * dimensions of the input image.
     */
    segmentMultiPerson(input: BodyPixInput, config?: MultiPersonInstanceInferenceConfig): Promise<PersonSegmentation[]>;
    /**
     * Given an image with many people, returns a dictionary containing: height,
     * width, a tensor with a part id from 0-24 for the pixels that are
     * part of a corresponding body part, and -1 otherwise. This does standard
     * ImageNet pre-processing before inferring through the model.  The image
     * should pixels should have values [0-255].
     *
     * @param input ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement)
     * The input image to feed through the network.
     *
     * @param internalResolution Defaults to 'medium'. The internal resolution
     * percentage that the input is resized to before inference. The larger the
     * internalResolution the more accurate the model at the cost of slower
     * prediction times. Available values are 'low', 'medium', 'high', 'full', or
     * a percentage value between 0 and 1. The values 'low', 'medium', 'high', and
     * 'full' map to 0.25, 0.5, 0.75, and 1.0 correspondingly.
     *
     * @param segmentationThreshold The minimum that segmentation values must have
     * to be considered part of the person.  Affects the clipping of the colored
     * part image.
     *
     * @return  A dictionary containing `partSegmentation`, `heatmapScores`,
     * `offsets`, and `padding`:
     * - `partSegmentation`: A 2d Tensor with a part id from 0-24 for
     * the pixels that are part of a corresponding body part, and -1 otherwise.
     * - `heatmapScores`: A 3d Tensor of the keypoint heatmaps used by
     * single-person pose estimation decoding.
     * - `offsets`: A 3d Tensor of the keypoint offsets used by single-person pose
     * estimation decoding.
     * - `displacementFwd`: A 3d Tensor of the keypoint forward displacement
     * used by pose estimation decoding.
     * - `displacementBwd`: A 3d Tensor of the keypoint backward displacement used
     * by pose estimation decoding.
     * - `padding`: The padding (unit pixels) being applied to the input image
     * before it is fed into the model.
     */
    segmentPersonPartsActivation(input: BodyPixInput, internalResolution: BodyPixInternalResolution, segmentationThreshold?: number): {
        partSegmentation: tf.Tensor2D;
        heatmapScores: tf.Tensor3D;
        offsets: tf.Tensor3D;
        displacementFwd: tf.Tensor3D;
        displacementBwd: tf.Tensor3D;
        padding: Padding;
        internalResolutionHeightAndWidth: [number, number];
    };
    /**
     * Given an image with many people, returns a PartSegmentation dictionary that
     * contains the body part segmentation mask for all people and a single pose.
     *
     * Note: The body part segmentation mask returned by this method covers all
     * people but the pose works well when there is one person. If you want to
     * estimate instance-level multiple person body part segmentation & pose for
     * each person, use `segmentMultiPersonParts` instead.
     *
     * @param input ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement)
     * The input image to feed through the network.
     *
     * @param config PersonInferenceConfig object that contains
     * parameters for the BodyPix inference using single person decoding.
     *
     * @return A SemanticPartSegmentation dictionary that contains height, width,
     * the flattened binary segmentation mask and the pose for the person. The
     * width and height correspond to the same dimensions of the input image.
     * - `height`: The height of the person part segmentation data in pixel unit.
     * - `width`: The width of the person part segmentation data in pixel unit.
     * - `data`: The flattened Int32Array of person part segmentation data with a
     * part id from 0-24 for the pixels that are part of a corresponding body
     * part, and -1 otherwise. The size of the array is equal to `height` x
     * `width` in row-major order.
     * - `allPoses`: The 2d poses of all people.
     */
    segmentPersonParts(input: BodyPixInput, config?: PersonInferenceConfig): Promise<SemanticPartSegmentation>;
    /**
     * Given an image with multiple people, returns an *array* of PartSegmentation
     * object. Each element in the array corresponding to one
     * of the people in the input image. In other words, it predicts
     * instance-level multiple person body part segmentation & pose for each
     * person.
     *
     * This does standard ImageNet pre-processing before inferring through
     * the model. The image pixels should have values [0-255].
     *
     * @param input
     * ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement) The input
     * image to feed through the network.
     *
     * @param config MultiPersonInferenceConfig object that contains
     * parameters for the BodyPix inference using multi-person decoding.
     *
     * @return An array of PartSegmentation object, each containing a width,
     * height, a flattened array (with part id from 0-24 for the pixels that are
     * part of a corresponding body part, and -1 otherwise) and 2D pose. The width
     * and height correspond to the dimensions of the image. Each flattened part
     * segmentation array size is equal to `height` x `width`.
     */
    segmentMultiPersonParts(input: BodyPixInput, config?: MultiPersonInstanceInferenceConfig): Promise<PartSegmentation[]>;
    dispose(): void;
}
/**
 * Loads the BodyPix model instance from a checkpoint, with the ResNet
 * or MobileNet architecture. The model to be loaded is configurable using the
 * config dictionary ModelConfig. Please find more details in the
 * documentation of the ModelConfig.
 *
 * @param config ModelConfig dictionary that contains parameters for
 * the BodyPix loading process. Please find more details of each parameters
 * in the documentation of the ModelConfig interface. The predefined
 * `MOBILENET_V1_CONFIG` and `RESNET_CONFIG` can also be used as references
 * for defining your customized config.
 */
export declare function load(config?: ModelConfig): Promise<BodyPix>;
