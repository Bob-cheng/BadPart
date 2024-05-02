# BadPart: Unified Black-box Adversarial Patch Attacks against Pixel-wise Regression Tasks

This is the official PyTorch implementation of our paper "BadPart: Unified Black-box Adversarial Patch Attacks against Pixel-wise Regression Tasks".

## Table of Contents

1. [Environment preparation](#environment-preparation)
2. [Code preparation](#code-preparation)
3. [Dataset preparation](#dataset-preparation)
5. [Configuration preparation](#configuration-preparation)
6. [Launch black-box attacks](#launch-black-box-attacks)
7. [Evaluate the patch](#evaluate-the-patch)
8. [Attack the Google online service](#attack-the-google-online-service)

## Environment preparation

Create a new conda environment of Python 3.8 called BadPart:
```
conda create -n BadPart python=3.8
conda activate BadPart
```

Install required packages:
- pytorch=1.11
- torchvision=0.12.0
- numpy=1.24.3
- tensorboard
- tensorboardx
- ...

## Code preparation

Clone this repository to folder `~/BadPart`

```
cd ~
git clone <repo_url> BadPart
cd BadPart
```

Prepare the target MDE networks ([Monodepth2](https://github.com/nianticlabs/monodepth2), [DepthHints](https://github.com/nianticlabs/depth-hints), [SQLDepth](https://github.com/hisfog/SfMNeXt-Impl), [PlaneDepth](https://github.com/svip-lab/PlaneDepth)) following their official instructions and put them in the directory of `DepthNetworks`. Download their official pretrained model weights (with the highest input resolution and best performance) into a sub-folder named `models` inside each network's directory (e.g., `DepthNetworks/monodepth2/models`).

Also, prepare the target OFE networks by cloning this [Repository](https://github.com/anuragranj/flowattack) to `FlowNetworks/flow_models`. Download the official pretrained model weights (including FlowNetC, FlowNet2 and PWC-Net) into the directory of `FlowNetworks/flow_models/pretrained`. 

The directories should be organized as:
```
BadPart
├── DepthNetworks
    ├── depth-hints
    ├── monodepth2
    ├── PlaneDepth
    ├── SQLdepth
├── FlowNetworks
    ├── flow_models
        ├── pretrained
```

## Dataset preparation
You will need to download the [KITTI flow dataset](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow) and organize it in the following way. Assume the path of the dataset is `/path/to/dataset/KITTI/flow/`.

```
KITTI
├── flow
    ├── testing
    ├── training
    ├── devkit
```


## Configuration preparation
Provide the log path and the dataset path in the file `config.py`:
```
kitti_dataset_root = "/path/to/dataset/KITTI/flow/"
log_dir = "/path/to/logs"
```

## Launch black-box attacks

Run the following code to launch `BadPart` on `Monodepth2`:

```
CUDA_VISIBLE_DEVICES=0 python main.py \
    --model_name monodepth2 \
    --attack_method ours \
    --patch_ratio 0.02 \
    --batch_size 5 \
    --n_batch 1 \
    --countermeasure None\
    --n_iter 10001 \
    --trail 20 \
    --p_init 0.025 \
    --p_sche v6 \
    --patch_only \
    --test_name name_for_this_test
```

You can change the target model with the option `--model_name` and available models are:
- MDE models: 
    - Monodepth2    -> `monodepth2`
    - DepthHints    -> `depthhints`
    - SQLDepth      -> `SQLdepth`
    - PlaneDepth    -> `planedepth`
- OFE models:
    - FlowNetC      -> `FlowNetC`
    - FlowNet2      -> `FlowNet2`
    - PWC-Net       -> `PWC-Net`

You can also change the attack method with the option `--attack_method` and available methods are:
- BadPart   -> `ours`
- Sparse-RS -> `S-RS`
- HardBeat  -> `hardbeat`
- GenAttack -> `GA_attack`
- White-Box -> `whitebox`

For detailed explanations of each options, please refer to the file `options.py`

You can see the visulized attack performance using Tensorboard by running:

```
tensorboard --logdir /path/to/logs --samples_per_plugin images=200
```

The generated patch file is saved to folder `/path/to/logs/name_for_this_test/` by default.
 
## Evaluate the patch
You can evaluate the attack performance of the generated universal adversarial patch by runnning `eval.py`:
```
CUDA_VISIBLE_DEVICES=0 python eval.py \
    --model_name SQLdepth \
    --patch_path /path/to/your/patch \
    --patch_ratio 0.01 \
    --batch_size 5 \
```
Prepare your adversarial patch and replace `/path/to/your/patch` with the actual path to your patch.

You can still change the target model with the option `--model_name`, but this target should be the same as the target you used to generate the patch.

## Attack the Google online service

### Step 1. Install node.js

Create another conda environment named `node`:
```
conda create -n node python=3.8
conda activate node
```
Install node.js using conda:
```
conda install -c conda-forge nodejs=20.9
```

### Step 2. Start the API server
Start the API server by running the following command in the `node` environment.
```
cd ~/BadPart/nodejs_depthe
CUDA_VISIBLE_DEVICES=0 node api_server_fast.js
```
Now the API server is running and waiting for the request from the client side. 

To make the API use GPU while inference, the cuDNN version on your machine should be 8.1.0 in order to match the tensorflow.js requirement. Otherwise, you may have to install the version 8.1.0 in your conda environment and add change the library path:
```
conda install conda-forge::cudnn=8.1
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

In the worst case, you could use CPU but it is much slower.

### Step 3. Launch the attack
Open another terminal and activate the `BadPart` conda environment:
```
conda activate BadPart
```
Run the following code to launch the attack. It will query the above API for portrait depth estimation.
```
CUDA_VISIBLE_DEVICES=0 python main.py \
    --model_name google_api \
    --attack_method ours \
    --patch_ratio 0.02 \
    --batch_size 1 \
    --n_batch 1 \
    --n_iter 10001 \
    --trail 20 \
    --square_steps 200\
    --p_init 0.1 \
    --p_sche v6 \
    --targeted_attack \
    --patch_only \
    --test_name name_for_this_test
```
You can also view the attack performance on Tensorboard.