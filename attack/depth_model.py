import os
import sys
import torch
import torch.nn as nn
import json
import numpy as np
sys.path.append('.')
from config import Config

import requests
from PIL import Image
import io
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

def depth_to_disp(depth, min_depth, max_depth):
    scalar = 5.4
    min_disp=1/max_depth
    max_disp=1/min_depth
    scaled_disp = 1 / torch.clip(torch.clip(depth, 0, max_depth) / scalar, min_depth, max_depth)
    disp = (scaled_disp - min_disp) / (max_disp-min_disp)
    return disp

file_dir = os.path.dirname(os.path.realpath(__file__))
file_dir = os.path.dirname(file_dir)

class GoogleAPIWrapper(torch.nn.Module):
    def __init__(self) -> None:
        super(GoogleAPIWrapper, self).__init__()
        self.url = 'http://localhost:9302/estimate-depth'
        self.executor = ThreadPoolExecutor(max_workers=20)
    
    def send_post(self, np_img):
        # print("np_img:")
        # print(type(np_img))
        # np_img = np.squeeze(np_img, axis=0)
        np_img = np.transpose(np_img, (1, 2, 0))
        image = Image.fromarray(np_img)
        
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')  # 或者使用其他格式，如 PNG
        img_byte_arr = img_byte_arr.getvalue()
        # 发送字节流数据
        response = requests.post(self.url, files={'imageData': img_byte_arr})
        if response.status_code == 200:
            depth_map = np.frombuffer(response.content, dtype=np.float32)
            # depth_map = np.array(response.json()['depthArray'])

            # 将其重塑为 [1, 1, h, w]
            # print(depth_map.shape)
            depth_map_reshaped = depth_map.reshape(1, 1, Config.input_H_GoogleAPI, Config.input_W_GoogleAPI)
            return depth_map_reshaped
        else:
            print("Error:", response.text)
            return None

    def forward(self, input_image):
        # return self.forward_serial(input_image)
        return self.forward_parallel(input_image)
        
    def forward_parallel(self, input_image):
        device = input_image.device
        res_list = []
        numpy_array = input_image.cpu().numpy()
        numpy_array = np.round(numpy_array * 255).astype(np.uint8)
        futures = [self.executor.submit(self.send_post, np_img) for np_img in numpy_array]
        for future in futures:
            depth_map_reshaped = future.result()
            res_list.append(depth_map_reshaped)
        res_array = np.vstack(res_list)
        print(np.array(res_array).shape)
        res_array = torch.from_numpy(res_array)
        # #reverse the target, encourate further
        # res_array[res_array == 0] = 0.01
        # res_array = - res_array + 1

        # save_pic(depth_map_reshaped, "api_img")
        # disp_viz(res_array[0], 'api_test.png')
        res_array = res_array.to(device)
        return res_array
    
    def forward_serial(self, input_image):
        res_list = []
        numpy_array = input_image.cpu().numpy()
        numpy_array = np.round(numpy_array * 255).astype(np.uint8)
        list_data = numpy_array.tolist()
        for img in list_data:
            # print(time.time())
            #print(np.array(img).shape)
            np_img = np.array(img)
            if np_img.dtype != np.uint8:
                np_img = np_img.astype(np.uint8)
            # 调整通道顺序为 (高度, 宽度, 通道数)
            np_img = np.transpose(np_img, (1, 2, 0))
            image = Image.fromarray(np_img)
            # print(time.time())
            # 将图像转换为字节流
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')  # 或者使用其他格式，如 PNG
            img_byte_arr = img_byte_arr.getvalue()
            # 发送字节流数据
            # print(time.time())
            response = requests.post(self.url, files={'imageData': img_byte_arr})
            #response = requests.post(self.url, json={'imageData': img})
            if response.status_code == 200:
                depth_data = response.json()['depthArray']
                # 将其重塑为 [1, 1, h, w]
                depth_map = np.array(depth_data)
                # print(depth_map.shape)
                depth_map_reshaped = depth_map.reshape(1, 1, depth_map.shape[0], depth_map.shape[1])
                # 输出新的形状查看
                # print(depth_map_reshaped.shape)  # 应该输出: (1, 1, h, w)
                res_list.append(depth_map_reshaped)
            else:
                print("Error:", response.text)
        res_array = np.vstack(res_list)
        print(np.array(res_array).shape)
        res_array = torch.from_numpy(res_array)

        res_array[res_array == 0] = 0.01
        # 2. 将所有数变为倒数
        # depth_map_reshaped = 1 / depth_map_reshaped
        res_array = - res_array + 1

        # print(depth_map_reshaped)
        # save_pic(depth_map_reshaped, "api_img")
        res_array = res_array.cuda()
        # disp_viz(res_array[0], 'api_test.png')
        
        return res_array


    
def load_and_preprocess_intrinsics(intrinsics_path, resize_width, resize_height):
    K = np.eye(4)
    with open(intrinsics_path, 'r') as f:
        K[:3, :3] = np.array(json.load(f))

    # Convert normalised intrinsics to 1/4 size unnormalised intrinsics.
    # (The cost volume construction expects the intrinsics corresponding to 1/4 size images)
    K[0, :] *= resize_width // 4
    K[1, :] *= resize_height // 4

    invK = torch.Tensor(np.linalg.pinv(K)).unsqueeze(0)
    K = torch.Tensor(K).unsqueeze(0)

    if torch.cuda.is_available():
        return K.cuda(), invK.cuda()
    return K, invK

class DepthModelWrapper(torch.nn.Module):
    def __init__(self, encoder, decoder) -> None:
        super(DepthModelWrapper, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, input_image):
        features = self.encoder(input_image)
        outputs = self.decoder(features)
        disp = outputs[("disp", 0)]
        # print(disp.shape)
        return disp

class SQLdepthModelWrapper(torch.nn.Module):
    def __init__(self, encoder, decoder) -> None:
        super(SQLdepthModelWrapper, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, input_image):
        features = self.encoder(input_image)
        outputs = self.decoder(features)
        disp = outputs[("disp", 0)]
        disp = nn.functional.interpolate(disp, input_image.shape[-2:], mode='bilinear', align_corners=True)
        # print(disp.shape)
        disp = depth_to_disp(disp, 0.1, 100)
        return disp

class PlaneDepthModelWrapper(torch.nn.Module):
    def __init__(self, encoder, decoder) -> None:
        super(PlaneDepthModelWrapper, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, input_color):
        grid = torch.meshgrid(torch.linspace(-1, 1, Config.input_W_PD), torch.linspace(-1, 1, Config.input_H_PD), indexing="xy")
        # grid = torch.meshgrid(torch.linspace(-1, 1, Config.input_W_PD), torch.linspace(-1, 1, Config.input_H_PD))
        # grid = [_.T for _ in grid]
        grid = torch.stack(grid, dim=0)
        grids = grid[None, ...].expand(input_color.shape[0], -1, -1, -1).cuda()
        output = self.decoder(self.encoder(input_color), grids)
        pred_disp = output["disp"]
        # pred_disp = output["disp"][:, 0]
        # print(pred_disp.shape)
        pred_disp = (pred_disp - 0.7424) / 741.6576
        return pred_disp


def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def import_depth_model(scene_size, model_type='monodepth2'):
    """
    import different depth model to attack:
    possible choices: monodepth2
    """
    if scene_size == (320, 1024):
        if model_type == 'monodepth2':
            model_name = 'mono+stereo_1024x320'
            code_path = os.path.join(file_dir, 'DepthNetworks', 'monodepth2')
            depth_model_dir = os.path.join(code_path, 'models')
            sys.path.append(code_path)
            import networks
        elif model_type == 'depthhints':
            model_name = 'DH_MS_320_1024'
            code_path = os.path.join(file_dir, 'DepthNetworks', 'depth-hints')
            depth_model_dir = os.path.join(code_path, 'models')
            sys.path.append(code_path)
            import networks
        elif model_type == 'SQLdepth':
            model_name = 'ConvNeXt_Large_SQLdepth'
            code_path = os.path.join(file_dir, 'DepthNetworks', 'SQLdepth')
            depth_model_dir = os.path.join(code_path, 'models')
            sys.path.append(code_path)
            import networks
        else:
            raise RuntimeError("depth model unfound")
    elif scene_size == (384, 1280):
        if model_type == 'planedepth':
            model_name = 'PD_distill_384_1280'
            code_path = os.path.join(file_dir, 'DepthNetworks', 'PlaneDepth')
            depth_model_dir = os.path.join(code_path, 'models')
            sys.path.append(code_path)
            import networks
    elif scene_size == (Config.input_H_GoogleAPI, Config.input_W_GoogleAPI):
        if model_type == 'google_api':
            api_model = GoogleAPIWrapper()
            # initilize google api model
            return api_model
    else:
        raise RuntimeError(f"scene size undefined! {scene_size}")
    model_path = os.path.join(depth_model_dir, model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    if model_type == 'monodepth2' or model_type == 'depthhints':
        loaded_dict_enc = torch.load(encoder_path, map_location='cpu')
        encoder = networks.ResnetEncoder(18, False)
        
        # extract the height and width of image that this model was trained with
        feed_height = loaded_dict_enc['height']
        feed_width = loaded_dict_enc['width']
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
        encoder.load_state_dict(filtered_dict_enc)

        
        print("   Loading pretrained decoder")
        depth_decoder = networks.DepthDecoder(
            num_ch_enc=encoder.num_ch_enc, scales=range(4))

        loaded_dict = torch.load(depth_decoder_path, map_location='cpu')
        depth_decoder.load_state_dict(loaded_dict)

        depth_model = DepthModelWrapper(encoder, depth_decoder)
    elif model_type == 'planedepth':
        encoder_dict = torch.load(encoder_path)
        encoder = networks.ResnetEncoder(50, False)
        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, 
                                                49, 
                                                2., 
                                                300., 
                                                8, 
                                                pe_type="neural",
                                                use_denseaspp=True, 
                                                xz_levels=14,
                                                yz_levels=0, 
                                                use_mixture_loss=True, 
                                                render_probability=False, 
                                                plane_residual=True)

        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(depth_decoder_path))

        depth_model = PlaneDepthModelWrapper(encoder, depth_decoder)
    
    elif model_type == 'SQLdepth':
        encoder = networks.Unet(
            pretrained=False, 
            backbone='convnext_large', 
            in_channels=3, 
            num_classes=32, 
            # decoder_channels=[1536, 768, 384, 192, 96])
            decoder_channels=[1024, 512, 256, 128])
            # decoder_channels=[1024, 512, 256, 128, 128])
        print("   Loading pretrained decoder")
        depth_decoder = networks.Depth_Decoder_QueryTr(in_channels=32, patch_size=32, dim_out=64, embedding_dim=32,
                                                        query_nums=64, num_heads=4, min_val=0.001, max_val=80)
        loaded_dict_enc = torch.load(encoder_path, map_location='cpu')
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
        encoder.load_state_dict(filtered_dict_enc)

        loaded_dict_enc = torch.load(depth_decoder_path, map_location='cpu')
        # filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.depth_decoder.state_dict()}
        # self.depth_decoder.load_state_dict(filtered_dict_enc)
        depth_decoder.load_state_dict(loaded_dict_enc)
        depth_model = SQLdepthModelWrapper(encoder, depth_decoder)
    return depth_model


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np
    from PIL import Image as pil
    from torchvision import transforms
    from config import Config
    from my_utils import read_scene_img, draw_a_patch

    model = 'SQLdepth' # 'depthhints'
    scene_size = (Config.input_H, Config.input_W)

    # model='planedepth'
    # scene_size = (Config.input_H_PD, Config.input_W_PD)

    depth_model = import_depth_model(scene_size, model).to(Config.device).eval()
    img_path = ''

    # crop image before resize
    bottom_gap = 230          
    side_crop = [100, 500]
    _, img = read_scene_img(img_path, bottom_gap, side_crop, scene_size)

    img = draw_a_patch(img, pos=(435, 170), WH=(110, 75))

    assert img.size[::-1] == scene_size
    img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(Config.device)
    with torch.no_grad():
        disp = depth_model(img_tensor)
        print(disp.size(), disp.max(), disp.min(), torch.median(disp))
        disp_np = disp.squeeze().cpu().numpy()
    
    vmax = np.percentile(disp_np, 95)
    plt.figure(figsize=(5,5))
    plt.subplot(211)
    plt.imshow(img, cmap='magma', vmax=vmax)
    plt.title('RGB Image')
    plt.axis('off')
    plt.subplot(212)
    plt.imshow(disp_np, cmap='magma', vmax=vmax)
    plt.title('Disparity')
    plt.axis('off')
    plt.savefig('temp_test.png')