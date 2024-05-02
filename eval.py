import argparse
import torch
import numpy as np
# from torchvision.utils import save_image
from torchvision import transforms
from my_utils import set_random_seed, get_mean_depth_diff, EPE
from config import Config
from my_utils import get_patch_area
import matplotlib.pyplot as plt
from attack.dataset import KittiDataset
from torch.utils.data.dataloader import DataLoader


def parse():
    parser = argparse.ArgumentParser(description='Define hyperparameters.')
    parser.add_argument('--model_name', type=str, choices=['monodepth2', 'planedepth', 'depthhints', 'SQLdepth', 'FlowNetC', 'PWC-Net','FlowNet2'], required=True, help='name of the subject model.')
    parser.add_argument('--attack_method', type=str, choices=['ours', 'GA_attack', 'S-RS', 'hardbeat', 'whitebox'], required=True, help='name of the attack method.')
    parser.add_argument('--patch_path', type=str, required=True, help='path of the adversarial patch.')
    parser.add_argument('--seed', type=int, default=1, help='Random Seed')
    parser.add_argument('--n_batch', type=int, default=40, help='number of ppictures for evaluation')
    parser.add_argument('--batch_size', type=int, default=5, help='batch size for evaluation')
    parser.add_argument('--patch_ratio', type=float, default=0.02, help='patch ratio')
    args = parser.parse_args()
    return args

def disp_viz(disp: torch.tensor, path, difference=False):
    disp = disp.detach().cpu().squeeze().numpy()
    if difference:
        plt.imshow(disp, cmap='magma')
        plt.axis('off')
        plt.savefig('eval/MDE/disp_difference_Scaled.png', bbox_inches='tight', pad_inches=0)
    vmax = np.percentile(disp, 95)
    plt.imshow(disp, cmap='magma', vmax=vmax, vmin=0)
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    
    plt.close()

def flow_visualize(flow: torch.tensor, path, difference=False):
    flow = flow[0].permute(1,2,0).detach().cpu().numpy()
    flow = flow_viz.flow_to_image(flow, difference=difference)
    plt.imshow(flow)
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()

def flow_visualize_v2(frame1: torch.tensor, flow_ori: torch.tensor, flow_attack: torch.tensor, path):
    frame1 = transforms.ToPILImage()(frame1.squeeze())
    flow_ori = flow_ori[0].permute(1,2,0).detach().cpu().numpy()
    flow_attack = flow_attack[0].permute(1,2,0).detach().cpu().numpy()
    flow_diff = flow_ori - flow_attack
    flow_ori = flow_viz.flow_to_image(flow_ori, difference=True)
    flow_attack = flow_viz.flow_to_image(flow_attack, difference=True)
    flow_diff = flow_viz.flow_to_image(flow_diff, difference=True)
    # plt.imshow(flow)
    # plt.axis('off')
    # plt.savefig(path, bbox_inches='tight', pad_inches=0)
    # plt.close()
    fig: Figure = plt.figure(figsize=(8, 4)) # width, height
    plt.subplot(221); plt.imshow(flow_ori); plt.title('original'); plt.axis('off')
    plt.subplot(222); plt.imshow(flow_attack); plt.title('attack'); plt.axis('off')
    plt.subplot(223); plt.imshow(flow_diff); plt.title('difference'); plt.axis('off')
    plt.subplot(224); plt.imshow(frame1); plt.title('frame1'); plt.axis('off')
    fig.canvas.draw()
    plt.savefig(path, pad_inches=0)
    plt.close()


def main(args):
    set_random_seed(args.seed, deterministic=True)

    model_name  = args.model_name

    if model_name in ['FlowNetC','PWC-Net','FlowNet2']:
        attack_task = 'OF' 
    elif model_name in ['depthhints','monodepth2','planedepth','SQLdepth']:
        attack_task = 'MDE'
    else:
        raise RuntimeError('The attack model is not supported!')
    scene_size  = Config.model_scene_sizes_WH[model_name]
    patch_area = get_patch_area(attack_task, scene_size, args.patch_ratio)

    eval_dataset = KittiDataset(model_name, main_dir=Config.kitti_dataset_root, mode='testing')
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)

    p_t, p_l, p_h, p_w = patch_area
    patch_slice = (slice(0, args.batch_size), slice(0, 3), slice(p_t, p_t + p_h), slice(p_l, p_l + p_w))

    error_patch_list = []
    error_whole_list = []

    if attack_task == 'MDE':
        # import depth model
        from attack.depth_model import import_depth_model
        model = import_depth_model(scene_size, model_name).to(Config.device).eval()
        patch = torch.load(args.patch_path)
        patch = torch.from_numpy(patch).unsqueeze(0)

        for i, (scene, _) in enumerate(eval_loader):
            print(i)
            if i == args.n_batch:
                break
            with torch.no_grad():
                disp_ref = model(scene.to(Config.device))
                # disp_viz(disp_ref, 'eval/MDE/original_disparity.png')

                scene[patch_slice] = patch
                # save_image(scene, 'eval/MDE/scene_patched.png')
                disp = model(scene.to(Config.device))
                
                patch_mask = torch.zeros_like(disp)
                patch_mask[patch_slice] = 1
                mean_depth_error_patch = get_mean_depth_diff(disp, disp_ref, patch_mask)
                mean_depth_error_whole = get_mean_depth_diff(disp, disp_ref)
                error_patch_list.append(mean_depth_error_patch)
                error_whole_list.append(mean_depth_error_whole)

        error_in_patch = torch.mean(torch.stack(error_patch_list)).item()
        error_in_scene = torch.mean(torch.stack(error_whole_list)).item()

        print(f'{args.attack_method}, attack model: {model_name}')
        print(f'{args.attack_method} mean depth error in patch area: {error_in_patch}')
        print(f'{args.attack_method} mean depth error in whole scene: {error_in_scene}')

    elif attack_task == 'OF':
        from attack.flow_model import import_optical_flow_model
        # import flow model
        model = import_optical_flow_model(Config.optical_flow_model_path[model_name], args).to(Config.device).eval()
        patch = torch.load(args.patch_path)
        patch = torch.from_numpy(patch).unsqueeze(0)
        error_patch_list = []
        error_whole_list = []
        for i, (frame1, frame2) in enumerate(eval_loader):
            # if i == args.n_batch:
            #     break
            print(i)
            with torch.no_grad():
                flow_ref = model(frame1.to(Config.device), frame2.to(Config.device))
                frame1[patch_slice] = patch
                frame2[patch_slice] = patch
                # flow_visualize(flow_ref, 'visualization/ori_flow/'+ str(i) +'_ori.png')

                flow = model(frame1.to(Config.device), frame2.to(Config.device))
                
                patch_mask = torch.zeros_like(flow)
                patch_mask[patch_slice] = 1
                mean_depth_error_patch = EPE(flow, flow_ref, patch_mask)
                mean_depth_error_whole = EPE(flow, flow_ref)
                error_patch_list.append(mean_depth_error_patch)
                error_whole_list.append(mean_depth_error_whole)

        error_in_patch = torch.mean(torch.stack(error_patch_list)).item()
        error_in_scene = torch.mean(torch.stack(error_whole_list)).item()

        print(f'{args.attack_method}, attack task: {model_name}')
        print(f'EPE in patch area: {error_in_patch}')
        print(f'EPE in whole scene: {error_in_scene}')


if __name__ == "__main__":
    args = parse()
    main(args)
    
'''
# eval
CUDA_VISIBLE_DEVICES=1 python eval.py \
    --model_name SQLdepth \
    --attack_method ours \
    --patch_ratio 0.01 \
    --batch_size 5
'''
