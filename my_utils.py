import torch
from PIL import Image as pil, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from torchvision import transforms
from attack.depth_model import DepthModelWrapper
import os
import random
from scipy.special import softmax
import math



def disp_to_depth(disp,min_depth,max_depth):
# """Convert network's sigmoid output into depth prediction
# The formula for this conversion is given in the 'additional considerations'
# section of the paper.
# """
    min_disp=1/max_depth
    max_disp=1/min_depth
    scaled_disp=min_disp+(max_disp-min_disp)*disp
    depth=1/scaled_disp
    return scaled_disp,depth

def depth_to_disp(depth, min_depth, max_depth):
    scalar = 5.4
    min_disp=1/max_depth
    max_disp=1/min_depth
    scaled_disp = 1 / torch.clip(torch.clip(depth, 0, max_depth) / scalar, min_depth, max_depth)
    disp = (scaled_disp - min_disp) / (max_disp-min_disp)
    return disp

def softmax_parent_selection(fitness, num_parents, ga_instance):
    softmax_fit = softmax(fitness)
    selected_idxs = np.random.choice(np.arange(len(fitness)), num_parents, p=softmax_fit)
    parents = ga_instance.population[selected_idxs, :].copy()
    return parents, selected_idxs

def prob_uniform_crossover(parents, offspring_size, ga_instance):
    offspring = np.empty(offspring_size)
    for i in range(offspring.shape[0]):
        gene_len = parents.shape[1]
        fitnesses = ga_instance.last_generation_fitness[ga_instance.last_generation_parents_indices]
        p1 = fitnesses[0] / np.sum(fitnesses)
        p2 = 1-p1
        idxs = np.random.choice(np.array([0, 1]), gene_len, p=np.array([p1, p2]))
        for j in range(gene_len):
            offspring[i, j] = parents[idxs[j], j]
    return offspring

def get_mean_depth_diff(adv_disp1, ben_disp2, scene_car_mask=None, use_abs=False, separate=False):
    scaler=5.4
    if scene_car_mask == None:
        scene_car_mask = torch.ones_like(adv_disp1)
    dep1_adv=torch.clamp(disp_to_depth(torch.abs(adv_disp1),0.1,100)[1]*scene_car_mask*scaler,max=100)
    dep2_ben=torch.clamp(disp_to_depth(torch.abs(ben_disp2),0.1,100)[1]*scene_car_mask*scaler,max=100)
    if not separate:
        mean_depth_diff = torch.sum(torch.abs(dep1_adv-dep2_ben))/torch.sum(scene_car_mask) if use_abs \
            else torch.sum(dep1_adv-dep2_ben)/torch.sum(scene_car_mask)
    else:
        mean_depth_diff = torch.sum(torch.abs(dep1_adv-dep2_ben), dim=[1,2,3])/torch.sum(scene_car_mask, dim=[1,2,3]) if use_abs \
            else torch.sum(dep1_adv-dep2_ben, dim=[1,2,3])/torch.sum(scene_car_mask, dim=[1,2,3])
    return mean_depth_diff

def eval_depth_diff(img1: torch.tensor, img2: torch.tensor, depth_model, filename=None, disp1=None, disp2=None):
    if disp1 == None:
        disp1 = depth_model(img1).detach().cpu().squeeze().numpy()
    else:
        disp1 = disp1.detach().cpu().squeeze().numpy()
    if disp2 == None:
        disp2 = depth_model(img2).detach().cpu().squeeze().numpy()
    else:
        disp2 = disp2.detach().cpu().squeeze().numpy()
    image1 = transforms.ToPILImage()(img1.squeeze())
    image2 = transforms.ToPILImage()(img2.squeeze())
    diff_disp = np.abs(disp1 - disp2)
    vmax = np.percentile(disp1, 95)
    
    fig: Figure = plt.figure(figsize=(12, 7)) # width, height
    plt.subplot(321); plt.imshow(image1); plt.title('Image 1'); plt.axis('off')
    plt.subplot(322); plt.imshow(image2); plt.title('Image 2'); plt.axis('off')
    plt.subplot(323)
    plt.imshow(disp1, cmap='magma', vmax=vmax, vmin=0); plt.title('Disparity 1'); plt.axis('off')
    plt.subplot(324)
    plt.imshow(disp2, cmap='magma', vmax=vmax, vmin=0); plt.title('Disparity 2'); plt.axis('off')
    plt.subplot(325)
    plt.imshow(diff_disp, cmap='magma', vmax=vmax, vmin=0); plt.title('Disparity difference'); plt.axis('off')
    plt.subplot(326)
    plt.imshow(diff_disp, cmap='magma'); plt.title('Disparity difference (scaled)'); plt.axis('off')
    fig.canvas.draw()
    if filename != None:
        plt.savefig('temp_' + filename + '.png')
    pil_image = pil.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    plt.close()
    return pil_image, disp1, disp2

def eval_depth_diff_jcl(img1: torch.tensor, img2: torch.tensor, depth_model, filename=None, disp1=None, disp2=None):
    if disp1 == None:
        disp1 = depth_model(img1).detach().cpu().squeeze().numpy()
    else:
        disp1 = disp1.detach().cpu().squeeze().numpy()
    if disp2 == None:
        disp2 = depth_model(img2).detach().cpu().squeeze().numpy()
    else:
        disp2 = disp2.detach().cpu().squeeze().numpy()
    image1 = transforms.ToPILImage()(img1.squeeze())
    image2 = transforms.ToPILImage()(img2.squeeze())
    diff_disp = np.abs(disp1 - disp2)
    vmax = np.percentile(disp1, 95)
    
    fig: Figure = plt.figure(figsize=(12, 7)) # width, height
    plt.subplot(311); plt.imshow(image1); plt.title('Image 1'); plt.axis('off')
    # plt.subplot(322); plt.imshow(image2); plt.title('Image 2'); plt.axis('off')
    plt.subplot(312)
    plt.imshow(disp1, cmap='magma', vmax=vmax, vmin=0); plt.title('Disparity 1'); plt.axis('off')
    # plt.subplot(324)
    # plt.imshow(disp2, cmap='magma', vmax=vmax, vmin=0); plt.title('Disparity 2'); plt.axis('off')
    plt.subplot(313)
    plt.imshow(diff_disp, cmap='magma', vmax=vmax, vmin=0); plt.title('Disparity difference'); plt.axis('off')
    # plt.subplot(326)
    # plt.imshow(diff_disp, cmap='magma'); plt.title('Disparity difference (scaled)'); plt.axis('off')
    fig.canvas.draw()
    if filename != None:
        plt.savefig('temp_' + filename + '.png')
    pil_image = pil.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    plt.close()
    return pil_image

def find_neighbor(sim_graph, curr_idx, hist_score): # return the best fit neighbor
    best_idx = -1
    best_metric = 0
    n = len(hist_score)
    for i in range(n):
        if i == curr_idx:
            continue
        metric = sim_graph[curr_idx][i] * hist_score[i]
        if metric > best_metric:
            best_metric = metric
            best_idx = i
    return best_idx

def save_depth_model(model: DepthModelWrapper, log_folder, epoch, width=1024, height=320, use_stereo=True):
    """Save model weights to disk
    """
    save_folder = os.path.join(log_folder, "models", "weights_{}".format(epoch))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    save_path = os.path.join(save_folder, "{}.pth".format('encoder'))
    to_save = model.encoder.state_dict()
    # save the sizes - these are needed at prediction time
    to_save['height'] = height
    to_save['width'] = width
    to_save['use_stereo'] = use_stereo
    torch.save(to_save, save_path)

    save_path = os.path.join(save_folder, "{}.pth".format('depth'))
    to_save = model.decoder.state_dict()
    torch.save(to_save, save_path)

def save_pic(tensor, i, log_dir=''):
    unloader = transforms.ToPILImage() # tensor to PIL image
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    if log_dir != '':
        file_path = os.path.join(log_dir, "{}.png".format(i))
    else:
        file_path = "{}.png".format(i)
    image.save(file_path, "PNG")


def color_trans(object_tensor):

    # the shape of object_tensor should be [3, H, W]
    color_jitter = transforms.ColorJitter(brightness=(0.9, 1.1), 
                                            contrast=(0.9, 1.1), 
                                            saturation=(0.9, 1.1), 
                                            hue=(-0.5, 0.5))

    jittered_tensor = color_jitter(object_tensor)

    return jittered_tensor


def read_scene_img(img_path, bottom_gap, side_crop, scene_size):
    H, W = scene_size
    scene_size = (W, H)
    img1_origin = pil.open(img_path).convert('RGB')
    img1_resize = resize_and_crop_img(img1_origin, scene_size, bottom_gap, side_crop)
    img1_tensor = transforms.ToTensor()(img1_resize).unsqueeze(0).to(torch.device("cuda"))
    return img1_tensor, img1_resize

def resize_and_crop_img(img, size_WH, bottom_gap, side_crop):
    original_w, original_h = img.size
    img = img.crop((side_crop[0], 0, original_w-side_crop[1], original_h))
    original_w, original_h = img.size
    scale = size_WH[0] / original_w
    img = img.resize((size_WH[0], int(original_h * scale)))
    now_w, now_h = img.size
    left = 0
    bottom = now_h - bottom_gap
    top = bottom - size_WH[1]
    img = img.crop((0, top, size_WH[0], bottom))
    return img

def draw_a_patch(img: pil, pos = (400, 100), WH = (30, 20)):
    shape = [pos, (pos[0] + WH[0], pos[1] + WH[1])] # top-left (w-idx, h-idx), bottom-right (w-idx, h-idx)
    img1 = ImageDraw.Draw(img)  
    img1.rectangle(shape, fill ="#00A36C", outline ="red")
    return img

def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # torch.use_deterministic_algorithms(True)


def noise_weight_decay_linear(init_weight, curr_iter, total_iters):

    return init_weight - (init_weight / total_iters) * curr_iter


def normalize_score(data):
    
    max_positive = data[data > 0].max(initial=0)
    max_negative = data[data < 0].min(initial=0)
    
    normalized_data = np.zeros_like(data)
    
    mask_positive = data > 0
    mask_negative = data < 0
    
    normalized_data[mask_positive] = data[mask_positive] / max_positive if max_positive != 0 else 0
    normalized_data[mask_negative] = data[mask_negative] / abs(max_negative) if max_negative != 0 else 0
    
    return normalized_data


def get_patch_area(attack_task, scene_size, ratio, height_pos=3/4):
    H, W = scene_size
    s = int(math.sqrt(H * W * ratio))
    if attack_task == 'OF':
        # assert ratio <= 0.25
        i, j = int(H * height_pos), int(W / 2)
    elif attack_task == 'MDE':
        i, j = int(H * height_pos), int(W / 2)
    p_t = i - int(s / 2)
    p_l = j - int(s / 2)

    if p_t < 0:
        p_t = 0
    if p_t + s > H:
        p_t = H - s
    if p_l < 0:
        p_l = 0
    if p_l + s > W:
        p_l = W - s

    return (p_t, p_l, s, s)


def EPE(f0, f1, object_mask=None):
        diff = f1 - f0
        del f0, f1
        diff = torch.sqrt(torch.square(diff[:, [0], :, :]) + torch.square(diff[:, [1], :, :]))
            
        if object_mask is None:
            epe = torch.mean(diff)
        else:
            epe = torch.sum(diff * object_mask) / torch.sum(object_mask)

        return epe