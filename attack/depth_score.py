from config import Config
import torch
import numpy as np

from torchvision import transforms
from PIL import Image as pil
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from torchvision.transforms import Resize
# from torchvision.utils import save_image

class DepthScore(object):
    def __init__(self, model, model_name, img_set, n_batch, batch_size, patch_area, tracker) -> None:
        self.model = model
        self.model_name = model_name
        self.img_set = img_set
        self.n_batch = n_batch
        self.batch_size = batch_size
        self.query_times = 0
        self.patch_area = patch_area
        p_t, p_l, p_h, p_w = patch_area
        self.patch_slice = (slice(0, 3), slice(p_t, p_t+p_h), slice(p_l, p_l+p_w))
        self.ori_disp = []
        self.disp_target = torch.zeros(1).float().to(Config.device)
        self.tracker = tracker
        if self.tracker != None:
            self.detection = 0
            self.resize = Resize(Config.blacklight_shape)

        with torch.no_grad():
            new_slice = (slice(0,batch_size),) + self.patch_slice
            for i in range(self.n_batch):
                scene = self.img_set[i]

                # pseudo ground truth
                scene_copy = scene.clone()
                if model_name != 'google_api':
                    scene_copy[new_slice] = 0 # 0 for black patch, 1 for white patch
                disp_ref = self.model(scene_copy.to(Config.device))
                self.ori_disp.append(disp_ref)


    def numpy2tensor(self, x):
        if len(x.shape) == 3:
            x = torch.from_numpy(x).unsqueeze(0)
        elif len(x.shape) == 4:
            x = torch.from_numpy(x)
        return x


    def score_creteria(self, est_disp, seperate=False, patch_only=False):
        '''
            the lower the better attacking performance
        '''
        if patch_only:
            b, _, h, w = est_disp.shape
            object_mask = torch.zeros([1, 1, h, w]) # est_mag: [B, 1, H, W]
            new_slice = (slice(0, 1), slice(0, 1),) + self.patch_slice[1:]
            # p_t, p_l, p_h, p_w = self.patch_area
            # new_slice = (slice(0, 1), slice(0, 1), slice(p_t-p_h, p_t+p_h), slice(p_l-p_w, p_l+p_w))
            object_mask[new_slice] = 1
            object_mask = object_mask.repeat(b, 1, 1, 1).to(Config.device)
            
            if seperate:
                score = torch.sum(torch.abs(est_disp - self.disp_target) * object_mask, dim=[1, 2, 3]) / torch.sum(object_mask, dim=[1,2,3])
            else:
                score = torch.sum(torch.abs(est_disp - self.disp_target) * object_mask) / torch.sum(object_mask)
        else:
            if seperate:
                score = torch.mean(est_disp, dim=[1, 2, 3])
            else:
                score = torch.mean(est_disp)

        return score


    def disp_to_depth(self, disp, min_depth, max_depth):
        """ 
            Convert network's sigmoid output into depth prediction.
            The formula for this conversion is given in the 'additional considerations' section of the paper.
        """
        min_disp=1/max_depth
        max_disp=1/min_depth
        scaled_disp=min_disp+(max_disp-min_disp)*disp
        depth=1/scaled_disp
        return scaled_disp,depth


    def get_mean_depth_diff(self, adv_disp1, ben_disp2, patch_only=False, use_abs=False, separate=False):
        scaler=5.4
        if patch_only:
            scene_mask = torch.zeros_like(adv_disp1)
            scene_mask[(slice(0,scene_mask.shape[0]),)+self.patch_slice] = 1
            # p_t, p_l, p_h, p_w = self.patch_area
            # new_slice = (slice(0,scene_mask.shape[0]), slice(0, 3), slice(p_t-p_h, p_t+p_h), slice(p_l-p_w, p_l+p_w))
            # scene_mask[new_slice] = 1
        else:
            scene_mask = torch.ones_like(adv_disp1)
        dep1_adv=torch.clamp(self.disp_to_depth(torch.abs(adv_disp1),0.1,100)[1]*scene_mask*scaler,max=100)
        dep2_ben=torch.clamp(self.disp_to_depth(torch.abs(ben_disp2),0.1,100)[1]*scene_mask*scaler,max=100)
        if not separate:
            mean_depth_diff = torch.sum(torch.abs(dep1_adv-dep2_ben))/torch.sum(scene_mask) if use_abs \
                else torch.sum(dep1_adv-dep2_ben)/torch.sum(scene_mask)
        else:
            mean_depth_diff = torch.sum(torch.abs(dep1_adv-dep2_ben), dim=[1,2,3])/torch.sum(scene_mask, dim=[1,2,3]) if use_abs \
                else torch.sum(dep1_adv-dep2_ben, dim=[1,2,3])/torch.sum(scene_mask, dim=[1,2,3])
        return mean_depth_diff
            

    def ACS(self, f0:torch.tensor, f1:torch.tensor, seperate=False, mean=True):
        assert f1.shape == f0.shape, 'The shape is not the same for cosine similarity computation.'

        dot_product = torch.sum(f1 * f0, dim=1)
        
        f1_magnitude = torch.sqrt(torch.sum(f1 ** 2, dim=1))
        f0_magnitude = torch.sqrt(torch.sum(f0 ** 2, dim=1))

        epsilon = 1e-6
        f1_magnitude = torch.clamp(f1_magnitude, min=epsilon)
        f0_magnitude = torch.clamp(f0_magnitude, min=epsilon)

        cosine_similarity = dot_product / (f1_magnitude * f0_magnitude)
        if mean:
            if seperate:
                average_similarity = torch.mean(cosine_similarity, dim=[1, 2])
            else:
                average_similarity = torch.mean(cosine_similarity)
            return average_similarity
        else:
            return cosine_similarity.unsqueeze(1)

            
    def score(self, patch, epe=False, patch_only=False, train=True):
        '''
            input:
                patch: numpy.arrary [c, p_h, p_w]
            arguments:
                epe: If true, calculate the mean depth error, else, calculate the loss score
                patch_only: If true, only foucus on the patch area, else, foucus on the whole scene

            output:
                mean_error: float
                loss: float
        '''
        with torch.no_grad():
            if isinstance(patch, np.ndarray):
                patch = self.numpy2tensor(patch)
            score_list = []
            if epe:
                error_whole = []
            new_slice = (slice(0, self.batch_size),) + self.patch_slice
            for i in range(self.n_batch):
                scene = self.img_set[i].clone()

                if self.tracker != None and train:
                    scene_noise = torch.rand(*scene.shape)
                    scene_noise = 2 * (scene_noise - 0.5) * Config.disturbance_weight
                    scene = torch.clip(scene + scene_noise, 0, 1)

                scene[new_slice] = patch

                if self.tracker != None and train:
                    if Config.benign_rate:
                        scene_tracker = scene_noise
                    else:
                        scene_tracker = scene.clone()
                    scene_tracker = self.resize(scene_tracker)
                    for j in range(self.batch_size):
                        match_num = self.tracker.add_img(scene_tracker[j:j+1,:,:,:])
                        if(match_num > Config.tracker_threshold):
                            self.detection += 1

                disp = self.model(scene.to(Config.device)) # [batch_size,1,h,w]

                # compute mean depth error
                if epe:
                    mean_error1 = self.get_mean_depth_diff(disp, self.ori_disp[i], patch_only=patch_only).item()
                    mean_error2 = self.get_mean_depth_diff(disp, self.ori_disp[i], patch_only=not patch_only).item()
                    score_list.append(mean_error1)
                    error_whole.append(mean_error2)
                # compute loss score for updating
                else:
                    loss = self.score_creteria(disp, patch_only=patch_only, seperate=False).item()
                    score_list.append(loss)

            if train:
                self.query_times += self.n_batch * self.batch_size

        if epe:
            if patch_only:
                return np.mean(score_list), np.mean(error_whole)
            else:
                return np.mean(error_whole), np.mean(score_list)
        else:
            return np.mean(score_list)


    # finished
    def disp_diff_compute(self, patch):
        '''
            input:
                patch: numpy.arrary [c, p_h, p_w]
                scenes: tensor [B,c,h,w]
            output:
                score_map: numpy.arrary [1,1,h,w]
        '''
        with torch.no_grad():
            patch = self.numpy2tensor(patch)
            score_map = []
            new_slice = (slice(0, self.batch_size),) + self.patch_slice
            for i in range(self.n_batch):
                scene = self.img_set[i].clone()
                scene[new_slice] = patch
                disp = self.model(scene.to(Config.device)) # [B,1,h,w]
                prob_map = self.ori_disp[i] - disp
                score_map.append(prob_map)

            # Concatenate along the batch dimension and compute the mean
            score_map = torch.cat(score_map, dim=0)
            score_map = score_map.mean(dim=0, keepdim=True)

        return score_map.squeeze().detach().cpu().numpy() # [H_scene, W_scene]


    def gredient_sample(self, patches_candi, patch_curr, scene, patch_only=False):
        '''
            Input:
                patches_candi: np.arrary[trail, c, p_h, p_w]
                patch_curr: np.arrary[c, p_h, p_w]
                scene: tensor[1, c, h, w]
            Output:
                scores: tensor[1, trail]
        '''
        with torch.no_grad():
            patches_candi = self.numpy2tensor(patches_candi)
            patch_curr = self.numpy2tensor(patch_curr)
            trail = patches_candi.shape[0]
            scenes = scene.repeat(trail, 1, 1, 1)

            if self.tracker != None:
                scene = scene.repeat(trail, 1, 1, 1)
                scene_noise = torch.rand(*scene.shape)
                scene_noise = 2 * (scene_noise - 0.5) * Config.disturbance_weight
                scene = torch.clip(scene + scene_noise, 0, 1)
                scenes = torch.clip(scenes + scene_noise, 0, 1)
                new_slice = (slice(0,trail),) + self.patch_slice
            else:
                new_slice = (slice(0,1),) + self.patch_slice

            scene[new_slice] = patch_curr
            disp_ref = self.model(scene.to(Config.device))

            # if self.tracker != None:
            #     scene_noise = torch.rand(*scenes.shape)
            #     scene_noise = 2 * (scene_noise - 0.5) * Config.disturbance_weight
            #     scenes = torch.clip(scenes + scene_noise, 0, 1)
            new_slice = (slice(0,trail),) + self.patch_slice
            scenes[new_slice] = patches_candi
            if self.tracker != None:
                if Config.benign_rate:
                    scene_tracker = scene_noise
                else:
                    scene_tracker = scenes.clone()
                scene_tracker = self.resize(scene_tracker)
                for j in range(trail):
                    match_num = self.tracker.add_img(scene_tracker[j:j+1,:,:,:])
                    if(match_num > Config.tracker_threshold):
                        self.detection += 1

            disp = self.model(scenes.to(Config.device))
            
            # the higher the better
            score_ref = self.score_creteria(disp_ref, seperate=True, patch_only=patch_only) # tensor[1]
            score_candi = self.score_creteria(disp, seperate=True, patch_only=patch_only) # tensor[1,trail]

            scores = score_ref - score_candi   # the higher the better for the difference of acs

            self.query_times += trail
        return scores


    def viz(self, patch):
        with torch.no_grad():
            scene = self.img_set[0][0:1,:,:,:].clone() # [1,3,H,W]
            disp_ref = self.ori_disp[0][0:1,:,:,:].clone() # [1,1,H,W]
            disp_ori= self.model(scene.to(Config.device))
            new_slice = (slice(0, 1),) + self.patch_slice
            scene_patched = scene.clone()
            scene_patched[new_slice] = patch
            disp = self.model(scene_patched.to(Config.device))

        disp_ori = disp_ori.detach().cpu().squeeze().numpy()
        disp_ref = disp_ref.detach().cpu().squeeze().numpy()
        disp = disp.detach().cpu().squeeze().numpy()

        scene = transforms.ToPILImage()(scene.squeeze())
        scene_patched = transforms.ToPILImage()(scene_patched.squeeze())
        diff_disp = np.abs(disp_ori - disp)
        diff_disp_patched = np.abs(disp_ref - disp)
        vmax = np.percentile(disp_ori, 95)

        fig: Figure = plt.figure(figsize=(12, 7)) # width, height
        plt.subplot(321); plt.imshow(scene); plt.title('original scene'); plt.axis('off')
        plt.subplot(322); plt.imshow(scene_patched); plt.title('patched scene'); plt.axis('off')
        plt.subplot(323)
        plt.imshow(disp_ori, cmap='magma', vmax=vmax, vmin=0); plt.title('original disparity'); plt.axis('off')
        plt.subplot(324)
        plt.imshow(disp, cmap='magma', vmax=vmax, vmin=0); plt.title('attacked disparity'); plt.axis('off')
        plt.subplot(325)
        plt.imshow(diff_disp, cmap='magma', vmax=vmax, vmin=0); plt.title('Disparity difference (ori)'); plt.axis('off')
        plt.subplot(326)
        plt.imshow(diff_disp_patched, cmap='magma', vmax=vmax, vmin=0); plt.title('Disparity difference (patched)'); plt.axis('off')
        fig.canvas.draw()
        pil_image = pil.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        plt.close()

        return pil_image
    

