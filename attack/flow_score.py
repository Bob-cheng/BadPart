from config import Config
import torch
import numpy as np

from torchvision import transforms
from PIL import Image as pil
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from torchvision.transforms import Resize


class FlowScore(object):
    def __init__(self, model, model_name, img_set, n_batch, batch_size, patch_area, tracker) -> None:
        self.model = model
        self.model_name = model_name
        self.img_set = img_set
        self.n_batch = n_batch
        self.batch_size = batch_size
        self.query_times = 0
        p_t, p_l, p_h, p_w = patch_area
        self.patch_slice = (slice(0, 3), slice(p_t, p_t+p_h), slice(p_l, p_l+p_w))
        self.ori_flow = []
        self.tracker = tracker
        if self.tracker != None:
            self.detection = 0
            self.resize = Resize(Config.blacklight_shape)

        self.patch_flow_slice = (slice(0, self.batch_size), slice(0, 2), slice(p_t, p_t+p_h), slice(p_l, p_l+p_w))
        with torch.no_grad():
            for i in range(self.n_batch):
                frame1, frame2 = self.img_set[i][0], self.img_set[i][1]

                # original
                flow_up_ref = self.model(frame1.to(Config.device), frame2.to(Config.device))
                # print(len(flow_up_ref))
                # print(flow_up_ref[0].shape)
                flow_up_ref[self.patch_flow_slice] = 0
                self.ori_flow.append(flow_up_ref)


    def numpy2tensor(self, x):
        if len(x.shape) == 3:
            x = torch.from_numpy(x).unsqueeze(0)
        elif len(x.shape) == 4:
            x = torch.from_numpy(x)
        return x
    

    def MSE(self, f0, f1, seperate=False, mean=True, patch_only=False, epe=False):
        diff = f1 - f0
        del f0, f1

        diff = torch.square(diff[:, [0], :, :]) + torch.square(diff[:, [1], :, :])

        if epe:
            diff = torch.sqrt(diff)
            
        if patch_only:
            b, _, h, w = diff.shape
            object_mask = torch.zeros([1, 1, h, w]) # est_mag: [B, 1, H, W]
            new_slice = (slice(0, 1), slice(0, 1),) + self.patch_slice[1:]
            object_mask[new_slice] = 1
            object_mask = object_mask.repeat(b, 1, 1, 1).to(Config.device)
            if mean:
                if seperate:
                    mse = torch.sum(diff * object_mask, dim=[1, 2, 3]) / torch.sum(object_mask, dim=[1, 2, 3])
                else:
                    mse = torch.sum(diff * object_mask) / torch.sum(object_mask)
            else:
                mse = diff

        else:
            if mean:
                if seperate:
                    mse = torch.mean(diff, dim=[1, 2, 3])
                else:
                    mse = torch.mean(diff)
            else:
                mse = diff

        return mse


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
                patch: numpy.arrary [c, p_h, p_w] or torch.tensor[1, c, p_h, p_w]

            output:
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
                frame1, frame2 = self.img_set[i][0].clone(), self.img_set[i][1].clone()

                if self.tracker != None and train:
                    scene_noise = torch.rand(*frame1.shape)
                    scene_noise = 2 * (scene_noise - 0.5) * Config.disturbance_weight
                    frame1 = torch.clip(frame1 + scene_noise, 0, 1)
                    frame2 = torch.clip(frame2 + scene_noise, 0, 1)

                frame1[new_slice] = patch
                frame2[new_slice] = patch

                if self.tracker != None and train:
                    frame1_tracker = frame1.clone()
                    frame1_tracker = self.resize(frame1_tracker)
                    for j in range(self.batch_size):
                        match_num = self.tracker.add_img(frame1_tracker[j:j+1,:,:,:])
                        if(match_num > Config.tracker_threshold):
                            self.detection += 1

                flow_up = self.model(frame1.to(Config.device), frame2.to(Config.device)) # [batch_size,2,h,w]
                mse = self.MSE(self.ori_flow[i], flow_up, patch_only=patch_only, epe=epe).item()

                if epe:
                    score_list.append(mse)
                    mean_error = self.MSE(self.ori_flow[i], flow_up, patch_only=not patch_only, epe=True).item()
                    error_whole.append(mean_error)
                else:
                    loss = 1 / (1 + mse)
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
            output:
                score_map: numpy.arrary [1,1,h,w]
        '''
        with torch.no_grad():
            patch = self.numpy2tensor(patch)
            score_map = []
            new_slice = (slice(0, self.batch_size),) + self.patch_slice
            for i in range(self.n_batch):
                frame1, frame2 = self.img_set[i][0].clone(), self.img_set[i][1].clone()

                frame1[new_slice] = patch
                frame2[new_slice] = patch
                flow_up = self.model(frame1.to(Config.device), frame2.to(Config.device))
                
                prob_map = self.MSE(self.ori_flow[i], flow_up, mean=False) # [1,1,h,w]

                score_map.append(prob_map)

            # Concatenate along the batch dimension and compute the mean
            score_map = torch.cat(score_map, dim=0)
            score_map = score_map.mean(dim=0, keepdim=True)

        return score_map.squeeze().detach().cpu().numpy() # [H_scene, W_scene]


    def gredient_sample(self, patches_candi, patch_curr, frame1, frame2, patch_only=False):
        '''
            input:
                candi_patches: np.arrary[trail, c, p_h, p_w]
                patch_curr: np.arrary[c, p_h, p_w]
                frame1: tensor[1, c, h, w]
                frame2: tensor[1, c, h, w]
        '''
        with torch.no_grad():
            patches_candi = self.numpy2tensor(patches_candi) # [trail,3,h,w]
            patch_curr = self.numpy2tensor(patch_curr) # [1,3,h,w]
            trail = patches_candi.shape[0]

            ## the flow of benchmark
            if self.model_name in [ 'PWC-Net']:
                flow_up_bench = self.model(frame1.to(Config.device), frame2.to(Config.device)) #[1,2,h,w]
            else:
                flow_up_bench = self.model(frame1.repeat(trail,1,1,1).to(Config.device), frame2.repeat(trail,1,1,1).to(Config.device)) #[trail,2,h,w]
            
            # the flow of patch_curr
            patched_curr1, patched_curr2 = frame1.clone(), frame2.clone()
            patched_curr1 = patched_curr1.repeat(trail,1,1,1)
            patched_curr2 = patched_curr2.repeat(trail,1,1,1)
            if self.tracker != None:
                scene_noise = torch.rand(*patched_curr1.shape)
                scene_noise = 2 * (scene_noise - 0.5) * Config.disturbance_weight
                patched_curr1 = torch.clip(patched_curr1 + scene_noise, 0, 1)
                patched_curr2 = torch.clip(patched_curr2 + scene_noise, 0, 1)
            new_slice = (slice(0,trail),) + self.patch_slice
            patched_curr1[new_slice] = patch_curr
            patched_curr2[new_slice] = patch_curr
            flow_up_ref = self.model(patched_curr1.to(Config.device), patched_curr2.to(Config.device)) #[1,2,h,w]

            patched_curr1[new_slice] = patches_candi
            patched_curr2[new_slice] = patches_candi

            if self.tracker != None:
                patched_frames1_tracker = patched_curr1.clone()
                patched_frames1_tracker = self.resize(patched_frames1_tracker)
                for j in range(trail):
                    match_num = self.tracker.add_img(patched_frames1_tracker[j:j+1,:,:,:])
                    if(match_num > Config.tracker_threshold):
                        self.detection += 1

            flow_up = self.model(patched_curr1.to(Config.device), patched_curr2.to(Config.device)) #[trail,2,h,w]

            score_ref = self.MSE(flow_up_bench, flow_up_ref, seperate=True, patch_only=patch_only) # tensor[1,trail]
            score_candi = self.MSE(flow_up_bench, flow_up, seperate=True, patch_only=patch_only) # tensor[1,trail]

            scores = score_candi - score_ref  # the higher the better for the difference of acs

            self.query_times += trail
        return scores


    def viz(self, patch):

        frame1, frame2 = self.img_set[0][0][0:1,:,:,:].clone(), self.img_set[0][1][0:1,:,:,:].clone() # [1,3,H,W]
        flow_ref = self.ori_flow[0][0:1,:,:,:].clone() # [1,2,H,W]
        flow_ori = self.model(frame1.to(Config.device), frame2.to(Config.device))

        new_slice = (slice(0, 1),) + self.patch_slice
        frame1_patched, frame2_patched = frame1.clone(), frame2.clone()
        frame1_patched[new_slice] = patch
        frame2_patched[new_slice] = patch

        with torch.no_grad():
            flow_up = self.model(frame1_patched.to(Config.device), frame2_patched.to(Config.device))

        difference = flow_up - flow_ori
        difference_patched = flow_up - flow_ref

        flow_ori = flow_ori[0].permute(1,2,0).detach().cpu().numpy()
        flow_ref = flow_ref[0].permute(1,2,0).detach().cpu().numpy()
        flow_up = flow_up[0].permute(1,2,0).detach().cpu().numpy()
        difference = difference[0].permute(1,2,0).detach().cpu().numpy()
        difference_patched = difference_patched[0].permute(1,2,0).detach().cpu().numpy()
        print(f'difference max:{np.abs(difference).max()}')

        # map flow to rgb image
        flow_ori = flow_viz.flow_to_image(flow_ori)
        flow_up = flow_viz.flow_to_image(flow_up)
        difference = flow_viz.flow_to_image(difference, difference=True)
        difference_patched = flow_viz.flow_to_image(difference_patched, difference=True)

        frame1 = transforms.ToPILImage()(frame1.squeeze())
        frame2 = transforms.ToPILImage()(frame2.squeeze())
        frame1_patched = transforms.ToPILImage()(frame1_patched.squeeze())
        frame2_patched = transforms.ToPILImage()(frame2_patched.squeeze())

        fig: Figure = plt.figure(figsize=(12, 9)) # width, height
        plt.subplot(421); plt.imshow(frame1); plt.title('original frame1'); plt.axis('off')
        plt.subplot(422); plt.imshow(frame1_patched); plt.title('patched frame1'); plt.axis('off')
        plt.subplot(423); plt.imshow(frame2); plt.title('original frame2'); plt.axis('off')
        plt.subplot(424); plt.imshow(frame2_patched); plt.title('patched frame2'); plt.axis('off')
        plt.subplot(425); plt.imshow(flow_ori); plt.title('original flow'); plt.axis('off')
        plt.subplot(426); plt.imshow(flow_up); plt.title('attack flow'); plt.axis('off')
        plt.subplot(427); plt.imshow(difference); plt.title('difference(ori)'); plt.axis('off')
        plt.subplot(428); plt.imshow(difference_patched); plt.title('difference(patched)'); plt.axis('off')
        fig.canvas.draw()
        pil_image = pil.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        plt.close()

        return pil_image
