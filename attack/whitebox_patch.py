import torch
import numpy as np
from torch.utils.data.dataloader import DataLoader
from config import Config
from attack.patch_attack import PatchAttack


class Whitebox_patch(PatchAttack):
    def __init__(self, task, model, model_name, patch_area, n_batch, batch_size, tb_logger=None, log_dir=None, tracker=None):
        super().__init__(task, model, model_name, patch_area, n_batch, batch_size, tb_logger, log_dir,tracker=None)


    def depth_loss(self, patch, scene, patch_only=True):
        b, _, h, w = scene.shape
        new_slice = (slice(0, b),) + self.patch_slice
        scene_patched = scene.clone().detach()
        scene_patched[new_slice] = patch
        est_disp = self.model(scene_patched)
        if patch_only:
            object_mask = torch.zeros([1, 1, h, w]) # est_mag: [B, 1, H, W]
            new_slice = (slice(0, 1), slice(0, 1),) + self.patch_slice[1:]
            object_mask[new_slice] = 1
            object_mask = object_mask.to(Config.device)
            loss = torch.sum(torch.abs(est_disp) * object_mask) / (b * torch.sum(object_mask))
        else:
            loss = torch.mean(torch.abs(est_disp))

        self.query_times += b

        return loss


    def flow_loss(self, patch, frame1, frame2, patch_only=True):
        b, _, h, w = frame1.shape

        flow_ori = self.model(frame1, frame2)
        new_slice = (slice(0, b), slice(0, 2),) + self.patch_slice[1:]
        flow_ori[new_slice] = 0
        flow_ori = flow_ori.detach()


        frame1_patched = frame1.clone().detach()
        frame2_patched = frame2.clone().detach()
        new_slice = (slice(0, b),) + self.patch_slice
        frame1_patched[new_slice] = patch
        frame2_patched[new_slice] = patch
        flow_at = self.model(frame1_patched, frame2_patched)
        diff = flow_ori - flow_at
        diff = torch.square(diff[:, [0], :, :]) + torch.square(diff[:, [1], :, :])
        # diff = torch.abs(diff)
        # diff = diff[:, 0, :, :] + diff[:, 1, :, :]
        # diff = diff.unsqueeze(1)
        if patch_only:
            object_mask = torch.zeros([b, 1, h, w]) # est_mag: [B, 1, H, W]
            new_slice = (slice(0, b), slice(0, 1),) + self.patch_slice[1:]
            object_mask[new_slice] = 1
            object_mask = object_mask.to(Config.device)
            assert diff.shape == object_mask.shape
            epe = torch.sum(diff * object_mask) / (torch.sum(object_mask))
        else:
            epe = torch.mean(diff)
        
        self.query_times += b

        return 1 / (1 + epe)
    

    def whitebox_patch_attack(self, n_iters, alpha=0.1, num_pos=3, patch_only=True):
        print(f'whitebox patch attack, iters={n_iters}, alpha={alpha}, patch_only={patch_only}')
        _, _, p_h, p_w = self.patch_area
        # scene_loader
        scene_loader = DataLoader(self.train_dataset, batch_size=Config.train_scenes, shuffle=False,\
                                    num_workers=2, pin_memory=True, drop_last=True)
        scene_loader_iter = iter(scene_loader)

        
        if self.task == 'OF':
            present_frame1, present_frame2 = next(scene_loader_iter) # tensor[trail,3,h,w]
            c = present_frame1.shape[1]
        else:
            scenes, _ = next(scene_loader_iter)
            c = scenes.shape[1]
        
        patch_curr = np.random.choice([-alpha, alpha], size=[c, 1, p_w])
        patch_curr = np.repeat(patch_curr, p_h, axis=1) # np.arrary[c,p_h,p_w]
        patch_curr += 0.5
        patch_curr = np.clip(patch_curr, 0, 1)
        patch_curr = self.numpy2tensor(patch_curr)
        patch_curr = patch_curr.to(Config.device)
        
        # optimizer
        patch_curr.requires_grad_(True)
        optimizer = torch.optim.Adam([patch_curr], lr=Config.lr, betas=(Config.beta1, Config.beta2), eps=Config.eps)

        for i_iter in range(n_iters):
            optimizer.zero_grad()
            patch_curr.data = torch.clip(patch_curr.data, 0, 1)
            indice = torch.randperm(Config.train_scenes)[:num_pos]
            if self.task == 'MDE':
                scene = scenes[indice]
                loss = self.depth_loss(patch_curr, scene.to(Config.device), patch_only)
            elif self.task == 'OF':
                frame1 = present_frame1[indice]
                frame2 = present_frame2[indice]
                loss = self.flow_loss(patch_curr, frame1.to(Config.device), frame2.to(Config.device), patch_only)
            loss.backward()
            optimizer.step()

            # log
            self.log(patch_curr, loss.item(), None, None, i_iter, 'whitebox', log_gap=1 ,img_gap=1000, patch_only=patch_only)
            # evaluation
            self.evaluate(patch_curr, i_iter, 'whitebox', log_gap=50, img_gap=1000, patch_only=patch_only)
