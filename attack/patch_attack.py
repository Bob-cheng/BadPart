import torch
import os
import logging
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import ToTensor, Resize
from torchvision.utils import save_image
from tensorboardX import SummaryWriter
from PIL import Image

from config import Config
from attack.dataset import KittiDataset


class PatchAttack:
    def __init__(self, task, model, model_name, patch_area, n_batch=1, batch_size=10, tb_logger: SummaryWriter=None, log_dir=None, tracker=None,targeted_attack=False):
        self.patch_area = patch_area # (t, l, h, w) --> t: top index, l: left index, w: width, h: height
        p_t, p_l, p_h, p_w = patch_area
        self.patch_slice = (slice(0, 3), slice(p_t, p_t+p_h), slice(p_l, p_l+p_w))
        self.task = task
        self.model = model
        self.tb_logger = tb_logger
        self.model_name = model_name
        self.train_eval_set = []
        self.test_set = []
        self.square_size = p_h
        self.query_times = 0
        self.log_dir = log_dir
        self.tracker = tracker
        self.targeted_attack = targeted_attack

        if self.model_name == 'google_api':
            from attack.depth_score import DepthScore
            image_path = Config.api_portrait_image
            image_list = []
            image = Image.open(image_path + "/portrait" + str(5) + ".png").convert('RGB')
            image_tensor = ToTensor()(image)
            image_tensor = Resize((Config.input_H_GoogleAPI, Config.input_W_GoogleAPI))(image_tensor).unsqueeze(0)
            print("shape of image_tensor:")
            print(image_tensor.shape)

            self.train_eval_set.append(image_tensor) # [1, 3, h, w] 
            self.test_set.append(image_tensor)
            self.Score = DepthScore(model, model_name, self.train_eval_set, n_batch, batch_size, patch_area, tracker)
            self.eval_Score = DepthScore(model, model_name, self.test_set, n_batch, batch_size, patch_area, tracker)
            self.batch_size = 1
        else:
            if targeted_attack:
                self.train_dataset = KittiDataset(self.model_name, main_dir=Config.kitti_dataset_root, mode='training')
                self.train_loader = DataLoader(self.train_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)
                self.batch_size = 1
                assert n_batch == 1, 'n_batch must be 1 for targeted attack'
                if self.task == 'OF':
                    from attack.flow_score import FlowScore
                    for i, (frame1, frame2) in enumerate(self.train_loader):
                        if i < 9:
                            continue
                        if i == 9 + n_batch:
                            break
                        self.train_eval_set.append([frame1, frame2])
                        self.test_set.append([frame1, frame2])

                    self.Score = FlowScore(model, model_name, self.train_eval_set, n_batch, batch_size, patch_area, tracker)
                    self.eval_Score = FlowScore(model, model_name, self.test_set, n_batch, batch_size, patch_area, tracker)
                elif self.task == 'MDE':
                    from attack.depth_score import DepthScore
                    for i, (scenes, _) in enumerate(self.train_loader):
                        if i < 10:
                            continue
                        if i == 10 + n_batch:
                            break
                        self.train_eval_set.append(scenes)
                        self.test_set.append(scenes)
                    
                    self.Score = DepthScore(model, model_name, self.train_eval_set, n_batch, batch_size, patch_area, tracker)
                    self.eval_Score = DepthScore(model, model_name, self.test_set, n_batch, batch_size, patch_area, tracker)

            else:
                self.train_dataset = KittiDataset(self.model_name, main_dir=Config.kitti_dataset_root, mode='training')
                self.test_dataset = KittiDataset(self.model_name, main_dir=Config.kitti_dataset_root, mode='testing')
                self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)
                self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)

                # optical flow
                if self.task == 'OF':
                    from attack.flow_score import FlowScore

                    for i, (frame1, frame2) in enumerate(self.train_loader):
                        if i < 9:
                            continue
                        if i == 9 + n_batch:
                            break
                        self.train_eval_set.append([frame1, frame2])

                    for i, (frame1, frame2) in enumerate(self.test_loader):
                        if i == n_batch:
                            break
                        self.test_set.append([frame1, frame2])

                    self.Score = FlowScore(model, model_name, self.train_eval_set, n_batch, batch_size, patch_area, tracker)
                    self.eval_Score = FlowScore(model, model_name, self.test_set, n_batch, batch_size, patch_area, tracker)

                # depth estimation
                elif self.task == 'MDE':
                    from attack.depth_score import DepthScore

                    for i, (scenes, _) in enumerate(self.train_loader):
                        if i < 10:
                            continue
                        if i == 10 + n_batch:
                            break
                        self.train_eval_set.append(scenes)

                    for i, (scenes, _) in enumerate(self.test_loader):
                        if i == n_batch:
                            break
                        self.test_set.append(scenes)

                    self.Score = DepthScore(model, model_name, self.train_eval_set, n_batch, batch_size, patch_area, tracker)
                    self.eval_Score = DepthScore(model, model_name, self.test_set, n_batch, batch_size, patch_area, tracker)

        for param in self.model.parameters():
            param.requires_grad_(False)

    def numpy2tensor(self, x):
        if len(x.shape) == 3:
            x = torch.from_numpy(x).unsqueeze(0)
        elif len(x.shape) == 4:
            x = torch.from_numpy(x)
        return x


    def log(self, patch_curr, best_loss, noise_weight, steps_counter, i_iter, method, log_gap=20, img_gap=200, patch_only=False):
        # For logging
        if i_iter % log_gap == 0:
            log_info = "Current iteration: {}, Best loss: {}".format(i_iter, best_loss)
            logging.info(log_info)
            if not isinstance(self.tb_logger, type(None)):
                self.tb_logger.add_scalar(method + '_' + self.task +'/Best_loss', best_loss, i_iter)
                error_patch, error_whole = self.Score.score(patch_curr, epe=True, patch_only=patch_only, train=False)
                if self.task == 'OF':
                    logging.info(f'mean point error in patch area: {error_patch}, mean point error in whole area: {error_whole}')
                    self.tb_logger.add_scalar(method + '_OF/Mean_point_error_patch', error_patch, i_iter)
                    self.tb_logger.add_scalar(method + '_OF/Mean_point_error_whole', error_whole, i_iter)
                elif self.task == 'MDE':
                    logging.info(f'mean depth error in patch area: {error_patch}, mean depth error in whole area: {error_whole}')
                    self.tb_logger.add_scalar(method + '_MDE/Mean_depth_error_patch', error_patch, i_iter)
                    self.tb_logger.add_scalar(method + '_MDE/Mean_depth_error_whole', error_whole, i_iter)
                
                if method == 'whitebox':
                    self.tb_logger.add_scalar(method + '_' + self.task +'/Query_times', self.query_times, i_iter)
                else:
                    self.tb_logger.add_scalar(method + '_' + self.task +'/Query_times', self.Score.query_times, i_iter)
                    if self.tracker != None:
                        self.tb_logger.add_scalar(method + '_' + self.task + '/detection_rate', (self.Score.detection / self.Score.query_times), self.Score.query_times)
                        self.tb_logger.add_scalar(method + '_' + self.task + '/detection_count', self.Score.detection, self.Score.query_times)

                if method == 'ours':
                    self.tb_logger.add_scalar(method +'_' + self.task + '/Noise_weight', noise_weight, i_iter)
                    self.tb_logger.add_scalar(method +'_' + self.task +'/Search_steps', steps_counter, i_iter)
                    if i_iter % 100 == 0:
                        self.tb_logger.add_scalar(method + '_' + self.task + '/Square_size', self.square_size, i_iter)

                
        if i_iter % img_gap == 0 and self.tb_logger != None:
            if not isinstance(patch_curr, torch.Tensor):
                patch_curr = self.numpy2tensor(patch_curr)
            # self.tb_logger.add_image(method + '_' + self.task +'/curr_patch', best_patch.squeeze(0), i_iter)
            log_img = self.Score.viz(patch_curr)
            if self.task == 'OF':
                self.tb_logger.add_image(method + '_OF/Scene_flow', ToTensor()(log_img), i_iter)
            elif self.task == 'MDE':
                self.tb_logger.add_image(method + '_MDE/Scene_depth', ToTensor()(log_img), i_iter)
            logging.info("Image logged.")
        

    def evaluate(self, best_patch, i_iter, method, log_gap=20, img_gap=200, patch_only=False):
        if i_iter % log_gap == 0:
            error_patch, error_whole = self.eval_Score.score(best_patch, epe=True, patch_only=patch_only, train=False)
            if self.task == 'OF':
                logging.info(f'mean point error in patch area: {error_patch}, mean point error in whole area: {error_whole}')
                self.tb_logger.add_scalar(method + '_OF/eval_mean_point_error_patch', error_patch, i_iter)
                self.tb_logger.add_scalar(method + '_OF/eval_mean_point_error_whole', error_whole, i_iter)
                if method == 'whitebox':
                    self.tb_logger.add_scalar(method + '_OF/eval_mean_point_error_patch_quary', error_patch, self.query_times)
                    if self.query_times % 10000 == 0:
                        self.tb_logger.add_scalar(method + '_OF/eval_mean_point_error_patch_quary2', error_patch, self.query_times)
                else:
                    self.tb_logger.add_scalar(method + '_OF/eval_mean_point_error_patch_quary', error_patch, self.Score.query_times)
                    if self.Score.query_times % 10000 == 0:
                        self.tb_logger.add_scalar(method + '_OF/eval_mean_point_error_patch_quary2', error_patch, self.Score.query_times)
            elif self.task == 'MDE':
                logging.info(f'evaluation mean depth error in patch area: {error_patch}, evaluation mean depth error in whole area: {error_whole}')
                self.tb_logger.add_scalar(method + '_MDE/eval_mean_depth_error_patch', error_patch, i_iter)
                self.tb_logger.add_scalar(method + '_MDE/eval_mean_depth_error_whole', error_whole, i_iter)
                if method == 'whitebox':
                    self.tb_logger.add_scalar(method + '_MDE/eval_mean_depth_error_patch_quary', error_patch, self.query_times)
                else:
                    self.tb_logger.add_scalar(method + '_MDE/eval_mean_depth_error_patch_quary', error_patch, self.Score.query_times)
            if (self.Score.query_times % 50000) <= 1000:
                torch.save(best_patch, self.log_dir + '/' + self.model_name + '_' + method +'_'+ str(self.Score.query_times // 1000) +'K_best_patch.pt')
                    

        if i_iter % img_gap == 0 and self.tb_logger != None:
            if not isinstance(best_patch, torch.Tensor):
                best_patch = self.numpy2tensor(best_patch)
            self.tb_logger.add_image(method + '_' + self.task +'/eval_Best_patch', best_patch.squeeze(0), i_iter)
            if self.task == 'OF':
                log_img = self.eval_Score.viz(best_patch)
                self.tb_logger.add_image(method + '_OF/eval_Scene_flow', ToTensor()(log_img), i_iter)
            elif self.task == 'MDE':
                log_img = self.eval_Score.viz(best_patch)
                self.tb_logger.add_image(method + '_MDE/eval_Scene_depth', ToTensor()(log_img), i_iter)
            logging.info('Evaluation Image logged')