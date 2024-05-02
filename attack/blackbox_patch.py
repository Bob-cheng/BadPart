import torch
import logging
import numpy as np
from numpy.linalg import norm
import math
from torch.utils.data.dataloader import DataLoader
from skimage.transform import resize
from scipy.special import softmax
from attack.patch_attack import PatchAttack
from config import Config
from my_utils import normalize_score, softmax_parent_selection, find_neighbor



class Blackbox_patch(PatchAttack):
    def __init__(self, task, model, model_name, patch_area, n_batch=1, batch_size=5, tb_logger=None, log_dir=None, tracker=None, targeted_attack=False):
        super().__init__(task, model, model_name, patch_area, n_batch, batch_size, tb_logger, log_dir, tracker, targeted_attack)


    def p_selection(self, p_init, it, n_iters, version='v6'):
        """ Piece-wise constant schedule for p (the fraction of pixels changed on every iteration). """
        it = int(it / n_iters * 10000)
        if version == 'v1':
            sche = [10, 50, 200, 500, 1000, 2000, 4000, 6000, 8000, 10000]
        elif version == 'v2':
            sche = [100, 500, 1200, 2000, 4000, 8000, 9000, 9500, 9800, 10000]
        elif version == 'v3':
            return p_init
        elif version == 'v4':
            sche = [4000, 7000, 9000, 10000]
        elif version == 'v5':
            sche = [100, 250, 500, 1000, 3000]
        elif version == 'v6':
            sche = [100, 500, 1500, 3000, 5000, 10000]
        elif version == 'v7':
            # this is for SQLdepth
            sche = [100, 500, 3000, 5000, 10000]
        elif version == 'v8':
            sche = [100, 500, 4000, 8000, 10000]
        else:
            raise NotImplementedError("Schedule not implemented.")
        
        if it <= sche[0]:
            p = p_init
        elif sche[0] < it <= sche[1]:
            p = p_init / 2
        elif sche[1] < it <= sche[2]:
            p = p_init / 4
        elif sche[2] < it <= sche[3]:
            p = p_init / 8
        elif sche[3] < it <= sche[4]:
            p = p_init / 16
        elif sche[4] < it <= sche[5]:
            p = p_init / 32
        elif sche[5] < it <= sche[6]:
            p = p_init / 64
        elif sche[6] < it <= sche[7]:
            p = p_init / 128
        elif sche[7] < it <= sche[8]:
            p = p_init / 256
        elif sche[8] < it <= sche[9]:
            p = p_init / 512
        else:
            raise NotImplementedError("No such interval.")

        return p


    def create_patch(self, params, p_h, p_w, mode):
        if mode == 'square':
            params_np = np.array(params).reshape((-1, 6))
            params_np = params_np[params_np[:, 0].argsort()[::-1]]
            patch = np.zeros([3, p_h, p_w])
            max_edge_r = 0.3
            for i in range(params_np.shape[0]):
                s = max(1, math.ceil(max_edge_r * min(p_h, p_w) * params_np[i, 0]))
                h_start = int((p_h-s) * params_np[i, 1])
                w_start = int((p_w-s) * params_np[i, 2])
                patch[:, h_start:h_start + s, w_start:w_start + s] = np.reshape(params_np[i, 3:6], (3, 1, 1))
        elif mode == 'pixel':
            patch = np.array(params).reshape((3, p_h, p_w))
        return patch


    def BadPart(self, alpha=0.1, n_iters=10000, init_iters=100, p_init=0.025, p_sche='v6', square_steps=200, num_pos=1, trail=10, patch_only=False):
        print(f'attack task:{self.task}, atttack model:{self.model_name}, alpha={alpha}, p_init={p_init}, p_sche={p_sche}, num_pos={num_pos}, patch_only: {patch_only}')
        if  self.targeted_attack:
            if self.task == 'MDE':
                scenes = self.train_eval_set[0] # [1, 1, h, w]
                train_batch, c, scene_H, scene_W = scenes.shape
            elif self.task == 'OF':
                present_frame1, present_frame2 = self.test_set[0][0], self.test_set[0][1]
                train_batch, c, scene_H, scene_W = present_frame1.shape
        else:
            # scene_loader
            scene_loader = DataLoader(self.train_dataset, batch_size=Config.train_scenes, shuffle=False,\
                                        num_workers=2, pin_memory=True, drop_last=True)
            scene_loader_iter = iter(scene_loader)
            if self.task == 'OF':
                # c = 3
                present_frame1, present_frame2 = next(scene_loader_iter) # tensor[trail,3,h,w]
                train_batch, c, scene_H, scene_W = present_frame1.shape
            else:
                scenes, _ = next(scene_loader_iter)
                train_batch, c, scene_H, scene_W = scenes.shape

        # Initialize
        p_t, p_l, p_h, p_w = self.patch_area
        init_trail = trail
        n_patch_features = c * p_h * p_w
        patch_curr = np.random.choice([-alpha, alpha], size=[c, 1, p_w])
        patch_curr = np.repeat(patch_curr, p_h, axis=1) # np.arrary[c,p_h,p_w]
        patch_curr += 0.5
        patch_curr = np.clip(patch_curr, 0, 1) # the best vehicle png

        patch_best = patch_curr.copy()
        best_loss = self.Score.score(patch_best, patch_only=patch_only)  # the initial loss
        print(f'initial_score: {best_loss}')

        # counter during process
        stable_count = 0 # how many squares that failing to get lower loss
        stable_count_inSquare = 0 # how many steps that failing to get lower loss in a square
        
        # initalize the parameters
        eps = Config.eps
        init_noise_weight = Config.init_noise_weight
        min_noise_weight = Config.min_noise_weight # 0.03 for previous exp
        noise_weight = init_noise_weight
        threshold_betwSquare = Config.threshold_betwSquare[self.model_name] # 10 for depthhints
        threshold_inSquare = Config.threshold_inSquare[self.model_name] # 15 for previous exp
        lr = Config.lr # 0.1 default
        beta1 = Config.beta1 # 0.5 for default
        beta2 = Config.beta2 # 0.5 for default
        # topk = Config.topk # False: One Way, True: Best K. Note: One way is better than top 1.
        minus_mean = Config.minus_mean # should always be set to False
        AdaptiveWeight = Config.AdaptiveWeight # V1 is better than V2
        Weight_Normalization = Config.Weight_Normalization
        prob_norm_times = Config.prob_norm_times
        for i_iter in range(n_iters):
            # stable_count_inSquare = 0 # how many steps that failing to get lower loss in a square
            # present_frame1, present_frame2 = next(scene_loader_iter) # tensor[trail,3,h,w]
            init_loss = best_loss # store the init score in every square slice
            # noise weight adjustment
            if stable_count >= threshold_betwSquare and noise_weight > min_noise_weight:
                if not Config.fixed_Noiseweight:
                    noise_weight *= 0.98
                stable_count = 0

            # calculate the square area for this iteration
            p = self.p_selection(p_init, i_iter, n_iters, p_sche)
            s = int(round(np.sqrt(p * n_patch_features / c)))
            s = min(max(s, 1), p_h)  # at least c x 1 x 1 window is taken and at most c x h-1 x h-1
            self.square_size = s
            print(f'No.square:{i_iter} square size:{self.square_size} noise_weight:{noise_weight}')
            if i_iter <= init_iters:
                # threshold_inSquare = 1
                # randomly sample square
                loc_h = np.random.randint(0, p_h - s + 1)
                loc_w = np.random.randint(0, p_w - s + 1)
                square_slice = (slice(0, c), slice(loc_h, loc_h + s), slice(loc_w, loc_w + s))
                print(f"initialize: randomly choose square slice: {(loc_h, loc_h + s, loc_w, loc_w + s)}")
            else:
                # threshold_inSquare = Config.threshold_inSquare[self.model_name] # 15 for previous exp
                x_diff_map = self.Score.disp_diff_compute(patch_curr) #image_size:whole scene
                
                # the xdiff of disparity in patch area
                patch_prob_map = x_diff_map[(slice(p_t, p_t+p_h), slice(p_l, p_l+p_w))] # [p_h, p_w]

                # compute mean loss for every pixle
                for row in range(p_h):
                    for col in range(p_w):
                        row_index, col_index = row + p_t, col + p_l
                        row_top = row_index - math.floor(self.square_size / 2) if row_index - math.floor(self.square_size / 2) >= 0 else 0
                        row_bottom = row_index + math.ceil(self.square_size / 2) if row_index + math.ceil(self.square_size / 2) <= scene_H else scene_H
                        col_left = col_index - math.floor(self.square_size / 2) if col_index - math.floor(self.square_size / 2) >= 0 else 0
                        col_right = col_index + math.ceil(self.square_size / 2) if col_index + math.ceil(self.square_size / 2) <= scene_W else scene_W
                        # sample_slice = (slice(row_index - math.floor(self.square_size / 2), row_index + math.ceil(self.square_size / 2)), \
                        #                 slice(col_index - math.floor(self.square_size / 2), col_index + math.ceil(self.square_size / 2)))
                        sample_slice = (slice(row_top, row_bottom), slice(col_left, col_right))
                        patch_prob_map[row][col] = np.mean(x_diff_map[sample_slice])


                if np.max(patch_prob_map) > 0:
                    if prob_norm_times:
                        patch_prob_map = 3 * patch_prob_map / np.max(patch_prob_map)
                    else:
                        patch_prob_map =  patch_prob_map / np.max(patch_prob_map) # 1 for previous experiments

                # compute probability for every pixle
                patch_prob_map = softmax(patch_prob_map)

                # chose the idx according to the probability
                idx = np.random.choice(np.arange(len(patch_prob_map.flatten())), p=patch_prob_map.flatten())
                idx = np.unravel_index(idx, patch_prob_map.shape)
                row_index, col_index = idx[0], idx[1]
                h_start = row_index - math.floor(self.square_size / 2) if row_index - math.floor(self.square_size / 2) >= 0 else 0
                h_end = row_index + math.ceil(self.square_size / 2) if row_index + math.ceil(self.square_size / 2) <= p_h else p_h
                w_start = col_index - math.floor(self.square_size / 2) if col_index - math.floor(self.square_size / 2) >= 0 else 0
                w_end = col_index + math.ceil(self.square_size / 2) if col_index + math.ceil(self.square_size / 2) <= p_w else p_w
                square_slice = (slice(0, c), slice(h_start, h_end), slice(w_start, w_end))
                print(f"targeted square: {h_start, h_end, w_start, w_end}")

            if Config.AdaptiveTrail:
                adapted_trail = 3 * self.square_size
                trail = adapted_trail if adapted_trail < init_trail else init_trail
            patch_square = patch_curr[square_slice]  # the current best patch
            steps_counter = 0
            for i in range(1, square_steps):
                steps_counter += 1
                print('-' * 30)
                deltas = []
                avg_mean = []
                for j in range(num_pos): # for each position
                    if Config.noise_type == 'square':
                        noise = np.random.choice([-alpha, alpha], size=[trail, c, 1, 1])
                    elif Config.noise_type == 'discrete':
                        noise = np.random.choice([0, 1], size=[trail, *patch_square.shape])
                        noise = 2.0 * (noise - 0.5) * noise_weight
                    else:
                        noise = np.random.rand(trail, *patch_square.shape)
                        noise = 2.0 * (noise - 0.5) * noise_weight
                    squares_candi = np.clip(noise + patch_square, 0, 1)
                    noise = squares_candi - patch_square
                    #apply square noise to patch
                    patches_candi = patch_curr.copy()
                    patches_candi = patches_candi[None, ...]
                    patches_candi = np.repeat(patches_candi, trail, axis=0)
                    patches_candi[(slice(0,trail),)+square_slice] = squares_candi

                    if self.task == 'MDE':
                        indice = torch.randperm(train_batch)[:1]
                        scene = scenes[indice]
                        # scene = scenes[0:1,:,:,:]
                        scores = self.Score.gredient_sample(
                                                patches_candi, #numpy[trail,3,h,w]
                                                patch_curr, #numpy[1,3,h,w]
                                                scene, #tensor[1,3,h,w]
                                                patch_only)
                    elif self.task == 'OF':
                        indice = torch.randperm(train_batch)[:1]
                        frame_curr1, frame_curr2 = present_frame1[indice], present_frame2[indice]
                        # frame_curr1, frame_curr2 = present_frame1[0:1,:,:,:], present_frame2[0:1,:,:,:]
                        scores = self.Score.gredient_sample(
                                                patches_candi, #numpy[trail,3,h,w]
                                                patch_curr, #numpy[trail,3,h,w]
                                                frame_curr1, #tensor[1,3,h,w]
                                                frame_curr2, #tensor[1,3,h,w]
                                                patch_only)

                    if Weight_Normalization:
                        candi_y = normalize_score(scores.cpu().numpy())
                        if np.sum(np.abs(candi_y)) == 0:
                            candi_y = np.ones_like(candi_y)
                    else:
                        # raise RuntimeError("baseline normalize needs to be done!")
                        candi_y = scores.cpu().numpy()
                    mean_y = np.mean(candi_y)   # mean of scores
                    avg_mean.append(mean_y)

                    if mean_y == -1 or mean_y == 1:
                        delta = mean_y * np.mean(noise, axis=0)
                    else:
                        if minus_mean:
                            delta = np.mean(noise * (candi_y - mean_y).reshape((trail, 1, 1, 1)), axis=0)
                        else:
                            # delta = np.mean(noise[candi_y > 0] * candi_y[candi_y > 0].reshape((-1, 1, 1, 1)), axis=0)
                            if AdaptiveWeight == 'None':
                                delta = np.mean(noise * candi_y.reshape((trail, 1, 1, 1)), axis=0)
                            else:
                                pos_cnt = np.sum(candi_y > 0)
                                neg_cnt = trail - pos_cnt
                                # v1
                                if AdaptiveWeight == 'V1':
                                    candi_y[candi_y > 0] *= (2/(pos_cnt + eps))
                                    candi_y[candi_y <= 0] *= (2/(neg_cnt + eps))
                                # # v2
                                elif AdaptiveWeight == 'V2':
                                    if int(pos_cnt) != 0 and int(neg_cnt) != 0:
                                        candi_y[candi_y > 0] *= ((2*neg_cnt)/trail)
                                        candi_y[candi_y <= 0] *= ((2*pos_cnt)/trail)
                                delta = np.mean(noise * candi_y.reshape((trail, 1, 1, 1)), axis=0)
                    
                    if np.isnan(delta).any() or np.isinf(delta).any():
                        print("NaN or inf")

                    if norm(delta) == 0:
                        print("norm == 0")
                    
                    delta = delta / norm(delta)

                    deltas.append(delta)
                delta = np.mean(deltas, axis=0)
                gradf = torch.from_numpy(delta)
                avg_mean = np.mean(avg_mean)
                
                if Config.UseAdam: 
                    ## Adam optimizer
                    gradf_flat = gradf.flatten()
                    if i == 1:
                        grad_momentum = gradf
                        full_matrix   = torch.outer(gradf_flat, gradf_flat)
                    else:
                        grad_momentum = beta1 * grad_momentum + (1 - beta1) * gradf
                        full_matrix   = beta2 * full_matrix\
                                        + (1 - beta2) * torch.outer(gradf_flat, gradf_flat)
                    grad_momentum /= (1 - beta1 ** (i + 1))
                    full_matrix   /= (1 - beta2 ** (i + 1))
                    factor = 1 / torch.sqrt(eps + torch.diagonal(full_matrix))
                    gradf = (factor * grad_momentum.flatten()).reshape_as(gradf)
                else:
                    gradf *= math.sqrt(gradf.numel())

                patch_square = patch_square + lr * gradf.numpy()
                patch_square = np.clip(patch_square, 0, 1)
                if Config.Oneway:
                    patch_curr[square_slice] = patch_square
                else:
                    patch_temp = patch_curr.copy()
                    patch_temp[square_slice] = patch_square

                # current loss score
                if i % Config.gap == 0:
                    if Config.Oneway:
                        loss_curr = self.Score.score(patch_curr, patch_only=patch_only)
                    else:
                        loss_curr = self.Score.score(patch_temp, patch_only=patch_only)
                    print("iter", i, "score_curr", loss_curr)

                    # store best
                    if loss_curr < best_loss:
                        best_loss = loss_curr
                        if not Config.Oneway:
                            patch_curr[:] = patch_temp
                        patch_best[:] = patch_curr
                        stable_count_inSquare = 0
                    else:
                        stable_count_inSquare += Config.gap

                    if stable_count_inSquare >= threshold_inSquare:
                        stable_count_inSquare = 0
                        break

            if init_loss > best_loss:
                stable_count = 0
            else:
                stable_count += 1

            # log
            self.log(patch_best, best_loss, noise_weight, steps_counter, i_iter, 'ours', log_gap=Config.gap ,img_gap=1000, patch_only=patch_only)
            # evaluation
            self.evaluate(patch_best, i_iter, 'ours', log_gap=50, img_gap=1000, patch_only=patch_only)

    def sparse_RS(self, n_iters, p_init, p_sche='v1', patch_only=True):        
        print(f'sparse RS attack, iters={n_iters}, p_init={p_init}, p_sche={p_sche}, patch_only={patch_only}')
        c = 3
        _, _, p_h, p_w = self.patch_area
        n_patch_features = c * p_h * p_w
        s = int(round(np.sqrt(p_init * n_patch_features / c)))
        s = min(max(s, 1), p_h)  # at least c x 1 x 1 window is taken and at most c x h-1 x h-1
        self.square_size = s
        patch_curr = np.full((c, p_h, p_w), 0.) # initialized to 0
        for i in range(1000):
            init_s = np.random.randint(1, min(p_h, p_w))
            loc_h = np.random.randint(0, p_h - init_s + 1)
            loc_w = np.random.randint(0, p_w - init_s + 1)
            square_slice = (slice(0, c), slice(loc_h, loc_h + init_s), slice(loc_w, loc_w + init_s))
            # patch_curr[square_slice] = np.random.choice([-alpha, alpha], size=[c, 1, 1])
            patch_curr[square_slice] = np.random.choice([0., 1.], size=[c, 1, 1])
        patch_curr = np.clip(patch_curr, 0., 1.)
        patch_best = patch_curr.copy()
        best_loss = self.Score.score(patch_curr, patch_only=patch_only)
        cn_ft_idx = None # the index to start channel-wise finetune
        for i_iter in range(n_iters):
            patch_curr[:] = patch_best
            # calculate the square area for this iteration
            p = self.p_selection(p_init, i_iter, n_iters, p_sche)
            s = int(round(np.sqrt(p * n_patch_features / c)))
            s = min(max(s, 1), p_h)  # at least c x 1 x 1 window is taken and at most c x h-1 x h-1
            self.square_size = s
            loc_h = np.random.randint(0, p_h - s + 1)
            loc_w = np.random.randint(0, p_w - s + 1)
            if s == 1 and cn_ft_idx is None: # calculate the index to start channel-wise finetune
                cn_ft_idx = i_iter + (n_iters - i_iter) // 2
            if cn_ft_idx is not None and i_iter > cn_ft_idx: # channel-wise finetune stage
                loc_channel = np.random.randint(0, 3) 
                square_slice = (slice(loc_channel, loc_channel + 1), slice(loc_h, loc_h + s), slice(loc_w, loc_w + s))
            else: # normal stage
                square_slice = (slice(0, c), slice(loc_h, loc_h + s), slice(loc_w, loc_w + s))

            # generate new delta
            ## prevent trying out a delta if it doesn't change x_curr (e.g. an overlapping patch)
            while np.sum(np.abs(np.clip(patch_curr[square_slice], 0, 1) - patch_best[square_slice]) < 10**-7) == patch_best[square_slice].size:
                if cn_ft_idx is not None and i_iter > cn_ft_idx:
                    patch_curr[square_slice] = 1 - patch_curr[square_slice]
                else:
                    patch_curr[square_slice] += np.random.choice([-1., 1.], size=[c, 1, 1])
                patch_curr = np.clip(patch_curr, 0., 1.)
            # calculate and update the loss
            curr_loss = self.Score.score(patch_curr, patch_only=patch_only)
            print("-" * 30)
            logging.info(f'Current iteration: {i_iter}, Current loss: {curr_loss}')
            if curr_loss < best_loss:
                patch_best[:] = patch_curr
                best_loss = curr_loss

            # log
            self.log(patch_best, best_loss, None, None, i_iter, 'square_attack', log_gap=1 ,img_gap=1000, patch_only=patch_only)
            # evaluation
            self.evaluate(patch_best, i_iter, 'square_attack', log_gap=50, img_gap=1000, patch_only=patch_only)

    
    def GA_attack(self, num_generations = 10000, num_parents_mating = 5, sol_per_pop = 20, mode='square', patch_only=True): # mode: 'pixel' or 'square',
        import pygad
        # Note: x, x_best, x_new are numpy.ndarray in the range of [0, 1]
        C = 3
        p_t, p_l, p_h, p_w = self.patch_area
        if mode == 'pixel':
            i_h, i_w = 10, 10 # p_h, p_w
            n_patch_features = C * i_h * i_w
        elif mode == 'square':
            i_h, i_w = p_h, p_w
            squares = 500
            n_patch_features = 6 * squares
        
        def fitness_function(ga, solution, solution_idx):
            patch = self.create_patch(solution, i_h, i_w, mode)
            patch = resize(patch, (C, p_h, p_w))
            patch = np.clip(patch, 0., 1.)
            with torch.no_grad():
                curr_loss = self.Score.score(patch, patch_only=True)
            fitness = -curr_loss
            return fitness

        def on_generation_func(ga):
            solution, solution_fitness, solution_idx = ga.best_solution()
            patch = self.create_patch(solution, i_h, i_w, mode)
            patch = resize(patch, (C, p_h, p_w))
            patch = np.clip(patch, 0., 1.)
            i_iter = ga.generations_completed
            # log
            self.log(patch, -solution_fitness, None, None, i_iter, 'GA_attack', log_gap=1 ,img_gap=100, patch_only=True)
            # evaluation
            self.evaluate(patch, i_iter, 'GA_attack', log_gap=5, img_gap=100, patch_only=True)

        gene_space = {'low': 0., 'high': 1.} 
        ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=n_patch_features,
                       init_range_low=0.,
                       init_range_high=1.,
                       parent_selection_type=softmax_parent_selection, #"sss", # rank
                    #    K_tournament=5,
                       keep_elitism=1,
                    #    keep_parents=-1,
                       crossover_type="single_point", # prob_uniform_crossover, # "uniform",
                    #    crossover_probability=1,
                       mutation_type="random",
                    #    mutation_probability=0.3,
                       mutation_percent_genes=50,
                       random_mutation_min_val=-0.3,
                       random_mutation_max_val=0.3,
                       gene_space=gene_space,
                       random_seed=17,
                       on_generation=on_generation_func
                       )
        ga_instance.run()


    def hardbeat_attack(self, total_steps=10000, K=4, num_pos=100, num_init=500, trail=30, patch_only=True):
        print(f'hardbeat attack, attack model: {self.model_name}, k={K}, n_init={num_init}, trail={trail}, num_pos={num_pos}, patch_only: {patch_only}')
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
        p_t, p_l, p_h, p_w = self.patch_area
        # Initialize pattern
        patch_curr = np.random.rand(c, p_h, p_w)
        patch_best = patch_curr.copy()
        score_best = self.Score.score(patch_curr, patch_only=patch_only) # lower score is better
        
        # Initialize pattern
        for i in range(num_init):
            patch_curr = np.random.rand(c, p_h, p_w)
            score = self.Score.score(patch_curr, patch_only=patch_only)
            if score < score_best:
                score_best = score
                patch_best[:] = patch_curr
                print(f"Initialize: step: {i} best score: {score_best}")

        # Choose pattern:
        hist_patch = [patch_best]
        hist_score = np.array([1 / score_best])
        sim_graph = np.eye(100)
        last_mean_y = np.array([0] * num_pos)
        # last_avg_mean = 0
        beta1 = Config.beta1
        beta2 = Config.beta1
        eps = Config.eps
        lr = Config.lr
        for i in range(1, total_steps):
            if not Config.hardbeat_oneway:
                if len(hist_score) > K:
                    topk_idx = np.argpartition(hist_score, -K)[-K:]
                else:
                    topk_idx = np.arange(len(hist_score))
                topk_prob = softmax(hist_score[topk_idx])
                curr_idx = np.random.choice(topk_idx, size=1, p=topk_prob)[0]
                u = np.random.rand(1)[0]
                if u <= min(1, hist_score[curr_idx] / hist_score[-1]):
                    patch_curr = hist_patch[curr_idx]
                    if u <= 0.5 and i > 2:
                        neighbor_idx = find_neighbor(sim_graph, curr_idx, hist_score)
                        neighbor_patch = hist_patch[neighbor_idx]
                        alpha = np.random.rand(1)[0]
                        patch_curr = alpha * patch_curr + (1-alpha) * neighbor_patch
                else:
                    patch_curr = hist_patch[-1]
                    curr_idx = len(hist_patch) - 1

            ## gradient estimate:
            deltas = []
            avg_mean = []
            for j in range(num_pos): # for each position
                noise = np.random.rand(trail, *patch_curr.shape)
                noise = 2.0 * (noise - 0.5) * 0.5
                patches_candi = np.clip(noise + patch_curr, 0, 1)
                noise = patches_candi - patch_curr

                if self.task == 'MDE':
                    indice = torch.randperm(scenes.shape[0])[:1]
                    scene = scenes[indice]
                    # scene = scenes[0:1,:,:,:]
                    scores = self.Score.gredient_sample(
                                            patches_candi, #numpy[trail,3,h,w]
                                            patch_curr, #numpy[1,3,h,w]
                                            scene, #tensor[1,3,h,w]
                                            patch_only)
                elif self.task == 'OF':
                    indice = torch.randperm(present_frame1.shape[0])[:1]
                    frame_curr1, frame_curr2 = present_frame1[indice], present_frame2[indice]
                    # frame_curr1, frame_curr2 = present_frame1[0:1,:,:,:], present_frame2[0:1,:,:,:]
                    scores = self.Score.gredient_sample(
                                            patches_candi, #numpy[trail,3,h,w]
                                            patch_curr, #numpy[trail,3,h,w]
                                            frame_curr1, #tensor[1,3,h,w]
                                            frame_curr2, #tensor[1,3,h,w]
                                            patch_only)

                candi_y = scores.cpu().numpy()  # scores(-1-1) of all sample points
                # if self.task == 'OF':
                candi_y = np.sign(candi_y)
                # candi_y = 1.0/ (1+ np.exp(np.negative(candi_y)))
                mean_y = np.mean(candi_y)   # mean of scores
                avg_mean.append(mean_y)
                diff_y = mean_y - last_mean_y[j]
                # diff_y = np.exp(diff_y) if diff_y > 0 else np.log(diff_y + 3)
                if diff_y > 0:
                    diff_y = np.exp(diff_y)
                    if mean_y >= 1:
                        diff_y /= 5
                else:
                    diff_y = np.log(diff_y + 3)
                if mean_y == -1 or mean_y == 1:
                    delta = mean_y * np.mean(noise, axis=0)
                else:
                    delta = np.mean(noise * (candi_y - mean_y).reshape((trail, 1, 1, 1)), axis=0)
                # delta = diff_y * delta / norm(delta)
                delta = delta / norm(delta)
                last_mean_y[j] = mean_y
                deltas.append(delta)
            gradf = torch.from_numpy(np.mean(deltas, axis=0))
            avg_mean = np.mean(avg_mean)

            ## optimizer
            gradf_flat = gradf.flatten()
            if i == 1:
                grad_momentum = gradf
                full_matrix   = torch.outer(gradf_flat, gradf_flat)
            else:
                grad_momentum = beta1 * grad_momentum + (1 - beta1) * gradf
                full_matrix   = beta2 * full_matrix\
                                + (1 - beta2) * torch.outer(gradf_flat, gradf_flat)
            grad_momentum /= (1 - beta1 ** (i + 1))
            full_matrix   /= (1 - beta2 ** (i + 1))
            factor = 1 / torch.sqrt(eps + torch.diagonal(full_matrix))
            gradf = (factor * grad_momentum.flatten()).reshape_as(gradf)
            
            patch_curr = patch_curr + lr * gradf.numpy()
            patch_curr = np.clip(patch_curr, 0, 1)

            if not Config.hardbeat_oneway:
                # record neighboring triggers #
                for t in range(len(hist_patch)):
                    p1 = hist_patch[t].flatten()
                    p2 = patch_curr.flatten()
                    sim = ((p1.dot(p2) / (norm(p1) * norm(p2))) + 1) / 2
                    sim_graph[i % 100, t] = sim
                    sim_graph[t, i % 100] = sim

            # update history
            
            score_curr = self.Score.score(patch_curr, patch_only=patch_only)
            print('-' * 30)
            print(f"score_curr: {score_curr}")
            if not Config.hardbeat_oneway:
                hist_patch.append(patch_curr)
                hist_score = np.append(hist_score, 1 / score_curr)

                if len(hist_patch) == 100:
                    hist_patch.pop(0)
                    hist_score = np.delete(hist_score, 0)
                    sim_graph = np.delete(sim_graph, 0, axis=0)
                    sim_graph = np.delete(sim_graph, 0, axis=1)
                    new_row = np.zeros((1, sim_graph.shape[1]))
                    sim_graph = np.vstack((sim_graph, new_row))
                    new_colomn = np.zeros((sim_graph.shape[0], 1))
                    sim_graph = np.hstack((sim_graph, new_colomn))
                    sim_graph[-1][-1] = 1

            # store best
            if score_curr < score_best:
                score_best = score_curr
                patch_best[:] = patch_curr

            # log
            self.log(patch_best, score_best, None, None, i, 'hardbeat', log_gap=1 ,img_gap=1000, patch_only=patch_only)
            # evaluation
            self.evaluate(patch_best, i, 'hardbeat', log_gap=50, img_gap=1000, patch_only=patch_only)