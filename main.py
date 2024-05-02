import os
import sys

import options
import logging
from tensorboardX import SummaryWriter

import torch
from my_utils import set_random_seed, get_patch_area
from config import Config


def main():
    args = options.parse()
    set_random_seed(args.seed, deterministic=True)
    # prepare log
    log_dir =  os.path.join(Config.log_dir, args.test_name)
    logfile_path = os.path.join(log_dir, 'log.txt')

    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            # logging.FileHandler(logfile_path, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(str(args))
    tb_logger = SummaryWriter(log_dir)
    tb_logger.add_text("CLI_options", str(args), 0)

    model_name  = args.model_name
    if model_name in ['FlowNetC','PWC-Net','FlowNet2']:
        attack_task = 'OF' 
    elif model_name in ['depthhints','monodepth2','planedepth', 'SQLdepth', 'google_api']:
        attack_task = 'MDE'
    else:
        raise RuntimeError(f'The attack model {model_name} is not supported!')
    scene_size  = Config.model_scene_sizes_WH[model_name]
    if model_name == 'google_api':
        patch_area = get_patch_area(attack_task, scene_size, args.patch_ratio, height_pos=5/6)
    else:
        patch_area = get_patch_area(attack_task, scene_size, args.patch_ratio)
    logging.info(f"patch_area: {patch_area}")

    # Countermeasure
    if args.countermeasure == 'Blacklight':
        from countermeasure.Blacklight.probabilistic_fingerprint import InputTracker
        shape = (1, 3,) + Config.blacklight_shape
        tracker = InputTracker(shape=shape, window_size=Config.window_size, num_hashes_keep=Config.hash_kept, round=Config.roundto, step_size=Config.step_size, workers=Config.workers)
        logging.info("Blacklight detector created.")
    else:
        tracker = None

    # attack optical flow model
    if attack_task == 'OF':
        # import optical flow object
        from attack.flow_model import import_optical_flow_model
        model = import_optical_flow_model(Config.optical_flow_model_path[model_name], args).to(Config.device).eval()

    # attack depth estimation model
    elif attack_task == 'MDE':
        # import depth object
        from attack.depth_model import import_depth_model
        model = import_depth_model(scene_size, model_name).to(Config.device).eval()
    else:
        raise RuntimeError(f'The attack task {attack_task} is not supported!')

    # whitebox attack
    if args.attack_method == 'whitebox':
        from attack.whitebox_patch import Whitebox_patch
        whitebox_patch = Whitebox_patch(attack_task, model, model_name, patch_area, args.n_batch,
                                    args.batch_size, tb_logger, log_dir, tracker)
        whitebox_patch.whitebox_patch_attack(args.n_iter, args.alpha, args.num_pos, patch_only=args.patch_only)

    # blackbox attack
    else:
        if model_name == 'google_api':
            assert args.targeted_attack is True, 'targeted_attack must be true for api option.'
        from attack.blackbox_patch import Blackbox_patch
        blackbox_patch = Blackbox_patch(attack_task, model ,model_name, patch_area, args.n_batch,
                                    args.batch_size, tb_logger, log_dir, tracker, args.targeted_attack)
        with torch.no_grad():
            if args.attack_method == 'ours':
                blackbox_patch.BadPart(alpha=args.alpha, n_iters=args.n_iter, init_iters=args.init_iters, p_init=args.p_init, p_sche=args.p_sche,
                                        square_steps=args.square_steps, num_pos=args.num_pos, trail=args.trail, patch_only=args.patch_only)

            elif args.attack_method == 'S-RS':
                # blackbox_patch.square_attack(alpha=args.alpha, n_iters=args.n_iter, p_init=args.p_init, p_sche=args.p_sche, patch_only=args.patch_only)
                blackbox_patch.sparse_RS(n_iters=args.n_iter, p_init=args.p_init, p_sche=args.p_sche, patch_only=args.patch_only)


            elif args.attack_method == 'hardbeat':
                blackbox_patch.hardbeat_attack(total_steps=args.n_iter, K=args.topk, num_pos=args.num_pos, num_init=args.init_iters, trail=args.trail, patch_only=args.patch_only)
            
            elif args.attack_method == 'GA_attack':
                blackbox_patch.GA_attack(num_generations = args.n_iter, num_parents_mating = 5, sol_per_pop = 20, patch_only=args.patch_only)
            else:
                raise RuntimeError(f'The attack method {args.attack_method} is not supported!')

if __name__ == "__main__":
    main()
    