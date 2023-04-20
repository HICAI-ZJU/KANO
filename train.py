import warnings
warnings.filterwarnings('ignore')
from rdkit import RDLogger  
RDLogger.DisableLog('rdApp.*')  
from argparse import Namespace
from logging import Logger
import os
from typing import Tuple
import numpy as np

from chemprop.train.run_training import run_training
from chemprop.data.utils import get_task_names
from chemprop.utils import makedirs
from chemprop.parsing import parse_train_args, modify_train_args
from chemprop.torchlight import initialize_exp


def run_stat(args: Namespace, logger: Logger = None) -> Tuple[float, float]:
    """k-time independent runs"""
    info = logger.info if logger is not None else print

    # Initialize relevant variables
    init_seed = args.seed
    save_dir = args.save_dir
    task_names = get_task_names(args.data_path)

    # Run training on different random seeds for each run
    all_scores = []
    for run_num in range(args.num_runs):
        info(f'Run {run_num}')
        args.seed = init_seed + run_num
        args.save_dir = os.path.join(save_dir, f'run_{run_num}')
        makedirs(args.save_dir)
        model_scores = run_training(args, False, logger)
        all_scores.append(model_scores)
    all_scores = np.array(all_scores)

    # Report results
    info(f'{args.num_runs}-time runs')

    # Report scores for each run
    for run_num, scores in enumerate(all_scores):
        info(f'Seed {init_seed + run_num} ==> test {args.metric} = {np.nanmean(scores):.6f}')

        if args.show_individual_scores:
            for task_name, score in zip(task_names, scores):
                info(f'Seed {init_seed + run_num} ==> test {task_name} {args.metric} = {score:.6f}')

    # Report scores across models
    avg_scores = np.nanmean(all_scores, axis=1)  # average score for each model across tasks
    mean_score, std_score = np.nanmean(avg_scores), np.nanstd(avg_scores)
    info(f'Overall test {args.metric} = {mean_score:.6f} +/- {std_score:.6f}')

    if args.show_individual_scores:
        for task_num, task_name in enumerate(task_names):
            info(f'Overall test {task_name} {args.metric} = '
                    f'{np.nanmean(all_scores[:, task_num]):.6f} +/- {np.nanstd(all_scores[:, task_num]):.6f}')

    return mean_score, std_score


if __name__ == '__main__':
    args = parse_train_args()
    modify_train_args(args)
    logger, args.save_dir = initialize_exp(Namespace(**args.__dict__))
    mean_auc_score, std_auc_score = run_stat(args, logger)
    print(f'Results: {mean_auc_score:.5f} +/- {std_auc_score:.5f}')