import argparse
import json
import os
import random
import time
import warnings

import numpy as np
import torch

import models
from configs import get_model_config


def seed_everything(seed, strengthen=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if strengthen:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
        os.environ['PYTHONHASHSEED'] = str(seed)


def resolve_model(model_name: str, dataset_name: str):
    if dataset_name.endswith('_hete') and not model_name.endswith('_hete'):
        if hasattr(models, model_name + '_hete'):
            return getattr(models, model_name + '_hete')
    return getattr(models, model_name)


if __name__ == '__main__':

    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--run', type=int, default=10)

    parser.add_argument('--dataset', type=str, default='yelp')
    parser.add_argument('--ratio', type=float, nargs='+', default=[0.4, 0.3, 0.3])
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--model', type=str, default='FDSGAD')
    parser.add_argument('--model_config', type=json.loads, default='{}')
    parser.add_argument('--save', type=str, default='', help='save weights path')
    args = parser.parse_args()

    args.model_config = get_model_config(args.model, args.dataset, args.model_config)
    model_cls = resolve_model(args.model, args.dataset)

    seed_everything(args.seed)

    print(f'dataset: {args.dataset}')
    print(f'model: {args.model}')
    print(f'model config: {args.model_config}')
    print(f'device: {args.device}')
    print()

    metrics = ['auc', 'recall', 'precision', 'f1_macro']
    performance = {m: [] for m in metrics}

    start = time.time()
    for t in range(args.run):
        if args.device == 'cuda':
            torch.cuda.empty_cache()

        print(f'trial: {t+1}/{args.run}')

        m, p = model_cls.trainfit(args)
        if args.save:
            torch.save(m.state_dict(), args.save)
        for metric in metrics:
            performance[metric].append(p[metric])

        print()
    end = time.time()
    dt = int(end - start)

    print('=' * 70)
    print(f'Dataset: {args.dataset}')
    print(f'Model: {args.model}')
    print(f'Config: {args.model_config}')
    print(f'Runs: {args.run}, Time: {dt//60}m{dt%60}s')
    print('=' * 70)

    print(f'{"Trial":<6}', end='')
    for m in metrics:
        print(f'{m:<12}', end='')
    print()
    print('-' * 54)

    for t in range(args.run):
        print(f'{t+1:<6}', end='')
        for m in metrics:
            print(f'{performance[m][t]:.5f}     ', end='')
        print()

    print('-' * 54)

    print(f'{"Mean":<6}', end='')
    for m in metrics:
        print(f'{np.mean(performance[m]):.5f}     ', end='')
    print()
    print(f'{"Std":<6}', end='')
    for m in metrics:
        print(f'{np.std(performance[m]):.5f}     ', end='')
    print()
    print()

    print('Paper format (mean+-std, x100):')
    for m in metrics:
        mean = np.mean(performance[m]) * 100
        std = np.std(performance[m]) * 100
        print(f'  {m}: {mean:.2f}+-{std:.2f}')
