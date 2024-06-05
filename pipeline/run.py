import sys
import click
import torch
from copy import deepcopy
from datetime import datetime

from torchsummary import summary

import lightning as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from pathlib import Path
import importlib
import os

import numpy as np
import pandas as pd

from loss.CompositionalLoss import parents
from metrics.Maokai import bone_lengths_np, distance_ordinal_histogram
from models.stride import calc_standing_foot_positions
import yaml
import subprocess
# import papermill as pm

from codecarbon import track_emissions

import logging
logger = logging.getLogger(__name__)
format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=format_str)


def resolve_repo_path():
    return str(Path(__file__).resolve().parent.parent) + '/'

def path_resolution():
    # path resolution
    paths = ["/home/yale1/ma-jonah-data/", "/share/data/yhartmann/data/ma-jonah/", '../data/']
    paths = [str(Path(p).resolve()) + '/' for p in paths]
    for base_path in paths:
        if os.path.exists(base_path):
            if not base_path == paths[-1]:
                print(' -- rsync --')
                call = ["rsync", "-av", paths[-1], base_path]
                print('Calling:', ' '.join(call))
                subprocess.call(call)
                print(' -- rsync finished --\n')

            print(f"Base path: {base_path}")
            base_path = base_path + '/'
            break
    return base_path

def extract_pred(r):
    prediction, target = list(zip(*r))

    # cast to numpy and flatten
    prediction = np.concatenate([itm.numpy() for itm in prediction])
    target = np.concatenate([itm.numpy() for itm in target])
    
    return prediction, target

def add_stride_length_metrics(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ['step amount', 
               'stride length mean', 
               'stride length median', 
               'stride length std', 
               'travelled distance'
              ]
    
    df[metrics[0]] = df['stride_len'].apply(len)
    df[metrics[1]] = df['stride_len'].apply(np.mean)
    df[metrics[2]] = df['stride_len'].apply(np.median)
    df[metrics[3]] = df['stride_len'].apply(np.std)
    df[metrics[4]] = df['stride_len'].apply(sum)
    df.loc['__all__'] = df[metrics].mean()
    return df, metrics

def calc_step_hists(prd_foots, trg_foots, bins = 80, hist_range = (0, 400)):
    gt_counts, gt_bins = np.histogram(trg_foots['stride_len']['LFoot'], bins=bins, range=hist_range)
    pred_counts, pred_bins = np.histogram(prd_foots['stride_len']['LFoot'], bins=bins, range=hist_range)

    return gt_counts, gt_bins, pred_counts, pred_bins

def calc_metrics(prediction, target, data_module, distance=15):
    metrics = {}

    # calc bone stability
    prd_bone_lengths = bone_lengths_np(prediction, False)
    metrics["prd_bone_stability"] = float(np.mean(np.std(prd_bone_lengths, axis=0)))
    metrics["prd_bone_stability_per_joint"] = {key: float(val) for key, val in (zip(parents.keys(), np.std(prd_bone_lengths, axis=0)))}

    # create dfs for easier use
    prd_df = pd.DataFrame(prediction)
    prd_df.columns = data_module.columns
    trg_df = pd.DataFrame(target)
    trg_df.columns = data_module.columns

    # calc standing foot positions
    prd_foots, prd_valley = calc_standing_foot_positions(prd_df, distance=distance)
    trg_foots, trg_valley = calc_standing_foot_positions(trg_df, distance=distance)

    trg_foots, _ = add_stride_length_metrics(trg_foots)
    prd_foots, _ = add_stride_length_metrics(prd_foots)

    logger.info(f'Distance: {distance}')
    step_percent = 100 * prd_foots.at['__all__', 'step amount'] / trg_foots.at['__all__', 'step amount']
    logger.info(f"SP: {np.round(step_percent, 2)}")

    mean_stride_length_difference = prd_foots.at['__all__', 'stride length mean'] - trg_foots.at['__all__', 'stride length mean']
    logger.info(f"MSLD: {np.round(mean_stride_length_difference, 2)}")

    distance_ratio = 100 * prd_foots.at['__all__', 'travelled distance'] / trg_foots.at['__all__', 'travelled distance']
    logger.info(f"DR: {np.round(distance_ratio, 2)}")

    # calc step histograms
    gt_counts, _, pred_counts, _ = calc_step_hists(trg_foots, prd_foots, bins=80)
    step_distribution_distance = distance_ordinal_histogram(pred_counts, gt_counts, bins=80)
    logger.info(f"SDD: {np.round(step_distribution_distance, 2)}")

    return (dict(**metrics, sp=float(step_percent), msld=float(mean_stride_length_difference), dr=float(distance_ratio), sdd=float(step_distribution_distance)), 
        dict(prd_foots=prd_foots, prd_valley=prd_valley, trg_foots=trg_foots, trg_valley=trg_valley))

@track_emissions(offline=True, country_iso_code="DEU", output_dir=resolve_repo_path(), log_level='warning') 
def train(model, trainer, data_module, model_checkpoint_path):
    try:
        logger.info('==== TRAIN ================================================')
            # construct trainer
        
        summary(model, data_module.data_shape)

        trainer.fit(model, data_module)
        # todo: reconsider using another strategy up top to then use the best version of that somehow and figure out how to get that from logs?
        # bc now during overfitting we have an issue
        trainer.save_checkpoint(model_checkpoint_path)
        
    except Exception as e:
        logger.exception(e)

@track_emissions(offline=True, country_iso_code="DEU", output_dir=resolve_repo_path(), log_level='warning') 
def evaluate(model, trainer, data_module, checkpoint_path, test=True, predict=True):
    res_dict = {}
    ft_dict = {}
    prd_dict = {}
    try:
        if test:
            logger.info('==== TEST ================================================')
            res_dict["skeleton"] = trainer.test(model, data_module, ckpt_path=checkpoint_path)

        if predict:
            logger.info('==== Predict ================================================')
            res = trainer.predict(model, data_module, ckpt_path=checkpoint_path)
            if data_module.num_data_loader('predict') < 2:
                res = [res]
            # re-zip
            for i, p in enumerate(res):
                prediction, target = extract_pred(p)
                prd_dict[f"gait_{i}"] = ((prediction, target))
                # for compatibility with old code
                # res_dict[f"gait_{i}"], ft_dict[f"gait_{i}"] = calc_metrics(prediction, target, data_module, distance=20)

                for distance in [15, 20, 25, 30, 35]:
                    res_dict[f"gait_{i}_d{distance}"], ft_dict[f"gait_{i}_d{distance}"] = calc_metrics(prediction, target, data_module, distance=distance)
                
                # pm.execute_notebook(
                #     'eval.ipynb',
                #     f'logs{interm_path}/{lib}.{params}.eval_1.ipynb',
                #     parameters = dict(**notebook_params, data_loader_to_eval=1),
                #     **pm_params
                # )
    except Exception as e:
        logger.exception(e)
    
    return res_dict, ft_dict, prd_dict


@click.command()
@click.option("--lib", default="OwnBaselineCNN", help="Model and file name")
@click.option("--data", default="RSO_LModule", help="Data module name")
@click.option("--params", default="hp_default", help="Hyperparams dict name in module")
@click.option("--mode", default="eval", help="Select from [train, eval]")
@click.option("--num_worker", default=40, help="Number of Workers to use")
@click.option("--debug", is_flag=True, show_default=True, default=False, help="If should run in debug mode")
@click.option("--computease", is_flag=True, show_default=True, default=False, help="Wether we're on computease or a rk")
@click.option("--dry-run", is_flag=True, show_default=True, default=False, help="If should just do a dry run")
def main(lib, data, params, mode, debug, computease, dry_run, num_worker):
    if mode not in ['train', 'eval']:
        raise ValueError(f"train must be one of ['train', 'eval'] but got {mode}")
    
    interm_path = '_debug' if debug else ''
    # create file handler which logs even debug messages
    # if log file already exists delete
    log_file = f"logs{interm_path}/{lib}.{params}.{mode}.log"
    if os.path.exists(log_file):
        os.remove(log_file)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(format_str))
    logging.getLogger().addHandler(fh)

    logger.info(f"Run at: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
    logger.info('==== SETUP ================================================')

    # path resolution
    base_path = path_resolution()
    
    # resolve checkpoints
    checkpoint_path = Path(f"./checkpoints{interm_path}/").resolve()
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    model_checkpoint_path = f"{str(checkpoint_path / lib)}.{params}.ckpt"
    logger.info(f"model_checkpoint_path: {model_checkpoint_path}")

    # import nn module
    NNModule = importlib.import_module(f"models.{lib}")

    # resolve hyperparameters
    hyper_params = getattr(NNModule, params)
    if debug:
        hyper_params['trainer_params']['max_epochs'] = 7
        # hyper_params['trainer_params']['profiler'] = 'simple'
    
    if mode != 'train':
        hyper_params['trainer_params']['devices'] = 1

    # use tensor cores
    if computease:
        torch.set_float32_matmul_precision('medium')
        logger.info('Using Tensor Cores')

    LOG = deepcopy(hyper_params)
    LOG['date'] = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    LOG['optimizer'] = None
    LOG['loss_function'] = None
    LOG['RUN'] = dict(
        mode=mode,
        lib=lib, 
        params=params, 
        base_path=base_path,
        debug=debug,
        computease=computease,
        dry_run=dry_run,
        num_worker=num_worker,
        model_checkpoint_path=model_checkpoint_path,
    )

    logger.info('--- Configuraion --------------------')
    for key, val in LOG.items():
        logger.info(f"{key}: {val}")
    logger.info('--------------------------------------')

    # construct data module
    DLModule = getattr(importlib.import_module(f"datasets.{data}"), data)
    data_module = DLModule(data_dir=base_path, 
        n_jobs=num_worker, 
        debug=debug, 
        **hyper_params['data_params'])

    NNModule = importlib.import_module(f"models.{lib}")
    NeuralNetwork = getattr(NNModule, "NeuralNetwork")

    model = NeuralNetwork(model_params=hyper_params['model_params'],
                        optimizer=hyper_params['optimizer'],
                        loss_function=hyper_params['loss_function'],
                        optimizer_params=hyper_params['optimizer_params'],
                        scheduler_params=hyper_params['scheduler_params'])

    callbacks = []
    if 'early_stopping_params' in hyper_params:
        callbacks.append(EarlyStopping(**hyper_params['early_stopping_params']))

    trainer = pl.Trainer(logger=TensorBoardLogger(checkpoint_path, name=lib, version=params), 
                    fast_dev_run=dry_run,
                    default_root_dir=checkpoint_path, 
                    callbacks=callbacks,
                    **hyper_params['trainer_params'])
    
    if mode == 'train':
        train(model, trainer, data_module, model_checkpoint_path)
    elif mode == 'eval':
        LOG['RESULTS'], _, _ = evaluate(model, trainer, data_module, model_checkpoint_path)
        logger.info(LOG['RESULTS'])

    LOG['date_end'] = datetime.now().strftime("%d.%m.%Y %H:%M:%S")

    # save results
    with open(f'logs{interm_path}/{lib}.{params}.{mode}.yml', 'w') as outfile:
        yaml.dump(LOG, outfile, sort_keys=False)

    logger.info('==== DONE ================================================')


if __name__ == "__main__":
    # from experiment_impact_tracker.compute_tracker import ImpactTracker
    # tracker = ImpactTracker('green_logs/')
    # tracker.launch_impact_monitor()

    main()