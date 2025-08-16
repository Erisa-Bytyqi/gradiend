import json
import os
import shutil
import time
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import logging
from pathlib import Path

from gradiend.evaluation.analyze_encoder import analyze_models
from gradiend.evaluation.encoder.de_encoder_analysis import DeEncoderAnalysis
from gradiend.training.trainer import train_all_layers_gradiend, train_multiple_layers_gradiend

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='conf', config_name='config')
def train(cfg: DictConfig): 
    
    log.info(f"Running GRADIEND with config: {OmegaConf.to_yaml(cfg)}")

    if cfg.eval_only and cfg.model_path: 
        model_analyser = DeEncoderAnalysis(cfg.pairing)
        for split in ["val", "test"]:
            analyze_models(cfg.model_path, config=cfg.pairing, split=split, multi_task=False)
            model_analyser.get_model_metrics_m_dim(cfg.model_path, split=split)

    else:
        if cfg.mode['name'] == 'gradiend':
            train_gradined(cfg)
        if cfg.mode['name'] == 'gradiend_ensemble':
            # this is the only one that need to construct objects first and then it can call 
            # the train function directly....
            pass


def train_gradined(cfg: DictConfig):
    base_path = Path.cwd()
    version = cfg.get('version', None)
    experiments_path = base_path / "output" / "experiments" / cfg.mode['name']
    metric = cfg.mode['metric']

    metrics = []
    total_start = time.time()
    times = []

    model_analyser = DeEncoderAnalysis(cfg.pairing)

    if version is None or version == '':
        version = ''
    else:
        version = f'/v{version}'

    for i in range(cfg.num_runs): 
        log.info(f"Run {i} for GRADIEND")
        start = time.time()
        output = experiments_path / cfg.pairing.plot_name / f"dim_{cfg.mode.model_config.num_dims}_{cfg.mode.model_config.source}" / cfg.base_model / f"{i}"
        metrics_file = f'{output}/metrics.json'
        base_model = cfg.base_model
        if os.path.exists(metrics_file):
            metrics.append(json.load(open(metrics_file)))
            print(f'Skipping training of {output} as it already exists')
            continue

        if not os.path.exists(output):
            print('Training', output)
            run_config = cfg.mode.model_config.copy()
            run_config.seed = i
            
            if 'layers' in run_config:
                train_multiple_layers_gradiend(model=base_model, output=output, **run_config)
            else:
                train_all_layers_gradiend(config=cfg.pairing, model=base_model, output=output, **run_config)
        else:
            print('Model', output, 'already exists, skipping training, but evaluate')

        log.info(f"Analyzing models in {output} for validation set")
        analyze_models(str(output), config=cfg.pairing, split='val', multi_task=False)
        log.info(f"Evaluating models in {output} for test set")
        analyze_models(str(output), config=cfg.pairing, split='test', multi_task=False)

        if cfg.mode.model_config.num_dims > 1: 
            model_analyser.get_model_metrics_m_dim(output, split='val')
            model_analyser.get_model_metrics_m_dim(output, split='test')
        else:
            model_metrics = model_analyser.get_model_metrics(output, split='val')
            model_analyser.get_model_metrics(output, split='test')
            metric_value = model_metrics[metric]
            json.dump(metric_value, open(metrics_file, 'w'))
            metrics.append(metric_value)

            # if clear_cache:
            #     cache_folder = f'data/cache/gradients/{det_combination}/{base_model}'
            #     if os.path.exists(cache_folder):
            #         shutil.rmtree(cache_folder)

            print(f'Metrics for model {base_model}: {metrics}')
            best_index = np.argmax(metrics)
            print('Best metric at index', best_index, 'with value', metrics[best_index])
        times.append(time.time() - start)

        total_time = time.time() - total_start
        
        if times:
            print(f'Trained {len(times)} models in {total_time}s')
            print(f'Average time per model: {np.mean(times)}')
        else:
            print('All models were already trained before!')

    return output



if __name__ == '__main__':
    train()