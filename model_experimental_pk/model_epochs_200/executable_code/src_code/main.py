import os
import traceback
import torch
import torch.nn as nn
from datetime import datetime

from utils import (
    Utils,
    save_history_to_csv,
    get_model,
    save_model_summary,
    load_config,
    save_model,
    load_model,
    countdown
)

import logging
from extraction_feature_test import extract_and_save_feature_maps

from setup_log import setup_logging_for_model
from model_train import train

from visualization import training_statistics, visualize_wrongly_class, model_predictions
from testing import test_model


def main():
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    config_path = r'C:\Users\jdah5454\Desktop\task_cifars10\model_experimental_pk\model_epochs_200\config\config.yaml'
    config = load_config(config_path)
    for model_cfg in config['models']:
        logger = setup_logging_for_model(model_cfg['name'], os.path.join(model_cfg['directories_dir'], 'logs'))
        try:
            start_time = datetime.now()
            utils = Utils()
            device = utils.device()

            train_loader, val_loader, test_loader, val_set, test_set = utils.get_data_loaders(
                batch_size=model_cfg['batch_size'],
                num_workers=0,
                input_size=tuple(model_cfg['input_size'])
            )

            backbone_path = model_cfg['backbone']
            class_name = backbone_path.split('.')[-1]
            module_path = '.'.join(backbone_path.split('.')[:-1])

            model_class = get_model(module_path, class_name)
            if model_class is None:
                logger.error(f"모델 클래스 로딩 실패: {model_cfg['name']}")
                continue

            net = model_class(in_channels=model_cfg['channels'], num_classes=10)

            if torch.cuda.device_count() > 1:
                logger.info(f"총 {torch.cuda.device_count()}개의 GPU가 감지되었습니다. DataParallel을 사용합니다.")
                net = nn.DataParallel(net)

            net.to(device)

            model_input_size = (model_cfg['channels'],) + tuple(model_cfg['input_size'])
            save_model_summary(
                net,
                input_size=model_input_size,
                save_dir=os.path.join(model_cfg['directories_dir'], 'model_summary'),
                filename="model_summary.txt"
            )

            load_pretrained = model_cfg.get('load_pretrained', False)
            if load_pretrained:
                logger.info("사전 학습된 모델을 로드합니다.")
                net = load_model(net, base_dir=model_cfg['directories_dir'], filename=model_cfg['model_path'], device=device)

                if net is None:
                    logger.error("Faield model Load.")
                    continue

                # 모델 평가 및 시각화
                plot_dir = os.path.join(model_cfg['directories_dir'], model_cfg.get('plot_dir', 'model_plot'))
                os.makedirs(plot_dir, exist_ok=True)

                pred_vec, images, labels = test_model(
                    net,
                    device,
                    test_loader,
                    test_set,
                    logger,
                    classes,
                    save_dir=plot_dir
                )
                if pred_vec is None:
                    logger.error("예측 벡터 실패.")
                    continue

                visualize_wrongly_class(
                    test_set,
                    pred_vec,
                    classes,
                    save_dir=plot_dir
                )
                model_predictions(
                    net,
                    test_loader,
                    classes,
                    device,
                    save_dir=plot_dir
                )

            else:
                # Model Learning start line
                net, history = train(
                    config,
                    model_cfg,
                    net,
                    train_loader,
                    val_loader,
                    device,
                    logger
                )
                if net is None or history is None:
                    logger.error("Error occurred during learning")
                    continue

                save_history_to_csv(
                    history,
                    base_dir=model_cfg['directories_dir'],
                    filename=model_cfg['history_file']
                )

                save_model(
                    net,
                    base_dir=model_cfg['directories_dir'],
                    filename=model_cfg['model_path']
                )

                plot_dir = os.path.join(model_cfg['directories_dir'], model_cfg.get('plot_dir', 'model_plot'))
                os.makedirs(plot_dir, exist_ok=True)

                training_statistics(
                    history,
                    save_dir=plot_dir
                )

                pred_vec, images, labels = test_model(
                    net,
                    device,
                    test_loader,
                    test_set,
                    logger,
                    classes,
                    save_dir=plot_dir
                )
                if pred_vec is None:
                    logger.error("예측 벡터 실패.")
                    continue

                visualize_wrongly_class(
                    test_set,
                    pred_vec,
                    classes,
                    save_dir=plot_dir
                )
                model_predictions(
                    net,
                    test_loader,
                    classes,
                    device,
                    save_dir=plot_dir
                )

                extract_and_save_feature_maps(
                    net,
                    model_cfg,
                    device,
                    train_loader,
                    val_loader,
                    logger
                )

                del net
                torch.cuda.empty_cache()
                countdown(180)

            end_time = datetime.now()
            elapsed_time = end_time - start_time
            logger.info(f'Total Training time: {elapsed_time}')

        except Exception as e:
            logger.error(f"error: {e}\n{traceback.format_exc()}")


if __name__ == "__main__":
    main()
