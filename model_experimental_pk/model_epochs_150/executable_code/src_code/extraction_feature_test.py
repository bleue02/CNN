import os
import torch
import torch.nn as nn
from torchvision.utils import make_grid, save_image
import logging
import matplotlib.pyplot as plt
from utils import Utils
import numpy as np
import traceback

def configure_env_path():
    base_path = ''
    if os.name == 'posix':
        base_path = r'/mnt/c/Users/jdah5454/PycharmProjects/task_cifar10/pth'
        print(f'Currently running env is posix (e.g., macOS/Linux)')
    elif os.name == 'nt':
        # base_path = r'C:\Users\jdah5454\PycharmProjects\task_cifar10\executable_code\pth'
        base_path = r'C:\src_runs\executable_code\pth'
        print(f'Currently running env is Windows')
    else:
        print('Automatic configuration is not possible, please set the path manually!')
        return None
    return base_path

def save_feature_map(feature_map, layer_dir, grid_size=(4, 4), logger=None):
    try:
        # 배치 내 첫 번째 이미지의 특징 맵만 저장
        if feature_map.dim() == 4:
            # [batch_size, C, H, W] -> [C, H, W]
            fmap = feature_map[0].cpu().detach()
            # if logger:
            #     logger.debug(f'Feature map shape after selecting first image: {fmap.shape}')
        elif feature_map.dim() == 3:
            fmap = feature_map.cpu().detach()
            # if logger:
            #     logger.debug(f'Feature map shape: {fmap.shape}')
        else:
            # if logger:
            #     logger.error(f'Invalid feature map shape: {feature_map.shape}. Expected 3 or 4 dimensions (C, H, W) or (N, C, H, W).')
            pass
            return

        # 데이터 타입 변환 (float으로 변환)
        if fmap.dtype == torch.uint8:
            fmap = fmap.float() / 255.0  # uint8을 float으로 변환 (0-1 범위)
            # if logger:
            #     logger.debug(f'Converted feature map dtype from uint8 to float.')

        # 텐서 형상 확인
        if len(fmap.shape) != 3:
            # if logger:
            #     logger.error(f'Invalid feature map shape after processing: {fmap.shape}. Expected 3 dimensions (C, H, W).')
            return

        # 그리드 이미지 생성 (nrow을 grid_size의 두 번째 값으로 설정)
        nrow = grid_size[1]
        grid = make_grid(fmap.unsqueeze(1), nrow=nrow, padding=2, normalize=True)

        # 레이어별 폴더 생성
        os.makedirs(layer_dir, exist_ok=True)

        # 그리드 이미지 저장
        save_path = os.path.join(layer_dir, 'feature_map_grid.png')
        save_image(grid, save_path)
        # if logger:
        #     logger.info(f'Feature map saved to {save_path}')

    except Exception as e:
        # if logger:
        #     logger.error(f'Failed to save feature map for layer {layer_dir}: {e}')
        pass
        # else:
        #     print(f'Failed to save feature map for layer {layer_dir}: {e}')

def extract_and_save_feature_maps(net, model_cfg, device, train_loader, val_loader, logger):
    try:
        # 실제 레이어 이름을 config.yaml에서 읽어옵니다
        target_layers = model_cfg.get('target_layers', [])
        # if not target_layers:
        #     logger.error("No 'target_layers' specified in model configuration.")
        pass
        return

        feature_maps = {layer: [] for layer in target_layers}
        hooks = []

        def hook_fn(layer_name):
            def hook(module, input, output):
                feature_maps[layer_name].append(output.detach().cpu())
            return hook

        # 모델의 레이어를 순회하며 후킹
        for name, module in net.named_modules():
            if name in target_layers:
                hooks.append(module.register_forward_hook(hook_fn(name)))
                # logger.info(f"레이어 '{name}'에 후크를 등록했습니다.")

        if not hooks:
            # logger.error("No target layers were found and hooked.")
            return

        # 학습 데이터에서 하나의 배치를 가져와서 포워드 패스 실행
        data_iter = iter(train_loader)
        images, labels = next(data_iter)
        images = images.to(device)

        net(images)

        # 후크 해제
        for hook in hooks:
            hook.remove()

        # 각 레이어의 특징 맵을 저장
        for layer_name, fmap_list in feature_maps.items():
            if not fmap_list:
                # logger.error(f"No feature maps captured from layer '{layer_name}'.")
                continue

            fm = fmap_list[0][0]
            layer_dir = os.path.join(model_cfg['directories_dir'], 'feature_maps', layer_name)
            save_feature_map(fm, layer_dir, grid_size=(4, 4), logger=logger)

    except Exception as e:
        # logger.error(f"Failed to visualize combined feature maps: {e}\n{traceback.format_exc()}")
        # print(f"Failed to visualize combined feature maps: {e}\n{traceback.format_exc()}")
        pass