import os
import torch.nn.functional as F
import traceback

from extraction_feature_test import extract_and_save_feature_maps
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from tqdm import tqdm
import gc
from utils import Utils, save_history_to_csv, save_model_summary
from torchvision.utils import save_image
from setup_log import *

def train(config, model_cfg, net, train_loader, val_loader, device, logger):
    try:
        start_time = datetime.now()

        history = []
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=model_cfg['learning_rate'])

        # num_epochs = model_cfg['num_epochs']
        num_epcohs = model_cfg.get('num_epochs')
        # logger.info(f'num_epochs from config: {num_epcohs}')
        for epoch in range(num_epcohs):
            net.train()
            train_loss = 0.0
            correct = 0
            total = 0

            for images, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1} [Training]', leave=True):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = net(images)  # forward
                loss = criterion(outputs, labels)

                loss.backward()  # backpropagation
                optimizer.step()

                train_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_loss = train_loss / len(train_loader.dataset)
            epoch_acc = 100.0 * correct / total

            net.eval()
            with torch.no_grad():
                val_loss = 0.0; val_correct = 0; val_total = 0
                for images, labels in tqdm(val_loader, desc=f'Epoch {epoch + 1}/{model_cfg["num_epochs"]} [Validation]', leave=True):
                    images, labels = images.to(device), labels.to(device)

                    outputs = net(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

                val_epoch_loss = val_loss / len(val_loader.dataset)
                val_epoch_acc = 100.0 * val_correct / val_total

            history.append({
                'train_loss': epoch_loss,
                'val_loss': val_epoch_loss,
                'val_acc': val_epoch_acc,
            })

            logger.info(f'Epoch[{epoch + 1} / {num_epcohs}], Train loss: {epoch_loss:.4f}, '
                        f'Train accuracy: {epoch_acc:.2f} %, Validation loss: {val_epoch_loss:.4f}, Validation accuracy: {val_epoch_acc:.2f}%')

            # Extract and save feature maps
            extract_and_save_feature_maps(net, model_cfg, device, train_loader, val_loader, logger)
            print('\n')

        end_time = datetime.now()
        elapsed_time = end_time - start_time
        logger.info(f'Training time: {elapsed_time}')
        return net, history

    except Exception as e:
        logger.exception("An error occurred during training")
        return None, None






