# visualization.py

import os
import matplotlib.pyplot as plt
import logging
import torch
import traceback
import numpy as np
from torchvision.utils import make_grid

def training_statistics(history, save_dir):
    try:
        plt.figure(figsize=(10, 5))
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot([h['train_loss'] for h in history], label='Train Loss')
        plt.plot([h['val_loss'] for h in history], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss over Epochs')
        plt.axis('off')
        plt.legend()

        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot([h['val_acc'] for h in history], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy over Epochs')
        plt.legend()

        # Save the figure
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'training_statistics.png'))
        plt.close()
        logging.getLogger().info(f'Training statistics plots saved to {save_dir}')
    except Exception as e:
        logging.getLogger().error(f"Failed to save training statistics plots: {e}\n{traceback.format_exc()}")


def get_targets(dataset):
    """
    원본 데이터셋의 targets를 재귀적으로 가져오는 함수.
    항상 numpy 배열을 반환하여 리스트 인덱싱 오류를 방지합니다.
    """
    if isinstance(dataset, torch.utils.data.Subset):
        return get_targets(dataset.dataset)[dataset.indices]
    else:
        return np.array(dataset.targets)


def visualize_wrongly_class(test_set, pred_vec, classes, save_dir):
    try:
        # 원본 데이터셋의 true_labels를 가져옵니다.
        true_labels = get_targets(test_set)

        # Ensure pred_vec is a 1D array
        pred_labels = pred_vec.cpu().numpy().flatten()

        # Find indices of wrongly classified images
        wrong_indices = np.where(pred_labels != true_labels)[0]
        if len(wrong_indices) == 0:
            print("No wrongly classified images.")
            return

        # Select a subset of wrongly classified images (up to 16)
        selected_indices = wrong_indices[:16]
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        for idx, ax in zip(selected_indices, axes.flatten()):
            image, label = test_set[idx]

            # Transpose the image to put the channels last (for visualization purposes)
            image = image.numpy().transpose((1, 2, 0))
            # Denormalize the image
            image = (image * 0.5) + 0.5  # Assuming normalization was done with mean=0.5, std=0.5
            image = np.clip(image, 0, 1)  # Ensure the pixel values are in [0, 1]

            ax.imshow(image)
            ax.set_title(f"True: {classes[true_labels[idx]]}\nPred: {classes[pred_labels[idx]]}", fontsize=10)
            ax.axis('off')

        plt.tight_layout()
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'wrongly_classified.png'))
        plt.close()
        logging.getLogger().info(f'Wrongly classified images saved to {save_dir}')
    except Exception as e:
        logging.getLogger().error(f"Failed to save wrongly classified images: {e}\n{traceback.format_exc()}")


def model_predictions(net, test_loader, classes, device, save_dir):
    try:
        # Ensure the model is in evaluation mode
        net.eval()

        all_preds = []
        all_labels = []
        all_images = []

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs, 1)
                all_preds.append(predicted.cpu())
                all_labels.append(labels.cpu())
                all_images.append(images.cpu())

        # Concatenate all batches
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        all_images = torch.cat(all_images)

        # Find wrongly classified indices
        wrong_indices = (all_preds != all_labels).nonzero(as_tuple=False).squeeze()

        if wrong_indices.numel() == 0:
            print("No wrongly classified images.")
            return

        # Select up to 16 wrongly classified images
        selected_indices = wrong_indices[:16]
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        for idx, ax in zip(selected_indices, axes.flatten()):
            image = all_images[idx].numpy().transpose((1, 2, 0))  # Convert to HWC format
            image = (image * 0.5) + 0.5  # Denormalize
            image = np.clip(image, 0, 1)  # Clip pixel values to be between 0 and 1

            ax.imshow(image)
            true_label = classes[all_labels[idx].item()]
            pred_label = classes[all_preds[idx].item()]
            ax.set_title(f"True: {true_label}\nPred: {pred_label}", fontsize=10, color='black')
            ax.axis('off')

        plt.tight_layout()
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'model_predictions_modified.png'))
        plt.close()
        logging.getLogger().info(f'Model predictions image saved to {save_dir}')
    except Exception as e:
        logging.getLogger().error(f"Failed to save model predictions image: {e}\n{traceback.format_exc()}")



# 'get_predictions_with_images' 함수는 'testing.py'로 이동되었으므로 제거했습니다.
