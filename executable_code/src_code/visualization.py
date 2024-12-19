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

def visualize_wrongly_class(test_set, pred_vec, classes, save_dir):
    try:
        wrong_indices = np.where(pred_vec.cpu().numpy() != np.array(test_set.targets))[0]
        if len(wrong_indices) == 0:
            print("No wrongly classified images.")
            return

        # Select a subset of wrongly classified images
        selected_indices = wrong_indices[:16]
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        for idx, ax in zip(selected_indices, axes.flatten()):
            image, label = test_set[idx]
            ax.imshow(np.transpose(image.numpy(), (1, 2, 0)))
            ax.set_title(f"True: {classes[label]}\nPred: {classes[pred_vec[idx]]}")
            ax.axis('off')

        plt.tight_layout()
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'wrongly_classified.png'))
        plt.close()
        # logging.getLogger().info(f'Wrongly classified images saved to {save_dir}')
    except Exception as e:
        logging.getLogger().error(f"Failed to save wrongly classified images: {e}\n{traceback.format_exc()}")

def model_predictions(net, test_loader, classes, device, save_dir):
    try:
        dataiter = iter(test_loader)
        images, labels = next(dataiter)
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)

        # Move images to CPU for visualization
        images = images.cpu()

        # Create a grid of images
        img_grid = make_grid(images[:16], nrow=4, padding=2, normalize=False)
        np_img = img_grid.numpy().transpose((1, 2, 0))

        # Clip to valid range [0, 1] for float values
        np_img = np.clip(np_img, 0, 1)

        # Plot the grid
        plt.figure(figsize=(8, 8))
        plt.imshow(np_img)
        plt.axis('off')
        plt.title('Model Predictions')
        plt.tight_layout()

        # Annotate predictions
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(np_img)
        ax.axis('off')
        for i in range(16):
            row = i // 4
            col = i % 4
            pred_label = classes[predicted[i].item()]
            ax.text(col * (images.size(3) + 2) + images.size(3) // 2,
                    row * (images.size(2) + 2) + images.size(2) // 2,
                    pred_label,
                    color='white',
                    fontsize=12,
                    ha='center',
                    va='center',
                    bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.5'))

        # Save the figure
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'model_predictions.png'))
        plt.close()
        # logging.getLogger().info(f'Model predictions image saved to {save_dir}')
    except Exception as e:
        logging.getLogger().error(f"Failed to save model predictions image: {e}\n{traceback.format_exc()}")

# 'get_predictions_with_images' 함수는 'testing.py'로 이동되었으므로 제거했습니다.
