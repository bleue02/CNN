# testing.py

import os
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np
import traceback

def test_model(net, device, test_loader, test_set, logger, classes, save_dir=None):
    try:
        net.eval()
        pred_vec = []
        correct = 0
        all_images = []
        all_labels = []

        with torch.no_grad():
            for data in test_loader:
                batch, labels = data
                batch, labels = batch.to(device), labels.to(device)
                outputs = net(batch)
                _, predicted = torch.max(outputs, 1)

                correct += (predicted == labels).sum().item()
                pred_vec.append(predicted.cpu())
                all_images.append(batch.cpu())
                all_labels.append(labels.cpu())

        pred_vec = torch.cat(pred_vec)
        all_images = torch.cat(all_images)
        all_labels = torch.cat(all_labels)

        accuracy = 100 * correct / len(test_set)
        print(f'Accuracy on the {len(test_set)} test images: {accuracy:.2f} %')
        logger.info(f'Accuracy on the {len(test_set)} test images: {accuracy:.2f} %')

        if save_dir:
            correct_indices = (pred_vec == all_labels)
            incorrect_indices = ~correct_indices

            if correct_indices.sum() > 0:
                save_images(all_images[correct_indices], pred_vec[correct_indices], all_labels[correct_indices],
                            classes, save_dir, 'correct_images.png', "Correctly Classified Images")
            if incorrect_indices.sum() > 0:
                save_images(all_images[incorrect_indices], pred_vec[incorrect_indices], all_labels[incorrect_indices],
                            classes, save_dir, 'incorrect_images.png', "Incorrectly Classified Images")

        return pred_vec, all_images, all_labels

    except Exception as e:
        logger.error(f"Failed to test the model: {e}\n{traceback.format_exc()}")
        print(f"Failed to test the model: {e}\n{traceback.format_exc()}")
        return None, None, None

def save_images(images, preds, labels, classes, save_dir, filename, title):
    try:
        os.makedirs(save_dir, exist_ok=True)
        num_images = min(len(images), 16)

        plt.figure(figsize=(12, 12))
        img_grid = make_grid(images[:num_images], nrow=4, padding=2, normalize=False)
        np_img = img_grid.numpy().transpose((1, 2, 0))
        plt.imshow(np.clip(np_img, 0, 1))
        plt.title(title)
        plt.axis('off')

        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(np.clip(np_img, 0, 1))
        ax.axis('off')

        for i in range(num_images):
            row = i // 4
            col = i % 4
            pred_label = classes[preds[i].item()]
            true_label = classes[labels[i].item()]
            ax.text(col * (images.size(3) + 2) + images.size(3) // 2,
                    row * (images.size(2) + 2) + images.size(2) // 2,
                    f"P: {pred_label}\nT: {true_label}",
                    color='white', fontsize=12, ha='center', va='center',
                    bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.5'))

        # 이미지 저장
        plt.tight_layout()
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path)
        plt.close()
        print(f'Saved: {save_path}')
    except Exception as e:
        print(f"Failed to save images {filename}: {e}")
