import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import traceback
from torchvision.utils import make_grid

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

        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        for i, ax in enumerate(axes.flatten()[:num_images]):
            # Extract image and normalize for better visualization
            image = images[i].numpy().transpose((1, 2, 0))
            image = (image - image.min()) / (image.max() - image.min())

            ax.imshow(np.clip(image, 0, 1))
            pred_label = classes[preds[i].item()]
            true_label = classes[labels[i].item()]
            ax.set_title(f"True: {true_label}\nPred: {pred_label}", fontsize=10, color='white', bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.5'))
            ax.axis('off')

        plt.tight_layout()
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path)
        plt.close()
        # print(f'Saved: {save_path}')
    except Exception as e:
        print(f"Failed to save images {filename}: {e}")
