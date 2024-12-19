from torch.utils.data import DataLoader, random_split
import torchvision
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.nn.functional as F

from matplotlib import pyplot as plt
from torchsummary import summary
import time
import sys
import PIL
import csv
import os
import random
import importlib
import logging
import yaml

class SingletonMeta(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

class ResizeFunction:
    def __init__(self):
        self.resize_transform = {
            '8x8': transforms.Resize((8, 8)),
            '16x16': transforms.Resize((16, 16)),
            '32x32': transforms.Resize((32, 32)),
            '64x64': transforms.Resize((64, 64)),
            '128x128': transforms.Resize((128, 128)),
            '256x256': transforms.Resize((256, 256)),
            '512x512': transforms.Resize((512, 512)),
            '1024x1024': transforms.Resize((1024, 1024)),
            '2048x2048': transforms.Resize((2048, 2048)),
            '4096x4096': transforms.Resize((4096, 4096))
        }

    def get_resize_transform(self, size):
        return self.resize_transform.get(size)

class Utils(metaclass=SingletonMeta):
    def __init__(self, dir='./data', train=True, download=True, transform=None):
        self.dir = dir
        self.train = train
        self.download = download
        self.transform = transform
        self.dataset = None

    def get_data_loaders(self, batch_size=64, num_workers=4, input_size=(32, 32)):
        resize_function = ResizeFunction()
        size_key = f'{input_size[0]}x{input_size[1]}'
        resize_transform = resize_function.get_resize_transform(size_key)

        if resize_transform is None:
            resize_transform = transforms.Resize(input_size)

        torchvision_version = torchvision.__version__
        major_version = int(torchvision_version.split('.')[0])
        minor_version = int(torchvision_version.split('.')[1])

        if (major_version > 0) or (major_version == 0 and minor_version >= 9):
            affine_fill_arg = 'fill'
        else:
            affine_fill_arg = 'fillcolor'

        # Define transforms for training data
        train_transform = transforms.Compose([
            resize_transform,
            transforms.Lambda(lambda img: img.convert("RGB")),  # Convert all images to RGB
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=(-5, 5),
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                **{affine_fill_arg: (0, 0, 0)}  # Use fill or fillcolor
            ),
            transforms.ToTensor(),  # ToTensor() should always be at the end
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            # print('train size:', resize_transform)
        ])

        # Define transforms for validation and test data
        test_transform = transforms.Compose([
            resize_transform,
            transforms.Lambda(lambda img: img.convert("RGB")),  # Ensure images are RGB
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            # print('test size:', resize_transform)

        ])

        # Load the full training set
        full_train_set = datasets.CIFAR10(root=self.dir, train=True, download=self.download, transform=train_transform)

        # Split the training set into training and validation sets
        torch.manual_seed(42)  # For reproducibility
        train_size = int(0.8 * len(full_train_set))
        val_size = len(full_train_set) - train_size
        train_set, val_set = random_split(full_train_set, [train_size, val_size])
        for image, label in train_set:
            print(image.shape)
            break

        # Create loaders for the training and validation sets
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # Load the test set
        test_set = datasets.CIFAR10(root=self.dir, train=False, download=self.download, transform=test_transform)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        return train_loader, val_loader, test_loader, val_set, test_set

    def device(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            # print(f"GPU is available. Using {torch.cuda.get_device_name(0)}.")
        else:
            device = torch.device("cpu")
            print("GPU not available. Using CPU.")
        return device

def save_model(net, base_dir='C:/Users/jdah5454/PycharmProjects/task_cifar10', filename='cifar10.pth'):
    save_dir = os.path.join(base_dir, 'pth')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    torch.save(net.state_dict(), save_path)

def save_history_to_csv(
        history, base_dir='C:/Users/jdah5454/PycharmProjects/task_cifar10',
        filename='training_history.csv'):

    model_csv_dir = os.path.join(base_dir, 'model_csv')
    os.makedirs(model_csv_dir, exist_ok=True)

    save_path = os.path.join(model_csv_dir, filename)

    with open(save_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'val_loss', 'val_acc'])
        writer.writeheader()
        for epoch, data in enumerate(history):
            writer.writerow({
                'epoch': epoch + 1,
                'train_loss': f"{data['train_loss']:.6f}",
                'val_loss': f"{data['val_loss']:.6f}",
                'val_acc': f"{data['val_acc']:.6f}"
            })

def get_model(module_path, class_name):
    try:
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
        return model_class
    except (ImportError, AttributeError) as e:
        logging.error(f"Error loading model {class_name} from {module_path}: {e}")
        return None

def save_model_summary(model, input_size, save_dir, filename="model_summary.txt"):
    os.makedirs(save_dir, exist_ok=True)
    summary_file_path = os.path.join(save_dir, filename)

    # 기존 stdout 저장
    original_stdout = sys.stdout
    with open(summary_file_path, 'w') as f:
        sys.stdout = f  # stdout을 파일로 리디렉션
        try:
            summary(model, input_size=input_size)
        finally:
            sys.stdout = original_stdout

def load_config(config_dir=None):
    config_path = config_dir
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_dirs = os.path.join(script_dir, config_path)
    with open(config_dirs, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def countdown(t):
    try:
        while t:
            mins, secs = divmod(t, 60)
            timeformat = f'{mins:02d}:{secs:02d}'
            print(f'Remaining time: {timeformat}', end='\r')
            time.sleep(1)
            t -= 1
        print(' ' * 30, end='\r')
    except KeyboardInterrupt:
        print("\n종료됨")
