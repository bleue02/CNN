import torch.nn as nn
import torch.nn.functional as F

# kernel: 5x5, stride:1, padding:2, conv layers: 4, epoch: 50, 100, 150, 200, 250, 300
# pooling: max
class Image_8x8_5x5_max(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(Image_8x8_5x5_max, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 16, 5, padding=2)
        self.bn_conv1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, 5, padding=2)
        self.bn_conv2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, 5, padding=2)
        self.bn_conv3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 64, 5, padding=2)
        self.bn_conv4 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout_conv = nn.Dropout2d(p=0.25)

        # Fully Connected layers
        self.fc1 = nn.Linear(64 * 2 * 2, 256)
        self.bn_dense1 = nn.BatchNorm1d(256)

        self.fc2 = nn.Linear(256, 128)
        self.bn_dense2 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(p=0.5)

        self.fc3 = nn.Linear(128, num_classes)

    def conv_layers(self, x):
        x = F.relu(self.bn_conv1(self.conv1(x)))  # (8 + 2 x 2) -5 - > / 1 + 1-> 8
        x = F.relu(self.bn_conv2(self.conv2(x)))  # (8 + 2 x 2) -5 - > / 1 + 1-> 8
        x = self.pool(x)  # 8 / 2 -> 4
        x = self.dropout_conv(x)

        x = F.relu(self.bn_conv3(self.conv3(x)))  # (4 + 2 x 2) -5 - > / 1 + 1-> 4
        x = F.relu(self.bn_conv4(self.conv4(x)))  # (4 + 2 x 2) -5 - > / 1 + 1-> 4
        x = self.pool(x)  # 4 / 2 -> 2
        x = self.dropout_conv(x)
        return x

    def dense_layers(self, x):
        x = F.relu(self.bn_dense1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn_dense2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.dense_layers(x)
        return x

class Image_8x8_5x5_avg(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(Image_8x8_5x5_avg, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 16, 5, padding=2)
        self.bn_conv1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, 5, padding=2)
        self.bn_conv2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, 5, padding=2)
        self.bn_conv3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 64, 5, padding=2)
        self.bn_conv4 = nn.BatchNorm2d(64)

        self.pool = nn.AvgPool2d(2, 2)  # Avg Pooling으로 변경
        self.dropout_conv = nn.Dropout2d(p=0.25)

        # Fully Connected layers
        self.fc1 = nn.Linear(64 * 2 * 2, 256)
        self.bn_dense1 = nn.BatchNorm1d(256)

        self.fc2 = nn.Linear(256, 128)
        self.bn_dense2 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(p=0.5)

        self.fc3 = nn.Linear(128, num_classes)

    def conv_layers(self, x):
        x = F.relu(self.bn_conv1(self.conv1(x)))
        x = F.relu(self.bn_conv2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout_conv(x)

        x = F.relu(self.bn_conv3(self.conv3(x)))
        x = F.relu(self.bn_conv4(self.conv4(x)))
        x = self.pool(x)
        x = self.dropout_conv(x)
        return x

    def dense_layers(self, x):
        x = F.relu(self.bn_dense1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn_dense2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.dense_layers(x)
        return x

if __name__ == '__main__':
    from torchsummary import summary
    from executable_code.logging_pck.utils import Utils
    utils_instance = Utils()
    device = utils_instance.device()

    model = Image_8x8_5x5_max().to(device)
    summary(model, (3, 8, 8))
