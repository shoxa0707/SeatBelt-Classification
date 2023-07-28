from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms

class SeatBelt(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding='same')
        self.norm1 = nn.BatchNorm2d(16)
        self.drop1 = nn.Dropout(0.2)

        self.conv2 = nn.Conv2d(16, 32, 3, padding='same')
        self.norm2 = nn.BatchNorm2d(32)
        self.drop2 = nn.Dropout(0.2)

        self.conv3 = nn.Conv2d(32, 64, 3, padding='same')
        self.norm3 = nn.BatchNorm2d(64)
        self.drop3 = nn.Dropout(0.2)

        self.conv4 = nn.Conv2d(64, 128, 3, padding='same')
        self.norm4 = nn.BatchNorm2d(128)
        self.drop4 = nn.Dropout(0.2)

        self.conv5 = nn.Conv2d(128, 256, 3, padding='same')
        self.norm5 = nn.BatchNorm2d(256)
        self.drop5 = nn.Dropout(0.2)

        self.conv6 = nn.Conv2d(256, 512, 3, padding='same')
        self.norm6 = nn.BatchNorm2d(512)
        self.drop6 = nn.Dropout(0.2)
        
        self.pool = nn.MaxPool2d(2)
        self.l1 = nn.Linear(512*5*5, 1024)
        self.l2 = nn.Linear(1024, 256)
        self.l3 = nn.Linear(256, 3)

        self.data_aug = transforms.Compose([
                                            transforms.RandomVerticalFlip(p=1),
                                            transforms.RandomHorizontalFlip(p=1),
                                            transforms.RandomRotation(0.2)
                                        ])
        
    def forward(self, x):
        x = self.data_aug(x)

        x = F.relu(self.conv1(x))
        x = self.pool(self.norm1(x))
        x = self.drop1(x)

        x = F.relu(self.conv2(x))
        x = self.pool(self.norm2(x))
        x = self.drop2(x)

        x = F.relu(self.conv3(x))
        x = self.pool(self.norm3(x))
        x = self.drop3(x)
        
        x = F.relu(self.conv4(x))
        x = self.pool(self.norm4(x))
        x = self.drop4(x)

        x = F.relu(self.conv5(x))
        x = self.pool(self.norm5(x))
        x = self.drop5(x)

        x = F.relu(self.conv6(x))
        x = self.pool(self.norm6(x))
        x = self.drop6(x)

        x = x.view(-1, 512*5*5)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        
        return x