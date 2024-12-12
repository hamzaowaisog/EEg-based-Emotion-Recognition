import torch.nn as nn
import torch

class DEBaseEncoder(nn.Module):
    def __init__(self):
        super(DEBaseEncoder, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(32*4,128)
        
    def forward(self, x):
        #  print(f"Input to encoder: {x.shape}")
         x = self.pool(nn.ReLU()(self.conv1(x)))
        #  print(f"After conv1: {x.shape}")
         x = self.pool(nn.ReLU()(self.conv2(x)))
        #  print(f"After conv2: {x.shape}")
         x = x.view(x.size(0), -1)  # Flatten
        #  print(f"Flattened Shape: {x.shape}")
         x = nn.ReLU()(self.fc(x))
         return x
    
class projector (nn.Module):
    def __init__(self, input_dim=128, output_dim=64):
        super(projector, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(input_dim,output_dim),
            nn.ReLU(),
            nn.Linear(output_dim,output_dim)
        )
    def forward(self, x):
        # print(f"Input to projector: {x.shape}")
        x = self.project(x)
        # print(f"Output of projector: {x.shape}")
        return x
    
class EmotionClassifier(nn.Module):
    def __init__(self, input_dim=128, num_classes=3):
        super(EmotionClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        z_i = nn.functional.normalize(z_i, dim=1)
        z_j = nn.functional.normalize(z_j, dim=1)
        similarity_matrix = torch.matmul(z_i, z_j.T) / self.temperature
        # print(f"Similarity Matrix: {similarity_matrix.shape}")
        labels = torch.arange(z_i.size(0)).to(z_i.device)
        # print(f"Labels: {labels.shape}")
        loss = nn.CrossEntropyLoss()(similarity_matrix, labels)
        # print(f"Contrastive Loss: {loss}")
        return loss

import torch.nn as nn

class CLISAModel(nn.Module):
    def __init__(self, input_dim=128, num_classes=3):
        super(CLISAModel, self).__init__()
        self.encoder = DEBaseEncoder()
        self.projector = projector(input_dim=input_dim)
        self.classifier = EmotionClassifier(input_dim=64, num_classes=num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.projector(x)
        x = self.classifier(x)
        return x
