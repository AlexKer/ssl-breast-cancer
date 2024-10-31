import torch
import torch.nn as nn
from resnet import resnet18, resnet34, resnet50 

class SSLModel(nn.Module):
    def __init__(self, resnet_type=50):
        super(SSLModel, self).__init__()
        if resnet_type == 18:
            self.encoder = resnet18()
        elif resnet_type == 34:
            self.encoder = resnet34()
        elif resnet_type == 50:
            self.encoder = resnet50()

        self.classify = nn.Sequential(
            nn.Linear(10*4, 10),
            nn.LeakyReLU(10),
            nn.Linear(10, 1)
        )

    def forward(self, first, second, third, fourth):
        f1 = self.encoder(first).squeeze() 
        f2 = self.encoder(second).squeeze() 
        f3 = self.encoder(third).squeeze() 
        f4 = self.encoder(fourth).squeeze() 

        concatenated = torch.cat([f1, f2, f3, f4], dim=1)

        predicted = self.classify(concatenated) 

        return predicted