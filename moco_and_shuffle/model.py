import torch
import torch.nn as nn
import numpy as np

class Model(nn.Module):
    def __init__(self, resnet, embedding_size = 64):
        super(Model, self).__init__()

        self.encoder = resnet

        output_channels = 128
        '''
        FCL FOR PROCESSING OUTPUT OF ENCODER
        '''
        self.latent_reducer = nn.Sequential(
            nn.Linear(output_channels, 500),
            nn.LeakyReLU(inplace=True),
            nn.LayerNorm(500),
            nn.Dropout(0.1),
            nn.Linear(500, 100),
            nn.LeakyReLU(inplace=True),
            nn.LayerNorm(100),
            nn.Dropout(0.1),
            # nn.Linear(100, 30),
            # nn.LeakyReLU(inplace=True),
            # nn.LayerNorm(30),
            nn.Linear(100, embedding_size),
            nn.LeakyReLU(inplace=True),
            nn.LayerNorm(embedding_size),
            nn.Dropout(0.2),
        ) 
        
        self.flatten = nn.Flatten()

        '''
        FINAL CLASSIFICATION LAYER FOR OUTPUT OF ATTENTION
        '''
        self.classify_mri = nn.Sequential(
            nn.Linear(embedding_size * 2, embedding_size),
            nn.LeakyReLU(inplace=True),
            nn.Linear(embedding_size, 10),
            nn.LeakyReLU(inplace=True),
            nn.Linear(10, 1),
        )

    '''
    non_mri_data: size 5, race_encoded is a six length ohe, metapausal status is a three lengths ohe
      [hr, her2, age, race_encoded, metapausal_status]
      ex: hr 0, her2 1, age 0.54, race [1, 0, 0, 0, 0, 0], metapause status [0, 0, 1]
    ''' 
    def forward(self, first_img, second_img, non_mri_data):
        '''
        ENCODE THE IMAGES

        1. FEED THRU RESNET 
        2. FEED THRU FCL
        3. ADD POSITIONAL ENCODING
        '''
        first_encoded = self.encoder(first_img).squeeze()
        second_encoded = self.encoder(second_img).squeeze()
        # print("shape after resnet", first_encoded.shape)

        batch_size = len(non_mri_data)
        # Ensure the tensors are at least 2D, especially for batch size 1
        if batch_size == 1:
            first_encoded = first_encoded.unsqueeze(0)
            second_encoded = second_encoded.unsqueeze(0)

        first_reduced = self.latent_reducer(first_encoded)
        second_reduced = self.latent_reducer(second_encoded)
     

        mri_features = torch.cat([first_reduced, second_reduced], dim = 1)
        logits = self.classify_mri(mri_features)

        return logits 
    