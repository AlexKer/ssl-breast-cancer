import pytorch_lightning as pl
import torch
import torchvision
from torch import nn
from lightly.loss import SwaVLoss
from lightly.models.modules import SwaVPrototypes
from lightly.models.modules.memory_bank import MemoryBankModule
from sklearn.model_selection import train_test_split
import json
import random
from dataloader import get_data_loader

accelerator = "gpu" if torch.cuda.is_available() else "cpu"
loss_record = []

class CustomProjectionHead(nn.Module):
    """
    Custom projection head with GroupNorm
    """
    def __init__(self, input_dim=2048, hidden_dim=512, output_dim=128, num_groups=16):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.GroupNorm(num_groups, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim, bias=False),
            nn.GroupNorm(num_groups, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

class SWaV(pl.LightningModule):
    """
    SWaV model
    """
    def __init__(self,
                 n_prototypes=1500,             # Number of prototypes
                 queue_length=3840,             # Queue length
                 lr=1e-3,                       # Learning rate
                 crops_for_assign=[0,1],        # Crops for assignment
                 temperature=0.1,               # Temperature
                 sinkhorn_iterations=3,         # Sinkhorn iterations
                 epsilon=0.05):                 # Sinkhorn epsilon
        super().__init__()
        self.save_hyperparameters()

        # ResNet50 backbone
        resnet = torchvision.models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # [B,2048,1,1]

        # Custom projection head with GroupNorm
        self.projection_head = CustomProjectionHead(
            input_dim=2048,
            hidden_dim=512,
            output_dim=128,
            num_groups=16
        )

        # Prototypes
        self.prototypes = SwaVPrototypes(
            input_dim=128,
            n_prototypes=n_prototypes
        )

        # Loss
        self.criterion = SwaVLoss(
            sinkhorn_iterations=sinkhorn_iterations,
            sinkhorn_epsilon=0.05,
            temperature=temperature
        )

        # Queues for global crops
        self.queues = nn.ModuleList(
            [MemoryBankModule(size=(queue_length, 128)) for _ in crops_for_assign]
        )

        self.batch_losses = []
        self.channel_reducer = self.ChannelReducer()

    class ChannelReducer(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(in_channels=4, out_channels=3, kernel_size=1, bias=False)
            
        def forward(self, x):
            return self.conv(x)

    def forward(self, x):
        # x: [B,4,T,H,W]
        x = torch.mean(x, dim=2)        # [B,4,H,W]
        x = self.channel_reducer(x)     # [B,3,H,W]

        with torch.no_grad():
            feats = self.backbone(x).flatten(start_dim=1)   # [B,2048]
        z = self.projection_head(feats)                     # [B,128]
        z = nn.functional.normalize(z, dim=1)
        return z

    def training_step(self, batch, batch_idx):
        (t0_first_images, t1_first_images, t2_first_images, t3_first_images), non_mri_features, labels = batch

        # 2 global + 4 local = 6 crops total
        N = 6  
        B_N = t0_first_images.size(0)
        B = B_N // N

        t0_first_images = t0_first_images.view(B, N, 1, t0_first_images.size(2), t0_first_images.size(3), t0_first_images.size(4))
        t1_first_images = t1_first_images.view(B, N, 1, t1_first_images.size(2), t1_first_images.size(3), t1_first_images.size(4))
        t2_first_images = t2_first_images.view(B, N, 1, t2_first_images.size(2), t2_first_images.size(3), t2_first_images.size(4))
        t3_first_images = t3_first_images.view(B, N, 1, t3_first_images.size(2), t3_first_images.size(3), t3_first_images.size(4))

        crops = []
        for i in range(N):
            crop_i = torch.cat((t0_first_images[:, i],
                                t1_first_images[:, i],
                                t2_first_images[:, i],
                                t3_first_images[:, i]), dim=1)
            crops.append(crop_i)

        self.prototypes.normalize()
        embeddings = [self.forward(crop) for crop in crops]
        prototypes = [self.prototypes(e, self.current_epoch) for e in embeddings]

        # first two = global
        high_resolution_prototypes = prototypes[:2]

        # last four = local
        low_resolution_prototypes = prototypes[2:]

        # Update queues
        queue_outputs = []
        with torch.no_grad():
            for i, queue_module in enumerate(self.queues):
                assign_crop_idx = self.hparams.crops_for_assign[i]
                _, features = queue_module(embeddings[assign_crop_idx], update=True)
                features = features.T
                queue_proto = self.prototypes(features, self.current_epoch)
                queue_outputs.append(queue_proto)

        loss = self.criterion(
            high_resolution_outputs=high_resolution_prototypes,
            low_resolution_outputs=low_resolution_prototypes,
            queue_outputs=queue_outputs
        )

        self.log('train_loss', loss)
        self.batch_losses.append(loss.item())

        return loss

    def on_train_epoch_end(self):
        if self.batch_losses:
            epoch_loss = round(sum(self.batch_losses) / len(self.batch_losses), 3)
            loss_record.append(epoch_loss)
            print(f"Epoch {self.current_epoch} loss: {epoch_loss}")
        self.batch_losses.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

def run_swav_model():
    with open('SSL_dataset.json', 'r') as f:
        data_list = json.load(f)
        print("Training starts.")

    # Use half of the samples randomly to reduce memory usages
    half_size = len(data_list) // 2
    data_list = random.sample(data_list, half_size)

    train_val_data_indices = list(range(len(data_list)))
    train_indices, val_indices = train_test_split(train_val_data_indices, test_size=0.10, shuffle=True, random_state=42)

    train_loader = get_data_loader("train", train_indices, data_list=data_list, batch_size=1, num_workers=0)

    model = SWaV(
        n_prototypes=1500,                                  # Number of prototypes
        queue_length=3840,                                  # Queue length
        lr=1e-3,                                            # Learning rate
        crops_for_assign=[0,1],                             # Crops for assignment
        temperature=0.1,                                    # Temperature
        sinkhorn_iterations=3,                              # Sinkhorn iterations
        epsilon=0.05                                        # Sinkhorn epsilon
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="checkpoints/",                             # Output directory
        filename="swav-{epoch:02d}-{train_loss:.3f}",       # Checkpoint filename
        save_top_k=3,                                       # Save top-k checkpoints            
        monitor="train_loss",                               # Metric to monitor
        mode="min",                                         # Minimize the metric
        save_last=True                                      # Save the last checkpoint
    )

    trainer = pl.Trainer(
        max_epochs=3,                                       # Number of epochs
        devices=1,                                          # Number of devices
        accelerator=accelerator,                            # Accelerator
        precision="16-mixed",                               # Precision
        callbacks=[checkpoint_callback],                    # Callbacks
        accumulate_grad_batches=2                           # Accumulate gradients
    )

    trainer.fit(model=model, train_dataloaders=train_loader)

if __name__ == "__main__":
    run_swav_model()
    print(loss_record)