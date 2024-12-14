#########################################################################################################################
# SWaV model from last time implementation of SWaV, still runable, not best performance but working and extremelly fast #
#########################################################################################################################
import pytorch_lightning as pl
import torch
import torchvision
from torch import nn
from lightly.loss import SwaVLoss
from lightly.models.modules import SwaVProjectionHead, SwaVPrototypes
from lightly.models.modules.memory_bank import MemoryBankModule
from sklearn.model_selection import train_test_split
import json
from dataloader import get_data_loader

accelerator = "gpu" if torch.cuda.is_available() else "CPU" 

loss_record = []

class SWaV(pl.LightningModule):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        self.projection_head = SwaVProjectionHead(2048, 512, 128)
        self.prototypes = SwaVPrototypes(128, 512, 1)

        self.start_queue_at_epoch = 2
        self.queues = nn.ModuleList([MemoryBankModule(size=(3840, 128)) for _ in range(2)])

        self.criterion = SwaVLoss()

        self.batch_losses = []

    def training_step(self, batch, batch_idx):
        #print(f"Processing batch {batch_idx}...")

        # Unpack the batch from the data loader
        (t0_first, t1_first, t2_first, t3_first), non_mri_features, labels = batch
        #print("Batch unpacked successfully.")

        # Only use the MRI images for SwaV training
        inputs = torch.cat((t0_first, t1_first, t2_first, t3_first), dim=0)
        print("Inputs shape:", inputs.shape)

        views = inputs.chunk(2)  # Split into high-resolution and low-resolution views
        high_resolution, low_resolution = views
        print("High-resolution shape:", high_resolution.shape)
        print("Low-resolution shape:", low_resolution.shape)

        self.prototypes.normalize()
        #print("Prototypes normalized.")

        # Extract features and compute prototypes
        high_resolution_features = self._subforward(high_resolution)
        low_resolution_features = self._subforward(low_resolution)
        print("High-resolution features shape:", high_resolution_features.shape)
        print("Low-resolution features shape:", low_resolution_features.shape)

        high_resolution_prototypes = self.prototypes(high_resolution_features, self.current_epoch)
        low_resolution_prototypes = self.prototypes(low_resolution_features, self.current_epoch)
        print("High-resolution prototypes shape:", high_resolution_prototypes.shape)
        print("Low-resolution prototypes shape:", low_resolution_prototypes.shape)

        queue_prototypes = self._get_queue_prototypes(high_resolution_features)
        queue_prototypes = torch.stack(queue_prototypes, dim=0)
        queue_prototypes = queue_prototypes.view(-1, queue_prototypes.size(-1))
        queue_prototypes = queue_prototypes[:4]
        print("Queue Prototypes shape:", queue_prototypes.shape)

        # Compute SwaV loss 
        loss = self.criterion([high_resolution_prototypes], [low_resolution_prototypes], [queue_prototypes])
        self.log('train_loss', loss)
        print("Loss computed:", loss.item())

        self.batch_losses.append(loss.item())

        return loss

    def on_train_epoch_end(self):
        if self.batch_losses:
            epoch_loss = round(sum(self.batch_losses) / len(self.batch_losses), 3)
            loss_record.append(epoch_loss)
            print(f"Epoch {self.current_epoch} loss: {epoch_loss}")
        self.batch_losses.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def _subforward(self, inputs):
        #print("Running subforward...")

        inputs = torch.mean(inputs, dim=2)  
        #print("Input shape:", inputs.shape)

        # Expand single channel to three channels
        inputs = inputs.repeat(1, 3, 1, 1)  # [B, 3, H, W]
        #print("Input shape after channel expansion:", inputs.shape)

        # Pass through ResNet backbone
        features = self.backbone(inputs).flatten(start_dim=1)  # ResNet expects [B, 3, H, W]
        #print("Features shape after backbone:", features.shape)

        # Pass through projection head
        features = self.projection_head(features)
        
        # Normalize features
        features = nn.functional.normalize(features, dim=1, p=2)

        #print("Subforward completed. Features shape:", features.shape)
        return features

    @torch.no_grad()
    def _get_queue_prototypes(self, high_resolution_features):
        #print("Getting queue prototypes...")
        try:
            #print("High-resolution features shape before chunking:", high_resolution_features.shape)
            
            # Check if chunking works properly
            chunks = high_resolution_features.chunk(2)
            #print(f"Number of chunks: {len(chunks)}")

            # Ensure the number of queues matches the number of chunks
            if len(self.queues) != len(chunks):
                raise ValueError(f"The number of queues ({len(self.queues)}) does not match the number of chunks ({len(chunks)})")

            queue_features = []
            for i, queue in enumerate(self.queues):
                # Access the chunk and compute features
                _, features = queue(chunks[i], update=True)
                features = features.T
                queue_features.append(features)
                #print(f"Queue {i} features shape:", features.shape)

            # Compute prototypes
            prototypes = [self.prototypes(q, self.current_epoch) for q in queue_features]
            #print("Queue prototypes computed.")
            return prototypes
        
        except Exception as e:
            print(f"Error in _get_queue_prototypes: {e}")
            raise

def run_swav_model():
    with open('SSL_dataset.json', 'r') as f:
        data_list = json.load(f)
        print("Training starts.")

    # Step 1: Split into train and test sets 
    train_val_data_indices = list(range(len(data_list)))
    train_indices, val_indices = train_test_split(train_val_data_indices, test_size=0.10, shuffle=True, random_state=42)

    train_loader = get_data_loader("train", train_indices, data_list=data_list, batch_size=2)

    # Define the model and trainer
    model = SWaV()

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="checkpoints/", 
        filename="swav-{epoch:02d}-{train_loss:.3f}",
        save_top_k=3, 
        monitor="train_loss", 
        mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=1, 
        devices=1, 
        accelerator=accelerator, 
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model=model, train_dataloaders=train_loader)

if __name__ == "__main__":
    run_swav_model()
    print(loss_record)