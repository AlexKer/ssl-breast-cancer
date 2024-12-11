import numpy as np
import torch
import torch.optim as optim
import logging
import random
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from torch.optim.lr_scheduler import CosineAnnealingLR
from warmup_scheduler import GradualWarmupScheduler
from MoCov2_shuffle import MoCov2, kendall_tau_distance, MoCov2DataAugmentation

class trainer:
    def __init__(
            self,
            dataloader,
            log_dir,
            epochs=50,
            warmup_epochs=5,
            lr=0.005,
            min_lr=1e-6,
            shuffle_prob=0.8
    ):  
        self.model = MoCov2().cuda()
        self.dataloader = dataloader
        self.log_dir = log_dir
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.lr = lr
        self.min_lr = min_lr
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.shuffle_prob = shuffle_prob
        self.argumentation = MoCov2DataAugmentation()
        self._setup_optimizer_and_scheduler()
        self.setup_logging()

    def setup_logging(self):
        """Setup logging directories and wandb"""
        self.log_dir = Path(self.log_dir) / datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup file and console logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / 'train.log'),
                logging.StreamHandler()
            ]
        )

    def _setup_optimizer_and_scheduler(self):
        """Setup optimizer and learning rate scheduler"""
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
        )

        scheduler_cosine = CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs - self.warmup_epochs,
            eta_min=self.min_lr
        )

        self.scheduler = GradualWarmupScheduler(
            self.optimizer,
            multiplier=1.0,
            total_epoch=self.warmup_epochs,
            after_scheduler=scheduler_cosine
        )

        self.optimizer.zero_grad()
        self.optimizer.step()
        if self.warmup_epochs == 0:
            self.scheduler.step()

    def save_checkpoint(self):
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_valid_acurracy': self.best_valid_acurracy,
        }
        checkpoint_path = self.log_dir / 'checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, path):
        """Load model and training state from checkpoint"""
        logging.info(f'Loading checkpoint from {path}')
        checkpoint = torch.load(path)
        
        self.current_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_valid_acurracy = checkpoint['best_valid_acurracy']
        
        logging.info(f'Loaded checkpoint from epoch {self.current_epoch}')

    def train_one_epoch(self):
        self.model.train()

        current_lr = self.scheduler.get_last_lr()[0]
        logging.info(f'Current learning rate: {current_lr:.8f}')
        total_loss = 0

        progress_bar = tqdm(enumerate(self.dataloader), total=len(self.dataloader), desc=f"Epoch {self.current_epoch}/{self.epochs}")
        for batch_index, batch in progress_bar:
            images, _ = batch

            batch_size, num_samples, *rest = images.shape
            images = images.view(batch_size * num_samples, *rest)  # Now shape: (18, 4, 60, 230, 230)

            first_images = images[:, 0, :, :, :].cuda()
            second_images = images[:, 1, :, :, :].cuda()
            third_images = images[:, 2, :, :, :].cuda()
            fourth_images = images[:, 3, :, :, :].cuda()

            q_0, k_0 = self.argumentation(first_images)
            q_1, k_1 = self.argumentation(second_images)
            q_2, k_2 = self.argumentation(third_images)
            q_3, k_3 = self.argumentation(fourth_images)

            qs = np.array([q_0, q_1, q_2, q_3])
            if random.random() < self.shuffle_prob:
                q_0, q_1, q_2, q_3 = qs[kendall_tau_distance(10)]

            self.optimizer.zero_grad()
            loss = self.model(q_0, q_1, q_2, q_3, k_0, k_1, k_2, k_3)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / (batch_index + 1):.4f}',
                'lr': f'{current_lr:.6f}'
            })
        return total_loss / len(self.dataloader)

    def train(self):
        logging.info(f'Starting training with config:\n{vars(self.config)}')
        logging.info(f'Model parameters: {sum(p.numel() for p in self.model.parameters())}')
        
        try:
            for epoch in range(self.current_epoch, self.config.epochs):
                self.current_epoch = epoch
                logging.info(f'Starting epoch {epoch}')
                
                train_loss = self.train_one_epoch()
                self.scheduler.step()

                logging.info(
                    f'Epoch {epoch}: train_loss={train_loss:.4f}, '
                    f'lr={self.scheduler.get_last_lr()[0]:.8f}'
                )

        except KeyboardInterrupt:
            logging.info('Training interrupted by user')
            self.save_checkpoint()
        
        except Exception as e:
            logging.error(f'Training failed with error: {str(e)}')
            self.save_checkpoint()
            raise e

        finally:
            logging.info(f'Training finished.')


