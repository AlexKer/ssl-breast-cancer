import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
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
            vis_loader,
            log_dir,
            dim=128,
            resnet_type=50,
            K=68,
            epochs=50,
            warmup_epochs=5,
            lr=0.005,
            min_lr=1e-6,
            shuffle_prob=0.8,
            resume_from = None
    ):  
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MoCov2(dim=dim,
                            resnet_type=resnet_type,
                            K=K,
                            ).to(self.device)
        self.dataloader = dataloader
        self.vis_loader = vis_loader
        self.log_dir = log_dir
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.lr = lr
        self.min_lr = min_lr
        self.shuffle_prob = shuffle_prob
        self.argumentation = MoCov2DataAugmentation()
        self._setup_optimizer_and_scheduler()
        self.setup_logging()

        self.current_epoch = 0

        if resume_from:
            self.load_checkpoint(resume_from)

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
            # 'best_valid_acurracy': self.best_valid_acurracy
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
        # self.best_valid_acurracy = checkpoint['best_valid_acurracy']
        
        logging.info(f'Loaded checkpoint from epoch {self.current_epoch}')

    def train_one_epoch(self):
        self.model.train()

        current_lr = self.scheduler.get_last_lr()[0]
        logging.info(f'Current learning rate: {current_lr:.8f}')
        total_loss = 0

        progress_bar = tqdm(enumerate(self.dataloader), total=len(self.dataloader), desc=f"Epoch {self.current_epoch}/{self.epochs}")
        for batch_index, batch in progress_bar:
            (t0_first_images, t1_first_images, t2_first_images, t3_first_images), _, _ = batch

            first_images = t0_first_images.squeeze(1)
            second_images = t1_first_images.squeeze(1)
            third_images = t2_first_images.squeeze(1)
            fourth_images = t3_first_images.squeeze(1)
            q_0, k_0 = self.argumentation(first_images)
            q_1, k_1 = self.argumentation(second_images)
            q_2, k_2 = self.argumentation(third_images)
            q_3, k_3 = self.argumentation(fourth_images)

            qs = np.array([q_0, q_1, q_2, q_3])
            condition = False
            if random.random() < self.shuffle_prob:
                condition = True
                q_0, q_1, q_2, q_3 = qs[kendall_tau_distance(10)]
            
            if condition:
                q_0 = torch.from_numpy(q_0).unsqueeze(1)
                q_1 = torch.from_numpy(q_1).unsqueeze(1)
                q_2 = torch.from_numpy(q_2).unsqueeze(1)
                q_3 = torch.from_numpy(q_3).unsqueeze(1)
            else:
                q_0 = q_0.unsqueeze(1)
                q_1 = q_1.unsqueeze(1)
                q_2 = q_2.unsqueeze(1)
                q_3 = q_3.unsqueeze(1)

            k_0 = k_0.unsqueeze(1)
            k_1 = k_1.unsqueeze(1)
            k_2 = k_2.unsqueeze(1)
            k_3 = k_3.unsqueeze(1)

            self.optimizer.zero_grad()
            logits, labels = self.model(q_0.to(self.device), q_1.to(self.device), q_2.to(self.device), q_3.to(self.device), 
                              k_0.to(self.device), k_1.to(self.device), k_2.to(self.device), k_3.to(self.device))
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / (batch_index + 1):.4f}',
                'lr': f'{current_lr:.6f}'
            })
        return total_loss / len(self.dataloader)

    def train(self):
        logging.info(f'Starting training')
        logging.info(f'Model parameters: {sum(p.numel() for p in self.model.parameters())}')
        
        try:
            for epoch in range(self.current_epoch, self.epochs):
                self.current_epoch = epoch
                logging.info(f'Starting epoch {epoch}')
                
                train_loss = self.train_one_epoch()
                self.scheduler.step()

                logging.info(
                    f'Epoch {epoch}: train_loss={train_loss:.4f}, '
                    f'lr={self.scheduler.get_last_lr()[0]:.8f}'
                )

                self.save_checkpoint()

        except KeyboardInterrupt:
            logging.info('Training interrupted by user')
            self.save_checkpoint()
        
        except Exception as e:
            logging.error(f'Training failed with error: {str(e)}')
            self.save_checkpoint()
            raise e

        finally:
            logging.info(f'Training finished.')



