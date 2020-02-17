# Finished by zfang 2020/02/16 12:11pm
from abc import ABC, abstractmethod
import os
import tensorboardX


class Trainer(ABC):
    def __init__(self, args, cfg):
        self.args = args
        self.cfg = cfg

        # States of the Trainer that is going to be shared across every subclasses
        self.lr = self.cfg['lr']

        self.model_save_dir = "%s/%s/checkpoints" % (self.args.log_dir, self.args.experiment_name)
        self.tb_log_dir = "%s/%s/tb_logs" % (self.args.log_dir, self.args.experiment_name)
        self.image_save_dir = "%s/%s/images" % (self.args.log_dir, self.args.experiment_name)

        self.model_resume_dir = "%s/%s/checkpoints" % (self.args.log_dir, self.args.resume_exp)
        self.create_log_dirs()

        self.tb_writer = tensorboardX.SummaryWriter(self.tb_log_dir)

    def create_log_dirs(self):
        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(self.tb_log_dir, exist_ok=True)
        os.makedirs(self.image_save_dir, exist_ok=True)

    @abstractmethod
    def train(self):
        """Train method"""
        pass

    @abstractmethod
    def test(self):
        """Testing"""
        pass

    @abstractmethod
    def log(self, loss_dict, epoch, it, pbar):
        """Logger to print results and stuff"""
        pass

    @abstractmethod
    def setup_data(self):
        """Set up dataloaders for training/val/test"""
        pass

    @abstractmethod
    def setup_nets(self):
        """Set up network components for training/val/test"""
        pass

    @abstractmethod
    def setup_losses(self):
        """Set up network loss functions"""
        pass

    @abstractmethod
    def setup_optimizers(self):
        """Set up the network optimizers"""
        pass

    @abstractmethod
    def save_model(self, ep):
        """Save the model checkpoints"""
        pass

    @abstractmethod
    def resume(self, ep):
        """Resume from checkpoint"""
        pass

