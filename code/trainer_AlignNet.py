
# Finished by zfang 2020/02/16 12:11pm
import torch
import torch.nn as nn
from torch.optim import Adam
import torchvision.transforms.functional as TF
from torch.autograd import Variable
import os.path as osp
from tqdm import tqdm
import random
import numpy as np
import scipy
import scipy.io
from functools import reduce

from code.trainer import Trainer

from dataset.lip_dataset import VoxCelebDataset
from model.Networks import AlignNet
from model.Losses import AlignNet_losses
from utils.utils import get_dataloader, COLORS

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class AlignNetTrainer(Trainer):
    def __init__(self, args, cfg):
        super(AlignNetTrainer, self).__init__(args, cfg)
        if self.args.mode == "train":
            self.train_dataloader, self.val_dataloader = self.setup_data()
        elif self.args.mode == "test":
            self.test_dataloader = self.setup_data()
        # Setup network
        self.setup_nets()
        # Setup network loss
        self.setup_losses()
        # Setup optimizer
        self.setup_optimizers()

    def setup_data(self):
        if self.args.mode == "train":
            train_dataset = VoxCelebDataset(self.args, self.cfg, mode="train")
            train_dataloader = get_dataloader(self.cfg, train_dataset)

            val_dataset = VoxCelebDataset(self.args, self.cfg, mode="val")
            val_dataloader = get_dataloader(self.cfg, val_dataset)

            return train_dataloader, val_dataloader
        elif self.args.mode == "test":
            test_dataset = VoxCelebDataset(self.args, self.cfg, mode="test")
            test_dataloader = get_dataloader(self.cfg, test_dataset)
            return test_dataloader

    def setup_nets(self):
        if not self.args.data_parallel:
            align_net = AlignNet(self.cfg).to(device)
        else:
            align_net = nn.DataParallel(AlignNet(self.cfg).to(device))
        setattr(self, "align_net", align_net)

    def setup_losses(self):
        losses = AlignNet_losses()
        setattr(self, 'losses', losses)

    def setup_optimizers(self):
        opt = Adam(params=list(self.align_net.parameters()),
                          lr=self.lr)
        setattr(self, "opt", opt)

    def train(self):
        loss_dict = {}
        ep_bar = tqdm(range(self.cfg['n_epoch']))

        if self.args.resume:
            self.resume(ep=0)

        for ep in ep_bar:
            self.align_net.train()
            batch_bar = tqdm(self.train_dataloader)
            for batch, data in enumerate(batch_bar):
                it = ep * len(self.train_dataloader) + batch
                ###############################################################
                # 0. Grab the data
                ###############################################################
                video_feature = data['video_feature'].to(device)
                audio_feature = data['audio_feature'].to(device)
                target = data['target'].to(device)
                sample_names = data['sample_name']

                ###############################################################
                # 1. Push thru :)
                ###############################################################
                preds = self.align_net(video_feature, audio_feature)

                ###############################################################
                # 2. Compute losses and optimize
                ###############################################################
                # Calculate loss
                loss_l1, loss_mono = self.losses(preds=preds, target=target, multilevel_supervision=self.cfg["multilevel_supervision"])
                loss_dict['loss_l1'] = loss_l1.item()
                loss_dict['loss_mono'] = loss_mono.item()
                loss = loss_l1 + loss_mono

                # Optimizer step
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                if it % self.cfg['log_batch'] == 0:
                    self.log(loss_dict, ep, it, batch_bar)

            batch_bar.close()

            # save model
            if ep % self.cfg['save_epoch'] == 0:
                # save model
                self.save_model(ep=ep)

                # save example from validation set
                self.align_net.eval()
                for batch, data in enumerate(self.val_dataloader):
                    ###############################################################
                    # 0. Grab the data
                    ###############################################################
                    video_feature = data['video_feature'].to(device)
                    audio_feature = data['audio_feature'].to(device)
                    target = data['target'].to(device)
                    sample_names = data['sample_name']

                    ###############################################################
                    # 1. Push thru :)
                    ###############################################################
                    preds = self.align_net(video_feature, audio_feature)

                    ###############################################################
                    # 2. Compute losses and optimize
                    ###############################################################
                    # Calculate loss
                    loss_l1, loss_mono = self.losses(preds, target, multilevel_supervision=False)
                    loss_l1 = loss_l1.item()
                    loss_mono = loss_mono.item()
                    loss_tot = loss_l1 + loss_mono

                    ###############################################################
                    # 3. Save targets and preds
                    ###############################################################
                    target = target.cpu().detach().numpy()
                    preds = [pred.cpu().detach().numpy() for pred in preds]
                    save_dict = {}
                    save_dict['target'] = target
                    for idx, pred in enumerate(preds):
                        save_dict['pred'+str(idx)] = pred
                    scipy.io.savemat(osp.join(self.image_save_dir, 'val-{:06d}-{:0.4f}.mat'.format(ep,loss_tot)), mdict=save_dict)

                    break

        ep_bar.close()

    def test(self):
        # video retargeting: to be implemented
        return None

    def log(self, loss_dict, epoch, it, pbar):
        print_str = "Epoch %03d, batch %06d:" % (epoch, it)
        for loss, val in loss_dict.items():
            print_str += "%s %s %.4f," % (COLORS.OKGREEN, loss, val)
            self.tb_writer.add_scalar(loss, val, global_step=it)

        print_str += COLORS.ENDC
        pbar.write(print_str)

    def save_model(self, ep):
        # Save the optimizer's state_dict
        torch.save(self.opt.state_dict(), osp.join(self.model_save_dir,
                                                     "opt-%03d.ckpt" % ep))

        # Save the network's state_dict
        torch.save(self.align_net.state_dict(), osp.join(self.model_save_dir,
                                                          "align_net-%03d.ckpt" % ep))

    def resume(self, ep):
        # Load the optimizer's state_dict
        self.opt.load_state_dict(torch.load(
            osp.join(self.model_resume_dir, "opt-%03d.ckpt" % ep)))

        # Load the networks' state_dict
        self.align_net.load_state_dict(torch.load(
            osp.join(self.model_resume_dir, "align_net-%03d.ckpt" % ep)))


