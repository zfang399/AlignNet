# Finished by zfang 2020/02/15 20:30pm
import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F

class AlignNet_losses(_Loss):
    def __init__(self):
        super(AlignNet_losses, self).__init__(True)

    def forward(self, preds, target, multilevel_supervision=True):
        loss_ol = 0
        loss_mono = 0
        for i in range(len(preds)):
            # get current level prediction
            B, L = preds[i].size()
            target_i = F.interpolate(target.view(B, 1, -1), L, mode = 'linear').reshape(B,L)

            # frame shift loss
            loss_ol_cur = torch.mean(torch.abs(preds[i] - target_i))

            # monotonic loss
            recon = torch.cat((torch.zeros((B, 1)).cuda(), preds[i]), 1)
            recon_diff = recon[:, :-1] - recon[: ,1:]
            mono = torch.max(recon_diff, torch.zeros((B, L)).cuda())
            loss_mono_cur = torch.mean(mono)

            # assemble loss components
            loss_ol += loss_ol_cur / (2**i)
            loss_mono += loss_mono_cur / (2**i)

            if not multilevel_supervision:
                break

        return loss_ol, loss_mono
