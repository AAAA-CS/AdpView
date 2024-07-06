import os
import random
import dotmap
import numpy as np
from dotmap import DotMap
from collections import OrderedDict
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision

from src.datasets import datasets
from src.models import resnet_small, resnet
from src.models.transfer import LogisticRegression
from src.objectives.memory_bank import MemoryBank
from src.objectives.adversarial import AdversarialSimCLRLoss, AdversarialNCELoss
from src.objectives.infonce import NoiseConstrastiveEstimation
from src.objectives.simclr import SimCLRObjective
from src.utils import utils

from src.models.composite_attack import CompositeAttack
from src.models import viewmaker

import torch_dct as dct
import pytorch_lightning as pl
import wandb
from torch.autograd import Variable


def create_dataloader(dataset, config, batch_size, shuffle=True, drop_last=True):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        drop_last=drop_last,
        num_workers=config.data_loader_workers,
    )
    return loader


class PretrainAttackSystem(pl.LightningModule):
    '''Pytorch Lightning System for self-supervised pretraining
    with adversarially generated views.
    '''

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config.optim_params.batch_size
        self.loss_name = self.config.loss_params.objective
        self.t = self.config.loss_params.t

        self.train_dataset, self.val_dataset = datasets.get_image_datasets(
            config.data_params.dataset,
            config.data_params.default_augmentations or 'none',
        )
        # Used for computing knn validation accuracy
        if config.data_params.dataset != 'potsdam' and config.data_params.dataset != 'loveDa' and config.data_params.dataset != 'vaihingen':
            train_labels = self.train_dataset.dataset.targets
            self.train_ordered_labels = np.array(train_labels)

        self.model = self.create_encoder()

        # Used for computing knn validation accuracy.
        self.memory_bank = MemoryBank(
            len(self.train_dataset),
            self.config.model_params.out_dim,
        )

        start_num = 1
        iter_num = 1
        inner_iter_num = 7
        self.attack = CompositeAttack(self.model, enabled_attack=(0, 1, 2, 3, 4), mode='train', dataset='imagenet',
                                       start_num=start_num, iter_num=iter_num, inner_iter_num=inner_iter_num,
                                       multiple_rand_start=True, order_schedule="fixed")
        print("攻击种类：", 0, 1, 2, 3, 4)

    def view(self, imgs):
        if 'Expert' in self.config.system:
            raise RuntimeError('Cannot call self.view() with Expert system')
        views = self.viewmaker(imgs)
        views = self.normalize(views)
        return views

    def create_encoder(self):
        '''Create the encoder model.'''
        if self.config.model_params.resnet_small:
            # ResNet variant for smaller inputs (e.g. CIFAR-10).
            encoder_model = resnet_small.ResNet18(self.config.model_params.out_dim)
        else:
            resnet_class = getattr(
                torchvision.models,
                self.config.model_params.resnet_version,
            )
            encoder_model = resnet_class(
                pretrained=False,
                num_classes=self.config.model_params.out_dim,
            )
        if self.config.model_params.projection_head:
            mlp_dim = encoder_model.fc.weight.size(1)
            encoder_model.fc = nn.Sequential(
                nn.Linear(mlp_dim, mlp_dim),
                nn.ReLU(),
                encoder_model.fc,
            )
        return encoder_model


    def noise(self, batch_size, device):
        shape = (batch_size, self.config.model_params.noise_dim)
        # Center noise at 0 then project to unit sphere.
        noise = utils.l2_normalize(torch.rand(shape, device=device) - 0.5)
        return noise

    def get_repr(self, img):
        '''Get the representation for a given image.'''
        if 'Expert' not in self.config.system:
            # The Expert system datasets are normalized already.
            img = self.normalize(img)
        return self.model(img)

    def normalize(self, imgs):
        # These numbers were computed using compute_image_dset_stats.py
        if 'cifar' in self.config.data_params.dataset:
            mean = torch.tensor([0.491, 0.482, 0.446], device=imgs.device)
            std = torch.tensor([0.247, 0.243, 0.261], device=imgs.device)
        elif 'potsdam' in self.config.data_params.dataset:
            mean = torch.tensor([0.485, 0.456, 0.406], device=imgs.device)
            std = torch.tensor([0.229, 0.224, 0.225], device=imgs.device)
        elif 'loveDa' in self.config.data_params.dataset:
            mean = torch.tensor([0.485, 0.456, 0.406], device=imgs.device)
            std = torch.tensor([0.229, 0.224, 0.225], device=imgs.device)
        elif 'vaihingen' in self.config.data_params.dataset:
            mean = torch.tensor([0.485, 0.456, 0.406], device=imgs.device)
            std = torch.tensor([0.229, 0.224, 0.225], device=imgs.device)
        else:
            raise ValueError(f'Dataset normalizer for {self.config.data_params.dataset} not implemented')
        imgs = (imgs - mean[None, :, None, None]) / std[None, :, None, None]
        return imgs

    def forward(self, batch, train=True):
        indices, img, img2, neg_img, _, = batch
        if self.loss_name == 'AdversarialNCELoss':
            view1 = self.view(img)
            view1_embs = self.model(view1)
            emb_dict = {
                'indices': indices,
                'view1_embs': view1_embs,
            }
        elif self.loss_name == 'AdversarialSimCLRLoss':
            if self.config.model_params.double_viewmaker:
                view1, view2 = self.view(img)
            else:
                # view1 = self.attack(img)
                # view2 = self.attack(img2)
                view1 = self.normalize(img)
                view2 = self.normalize(img2)
                # view1 = self.view(img)
                # view2 = self.view(img2)
            emb_dict = {
                'indices': indices,
                'view1_embs': self.model(view1),
                'view2_embs': self.model(view2),
            }
        else:
            raise ValueError(f'Unimplemented loss_name {self.loss_name}.')

        # if self.global_step % 200 == 0:
        #     # Log some example views.
        #     views_to_log = view1.permute(0,2,3,1).detach().cpu().numpy()[:10]
        # wandb.log({"examples": [wandb.Image(view, caption=f"Epoch: {self.current_epoch}, Step {self.global_step}, Train {train}") for view in views_to_log]})

        return emb_dict

    def get_losses_for_batch(self, emb_dict, train=True):
        if self.loss_name == 'AdversarialSimCLRLoss':
            view_maker_loss_weight = self.config.loss_params.view_maker_loss_weight
            loss_function = AdversarialSimCLRLoss(
                embs1=emb_dict['view1_embs'],
                embs2=emb_dict['view2_embs'],
                t=self.t,
                view_maker_loss_weight=view_maker_loss_weight
            )
            encoder_loss = loss_function.get_loss()
            img_embs = emb_dict['view1_embs']
        elif self.loss_name == 'AdversarialNCELoss':
            view_maker_loss_weight = self.config.loss_params.view_maker_loss_weight
            loss_function = AdversarialNCELoss(
                emb_dict['indices'],
                emb_dict['view1_embs'],
                self.memory_bank,
                k=self.config.loss_params.k,
                t=self.t,
                m=self.config.loss_params.m,
                view_maker_loss_weight=view_maker_loss_weight
            )
            encoder_loss, _ = loss_function.get_loss()
            img_embs = emb_dict['view1_embs']
        else:
            raise Exception(f'Objective {self.loss_name} is not supported.')

            # Update memory bank.
        if train:
            with torch.no_grad():
                if self.loss_name == 'AdversarialNCELoss':
                    new_data_memory = loss_function.updated_new_data_memory()
                    self.memory_bank.update(emb_dict['indices'], new_data_memory)
                else:
                    new_data_memory = utils.l2_normalize(img_embs, dim=1)
                    self.memory_bank.update(emb_dict['indices'], new_data_memory)

        return encoder_loss

    def training_step(self, batch, batch_idx):
        #生成攻击样本
        self.model.eval()
        # print(batch[1])     #[0, 1]
        pos_1 = batch[1].detach().clone()
        pos_2 = batch[2].detach().clone()
        # data_adv_1 = batch[1].detach() + 0.001 * torch.randn(batch[1].shape).cuda(0, non_blocking=True).detach()
        # data_adv_2 = batch[2].detach() + 0.001 * torch.randn(batch[2].shape).cuda(0, non_blocking=True).detach()
        # 噪声大小[0, 1]
        data_adv_1 = batch[1].detach() + 0.1 * torch.randn(batch[1].shape).cuda(0, non_blocking=True).detach()
        data_adv_2 = batch[2].detach() + 0.1 * torch.randn(batch[2].shape).cuda(0, non_blocking=True).detach()
        data_adv_1 = self.attack(data_adv_1, pos_2, True)
        data_adv_2 = self.attack(data_adv_2, pos_1, False)
        data_adv_1 = Variable(torch.clamp(data_adv_1, 0.0, 1.0), requires_grad=False)   #限制扰动大小
        data_adv_2 = Variable(torch.clamp(data_adv_2, 0.0, 1.0), requires_grad=False)

        self.model.train()
        batch[1] = data_adv_1
        batch[2] = data_adv_2
        emb_dict = self.forward(batch)
        # emb_dict['optimizer_idx'] = torch.tensor(optimizer_idx, device=self.device)
        return emb_dict

    def training_step_end(self, emb_dict):
        encoder_loss = self.get_losses_for_batch(emb_dict, train=True)
        # print("endocer_loss", encoder_loss)
        # Handle Tensor (dp) and int (ddp) cases
        # if emb_dict['optimizer_idx'].__class__ == int or emb_dict['optimizer_idx'].dim() == 0:
        #     optimizer_idx = emb_dict['optimizer_idx']
        # else:
        #     optimizer_idx = emb_dict['optimizer_idx'][0]
        # if optimizer_idx == 0:
        metrics = {
            'encoder_loss': encoder_loss, 'temperature': self.t
        }
        return {'loss': encoder_loss, 'log': metrics}
        # else:
        #     metrics = {
        #         'view_maker_loss': view_maker_loss,
        #     }
        #     return {'loss': view_maker_loss, 'log': metrics}

    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_idx,
                       second_order_closure=None, on_tpu=False, using_native_amp=False, using_lbfgs=False):
        if not self.config.optim_params.viewmaker_freeze_epoch:
            super().optimizer_step(current_epoch, batch_nb, optimizer, optimizer_idx)
            return

        if optimizer_idx == 0:
            optimizer.step()
            optimizer.zero_grad()
        elif current_epoch < self.config.optim_params.viewmaker_freeze_epoch:
            # Optionally freeze the viewmaker at a certain pretraining epoch.
            optimizer.step()
            optimizer.zero_grad()

    def configure_optimizers(self):
        # Optimize temperature with encoder.
        if type(self.t) == float or type(self.t) == int:
            encoder_params = self.model.parameters()
        else:
            encoder_params = list(self.model.parameters()) + [self.t]
        encoder_optim = torch.optim.SGD(
            encoder_params,
            lr=self.config.optim_params.learning_rate,
            momentum=self.config.optim_params.momentum,
            weight_decay=self.config.optim_params.weight_decay,
        )

        # return [encoder_optim, view_optim], []
        # return [encoder_optim], []
        return encoder_optim

    def train_dataloader(self):
        return create_dataloader(self.train_dataset, self.config, self.batch_size)
