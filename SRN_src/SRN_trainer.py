import sys
import cv2
import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn.functional as F

from SRN_src.dataset import TrainDataset, ValDataset
from SRN_src.SRN_network import *
from SRN_src.losses import CrossCorrelationLoss

from SRN_src.utils import *

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.autograd.set_detect_anomaly(True)

# Sketch Refinement Network (SRN)
class SRNTrainer:
    def __init__(self, configs):
    
        self.configs = configs
        if not os.path.exists(self.configs.output):
            os.makedirs(self.configs.output)

        # tensorboard writer
        tblog_dir = os.path.join(self.configs.output, 'tb_logs')
        self.writer = SummaryWriter(log_dir=tblog_dir)

        # train data preparation
        self.dataset = TrainDataset(configs)
        self.data_loader = DataLoader(
            self.dataset,
            batch_size=self.configs.batch_size,
            num_workers=self.configs.num_workers,
            shuffle=True,
            drop_last=True)
        print('')
        print('-' * 50 + 'TRAIN CONFIGURATION' + '-' * 50)
        print(f"total batches: {len(self.data_loader)}")
        print(f"edge path: {self.configs.edges_prefix}")
        print(f"batch size: {self.configs.batch_size}")
        print(f"number of workers: {self.configs.num_workers}")
        print('-' * 100)
        print('')

        # validation data preparation
        if self.configs.val_interval > 0:
            self.val_dataset = ValDataset(configs)
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=1,
                num_workers=self.configs.num_workers,)
            print('')
            print('-' * 50 + 'VALIDATION CONFIGURATION' + '-' * 50)
            print(f"total batches: {len(self.val_loader)}")
            print(f"sketch path: {self.configs.sketches_prefix_val}")
            print(f"edge path: {self.configs.edges_prefix_val}")
            print(f"batch size: {self.configs.batch_size}")
            print(f"number of workers: {self.configs.num_workers}")
            print('-' * 100)
            print('')

        # network
        # initialize registration module
        self.registration_module = RegistrationModule()

        # initialize enhancement module
        self.enhancement_module = EnhancementModule()

        self.registration_module.to(device)
        self.enhancement_module.to(device)

        # the checkpoint is saved as dict with keys of 'name', 'model' and 'parameters'
        if self.configs.train_EM:
            self.registration_module.load_state_dict(torch.load(self.configs.RM_checkpoint)['parameters'])

        print('')
        print('-' * 50 + 'NETWORK CONFIGURATION' + '-' * 50)
        print(self.registration_module)
        print(self.enhancement_module)
        print('-' * 100)
        print('')

        # loss function
        self.l1_loss = nn.L1Loss()
        self.cc_loss = CrossCorrelationLoss()

        # optimizers
        self.optimizer_rm = torch.optim.Adam(self.registration_module.parameters(), lr=self.configs.lr)
        self.optimizer_em = torch.optim.Adam(self.enhancement_module.parameters(), lr=self.configs.lr)

        # initialize progressive bar
        self.progbar = Progbar(len(self.data_loader), width=20, stateful_metrics=['epoch', 'iter'],
                              verbose=1)

    # train enhancement module
    def train_EM(self):

        self.iteration = 0
        self.registration_module.eval()
        self.enhancement_module.train()
        for epoch in range(self.configs.epochs):
            self.epoch = epoch
            print('')
            print('-' * 50 + f'STARTING EPOCH {epoch + 1}' + '-' * 50)
            for batch_index, data in enumerate(self.data_loader):
                if self.iteration > self.configs.max_iters:
                    sys.exit(0)
                self.iteration = epoch * len(self.dataset) + (batch_index + 1) * self.configs.batch_size
                
                data['image'] = data['image'].to(device)
                data['mask'] = data['mask'].to(device)
                data['sketch'] = data['sketch'].to(device)
                data['edge'] = data['edge'].to(device)

                # forward
                masked_image = data['image'] * (1 - data['mask']) + data['mask']
                data['masked_image'] = masked_image
                                    
                data['sketch'] = data['sketch'] * data['mask'] + data['edge'] * (1 - data['mask'])
                    
                data['visualize_sketch'] = data['sketch']

                # input of registration module
                x = torch.cat([masked_image, data['mask'], data['sketch']], dim=1)

                # forward
                data['rm_out'] = self.registration_module(x)
                data['rm_out'] = torch.clamp(data['rm_out'], 0.0, 1.0)
            
                data['em_in'] = data['rm_out']

                # input of enhancement module
                thresh = torch.mean(data['em_in'])
                data['em_in'] = binary_value(data['em_in'], thresh)
                data['em_in'] = data['em_in'] * data['mask'] + data['edge'] * (1 - data['mask'])
                em_x = data['em_in']

                # forward
                data['em_out'] = self.enhancement_module(em_x)
                data['em_out'] = torch.clamp(data['em_out'], 0.0, 1.0)
                data['em_out'] = data['em_out'] * data['mask'] + data['edge'] * (1 - data['mask'])

                # calculate loss value and update tensorboard logs

                # update enhancement module
                em_loss = 0.0
                em_metrics = {}

                # l1 loss
                em_l1_loss = self.l1_loss(data['em_out'], data['edge'].detach())
                em_loss += em_l1_loss * self.configs.em_l1_weight
                self.writer.add_scalar('em_l1_loss/train', em_l1_loss, self.iteration)
                em_metrics['l1_loss'] = em_l1_loss.item()

                # cc loss
                em_cc_loss = self.cc_loss.loss(data['em_out'], data['edge'].detach())
                em_loss += em_cc_loss * self.configs.em_cc_weight
                self.writer.add_scalar('em_cc_loss/train', em_cc_loss, self.iteration)
                em_metrics['cc_loss'] = em_cc_loss.item()

                self.writer.add_scalar('em_loss/train', em_loss, self.iteration)

                self.optimizer_em.zero_grad()
                em_loss.backward()
                self.optimizer_em.step()

                logs = [em_metrics]

                logs = [("epoch", epoch), ("iter", self.iteration)] + \
                        [(i, logs[0][i]) for i in logs[0]]

                self.progbar.add(len(data['image']), values=logs)

                # print log
                if self.iteration % self.configs.log_interval == 0:
                    sys.stdout.flush()

                # visualization training samples
                if self.iteration % self.configs.sample_interval == 0:
                    self.visualize(data=data, keys=['image', 'masked_image', 'visualize_sketch', 'edge', 'rm_out', 'em_out'], path=self.configs.output, epoch=epoch, iteration=self.iteration)
                    print('')
                    print('-' * 50 + f"SAVING SAMPLES OF ITERATION {self.iteration}" + '-' * 50)
                    print(f"samples saved at: {os.path.join(self.configs.output, 'samples')}")
                    print('-' * 50 + 'END OF VISUALIZAITON' + '-' * 50)
                    print('')

                # validation
                # set `self.configs.val_interval` > 0 for validation
                if self.configs.val_interval > 0 and self.iteration % self.configs.val_interval == 0:
                    self.val_with_two_stage()

                # save checkpoint
                if self.iteration % self.configs.checkpoint_interval == 0:
                    self.save_checkpoint('enhancement_module', self.enhancement_module, self.configs.output, epoch + 1, self.iteration)
                    self.save_checkpoint('optimizer_em', self.optimizer_em, self.configs.output, epoch + 1, self.iteration)

            print('-' * 50 + f'END OF EPOCH {epoch + 1}' + '-' * 50)
            epoch += 1
            print('')
                
    # train registration module
    def train_RM(self):

        self.iteration = 0
        self.registration_module.train()
        for epoch in range(self.configs.epochs):
            self.epoch = epoch
            print('')
            print('-' * 50 + f'STARTING EPOCH {epoch + 1}' + '-' * 50)
            for batch_index, data in enumerate(self.data_loader):
                
                # kill the program if iteration achieve maximum
                if self.iteration > self.configs.max_iters:
                    sys.exit(0)
                    
                self.iteration = epoch * len(self.dataset) + (batch_index + 1) * self.configs.batch_size
                
                data['image'] = data['image'].to(device)
                data['mask'] = data['mask'].to(device)
                data['sketch'] = data['sketch'].to(device)
                data['edge'] = data['edge'].to(device)

                # forward
                masked_image = data['image'] * (1 - data['mask']) + data['mask']
                data['masked_image'] = masked_image
                
                data['sketch'] = data['sketch'] * data['mask'] + data['edge'] * (1 - data['mask'])
                data['visualize_sketch'] = data['sketch']

                #  input
                x = torch.cat([masked_image, data['mask'], data['sketch']], dim=1)

                # forward
                data['out'] = self.registration_module(x)
                
                data['out'] = torch.clamp(data['out'], 0.0, 1.0)
                data['out'] = data['out'] * data['mask'] + data['edge'] * (1 - data['mask'])

                losses = 0.0
                metrics = {}

                # calculate loss value and update tensorboard logs
                # l1 loss
                total_l1_loss = self.l1_loss(data['out'], data['edge'])
                losses += total_l1_loss * self.configs.rm_l1_weight
                metrics['l1_loss'] = total_l1_loss.item()
                
                self.writer.add_scalar('total_l1_loss/train', total_l1_loss, self.iteration)

                # cc loss
                cc_loss = self.cc_loss.loss(data['out'], data['edge'])
                losses += cc_loss * self.configs.rm_cc_weight
                self.writer.add_scalar('cc_loss/train', cc_loss, self.iteration)
                metrics['cc_loss'] = cc_loss.item()

                self.writer.add_scalar('losses/train', losses, self.iteration)

                # update
                self.optimizer_rm.zero_grad()
                losses.backward()
                self.optimizer_rm.step()

                logs = [metrics]

                logs = [("epoch", epoch), ("iter", self.iteration)] + \
                        [(i, logs[0][i]) for i in logs[0]]

                # print logs
                self.progbar.add(len(data['image']), values=logs)
                
                # visualization training samples
                if self.iteration % self.configs.sample_interval == 0:
                    self.visualize(data=data, keys=['image', 'masked_image', 'visualize_sketch', 'edge', 'out'], path=self.configs.output, epoch=epoch, iteration=self.iteration)
                    print('')
                    print('-' * 50 + f"SAVING SAMPLES OF ITERATION {self.iteration}" + '-' * 50)
                    print(f"samples saved at: {os.path.join(self.configs.output, 'samples')}")
                    print('-' * 50 + 'END OF VISUALIZAITON' + '-' * 50)
                    print('')

                # validation
                # set `self.configs.val_interval` > 0 for validation
                if self.configs.val_interval > 0 and self.iteration % self.configs.val_interval == 0:
                    self.val()

                # save checkpoint
                if self.iteration % self.configs.checkpoint_interval == 0:
                    self.save_checkpoint('registration_module', self.registration_module, self.configs.output, epoch + 1, self.iteration)
                    self.save_checkpoint('optimizer_rm', self.optimizer_rm, self.configs.output, epoch + 1, self.iteration)

            print('-' * 50 + f'END OF EPOCH {epoch + 1}' + '-' * 50)
            epoch += 1
            print('')


    def val_with_two_stage(self):
        if not os.path.exists(os.path.join(self.configs.output, 'validation')):
            os.makedirs(os.path.join(self.configs.output, 'validation'))

        print('')
        print('-' * 50 + f"STARTING VALIDATION OF ITERATION {self.iteration}" + '-' * 50)
        with torch.no_grad():
            rm_loss, em_loss = 0.0, 0.0       
            for i, data in enumerate(self.val_loader):
                filename = data['filename']
                data['image'] = data['image'].to(device)
                data['sketch'] = data['sketch'].to(device)
                data['edge'] = data['edge'].to(device)
                data['mask'] = data['mask'].to(device)

                masked_image = data['image'] * (1 - data['mask']) + data['mask']
                data['masked_image'] = masked_image

                # forward
                x = torch.cat([data['masked_image'], data['mask'], data['sketch']], dim=1)
                data['rm_out'] = self.registration_module(x)

                em_x = torch.cat([data['masked_image'], data['mask'], data['rm_out']], dim=1)
                data['em_out'] = self.enhancement_module(em_x)

                rm_l1_loss = self.l1_loss(data['rm_out'], data['edge'])
                em_l1_loss = self.l1_loss(data['em_out'], data['edge'])
                rm_loss += rm_l1_loss
                em_loss += em_l1_loss
                
                rm_cc_loss = self.cc_loss.loss(data['rm_out'], data['edge'])
                em_cc_loss = self.cc_loss.loss(data['em_out'], data['edge'])
                rm_loss += rm_cc_loss
                em_loss += em_cc_loss

                self.visualize_validation(data, ['image', 'masked_image', 'sketch', 'edge', 'rm_out', 'em_out'], self.configs.output, self.epoch, self.iteration)


            # visualize & log printing
            print(f"rm_loss: {rm_loss}")
            print(f"em_loss: {em_loss}")
            print(f"validation samples saved at: {os.path.join(self.configs.output, 'samples')}")
            print('-' * 50 + 'END OF VALIDATION' + '-' * 50)
            print('')
            
    
    def val_with_rm(self):
        if not os.path.exists(os.path.join(self.configs.output, 'validation')):
            os.makedirs(os.path.join(self.configs.output, 'validation'))

        print('')
        print('-' * 50 + f"STARTING VALIDATION OF ITERATION {self.iteration}" + '-' * 50)
        with torch.no_grad():
            losses = 0.0   
            for i, data in enumerate(self.val_loader):
                filename = data['filename']
                data['image'] = data['image'].to(device)
                data['sketch'] = data['sketch'].to(device)
                data['edge'] = data['edge'].to(device)
                data['mask'] = data['mask'].to(device)

                masked_image = data['image'] * (1 - data['mask']) + data['mask']
                data['masked_image'] = masked_image

                # forward
                x = torch.cat([data['masked_image'], data['mask'], data['sketch']], dim=1)
                data['out'] = self.registration_module(x)

                valid_l1_loss = self.l1_loss(data['out'], data['edge']) * (1 - data['mask'])
                masked_l1_loss = self.l1_loss(data['out'], data['edge']) * data['mask']
                losses = losses + valid_l1_loss + masked_l1_loss
                
                cc_loss = self.cc_loss.loss(data['rm_out'], data['edge'])
                losses += cc_loss

                self.visualize_validation(data, ['image', 'masked_image', 'sketch', 'edge', 'out'], self.configs.output, self.epoch, self.iteration)


            # visualize & log printing
            print(f"valid_l1_loss: {valid_l1_loss}")
            print(f"masked_l1_loss: {masked_l1_loss}")
            
            if self.configs.use_cc:
                print(f"cc_loss: {cc_loss}")
                            
            print(f"validation samples saved at: {os.path.join(self.configs.output, 'val_samples')}")
            print('-' * 50 + 'END OF VALIDATION' + '-' * 50)
            print('')


    def visualize(self, data, keys, path, epoch, iteration):
        
        # create sample path if not exists
        if not os.path.exists(os.path.join(path, 'samples')):
            os.makedirs(os.path.join(path, 'samples'))

        data_list = []

        for key in keys:
            item = data[key]
            # [B, C=1, H, W] -> [H, W, C=1]
            if item.size(1) == 1:
                item = torch.cat([item, item, item], dim=1)
            item = item[0, :, :, :,].permute(1, 2, 0)
            item = (item * 255.).cpu().detach().numpy().astype(np.uint8)
            data_list.append(item)

        # concate on `width` dimension
        sample = np.concatenate(data_list, axis=1)

        cv2.imwrite(os.path.join(path, 'samples') + f"/sample_epoch{epoch}_iters{iteration}.png", sample)


    def visualize_validation(self, data, keys, path, epoch, iteration):
        
        validation_out_dir = os.path.join(path, 'val_samples')
        filename = data['filename'][0]

        # create sample path if not exists
        if not os.path.exists(validation_out_dir):
            os.makedirs(validation_out_dir)

        if not os.path.exists(os.path.join(validation_out_dir, f'epoch{epoch}_iters{iteration}')):
            os.makedirs(os.path.join(validation_out_dir, f'epoch{epoch}_iters{iteration}'))

        save_path = os.path.join(validation_out_dir, f'epoch{epoch}_iters{iteration}')

        data_list = []

        for key in keys:
            item = data[key]
            # [B, C=1, H, W] -> [H, W, C=1]
            if item.size(1) == 1:
                item = torch.cat([item, item, item], dim=1)
            item = item[0, :, :, :,].permute(1, 2, 0)
            item = (item * 255.).cpu().detach().numpy().astype(np.uint8)
            data_list.append(item)

        # concate on `width` dimension
        sample = np.concatenate(data_list, axis=1)

        cv2.imwrite(save_path + f"/{filename}", sample)


    def save_checkpoint(self, name, model, path, epoch, iteration):

        print('')
        print('-' * 50 + f"START SAVING CHECKPOINTS OF ITERATION {self.iteration}" + '-' * 50)

        # create checkpoint path if not exists
        if not os.path.exists(os.path.join(path, 'checkpoints')):
            os.makedirs(os.path.join(path, 'checkpoints'))

        if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()

        state = {
            'iteration': iteration,
            'model': name,
            'parameters': model_state,
        }

        save_path = os.path.join(path, 'checkpoints') + f"/checkpoint_epoch{epoch}_iters{iteration}_{name}.pth"
        torch.save(state, save_path)

        # log printing
        print(f"model: {name}")
        print(f"iteration: {iteration}")
        print(f"checkpoints saved at: {os.path.join(self.configs.output, 'samples')}")
        print('-' * 50 + 'END OF SAVING CHECKPOINTS' + '-' * 50)
        print('')
       
