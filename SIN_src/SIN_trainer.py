import time

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from SIN_src.dataset import *
from SIN_src.models.SIN_model import *
from SIN_src.utils import Progbar, create_dir, stitch_images, SampleEdgeLineLogits

# Sketch-modulated Inpainting Network (SIN)
class SINTrainer:
    def __init__(self, config, gpu, rank, test=False):
        create_dir(config.OUTPUT_DIR)
        create_dir(os.path.join(config.OUTPUT_DIR, 'tb_logs'))

        self.writer = SummaryWriter(os.path.join(config.OUTPUT_DIR, 'tb_logs'))
        self.config = config
        self.device = gpu
        self.global_rank = rank

        self.model_name = 'inpaint'

        kwargs = dict(config.training_model)
        kwargs.pop('kind')

        self.inpaint_model = SINInpaintingTrainingModule(config, gpu=gpu, rank=rank, test=test, **kwargs).to(gpu)

        if config.round is None:
            round = 1
        else:
            round = config.round

        if not test:
            self.train_dataset = TrainDataset(config.TRAIN_FLIST, sketch_path=config.SKETCH_PATH,
                                                batch_size=config.BATCH_SIZE // config.world_size,
                                                world_size=config.world_size, round=round, input_size=config.size if config.size else None)
            if config.EVAL_INTERVAL:
                self.val_dataset = ValidationDataset(config.VAL_FLIST, sketch_path=config.VAL_SKETCH_PATH, mask_path=config.VAL_MASK_PATH,
                                                        batch_size=config.BATCH_SIZE // config.world_size,
                                                        world_size=config.world_size, round=round, input_size=config.size if config.size else None)
        if config.DDP:
            self.train_sampler = DistributedSampler(self.train_dataset, num_replicas=config.world_size,
                                                    rank=self.global_rank, shuffle=True)
        else:
            self.train_sampler = DistributedSampler(self.train_dataset, num_replicas=1, rank=0, shuffle=True)

        self.samples_path = os.path.join(config.OUTPUT_DIR, 'samples')
        create_dir(self.samples_path)

        self.log_file = os.path.join(config.PATH, 'log_' + self.model_name + '.dat')

        self.best = float("inf") if self.inpaint_model.best is None else self.inpaint_model.best

    def save(self, epoch, iteration):
        if self.global_rank == 0:
            self.inpaint_model.save(epoch, iteration)

    def train(self):
        epoch = 0

        if self.config.DDP:
            train_loader = DataLoader(self.train_dataset, shuffle=True, pin_memory=True,
                                      batch_size=self.config.BATCH_SIZE // self.config.world_size,
                                      num_workers=8, sampler=self.train_sampler)
        else:
            train_loader = DataLoader(self.train_dataset, pin_memory=True,
                                      batch_size=self.config.BATCH_SIZE, num_workers=8,
                                      sampler=self.train_sampler)
        keep_training = True
        max_iteration = int(float((self.config.MAX_ITERS)))
        if self.config.max_epochs:
            max_epochs = self.config.max_epochs
        else:
            max_epochs = max_iteration // len(train_loader)
        total = len(self.train_dataset) // self.config.world_size

        if total == 0 and self.global_rank == 0:
            print('No training data was provided! Check \'TRAIN_FLIST\' value in the configuration file.')
            return

        while keep_training:

            if self.config.DDP or self.config.DP:
                self.train_sampler.set_epoch(epoch + 1)
            if self.config.fix_256 is None or self.config.fix_256 is False:
                self.train_dataset.reset_dataset(self.train_sampler)

            epoch_start = time.time()
            if self.global_rank == 0:
                print('\n\nTraining epoch: %d' % epoch)

            progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter', 'loss_scale',
                                                                 'g_lr', 'd_lr', 'str_lr', 'img_size'],
                              verbose=1 if self.global_rank == 0 else 0)

            for epoch in range(max_epochs):
                for batch_idx, items in enumerate(train_loader):
                    # iteration = self.inpaint_model.iteration
                    iteration = (batch_idx + 1) * self.config.BATCH_SIZE + epoch * len(train_loader) * self.config.BATCH_SIZE
                    self.iteration = iteration

                    self.inpaint_model.train()
                    for k in items:
                        if type(items[k]) is torch.Tensor:
                            items[k] = items[k].to(self.device)

                    # train
                    outputs, gen_loss, dis_loss, logs, batch = self.inpaint_model.process(items)

                    # write tensorboard logs
                    dis_losses, gen_losses = logs
                    self.writer.add_scalar("gen_l1", gen_losses['gen_l1'], iteration)
                    self.writer.add_scalar("gen_adv", gen_losses['gen_adv'], iteration)
                    self.writer.add_scalar("gen_fm", gen_losses['gen_fm'], iteration)
                    self.writer.add_scalar("gen_resnet_pl", gen_losses['gen_resnet_pl'], iteration)
                    self.writer.add_scalar("gen_total_loss", gen_losses['gen_total_loss'], iteration)

                    self.writer.add_scalar("dis_real_loss", dis_losses['dis_real_loss'], iteration)
                    self.writer.add_scalar("dis_fake_loss", dis_losses['dis_fake_loss'], iteration)
                    self.writer.add_scalar("grad_penalty", dis_losses['grad_penalty'], iteration)

                    if iteration >= max_iteration:
                        keep_training = False
                        break
                    logs = [("epoch", epoch), ("iter", iteration)] + \
                        [(i, logs[0][i]) for i in logs[0]] + [(i, logs[1][i]) for i in logs[1]]
                    logs.append(("g_lr", self.inpaint_model.g_scheduler.get_lr()[0]))
                    logs.append(("d_lr", self.inpaint_model.d_scheduler.get_lr()[0]))
                    logs.append(("str_lr", self.inpaint_model.str_scheduler.get_lr()[0]))
                    progbar.add(len(items['image']),
                                values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('l_')])

                    # validation
                    if self.config.EVAL_INTERVAL and iteration > 0 and iteration % self.config.EVAL_INTERVAL == 0 and self.global_rank == 0:
                        psnr, ssim = self.eval()
                        self.writer.add_scalar("psnr/val", psnr, iteration)
                        self.writer.add_scalar("ssim/val", ssim, iteration)

                    # sample model at checkpoints
                    if self.config.SAMPLE_INTERVAL and iteration > 0 and iteration % self.config.SAMPLE_INTERVAL == 0 and self.global_rank == 0:
                        self.visualize(batch, keys=['image', 'masked_image', 'sketch', 'inpainted'], path=self.config.OUTPUT_DIR, epoch=epoch, iteration=iteration)

                    # save model at checkpoints
                    if self.config.SAVE_INTERVAL and iteration > 0 and iteration % self.config.SAVE_INTERVAL == 0 and self.global_rank == 0:
                        self.save(epoch, iteration)
                if self.global_rank == 0:
                    print("Epoch: %d, time for one epoch: %d seconds" % (epoch, time.time() - epoch_start))
                    logs = [('Epoch', epoch), ('time', time.time() - epoch_start)]

                # psnr, ssim = self.eval()
                # self.writer.add_scalar("psnr/val", psnr, iteration)
                # self.writer.add_scalar("ssim/val", ssim, iteration)

                self.visualize(batch, keys=['image', 'masked_image', 'sketch', 'inpainted'], path=self.config.OUTPUT_DIR, epoch=epoch, iteration=0)

                self.save(epoch, iteration=0)

        print('\nEnd training....')

    def eval(self):
        create_dir(os.path.join(self.config.OUTPUT_DIR, "validation"))
        if self.config.DDP:
            val_loader = DataLoader(self.val_dataset, shuffle=False, pin_memory=True,
                                    batch_size=self.config.BATCH_SIZE // self.config.world_size,  ## BS of each GPU
                                    num_workers=8)
        else:
            val_loader = DataLoader(self.val_dataset, shuffle=False, pin_memory=True,
                                    batch_size=self.config.BATCH_SIZE, num_workers=8)

        total = len(self.val_dataset)

        self.inpaint_model.eval()

        if self.config.No_Bar:
            pass
        else:
            progbar = Progbar(total, width=20, stateful_metrics=['it'])
        iteration = 0
        with torch.no_grad():
            for items in tqdm(val_loader):
                iteration += 1
                items['image'] = items['image'].to(self.device)
                items['mask'] = items['mask'].to(self.device)
                items['sketch'] = items['sketch'].to(self.device)
                b, _, _, _ = items['image'].size()

                # inpaint model
                # eval
                items = self.inpaint_model(items)
                outputs_merged = (items['predicted_image'] * items['mask']) + (items['image'] * (1 - items['mask']))
                # save
                outputs_merged *= 255.0
                outputs_merged = outputs_merged.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
                for img_num in range(b):
                    img_rgb = outputs_merged[img_num, :, :, ::-1]
                    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(os.path.join(self.config.OUTPUT_DIR, "validation") + '/' + items['name'][img_num], img_rgb)

        # our_metric = get_inpainting_metrics(os.path.join(self.config.OUTPUT_DIR, "validation"), self.config.VAL_FLIST, None, fid_test=False)

        if self.global_rank == 0:
            print("iter: %d, PSNR: %f, SSIM: %f" % (self.iteration, float(our_metric['psnr']), float(our_metric['ssim'])))

        self.inpaint_model.train()
        return float(our_metric['psnr']), float(our_metric['ssim'])

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

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()

