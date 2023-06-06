import os

from SIN_src.losses.adversarial import NonSaturatingWithR1
from SIN_src.losses.feature_matching import masked_l1_loss, feature_matching_loss
from SIN_src.losses.perceptual import ResNetPL
from SIN_src.models.SIN_network import *
from SIN_src.utils import get_lr_schedule_with_warmup, torch_init_model


def make_optimizer(parameters, kind='adamw', **kwargs):
    if kind == 'adam':
        optimizer_class = torch.optim.Adam
    elif kind == 'adamw':
        optimizer_class = torch.optim.AdamW
    else:
        raise ValueError(f'Unknown optimizer kind {kind}')
    return optimizer_class(parameters, **kwargs)


def set_requires_grad(module, value):
    for param in module.parameters():
        param.requires_grad = value


def add_prefix_to_keys(dct, prefix):
    return {prefix + k: v for k, v in dct.items()}


class BaseInpaintingTrainingModule(nn.Module):
    def __init__(self, config, gpu, name, rank, *args, test=False, **kwargs):
        super().__init__(*args, **kwargs)
        print('BaseInpaintingTrainingModule init called')
        self.config = config
        self.global_rank = rank
        self.config = config
        self.iteration = 0
        self.name = name
        self.test = test
        self.gen_weights_path = os.path.join(config.OUTPUT_DIR, name + '_gen.pth')
        self.dis_weights_path = os.path.join(config.OUTPUT_DIR, name + '_dis.pth')

        self.str_encoder = PartialSketchEncoder().cuda(gpu)
        self.generator = TextureRestorationModule().cuda(gpu)
        self.best = None

        if not test:
            self.discriminator = NLayerDiscriminator(**self.config.discriminator).cuda(gpu)
            self.adversarial_loss = NonSaturatingWithR1(**self.config.losses['adversarial'])
            self.generator_average = None
            self.last_generator_averaging_step = -1

            if self.config.losses.get("l1", {"weight_known": 0})['weight_known'] > 0:
                self.loss_l1 = nn.L1Loss(reduction='none')

            if self.config.losses.get("mse", {"weight": 0})['weight'] > 0:
                self.loss_mse = nn.MSELoss(reduction='none')

            assert self.config.losses['perceptual']['weight'] == 0

            if self.config.losses.get("resnet_pl", {"weight": 0})['weight'] > 0:
                self.loss_resnet_pl = ResNetPL(**self.config.losses['resnet_pl'])
            else:
                self.loss_resnet_pl = None
            self.gen_optimizer, self.dis_optimizer = self.configure_optimizers()
            self.str_optimizer = torch.optim.Adam(self.str_encoder.parameters(), lr=config.optimizers['generator']['lr'])
        if self.config.AMP:  # use AMP
            self.scaler = torch.cuda.amp.GradScaler()

        # reset lr
        if not test:
            for group in self.gen_optimizer.param_groups:
                group['lr'] = config.optimizers['generator']['lr']
                group['initial_lr'] = config.optimizers['generator']['lr']
            for group in self.dis_optimizer.param_groups:
                group['lr'] = config.optimizers['discriminator']['lr']
                group['initial_lr'] = config.optimizers['discriminator']['lr']

        if self.config.DDP and not test:
            import apex
            self.generator = apex.parallel.convert_syncbn_model(self.generator)
            self.discriminator = apex.parallel.convert_syncbn_model(self.discriminator)
            self.generator = apex.parallel.DistributedDataParallel(self.generator)
            self.discriminator = apex.parallel.DistributedDataParallel(self.discriminator)

        if self.config.optimizers['decay_steps'] is not None and self.config.optimizers['decay_steps'] > 0 and not test:
            self.g_scheduler = torch.optim.lr_scheduler.StepLR(self.gen_optimizer, config.optimizers['decay_steps'],
                                                               gamma=config.optimizers['decay_rate'])
            self.d_scheduler = torch.optim.lr_scheduler.StepLR(self.dis_optimizer, config.optimizers['decay_steps'],
                                                               gamma=config.optimizers['decay_rate'])
            self.str_scheduler = get_lr_schedule_with_warmup(self.str_optimizer,
                                                             num_warmup_steps=config.optimizers['warmup_steps'],
                                                             milestone_step=config.optimizers['decay_steps'],
                                                             gamma=config.optimizers['decay_rate'])
        else:
            self.g_scheduler = None
            self.d_scheduler = None
            self.str_scheduler = None

    def save(self, epoch, iteration):
        save_dir = os.path.join(self.config.OUTPUT_DIR, "checkpoints")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.gen_weights_path = os.path.join(save_dir, f'epoch{epoch}_iteration{iteration}_gen.pth')
        self.dis_weights_path = os.path.join(save_dir, f'epoch{epoch}_iteration{iteration}_dis.pth')
        print('\nsaving %s...\n' % self.name)
        raw_model = self.generator.module if hasattr(self.generator, "module") else self.generator
        raw_encoder = self.str_encoder.module if hasattr(self.str_encoder, "module") else self.str_encoder
        torch.save({
            'iteration': self.iteration,
            'optimizer': self.gen_optimizer.state_dict(),
            'str_opt': self.str_optimizer.state_dict(),
            'str_encoder': raw_encoder.state_dict(),
            'generator': raw_model.state_dict()
        }, self.gen_weights_path)
        raw_model = self.discriminator.module if hasattr(self.discriminator, "module") else self.discriminator
        torch.save({
            'optimizer': self.dis_optimizer.state_dict(),
            'discriminator': raw_model.state_dict()
        }, self.dis_weights_path)

    def configure_optimizers(self):
        discriminator_params = list(self.discriminator.parameters())
        return [
            make_optimizer(self.generator.parameters(), **self.config.optimizers['generator']),
            make_optimizer(discriminator_params, **self.config.optimizers['discriminator'])
        ]

class SINInpaintingTrainingModule(BaseInpaintingTrainingModule):
    def __init__(self, *args, gpu, rank, image_to_discriminator='predicted_image', test=False, **kwargs):
        super().__init__(*args, gpu=gpu, name='InpaintingModel', rank=rank, test=test, **kwargs)
        self.image_to_discriminator = image_to_discriminator
        if self.config.AMP:  # use AMP
            self.scaler = torch.cuda.amp.GradScaler()

    def forward(self, batch):
        img = batch['image']
        mask = batch['mask']
        sketch = batch['sketch']
        masked_img = img * (1 - mask)
        batch['masked_image'] = masked_img

        masked_img = torch.cat([masked_img, mask], dim=1)

        str_feats = self.str_encoder(sketch)
        batch['predicted_image'] = self.generator(masked_img.to(torch.float32), str_feats)
        batch['inpainted'] = mask * batch['predicted_image'] + (1 - mask) * batch['image']
        batch['mask_for_losses'] = mask
        return batch

    def process(self, batch):
        # self.iteration += 1
        self.discriminator.zero_grad()
        # discriminator loss
        dis_loss, batch, dis_metric = self.discriminator_loss(batch)
        self.dis_optimizer.step()
        if self.d_scheduler is not None:
            self.d_scheduler.step()

        # generator loss
        self.generator.zero_grad()
        self.str_optimizer.zero_grad()
        # generator loss
        gen_loss, gen_metric = self.generator_loss(batch)

        if self.config.AMP:
            self.scaler.step(self.gen_optimizer)
            self.scaler.update()
            self.scaler.step(self.str_optimizer)
            self.scaler.update()
        else:
            self.gen_optimizer.step()
            self.str_optimizer.step()

        if self.str_scheduler is not None:
            self.str_scheduler.step()
        if self.g_scheduler is not None:
            self.g_scheduler.step()

        # create logs
        if self.config.AMP:
            gen_metric['loss_scale'] = self.scaler.get_scale()
        logs = [dis_metric, gen_metric]

        return batch['predicted_image'], gen_loss, dis_loss, logs, batch

    def generator_loss(self, batch):
        img = batch['image']
        predicted_img = batch[self.image_to_discriminator]
        original_mask = batch['mask']
        supervised_mask = batch['mask_for_losses']

        # L1
        l1_value = masked_l1_loss(predicted_img, img, supervised_mask,
                                  self.config.losses['l1']['weight_known'],
                                  self.config.losses['l1']['weight_missing'])

        total_loss = l1_value
        metrics = dict(gen_l1=l1_value.item())

        # discriminator
        # adversarial_loss calls backward by itself
        mask_for_discr = original_mask
        self.adversarial_loss.pre_generator_step(real_batch=img, fake_batch=predicted_img,
                                                 generator=self.generator, discriminator=self.discriminator)
        discr_fake_pred, discr_fake_features = self.discriminator(predicted_img.to(torch.float32))
        adv_gen_loss, adv_metrics = self.adversarial_loss.generator_loss(discr_fake_pred=discr_fake_pred,
                                                                         mask=mask_for_discr)
        total_loss = total_loss + adv_gen_loss
        metrics['gen_adv'] = adv_gen_loss.item()
        metrics.update(add_prefix_to_keys(adv_metrics, 'adv_'))
        # feature matching
        if self.config.losses['feature_matching']['weight'] > 0:
            discr_real_pred, discr_real_features = self.discriminator(img)
            need_mask_in_fm = self.config.losses['feature_matching'].get('pass_mask', False)
            mask_for_fm = supervised_mask if need_mask_in_fm else None
            fm_value = feature_matching_loss(discr_fake_features, discr_real_features,
                                             mask=mask_for_fm) * self.config.losses['feature_matching']['weight']
            total_loss = total_loss + fm_value
            metrics['gen_fm'] = fm_value.item()

        if self.loss_resnet_pl is not None:
            resnet_pl_value = self.loss_resnet_pl(predicted_img, img)
            total_loss = total_loss + resnet_pl_value
            metrics['gen_resnet_pl'] = resnet_pl_value.item()

        if self.config.AMP:
            self.scaler.scale(total_loss).backward()
        else:
            total_loss.backward()

        metrics['gen_total_loss'] = total_loss.item()

        return total_loss.item(), metrics

    def discriminator_loss(self, batch):
        self.adversarial_loss.pre_discriminator_step(real_batch=batch['image'], fake_batch=None,
                                                     generator=self.generator, discriminator=self.discriminator)
        discr_real_pred, discr_real_features = self.discriminator(batch['image'])
        real_loss, dis_real_loss, grad_penalty = self.adversarial_loss.discriminator_real_loss(
            real_batch=batch['image'],
            discr_real_pred=discr_real_pred)
        real_loss.backward()
        if self.config.AMP:
            with torch.cuda.amp.autocast():
                batch = self.forward(batch)
        else:
            batch = self(batch)
        batch[self.image_to_discriminator] = batch[self.image_to_discriminator].to(torch.float32)
        predicted_img = batch[self.image_to_discriminator].detach()
        discr_fake_pred, discr_fake_features = self.discriminator(predicted_img.to(torch.float32))
        fake_loss = self.adversarial_loss.discriminator_fake_loss(discr_fake_pred=discr_fake_pred, mask=batch['mask'])
        fake_loss.backward()
        total_loss = fake_loss + real_loss
        metrics = {}
        metrics['dis_real_loss'] = dis_real_loss.mean().item()
        metrics['dis_fake_loss'] = fake_loss.item()
        metrics['grad_penalty'] = grad_penalty.mean().item()

        return total_loss.item(), batch, metrics
