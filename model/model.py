import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import os
import model.networks as networks
from model.style_transfer import VGGPerceptualLoss
from .ddpm_trans_modules.discriminator import D2
from .base_model import BaseModel
logger = logging.getLogger('base')


class DDPM(BaseModel):
    def __init__(self, opt):
        super(DDPM, self).__init__(opt)
        # define network and load pretrained models
        self.netG = self.set_device(networks.define_G(opt))
        # self.netG_air = self.set_device(networks.define_G(opt))
        # self.dis_water = self.set_device(D2())
        # self.dis_air = self.set_device(D2())
        self.schedule_phase = None

        # set loss and load resume state
        self.set_loss()
        self.loss_func = nn.L1Loss(reduction='sum').to(self.device)
        self.set_new_noise_schedule(
            opt['model']['beta_schedule']['train'], schedule_phase='train')
        if self.opt['phase'] == 'train':
            self.netG.train()
            # self.netG_air.train()
            # self.dis_water.train()
            # self.dis_air.train()
            # find the parameters to optimize
            if opt['model']['finetune_norm']:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    v.requires_grad = False
                    if k.find('transformer') >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info(
                            'Params [{:s}] initialized to 0 and will optimize.'.format(k))
            else:
                optim_params = list(self.netG.parameters())
                # optim_params = list(self.netG.parameters()) + list(self.netG_air.parameters())
                # optim_params_dis = list(self.dis_air.parameters()) + list(self.dis_water.parameters())

            self.optG = torch.optim.Adam(
                optim_params, lr=opt['train']["optimizer"]["lr"])
            # self.optD = torch.optim.Adam(
            #     optim_params_dis, lr=opt['train']["optimizer"]["lr"])
            self.log_dict = OrderedDict()
        self.load_network()
        self.print_network()
        # self.style_loss = VGGPerceptualLoss().to(self.device)

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def feed_data(self, data):
        self.data = self.set_device(data)

    def optimize_parameters(self):
        self.optG.zero_grad()
        l_pix = self.netG(self.data, 'air')
        # need to average in multi-gpu
        b, c, h, w = self.data['HR'].shape
        l_pix = l_pix.sum()/int(b*c*h*w)
        l_pix.backward()
        self.optG.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()

    # def optimize_parameters(self):
    #     real_air = self.data['style']
    #     real_water = self.data['SR']
    #
    #     b, c, h, w = real_water.shape
    #     self.data['HR'] = real_water
    #     x_recon, noise, x_recover_fake_air = self.netG(self.data, 'air')
    #     self.data['HR'] = x_recover_fake_air
    #     x_recon2, noise2, reconst_water = self.netG_air(self.data, 'water')
    #
    #     self.data['HR'] = real_air
    #     x_recon3, noise3, x_recover_fake_water = self.netG_air(self.data, 'water')
    #     self.data['HR'] = x_recover_fake_water
    #     x_recon4, noise4, reconst_air = self.netG(self.data, 'air')
    #
    #     l_pix = self.loss_func(x_recon, noise)
    #     l_pix2 = self.loss_func(x_recon2, noise2)
    #     l_pix3 = self.loss_func(x_recon3, noise3)
    #     l_pix4 = self.loss_func(x_recon4, noise4)
    #     l_pix = 0.25 * (l_pix + l_pix2 + l_pix3 + l_pix4)
    #     l_pix = l_pix.sum() / int(b * c * h * w)
    #     # print(l_pix)
    #     self.set_requires_grad([self.dis_air, self.dis_water], False)
    #     self.optG.zero_grad()
    #     out = self.dis_air(x_recover_fake_air)
    #     g_loss = torch.mean((out - 1) ** 2)
    #     out2 = self.dis_water(x_recover_fake_water)
    #     g_loss2 = torch.mean((out2 - 1) ** 2)
    #     g_rec_loss = torch.mean((real_water - reconst_water) ** 2)
    #     g_rec_loss2 = torch.mean((real_air - reconst_air) ** 2)
    #     g_total = 0.25 * (g_loss + g_loss2 + 4 * g_rec_loss + 4 * g_rec_loss2) + l_pix
    #     # print(g_total)
    #     g_total.backward()
    #     self.optG.step()
    #
    #     self.set_requires_grad([self.dis_air, self.dis_water], True)
    #     self.optD.zero_grad()
    #
    #     out_real = self.dis_air(real_air)
    #     d1_loss_real = torch.mean((out_real - 1) ** 2)
    #     x_recover_fake_air2 = x_recover_fake_air.detach()
    #     out_fake = self.dis_air(x_recover_fake_air2)
    #     d1_loss_fake = torch.mean(out_fake ** 2)
    #     d1_total = (d1_loss_real + d1_loss_fake) * 0.5
    #     d1_total.backward()
    #
    #     out2_real = self.dis_water(real_water)
    #     d2_loss_real = torch.mean((out2_real - 1) ** 2)
    #     x_recover_fake_water2 = x_recover_fake_water.detach()
    #     out2_fake = self.dis_water(x_recover_fake_water2)
    #     d2_loss_fake = torch.mean(out2_fake ** 2)
    #     d2_total = (d2_loss_real + d2_loss_fake) * 0.5
    #     d2_total.backward()
    #     self.optD.step()
    #
    #
    #     # # ----------------- train D ----------------
    #     # self.optG.zero_grad()
    #     # self.optD.zero_grad()
    #     # out = self.dis_air(self.data['style'])
    #     # d1_loss = torch.mean((out - 1) ** 2)
    #     # out2 = self.dis_water(self.data['SR'])
    #     # d2_loss = torch.mean((out2 - 1) ** 2)
    #     # d_real_loss = d1_loss + d2_loss
    #     # d_real_loss.backward()
    #     # self.optD.step()
    #     #
    #     # self.optG.zero_grad()
    #     # self.optD.zero_grad()
    #     # x_recon, noise, x_recover_fake_air = self.netG(self.data, 'air')
    #     # l_pix = self.loss_func(x_recon, noise)
    #     # out = self.dis_air(x_recover_fake_air)
    #     # d1_loss = torch.mean(out ** 2)
    #     #
    #     # x_recon, noise, x_recover_fake_water = self.netG_air(self.data, 'water')
    #     # l_pix = self.loss_func(x_recon, noise)
    #     # out2 = self.dis_air(x_recover_fake_water)
    #     # d2_loss = torch.mean(out2 ** 2)
    #     # d_fake_loss = d1_loss + d2_loss
    #     # d_fake_loss.backward()
    #     # self.optD.step()
    #     #
    #     # # ----------------- train G ----------------
    #     # self.optG.zero_grad()
    #     # self.optD.zero_grad()
    #     # real_water = self.data['SR']
    #     # x_recon, noise, x_recover_fake_air = self.netG(self.data, 'air')
    #     # l_pix = self.loss_func(x_recon, noise)
    #     # out = self.dis_air(x_recover_fake_air)
    #     # self.data['SR'] = x_recover_fake_air
    #     # x_recon, noise, reconst_water = self.netG_air(self.data, 'water')
    #     # # need to average in multi-gpu
    #     # l_pix2 = self.loss_func(x_recon, noise)
    #     # g_loss = torch.mean((out - 1) ** 2)
    #     # g_loss += torch.mean((real_water - reconst_water) ** 2)
    #     # b, c, h, w = self.data['HR'].shape
    #     # l_pix = l_pix.sum()/int(b*c*h*w) + 0.1 * g_loss
    #     # l_pix.backward(retain_graph=True)
    #     # self.optG.step()
    #     #
    #     #
    #     # self.optG.zero_grad()
    #     # self.optD.zero_grad()
    #     # real_air = self.data['style']
    #     # x_recon, noise, x_recover_fake_water = self.netG_air(self.data, 'water')
    #     # l_pix3 = self.loss_func(x_recon, noise)
    #     # out = self.dis_water(x_recover_fake_water)
    #     # self.data['style'] = x_recover_fake_water
    #     # x_recon, noise, reconst_air = self.netG(self.data, 'air')
    #     # l_pix4 = self.loss_func(x_recon, noise)
    #     #
    #     # g_loss = torch.mean((out - 1) ** 2)
    #     # g_loss += torch.mean((real_air - reconst_air) ** 2)
    #     # b, c, h, w = self.data['HR'].shape
    #     # l_pix3 = l_pix3.sum() / int(b * c * h * w) + 0.1 * g_loss
    #     # l_pix3.backward()
    #     # self.optG.step()
    #
    #     # set log
    #     self.log_dict['l_pix'] = l_pix.item()
    #     self.log_dict['g_total'] = (g_total - l_pix).item()
    #     self.log_dict['d1_total'] = d1_total.item()

    def finetune_parameters(self):
        self.optG.zero_grad()
        # l_pix = self.netG(self.data)
        # need to average in multi-gpu
        b, c, h, w = self.data['HR'].shape
        # l_pix = l_pix.sum()/int(b*c*h*w)
        if isinstance(self.netG, nn.DataParallel):
            self.SR = self.netG.module.fine_tune(self.data)
        else:
            self.SR = self.netG.fine_tune(self.data)
        # l_pix = self.netG.loss_func(self.data['HR'], self.SR)
        # l_pix = l_pix.sum() / int(b * c * h * w)
        # if len(self.SR.size()) == 3:
        #     self.SR = torch.reshape(self.SR, (b, c, h, w))
        content_loss, style_loss = self.style_loss(self.SR, self.data['SR'], self.data['style'])
        # print('content:', content_loss, ' style:', style_loss)
        l_pix = content_loss + 100000 * style_loss
        l_pix.backward()
        self.optG.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()


    def test(self, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.super_resolution(
                    self.data, continous)
            else:
                self.SR = self.netG.super_resolution(
                    self.data, continous)
        self.netG.train()

    def sample(self, batch_size=1, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.sample(batch_size, continous)
            else:
                self.SR = self.netG.sample(batch_size, continous)
        self.netG.train()

    def set_loss(self):
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

        # if isinstance(self.netG_air, nn.DataParallel):
        #     self.netG_air.module.set_loss(self.device)
        # else:
        #     self.netG_air.set_loss(self.device)

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(
                    schedule_opt, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)

            # if isinstance(self.netG_air, nn.DataParallel):
            #     self.netG_air.module.set_new_noise_schedule(
            #         schedule_opt, self.device)
            # else:
            #     self.netG_air.set_new_noise_schedule(schedule_opt, self.device)

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_LR=True, sample=False):
        out_dict = OrderedDict()
        if sample:
            out_dict['SAM'] = self.SR.detach().float().cpu()
        else:
            out_dict['SR'] = self.SR.detach().float().cpu()
            out_dict['INF'] = self.data['SR'].detach().float().cpu()
            out_dict['HR'] = self.data['HR'].detach().float().cpu()
            if need_LR and 'LR' in self.data:
                out_dict['LR'] = self.data['LR'].detach().float().cpu()
            else:
                out_dict['LR'] = out_dict['INF']
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info(
            'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def save_network(self, epoch, iter_step):
        gen_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_gen.pth'.format(iter_step, epoch))
        opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_opt.pth'.format(iter_step, epoch))
        # gen
        network = self.netG
        if isinstance(self.netG, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step,
                     'scheduler': None, 'optimizer': None}
        opt_state['optimizer'] = self.optG.state_dict()
        torch.save(opt_state, opt_path)

        logger.info(
            'Saved model in [{:s}] ...'.format(gen_path))

    def load_network(self):
        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            logger.info(
                'Loading pretrained model for G [{:s}] ...'.format(load_path))
            gen_path = '{}_gen.pth'.format(load_path)
            opt_path = '{}_opt.pth'.format(load_path)
            # gen
            network = self.netG
            if isinstance(self.netG, nn.DataParallel):
                network = network.module
            network.load_state_dict(torch.load(
                gen_path), strict=(not self.opt['model']['finetune_norm']))
            # network.load_state_dict(torch.load(
            #     gen_path), strict=False)
            if self.opt['phase'] == 'train':
                # optimizer
                opt = torch.load(opt_path)
                self.optG.load_state_dict(opt['optimizer'])
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']
