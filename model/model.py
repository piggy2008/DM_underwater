import logging
from collections import OrderedDict
import torch.nn.functional as F
import torch
import torch.nn as nn
import os
import model.networks as networks
from model.utils import load_part_of_model2, load_part_of_model3, load_part_of_model4
from model.style_transfer import VGGPerceptualLoss
# from .ddpm_trans_modules.discriminator import D2
from .base_model import BaseModel
# from .ddpm_trans_modules.style_loss import LossNetwork
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
        self.loss_func = nn.MSELoss(reduction='sum').to(self.device)
        # self.loss_style = LossNetwork().to(self.device)
        # self.style_loss = VGGPerceptualLoss().to(self.device)
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
                    if k.find('style') >= 0 or k.find('control') >= 0 or k.find('blocks2') >= 0:
                        v.requires_grad = True
                        # v.data.zero_()
                        # print(k)
                        optim_params.append(v)
                        logger.info(
                            'Params [{:s}] initialized to 0 and will optimize.'.format(k))
                    else:
                        v.requires_grad = False
                        optim_params.append(v)
            else:
                optim_params = list(self.netG.parameters())
                # optim_params = list(self.netG.parameters()) + list(self.netG_air.parameters())
                # optim_params_dis = list(self.dis_air.parameters()) + list(self.dis_water.parameters())
            # print(optim_params)
            self.optG = torch.optim.Adam(
                optim_params, lr=opt['train']["optimizer"]["lr"])
            # self.optG = torch.optim.Adam(
            #     self.netG.parameters(), lr=opt['train']["optimizer"]["lr"])
            # self.optD = torch.optim.Adam(
            #     optim_params_dis, lr=opt['train']["optimizer"]["lr"])
            self.log_dict = OrderedDict()
        self.load_network()
        self.print_network()


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

    def optimize_parameters(self, flag=None):
        # need to average in multi-gpu
        if flag is None:
            # l_pix = self.loss_func(x_recon, noise)
            # b, c, h, w = self.data['SR'].shape
            # l_pix = l_pix.sum()/int(b*c*h*w)
            self.optG.zero_grad()
            l_pix = self.netG(self.data, flag=None)
            # x_0_recover = self.netG(self.data, flag=None)
            # l_pix = self.loss_func(x_0_recover, self.data['SR'])
            # print('x0 shape:', self.data['HR'].shape)
            # l_pix = l_pix / int(b)
            # print('l_pix:', l_pix)
            l_pix.backward()
            self.optG.step()
            # print('single mse:', l_pix.item())
            # set log
            self.log_dict['l_pix'] = l_pix.item()

        else:
            self.optG.zero_grad()
            x_0_recover = self.netG(self.data, flag=flag)
            # x_0_recover.clamp_(-1., 1.)
            x_0_recover = (x_0_recover + 1) / 2
            style = self.data['style']
            style = (style + 1) / 2
            x0 = self.data['SR']
            x0 = (x0 + 1) / 2

            content_score, style_score = self.loss_style(x_0_recover, x0, style)
            # print('content loss:', content_score, '----- style loss:', style_score)
            content_loss = 0.0001 * content_score
            style_loss = 1.5 * style_score
            # l_pix = self.loss_func(x_recon, noise)
            # b, c, h, w = self.data['HR'].shape
            # l_pix = l_pix.sum() / int(b * c * h * w)
            l_pix = content_loss + style_loss
            l_pix.backward()
            self.optG.step()
            # print('content and style mse:', l_pix.item())
            # set log
            self.log_dict['l_pix'] = l_pix.item()

    def optimize_parameters2(self):
        # need to average in multi-gpu
        self.optG.zero_grad()
        l_pix = self.netG(self.data)
        # need to average in multi-gpu
        b, c, h, w = self.data['HR'].shape
        l_pix = l_pix.sum() / int(b * c * h * w)
        l_pix.backward()
        self.optG.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()

    def finetune_parameters(self):
        self.optG.zero_grad()
        # l_pix = self.netG(self.data)
        # need to average in multi-gpu
        b, c, h, w = self.data['HR'].shape
        # l_pix = l_pix.sum()/int(b*c*h*w)
        if isinstance(self.netG, nn.DataParallel):
            l_pix = self.netG.module.fine_tune(self.data)
        else:
            l_pix = self.netG.fine_tune(self.data)
        # l_pix = self.netG.loss_func(self.data['HR'], self.SR)
        # l_pix = l_pix.sum() / int(b * c * h * w)
        # if len(self.SR.size()) == 3:
        #     self.SR = torch.reshape(self.SR, (b, c, h, w))
        # content_loss, style_loss = self.style_loss(self.SR, self.data['SR'], self.data['style'])
        # print('content:', content_loss, ' style:', style_loss)
        # l_pix = content_loss + 100000 * style_loss
        l_pix.backward()
        self.optG.step()
        # set log
        self.log_dict['l_pix'] = l_pix.item()


    def test(self, cand=None, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.super_resolution(
                    self.data, continous)
            else:
                # n = None
                # self.temp, miu, var = self.netG_air.super_resolution(
                #     self.data, continous, flag='style', n=n)
                self.SR = self.netG.super_resolution(
                    self.data, continous, cand=cand)

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
            # network.load_state_dict(torch.load(
            #     gen_path), strict=(not self.opt['model']['finetune_norm']))

            # gen_path2 = 'experiments_train/sr_ffhq_230617_110652/checkpoint/I1200000_E4257_gen.pth'
            # network2 = self.netG_air
            # network2.load_state_dict(torch.load(
            #     gen_path2), strict=(not self.opt['model']['finetune_norm']))

            load_part_of_model3(network, gen_path)
            if self.opt['phase'] == 'train':
                # optimizer
                opt = torch.load(opt_path)
                # self.optG.load_state_dict(opt['optimizer'])
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']
