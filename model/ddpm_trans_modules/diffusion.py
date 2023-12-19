import math
import torch
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
from torch.distributions import kl
import core.metrics as Metrics
from model.style_transfer import VGGPerceptualLoss
from model.utils import load_part_of_model2
import random
from model.clip_loss2 import FrozenCLIPEmbedder
from model.clip_loss2 import generate_src_target_txt
# from model.utils import generate_src_target_txt


def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


# def extract(a, t, x_shape):
#     b, *_ = t.shape
#     out = a.gather(-1, t)
#     return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def extract(a, t, x_shape):
    """Extract coefficients from a based on t and reshape to make it
    broadcastable with x_shape."""
    bs, = t.shape
    assert x_shape[0] == bs
    out = torch.gather(torch.tensor(a, dtype=torch.float, device=t.device), 0, t.long())
    assert out.shape == (bs,)
    out = out.reshape((bs,) + (1,) * (len(x_shape) - 1))
    return out


def noise_like(shape, device, repeat=False):
    def repeat_noise(): return torch.randn(
        (1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))

    def noise(): return torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        image_size,
        channels=3,
        loss_type='l1',
        conditional=True,
        schedule_opt=None,
        teacher_model=None,
        teacher_param=None
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.conditional = conditional
        self.loss_type = loss_type
        # print(teacher_param)
        if teacher_param is not None:
            self.teacher = load_part_of_model2(teacher_model, teacher_param)
        else:
            self.teacher = None
        if schedule_opt is not None:
            pass
            # self.set_new_noise_schedule(schedule_opt)
        self.eta = 0
        self.sample_proc = 'ddim'
    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss().to(device)
            self.loss_func2 = nn.MSELoss().to(device)
            # self.style_loss = VGGPerceptualLoss().to(device)
            self.clip_embedding = FrozenCLIPEmbedder().to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss().to(device)
        else:
            raise NotImplementedError()

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        ddim_sigma = (self.eta * ((1 - alphas_cumprod_prev) / (1 - alphas_cumprod) * (1 - alphas_cumprod / alphas_cumprod_prev)) ** 0.5)
        self.ddim_sigma = to_torch(ddim_sigma)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # self.register_buffer('ddim_sigma',
        #                      to_torch(ddim_sigma))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(
            self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None, style=None):
        if condition_x is not None:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(torch.cat([condition_x, x], dim=1), t))
        else:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(x, t))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    def p_mean_variance_ddim(self, x, t, clip_denoised: bool, condition_x=None, style=None):
        if condition_x is not None:
            # x_recon = self.predict_start_from_noise(
            #     x, t=t, noise=self.denoise_fn(torch.cat([condition_x, x], dim=1), t, style))
            x_recon = self.denoise_fn(torch.cat([condition_x, x], dim=1), t, style)
        else:
            # x_recon = self.predict_start_from_noise(
            #     x, t=t, noise=self.denoise_fn(x, t))
            x_recon = self.denoise_fn(x, t)
        # if clip_denoised:
        #     x_recon.clamp_(-1., 1.)

        alpha = extract(self.alphas_cumprod, t, x_recon.shape)
        alpha_prev = extract(self.alphas_cumprod_prev, t, x_recon.shape)
        sigma = extract(self.ddim_sigma, t, x_recon.shape)
        sqrt_one_minus_alphas = extract(self.sqrt_one_minus_alphas_cumprod, t, x_recon.shape)
        pred_x0 = (x - sqrt_one_minus_alphas * x_recon) / (alpha ** 0.5)

        dir_xt = torch.sqrt(1. - alpha_prev - sigma ** 2) * x_recon
        noise = torch.randn(x.shape, device=x.device)
        x_prev = (alpha_prev ** 0.5) * pred_x0 + dir_xt + sigma * noise
        return x_prev, pred_x0


    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False, condition_x=None, style=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x, style=style)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b,
                                                      *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample2(self, x, t, clip_denoised=True, repeat_noise=False, condition_x=None, style=None):
        bt = extract(self.betas, t, x.shape)
        at = extract((1.0 - self.betas).cumprod(dim=0), t, x.shape)
        logvar = extract(
            self.posterior_log_variance_clipped, t, x.shape)
        weight = bt / torch.sqrt(1 - at)
        et = self.denoise_fn(torch.cat([condition_x, x], dim=1), t, style)
        # if clip_denoised:
        #     et.clamp_(-1., 1.)
        mean = 1 / torch.sqrt(1.0 - bt) * (x - weight * et)
        noise = torch.randn_like(x)
        mask = 1 - (t == 0).float()
        mask = mask.reshape((x.shape[0],) + (1,) * (len(x.shape) - 1))
        xt_next = mean + mask * torch.exp(0.5 * logvar) * noise
        xt_next = xt_next.float()
        return xt_next

    @torch.no_grad()
    def p_sample_ddim(self, x, t, clip_denoised=True, repeat_noise=False, condition_x=None, style=None):
        b, *_, device = *x.shape, x.device
        x_prev, pred_x0 = self.p_mean_variance_ddim(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x, style=style)
        # noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0

        return x_prev

    def p_sample_ddim2(self, x, t, t_next, clip_denoised=True, repeat_noise=False, condition_x=None, style=None, context=None):
        b, *_, device = *x.shape, x.device
        bt = extract(self.betas, t, x.shape)
        at = extract((1.0 - self.betas).cumprod(dim=0), t, x.shape)
        # print('betas=', self.betas)
        # print('bt=', bt)

        ######## use mask to extract foreground #######
        # style = torch.mean(style, dim=1, keepdim=True)
        # style = (style + 1) / 2
        if condition_x is not None:
            et = self.denoise_fn(torch.cat([condition_x, x], dim=1), t,
                                 style, context)
            # et = self.denoise_fn(torch.cat([condition_x, x], dim=1), t, context)
        else:
            et = self.denoise_fn(x, t)


        x0_t = (x - et * (1 - at).sqrt()) / at.sqrt()
        # x0_air_t = (x_air - et_air * (1 - at).sqrt()) / at.sqrt()
        if t_next == None:
            at_next = torch.ones_like(at)
        else:
            at_next = extract((1.0 - self.betas).cumprod(dim=0), t_next, x.shape)
        if self.eta == 0:
            xt_next = at_next.sqrt() * x0_t + (1 - at_next).sqrt() * et
            # xt_air_next = at_next.sqrt() * x0_air_t + (1 - at_next).sqrt() * et_air
        elif at > (at_next):
            print('Inversion process is only possible with eta = 0')
            raise ValueError
        else:
            c1 = self.eta * ((1 - at / (at_next)) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c2 * et + c1 * torch.randn_like(x0_t)
            # xt_air_next = at_next.sqrt() * x0_air_t + c2 * et_air + c1 * torch.randn_like(x0_t)

        # noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0

        return xt_next

    def p_sample_ddim_withx0(self, x, x_air, t, t_next, clip_denoised=True, repeat_noise=False, condition_x=None, style=None):

        if condition_x is not None:
            x0_t = self.denoise_fn(torch.cat([condition_x, x], dim=1), t)
        else:
            # et, _, _, _, _, _ = self.denoise_fn(x, x_air, t)
            x0_t, x0_t_air, _, _, _, _, _, _ = self.denoise_fn(x, x_air, t)

        # x0_t = (x - et * (1 - at).sqrt()) / at.sqrt()
        if t_next == None:
            return x0_t, x0_t_air
        else:
            xt_next = self.q_sample(x_start=x0_t, t=t_next)
            xt_next_air = self.q_sample(x_start=x0_t_air, t=t_next)
            return xt_next, xt_next_air

        # noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0



    def p_sample_ddim_style(self, x, t, t_next, clip_denoised=True, repeat_noise=False, condition_x=None, style=False, miu=None, var=None):
        b, *_, device = *x.shape, x.device
        bt = extract(self.betas, t, x.shape)
        at = extract((1.0 - self.betas).cumprod(dim=0), t, x.shape)

        if condition_x is not None:
            et = self.denoise_fn(torch.cat([condition_x, x], dim=1), t)
        else:
            if style:
                et, miu, var = self.denoise_fn(x, t)
            else:
                et = self.denoise_fn(x, t, miu, var)

        x0_t = (x - et * (1 - at).sqrt()) / at.sqrt()
        if t_next == None:
            at_next = torch.ones_like(at)
        else:
            at_next = extract((1.0 - self.betas).cumprod(dim=0), t_next, x.shape)
        if self.eta == 0:
            xt_next = at_next.sqrt() * x0_t + (1 - at_next).sqrt() * et
        elif at > (at_next):
            print('Inversion process is only possible with eta = 0')
            raise ValueError
        else:
            c1 = self.eta * ((1 - at / (at_next)) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c2 * et + c1 * torch.randn_like(x0_t)

        # noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        if style:
            # print('extract style----')
            # print(xt_next[0, 0, 0, 1])
            return xt_next, miu, var
        else:
            # print('insert style----')
            # print(xt_next[0, 0, 0, 1])
            return xt_next

    @torch.no_grad()
    def p_sample_loop(self, x_in, continous=False, cand=None):
        device = self.betas.device
        sample_inter = 10
        g_gpu = torch.Generator(device=device).manual_seed(44444)
        if not self.conditional:
            x = x_in['SR']
            shape = x.shape
            b = shape[0]
            img = torch.randn(shape, device=device, generator=g_gpu)
            # img = torch.randn(shape, device=device)
            img0 = img
            ret_img = img
            if cand is not None:
                time_steps = np.array(cand)
            else:
                num_timesteps_ddim = np.array([0, 245, 521, 1052, 1143, 1286, 1475, 1587, 1765, 1859])  # searching
                time_steps = np.flip(num_timesteps_ddim)
            for j, i in enumerate(tqdm(time_steps, desc='sampling loop time step', total=len(time_steps))):
                # print('i = ', i)
                t = torch.full((b,), i, device=device, dtype=torch.long)
                if j == len(time_steps) - 1:
                    t_next = None
                else:
                    t_next = torch.full((b,), time_steps[j + 1], device=device, dtype=torch.long)
                style = self.q_sample(x_in['style'], t, img0)
                img = self.p_sample_ddim2(img, t, t_next, style=style)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
            return img
        else:
            x = x_in['SR']
            labels = x_in['label']
            ret_img = x
            condition_x = torch.mean(x, dim=1, keepdim=True)
            shape = x.shape
            b = shape[0]
            img = torch.randn(shape, device=device)
            src_txt_inputs, target_txt_inputs = generate_src_target_txt(labels)
            context = self.clip_embedding(target_txt_inputs)

            if self.sample_proc == 'ddpm':
                for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                    # print('i = ', i)
                    img = self.p_sample(img, torch.full(
                        (b,), i, device=device, dtype=torch.long), condition_x=condition_x)
                    if i % sample_inter == 0:
                        ret_img = torch.cat([ret_img, img], dim=0)
            else:
                # step = 100
                c = 100
                num_timesteps_ddim = np.asarray(list(range(0, self.num_timesteps, c)))
                if cand is not None:
                    time_steps = np.array(cand)
                    # print(time_steps)
                else:
                    # time_steps = np.array([460, 636, 854, 1174, 1270, 1586, 1709, 1853, 1924, 1958])  # searching
                    # time_steps = np.array([1898, 1844, 1554, 1379, 1325, 1000, 904, 764, 479, 340]) # best candicate
                    # time_steps = np.array([1898, 1640, 1539, 1491, 1370, 1136, 972, 858, 680, 340])
                    # time_steps = np.asarray(list(range(0, 1000, int(1000/4))) + list(range(1000, 2000, int(1000/6))))
                    time_steps = np.flip(num_timesteps_ddim[:-1])
                for j, i in enumerate(time_steps):
                    # print('i = ', time_steps)
                    t = torch.full((b,), i, device=device, dtype=torch.long)
                    if j == len(time_steps) - 1:
                        t_next = None
                    else:
                        t_next = torch.full((b,), time_steps[j + 1], device=device, dtype=torch.long)
                    img = self.p_sample_ddim2(img, t, t_next, condition_x=condition_x, style=x_in['style'], context=context)
                    if i % sample_inter == 0:
                        ret_img = torch.cat([ret_img, img], dim=0)
        if continous:
            return ret_img
        else:
            return ret_img[-1]

    @torch.no_grad()
    def p_sample_loop2(self, x_in, continous=False):
        device = self.betas.device
        sample_inter = 10


        x_air = x_in['HR']
        x = x_in['SR']
        condition_x = torch.mean(x, dim=1, keepdim=True)
        shape = x.shape
        b = shape[0]
        img = torch.randn(shape, device=device)
        start = 800
        t = torch.full((b,), start, device=device, dtype=torch.long)

        noise = torch.randn(shape, device=device)
        # img = self.q_sample(x_start=x, t=t, noise=noise)
        img_air = self.q_sample(x_start=x_air, t=t, noise=noise)

        ret_img = img
        # num_timesteps_ddim = np.array([0, 245, 521, 1052, 1143, 1286, 1475, 1587, 1765, 1859])  # searching
        c = 100
        num_timesteps_ddim = np.asarray(list(range(0, start+1, c)))
        time_steps = np.flip(num_timesteps_ddim)
        for j, i in enumerate(tqdm(time_steps, desc='sampling loop time step', total=len(time_steps))):
            # print('i = ', i)
            t = torch.full((b,), i, device=device, dtype=torch.long)
            if j == len(time_steps) - 1:
                t_next = None
            else:
                t_next = torch.full((b,), time_steps[j + 1], device=device, dtype=torch.long)
            img = self.p_sample_ddim2(img, img_air, t, t_next, condition_x=condition_x)
            # print('i=', i)
            if i % sample_inter == 0:
                ret_img = torch.cat([ret_img, img], dim=0)

        if continous:
            return ret_img
        else:
            return ret_img[-1]

    @torch.no_grad()
    def p_sample_loop2_withx0(self, x_in, continous=False):
        device = self.betas.device
        sample_inter = 10

        if not self.conditional:
            x_air = x_in['HR']
            x = x_in['SR']
            shape = x.shape
            b = shape[0]
            # img = torch.randn(shape, device=device)
            start = 200
            t = torch.full((b,), start, device=device, dtype=torch.long)

            noise = torch.randn(shape, device=device)
            img = self.q_sample(x_start=x, t=t, noise=noise)
            img_air = self.q_sample(x_start=x_air, t=t, noise=noise)

            ret_img = img
            # num_timesteps_ddim = np.array([0, 245, 521, 1052, 1143, 1286, 1475, 1587, 1765, 1859])  # searching
            c = 20
            num_timesteps_ddim = np.asarray(list(range(0, start + 1, c)))
            time_steps = np.flip(num_timesteps_ddim)
            for j, i in enumerate(tqdm(time_steps, desc='sampling loop time step', total=len(time_steps))):
                # print('i = ', i)
                t = torch.full((b,), i, device=device, dtype=torch.long)
                if j == len(time_steps) - 1:
                    t_next = None
                else:
                    t_next = torch.full((b,), time_steps[j + 1], device=device, dtype=torch.long)
                # img_air = torch.cat([img, img_air], dim=1)
                img, img_air = self.p_sample_ddim_withx0(img, img_air, t, t_next)
                # print('i=', i)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        else:
            print('abc')
        if continous:
            return ret_img
        else:
            return ret_img[-1]

    @torch.no_grad()
    def p_sample_loop2_style(self, x_in, continous=False, flag='style', miu=None, var=None, n=None):
        device = self.betas.device
        sample_inter = 10
        miu_dict = {}
        var_dict = {}
        if flag == 'style':
            x = x_in['HR']
        else:
            x = x_in['SR']
        shape = x.shape
        b = shape[0]
        # img = torch.randn(shape, device=device)
        start = 200
        t = torch.full((b,), start, device=device, dtype=torch.long)
        if n is not None:
            noise = n
        else:
            noise = torch.randn(shape, device=device)
        img = self.q_sample(x_start=x, t=t, noise=noise)

        ret_img = img
        # num_timesteps_ddim = np.array([0, 245, 521, 1052, 1143, 1286, 1475, 1587, 1765, 1859])  # searching
        c = 20
        num_timesteps_ddim = np.asarray(list(range(0, start, c)))
        time_steps = np.flip(num_timesteps_ddim)
        for j, i in enumerate(tqdm(time_steps, desc='sampling loop time step', total=len(time_steps))):
            # print('i = ', i)
            t = torch.full((b,), i, device=device, dtype=torch.long)
            if j == len(time_steps) - 1:
                t_next = None
            else:
                t_next = torch.full((b,), time_steps[j + 1], device=device, dtype=torch.long)
            if flag == 'style':
                img, miu_, var_ = self.p_sample_ddim2(img, t, t_next, style=True)
                miu_dict[j] = miu_
                var_dict[j] = var_
            else:
                img = self.p_sample_ddim2(img, t, t_next, style=False, miu=miu[j], var=var[j])
            # print('i=', i)
            if i % sample_inter == 0:
                ret_img = torch.cat([ret_img, img], dim=0)

        if continous:
            return ret_img
        else:
            if flag == 'style':
                # print('extract style----')
                # print(ret_img[-1][0])
                return ret_img[-1], miu_dict, var_dict
            else:
                # print('insert style----')
                # print(ret_img[-1][0])
                return ret_img[-1]

    @torch.no_grad()
    def sample(self, batch_size=1, continous=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size), continous)

    # @torch.no_grad()
    # def super_resolution(self, x_in, continous=False, flag='style', miu=None, var=None, n=None):
    #
    #     if flag is None:
    #         return self.p_sample_loop2_style(x_in, continous, flag=None, miu=miu, var=var, n=n)
    #     else:
    #         img, miu, var = self.p_sample_loop2_style(x_in, continous=False, flag=flag, n=n)
    #         # return self.p_sample_loop2_style(x_in, continous, flag='None', miu=miu, var=var)
    #         return img, miu, var

    @torch.no_grad()
    def super_resolution(self, x_in, continous=False, cand=None):
        return self.p_sample_loop(x_in, continous, cand=cand)

    @torch.no_grad()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full(
                (b,), i, device=device, dtype=torch.long))

        return img

    def fine_tune(self, x_in):
        return self.p_sample_loop_finetune(x_in)

    def p_sample_loop_finetune(self, x_in):
        device = self.betas.device
        x = x_in['SR']
        start = x_in['HR']
        label = x_in['label']
        style = x_in['style']
        # print(label)
        condition_x = torch.mean(x, dim=1, keepdim=True)
        shape = x.shape
        b = shape[0]
        # print('b:', b)
        img = torch.randn(shape, device=device)

        # step = 100
        c = 200
        num_timesteps_ddim = np.asarray(list(range(0, self.num_timesteps, c)))

        time_steps = np.flip(num_timesteps_ddim)
        for j, i in enumerate(time_steps):
            # print('i = ', time_steps)
            t = torch.full((b,), i, device=device, dtype=torch.long)
            if j == len(time_steps) - 1:
                t_next = None
            else:
                t_next = torch.full((b,), time_steps[j + 1], device=device, dtype=torch.long)
            img = self.p_sample_ddim2(img, t, t_next, condition_x=condition_x, style=x_in['style'])
            # generated_images.append(img)
            # if i % sample_inter == 0:
        # print(torch.unique(img))
        # img.clamp_(-1., 1.)
        # print(torch.unique(img))
        loss_content = self.loss_func(img, start)
        # src_txt = 'underwater scenes'
        # trg_txt = 'underwater scenes'
        loss_clip = self.clip_loss(start, img, label)
        # loss_style = self.style_loss(img, style)
        # loss_clip = -torch.log(loss_clip)
        # print('l2 loss:', loss_content)
        # print('clip loss:', loss_clip)
        # print('style loss', loss_style)
        return loss_content + 2.5 * loss_clip

        # choose = random.randint(0, len(time_steps) - 1)
        # t = torch.full((b,), time_steps[choose], device=device, dtype=torch.long)
        # img = self.q_sample(start, t, noise)
        # if choose == len(time_steps) - 1:
        #     t_next = None
        #     style_label = x_in['style']
        # else:
        #     t_next = torch.full((b,), time_steps[choose + 1], device=device, dtype=torch.long)
        #     style_label = self.q_sample(x_in['style'], t_next, noise)
        # img = self.p_sample_ddim2(img, t, t_next, condition_x=condition_x, style=x_in['style'])
        # # generated_images.append(img)
        # # if i % sample_inter == 0:
        # loss_style = self.style_loss(img, style_label)
        # if choose == len(time_steps) - 1:
        #     loss_content = self.loss_func(torch.mean(img, dim=1, keepdim=True), torch.mean(start, dim=1, keepdim=True))
        #     loss = loss_style + loss_content
        #     # print('loss_style:', loss_style)
        #     # print('loss_content:', loss_content)
        # else:
        #     loss = loss_style
        #     # print('loss_style:', loss_style)
        # return loss

    def p_sample_finetune(self, x, t, clip_denoised=True, repeat_noise=False, condition_x=None, style=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x, style=style)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b,
                                                      *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def q_sample_recover(self, x_noisy, t, predict_noise=None):
        # noise = default(noise, lambda: torch.randn_like(x_start))
        return (x_noisy - extract(self.sqrt_one_minus_alphas_cumprod,
                    t, x_noisy.shape) * predict_noise) / extract(self.sqrt_alphas_cumprod, t, x_noisy.shape)


    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # fix gama
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod,
                    t, x_start.shape) * noise
        )
        # random gama
        # x_shape = x_start.shape
        # l = self.alphas_cumprod .gather(-1, t)
        # r = self.alphas_cumprod .gather(-1, t+1)
        # gama = (r - l) * torch.rand(0, 1) + l
        # gama = gama.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))
        # return (
        #     nq.sqrt(gama) * x_start + nq.sqrt(1-gama)* noise
        # )

    def p_losses(self, x_in, noise=None):
        x_start = x_in['HR']
        x = x_in['SR']
        x_style = x_in['style']
        labels = x_in['label']
        condition_x = torch.mean(x, dim=1, keepdim=True)
        ##### use mask to extract foreground #########
        # x_style = torch.mean(x_style, dim=1, keepdim=True)
        # x_mask = (x_style + 1) / 2
        # condition_x   = x_in['SR']
        [b, c, h, w] = x_start.shape
        t = torch.randint(0, self.num_timesteps, (b,),
                          device=x_start.device).long()

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy_style = self.q_sample(x_start=x_style, t=t, noise=noise)
        src_txt_inputs, target_txt_inputs = generate_src_target_txt(labels)
        context = self.clip_embedding.encode(target_txt_inputs)
        if self.teacher is not None:
            x_recon_teacher = self.teacher(
                torch.cat([condition_x, x_noisy], dim=1), t)
        # x_noisy_air = self.q_sample(x_start=x_start_air, t=t, noise=noise)

        # x_noisy_air = torch.cat([x_noisy, x_noisy_air], dim=1)

        if not self.conditional:
            x_recon = self.denoise_fn(x_noisy, t)
        else:
            x_recon = self.denoise_fn(
                torch.cat([condition_x, x_noisy], dim=1), t,
                x_noisy_style, context)

        # x_0_recover = self.q_sample_recover(x_noisy, t, predict_noise=x_recon)
        # x_0_recover_style = self.q_sample_recover(x_noisy_style, t, predict_noise=x_recon_teacher)
        # x_0_recover_style.detach()
        # x_0_recover_support = self.q_sample_recover(support, t, predict_noise=x_recon)
        # x_0_recover_support2 = self.q_sample_recover(support2, t, predict_noise=x_recon)
        # x_0_recover = F.normalize(x_0_recover, p=2, dim=2)
        # x_0_recover_air = self.q_sample_recover(x_noisy_air, t, predict_noise=x_recon_air)
        # x_0_recover2 = x_0_recover.detach()
        # x_recover = Metrics.tensor2img(x_0_recover2)
        # Metrics.save_img(x_recover, 'experiments/x0_recover.png')
        #
        # x_0_recover3 = x_in['style'].detach()
        # x_recover2 = Metrics.tensor2img(x_0_recover3)
        # Metrics.save_img(x_recover2, 'experiments/x0.png')
        loss = self.loss_func(noise, x_recon)
        # loss2 = self.loss_func(x_start, x0_recon)
        # loss_gram = self.loss_func2(gram_water, gram_air)
        # loss_air = self.loss_func(noise, x_recon_air)
        # kl_div_u = torch.mean(kl.kl_divergence(water_u, air_u))
        # kl_div_s = torch.mean(kl.kl_divergence(water_s, air_s))
        # loss_x0 = self.style_loss(x_0_recover, x_style)

        # src_txt = 'underwater background'
        # trg_txt = 'normal background'
        # loss_clip = (2 - self.clip_loss(x, src_txt, x_0_recover, trg_txt)) / 2
        # loss_clip = -torch.log(loss_clip)
        # print('loss_clip:', loss_clip)
        # return x_recon, noise, x_0_recover
        # print('loss:', loss)
        # print('loss_air:', loss_air)
        # print('loss_x0', loss_x0)
        # print('loss', loss)
        # print('kl_div_u', kl_div_u)
        # print('kl_div_s', kl_div_s)
        # print('loss_gram', loss_gram)
        return loss

    def p_losses2(self, x_in, noise=None):
        x_start = x_in['style']
        [b, c, h, w] = x_start.shape
        t = torch.randint(0, self.num_timesteps, (b,),
                          device=x_start.device).long()

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        if not self.conditional:
            x_recon = self.denoise_fn(x_noisy, t)
        else:
            x_recon = self.denoise_fn(
                torch.cat([x_in['HR'], x_noisy], dim=1), t, x_in['style'])

        x_0_recover = self.q_sample_recover(x_noisy, t, predict_noise=x_recon)
        # x_0_recover2 = x_0_recover.detach()
        # x_recover = Metrics.tensor2img(x_0_recover2)
        # Metrics.save_img(x_recover, 'experiments/x0_recover.png')
        #
        # x_0_recover3 = x_in['style'].detach()
        # x_recover2 = Metrics.tensor2img(x_0_recover3)
        # Metrics.save_img(x_recover2, 'experiments/x0.png')
        # loss = self.loss_func(noise, x_recon)

        return x_recon, noise, x_0_recover
    def p_losses_style(self, x_in, noise=None, flag=None):
        x_start = x_in['SR']
        [b, c, h, w] = x_start.shape
        t = torch.randint(0, self.num_timesteps, (b,),
                          device=x_start.device).long()

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        if flag is not None:
            x_recon = self.denoise_fn(
                torch.cat([x_in['SR'], x_noisy], dim=1), t, x_in['style'])
        else:
            x_recon = self.denoise_fn(
                torch.cat([x_in['SR'], x_noisy], dim=1), t)

        x_0_recover = self.q_sample_recover(x_noisy, t, predict_noise=x_recon)
        return x_recon, noise, x_0_recover

    # def forward(self, x, flag, *args, **kwargs):
    #     if flag == 'air':
    #         return self.p_losses(x, *args, **kwargs)
    #     else:
    #         return self.p_losses2(x, *args, **kwargs)
    def forward(self, x, flag, *args, **kwargs):

        return self.p_losses(x, *args, **kwargs)
    # def forward(self, x, continous=False, flag=None):
    #     return self.p_sample_loop_finetune(x, continous=False, flag=flag)
    # def forward(self, x, flag, *args, **kwargs):
    #     return self.p_losses_style(x, flag=flag)
