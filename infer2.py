import data as Data
import model as Model
import core.logger as Logger
import torch
import argparse
import core.metrics as Metrics


def get_cand_err2(model, cand, data):
    avg_psnr = 0.0
    idx = 0
    for _,  val_data in enumerate(data):
        idx += 1
        model.feed_data(val_data)
        model.test(cand=cand, continous=True)

        visuals = model.get_current_visuals(need_LR=False)

        hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
        sr_img = Metrics.tensor2img(visuals['SR'][-1])
        psnr = Metrics.calculate_psnr(sr_img, hr_img)
        avg_psnr += psnr
    avg_psnr = avg_psnr / idx
    return avg_psnr

# diffusion model init
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='config/underwater.json',
                    help='JSON file for configuration')
parser.add_argument('-p', '--phase', type=str, choices=['val'], help='val(generation)', default='val')
parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
parser.add_argument('-debug', '-d', action='store_true')
parser.add_argument('-enable_wandb', action='store_true')
parser.add_argument('-log_infer', action='store_true')

# parse configs
args2 = parser.parse_args()
opt = Logger.parse(args2)
# Convert to NoneDict, which return None for missing key.
opt = Logger.dict_to_nonedict(opt)

# logging
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# dataset
for phase, dataset_opt in opt['datasets'].items():
    if phase == 'val':
        val_set = Data.create_dataset(dataset_opt, phase)
        val_loader = Data.create_dataloader(
            val_set, dataset_opt, phase)

# model
diffusion = Model.create_model(opt)

diffusion.set_new_noise_schedule(
    opt['model']['beta_schedule']['val'], schedule_phase='val')



result_dict = torch.load('log/checkpoint.pth.tar')
cands = result_dict['candidates']

for cand in cands:
    psnr = get_cand_err2(diffusion, cand, val_loader)
    print('PSNR:', psnr, '---- cand:', cand)
