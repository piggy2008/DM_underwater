import argparse
import core.metrics as Metrics
from PIL import Image
import numpy as np
import glob
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str,
                        default='experiments_val/sr_ffhq_230804_150250/results')
    args = parser.parse_args()
    real_names = list(glob.glob('{}/*_hr.png'.format(args.path)))
    fake_names = list(glob.glob('{}/*_sr.png'.format(args.path)))

    real_names.sort()
    fake_names.sort()

    avg_psnr = 0.0
    avg_ssim = 0.0
    idx = 0
    for rname, fname in zip(real_names, fake_names):
        idx += 1
        ridx = rname.rsplit("_hr")[0]
        fidx = rname.rsplit("_sr")[0]
        # assert ridx == fidx, 'Image ridx:{ridx}!=fidx:{fidx}'.format(
        #     ridx, fidx)
        img = Image.open(rname).convert('RGB')
        # img = img.resize((256, 256))
        hr_img = np.array(img)

        img2 = Image.open(fname).convert('RGB')
        # img2 = img2.resize((256, 256))
        sr_img = np.array(img2)
        psnr = Metrics.calculate_psnr(sr_img, hr_img)
        ssim = Metrics.calculate_ssim(sr_img, hr_img)
        avg_psnr += psnr
        avg_ssim += ssim
        if idx % 20 == 0:
            print('Image:{}, PSNR:{:.4f}, SSIM:{:.4f}'.format(idx, psnr, ssim))

    avg_psnr = avg_psnr / idx
    avg_ssim = avg_ssim / idx

    # log
    print('# Validation # PSNR: {:.4e}'.format(avg_psnr))
    print('# Validation # SSIM: {:.4e}'.format(avg_ssim))
