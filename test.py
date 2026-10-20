import os
import numpy as np
import torch
import torch.nn as nn
from option import opt
from eval import PSNR, SSIM, SAM
from torch.utils.data import DataLoader
from data_utils import TrainsetFromFolder, ValsetFromFolder
import scipy.io as scio
from model.net import Net, Net_combine

def main():
    out_path = '.\\out_img\\'

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    PSNRs = []
    SSIMs = []
    SAMs = []

    test_set = ValsetFromFolder(r'E:\dataset\cave4x_input32\test')
    test_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)

    if opt.cuda:
        print("=> use gpu id: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    model = Net_combine(channel1= 256, channel2= 256, num1= 2, num2= 8, scale= 4)

    if opt.cuda:
        model = nn.DataParallel(model).cuda()

    checkpoint  = torch.load(opt.model_name)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    val_psnr = 0
    val_ssim = 0
    val_sam = 0
    with torch.no_grad():
        for iteration, batch in enumerate(test_loader, 1):
            lr, label = batch[0], batch[1]
            if opt.cuda:
                lr = lr.cuda()
                label = label.cuda()

            SR = model(lr)
            SR = SR.squeeze(0)

            val_psnr += PSNR(SR.cpu().numpy(), label.cpu().data[0].numpy())
            val_ssim += SSIM(SR.cpu().numpy(), label.cpu().data[0].numpy())
            val_sam += SAM(SR.cpu().numpy(), label.cpu().data[0].numpy())

            HR = np.squeeze(label.cpu().data[0].numpy())
            SR = SR.cpu().numpy()
            SR, HR = np.transpose(SR, (1, 2, 0)), np.transpose(HR, (1, 2, 0))
            # # print("PSNR = {:.4f}, SSIM = {:.4f}, SAM = {:.4f}".format(psnr, ssim, sam))
            # if not os.path.exists(out_path + 'epoch'  + str(num)):
            #     os.makedirs(out_path + 'epoch'  + str(num))
            scio.savemat(out_path + str(iteration) + '.mat', {'SR': SR, 'label': HR})

    val_psnr = val_psnr / len(test_loader)
    val_ssim = val_ssim / len(test_loader)
    val_sam = val_sam / len(test_loader)

    print("=====averPSNR:{:.4f}=====averSSIM:{:.4f}=====averSAM:{:.4f}".format(val_psnr, val_ssim, val_sam))


if __name__ == "__main__":
    main()
