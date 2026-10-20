import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

from option import opt
from model.net import Net, Net_combine
from data_utils import TrainsetFromFolder, ValsetFromFolder
from eval import *
from torch.optim.lr_scheduler import MultiStepLR

logs_name = '/1_25625682'
# train_dir = './dataset/harvard4x_input32/train'
# val_dir = './dataset/harvard4x_input32/test'
train_dir = r'E:/dataset/pavia4x_input32/train'
val_dir = r'E:/dataset/pavia4x_input32/val'
dataset_name = '\pavia'

def main():
    if opt.show:
        if not os.path.exists(opt.log_dir + logs_name):
            os.makedirs(opt.log_dir + logs_name)

        global writer
        writer = SummaryWriter(log_dir=opt.log_dir + logs_name)

    if opt.cuda:
        print("=> Use GPU ID: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)
    cudnn.benchmark = True

    # Loading datasets
    train_set = TrainsetFromFolder(train_dir)
    train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
    val_set = ValsetFromFolder(val_dir)
    val_loader = DataLoader(dataset=val_set, num_workers=opt.threads, batch_size=1, shuffle=False)

    # Buliding model
    # model = net(num_bands= opt.num_bands, in_channels= 128, out_channels= 128, num_blocks = 8, upscale_factor = opt.upscale_factor)
    # model = Net(channel = 256, num = 8, scale = 4)
    model = Net_combine(channel1= 256, channel2= 256, num1= 8, num2= 2, scale= 4)
    criterion = nn.L1Loss()

    if opt.cuda:
        model = nn.DataParallel(model).cuda()
        criterion = criterion.cuda()
    else:
        model = model.cpu()
    print('# parameters:', sum(param.numel() for param in model.parameters()))

    # Setting Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-08)

    # optionally resuming from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

            # Setting learning rate
    scheduler = MultiStepLR(optimizer, milestones=[35, 70, 105, 140, 175], gamma=0.5, last_epoch=-1)

    # Training
    # for epoch in tqdm(range(opt.start_epoch, opt.nEpochs + 1), desc="Training Epochs"):
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        scheduler.step()
        print("Epoch = {}, lr = {}".format(epoch, optimizer.param_groups[0]["lr"]))
        train(train_loader, optimizer, model, criterion, epoch)
        val(val_loader, model, epoch)
        save_checkpoint(model, epoch, optimizer)


# def train(train_loader, optimizer, model, criterion, epoch):
#     model.train()
#
#     for iteration, batch in enumerate(train_loader, 1):
#         input1, label = batch[0], batch[1]
#         if opt.cuda:
#             input1 = input1.cuda()
#             label = label.cuda()
#
#         SR = model(input1)
#         loss = criterion(SR, label)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, len(train_loader), loss.item()))
#
#         if opt.show:
#             writer.add_scalar('Train/Loss', loss.item())

def train(train_loader, optimizer, model, criterion, epoch):
    model.train()
    with tqdm(total=len(train_loader), desc=f"Epoch {epoch}", unit="batch") as pbar:

        for iteration, batch in enumerate(train_loader, 1):
            input1, label = batch[0], batch[1]
            if opt.cuda:
                input1 = input1.cuda()
                label = label.cuda()

            SR = model(input1)
            loss = criterion(SR, label)

            # 检查损失是否包含NaN
            assert torch.isnan(loss).sum() == 0, "Loss contains NaN!"

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, len(train_loader), loss.item()))

            if opt.show:
                writer.add_scalar('Train/Loss', loss.item())

def val(val_loader, model, epoch):
    model.eval()
    val_psnr = 0
    val_ssim = 0
    val_sam = 0

    with torch.no_grad():
        for iteration, batch in enumerate(val_loader, 1):
            input1, label = batch[0], batch[1]
            if opt.cuda:
                input1 = input1.cuda()

            SR = model(input1)
            SR = SR.squeeze(0)
            val_psnr += PSNR(SR.cpu().numpy(), label.cpu().data[0].numpy())
            val_ssim += SSIM(SR.cpu().numpy(), label.cpu().data[0].numpy())
            val_sam += SAM(SR.cpu().numpy(), label.cpu().data[0].numpy())

    val_psnr = val_psnr / len(val_loader)
    val_ssim = val_ssim / len(val_loader)
    val_sam = val_sam / len(val_loader)
    print("PSNR = {:.4f}, SSIM = {:.4f}, SAM = {:.4f}".format(val_psnr, val_ssim, val_sam))
    if opt.show:
        writer.add_scalar('Val/PSNR', val_psnr, epoch)
        writer.add_scalar('Val/SSIM', val_ssim, epoch)
        writer.add_scalar('Val/SAM', val_sam, epoch)


def save_checkpoint(model, epoch, optimizer):
    if not os.path.exists(r'checkpoint/' + opt.method + '/' + dataset_name + logs_name):
        os.makedirs(r'checkpoint/' + opt.method + '/' + dataset_name + logs_name)
    model_out_path = r'checkpoint/' + opt.method + '/' + dataset_name + logs_name + '/' + "model_{}_epoch_{}.pth".format(opt.upscale_factor, epoch)
    state = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()}
    torch.save(state, model_out_path)


if __name__ == "__main__":
    main()
