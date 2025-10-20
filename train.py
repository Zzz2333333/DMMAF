import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
import time
import random
from dataset2 import HyperDatasetValid, HyperDatasetTrain1  # Clean Data set
from DMMFA import DMMFA
# from architecture import AWAN
from utils import AverageMeter, initialize_logger, save_checkpoint, record_loss2, Loss_RMSE, Loss_PSNR, Loss_train3, \
    LossTrainCSS2, Loss_valid, Loss_ssim_hyper
import cv2
import math
import torch.multiprocessing as mp

# utils

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 19 1e-4 pretrain 1e-5
parser = argparse.ArgumentParser(description="SSR")
parser.add_argument("--batchSize", type=int, default=20, help="batch size")
parser.add_argument("--end_epoch", type=int, default=100 + 1, help="number of epochs")
parser.add_argument("--init_lr", type=float, default=1e-4, help="initial learning rate")
parser.add_argument("--decay_power", type=float, default=1.5, help="decay power")
parser.add_argument("--trade_off", type=float, default=0, help="trade_off")
parser.add_argument("--max_iter", type=float, default=94500, help="max_iter")
parser.add_argument("--outf", type=str, default="results/MFormer", help='path log files')
parser.add_argument("--lambda1", type=float, default=1.0, help="smooth loss weight")
parser.add_argument("--lambda2", type=float, default=0.5, help="curvature loss weight")
parser.add_argument("--lambda3", type=float, default=0.1, help="attention loss weight")
opt = parser.parse_args()


# 结构感知平滑损失函数
class StructureAwareSmoothLoss(nn.Module):
    def __init__(self, lambda1=1.0, lambda2=1.0, lambda3=1.0):
        super(StructureAwareSmoothLoss, self).__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

    def forward(self, output, attention_maps=None):
        """
        output: 模型输出 [B, C, H, W]
        attention_maps: 注意力图 [B, 1, H, W] 或 [B, H, W]
        """
        total_loss = 0

        # 平滑损失
        if self.lambda1 > 0:
            smooth_loss = self.smoothness_loss(output)
            total_loss += self.lambda1 * smooth_loss

        # 曲率损失
        if self.lambda2 > 0:
            curvature_loss = self.curvature_loss(output)
            total_loss += self.lambda2 * curvature_loss

        # 注意力监督损失
        if self.lambda3 > 0 and attention_maps is not None:
            attention_loss = self.attention_supervision_loss(attention_maps)
            total_loss += self.lambda3 * attention_loss

        return total_loss

    def smoothness_loss(self, x):
        """
        平滑损失 - 公式(28)
        L_smooth = 1/N * Σ(|I_i,j - I_i+1,j| + |I_i,j - I_i,j+1|)
        """
        batch_size, channels, height, width = x.shape

        # 计算水平方向梯度差异
        diff_x = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])

        # 计算垂直方向梯度差异
        diff_y = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])

        # 计算平均损失
        loss = torch.mean(diff_x) + torch.mean(diff_y)

        return loss

    def curvature_loss(self, x):
        """
        曲率损失 - 公式(29)
        L_curvature = 1/N * Σ(|I_i+1,j + I_i-1,j + I_i,j+1 + I_i,j-1 - 4*I_i,j|)
        """
        batch_size, channels, height, width = x.shape

        # 使用拉普拉斯算子计算二阶导数
        # 创建拉普拉斯核
        laplacian_kernel = torch.tensor([[0, 1, 0],
                                         [1, -4, 1],
                                         [0, 1, 0]], dtype=torch.float32, device=x.device)

        laplacian_kernel = laplacian_kernel.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)

        # 对每个通道应用拉普拉斯卷积
        curvature = F.conv2d(x, laplacian_kernel, padding=1, groups=channels)

        # 计算绝对值的平均值
        loss = torch.mean(torch.abs(curvature))

        return loss

    def attention_supervision_loss(self, attention_maps):
        """
        注意力监督损失 - 公式(30)
        L_attention = 1/N * Σ(A_i,j - M_i,j)^2
        这里使用自监督方式，M_i,j 通过注意力图自身生成
        """
        if attention_maps.dim() == 3:
            attention_maps = attention_maps.unsqueeze(1)

        batch_size, _, height, width = attention_maps.shape

        # 生成自监督信号 - 使用高斯模糊后的注意力图作为目标
        with torch.no_grad():
            # 应用高斯模糊生成平滑的监督信号
            kernel_size = 3
            padding = kernel_size // 2
            target_maps = F.avg_pool2d(attention_maps, kernel_size=kernel_size, stride=1, padding=padding)

        # 计算均方误差
        loss = F.mse_loss(attention_maps, target_maps)

        return loss


def main():
    cudnn.benchmark = True
    # load dataset
    print("\nloading dataset ...")
    train_data1 = HyperDatasetTrain1(mode='train')
    print("Train set samples: ", len(train_data1))
    val_data = HyperDatasetValid(mode='valid')
    print("Validation set samples: ", len(val_data))
    # Data Loader (Input Pipeline)
    train_loader1 = DataLoader(dataset=train_data1, batch_size=opt.batchSize, shuffle=True, num_workers=10,
                               pin_memory=True, drop_last=True)
    train_loader = [train_loader1]
    val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    # model
    print("\nbuilding models_baseline ...")
    model = MFormer(3, 31, 48, 1)
    print('Parameters number is ', sum(param.numel() for param in model.parameters()))
    criterion_train = LossTrainCSS2()
    criterion_train_2 = Loss_ssim_hyper()
    criterion_train_3 = Loss_train3()
    criterion_valid_mrae = Loss_valid()  # mrae
    criterion_valid_psnr = Loss_PSNR()
    criterion_valid_rmse = Loss_RMSE()

    # 新增结构感知损失
    structure_loss = StructureAwareSmoothLoss(
        lambda1=opt.lambda1,
        lambda2=opt.lambda2,
        lambda3=opt.lambda3
    )

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)  # batchsize integer times

    if torch.cuda.is_available():
        model.cuda()
        criterion_train.cuda()
        criterion_train_2.cuda()
        criterion_train_3.cuda()
        criterion_valid_mrae.cuda()
        criterion_valid_psnr.cuda()
        criterion_valid_rmse.cuda()
        structure_loss.cuda()

    # Parameters, Loss and Optimizer
    start_epoch = 0
    iteration = 0
    record_val_loss = 1000
    # 原来是1e-8
    optimizer = optim.Adam(model.parameters(), lr=opt.init_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    # visualzation
    if not os.path.exists(opt.outf):
        os.makedirs(opt.outf)
    loss_csv = open(os.path.join(opt.outf, 'loss.csv'), 'a+')
    log_dir = os.path.join(opt.outf, 'train.log')
    logger = initialize_logger(log_dir)

    # Resume
    resume_file = ''
    if resume_file:
        if os.path.isfile(resume_file):
            print("=> loading checkpoint '{}'".format(resume_file))
            checkpoint = torch.load(resume_file)
            start_epoch = checkpoint['epoch']
            iteration = checkpoint['iter']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

    pretrain_file = ''
    if pretrain_file:
        if os.path.isfile(pretrain_file):
            print("=> loading checkpoint '{}'".format(pretrain_file))
            checkpoint = torch.load(pretrain_file)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

    for epoch in range(start_epoch + 1, opt.end_epoch):
        start_time = time.time()
        train_loss, losses_hyper, losses_mask, losses_all, losses_structure, iteration, lr = train(
            train_loader, model, criterion_train, criterion_train_2, criterion_train_3,
            structure_loss, optimizer, epoch, iteration, opt.init_lr, opt.decay_power, opt.trade_off
        )
        val_loss, val_rmse, val_psnr = validate(val_loader, model, criterion_valid_mrae, criterion_valid_rmse,
                                                criterion_valid_psnr)
        # Save model
        if torch.abs(val_loss - record_val_loss) < 0.0001 or val_loss < record_val_loss or epoch == (opt.end_epoch - 1):
            save_checkpoint(opt.outf, epoch, iteration, model, optimizer)
            if val_loss < record_val_loss:
                record_val_loss = val_loss
        # print loss
        end_time = time.time()
        epoch_time = end_time - start_time
        print(
            "Epoch [%02d], Iter[%06d], Time:%.9f, learning rate : %.9f, Train Loss: %.9f Hyper Loss: %.9f  Mask Loss: %.9f All Loss: %.9f Structure Loss: %.9f Test RMAE: %.9f Test RMSE: %.9f Test PSNR: %.9f"
            % (epoch, iteration, epoch_time, lr, train_loss, losses_hyper, losses_mask, losses_all, losses_structure,
               val_loss, val_rmse, val_psnr))
        # save loss
        record_loss2(loss_csv, epoch, iteration, epoch_time, lr, train_loss, losses_hyper, losses_mask, losses_all,
                     val_loss, val_rmse, val_psnr, losses_structure)
        logger.info(
            "Epoch [%02d], Iter[%06d], Time:%.9f, learning rate : %.9f, Train Loss: %.9f Hyper Loss: %.9f Mask Loss: %.9f All Loss: %.9f Structure Loss: %.9f Test RMAE: %.9f Test RMSE: %.9f Test PSNR: %.9f "
            % (epoch, iteration, epoch_time, lr, train_loss, losses_hyper, losses_mask, losses_all, losses_structure,
               val_loss, val_rmse, val_psnr))


# Training
def train(train_loader, model, criterion, criterion_2, criterion_3, structure_loss, optimizer, epoch, iteration,
          init_lr, decay_power, trade_off):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()
    random.shuffle(train_loader)

    losses = AverageMeter()
    losses_hyper = AverageMeter()
    losses_mask = AverageMeter()
    losses_all = AverageMeter()
    losses_structure = AverageMeter()  # 新增结构损失记录

    for k, train_data_loader in enumerate(train_loader):
        for i, (images, labels) in enumerate(train_data_loader):
            labels = labels.cuda()
            images = images.cuda()

            images = Variable(images)
            labels = Variable(labels)

            lr = poly_lr_scheduler(optimizer, init_lr, iteration, max_iter=opt.max_iter, power=decay_power)
            iteration = iteration + 1

            # 前向传播
            output, before, after = model(images)

            # 生成注意力图（使用输出的空间方差）
            attention_maps = torch.var(output, dim=1, keepdim=True)  # [B, 1, H, W]

            # 计算各种损失
            loss = criterion(output, labels, images)
            loss_hyper = criterion_2(output)
            loss_mask = criterion_3(before, after)

            # 新增结构感知平滑损失
            loss_structure = structure_loss(output, attention_maps)

            # 总损失组合 - 调整权重
            loss_all = loss + 0.3 * loss_hyper + 0.1 * loss_structure

            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()

            # 更新记录器
            losses.update(loss.data)
            losses_hyper.update(loss_hyper.data)
            losses_mask.update(loss_mask.data)
            losses_all.update(loss_all.data)
            losses_structure.update(loss_structure.data)

            print(
                '[Epoch:%02d],[Process:%d/%d],[iter:%d],lr=%.9f,train_losses.avg=%.9f,train_loss_hyper.avg=%.9f,loss_mask_all.avg=%.9f,loss_all.avg=%.9f,structure_loss.avg=%.9f'
                % (epoch, k + 1, len(train_loader), iteration, lr, losses.avg, losses_hyper.avg, losses_mask.avg,
                   losses_all.avg, losses_structure.avg))

    return losses.avg, losses_hyper.avg, losses_mask.avg, losses_all.avg, losses_structure.avg, iteration, lr


def validate(val_loader, model, criterion_mrae, criterion_rmse, criterion_psnr):
    model.eval()
    losses_mrae = AverageMeter()
    losses_rmse = AverageMeter()
    losses_psnr = AverageMeter()

    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()
        with torch.no_grad():
            # compute output
            output, before, after = model(input)

            # 调整 target 的大小，使其匹配 output 的维度
            target_resized = F.interpolate(target, size=output.shape[2:], mode='bilinear', align_corners=False)

            # 如果目标通道数为 482，将其降维到 31
            target_resized = F.adaptive_avg_pool3d(target_resized,
                                                   (31, target_resized.shape[2], target_resized.shape[3]))

            # 使用调整后的 target_resized 来计算损失
            loss_mrae = criterion_mrae(output, target_resized)
            loss_rmse = criterion_rmse(output, target_resized)
            loss_psnr = criterion_psnr(output, target_resized)

        # record loss
        losses_mrae.update(loss_mrae.data)
        losses_rmse.update(loss_rmse.data)
        losses_psnr.update(loss_psnr.data)

    return losses_mrae.avg, losses_rmse.avg, losses_psnr.avg


# Learning rate
def poly_lr_scheduler(optimizer, init_lr, iteraion, lr_decay_iter=1, max_iter=100, power=0.9):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power

    """
    if iteraion % lr_decay_iter or iteraion > max_iter:
        return optimizer

    lr = init_lr * (1 - iteraion / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
    print(torch.__version__)