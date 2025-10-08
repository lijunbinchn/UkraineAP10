import argparse
import logging
import os
import time
import datetime
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import S2ParcelDataset
from HBGNet import Field
from losses import *
from utils import evaluate_net
import random
import numpy as np

# 参数都在这里设置。
def parse_args():
    parser = argparse.ArgumentParser(description="train setup for segmentation")
    parser.add_argument("--data-path", default=r"E:\Codes\MyProjectCodes\Agricultural_Parcel\data\UKR\UKR_2023\data4train")
    parser.add_argument("--save-path", default=r"E:\Codes\MyProjectCodes\Agricultural_Parcel\ukraine_parcel\HBGNet_improve\save_weights\data4train", type=str, help="Model save path.")
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    # parser.add_argument("--backbone-path", default=r"./pre_weights/pvt_v2_b2.pth")
    parser.add_argument("--use_pretrained", type=bool, default=False, help="Load pretrained checkpoint.")
    parser.add_argument("--pretrained_model_path", type=str, default=None, help="If use_pretrained is true, provide checkpoint.")

    parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')
    parser.add_argument("-b", "--batch-size", default=8, type=int)
    parser.add_argument("--epochs", default=150, type=int, metavar="N", help="number of total epochs to train")
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--seed", default=2025, type=int)

    args = parser.parse_args()

    return args


# 保存训练和验证的结果到 csv 文件
def save_results_to_csv(train_info_path, train_loss, val_loss, val_iou, acc, precision, recall, f1, epoch, goc, guc, gtc):
    # 定义 CSV 文件路径
    csv_path = train_info_path

    # 如果 CSV 文件不存在，则创建并写入列名
    if not os.path.exists(csv_path):
        df = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss', 'acc', 'precision', 'recall', 'f1', 'val_iou', 'goc', 'guc', 'gtc'])
        df.to_csv(csv_path, index=False)

    # 创建当前训练信息的 DataFrame
    new_data = pd.DataFrame({
        'epoch': [epoch],
        'train_loss': [train_loss],
        'val_loss': [val_loss],
        'acc': [acc],
        'precision': [precision],
        'recall': [recall],
        'f1': [f1],
        'val_iou': [val_iou],
        'goc': [goc],
        'guc': [guc],
        'gtc': [gtc]
    })

    # 追加数据到 CSV 文件中
    new_data.to_csv(csv_path, mode='a', header=False, index=False)

def train_net(net, data_path, epochs, batch_size, lr, num_workers, device, save_path, use_pretrained, pretrained_model_path):

    print('Starting training...')

    # 如果路径不存在，则创建
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
        print('Save path created:', save_path)

    folder_name = os.path.basename(save_path)

    train_dataset = S2ParcelDataset(data_path, train_flag=True, aug_flag=True, normalize_flag=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)
    val_dataset = S2ParcelDataset(data_path, train_flag=False, aug_flag=False, normalize_flag=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {len(train_dataset)}
        Validation size: {len(val_dataset)}
        Device:          {device.type}
    ''')

    epoch_start = "0"
    if use_pretrained:
        print("Loading Model {}".format(os.path.basename(pretrained_model_path)))
        net.load_state_dict(torch.load(pretrained_model_path))
        epoch_start = os.path.basename(pretrained_model_path).split(".")[0]
        print(epoch_start)

    # 定义优化器和损失函数等
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    optimizer_awl = torch.optim.Adam(awl.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(epochs), eta_min=1e-5)
    scheduler_awl = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_awl, int(epochs), eta_min=1e-5) ##

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(1e10), eta_min=1e-5)

    # 打印当前学习率
    print('Learning rate: ', optimizer.state_dict()['param_groups'][0]['lr'])

    criterion1 = BCEDiceLoss()  # mask_loss BCE loss 参考SEANet
    criterion2 = LossMulti(num_classes=2)  # contour_loss NLL 参考bsinet
    criterion3 = nn.MSELoss()

    best_iou = 0.0
    # 开始训练模型
    start_time = time.time()
    for epoch in range(int(epoch_start) + 1, 1 + epochs):

        print("——————第 {} 轮训练开始——————".format(epoch))
        # 进行训练
        net.train()

        # 记录保存损失
        train_loss = 0
        with tqdm(total=len(train_dataset), desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:

            # 加载数据数据，进行每轮的模型训练
            for step, (image_name, image, mask, boundary, dist) in enumerate(train_loader):

                # 将取出的这一批次的数据放到GPU上
                image = image.to(device)
                mask = mask.to(device)
                boundary = boundary.to(device)
                dist = dist.to(device)

                # 将数据输入到网络中，得到输出
                mask_pred, boundary_pred, dist_pred = net(image)

                # 根据输出计算损失
                loss1 = criterion1(mask_pred, mask)
                loss2 = criterion2(boundary_pred, boundary)
                loss3 = criterion3(dist_pred, dist)
                loss = awl(loss1, loss2, loss3)

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
                })

                # 累加epoch损失，记录训练集的损失
                train_loss += loss.item() * image.shape[0]  # 计算一个batch的损失,其中image为batch的图片

                optimizer.zero_grad()
                optimizer_awl.zero_grad()
                loss.backward()
                optimizer.step()
                optimizer_awl.step()

                pbar.update(image.shape[0])

            # 循环结束，完成一轮训练后更新学习率
            train_loss = train_loss / len(train_loader.dataset)

            # 更新学习率后，打印当前学习率
            scheduler.step()
            scheduler_awl.step()
            print('Scheduler step!')
            Learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
            print('Learning rate: ', Learning_rate)
            Learning_rate_awl = optimizer_awl.state_dict()['param_groups'][0]['lr']
            print('Learning rate 1: ', Learning_rate_awl)

            # 经过一轮训练后，对训练好的模型进行验证
            val_loss, accuracy, precision, recall, f1, val_iou, val_goc, val_guc, val_gtc, val_time = evaluate_net(device, net, val_loader)

            # 保存损失和精度结果
            save_results_to_csv(os.path.join(save_path, folder_name + "_train_info.csv"), train_loss, val_loss, val_iou, accuracy,
                                precision, recall, f1, epoch, val_goc, val_guc, val_gtc)
            print('epoch:{}: train_loss:{}, val_loss:{}'.format(epoch, train_loss, val_loss))

            # 根据IOU的值来进行模型保存
            if val_iou > best_iou:
                best_iou = val_iou
                torch.save(net.state_dict(), os.path.join(save_path, f"{epoch}_best.pt"))
                print("已保存模型！", "当前模型的IOU为：", val_iou)
                logging.info(f'Checkpoint "{epoch}_best.pt saved !')

            # 保存模型
            if epoch % 1 == 0:
                torch.save(
                    net.state_dict(), os.path.join(save_path, str(epoch) + ".pt")
                )

    # 计算训练时间
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


# 在main函数里将参数传入train_net函数。
def main(args):

    # 设置随机种子
    seed_torch(seed=args.seed)
    print(f"设置的随机种子为：{args.seed}")

    # 设置 device 为 GPU 或者 CPU
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    net = Field(num_classes=2)
    print('Net parameters: %d' % sum(p.numel() for p in net.parameters()))
    net = net.to(device)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'Using device {device}')

    # todo 开启了这个方法，模型参数的名字对多一个module
    # 如果有多个GPU，则并行计算
    # net = torch.nn.parallel.DataParallel(net.to(device))

    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs!")
    #     net = torch.nn.DataParallel(net)

    # 将参数闯入train_net函数，进行模型训练
    train_net(net=net,
              data_path=args.data_path,
              epochs=args.epochs,
              batch_size=args.batch_size,
              lr=args.lr,
              num_workers=args.num_workers,
              device=device,
              save_path=args.save_path,
              use_pretrained=args.use_pretrained,
              pretrained_model_path=args.pretrained_model_path)

# 固定随机数种子
def seed_torch(seed=3407):
    random.seed(seed)   # Python的随机性
    os.environ['PYTHONHASHSEED'] = str(seed)    # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)   # numpy的随机性
    torch.manual_seed(seed)   # torch的CPU随机性，为CPU设置随机种子
    torch.cuda.manual_seed(seed)   # torch的GPU随机性，为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.   torch的GPU随机性，为所有GPU设置随机种子
    torch.backends.cudnn.benchmark = False   # if benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = True   # 选择确定性算法

# 执行函数，进行训练
if __name__ == '__main__':
   
    # 解析参数
    args = parse_args()

    # 传入参数，进行训练
    main(args)
