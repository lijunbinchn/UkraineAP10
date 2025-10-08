import torch
from tqdm import tqdm
import numpy as np
import time
from scipy import stats
from losses import BCEDiceLoss
from scipy.ndimage import label
from losses import *

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, count, weight):
        self.val = val
        self.avg = val
        self.count = count
        self.sum = val * weight
        self.initialized = True

    def update(self, val, count=1, weight=1):
        if not self.initialized:
            self.initialize(val, count, weight)
        else:
            self.add(val, count, weight)

    def add(self, val, count, weight):
        self.val = val
        self.count += count
        self.sum += val * weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

def align_dims(np_input, expected_dims=2):
    dim_input = len(np_input.shape)
    np_output = np_input
    if dim_input>expected_dims:
        np_output = np_input.squeeze(0)
    elif dim_input<expected_dims:
        np_output = np.expand_dims(np_input, 0)
    assert len(np_output.shape) == expected_dims
    return np_output

def binary_accuracy(pred, label):
    pred = align_dims(pred, 2)
    label = align_dims(label, 2)

    # 将output_mask二值化
    pred = (pred >= 0.5)
    label = (label >= 0.5)

    TP = float((pred * label).sum())
    FP = float((pred * (1 - label)).sum())
    FN = float(((1 - pred) * (label)).sum())
    TN = float(((1 - pred) * (1 - label)).sum())
    precision = TP / (TP + FP + 1e-10)
    recall = TP / (TP + FN + 1e-10)
    IoU = TP / (TP + FP + FN + 1e-10)
    acc = (TP + TN) / (TP + FP + FN + TN)
    F1 = 0
    if acc > 0.99 and TP == 0:
        precision = 1
        recall = 1
        IoU = 1
    if precision > 0 and recall > 0:
        F1 = stats.hmean([precision, recall])
    return acc, precision, recall, F1, IoU

def obejct_eval(pred, target):
    try:
        pred = align_dims(pred, 2)
        target = align_dims(target, 2)

        # 将output_mask二值化
        pred = (pred == 1).astype(np.int8)
        target = (target == 1).astype(np.int8)

        structure = np.ones((3, 3), dtype=int)
        labeled_M, num_M = label(pred, structure=structure)
        labeled_O, num_O = label(target, structure=structure)

        GOC, GUC, GTC = 0.0, 0.0, 0.0
        sum_area_Mi = 0.0
        for i in range(1, num_M + 1):
            Mi = labeled_M == i
            Oj_list = []
            Mi_n_Oj_list = []
            for j in range(1, num_O + 1):
                Oj = labeled_O == j
                Oj_list.append(Oj)
                Mi_n_Oj_list.append(np.sum(Mi & Oj))
            over_max_index = np.argmax(Mi_n_Oj_list)
            Oi = Oj_list[over_max_index]
            Mi_n_Oi = Mi_n_Oj_list[over_max_index]

            area_Mi_n_Oi = np.sum(Mi_n_Oi)
            area_Oi = np.sum(Oi)
            area_Mi = np.sum(Mi)

            sum_area_Mi += area_Mi

            OC = 1 - area_Mi_n_Oi / area_Oi
            UC = 1 - area_Mi_n_Oi / area_Mi

            TC = np.sqrt((OC ** 2 + UC ** 2) / 2)

            GOC += (OC * area_Mi)
            GUC += (UC * area_Mi)
            GTC += (TC * area_Mi)

        GOC /= sum_area_Mi
        GUC /= sum_area_Mi
        GTC /= sum_area_Mi
        return GOC, GUC, GTC

    except Exception as e:
        return 0, 0, 0


def calculate_batch_accuracy(outputs, targets):

    '''
    outputs: batch_size * 256 *256
    targets: batch_size * 256 *256
    '''

    batch_size = outputs.shape[0]  # 批次大小
    acc_list, precision_list, recall_list, F1_list, IoU_list, GOC_list, GUC_list, GTC_list = [], [], [], [], [], [], [], []

    for i in range(batch_size):

        # 值在0-1之间
        output_mask = outputs[i]  # 获取每张图片的输出
        target = targets[i]  # 获取每张图片的目标标签
        # print(output_mask.shape)
        # 获取 output_mask 的大小
        (height, width) = output_mask.shape

        # 将output_mask二值化
        res = np.zeros((height, width))

        res[output_mask > 0.5] = 1
        res[output_mask <= 0.5] = 0

        # 计算二分类指标
        acc, precision, recall, F1, IoU = binary_accuracy(res, target)
        # 计算 GOC，GUC，GTC
        GOC, GUC, GTC = obejct_eval(res, target)

        # 将每张图片的结果存入对应的列表
        acc_list.append(acc)
        precision_list.append(precision)
        recall_list.append(recall)
        F1_list.append(F1)
        IoU_list.append(IoU)
        GOC_list.append(GOC)
        GUC_list.append(GUC)
        GTC_list.append(GTC)

    # 计算批次的平均值
    avg_acc = np.mean(acc_list)
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_F1 = np.mean(F1_list)
    avg_IoU = np.mean(IoU_list)
    avg_GOC = np.mean(GOC_list)
    avg_GUC = np.mean(GUC_list)
    avg_GTC = np.mean(GTC_list)

    return avg_acc, avg_precision, avg_recall, avg_F1, avg_IoU, avg_GOC, avg_GUC, avg_GTC

def evaluate_net(device, model, data_loader):
    model.eval()
    losses = []
    start = time.perf_counter()

    acc_meter = AverageMeter()
    precision_meter = AverageMeter()
    recall_meter = AverageMeter()
    F1_meter = AverageMeter()
    IoU_meter = AverageMeter()
    GOC_meter = AverageMeter()
    GUC_meter = AverageMeter()
    GTC_meter = AverageMeter()

    with torch.no_grad():

        for iter, data in enumerate(tqdm(data_loader)):

            val_name, inputs, target_mask, target_boundary, targets_distance = data
            inputs = inputs.to(device)
            target_mask = target_mask.to(device)
            target_boundary = target_boundary.to(device)
            targets_distance = targets_distance.to(device)

            # outputs 是一个列表, 第一个元素是掩膜输出结果, 第二个元素是边界输出结果, 第三个元素是距离图输出结果
            outputs = model(inputs.float())

            # 获取当前 batch 的 batch_size
            # batch_size = inputs.size(0)  # inputs 的第一个维度是 batch_size

            # outputs[0] 是 一个（batchsize, 1, width,height)大小的张量
            # outputs_mask（batchsize,width,height)大小的数组

            # 这里进行了sigmiod处理。
            outputs_masks = outputs[0].sigmoid().detach().cpu().numpy().squeeze(axis=1) # 移除通道维度
            target_masks = target_mask.detach().cpu().numpy()

            # res = np.zeros((256, 256))
            # # res = np.zeros((batch_size, 256, 256))
            # # indices = np.argmax(output_mask, axis=0)
            # res[output_mask > 0.5] = 255
            # res[output_mask <= 0.5] = 0
            # res = morphology.remove_small_objects(res.astype(int), 1000)

            # acc, precision, recall, F1, IoU = binary_accuracy(res, targets1)
            avg_acc, avg_precision, avg_recall, avg_F1, avg_IoU, avg_GOC, avg_GUC, avg_GTC = calculate_batch_accuracy(outputs_masks, target_masks)

            acc_meter.update(avg_acc)
            precision_meter.update(avg_precision)
            recall_meter.update(avg_recall)
            F1_meter.update(avg_F1)
            IoU_meter.update(avg_IoU)
            GOC_meter.update(avg_GOC)
            GUC_meter.update(avg_GUC)
            GTC_meter.update(avg_GTC)

            # 这里就没有使用sigmoid,因为BCEDiceLoss会进行sigmoid处理。
            # 这里的损失应该和训练时一样。
            criterion1 = BCEDiceLoss()  # mask_loss BCE loss 参考SEANet
            criterion2 = LossMulti(num_classes=2)  # contour_loss NLL 参考bsinet
            criterion3 = nn.MSELoss()
            loss1 = criterion1(outputs[0], target_mask)
            loss2 = criterion2(outputs[1], target_boundary)
            loss3 = criterion3(outputs[2], targets_distance)
            loss = awl(loss1, loss2, loss3)

            losses.append(loss.item())

        print('avg Acc %.2f, Pre %.2f, Recall %.2f, F1 %.2f, IOU %.2f, GOC %.2f, GUC %.2f, GTC %.2f' % (
            acc_meter.avg * 100, precision_meter.avg * 100, recall_meter.avg * 100, F1_meter.avg * 100,
            IoU_meter.avg * 100, GOC_meter.avg, GUC_meter.avg, GTC_meter.avg))

    return (np.mean(losses), acc_meter.avg * 100, precision_meter.avg * 100, recall_meter.avg * 100,
            F1_meter.avg * 100, IoU_meter.avg * 100, GOC_meter.avg, GUC_meter.avg, GTC_meter.avg, time.perf_counter() - start)
