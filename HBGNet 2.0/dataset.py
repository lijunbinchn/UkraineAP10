import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from scipy import io
import os
from osgeo import gdal
from skimage import io
import matplotlib.pyplot as plt
import imgaug.augmenters as iaa

def readTif(fileName, xoff = 0, yoff = 0, data_width = 0, data_height = 0):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")
    #  栅格矩阵的列数
    width = dataset.RasterXSize
    #  栅格矩阵的行数
    height = dataset.RasterYSize
    #  波段数
    bands = dataset.RasterCount
    #  获取数据
    if(data_width == 0 and data_height == 0):
        data_width = width
        data_height = height
    data = dataset.ReadAsArray(xoff, yoff, data_width, data_height)
    #  获取仿射矩阵信息
    geotrans = dataset.GetGeoTransform()
    #  获取投影信息
    proj = dataset.GetProjection()
    return width, height, bands, data, geotrans, proj

#保存遥感影像
def writeTiff(im_data, im_geotrans, im_proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
          im_bands, (im_height, im_width) = 1, im_data.shape
    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if (dataset != None):
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
    if im_bands == 1:
      dataset.GetRasterBand(1).WriteArray(im_data)
    else:
        for i in range(im_bands):
           dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset

class S2ParcelDataset(Dataset):

    # 类的构造函数，训练和验证数据分开读取。
    # 传入保存训练和测试数据的文件夹根路径
    # 获取训练集和测试集的数据路径，分别得到训练和验证的数据集。然后需要将路径传入len 和 getitem，获取数据集数量和遍历该数据。
    def __init__(self, data_path, train_flag=True, aug_flag=True, normalize_flag=True):
        """
        FHAP数据集类，支持训练和验证数据的加载，并添加数据增强操作。

        :param data_path: 数据集根目录
        :param train: 是否为训练模式
        :param transform: 是否应用数据增强
        """

        # 数据集文件夹路径
        self.data_path = data_path
        # 是否为训练模式
        self.train_flag = train_flag
        # 是否进行数据增强
        self.aug_flag = aug_flag
        # 是否进行数据标准化
        self.normalize_flag = normalize_flag

        # 获取训练数据文件夹, 需要读取影像和多任务数据
        self.train_data_path = os.path.join(data_path, 'train')
        self.train_images_path = os.path.join(self.train_data_path, 'train_image')
        # 获取验证数据文件夹， 只需读取影像和掩膜数据
        self.val_data_path = os.path.join(data_path, 'val')
        self.val_images_path = os.path.join(self.val_data_path, 'val_image')

        # 如果是训练模式，则读取训练数据，否则读取验证数据

        if train_flag:
            # 获取训练数据的所有图片路径
            self.images_path = [os.path.join(self.train_images_path, x) for x in os.listdir(self.train_images_path)]

        else:
            # 如果是验证模式，获取验证数据的所有图片路径
            self.images_path = [os.path.join(self.val_images_path, x) for x in os.listdir(self.val_images_path)]

        # 定义数据增强流程，训练数据进行数据增强，验证数据不进行。
        self.transform = iaa.Sequential([
            iaa.Rot90([0, 1, 2, 3]),
            iaa.VerticalFlip(p=0.5),
            iaa.HorizontalFlip(p=0.5)
        ])

        # 定义图像标准化的参数
        # UKR_2023
        self.normalize = transforms.Normalize([675.6458129882812, 656.1358642578125, 447.96978759765625], [264.444580078125, 167.1501922607422, 138.0218505859375])

        # 定义转换为张量的操作
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.images_path)

    # 训练和验证的获取方式是不一样的，训练需要增强，验证不需要增强。
    def __getitem__(self, idx):
        # 如果是训练模式，按照训练模式读取数据
        if self.train_flag:
            # 获取图片，掩膜，边界和距离等数据的路径
            image_path = self.images_path[idx]
            mask_path = image_path.replace('image', 'mask')
            boundary_path = image_path.replace('image', 'boundary')
            distance_path = image_path.replace('image', 'distance')

            # 加载图片，掩膜，边界和距离等数据
            image = self.load_image(image_path)
            mask = self.load_mask(mask_path)
            boundary = self.load_boundary(boundary_path)
            distance = self.load_distance(distance_path)

            image_name = os.path.basename(image_path)

            # 对 image 进行通道变换
            image = np.transpose(image, (1, 2, 0))
            # 如果进行数据增强
            # todo 检查经过数据增强后的数据和类型, 数据类型不会改变。
            if self.aug_flag:
                # 注意 img_aug的输入类型是numpy.ndarray, 输入维度位 H,W,C
                image, label = self.transform(image=image, segmentation_maps=np.stack(
                    (mask[np.newaxis, :, :], boundary[np.newaxis, :, :], distance[np.newaxis, :, :]), axis=-1))

                mask, boundary, distance = label[0, :, :, 0], label[0, :, :, 1], label[0, :, :, 2]

            # 对影像进行归一化和标准化
            # 对影像进行归一化
            # image = image / 65535.0
            image = image.copy()
            # 转换为 PyTorch Tensor 张量
            image = np.transpose(image, (2, 0, 1))
            image = torch.from_numpy(image).float()
            # 如果进行数据标准化
            if self.normalize_flag:
                image = self.normalize(image)

            # 对多任务数据进行归一化
            mask = (mask / 255.0).astype(np.float32)
            boundary = (boundary / 255.0).astype(np.float32)
            distance = (distance / 255.0).astype(np.float32)

            # 对多任务数据进行维度拓展和转换为张量
            mask = torch.from_numpy(np.expand_dims(mask, 0)).float()
            boundary = torch.from_numpy(np.expand_dims(boundary, 0)).long()
            distance = torch.from_numpy(np.expand_dims(distance, 0)).float()

            return image_name, image, mask, boundary, distance

        # 如果是验证模式，按照验证模式读取数据
        else:
            # 验证数据只需要影像和掩膜数据, 也需要边界和距离数据
            image_path = self.images_path[idx]
            mask_path = image_path.replace('image', 'mask')
            boundary_path = image_path.replace('image', 'boundary')
            distance_path = image_path.replace('image', 'distance')
            # 加载图片，掩膜，边界和距离等数据
            image = self.load_image(image_path)
            mask = self.load_mask(mask_path)
            boundary = self.load_boundary(boundary_path)
            distance = self.load_distance(distance_path)

            image_name = os.path.basename(image_path)

            # 不需要进行数据增强

            # 对影像进行归一化和标准化
            # 对影像进行归一化
            # image = image / 65535.0
            image = image.copy()
            image = torch.from_numpy(image).float()
            # 如果进行数据标准化
            if self.normalize_flag:
                image = self.normalize(image)

            # 进行归一化
            mask = (mask / 255.0).astype(np.float32)
            boundary = (boundary / 255.0).astype(np.float32)
            distance = (distance / 255.0).astype(np.float32)
            # 进行维度拓展和转换为张量
            mask = torch.from_numpy(np.expand_dims(mask, 0)).float()
            boundary = torch.from_numpy(np.expand_dims(boundary, 0)).long()
            distance = torch.from_numpy(np.expand_dims(distance, 0)).float()

            return image_name, image, mask, boundary, distance


    def load_image(self, path):
        im_width, im_height, im_bands, img_data, im_geotrans, im_proj = readTif(path)
        return img_data

    def load_mask(self, path):
        im_width, im_height, im_bands, mask, im_geotrans, im_proj = readTif(path)
        return mask

    def load_boundary(self, path):
        # 得到的边界，0~255之间
        im_width, im_height, im_bands, boundary, im_geotrans, im_proj = readTif(path)
        return boundary

    def load_distance(self, path):
        im_width, im_height, im_bands, dist, im_geotrans, im_proj = readTif(path)
        return dist




if __name__ == '__main__':

    def scale_data(rgb_data):
        # 可选：进行线性拉伸以增强对比度
        percentile_min = 2
        percentile_max = 98

        for i in range(3):
            vmin = np.percentile(rgb_data[:, :, i], percentile_min)
            vmax = np.percentile(rgb_data[:, :, i], percentile_max)
            rgb_data[:, :, i] = np.clip((rgb_data[:, :, i] - vmin) / (vmax - vmin), 0, 1)

        # 可选：应用 Gamma 校正
        gamma = 0.8
        rgb_data = np.power(rgb_data, gamma)

        return rgb_data

    def normalize_image(image):
        """将整个图像数据归一化到0-1之间"""
        image_min = np.min(image)
        image_max = np.max(image)
        normalized = (image - image_min) / (image_max - image_min)
        return normalized

    # 数据集路径
    data_path = r'E:\Codes\MyProjectCodes\Agricultural_Parcel\data\UKR\UKR_2023\data4train'
    # data_path = r'E:\Codes\MyProjectCodes\Agricultural_Parcel\data\FHAPD\FHAPD_Test\data100'
    # train_dataset = PlanetParcelDataset(data_path, train=True, transform=True)
    # val_dataset = PlanetParcelDataset(data_path, train=False, transform=True)

    train_flag = True
    aug_flag = True
    normalize_flag = True
    scale_flag = normalize_flag

    # dataset = AI4Boundaries(data_path, train_flag, aug_flag, normalize_flag)
    dataset = S2ParcelDataset(data_path, train_flag, aug_flag, normalize_flag)
    # dataset = FHAPDataset(data_path, train_flag, aug_flag, normalize_flag)
    # 创建数据加载器
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)

    # 读取训练数据的一张图片及其对应的标签
    data_size = len(dataset)
    print("data_size:", data_size)
    print("data_loader_batch:", len(data_loader))
    print("data_loader.dataset:", len(data_loader.dataset))

    # 选择一张图片
    index = random.randint(0, data_size)
    # index = 50
    print("index:", index)
    if train_flag:
        image_name, img, mask, boundary, dist = dataset[index]
        # image_path = os.path.join(data_path, 'images', 'train', image_name)
        image_path = os.path.join(data_path, 'train', 'train_image', image_name)
        # image_original = np.array(Image.open(image_path))

        im_width, im_height, im_bands, image_original, im_geotrans, im_proj = readTif(image_path)
        # 对整个图像进行归一化
        image_original = normalize_image(image_original)
        image_original = scale_data(image_original)
    else:
        image_name, img, mask, boundary, dist = dataset[index]
        image_path = os.path.join(data_path, 'val', 'val_image', image_name)
        # image_path = os.path.join(data_path, 'images', 'val', image_name)
        # image_original = np.array(Image.open(image_path))

        im_width, im_height, im_bands, image_original, im_geotrans, im_proj = readTif(image_path)
        # 对整个图像进行归一化
        image_original = normalize_image(image_original)
        image_original = scale_data(image_original)


    # 将图像转换为numpy数组
    img = np.array(img)
    print("图片的数据类型", img.dtype)
    print(type(mask))
    img = img.squeeze()#.numpy()
    print(img.shape)

    # 将张量转换为 NumPy 数组，并将维度从 (C, H, W) 转换为 (H, W, C)
    rgb_data = img.transpose(1, 2, 0)
    print(rgb_data.shape)
    # 输出值的范围
    print("img_np.min(), img_np.max():", rgb_data.min(), rgb_data.max())

    # 显示原始图像
    plt.figure(figsize=(10, 10))
    print("image_original.shape", image_original.shape)
    image_original = image_original.transpose(1, 2, 0)
    plt.imshow(image_original)
    plt.title("Image Original", fontsize=16)
    plt.axis('off')
    plt.show()

    # 显示图像
    if scale_flag:
        rgb_data = scale_data(rgb_data)
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb_data)
    plt.title("Image", fontsize=16)
    plt.axis('off')
    plt.show()

    mask = mask.numpy().transpose(1, 2, 0)
    # 输出值的范围
    print("mask.min(), mask.max():", mask.min(), mask.max())
    print(mask.shape)
    plt.imshow(mask)
    plt.axis('off')
    plt.show()

    boundary = boundary.numpy().transpose(1, 2, 0)
    # 输出值的范围
    print("boundary.min(), boundary.max():", boundary.min(), boundary.max())
    print(boundary.shape)
    plt.imshow(boundary)
    plt.axis('off')
    plt.show()

    dist = dist.numpy().transpose(1, 2, 0)
    # 输出值的范围
    print("dist.min(), dist.max():", dist.min(), dist.max())
    print(dist.shape)
    plt.imshow(dist)
    plt.axis('off')
    plt.show()








