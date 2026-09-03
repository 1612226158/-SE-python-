import configparser
import json
import logging
import os
import sys
from copy import deepcopy
from datetime import datetime

import numpy as np
import pandas as pd

import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from tqdm import tqdm
from PIL import Image, ImageFile
import pillow_avif  # 导入 AVIF 支持插件

from src_v2 import merged_dict, metrics_file, UPDATE_N

from src_v2 import extensions_suffix

ImageFile.LOAD_TRUNCATED_IMAGES = True  # 忽略部分损坏的图片

# ===== 路径基于本文件定位（与 PyCharm 工作目录无关） =====
_CALC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CALC_DIR)
_ORIGINAL_SRC_DIR = os.path.join(_PROJECT_ROOT, 'src')   # 只读复用原缓存与配置，绝不写入


class __OtherImport:
    __temporary = [pillow_avif]


def get_classnames():
    with open(os.path.join(_PROJECT_ROOT, 'classname.txt'), mode='r', encoding='utf-8') as fp:
        class_labels = fp.read().split('\n')
    # print(f"get_classnames, {class_labels}")
    return class_labels


def custom_collate_fn(batch):
    # 过滤掉 None
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        raise ValueError("All samples in the batch are invalid.")
    return torch.utils.data.default_collate(batch)


class SafeImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 train=True,
                 id_to_main_class=None,
                 renew_class_to_index=None,
                 # id_to_child_class=None,
                 # main_class_to_index=None,
                 # child_class_to_index=None
                 ):
        """
        扩展的 SafeImageFolder 类，支持过滤损坏图片、缓存机制以及动态生成主类和子类标签。
        :param root: 数据集根目录
        :param transform: 数据变换
        :param target_transform: 标签变换
        :param train: 根据训练或验证选择缓存文件
        :param id_to_main_class: 主类映射
        # :param id_to_child_class: 子类映射
        # :param main_class_to_index: 子类映射
        """
        if train:
            cache_file = os.path.join(_CALC_DIR, "images_cache_train_v2.json")       # v2 增量缓存（src_v2 内）
            seed_cache = os.path.join(_ORIGINAL_SRC_DIR, "images_cache_train.json")  # 原缓存仅作首次种子
        else:
            cache_file = os.path.join(_CALC_DIR, "images_cache_val_v2.json")
            seed_cache = os.path.join(_ORIGINAL_SRC_DIR, "images_cache_val.json")
        self.cache_file = cache_file
        self.seed_cache = seed_cache
        self.root = root
        # self.id_to_main_class = id_to_main_class
        # self.id_to_child_class = id_to_child_class
        # self.main_class_to_index = main_class_to_index
        # print(id_to_main_class)
        # print(id_to_child_class)
        # print(main_class_to_index)

        # self.id_to_main_class = {str(k): v for k, v in id_to_main_class.items()} if id_to_main_class else None
        self.id_to_main_class = id_to_main_class
        self.renew_class_to_index = renew_class_to_index
        # self.id_to_child_class = {str(k): v for k, v in id_to_child_class.items()} if id_to_child_class else None
        # self.main_class_to_index = main_class_to_index
        # self.child_class_to_index = child_class_to_index

        super(SafeImageFolder, self).__init__(root, transform=transform, target_transform=target_transform)
        self.valid_samples = self._load_or_build_cache()

        # 重写 samples，使用过滤后的有效样本
        self.samples = self.valid_samples
        self.imgs = self.samples
        self.targets = [s[1] for s in self.samples]   # 同步 targets（修复历史不同步隐患）

        # 确保数据集类别与映射一致
        # self.dataset_classes = sorted([str(cls) for cls in os.listdir(root) if cls.isdigit()])
        # print(f"{self.dataset_classes}")
        # print(f"{self.dataset_classes=}")

    def _is_image_valid(self, path):
        """
        检查单张图片是否有效。
        :param path: 图片路径
        :return: bool
        """
        try:
            with Image.open(path) as img:
                img.verify()
            return True
        except (OSError, ValueError):
            return False

    def _scan_disk(self):
        """列磁盘当前图片文件 -> {相对路径: (mtime, size)}。只列名+stat，不做内容验证。"""
        disk = {}
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.jfif')
        for folder in os.listdir(self.root):
            folder_path = os.path.join(self.root, folder)
            if not folder.isdigit() or not os.path.isdir(folder_path):
                continue
            for name in os.listdir(folder_path):
                if not name.lower().endswith(valid_exts):
                    continue
                full = os.path.join(folder_path, name)
                try:
                    st = os.stat(full)
                except OSError:
                    continue
                disk[os.path.join(folder, name)] = (int(st.st_mtime), st.st_size)
        return disk

    def _load_or_build_cache(self):
        """v2 缓存：每次启动校验一遍磁盘并增量自动更新（新增/删除/变动），原子写入。
        首启动用原缓存当种子（已验证条目），只需补验新增/变动文件，秒级完成。"""
        entries = {}   # 相对路径 -> [mtime, size, class_idx]
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                entries = {p: list(v) for p, v in data.get('samples', {}).items()}
                logging.info(f'[缓存v2] 加载 {len(entries)} 条: {self.cache_file}')
            except Exception as e:
                logging.warning(f'[缓存v2] 缓存损坏({e})，将重建')
                entries = {}

        if not entries and os.path.exists(self.seed_cache):
            # 首次：用原缓存（已验证过的条目）做种子，只需 stat 一遍拿 mtime/size
            try:
                with open(self.seed_cache, 'r', encoding='utf-8') as f:
                    old = json.load(f)
                for path, idx in old:
                    full = os.path.normpath(os.path.join(self.root, path))
                    if os.path.isfile(full):
                        try:
                            st = os.stat(full)
                            rel = os.path.relpath(full, self.root)
                            entries[rel] = [int(st.st_mtime), st.st_size, int(idx)]
                        except OSError:
                            continue
                logging.info(f'[缓存v2] 以原缓存为种子，导入 {len(entries)} 条')
            except Exception as e:
                logging.warning(f'[缓存v2] 原缓存读取失败({e})，改为全量扫描')

        disk = self._scan_disk()
        added = [p for p in disk if p not in entries]
        removed = [p for p in entries if p not in disk]
        changed = [p for p in entries if p in disk and tuple(entries[p][:2]) != disk[p]]

        verified = 0
        skipped = 0
        for p in added + changed:
            full = os.path.join(self.root, p)
            folder = p.split(os.sep, 1)[0]
            idx = self.classes.index(folder) if folder in self.classes else int(folder)
            if self._is_image_valid(full):
                try:
                    st = os.stat(full)
                except OSError:
                    continue
                entries[p] = [int(st.st_mtime), st.st_size, idx]
                verified += 1
            else:
                entries.pop(p, None)
                skipped += 1
        for p in removed:
            entries.pop(p, None)

        if added or removed or changed or not os.path.exists(self.cache_file):
            tmp = self.cache_file + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump({'version': 2,
                           'updated': datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'),
                           'samples': {p: list(v) for p, v in entries.items()}}, f)
            os.replace(tmp, self.cache_file)   # 原子替换，断电不损坏
            logging.info(f'[缓存v2] 已更新: +新增{len(added)} -删除{len(removed)} 变动{len(changed)} '
                         f'(通过验证{verified}, 剔除{skipped})')

        valid_samples = []
        for p in sorted(entries.keys()):
            valid_samples.append((os.path.normpath(os.path.join(self.root, p)), entries[p][2]))
        logging.info(f'[缓存v2] 本次数据集有效样本: {len(valid_samples)}')
        return valid_samples

    def __getitem__(self, index):
        exit_flag = 1
        while True:
            if exit_flag >= 100:
                print("异常!多次未找到合适的图片!程序已经退出!")
                sys.exit()

            path, target = self.samples[index]
            full_path = os.path.join(self.root, path)

            if not os.path.exists(full_path):
                index = (index + 1) % len(self.samples)
                exit_flag += 1
                continue

            try:
                sample = self.loader(full_path)
                if self.transform is not None:
                    sample = self.transform(sample)

                # 处理层级标签
                if self.id_to_main_class is not None:
                    # class_id = sorted(self.dataset_classes, key=lambda x: int(x))[target]  # 获取实际的类别ID
                    class_id = self.classes[target]
                    # print(f"处理层级标签, {class_id=}")
                    # 使用实际的类别ID而不是数据集索引
                    class_idx_str = str(class_id)
                    # class_idx_str = str(target)

                    # 获取主类和子类名称
                    main_class = self.id_to_main_class.get(class_idx_str)
                    # child_class = self.id_to_child_class.get(class_idx_str)

                    # 处理未知类别
                    if main_class is None:
                        main_class = f"公共类-{class_idx_str}"
                        print(f"警告: 类别 {class_idx_str} 没有主类映射，使用: {main_class}")

                    parent_index = self.renew_class_to_index[main_class]
                    return sample, str(parent_index)

                # 如果没有映射，返回原始标签
                if self.target_transform is not None:
                    target = self.target_transform(target)
                    print(f"原始标签, {target=}")
                return sample, target

            except (OSError, ValueError) as e:
                print(f"加载图像错误: {full_path}, 错误: {e}")
                exit_flag += 1
                index = (index + 1) % len(self.samples)

    def _validate_label_consistency(self, main_class, child_class):
        """验证父类和子类标签是否逻辑一致，基于 merged_dict 的结构"""
        # 提取主类的关键部分（去掉"公共类-"或"私有类-"前缀）
        if main_class.startswith("公共类-"):
            main_key = main_class[4:]  # 去掉"公共类-"前缀
        elif main_class.startswith("私有类-"):
            # 对于私有类，我们不需要验证，因为它们是一对一的关系
            return True
        else:
            main_key = main_class

        # 检查这个主类是否在 merged_dict 中
        if main_key in merged_dict:
            # 获取该主类下的所有子类名称
            valid_child_names = [child_name for _, child_name in merged_dict[main_key]]

            # 检查子类名称是否在有效子类列表中
            if child_class in valid_child_names:
                return True
            else:
                # 如果不在列表中，检查是否有部分匹配（例如，"毛巾"匹配"抹布毛巾"）
                for valid_name in valid_child_names:
                    if child_class in valid_name or valid_name in child_class:
                        return True
                return False
        else:
            # 如果主类不在 merged_dict 中，可能是私有类或其他情况
            # 尝试使用简单的关键词匹配
            main_keywords = main_key.split('-')

            # 检查子类是否包含主类的关键词
            for keyword in main_keywords:
                if keyword in child_class:
                    return True

            # 如果没有匹配的关键词，返回False
            return False

    def is_valid_file(self, path):
        """
        重写 is_valid_file，扩展支持的文件类型。
        """
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.jfif')
        return path.lower().endswith(valid_extensions)


class ConvertToRGB:
    # 解决带有透明图片的问题
    """
    UserWarning: Palette images with Transparency expressed in bytes should be converted to RGBA images
    """

    def __call__(self, img):
        if img.mode in ("P", "PA"):  # 检查是否为调色板模式
            img = img.convert("RGBA")  # 转为 RGBA 模式
        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            # 如果是 RGBA 或者带透明度的图片，转换为 RGB
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'PA':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[3])  # 使用 alpha 通道作为 mask
            return background
        return img.convert('RGB')  # 其他情况直接转换为 RGB


def calculate_mean_and_std(file: str = None,
                           retrieve_file: bool = True,
                           update: bool = False,
                           color_bias_threshold: float = 0.8
                           ):
    """
    :param file: 模型文件夹
    :param retrieve_file: 是否从历史文件中读取
    :param update: 是否更新文件, 前提是retrieve_file为False
    :param color_bias_threshold: 定义剔除阈值, 默认0.8
    :return: mean, std, skip_number
    """
    file_calculate_number = os.path.join(_ORIGINAL_SRC_DIR, 'calculate.config')
    if retrieve_file and os.path.exists(file_calculate_number):
        config = configparser.ConfigParser()
        config.read(file_calculate_number)
        try:
            mean = torch.tensor(list(map(float, config["Statistics"]["mean"].split(","))))
            std = torch.tensor(list(map(float, config["Statistics"]["std"].split(","))))
            skip_number = config["Statistics"]["skip_number"]
            return mean, std, int(skip_number)
        except KeyError:
            pass

    # 数据集加载
    transform = transforms.Compose([
        ConvertToRGB(),
        transforms.ToTensor()  # 将图片转换为Tensor
    ])

    dataset = SafeImageFolder(root=file, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)  # 每次处理一张图片

    train_loader_len = len(loader)

    # 初始化累积量
    mean = torch.zeros(3)  # RGB三通道
    std = torch.zeros(3)
    valid_image_count = 0  # 有效图片计数
    skip_number = 0

    # 定义剔除阈值
    # color_bias_threshold = 0.8  # 如果某通道均值超过总均值的80%，认为有明显偏向

    with tqdm(total=train_loader_len) as tq:
        for images, _ in loader:
            # print("断点1")
            # images形状为 (1, 3, H, W)
            images = images.view(3, -1)  # 展平为 (3, H*W)
            # print("断点2")
            channel_means = images.mean(1)  # 计算每个通道的均值
            # print("断点3")
            channel_total_mean = channel_means.mean()  # 所有通道的整体均值
            # print("断点4")
            tq.update(1)
            max_deviation = torch.abs(channel_means - channel_total_mean).max() / channel_total_mean

            if max_deviation > color_bias_threshold:
                skip_number += 1
                continue  # 跳过偏色图片

            image_std = images.std(dim=1)
            # 检查标准差是否为零（即纯色图像）
            if torch.all(image_std == 0):  # 如果所有通道的标准差都是零，说明是纯色图片
                skip_number += 1
                continue  # 跳过纯色图片

            # 检查标准差是否为 NaN
            if torch.any(torch.isnan(image_std)):
                skip_number += 1
                continue

            # 累积均值和标准差
            mean += channel_means
            std += image_std
            valid_image_count += 1

    # 计算最终均值和标准差
    if valid_image_count > 0:
        mean /= valid_image_count
        std /= valid_image_count

    if update or not os.path.exists(file_calculate_number):
        config = configparser.ConfigParser()
        config["Statistics"] = {
            "mean": ",".join(map(str, mean.tolist())),  # 转为逗号分隔的字符串
            "std": ",".join(map(str, std.tolist())),
            "skip_number": str(skip_number)
        }
        with open(file_calculate_number, "w") as fp:
            config.write(fp)

    return mean, std, skip_number


# 计算 Gini 系数
def gini(array):
    array = np.sort(array)  # 排序
    cumulative_array = np.cumsum(array)  # 累加
    relative_cumulative = cumulative_array / cumulative_array[-1]
    n = len(array)
    gini_index = (n + 1 - 2 * np.sum(relative_cumulative)) / n
    return gini_index


def get_class_labels_number():
    class_name = get_classnames()
    # print(class_name)
    path1 = os.path.join(_PROJECT_ROOT, "train")  # 输入一级文件夹地址
    files1 = os.listdir(path1)  # 读入一级文件夹
    num1 = len(files1)  # 统计一级文件夹中的二级文件夹个数
    num2 = []  # 建立空列表
    file_and_num = []
    for i in range(num1):  # 遍历所有二级文件夹
        path2 = path1 + '//' + files1[i]  # 某二级文件夹的路径
        try:
            files2 = os.listdir(path2)  # 读入二级文件夹
        except NotADirectoryError:
            continue
        num2.append(len(files2))  # 二级文件夹中的文件个数
        file_and_num.append([files1[i], class_name[int(files1[i])], len(files2)])
    # print("所有二级文件夹名:")
    # print(files1)  # 打印二级文件夹名称
    # print("所有二级文件夹中的文件个数:")
    # print(num2)  # 打印二级文件夹中的文件个数

    # print("对应输出:")
    # xinhua = dict(zip(files1, num2))  # 将二级文件夹名称和所含文件个数组合成字典

    # if sortYN:
    #     xinhua = sorted(file_and_num, key=lambda x: x[-1], reverse=True)
    # else:
    #     xinhua = sorted(file_and_num, key=lambda x: int(x[0]), reverse=False)

    # xinhua
    xinhua = sorted(file_and_num, key=lambda x: int(x[0]), reverse=False)
    #
    # for key, name, value in xinhua:  # 将二级文件夹名称和所含文件个数对应输出
    #     print('id:{key}-{name}, 数量:{value}'.format(key=key, name=name, value=value))
    # print(f"Gini系数:{gini(num2)}")
    # print(f"类别最少:{min(num2)}")
    # print(f"类别最少:{max(num2)}")
    return xinhua


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

        if isinstance(self.alpha, (float, int)):
            self.alpha = torch.tensor([1 - self.alpha, self.alpha])
        if isinstance(self.alpha, list):
            self.alpha = torch.tensor(self.alpha, dtype=torch.float)

    def forward(self, inputs, targets):
        log_pt = F.log_softmax(inputs, dim=1)
        pt = torch.exp(log_pt)

        # 获取targets对应的预测概率与log概率
        log_pt = log_pt.gather(1, targets.unsqueeze(1)).view(-1)
        pt = pt.gather(1, targets.unsqueeze(1)).view(-1)

        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            log_pt = alpha_t * log_pt

        loss = - (1 - pt) ** self.gamma * log_pt
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def add_record_metrics(epoch=-1,
                       train_acc_parent=0.0,
                       train_acc_child=0.0,
                       val_acc_parent=0.0,
                       val_acc_child=0.0,
                       train_loss=0.0,
                       val_loss=0.0,
                       precision_parent=0.0,
                       precision_child=0.0,
                       recall_parent=0.0,
                       recall_child=0.0,
                       f1_parent=0.0,
                       f1_child=0.0,
                       gradient_norm=0.0,
                       learning_rate=0.0,
                       time_spend=0,
                       state=1
                       ):
    """
    :param epoch: 当前轮数
    :param train_acc_parent: 含有公共类的训练准确度
    :param train_acc_child: 所有子类的训练准确度

    :param val_acc_parent: 含有公共类的验证准确度
    :param val_acc_child: 所有子类的验证准确度

    :param train_loss: 训练损失
    :param val_loss: 所有子类的验证损失

    :param precision_parent: 含有公共类的精确率
    :param precision_child: 所有子类的精确率

    :param recall_parent: 含有公共类的召回率
    :param recall_child: 所有子类的召回率

    :param f1_parent: 含有公共类的f1分数
    :param f1_child: 所有子类的f1分数

    :param gradient_norm: 梯度范数
    :param learning_rate: 学习率

    :param time_spend: 花费时长, 单位秒
    :return: epoch
    """

    learning_rate = learning_rate.cpu().item() if isinstance(learning_rate, torch.Tensor) else learning_rate
    lr_str = f"{learning_rate:.20f}_lr"
    gradient_norm = gradient_norm.cpu().item() if isinstance(gradient_norm, torch.Tensor) else gradient_norm
    grad_str = f"{gradient_norm:.20f}_grad" if abs(gradient_norm) < 1e-6 else f"{gradient_norm}_grad"

    metrics_data = {
        "Epoch": [epoch],

        "Train_Accuracy_Parent": [train_acc_parent],
        "Train_Accuracy_Child": [train_acc_child],

        "Validation_Accuracy_Parent": [val_acc_parent],
        "Validation_Accuracy_Child": [val_acc_child],

        "Train_Loss": [train_loss],
        "Validation_Loss": [val_loss],

        "Precision_Parent": [precision_parent],
        "Precision_Child": [precision_child],

        "Recall_Parent": [recall_parent],
        "Recall_Child": [recall_child],

        "F1_Score_Parent": [f1_parent],
        "F1_Score_Child": [f1_child],

        "Gradient_Norm": [grad_str],  # 使用字符串格式
        "Learning_Rate": [lr_str],  # 使用字符串格式
        "LR_Original": [learning_rate],  # 保留原始值

        "Time_Spend": time_spend,  # 该epoch花费时常, 单位秒
        "Save_Now": datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'),  # 保存指标的时间

        "state": [state]
    }

    if not os.path.isfile(metrics_file):
        pd.DataFrame(metrics_data).to_csv(metrics_file, index=False)
    else:
        pd.DataFrame(metrics_data).to_csv(metrics_file, mode='a', header=False, index=False)

    return epoch


def add_record_metrics_v2(epoch=-1,
                          config_id='',
                          seed=0,
                          train_acc=0.0,
                          train_loss=0.0,
                          val_acc=0.0,
                          val_loss=0.0,
                          val_macro_f1=0.0,
                          val_weighted_f1=0.0,
                          lr=0.0,
                          gradient_norm=0.0,
                          time_spend=0,
                          state=1):
    """新增实验记录 v2：纯数值列 + config_id/seed，写入 src_v2/runs/metrics_record.csv。
    原 metrics.csv（旧 18 列）仍由 add_record_metrics 照常写入，二者并存互不影响。"""
    runs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs')
    os.makedirs(runs_dir, exist_ok=True)
    record_file = os.path.join(runs_dir, 'metrics_record.csv')

    lr = lr.cpu().item() if isinstance(lr, torch.Tensor) else float(lr)
    gradient_norm = gradient_norm.cpu().item() if isinstance(gradient_norm, torch.Tensor) else float(gradient_norm)

    row = {
        "config_id": [config_id],
        "seed": [seed],
        "epoch": [epoch],
        "train_acc": [train_acc],
        "train_loss": [train_loss],
        "val_acc": [val_acc],
        "val_loss": [val_loss],
        "val_macro_f1": [val_macro_f1],
        "val_weighted_f1": [val_weighted_f1],
        "lr": [lr],
        "gradient_norm": [gradient_norm],
        "time_spend_s": [time_spend],
        "state": [state],
        "save_time": [datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')],
    }
    if not os.path.isfile(record_file):
        pd.DataFrame(row).to_csv(record_file, index=False)
    else:
        pd.DataFrame(row).to_csv(record_file, mode='a', header=False, index=False)
    return epoch


# def generate_mappings(merged_dict):
#     class_labels = get_classnames()
#     # total_classes = len(class_labels)
#
#     id_to_main_class = {}
#     id_to_child_class = {}
#
#     # 遍历 merged_dict
#     for main_class, items in merged_dict.items():
#         for item_id, child_name in items:
#             id_to_main_class[str(item_id)] = f"公共类-{main_class}"
#             id_to_child_class[str(item_id)] = child_name
#
#     # 为未列出的类别设置默认值
#     # print()
#     # for item_id, class_name in enumerate(class_labels):
#     for item_id in range(len(class_labels)):
#         class_name = class_labels[item_id]
#         if str(item_id) not in id_to_main_class:
#             id_to_main_class[str(item_id)] = f"私有类-{class_name}"
#             id_to_child_class[str(item_id)] = class_name  # 子类名称与类别名称相同
#
#     # print(len(id_to_main_class))
#     # print(len(id_to_child_class))
#
#     # print(id_to_main_class)
#     # print(len(id_to_main_class.values()))
#
#     # main_class_to_index = {main_class: str(idx) for idx, main_class in enumerate(sorted(set(id_to_main_class.values())))}
#
#     # 使用整数索引而不是字符串
#     main_class_to_index = {}
#     for idx, main_class in enumerate(sorted(set(id_to_main_class.values()))):
#         main_class_to_index[main_class] = idx  # 使用整数索引
#
#     # print(main_class_to_index)
#     # print(len(list(enumerate(id_to_main_class.values()))))
#     # print(len(main_class_to_index))
#     return id_to_main_class, id_to_child_class, main_class_to_index

# 修改 generate_mappings 函数，确保与数据集排序一致
def generate_mappings(merged_dict):
    class_labels = get_classnames()
    id_to_main_class = {}  # id: 父类

    # 遍历 merged_dict
    for main_class in sorted(merged_dict.keys()):
        items = merged_dict[main_class]
        # main_class重新定义的公共父类
        for item_id, child_name in items:
            id_to_main_class[item_id] = f"公共类-{main_class}"
            # id_to_child_class[item_id] = child_name

    # print(f"temp_id_to_main_class, {id_to_main_class}", end="\n" + '*' * 100)

    # 为未列出的类别设置默认值
    # 按照数据集的顺序（字符串排序）处理类别
    dataset_classes = sorted([int(cls) for cls in os.listdir(os.path.join(_PROJECT_ROOT, 'train')) if cls.isdigit()])
    # print(f"{dataset_classes=}")

    for class_id in dataset_classes:
        if str(class_id) not in id_to_main_class:
            class_name = class_labels[class_id]
            id_to_main_class[str(class_id)] = f"私有类-{class_name}"
            # id_to_child_class[str(class_id)] = class_name

    # print(f"all_id_to_main_class, {id_to_main_class}", end="\n" + '*' * 100)

    # print(f"{id_to_child_class=}")

    # 将父类索引重新标写, 从"0"开始
    # 含公共类和私有类, 重新排序
    renew_class_to_index = {}
    sort_idx = sorted(set(id_to_main_class.values()))
    for idx, main_class in enumerate(sort_idx):
        renew_class_to_index[main_class] = str(idx)
    # print(f"start's {sorted(set(id_to_main_class.values()))=}")
    # print(f"{len(renew_class_to_index)=}")
    # print(f"{id_to_main_class=}")
    # print(f"{sorted(set(id_to_main_class.values()))=}")
    # print(f"{set(id_to_main_class.values())=}")
    # print(f"{len(set(id_to_main_class.values()))=}, {len(id_to_main_class.values())=}")
    # print(f"{main_class_to_index=}")

    # 创建子类到索引的映射（按主类分组）
    # child_class_to_index = {}
    # f = True
    # idx_public = 0
    # for main_class in main_class_to_index.keys():
    #     # 获取该主类下的所有子类
    #     children = [
    #         child for class_id, child in id_to_child_class.items()
    #         if id_to_main_class[class_id] == main_class
    #     ]
    #     # print(f"{children=}") if f else ...
    #     f = False
    #     # 为每个子类分配索引
    #     for idx, child in enumerate(set(children)):
    #         if main_class.split("-")[0] == '私有类':
    #             child_key = main_class
    #         else:
    #             child_key = f"{main_class}___{child}"
    #         child_class_to_index[child_key] = idx_public
    #         idx_public += 1

    # print(f"{child_class_to_index=}")
    return id_to_main_class, renew_class_to_index

    # return id_to_main_class, id_to_child_class, main_class_to_index, child_class_to_index


def get_class_proportion(pq='p'):
    """
    :param pq:
        p: percentage 返回百分比
        q: quantity   返回数量
    :return:
    """
    # class_weights = torch.tensor([0.1, 0.9]).to(device)
    class_name = get_classnames()
    # print(class_name)

    path1 = os.path.join(_PROJECT_ROOT, "train")  # 输入一级文件夹地址
    files1 = os.listdir(path1)  # 读入一级文件夹
    num1 = len(files1)  # 统计一级文件夹中的二级文件夹个数
    # num2 = []  # 建立空列表
    file_and_num = []
    all_files_number = 0
    for i in range(num1):  # 遍历所有二级文件夹
        path2 = path1 + '//' + files1[i]  # 某二级文件夹的路径
        try:
            files2 = os.listdir(path2)  # 读入二级文件夹
            all_files_number += len(files2)
            name_one = class_name[int(files1[i])]
        except NotADirectoryError:
            continue
        # num2.append(len(files2))  # 二级文件夹中的文件个数
        file_and_num.append([files1[i], name_one, len(files2)])
    # print(file_and_num)
    # 计算类别权重
    if pq == 'p':
        for entry in file_and_num:
            entry.append(entry[-1] / all_files_number)  # 样本比例

    # _ = [file_and_num[index].append(key[-1] / all_files_number) for index, key in enumerate(file_and_num)]
    class_weights = torch.tensor([value[-1] for value in file_and_num])
    return class_weights


# def get_class_weight(main_class_to_index, id_to_child_class, merged_dict_user=None):
#     """
#     :param main_class_to_index:
#         {'公共类-其他垃圾-抹布毛巾': 0, '公共类-厨余垃圾-哈密瓜': 1, ...,
#          '私有类-其他垃圾-U型回形针': 5, '私有类-其他垃圾-一次性杯子': 6, ...}
#     :param id_to_child_class:
#         {57: '玻璃器皿', 56: '玻璃瓶', 55: '玻璃壶', 59: '玻璃制品',
#         117: '金属罐', 118: '金属制品', 26: '哈密瓜', 49: '哈密瓜2',
#         213: '厨房抹布', 232: '毛巾',
#         24: '果壳', 225: '果壳',
#         0: '厨余垃圾-八宝粥', 1: '厨余垃圾-冰激凌', ...}
#     :param merged_dict_user:
#         见__init__.py中, 形如merged_dict
#     :return: child_weight, parent_weight
#     """
#     if merged_dict_user is None:
#         merged_dict_user = merged_dict
#     all_numbers = get_class_labels_number(sortYN=False)
#     child_numbers = []
#     child_weight = []
#     for child in id_to_child_class:
#         child_numbers.append(all_numbers[int(child)][-1])
#
#     sum_child_number = sum(child_numbers)
#
#     for n in child_numbers:
#         child_weight.append(n / sum_child_number)
#
#     # print(all_numbers)
#     all_numbers_copy = deepcopy(all_numbers)
#     parent_numbers = []
#     for parent, id_ in main_class_to_index.items():
#         # print()
#         # print(parent)
#         label_name = parent[4:]
#         if parent.startswith('公共类'):
#
#             parent_numbers.append(0)
#             for child_for_parent in merged_dict[label_name]:
#                 parent_numbers[-1] += all_numbers[child_for_parent[0]][-1]
#         else:
#             for index, label_one in enumerate(all_numbers_copy):
#                 if label_one[1].startswith(label_name):
#                     parent_numbers.append(label_one[-1])
#                     all_numbers_copy.pop(index)
#                     break
#     sum_parent_number = sum(parent_numbers)
#     parent_weight = []
#     for n in parent_numbers:
#         parent_weight.append(n / sum_parent_number)
#
#     # print(child_weight, parent_weight)
#     # print(len(child_weight), len(parent_weight))
#
#     return child_weight, parent_weight
#
#     # print(child_weight, sum(child_weight))


def get_class_weight(
        renew_class_to_index,
        # id_to_child_class,
        id_to_main_class,
        merged_dict_user=None
):
    """
    :param main_class_to_index:
        {'公共类-其他垃圾-抹布毛巾': 0, '公共类-厨余垃圾-哈密瓜': 1, ...,
         '私有类-其他垃圾-U型回形针': 5, '私有类-其他垃圾-一次性杯子': 6, ...}
    :param id_to_child_class:
        {57: '玻璃器皿', 56: '玻璃瓶', 55: '玻璃壶', 59: '玻璃制品',
        117: '金属罐', 118: '金属制品', 26: '哈密瓜', 49: '哈密瓜2',
        213: '厨房抹布', 232: '毛巾',
        24: '果壳', 225: '果壳',
        0: '厨余垃圾-八宝粥', 1: '厨余垃圾-冰激凌', ...}
    :param id_to_main_class:
        {'57': '公共类-可回收物-玻璃制品类', '56': '公共类-可回收物-玻璃制品类', ...}
    :param merged_dict_user:
        见__init__.py中, 形如merged_dict
    :return: child_weight, parent_weight
    """
    if merged_dict_user is None:
        merged_dict_user = merged_dict

    dataset_path = '../train'
    dataset_classes = sorted([int(f) for f in os.listdir(dataset_path) if f.isdigit()])
    print(f"{dataset_classes=}")
    # 获取所有类别的样本数量信息
    all_numbers = get_class_labels_number()
    # print(f"{all_numbers=}")
    # print(f"{id_to_main_class=}")

    # 创建一个从类别ID到样本数量的映射
    id_to_count = {}
    for real_id, _name, number in all_numbers:
        # class_id = int(real_id)  # 文件夹名就是类别ID
        # count = item[2]  # 样本数量
        id_to_count[real_id] = number
    # print(f"{id_to_count=}")

    # # 计算子类权重
    # child_numbers = []
    # for child_id in dataset_classes:
    #     if child_id in id_to_count:
    #         child_numbers.append(id_to_count[child_id])
    #     else:
    #         # 如果找不到该子类的样本数量，使用默认值1
    #         print(f"警告: 找不到子类 {child_id} 的样本数量，使用默认值1")
    #         child_numbers.append(1)
    #
    # # 使用逆频率计算权重（样本数越少，权重越大）
    # sum_child_number = sum(child_numbers)
    # child_weight = [sum_child_number / n for n in child_numbers]

    # 计算父类权重
    parent_numbers = []
    parent_to_public = {}
    merged_child_ids = []
    private_child_ids = []

    # 处理公共类（来自merged_dict）
    for parent_name, children_list in merged_dict_user.items():
        full_parent_name = f"公共类-{parent_name}"
        child_ids = [child[0] for child in children_list]  # 提取子类ID
        merged_child_ids.extend(child_ids)
        parent_to_public[full_parent_name] = child_ids
    # print(f"{parent_to_public=}")
    private_child_ids = list(id_to_count.keys())
    # private_child_ids = list(id_to_count.keys()).remove(merged_child_ids)
    for i in merged_child_ids:
        private_child_ids.remove(i)

    # # 为每个私有类创建一个独立的父类
    for child_id in private_child_ids:
        # 从id_to_main_class获取父类名称
        parent_name = id_to_main_class.get(str(child_id), f"私有类-{child_id}")
        parent_to_public[parent_name] = [child_id]

    # print(f"{parent_to_public=}")

    # 计算每个父类的样本数量
    for parent_name in renew_class_to_index:
        total_count = 0
        if parent_name in parent_to_public:
            for child_id in parent_to_public[parent_name]:
                total_count += id_to_count[child_id]
        else:
            print(f"异常类{parent_name}没有数量!")
            total_count = 0

        parent_numbers.append(total_count)

    # 使用逆频率计算父类权重
    sum_parent_number = sum(parent_numbers)
    parent_weight = [sum_parent_number / n for n in parent_numbers]

    # 归一化权重（可选，但通常有助于训练稳定性）
    # child_weight = np.array(child_weight) / np.sum(child_weight)
    parent_weight = np.array(parent_weight) / np.sum(parent_weight)

    # return child_weight.tolist(), parent_weight.tolist()
    return parent_weight.tolist()


def hierarchical_collate_fn(batch):
    """
    自定义 collate 函数，处理层级标签
    """
    images = []
    parent_labels = []
    # child_labels = []

    for item in batch:
        # print(f"batch={batch}")
        images.append(item[0])
        # parent_labels.append(str(item[1][0]))  # 父类索引
        parent_labels.append(str(item[1]))  # 父类索引
        # child_labels.append(str(item[1][1]))  # 子类索引

    # 将列表转换为张量
    images = torch.stack(images, 0)
    # print(f"{parent_labels=}")
    # print(f"{child_labels=}")
    # parent_labels = [int(i) for i in parent_labels]
    # print(f"{parent_labels=}")
    # print(f"{child_labels=}")
    # parent_labels = torch.tensor(parent_labels, dtype=torch.long)
    # child_labels = torch.tensor(child_labels, dtype=torch.long)
    # 将字符串转换为整数张量（仅用于损失计算）
    parent_labels_int = torch.tensor([int(i) for i in parent_labels], dtype=torch.long)
    # child_labels_int = torch.tensor([int(i) for i in child_labels], dtype=torch.long)

    # print(f"{parent_labels_int=}")
    # print(f"{child_labels_int=}")

    # return images, (parent_labels_int, child_labels_int)
    return images, parent_labels_int
    # return images, (parent_labels, child_labels)
    # return images, (parent_labels_int, child_labels_int), (parent_labels, child_labels)


def count_by_category(class_data):
    """
    统计每个大类别下的物品数量和图片总数

    参数:
        class_data: 通过get_class_labels_number()获取的二维数组

    返回:
        dict: 包含每个大类别统计结果的字典
    """
    category_stats = {}

    for item in class_data:
        # 从类别名称中提取大类别（如"可回收物"、"厨余垃圾"等）
        category = item[1].split('-')[0]
        item_name = item[1].split('-')[1] if '-' in item[1] else item[1]
        count = item[2]

        # 如果该大类别还未在字典中，初始化
        if category not in category_stats:
            category_stats[category] = {
                'item_count': 0,
                'image_count': 0,
                'items': []
            }

        # 更新统计信息
        category_stats[category]['item_count'] += 1
        category_stats[category]['image_count'] += count
        category_stats[category]['items'].append({
            'id': item[0],
            'name': item_name,
            'count': count
        })

    return category_stats


def should_update_simple(csv_file="metrics.csv",
                         n=UPDATE_N,
                         acc_col='Validation_Accuracy_Parent',
                         incr_update=0.15):
    df = pd.read_csv(csv_file)
    if len(df) < 3*n+2:
        return True
    # 2. 比较最后一行与倒数第31行（k-30）的 'state'
    last_state = df['state'].iloc[-1]
    comparison_state = df['state'].iloc[-(5*n+1)]
    if last_state != comparison_state:
        # 如果状态不一致，说明在过去30轮内发生过状态切换。
        # print(f"DEBUG: 最近30轮内状态已改变 (从 {comparison_state} -> {last_state})，处于宽限期，不进行解冻判断。")
        return True  # 返回 True 表示“不解冻”，给新状态适应时间

    last_n = df.tail(n)[acc_col].values
    last_3n = df.tail(3 * n)[acc_col].values

    # 条件1：连续递增量检查
    # incrs = [(last_n[i + 1] - last_n[i]) / last_n[i] for i in range(len(last_n) - 1)]
    # if incrs and sum(incr < incr_update for incr in incrs) >= len(incrs) // 2:
    #     return False

    deltas = [(last_n[i + 1] - last_n[i]) for i in range(len(last_n) - 1)]

    # 你的逻辑：如果一半以上的增量 < incr_update → 无提升 → 返回 False
    if deltas and sum(delta < incr_update for delta in deltas) >= len(deltas) // 2:
        return False

    # 条件2：最佳准确率对比
    best_acc = np.max(last_3n) * 0.99
    avg_acc = np.mean(last_n)
    if avg_acc < best_acc:
        return False

    # 条件3：平均提升检查
    # if len(last_n) > 2:
    #     trimmed = np.sort(last_n)[1:-1]
    #     mean_imp = np.mean([trimmed[i + 1] - trimmed[i] for i in range(len(trimmed) - 1)])
    #     if mean_imp <= 0:
    #         return False
    values = last_n.copy()
    max_idx = np.argmax(values)
    min_idx = np.argmin(values)
    mask = np.ones_like(values, dtype=bool)
    mask[max_idx] = False
    mask[min_idx] = False
    trimmed = values[mask]

    if len(trimmed) > 1:
        diffs = [trimmed[i + 1] - trimmed[i] for i in range(len(trimmed) - 1)]
        mean_imp = np.mean(diffs)
        if mean_imp <= 0:
            return False

    return True


if __name__ == '__main__':
    # print(generate_mappings(merged_dict))
    # 26
    # class_data = get_class_labels_number()
    # stats = count_by_category(class_data)
    #
    # # 打印统计结果
    # for category, data in stats.items():
    #     print(f"类别: {category}")
    #     print(f"物品数量: {data['item_count']}")
    #     print(f"图片总数: {data['image_count']}")
    # print("包含物品:")
    # for item in data['items']:
    #     print(f"  ID:{item['id']} {item['name']}: {item['count']}张")
    # print("-" * 50)
    id_to_main_class, renew_class_to_index = generate_mappings(merged_dict=merged_dict)

    print(f"{id_to_main_class=}")
    print(f"{len(id_to_main_class)=}")
    print(f"{renew_class_to_index=}")
    print(f"{len(renew_class_to_index)=}")

    # print(id_to_main_class)
    # print(len(id_to_main_class))
    # print(len(id_to_child_class))
    # # print(len(get_classnames()))
    # file_and_num = get_class_proportion(pq='p')
    # print(file_and_num, type(file_and_num))
    # print(get_class_proportion())
    # print(get_class_labels_number())
    # print(len(main_class_to_index))
    # print(get_classnames())
    # id_to_main_class, id_to_child_class, main_class_to_index, child_class_to_index = generate_mappings(
    #     merged_dict=merged_dict)
    # print(f"{id_to_main_class=}", end='\n' * 5)
    # print(f"{id_to_child_class=}", end='\n' * 5)
    # print(f"{main_class_to_index=}", end='\n' * 5)
    # print(f"{child_class_to_index=}", end='\n' * 5)
    # print(f"{len(main_class_to_index)}")
    # print(f"{len(child_class_to_index)}")
    # # width = get_class_weight(main_class_to_index=main_class_to_index,
    # #                  id_to_child_class=id_to_child_class,
    # #                  merged_dict_user=merged_dict)
    # print(f"{id_to_main_class=}", f"{id_to_child_class=}", f"{main_class_to_index=}", sep=f"{'=' * 100}\n")
    # print(id_to_child_class)
    # file = '../train'
    # # file = '../val'
    # mean, std, skip_number = calculate_mean_and_std(file, retrieve_file=False, update=True)
    # print('Filtered Mean:', mean)
    # print('Filtered Std:', std)
    # print(f"略过偏色图片数量: {skip_number}")
