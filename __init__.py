import logging
import os.path
from datetime import datetime

import torch
from torchvision import transforms
import numpy as np

# 如果模型准确度超过预定的值, 将会强制保存
ACC_STOP = 96
ERROR_RATE = 0.1

# 每n轮检查模型指标, 以判断是否需要更新模型层
UPDATE_N = 10

# ===== 输出路径：全部落到 src_v2/runs/（与工作目录无关） =====
_runs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs')
os.makedirs(_runs_dir, exist_ok=True)

pth_file_name = os.path.join(_runs_dir, 'garbage_classification_model.pth')
metrics_file = os.path.join(_runs_dir, "metrics.csv")
extensions_suffix = ('jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff', 'avif', '.jfif')

# merged_dict = {
#     '可回收物-玻璃制品类': [(57, '玻璃器皿'),
#                             (56, '玻璃瓶'),
#                             (55, '玻璃壶'),
#                             (59, '玻璃制品'),
#                             # (58, '玻璃球')
#                             ],
#     '可回收物-金属制品': [(117, '金属罐'), (118, '金属制品')],
#     '厨余垃圾-哈密瓜': [(26, '哈密瓜'), (49, '哈密瓜2')],
#     '其他垃圾-抹布毛巾': [(213, '厨房抹布'), (232, '毛巾')],
#     '厨余垃圾-果壳': [(24, '果壳'), (225, '果壳')]
# }
merged_dict = {
    '可回收物-玻璃制品类': [('57', '玻璃器皿'),
                            ('56', '玻璃瓶'),
                            ('55', '玻璃壶'),
                            ('59', '玻璃制品'),
                            ],
    '可回收物-金属制品': [('117', '金属罐'), ('118', '金属制品'), ('60', '不锈钢制品')],
    '厨余垃圾-哈密瓜': [('26', '哈密瓜'), ('49', '哈密瓜2')],
    '其他垃圾-抹布毛巾': [('213', '厨房抹布'), ('232', '毛巾')],
    '厨余垃圾-果壳': [('24', '果壳'), ('225', '果壳2')],
    '可回收物-鞋': [('183', '鞋'), ('184', '鞋子')],
    '其他垃圾-陶瓷': [('250', '陶瓷'), ('234', '破碎陶瓷'), ('209', '茶壶碎片')],
}


class DynamicMeanPad:
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, img):
        # 转换为 NumPy 数组
        img_array = np.array(img)

        # 计算每个通道的均值
        mean_values = img_array.mean(axis=(0, 1))  # (R_mean, G_mean, B_mean)

        # 计算图片尺寸和填充大小
        W, H = img.size
        target_height, target_width = self.target_size, self.target_size

        padding_top = (target_height - H) // 2
        padding_bottom = target_height - H - padding_top
        padding_left = (target_width - W) // 2
        padding_right = target_width - W - padding_left

        # 填充图片
        padding = (padding_left, padding_top, padding_right, padding_bottom)
        img = transforms.Pad(padding, fill=tuple(mean_values.astype(int)))(img)

        return img


def save_model_after_epoch(epoch,
                           model,
                           optimizer,
                           scheduler,
                           renew_class_to_index,
                           state,
                           other_pth_file_name=None,
                           config_id='',
                           seed=None):
    today = datetime.strftime(datetime.now(), '%Y-%m-%d')
    # 执行原始的训练函数
    checkpoint = {
        'epoch': epoch,
        'config_id': config_id,  # 实验标识（归档后可溯源）
        'seed': seed,
        'model': model,  # 保存整个模型对象
        'model_state_dict': model.state_dict(),  # 模型权重
        'scheduler_state_dict': scheduler.state_dict(),  # 修复：续跑恢复调度器状态
        'optimizer_state_dict': optimizer.state_dict(),
        'renew_class_to_index': renew_class_to_index,
        'today': today,
        'state': state
    }
    # *****************************************************
    # sorted_items = sorted(renew_class_to_index.items(), key=lambda x: int(x[1]))
    # print("Saving mapping (first 20):")
    # for k, v in sorted_items[:20]:
    #     print(v, "<-", k)
    # *****************************************************
    if other_pth_file_name is None:
        torch.save(checkpoint, pth_file_name)
    else:
        if not os.path.isfile(other_pth_file_name):
            torch.save(checkpoint, other_pth_file_name)
            s = f"特殊模型{other_pth_file_name}已保存, epoch {epoch} 完成!"
            print(s)
            logging.info(s)
    s = f"模型已保存, epoch {epoch} 完成!{today=}"
    print(s)
    logging.info(s)


# transforms_setting = transforms.Compose([
#     transforms.Resize([96, 96]),  # 设置图片大小
#     # 数据增强
#     transforms.RandomRotation(45),  # 随机旋转
#     transforms.CenterCrop(64),  # 从中心开始裁剪
#     transforms.RandomHorizontalFlip(p=0.5),  # 随机概率水平翻转, p为概率
#     transforms.RandomVerticalFlip(p=0.5),  # 随机概率垂直翻转, p为概率
#     transforms.ColorJitter(
#         brightness=0.2,  # 亮度
#         contrast=0.1,  # 对比度
#         saturation=0.1,  # 饱和度
#         hue=0.1,  # 色相
#     ),
#     transforms.RandomGrayscale(p=0.025),  # 概率转换为灰度图
#
#     # 这两件事必须做
#     transforms.ToTensor(),  # 转换为张量类型
#     transforms.Normalize([0.485, 0.456, 0.409], [0.229, 0.224, 0.225]),  # 均值, 标准差
# ])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
