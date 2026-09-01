import os
import json
import sys
import cv2  # <--- 导入 OpenCV
import numpy as np # <--- 导入 NumPy
from PIL import Image
from torchvision import datasets
from torchvision.datasets.folder import IMG_EXTENSIONS


class SafeImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 train=True,
                 id_to_main_class=None,
                 renew_class_to_index=None,
                 ):
        super(SafeImageFolder, self).__init__()  # <-- 调用基类的构造函数

        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        # --- 手动执行 ImageFolder 的核心逻辑 ---
        classes, class_to_idx = self._find_classes(self.root)
        samples = self._make_dataset(self.root, class_to_idx, IMG_EXTENSIONS)

        if len(samples) == 0:
            raise RuntimeError(f"Found 0 files in subfolders of: {self.root}")
        # self.loader = default_loader  # <-- 直接使用导入的 default_loader
        self.extensions = IMG_EXTENSIONS

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples  # <-- 此时是未经缓存过滤的完整列表
        self.targets = [s[1] for s in samples]
        # --- 手动执行结束 ---
        if train:
            cache_file = "images_cache_train.json"
        else:
            cache_file = "images_cache_val.json"
        self.cache_file = cache_file

        self.id_to_main_class = id_to_main_class
        self.renew_class_to_index = renew_class_to_index
        # 现在 self.samples 已经存在，可以构建缓存了
        self.valid_samples = self._load_or_build_cache()

        # 重写 samples，使用过滤后的有效样本
        self.samples = self.valid_samples
        self.imgs = self.samples
        self.targets = [s[1] for s in self.samples]

    # --- 将 ImageFolder 的内部方法复制过来 ---
    def _find_classes(self, dir):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _make_dataset(self, directory, class_to_idx, extensions):
        instances = []
        directory = os.path.expanduser(directory)
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    if fname.lower().endswith(extensions):
                        path = os.path.join(root, fname)
                        item = (path, class_index)
                        instances.append(item)
        return instances

    # --- 为了兼容性，添加一个简单的静态方法来定义支持的扩展名 ---
    @staticmethod
    def get_extensions():
        return ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.jfif')

    # --- _is_image_valid 方法保持不变，因为它使用 PIL.Image.verify()，这依然是最好的选择 ---
    def _is_image_valid(self, path):
        try:
            with Image.open(path) as img:
                img.verify()
            return True
        except (OSError, ValueError, Image.DecompressionBombError):  # 添加更多可能的异常
            return False

    def _load_or_build_cache(self):
        """
        加载或构建有效图片缓存。
        :return: List[(path, class_idx)]
        """
        cache_path = os.path.join(os.path.dirname(self.root), self.cache_file)
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                valid_samples_relative = json.load(f)
            # 将相对路径转换回绝对路径进行检查
            valid_samples = [
                (os.path.join(self.root, rel_path), class_idx)
                for rel_path, class_idx in valid_samples_relative
                if os.path.exists(os.path.join(self.root, rel_path))
            ]
        else:
            print(f"Building valid image cache for dataset at {self.root}...")
            valid_samples = []
            # 注意：这里的 self.samples 是未经缓存过滤的完整列表
            for path, class_idx in self.samples:
                if self._is_image_valid(path):
                    valid_samples.append((path, class_idx))
            # 保存缓存时，保存相对路径以增加可移植性
            valid_samples_relative = [
                (os.path.relpath(path, self.root), class_idx)
                for path, class_idx in valid_samples
            ]
            with open(cache_path, "w") as f:
                json.dump(valid_samples_relative, f)

        return valid_samples

    def __getitem__(self, index):
        exit_flag = 1
        while True:
            if exit_flag >= 100:
                print(f"异常! 尝试了 {exit_flag} 次仍未找到合适的图片! 程序已退出!")
                sys.exit()
            # self.samples 存储的是 (绝对路径, class_idx)
            path, target = self.samples[index]
            if not os.path.exists(path):
                print(f"路径不存在，跳过: {path}")
                index = (index + 1) % len(self.samples)
                exit_flag += 1
                continue
            try:
                # --- 主要修改点 ---
                # 1. 使用 OpenCV 读取图像，而不是 self.loader(PIL)
                image = cv2.imread(path)
                if image is None:  # 检查图像是否成功读取
                    raise IOError(f"cv2.imread failed to read image: {path}")
                # 2. 将 BGR 转换为 RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # 3. 应用 Albumentations 变换
                if self.transform is not None:
                    # Albumentations 的调用方式是关键字参数
                    augmented = self.transform(image=image)
                    sample = augmented['image']
                else:
                    # 如果没有变换，需要手动转为 tensor (如果需要的话)
                    # 但通常使用 Albumentations 时总会有一个 ToTensorV2 的变换
                    sample = image
                    # --- 修改结束 ---
                # 处理层级标签 (这部分逻辑保持不变)
                if self.id_to_main_class is not None:
                    class_id = self.classes[target]
                    class_idx_str = str(class_id)
                    main_class = self.id_to_main_class.get(class_idx_str)

                    if main_class is None:
                        main_class = f"公共类-{class_idx_str}"
                        print(f"警告: 类别 {class_idx_str} 没有主类映射，使用: {main_class}")
                    parent_index = self.renew_class_to_index[main_class]
                    # 注意：PyTorch 的 CrossEntropyLoss 通常需要 int 或 long 类型的标签
                    return sample, int(parent_index)
                # 如果没有映射，返回原始标签
                if self.target_transform is not None:
                    target = self.target_transform(target)

                return sample, target
            except Exception as e:  # 捕获更广泛的异常
                print(f"加载或处理图像时出错: {path}, 错误: {e}")
                exit_flag += 1
                index = (index + 1) % len(self.samples)

    # is_valid_file 和 _validate_label_consistency 方法保持不变
    # 但由于我们重写了构造函数，is_valid_file 不再被直接调用
    # make_dataset 会使用 self.extensions

    # 为了完整性，保留这些方法
    def _validate_label_consistency(self, main_class, child_class):
        pass  # 省略具体实现...

    def is_valid_file(self, path):
        return path.lower().endswith(self.get_extensions())

    def __len__(self):
        return len(self.samples)