import os
import torch as t
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


class HKO_7(Dataset):
    def __init__(self, root, seq_len=35, seq_interval=5, transforms=None, train=True, test=False, nonzero_points_threshold=None):
        self.root = root
        self.train = train
        self.test = test
        self.seq_len = seq_len
        self.seq_interval = seq_interval
        self.nonzero_points_threshold = nonzero_points_threshold
        self.rainy_days_imgs_indexs = []
        if self.train:
            f = open(os.path.join(self.root, 'hko7_rainy_train_days_statistics.txt'), 'r')
        elif self.test:
            f = open(os.path.join(self.root, 'hko7_rainy_test_days_statistics.txt'), 'r')
        else:
            f = open(os.path.join(self.root, 'hko7_rainy_valid_days_statistics.txt'), 'r')
        rainy_days = []
        for line in f.readlines():
            if self.nonzero_points_threshold is None:
                rainy_days.append(line.split(',')[0])
            else:
                nonzero_points = float(line.split(',')[1])
                if nonzero_points > self.nonzero_points_threshold:
                    rainy_days.append(line.split(',')[0])
        for rainy_day in rainy_days:
            rainy_day_root = os.path.join(self.root, 'radarPNG', rainy_day[0:4], rainy_day[4:6], rainy_day[6:8])
            rainy_day_imgs_length = len(os.listdir(rainy_day_root))
            for i in range(0, rainy_day_imgs_length - self.seq_len + 1, self.seq_interval):
                self.rainy_days_imgs_indexs.append([i, rainy_day_root])
        f.close()

        if transforms is None:
            self.transforms = T.Compose([T.ToTensor()])
        else:
            self.transforms = transforms

    # output: rainy_day_sequence_imgs [seq_len, channels, height, width]
    def __getitem__(self, item):
        index = self.rainy_days_imgs_indexs[item][0]
        rainy_day_root = self.rainy_days_imgs_indexs[item][1]
        rainy_day_imgs_paths = [os.path.join(rainy_day_root, img) for img in os.listdir(rainy_day_root)]
        rainy_day_sequence_imgs = []
        for i in range(self.seq_len):
            rainy_day_img = Image.open(rainy_day_imgs_paths[index + i])
            rainy_day_img = rainy_day_img.resize((256, 256), Image.BILINEAR)
            rainy_day_img = self.transforms(rainy_day_img)
            rainy_day_sequence_imgs.append(rainy_day_img)
        rainy_day_sequence_imgs = t.stack(rainy_day_sequence_imgs, dim=0)
        return rainy_day_sequence_imgs

    def __len__(self):
        return len(self.rainy_days_imgs_indexs)


class HKO_7_samples(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        test_samples = os.listdir(self.root)
        self.test_samples_path = [os.path.join(self.root, test_sample) for test_sample in test_samples]

        if transforms is None:
            self.transforms = T.Compose([T.ToTensor()])
        else:
            self.transforms = transforms

    # output: test_sample_sequence_imgs [seq_len, channels, height, width]
    def __getitem__(self, item):
        test_sample_path = self.test_samples_path[item]
        test_sample_imgs_paths = [os.path.join(test_sample_path, img) for img in os.listdir(test_sample_path)]
        test_sample_sequence_imgs = []
        for i in range(len(test_sample_imgs_paths)):
            test_sample_img = Image.open(test_sample_imgs_paths[i])
            test_sample_img = test_sample_img.resize((256, 256), Image.BILINEAR)
            test_sample_img = self.transforms(test_sample_img)
            test_sample_sequence_imgs.append(test_sample_img)
        test_sample_sequence_imgs = t.stack(test_sample_sequence_imgs, dim=0)
        return test_sample_sequence_imgs

    def __len__(self):
        return len(self.test_samples_path)
