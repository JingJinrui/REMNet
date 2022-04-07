import os
import torch as t
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


class Shanghai_2020(Dataset):
    def __init__(self, root, seq_len=20, seq_interval=None, transforms=None, train=True, test=False, nonzero_points_threshold=None):
        self.root = root
        self.train = train
        self.test = test
        self.seq_len = seq_len
        self.seq_interval = seq_interval
        self.nonzero_points_threshold = nonzero_points_threshold

        self.examples_path = []
        if self.train:
            f = open(os.path.join(self.root, 'Shanghai_2020_train_examples.txt'), 'r')
        elif self.test:
            f = open(os.path.join(self.root, 'Shanghai_2020_test_examples.txt'), 'r')
        else:
            f = open(os.path.join(self.root, 'Shanghai_2020_valid_examples.txt'), 'r')
        for line in f.readlines():
            self.examples_path.append(line.split('\n')[0])
        f.close()

        if transforms is None:
            self.transforms = T.Compose([T.ToTensor()])
        else:
            self.transforms = transforms

    # output: example_imgs [seq_len, channels, height, width]
    def __getitem__(self, item):
        example_path = self.examples_path[item]
        example_index = example_path.split('\\')[-1]
        example_imgs = []
        f_inputs = open(os.path.join(example_path, example_index + '-inputs-train.txt'), 'r')
        f_targets = open(os.path.join(example_path, example_index + '-targets-train.txt'), 'r')
        for line in f_inputs.readlines():
            example_img_path = os.path.join(self.root, 'train', 'data', line.split('\n')[0])
            example_img = Image.open(example_img_path)
            example_img = example_img.resize((256, 256), Image.BILINEAR)
            example_img = self.transforms(example_img)
            example_imgs.append(example_img)
        for line in f_targets.readlines():
            example_img_path = os.path.join(self.root, 'train', 'data', line.split('\n')[0])
            example_img = Image.open(example_img_path)
            example_img = example_img.resize((256, 256), Image.BILINEAR)
            example_img = self.transforms(example_img)
            example_imgs.append(example_img)
        example_imgs = t.stack(example_imgs, dim=0)
        return example_imgs

    def __len__(self):
        return len(self.examples_path)


class Shanghai_2020_samples(Dataset):
    def __init__(self, root, transforms=None):
        test_samples = os.listdir(root)
        self.test_samples_path = [os.path.join(root, test_sample) for test_sample in test_samples]

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
