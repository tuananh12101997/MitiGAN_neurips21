import csv
import os

import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from PIL import Image


class ColorDepthShrinking(object):
    def __init__(self, c=3):
        self.t = 1 << int(8 - c)

    def __call__(self, img):
        im = np.asarray(img)
        im = (im / self.t).astype("uint8") * self.t
        img = Image.fromarray(im.astype("uint8"))
        return img

    def __repr__(self):
        return self.__class__.__name__ + "(t={})".format(self.t)


class Smoothing(object):
    def __init__(self, k=3):
        self.k = k

    def __call__(self, img):
        im = np.asarray(img)
        im = cv2.GaussianBlur(im, (self.k, self.k), 0)
        img = Image.fromarray(im.astype("uint8"))
        return img

    def __repr__(self):
        return self.__class__.__name__ + "(k={})".format(self.k)


def get_transform(opt, train=True, c=0, k=0):
    transforms_list = []
    transforms_list.append(transforms.Resize((opt.input_height, opt.input_width)))
    if train:
        transforms_list.append(
            transforms.RandomCrop(
                (opt.input_height, opt.input_width), padding=opt.random_crop
            )
        )
        if opt.dataset != "mnist":
            transforms_list.append(transforms.RandomRotation(opt.random_rotation))

        if opt.dataset == "cifar10":
            pass
    if c > 0:
        transforms_list.append(ColorDepthShrinking(c))
    if k > 0:
        transforms_list.append(Smoothing(k))

    transforms_list.append(transforms.ToTensor())
    if opt.dataset == "cifar10":
        transforms_list.append(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
    elif opt.dataset == "gtsrb":
        transforms_list.append(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
    elif opt.dataset == "mnist":
        transforms_list.append(transforms.Normalize([0.5], [0.5]))
    elif opt.dataset == "TinyImageNet":
        transforms_list.append(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
    else:
        raise Exception("Invalid Dataset")
    return transforms.Compose(transforms_list)


class GTSRB(data.Dataset):
    def __init__(self, opt, train, transforms):
        super(GTSRB, self).__init__()
        if train:
            self.data_folder = os.path.join(opt.data_root, "GTSRB/Train")
            self.images, self.labels = self._get_data_train_list()
        else:
            self.data_folder = os.path.join(opt.data_root, "GTSRB/Test")
            self.images, self.labels = self._get_data_test_list()

        self.transforms = transforms

    def _get_data_train_list(self):
        images = []
        labels = []
        for c in range(0, 43):
            prefix = self.data_folder + "/" + format(c, "05d") + "/"
            gtFile = open(prefix + "GT-" + format(c, "05d") + ".csv")
            gtReader = csv.reader(gtFile, delimiter=";")
            next(gtReader)
            for row in gtReader:
                images.append(prefix + row[0])
                labels.append(int(row[7]))
            gtFile.close()
        return images, labels

    def _get_data_test_list(self):
        images = []
        labels = []
        prefix = os.path.join(self.data_folder, "GT-final_test.csv")
        gtFile = open(prefix)
        gtReader = csv.reader(gtFile, delimiter=";")
        next(gtReader)
        for row in gtReader:
            images.append(self.data_folder + "/" + row[0])
            labels.append(int(row[7]))
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image = Image.open(self.images[index])

        image = self.transforms(image)
        label = self.labels[index]
        return image, label


class TinyImageNet(data.Dataset):
    def __init__(self, opt, train, transforms):
        super(TinyImageNet, self).__init__()
        self.class_dict = self._get_class_name_dict(opt)
        self.train = train
        if train:
            self.data_folder = os.path.join(opt.data_root, "tiny-imagenet-200/train")
            self.images, self.labels = self._get_data_train_list()
        else:
            self.data_folder = os.path.join(opt.data_root, "tiny-imagenet-200/val")
            self.images, self.labels = self._get_data_test_list()
        self.transforms = transforms

    def _get_data_train_list(self):
        images = []
        labels = []
        class_dirs = os.listdir(self.data_folder)
        class_dirs = [os.path.join(self.data_folder, x, "images") for x in class_dirs]
        for class_dir in class_dirs:
            file_list = os.listdir(class_dir)
            # print(img.size)
            # print(os.path.join(self.data_folder, "images", line[0]))
            # img = Image.open(os.path.join(self.data_folder, "images", line[0]))
            # if img.size[0] != 3:
            #     continue
            file_list = [os.path.join(class_dir, x) for x in file_list]
            class_name = class_dir.split("/")[-2]
            class_id = self.class_dict[class_name]

            images += file_list
            labels += [class_id] * len(file_list)
            # import pdb; pdb.set_trace()

        return images, labels

    def _get_data_test_list(self):
        images = []
        labels = []
        annotation_path = os.path.join(self.data_folder, "val_annotations.txt")
        with open(annotation_path, "r") as f:
            contents = f.readlines()
            for line in contents:
                line = line.split()
                # img = Image.open(os.path.join(self.data_folder, "images", line[0]))
                # print(img.size)
                # print(os.path.join(self.data_folder, "images", line[0]))
                # if img.size[0] != 3:
                #     continue
                labels.append(self.class_dict[line[1]])
                images.append(os.path.join(self.data_folder, "images", line[0]))
        return images, labels

    def _get_class_name_dict(self, opt):
        winds_path = os.path.join(opt.data_root, "tiny-imagenet-200", "wnids.txt")
        with open(winds_path, "r") as f:
            contents = f.readlines()
        contents = {class_name.strip("\n"): i for i, class_name in enumerate(contents)}
        return contents

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image = Image.open(self.images[index])
        if image.size[0] != 3:
            image = image.convert("RGB")
        # to_tensor = transforms.ToTensor()
        # print(to_tensor(image).shape)

        image = self.transforms(image)
        label = self.labels[index]
        return image, label


def get_dataloader(opt, train=True, c=0, k=0, shuffle=True):
    transform = get_transform(opt, train, c=c, k=k)
    if opt.dataset == "gtsrb":
        dataset = GTSRB(opt, train, transform)
    elif opt.dataset == "mnist":
        dataset = torchvision.datasets.MNIST(
            opt.data_root, train, transform, download=True
        )
    elif opt.dataset == "cifar10":
        dataset = torchvision.datasets.CIFAR10(
            opt.data_root, train, transform, download=True
        )
    elif opt.dataset == "TinyImageNet":
        dataset = TinyImageNet(opt, train, transform)
        print("-------tiny ImageNet")
    else:
        raise Exception("Invalid dataset")
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batchsize, num_workers=opt.num_workers, shuffle=shuffle
    )
    return dataloader
