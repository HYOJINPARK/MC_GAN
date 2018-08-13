import numpy as np
import torch
import random
from PIL import Image

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size, interpolation=Image.BILINEAR):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        self.interpolation = Image.BILINEAR

    def __call__(self, sample):
        image, seg = sample['image'], sample['seg']

        w, h = image.size
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = image.resize((new_h, new_w), self.interpolation)
        seg_img = seg.resize((new_h, new_w), Image.NEAREST)

        return {'image': img, 'seg': seg_img}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, seg = sample['image'], sample['seg']

        w, h = image.size
        th, tw = self.output_size
        if w == tw and h == th:
            return {'image': image, 'seg': seg}

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        image = image.crop((x1, y1, x1 + tw, y1 + th))
        seg = seg.crop((x1, y1, x1 + tw, y1 + th))

        return {'image': image, 'seg': seg}


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __call__(self, sample):
        img, seg = sample['image'], sample['seg']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            seg = seg.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img, 'seg': seg}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, seg = sample['image'], sample['seg']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'seg': torch.from_numpy(seg)}