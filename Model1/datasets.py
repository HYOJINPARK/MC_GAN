from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
import PIL
import os
import os.path
import pickle
import random
import numpy as np
import pandas as pd
from miscc.config import cfg

import torch.utils.data as data
from PIL import Image
import os
import os.path
import six
import string
import sys
import torch
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_imgs(img_path, seg_name, imsize, bbox=None,
             transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    seg = Image.open(seg_name).convert('L')
    temp_seg = np.array(seg) > 51
    temp_seg = (temp_seg * 255).astype(np.uint8)
    seg = Image.fromarray(temp_seg)
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]))
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])
        seg = seg.crop([x1, y1, x2, y2])

    if transform is not None:
        sample = {'image': img, 'seg': seg}
        sample = transform(sample)
    img = sample['image']
    seg = sample['seg']
    # img.show()
    # seg.show()
    ret_img, ret_seg= [], []
    for i in range(cfg.TREE.BRANCH_NUM):
        if i < (cfg.TREE.BRANCH_NUM - 1):
            re_img = transforms.Scale(imsize[i])(img)
            re_seg = transforms.Scale(imsize[i])(seg)
        else:
            re_img = img
            re_seg = seg
        ret_img.append(normalize(re_img))
        ret_seg.append(normalize(re_seg))

    return ret_img, ret_seg

def load_txt(dir):
    this_txt = open(dir, 'r')
    txt_line = this_txt.read()
    loc_split = txt_line.split(' ')
    next_loc = int(loc_split[1]) - 1
    return next_loc

class TextDataset(data.Dataset):
    def __init__(self, data_dir, split='train',
                 base_size=64, transform=None, target_transform=None):
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.target_transform = target_transform

        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2

        self.data = []
        self.data_dir = data_dir

        self.bbox = self.load_bbox()

        split_dir = os.path.join(data_dir,'stack_gan/', split)
        print(split_dir)
        self.filenames = self.load_filenames(split_dir)
        self.embeddings = self.load_embedding(split_dir)

        self.class_id = self.load_class_id(split_dir, len(self.filenames))
        self.captions = self.load_all_captions()

        if cfg.TRAIN.FLAG:
            self.iterator = self.prepair_training_pairs
        else:
            self.iterator = self.prepair_test_pairs

    def load_bbox(self):
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'cub/bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        #
        filepath = os.path.join(data_dir, 'cub/images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        print('Total filenames: ', len(filenames), filenames[0])
        #
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in xrange(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()

            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        #
        return filename_bbox

    def load_all_captions(self):
        def load_captions(caption_name):  # self,
            cap_path = caption_name
            with open(cap_path, "r") as f:
                captions = f.read().decode('utf8').split('\n')
            captions = [cap.replace("\ufffd\ufffd", " ")
                        for cap in captions if len(cap) > 0]
            return captions

        caption_dict = {}
        for key in self.filenames:
            caption_name = '%s/stack_gan/text/%s.txt' % (self.data_dir, key)
            captions = load_captions(caption_name)
            caption_dict[key] = captions
        return caption_dict

    def load_embedding(self, data_dir):

        embedding_filename = '/char-CNN-RNN-embeddings.pickle'

        with open(data_dir + embedding_filename, 'rb') as f:
            embeddings = pickle.load(f)
            embeddings = np.array(embeddings)
            # embedding_shape = [embeddings.shape[-1]]
            print('embeddings: ', embeddings.shape)
        return embeddings

    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f)
        else:
            class_id = np.arange(total_num)
        return class_id

    def load_filenames(self, data_dir):
        filepath = os.path.join(data_dir, 'filenames.pickle')
        with open(filepath, 'rb') as f:
            filenames = pickle.load(f)
        print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        return filenames


    def prepair_training_pairs(self, index):
        key = self.filenames[index]
        if self.bbox is not None:
            bbox = self.bbox[key]
            data_dir = '%s' % self.data_dir
        else:
            bbox = None
            data_dir = self.data_dir
        # captions = self.captions[key]
        embeddings = self.embeddings[index, :, :]
        img_name = '%s/cub/images/%s.jpg' % (data_dir, key)
        seg_name = '%s/cub/segmentations/%s.png' %(data_dir, key)
        (imgs, segs) = get_imgs(img_name, seg_name, self.imsize,
                        bbox, self.transform, normalize=self.norm)

        wrong_ix = random.randint(0, len(self.filenames) - 1)
        if(self.class_id[index] == self.class_id[wrong_ix]):
            wrong_ix = random.randint(0, len(self.filenames) - 1)
        wrong_key = self.filenames[wrong_ix]
        if self.bbox is not None:
            wrong_bbox = self.bbox[wrong_key]
        else:
            wrong_bbox = None
        wrong_img_name = '%s/cub/images/%s.jpg' % (data_dir, wrong_key)
        wrong_seg_name = '%s/cub/segmentations/%s.png' %(data_dir, wrong_key)
        wrong_imgs, wrong_segs = get_imgs(wrong_img_name,wrong_seg_name, self.imsize,
                              wrong_bbox, self.transform, normalize=self.norm)

        embedding_ix = random.randint(0, embeddings.shape[0] - 1)
        embedding = embeddings[embedding_ix, :]
        if self.target_transform is not None:
            embedding = self.target_transform(embedding)

        base_key = key.split('/')
        base_img_name = '%s/cub/base_images/%s' % (data_dir, base_key[0])

        return imgs, segs, wrong_imgs, wrong_segs, embedding, base_img_name # captions

    def prepair_test_pairs(self, index):
        key = self.filenames[index]
        if self.bbox is not None:
            bbox = self.bbox[key]
            data_dir = '%s' % self.data_dir
        else:
            bbox = None
            data_dir = self.data_dir
        # captions = self.captions[key]
        embeddings = self.embeddings[index, :, :]
        img_name = '%s/cub/images/%s.jpg' % (data_dir, key)
        seg_name = '%s/cub/segmentations/%s.png' % (data_dir, key)
        imgs, _ = get_imgs(img_name, seg_name, self.imsize,
                        bbox, self.transform, normalize=self.norm)

        if self.target_transform is not None:
            embeddings = self.target_transform(embeddings)

        base_key = key.split('/')
        base_img_name = '%s/cub/base_images/%s' % (data_dir, base_key[0])

        return imgs, embeddings, key, base_img_name # captions

    def __getitem__(self, index):
        return self.iterator(index)

    def __len__(self):
        return len(self.filenames)
