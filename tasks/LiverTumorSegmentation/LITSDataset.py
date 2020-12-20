#####################################################
# Copyright (C) 2020 Chae Eun Lee
#
# Author:   Chae Eun Lee
# Email:    nuguziii@cglab.snu.ac.kr
#####################################################

""" LiTS (Liver Tumor Segmentation Challenge) Dataset

download: https://competitions.codalab.org/competitions/17094
"""

import glob

from torch.utils.data import Dataset
import nibabel as nib
import numpy as np

from src.image_tools.augmentation import image_augmentation
from src.image_tools.image_processing import normalize, resize

class LITSDataset(Dataset):
    def __init__(self,
                 width,
                 height,
                 depth,
                 image_dir_path,
                 label_dir_path=None,
                 aug=[]):
        """
        :param aug: ['transformation', 'gaussian_noise', 'cutout', 'flip']
        """

        self.shape = (depth, width, height)
        self.image_dir_path = image_dir_path
        self.label_dir_path = label_dir_path
        self.aug = aug

        image_path_list = glob.glob(self.image_dir_path + '\*')
        self.num_images = len(image_path_list)

        self.data_list = {}

        for idx, image_path in enumerate(image_path_list):
            data = {}
            data['image_path'] = image_path

            if label_dir_path:
                label_path = image_path.replace("train", "label").replace("volume", "segmentation")

                data['label_path'] = label_path

            self.data_list[idx] = data

    def __len__(self):
        return self.num_images

    def __getitem__(self, index):

        # load image
        image_path = self.data_list[index]['image_path']
        image_file = nib.load(image_path)
        image = image_file.get_data()

        # load label
        if self.label_dir_path:
            label_path = self.data_list[index]['label_path']
            label = nib.load(label_path).get_data()

        image_shape = image.shape

        # preprocessing
        image = image.transpose((-1, 0, 1))
        image = resize(image.astype(float), self.shape, is_labeled=False)

        if self.label_dir_path:
            label = label.transpose((-1, 0, 1))
            label = resize(label.astype(int), self.shape, is_labeled=True)

        # augmentation
        if self.label_dir_path:
            image, label = image_augmentation(image, label, self.aug)
            label = label.astype(np.uint8)
        else:
            image = image_augmentation(image, aug=self.aug)

        image = normalize(image, -340, 360)

        if self.label_dir_path:
            return {'image_path': self.data_list[index]['image_path'],
                    'image': image,
                    'original_size': image_shape,
                    'label': label}
        else:
            return {'image_path': self.data_list[index]['image_path'],
                    'image': image,
                    'original_size': image_shape}

if __name__ == '__main__':
    data = LITSDataset(64, 64, 128,
                       'E:\Dataset_Medical\LiTS\\train',
                       'E:\Dataset_Medical\LiTS\\label',
                       aug=[])

    item = data.__getitem__(0)
    print(item['image_path'], item['label'].shape, item['image'].shape, item['original_size'])

    from src.utils.io import save_image_to_nib

    save_image_to_nib(item['label'].transpose(1, 2, 0).astype(np.uint8), './', 'aug' + item['image_path'][-6:-4])