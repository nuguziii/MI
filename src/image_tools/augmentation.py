import numpy as np
import scipy.ndimage as ni

from src.utils.random_function import get_random
from src.image_tools.transformation_matrix import generate_random_rotation_around_axis, generate_random_shear

def image_augmentation(image, label=None, aug=[]):
    '''
    :param image: DHW
    :param aug: ['transformation', 'gaussian_noise', 'cutout', 'flip' ...]
    :return: image, label
    '''
    if 'transformation' in aug:
        if get_random() > 0.3:
            if label is not None:
                image, label = random_affine_transformation(image, label, axis=(1, 0, 0), rotation_angle=20,
                                                            shear_angle=5)
            else:
                image = random_affine_transformation(image, axis=(1, 0, 0), rotation_angle=20, shear_angle=5)

    if 'gaussian_noise' in aug:
        image = add_gaussian_noise(image)

    if 'cutout' in aug:
        image = cutout(image)

    if 'flip' in aug:
        if label is not None:
            image, label = random_flip(image, label)
        else:
            image = random_flip(image)

    if label is not None:
        return image, label
    else:
        return image

def random_flip(image, label=None):
    if get_random() > 0.5:
        if label is not None:
            return np.fliplr(image), np.fliplr(label)
        else:
            return np.fliplr(image)


def random_affine_transformation(image, label=None, axis=(0,0,0), rotation_angle=0, shear_angle=0 , cval=-1024.0):
    mat = None
    if ((axis[0] + axis[1] + axis[2]) > 0) and (rotation_angle > 0):
        mat = generate_random_rotation_around_axis(axis, rotation_angle)

    if (shear_angle > 0):
        if mat is not None:
            mat = mat * generate_random_shear(shear_angle)
        else:
            mat = generate_random_shear(shear_angle)

    if mat is None:
        if label is not None:
            return image, label
        else:
            return image

    if label is not None:
        return ni.affine_transform(image, matrix=mat, cval=cval), np.round_(ni.affine_transform(label, matrix=mat))
    else:
        return ni.affine_transform(image, matrix=mat, cval=cval)

def add_gaussian_noise(img):
    mean = 0
    var = 0.01

    sigma = var ** 0.5
    gaussian = np.random.normal(mean, sigma, (img.shape[0], img.shape[1], img.shape[2]))
    noise_img = img + gaussian

    return noise_img

def cutout(data):
    data_type = data.dtype

    mask = np.ones((data.shape[0], data.shape[1], data.shape[2]), np.float32)

    n_holes = 1
    # if get_random() > 0.5:
    #     n_holes = 2

    # set range to width/5 ~ width/3
    len_plane = int(data.shape[2]/5) + int(get_random() * (data.shape[2]/4 - data.shape[2]/5))
    # set range to depth/5 ~ depth/3
    len_depth = int(data.shape[0]/5) + int(get_random() * (data.shape[0]/4 - data.shape[0]/5))

    for n in range(n_holes):
        # x = np.random.randint(data.shape[2])
        # y = np.random.randint(data.shape[1])
        # z = np.random.randint(data.shape[0])
        x = int(get_random() * data.shape[2])
        y = int(get_random() * data.shape[1])
        z = int(get_random() * data.shape[0])

        x1 = np.clip(x-len_plane//2, 0, data.shape[2])
        x2 = np.clip(x+len_plane//2, 0, data.shape[2])
        y1 = np.clip(y-len_plane//2, 0, data.shape[1])
        y2 = np.clip(y+len_plane//2, 0, data.shape[1])
        z1 = np.clip(z-len_depth//2, 0, data.shape[0])
        z2 = np.clip(z+len_depth//2, 0, data.shape[0])

        mask[z1:z2, y1:y2, x1:x2] = 0.

    data = data * mask

    return data.astype(data_type)


