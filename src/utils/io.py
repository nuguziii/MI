#####################################################
# author: Chae Eun Lee
# email: nuguziii@cglab.snu.ac.kr
#####################################################

import os

import numpy as np
import nibabel as nib

def save_image_to_nib(img, path, file_name):
    '''
    save numpy array to nii.gz file
    :param img: (H, W, D)
    :param path: directory
    '''
    img = nib.Nifti1Image(img, np.eye(4))
    nib.save(img, os.path.join(path, file_name+'.nii.gz'))

def print_raw(image, data_type, path):
    with open(path, "wb") as file:
        file.write(image.astype(data_type).tobytes('Any'))


def save_image_to_raw(img, path, file_name, endian_convert=False):
    outFile = open(os.path.join(path, '%s' % file_name + '.raw'), 'wb')
    if endian_convert:
        img = img.byteswap()
    outFile.write(img.tobytes())
    outFile.close()