import numpy as np
from scipy import ndimage
from skimage import measure, morphology
from skimage.transform import resize as r

def resize(image, shape, is_labeled=False):
    if is_labeled:
        return r(image.astype(int), shape, anti_aliasing=False, order=0, preserve_range=True)
    else:
        return r(image.astype(float), shape)

def normalize(data, min, max):
    ptp = max - min
    nimage = (data - min) / ptp
    nimage = np.clip(nimage, 0, 1)

    return nimage.astype(np.float32)

def get_gradient_image(data):
    sx = ndimage.filters.prewitt(data.astype(float), axis=0)
    sy = ndimage.filters.prewitt(data.astype(float), axis=1)
    sz = ndimage.filters.prewitt(data.astype(float), axis=2)
    return np.sqrt(sx**2 + sy**2 + sz**2).astype(data.dtype)

def get_edge_image(data):
    inside = np.empty_like(data)
    for z in range(data.shape[0]):
        inside[z] = ndimage.binary_erosion(data[z]).astype(data.dtype)
    return data - inside

def CCL(image_BW):
    image_labeled = measure.label(image_BW, background=0)
    areas = [r.area for r in measure.regionprops(image_labeled)]
    if len(areas) >= 2:
        areas.sort()
        maxArea = areas[len(areas) - 1]
        result = morphology.remove_small_objects(image_labeled, maxArea - 1)

        result = result > 0
        result = result.astype('uint8')

        return result
    return image_BW