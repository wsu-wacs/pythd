"""
Code for working with image data
"""
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import cv2

def minmax_normalize_grayscale(image):
    minv = image.min()
    maxv = image.max()
    rng = maxv - minv
    if rng > 0.0:
        return (image - minv) / rng
    elif maxv > 0.0:
        return image / maxv
    else:
        return image - minv

def normalize_image(image, maxval=1.0, dtype=np.float32, minmax=True):
    """
    Normalize an image to have a given maximum value
    
    Parameters
    ----------
    image : numpy.ndarray
        The image to normalize
    maxval : int or float
        The maximum value of the normalized image
    dtype : numpy dtype
        The data type (np.uint8, np.float32, etc) of the normalized image
    minmax : bool
        Whether to use min-max normalization (True) or max normalization (False)
    
    Returns
    -------
    numpy.ndarray
        The normalized image
    """
    is_rgb = len(image.shape) > 2
    
    if is_rgb:
        result = image.astype(np.float32)
        for c in range(0, image.shape[2]):
            if minmax:
                result[:,:,c] = minmax_normalize_grayscale(result[:,:,c])
            else:
                result[:,:,c] = result[:,:,c] / result[:,:,c].max()
    else:
        if minmax:
            result = minmax_normalize_grayscale(image)
        else:
            result = image / image.max()
    
    return (maxval * result).astype(dtype)

def plot_images(images, dpi=300.0, **kwargs):
    """Show multiple images in one plot"""
    fig, ax = plt.subplots(1, len(images), dpi=dpi)

    if len(images) == 1:
        ax.imshow(images[0], **kwargs)
        ax.axis('off')
    else:
        for i, img in enumerate(images):
            ax[i].imshow(img, **kwargs)
            ax[i].axis("off")
    return (fig, ax)

def overlay_mask(image, mask, image_alpha=0.6, color=[1., 1., 0.]):
    """
    Overlay a binary mask on an image
    
    Parameters
    ----------
    image : numpy.ndarray
        The image to overlay the binary mask on
    mask : numpy.ndarray
        The binary mask
    image_alpha : float
        The alpha value to use for the original image when blending
    color : list or numpy.ndarray
        The color value (RGB) to use for the overlay.
        The default is yellow
    
    Returns
    -------
    numpy.ndarray
        The image with the binary mask overlaid on it
    """
    if len(image.shape) < 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image = normalize_image(image)

    color = np.array(color)
    cond = mask > 0
    image[cond] = image_alpha*image[cond] + (1.0 - image_alpha)*color
    
    return image

def overlay_rgb(image, mask, image_alpha=0.6):
    """
    Overlay one RGB image on another
    
    Parameters
    ----------
    image : numpy.ndarray
        The target image to overlay on
    mask : numpy.ndarray
        The RGB image to overlay
    image_alpha : float
        The alpha value to use for the original image when blending
    
    Returns
    -------
    numpy.ndarray
        The image with the RGB mask overlaid on it
    """
    if len(image.shape) < 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image = normalize_image(image)
    mask = mask.astype(float) / mask.max()
    
    cond = np.any(mask > 0, axis=2)
    image[cond] = image_alpha*image[cond] + (1.0 - image_alpha)*mask[cond]

    return normalize_image(image)

class ChippedImage:
    def __init__(self, df, shape, stride, coord_columns, pixel_columns):
        self.df = df
        self.cw, self.ch = shape
        self.sx, self.sy = stride
        self.coord_cols = coord_columns
        self.pixel_cols = pixel_columns
    
    def get_data(self, include_pixels=True):
        columns = []
        if include_pixels:
            columns += self.pixel_cols
        
        return self.df.loc[:, columns].values
    
    @classmethod
    def FromRGB(cls, img, shape, stride=None):
        """Chip an RGB image"""
        shape = (shape, shape) if isinstance(shape, int) else shape
        stride = shape if (stride is None) else stride
        stride = (stride, stride) if isinstance(stride, int) else stride
        cw, ch = shape
        sx, sy = stride
        
        h, w, c = (img.shape[0], img.shape[1], img.shape[2])
        ncol = (cw * ch * c) + 2
        
        maxx = w - cw
        maxy = h - ch
        nx = 1 + (maxx // sx)
        ny = 1 + (maxy // sy)
        nrow = nx * ny
        chips = np.zeros((nrow, ncol))
        row = 0
        coord_columns = ["x", "y"]
        pixel_columns = ["pixel {}".format(i+1) for i in range(ch*cw*c)]
        
        # Want normalized pixel values
        img = normalize_image(img)
        
        for y in range(0, h-sy, sy):
            for x in range(0, w-sx, sx):
                chips[row, 2:] = img[y:(y+ch), x:(x+cw), :].flatten()
                chips[row, 0] = x
                chips[row, 1] = y
                row += 1
                
        chips = pd.DataFrame(chips, columns=(coord_columns + pixel_columns))
        chips.astype({"x": "int32", "y": "int32"}, copy=False)
        return cls(chips, shape, stride, coord_columns, pixel_columns)
