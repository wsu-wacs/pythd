"""
Code for working with image data
"""
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

def plot_images(images, dpi=300.0, **kwargs):
    """Show multiple images in one plot"""
    fig, ax = plt.subplots(1, len(images), dpi=dpi)
    for i, img in enumerate(images):
        ax[i].imshow(img, **kwargs)
        ax[i].axis("off")
    return (fig, ax)

def chip_image(img, shape, stride=None):
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
    columns = ["x", "y"]
    columns += ["pixel {}".format(i+1) for i in range(ch*cw*c)]
    
    for y in range(0, h-sy, sy):
        for x in range(0, w-sx, sx):
            chips[row, 2:] = img[y:(y+ch), x:(x+cw), :].flatten()
            chips[row, 0] = x
            chips[row, 1] = y
            row += 1
            
    return pd.DataFrame(chips, columns=columns)

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
        img = img.copy().astype(float)
        for i in range(c):
            img[:,:,i] = (img[:,:,i] - img[:,:,i].min()) / (img[:,:,i].max() - img[:,:,i].min())
        
        for y in range(0, h-sy, sy):
            for x in range(0, w-sx, sx):
                chips[row, 2:] = img[y:(y+ch), x:(x+cw), :].flatten()
                chips[row, 0] = x
                chips[row, 1] = y
                row += 1
                
        chips = pd.DataFrame(chips, columns=(coord_columns + pixel_columns))
        return cls(chips, shape, stride, coord_columns, pixel_columns)
