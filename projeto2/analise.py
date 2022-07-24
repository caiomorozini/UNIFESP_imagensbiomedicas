import cv2
import scipy
import skimage
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import os
from skimage.segmentation import clear_border
from skimage import measure
from skimage.measure import label,regionprops
from scipy import ndimage as ndi
from scipy.ndimage import measurements, center_of_mass, binary_dilation, zoom


images_path = './imagens/'
images = list()

for dir, _, files in os.walk(images_path):
    for file in files:
        if 'img' in file:
            images.append(
                {
                    'original': skimage.img_as_float(
                        cv2.imread(os.path.join(dir, file), 0)), 
                    'gold_standard': skimage.img_as_float(
                        cv2.imread(os.path.join(dir, file.replace('img', 'gsmab'))), 0),
                })

fig = plt.figure(1, (50., 40.))
grid = ImageGrid(fig, 111,
                 nrows_ncols=(5, 2),
                 axes_pad=0.1,
                 )
for index, img in enumerate(images[0:5]):
    grid[0+(index*2)].imshow(images[index]['original'])
    grid[(index*2)+1].imshow(images[index]['gold_standard'])


m1 = images[0]['original']
m2 = images[0]['gold_standard']