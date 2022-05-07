#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 19:47:04 2022

@author: caiomorozini
"""

import numpy as np
import cv2
import skimage
import matplotlib.pyplot as plt
f = np.array([0,0,0,1,0,0,0,0])
w = np.array([1,2,3,2,8])

l0 = len(f)
lw = len(w)
pad = np.zeros(4)

fpadding = np.concatenate((pad,f,pad))

(L, )= np.shape(fpadding)
Lc = L - (lw-1)

cor = np.zeros(Lc)
for index in range(Lc):
    cor[index] = np.sum(w[0:(lw-1)] * fpadding[index:index+(lw-1)]) 
    

corFuncao = np.correlate(f,w,'full')

ccrop = cor[len(pad)-2:len(f)+len(pad)-2]

f = np.zeros(shape=[5,5], dtype='float') 
c = np.zeros_like(f)

f[2,2] = 1
w = np.matrix([[1,2,3],[4,5,6],[7,8,9]])
M,N = np.shape(f)

cor2= np.zeros_like(f)
for i in range(M-2):
    for j in range(N-2):
        cor2[i+1,j+1] = np.sum(w[0:3,0:3] * f[i:i+3,j:j+3])
        
# Q9
import scipy.signal

corFuncao2 = scipy.signal.correlate2d(f, w, boundary='symm', mode='same')

i0 = cv2.imread('../imagens/Mamography.pgm', 0) # Gray
in0 = skimage.img_as_float(i0)
w3 = np.ones([3,3]) * 1/9
in0Filt = scipy.signal.correlate2d(in0,w3);
in0Filt = skimage.exposure.rescale_intensity(in0Filt, in_range=(0,1))
plt.figure()
plt.title('imFilt0')
plt.imshow(in0Filt, cmap='gray')

m5 = np.ones((5,5), dtype='float') * 1/25
m10 = np.ones((10,10),dtype='float') * 1/100

in0Filt = scipy.signal.correlate2d(in0,m5);
in0Filt = skimage.exposure.rescale_intensity(in0Filt, in_range=(0,1))
plt.figure()
plt.title('imFilt0')
plt.imshow(in0Filt, cmap='gray')

in0Filt = scipy.signal.correlate2d(in0,m10);
in0Filt = skimage.exposure.rescale_intensity(in0Filt, in_range=(0,1))
plt.figure()
plt.title('imFilt0')
plt.imshow(in0Filt, cmap='gray')
