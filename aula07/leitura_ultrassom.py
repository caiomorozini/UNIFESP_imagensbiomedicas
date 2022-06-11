#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 19:13:20 2022

@author: caiomorozini
"""

import numpy as np
import cv2
import skimage
import matplotlib.pyplot as plt
import scipy.signal
import scipy

ultrassom_bebe = cv2.imread('../imagens/UltrassomBebe.pgm', 0) # Gray
ultrassom_bebe = skimage.img_as_float(ultrassom_bebe)
plt.figure()
plt.title('ultrassom_bebe')
plt.imshow(ultrassom_bebe, cmap='gray')

roi = cv2.selectROI(ultrassom_bebe)

c_min = roi[0]
l_min = roi[1]
c_max = roi[0] + roi[2]
l_max = roi[1] + roi[3]

variancia_homogenia = np.var(ultrassom_bebe[l_min:c_min, l_max:c_max])
media = np.mean(ultrassom_bebe[l_min:c_min, l_max:c_max])

def fazer_mascara_ideal_2D(m, n, fc):
    h_ideal = np.zeros((m,n), complex)
    d0 = fc * (m/2)
    for l in range(m):
        for c in range(n):
            dist_c = c - (n/2)
            dist_l = l - (m/2)
            d = np.sqrt((dist_c ** 2) + (dist_l ** 2))
            if d < d0:
                h_ideal[l,c] = 1 + 0j
    return h_ideal

W = np.ones((7,7), float)/49
i_media = scipy.signal.convolve2d(ultrassom_bebe, W,'same')
i_lee = np.zeros_like(ultrassom_bebe)

(M,N) = np.shape(ultrassom_bebe)
k = np.zeros_like(ultrassom_bebe)
for l in range(M-7):
    for c in range(N-7):
        var_local = np.var(ultrassom_bebe[l:l+7,c:c+7]) + 0.0000001
        k[l+3,c+3] = 1 - (variancia_homogenia/var_local)
        k[l+3,c+3] = np.clip(k[l+3,c+3], 0, 1)

i_lee = i_media + k * (ultrassom_bebe - i_media)

plt.figure()
plt.title('i_media')
plt.imshow(np.abs(i_media), cmap='gray')
plt.figure()
plt.title('i_lee')
plt.imshow(np.abs(i_lee), cmap='gray')

wx_sobel = np.matrix('-1 0 1; -2 0 2; -1 0 1')
wy_sobel = np.matrix('-1 -2 -1; -0 0 0; 1 2 1')
dx = scipy.signal.convolve2d(ultrassom_bebe, wx_sobel,'same')
dy = scipy.signal.convolve2d(ultrassom_bebe, wy_sobel,'same')

grad_sobel = np.power(np.power(dx, 2) + np.power(dy,2), 0.5)
k_sobel = np.clip(grad_sobel, 0, 1)
i_lee_modificado = i_media + k_sobel * (ultrassom_bebe - i_media)

plt.figure()
plt.title('i_lee_modificado')
plt.imshow(np.abs(i_lee), cmap='gray')