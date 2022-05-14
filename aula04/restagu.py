#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 13 20:21:28 2022

@author: caiomorozini
"""
import numpy as np
import cv2
import skimage
import matplotlib.pyplot as plt
import scipy.signal
import scipy
import bibMascara
amostra = np.array([15, 29, 5, 8, 255, 40, 1, 0, 10])

amostraOrdenada = np.sort(amostra)

mediana = np.median(amostraOrdenada)

IMRI = cv2.imread("../imagens/TransversalMRI_salt-and-pepper.pgm", 0)
IMRI = skimage.img_as_float(IMRI)



m3 = np.ones(shape=(3,3))

def mask(matrix):
    M,N = np.shape(matrix)
    filtrada = np.zeros_like(matrix)
    for i in range(M-2):
        for j in range(N-2):
            mask = m3 * matrix[i:i+3,j:j+3]
            vetor_amostras = np.concatenate(mask, axis=None)
            vetor_amotras_ordenado = np.sort(vetor_amostras)
            filtrada[i+1, j+1] = vetor_amotras_ordenado[4]
    return filtrada

mask_IMRI = mask(IMRI)

mask_compare = scipy.signal.medfilt2d(IMRI)
IMRI_gauss = scipy.signal.gaussian(9, std=1)
plt.figure()
plt.ylabel('linhas - M')
plt.xlabel('colunas - N')
plt.title('image1 - blue')
plt.plot(IMRI_gauss)

g1 = np.zeros((9, 9), float)
g1[4,:] = IMRI_gauss
gtranspose1 = np.transpose(g1)
w_Gauss2D = scipy.signal.convolve2d(g1, gtranspose1, 'same')
w_gauss2DNormal = w_Gauss2D/np.sum(w_Gauss2D)

plt.figure()
plt.title('w_Gauss2D ')
plt.imshow(w_Gauss2D , cmap='gray') # cmap='jet'

mamo = cv2.imread('../imagens/Mamography.pgm', 0)
k = scipy.signal.convolve2d(mamo, w_gauss2DNormal, 'same')

plt.figure()
plt.title('w_Gauss2D ')
plt.imshow(k , cmap='gray') # cmap='jet'

w_Gauss2DNormalizado = bibMascara.fazerMascaraGauss2D(media=4, desvio_padrao=1)

w_Gauss2D_2 = scipy.signal.convolve2d(IMRI, w_Gauss2DNormalizado, 'same')

plt.figure()
plt.title('w_Gauss2D_2')
plt.imshow(w_Gauss2D_2 , cmap='gray') # cmap='jet'
w_Gauss2DNormalizado = bibMascara.fazerMascaraGauss2D(media=10, desvio_padrao=3)


plt.figure()
plt.title('w_Gauss2DNormalizado')
plt.imshow(w_Gauss2DNormalizado , cmap='gray') # cmap='jet'

w_Gauss2D_3 = scipy.signal.convolve2d(IMRI, w_Gauss2DNormalizado, 'same')

plt.figure()
plt.title('w_Gauss2D_3')
plt.imshow(w_Gauss2D_3 , cmap='gray') # cmap='jet'