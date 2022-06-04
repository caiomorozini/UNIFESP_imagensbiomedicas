#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 20:03:16 2022

@author: caiomorozini
"""

from funcoes import bibFucaoTransferencia as bft
import numpy as np
import cv2
import skimage
import matplotlib.pyplot as plt
import scipy.signal
import scipy

M = 100
N = 100
FC = 0.5
x = bft.fazer_mascara_ideal_2D(M, N, FC)
y = bft.fazer_mascara_gaussiana_2d(M, N, FC)
z = bft.fazer_mascara_butterworth_2d(M, N, FC, 2)
plt.figure()
plt.title('Ideal')
plt.imshow(np.abs(x), cmap='gray')

plt.figure()
plt.title('gauss')
plt.imshow(np.abs(y), cmap='gray')

plt.figure()
plt.title('butter')
plt.imshow(np.abs(z), cmap='gray')

mamo = cv2.imread('../imagens/Mamography.pgm', 0) # Gray
mamo = skimage.img_as_float(mamo)
plt.figure()
plt.title('mamo')
plt.imshow(mamo, cmap='gray')


dim = (400,400)
abc = cv2.resize(mamo, dim, interpolation = cv2.INTER_AREA)

PulsoFrequencia = np.fft.fft2(abc)
plt.figure()
plt.title('ModuloPulsoFrequencia')
plt.imshow(np.abs(PulsoFrequencia), cmap='gray')

FrequenciaDeslocado = np.fft.fftshift(PulsoFrequencia)
plt.figure()
plt.title('FrequenciaDeslocado')
plt.imshow(np.abs(FrequenciaDeslocado), cmap='gray')

mascara = bft.fazer_mascara_ideal_2D(400, 400, 0.2)

Ffiltrado = FrequenciaDeslocado * mascara
plt.figure()
plt.title('Ffiltrado')
plt.imshow(np.abs(Ffiltrado), cmap='gray')

reverse_Ffiltrado = np.fft.ifft2(Ffiltrado)
plt.figure()
plt.title('Inversa Ffltrado')
plt.imshow(np.abs(reverse_Ffiltrado), cmap='gray')

mascara_gauss = bft.fazer_mascara_gaussiana_2d(400, 400, 0.2)

Ffiltrado_gauss = FrequenciaDeslocado * mascara_gauss
plt.figure()
plt.title('Ffiltrado_gauss')
plt.imshow(np.abs(Ffiltrado_gauss), cmap='gray')

reverse_Ffiltrado_gauss = np.fft.ifft2(Ffiltrado_gauss)
plt.figure()
plt.title('Inversa Ffltrado gauss')
plt.imshow(np.abs(reverse_Ffiltrado_gauss), cmap='gray')

mascara_butter = bft.fazer_mascara_butterworth_2d(400, 400, 0.2, 2)

Ffiltrado_butter = FrequenciaDeslocado * mascara_butter
plt.figure()
plt.title('Ffiltrado_butter')
plt.imshow(np.abs(Ffiltrado_butter), cmap='gray')

reverse_Ffiltrado_butter = np.fft.ifft2(Ffiltrado_butter)
plt.figure()
plt.title('Inversa Ffltrado_butter')
plt.imshow(np.abs(reverse_Ffiltrado_butter), cmap='gray')

alta_ideal = bft.fazer_mascara_ideal_2D(400, 400, 0.1)
Ffiltrado_alta_ideal = FrequenciaDeslocado * (1 - alta_ideal)

plt.figure()
plt.title('Ffiltrado_alta_ideal')
plt.imshow(np.abs(Ffiltrado_alta_ideal), cmap='gray')

reverse_Ffiltrado_alta_ideal = np.fft.ifft2(Ffiltrado_alta_ideal)
plt.figure()
plt.title('Inversa Ffiltrado_alta_ideal')
plt.imshow(np.abs(reverse_Ffiltrado_alta_ideal), cmap='gray')