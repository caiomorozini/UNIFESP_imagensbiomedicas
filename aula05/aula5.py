#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 19:11:39 2022

@author: caiomorozini
"""

import numpy as np
import cv2
import skimage
import matplotlib.pyplot as plt
import scipy.signal
import scipy

f = [1,3,5]
t = np.arange(start=0, stop=10, step=0.01)
s = list()
for index, value in enumerate(f):
    s.append(np.sin(2*np.pi*value*t))
    plt.figure()
    plt.subplot(4,1,1)
    plt.plot(t,s[index])
    plt.ylabel(f's{index+1}')
    if index >= 2:
        plt.figure()
        signal = sum(s)
        plt.subplot(4,1,1)
        plt.plot(t,signal)
        plt.ylabel('sum s')

X = np.zeros(10, complex)
for f in range(0, 10):
    euler = np.exp(-1j*2*(np.math.pi)*f*t)
    X[f] = np.abs(np.sum(signal*euler))

plt.figure()
plt.stem(range(10), X)
    

pulsoQuadrado = cv2.imread('imagens/PulsoQuadrado1.pgm', 0) # Gray
pulsoQuadrado = skimage.img_as_float(pulsoQuadrado)
plt.figure()
plt.title('pulsoQuadrado')
plt.imshow(pulsoQuadrado, cmap='gray')

PulsoFrequencia = np.fft.fft2(pulsoQuadrado)
plt.figure()
plt.title('ModuloPulsoFrequencia')
plt.imshow(np.abs(PulsoFrequencia), cmap='gray')

FrequenciaDeslocado = np.fft.fftshift(PulsoFrequencia)
plt.figure()
plt.title('FrequenciaDeslocado')
plt.imshow(np.abs(FrequenciaDeslocado), cmap='gray')

plt.figure()
plt.title('FrequenciaDeslocado em logaritmo')
plt.imshow(np.log(1 + np.abs(FrequenciaDeslocado)), cmap='gray')


H = np.zeros_like(FrequenciaDeslocado)
M, N = np.shape(H)
fc = 0.1
d0 = fc * (M/2)

for l in range(M):
    for c in range(N):
        dist_c = c - (N/2)
        dist_l = l - (M/2)
        dist = np.sqrt((dist_c ** 2) + (dist_l ** 2))
        if dist < d0:
            H[l,c] = 1 + 0j
 
plt.figure()
plt.title('FTransferencia')
plt.imshow(np.abs(H), cmap='gray')

Ffiltrado = FrequenciaDeslocado * H
plt.figure()
plt.title('Ffiltrado')
plt.imshow(np.abs(Ffiltrado), cmap='gray')

reverse_Ffiltrado = np.fft.ifft2(Ffiltrado)
plt.figure()
plt.title('Inversa Ffltrado')
plt.imshow(np.abs(reverse_Ffiltrado), cmap='gray')

mamo = cv2.imread('imagens/Mamography.pgm', 0)
mamo = skimage.img_as_float(mamo)
plt.figure()
plt.title('Mamography')
plt.imshow(mamo, cmap='gray')

PulsoFrequencia = np.fft.fft2(mamo)
plt.figure()
plt.title('ModuloPulsoFrequencia')
plt.imshow(np.abs(PulsoFrequencia), cmap='gray')

FrequenciaDeslocado = np.fft.fftshift(PulsoFrequencia)
plt.figure()
plt.title('FrequenciaDeslocado')
plt.imshow(np.abs(FrequenciaDeslocado), cmap='gray')

plt.figure()
plt.title('FrequenciaDeslocado em logaritmo')
plt.imshow(np.log(1 + np.abs(FrequenciaDeslocado)), cmap='gray')


H = np.zeros_like(FrequenciaDeslocado)
M, N = np.shape(H)
fc = 0.1
d0 = fc * (M/2)

for l in range(M):
    for c in range(N):
        dist_c = c - (N/2)
        dist_l = l - (M/2)
        dist = np.sqrt((dist_c ** 2) + (dist_l ** 2))
        if dist < d0:
            H[l,c] = 1 + 0j
 
plt.figure()
plt.title('FTransferencia mamo')
plt.imshow(np.abs(H), cmap='gray')

mamo_Ffiltrado = FrequenciaDeslocado * H
plt.figure()
plt.title('mamofiltrado')
plt.imshow(np.abs(mamo_Ffiltrado), cmap='gray')

mamo_reverse_Ffiltrado = np.fft.ifft2(mamo_Ffiltrado)
plt.figure()
plt.title('Inversa mamo filtrado')
plt.imshow(np.abs(mamo_reverse_Ffiltrado), cmap='gray')


trans_imr = cv2.imread('imagens/TransversalMRI2.pgm', 0)
trans_imr = skimage.img_as_float(trans_imr)
plt.figure()
plt.title('trans_imr')
plt.imshow(trans_imr, cmap='gray')

PulsoFrequencia = np.fft.fft2(trans_imr)
plt.figure()
plt.title('ModuloPulsoFrequencia trans_imr')
plt.imshow(np.abs(PulsoFrequencia), cmap='gray')

FrequenciaDeslocado = np.fft.fftshift(PulsoFrequencia)
plt.figure()
plt.title('FrequenciaDeslocado trans_imr')
plt.imshow(np.abs(FrequenciaDeslocado), cmap='gray')

plt.figure()
plt.title('FrequenciaDeslocado em logaritmo')
plt.imshow(np.log(1 + np.abs(FrequenciaDeslocado)), cmap='gray')


H = np.zeros_like(FrequenciaDeslocado)
M, N = np.shape(H)
fc = 0.1
d0 = fc * (M/2)

for l in range(M):
    for c in range(N):
        dist_c = c - (N/2)
        dist_l = l - (M/2)
        dist = np.sqrt((dist_c ** 2) + (dist_l ** 2))
        if dist < d0:
            H[l,c] = 1 + 0j
 
plt.figure()
plt.title('FTransferencia')
plt.imshow(np.abs(H), cmap='gray')

Ffiltrado = FrequenciaDeslocado * H
plt.figure()
plt.title('Ffiltrado')
plt.imshow(np.abs(Ffiltrado), cmap='gray')

reverse_Ffiltrado = np.fft.ifft2(Ffiltrado)
plt.figure()
plt.title('Inversa Ffltrado')
plt.imshow(np.abs(reverse_Ffiltrado), cmap='gray')


