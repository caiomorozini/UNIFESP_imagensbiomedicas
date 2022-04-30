#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 19:07:15 2022

@author: caiomorozini
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2 # OpenCV
import skimage
import skimage.exposure
from bib_funcoes_histograma import fazer_histograma


image_dir = '../imagens'
mamo = cv2.imread(f"{image_dir}/Mamography.pgm", 0)
stent = cv2.imread(f"{image_dir}/Stent.pgm", 0)

mamo_n = skimage.img_as_float(mamo)
stent_n = skimage.img_as_float(stent)

plt.figure(0)
plt.ylabel('linhas - M')
plt.xlabel('colunas - N')
plt.title('mamo_n')
plt.imshow(mamo_n, cmap='gray')
plt.colorbar()

[M,N] = np.shape(mamo_n)
negativo_mamo_por_pixel = np.zeros((M,N), float)
negativo_mamo_direto = np.zeros((M,N), float)

for i in range(M):
    for c in range(N):
        negativo_mamo_por_pixel[i,c] = 255 - mamo_n[i,c]
        
plt.figure(1)
plt.ylabel('linhas - M')
plt.xlabel('colunas - N')
plt.title('negativo_mamo_por_pixel')
plt.imshow(negativo_mamo_por_pixel, cmap='gray') # cmap='jet'
plt.colorbar()

negativo_mamo_direto[::1] = 255 - mamo_n[::1]

plt.figure(1)
plt.ylabel('linhas - M')
plt.xlabel('colunas - N')
plt.title('negativo_mamo_direto')
plt.imshow(negativo_mamo_direto, cmap='gray') # cmap='jet'
plt.colorbar()

# mamo_direto é mais rapido pois não precisa iterar sobre as linhas e colunas.
histograma = fazer_histograma(mamo, M, N)
plt.figure(3)
plt.stem(histograma, use_line_collection=True)
plt.show()

histograma2 = skimage.exposure.histogram(mamo)
x = histograma2[1] # Classes
y = histograma2[0] # Numero de Ocorrência
plt.figure(4)
plt.stem(x, y, use_line_collection=True)
plt.ylabel('Numero de Ocorrência')
plt.xlabel('Classes')
plt.ylim([0,1400])
plt.show()

histograma3 = skimage.exposure.histogram(mamo_n)
x = histograma3[1] # Classes
y = histograma3[0] # Numero de Ocorrência
plt.figure(5)
plt.stem(x, y, use_line_collection=True)
plt.ylabel('Numero de Ocorrência')
plt.xlabel('Classes')
plt.ylim([0,1400])
plt.show()

plt.figure()
plt.imshow(stent_n, cmap='gray') # cmap='jet'
stent_brilho_n = stent_n + 0.2
stent_brilho_n = skimage.exposure.rescale_intensity(stent_brilho_n,in_range=(0,1))
plt.figure()
plt.imshow(stent_brilho_n, cmap='gray') # cmap='jet'
histograma4 = skimage.exposure.histogram(stent_brilho_n)
x = histograma4[1] # Classes
y = histograma4[0] # Numero de 
plt.figure()
plt.stem(x, y, use_line_collection=True)
plt.ylabel('Numero de Ocorrência')
plt.xlabel('Classes')
plt.ylim([0,10000])
plt.show()

stent_alongada = skimage.exposure.rescale_intensity(stent_brilho_n, in_range=(0.2,0.7))
plt.figure()
plt.imshow(stent_alongada, cmap='gray') # cmap='jet'
histograma5 = skimage.exposure.histogram(stent_alongada)
x = histograma5[1] # Classes
y = histograma5[0] # Numero de 
plt.figure()
plt.stem(x, y, use_line_collection=True)
plt.ylabel('Numero de Ocorrência')
plt.xlabel('Classes')
plt.ylim([0,10000])
plt.show()

stent_gama = skimage.exposure.adjust_gamma(stent_alongada, 1)
plt.figure()
plt.imshow(stent_gama, cmap='gray') # cmap='jet'
histograma6 = skimage.exposure.histogram(stent_gama)
x = histograma6[1] # Classes
y = histograma6[0] # Numero de 
plt.figure()
plt.stem(x, y, use_line_collection=True)
plt.ylabel('Numero de Ocorrência')
plt.xlabel('Classes')
plt.ylim([0,10000])
plt.show()

stent_gama = skimage.exposure.adjust_gamma(stent_alongada, 2)
plt.figure()
plt.imshow(stent_gama, cmap='gray') # cmap='jet'
histograma6 = skimage.exposure.histogram(stent_gama)
x = histograma6[1] # Classes
y = histograma6[0] # Numero de 
plt.figure()
plt.stem(x, y, use_line_collection=True)
plt.ylabel('Numero de Ocorrência')
plt.xlabel('Classes')
plt.ylim([0,10000])
plt.show()


stent_gama = skimage.exposure.adjust_gamma(stent_alongada, 0.5)
plt.figure()
plt.imshow(stent_gama, cmap='gray') # cmap='jet'
histograma6 = skimage.exposure.histogram(stent_gama)
x = histograma6[1] # Classes
y = histograma6[0] # Numero de 
plt.figure()
plt.stem(x, y, use_line_collection=True)
plt.ylabel('Numero de Ocorrência')
plt.xlabel('Classes')
plt.ylim([0,10000])
plt.show()

def plota_histograma_e_grafico(valor, gama):
    plt.figure()
    plt.title(f'gama = {gama}')
    plt.imshow(valor, cmap='gray') # cmap='jet'
    histograma = skimage.exposure.histogram(valor)
    x = histograma6[1] # Classes
    y = histograma6[0] # Numero de 
    plt.figure()
    plt.title(f'gama = {gama}')
    plt.stem(x, y, use_line_collection=True)
    plt.ylabel('Numero de Ocorrência')
    plt.xlabel('Classes')
    plt.ylim([0,10000])
    plt.show()

for i in range(1,10):
    gama = i/10
    stent_gama = skimage.exposure.adjust_gamma(stent_alongada, gama)
    plota_histograma_e_grafico(stent_gama, gama)

for gama in range(1,11,1):
    stent_gama = skimage.exposure.adjust_gamma(stent_alongada, gama)
    plota_histograma_e_grafico(stent_gama, gama)