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

# questão 7
IMRI2 = cv2.imread("../imagens/TransversalMRI2.pgm", 0)
IMRI2 = skimage.img_as_float(IMRI2)
plt.figure()
plt.title('IMRI2_original')
plt.imshow(IMRI2 , cmap='gray') # cmap='jet'

w_Gauss2D_IMRI2_normalizado = bibMascara.fazerMascaraGauss2D(7, 3) 
plt.figure()
plt.title('w_Gauss2DNormalizado')
plt.imshow(w_Gauss2D_IMRI2_normalizado , cmap='gray') # cmap='jet'

f_borrado = scipy.signal.convolve2d(IMRI2, w_Gauss2D_IMRI2_normalizado, 'same')
plt.figure()
plt.title('w_Gauss2DNormalizado')
plt.imshow(f_borrado , cmap='gray') # cmap='jet'

gxy = IMRI2 - f_borrado
fxy = IMRI2 + gxy
plt.figure()
plt.title('w_Gauss2DNormalizado')
plt.imshow(fxy , cmap='gray') # cmap='jet'

# questão 8
stent = cv2.imread("../imagens/Stent.pgm", 0)
stent = skimage.img_as_float(stent)

wxpriwitt = np.matrix('-1 0 1; -1 0 1; -1 0 1')
wypriwitt = np.matrix('-1 -1 -1; 0 0 0; 1 1 1')
wxsobel = np.matrix('-1 0 1; -2 0 2; -1 0 1')
wysobel = np.matrix('-1 -2 -1; -0 0 0; 1 2 1')

dx = scipy.signal.convolve2d(stent, wxpriwitt,'same')
dy = scipy.signal.convolve2d(stent, wypriwitt,'same')

grad_priwitt = np.abs(np.power(dx, 2) + np.power(dy,2))
plt.figure()
plt.title('stent_grad_priwitt')
plt.imshow(grad_priwitt , cmap='gray') # cmap='jet'

dx = scipy.signal.convolve2d(stent, wxsobel,'same')
dy = scipy.signal.convolve2d(stent, wysobel,'same')
grad_priwitt = np.abs(np.power(dx, 2) + np.power(dy,2))

plt.figure()
plt.title('stent_grad_sobel')
plt.imshow(grad_priwitt , cmap='gray') # cmap='jet'

gradgeneric = np.matrix('1 0 1; 1 -4 0; 0 1 0')
gradgeneric2 = np.matrix('1 1 1; 1 -8 1; 1 1 1')
laplace_gradgeneric = scipy.signal.convolve2d(stent, gradgeneric,'same')
laplace_gradgeneric2 = scipy.signal.convolve2d(stent, gradgeneric2,'same')
plt.figure()
plt.title('stent_grad_sobel')
plt.imshow(laplace_gradgeneric , cmap='gray') # cmap='jet'
plt.figure()
plt.title('stent_grad_sobel')
plt.imshow(laplace_gradgeneric2 , cmap='gray') # cmap='jet'

# Desafio

w01kirsch = np.matrix('5 -3 -3; 5 0 -3; 5 -3 -3')
w02kirsch = np.matrix('-3 -3 -3; 5 0 -3; 5 5 -3')
w03kirsch = np.matrix('-3 -3 -3; -3 0 -3; 5 5 5')
w04kirsch = np.matrix('-3 -3 -3; -3 0 5; -3 5 5')
w05kirsch = np.matrix('-3 -3 5; -3 0 5; -3 -3 5')
w06kirsch = np.matrix('-3 5 5; -3 0 5; -3 -3 -3')
w07kirsch = np.matrix('5 5 5; -3 0 -3; -3 -3 -3')
w08kirsch = np.matrix('5 5 -3; 5 0 -3; -3 -3 -3')

masks = (w01kirsch, w02kirsch, w03kirsch, w04kirsch,w05kirsch, w06kirsch, w07kirsch, w08kirsch)
x = list()

for mask in masks:
    gradient = scipy.signal.convolve2d(stent, mask, 'same')
    x.append(gradient)

abc = np.dstack(x)
linha, coluna = np.shape(stent)
new_matrix = np.zeros_like(stent)
for i in range(linha-1):
    for j in range(coluna-1):
        new_matrix[i,j] = np.max(abc[i,j,:])
    

plt.figure()
plt.title('stent_desafio_kirsch')
plt.imshow(new_matrix , cmap='gray') # cmap='jet'

w01_robinson = np.matrix('1 0 -1; 2 0 -2; 1 0 -1')
w02_robinson = np.matrix('0 -1 -2; 1 0 -1; 2 1 0')
w03_robinson = np.matrix('-1 -2 -1; 0 0 0; 1 2 1')
w04_robinson = np.matrix('-2 -1 0; -1 0 1; 0 1 2')
w05_robinson = np.matrix('-1 0 1; -2 0 2; -1 0 1')
w06_robinson = np.matrix('0 1 2; -1 0 1; -2 -1 0')
w07_robinson = np.matrix('1 2 1; 0 0 0; -1 -2 -1')
w08_robinson = np.matrix('2 1 0; 1 0 -1; 0 -1 -2')

masks = (w01_robinson, w02_robinson, w03_robinson, w04_robinson, 
         w05_robinson, w06_robinson, w07_robinson, w08_robinson)
x = list()

for mask in masks:
    gradient = scipy.signal.convolve2d(stent, mask, 'same')
    x.append(gradient)

abc = np.dstack(x)
linha, coluna = np.shape(stent)
new_matrix = np.zeros_like(stent)
for i in range(linha-1):
    for j in range(coluna-1):
        new_matrix[i,j] = np.max(abc[i,j,:])
    

plt.figure()
plt.title('stent_desafio_robinson')
plt.imshow(new_matrix , cmap='gray') # cmap='jet'

w01_freichen = (1/(2*np.sqrt(2))) * np.matrix(f'1 {np.sqrt(2)} 1; 0 0 0; -1 -{np.sqrt(2)} -1')
w02_freichen = (1/(2*np.sqrt(2))) * np.matrix(f'1 0 -1; {np.sqrt(2)} 0 -{np.sqrt(2)}; 1 0 -1')
w03_freichen = (1/(2*np.sqrt(2))) * np.matrix(f'0 -1 {np.sqrt(2)}; 1 0 -1; -{np.sqrt(2)} 1 0')
w04_freichen = (1/(2*np.sqrt(2))) * np.matrix(f'{np.sqrt(2)} -1 0; -1 0 1; 0 1 -{np.sqrt(2)}')
w05_freichen = (1/2) * np.matrix('0 1 0; -1 0 -1; 0 1 0')
w06_freichen = (1/2) * np.matrix('-1 0 1; 0 0 0; 1 0 -1')
w07_freichen = (1/2) * np.matrix('1 -2 1; -2 4 -2; 1 -2 1')
w08_freichen = (1/6) * np.matrix('-2 1 -2; 1 4 1; -2 1 -2')
w09_freichen = (1/3) *np.ones_like(w07_freichen)

masks = (w01_freichen, w02_freichen, w03_freichen, w04_freichen, 
         w05_freichen, w06_freichen, w07_freichen, w08_freichen, w09_freichen)
x = list()

for mask in masks:
    gradient = scipy.signal.convolve2d(stent, mask, 'same')
    x.append(gradient)

abc = np.dstack(x)
linha, coluna = np.shape(stent)
new_matrix = np.zeros_like(stent)
for i in range(linha-1):
    for j in range(coluna-1):
        new_matrix[i,j] = np.max(abc[i,j,:])
    

plt.figure()
plt.title('stent_desafio_freichen')
plt.imshow(new_matrix , cmap='gray') # cmap='jet'