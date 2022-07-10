import cv2
import scipy
import skimage
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

ivus_referencia = cv2.imread('../imagens/IVUSReferencia.pgm', 0) # Gray
ivus_referencia_float = skimage.img_as_float(ivus_referencia)
plt.figure()
plt.title('IVUSReferencia')
plt.imshow(ivus_referencia, cmap='gray')

f_noise = list()
fig = plt.figure(1, (10., 10.))
grid = ImageGrid(fig, 111,
                 nrows_ncols=(2, 5),
                 axes_pad=0.1,
                 )
for i in range(1, 6):
    f_noise.append(
        skimage.util.random_noise(
            image=ivus_referencia,
            var=i*0.001,
            mode='gaussian',
        )
    )
    grid[i-1].imshow(f_noise[i-1])
    

objeto_gold_standard = cv2.imread('../imagens/ObjetoGoldStandard.pgm', 0) # Gray
objeto_gold_standard = skimage.img_as_float(objeto_gold_standard) 
plt.figure()
plt.title('objeto_gold_standard')
plt.imshow(objeto_gold_standard, cmap='gray')

roi = cv2.selectROI(ivus_referencia_float)

c_min = roi[0]
l_min = roi[1]
c_max = roi[0] + roi[2]
l_max = roi[1] + roi[3]
variancia_homogenia = np.var(ivus_referencia[l_min:l_max, c_min:c_max])
media = np.mean(ivus_referencia[l_min:l_max, c_min:c_max])
desvio_padrao = np.std(ivus_referencia[l_min:l_max, c_min:c_max])

(M,N) = np.shape(ivus_referencia)
k = np.zeros((M,N),float)

for m in range(M):
    for n in range(N):
        if (ivus_referencia[m,n] >= media-desvio_padrao) and (ivus_referencia[m,n] <= media+desvio_padrao):
            k[m,n] = ivus_referencia[m,n]


plt.figure()
plt.title('k')
plt.imshow(k, cmap='gray')

def fazer_avaliacao_segmentacao(objeto_segmentado, gold_standard):
    objeto_segmentado = ob