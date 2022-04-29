import numpy as np
import matplotlib.pyplot as plt
import cv2 # OpenCV
import skimage
import skimage.exposure

i0 = cv2.imread('../imagens/raioXTorax.pgm', 0)
in0 = skimage.img_as_float(i0)
(M,N) = np.shape(in0)
maximo = np.max(in0)
minimo = np.min(in0)
media = np.mean(in0)
dP = np.std(in0)
print(
    f'maximo= {maximo}\n' + \
        f'minimo= {minimo}\n' + \
            f'media = {media}\n' + \
                f'dP = {dP}'
    )

plt.figure()
plt.ylabel('linhas - M')
plt.xlabel('colunas - N')
plt.title('image0')
plt.imshow(in0, cmap='gray') 
cmap='jet'
plt.colorbar()

i1 = cv2.imread('../imagens/Lamina-biopsia.jpg', 1)
in1 = skimage.img_as_float(i1)
(M,N,K) = np.shape(in1)

maximo = np.max(in1)
minimo = np.min(in1)
media = np.mean(in1)
dP = np.std(in1)
print(
    f'maximo= {maximo}\n' + \
        f'minimo= {minimo}\n' + \
            f'media = {media}\n' + \
                f'dP = {dP}'
    )

plt.figure()
plt.ylabel('linhas - M')
plt.xlabel('colunas - N')
plt.title('image1 - blue')
plt.imshow(in1[:,:,0], 'gray') 
plt.colorbar()

plt.figure()
plt.ylabel('linhas - M')
plt.xlabel('colunas - N')
plt.title('image1 - blue')
plt.imshow(in1[:,:,1], 'gray') 
plt.colorbar()


plt.figure()
plt.ylabel('linhas - M')
plt.xlabel('colunas - N')
plt.title('image1 - red')
plt.imshow(in1[:,:,2], 'gray') 

plt.colorbar()