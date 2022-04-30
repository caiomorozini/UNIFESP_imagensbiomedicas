#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 20:46:58 2022

@author: caiomorozini
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2

def fazer_histograma(imagem, M, N):
    histograma = np.zeros((256), dtype=int)
    for i in range(M):
        for j in range(N):
            if imagem[i,j] > 0:
                histograma[imagem[i,j]] += 1
            
    return histograma
    
        