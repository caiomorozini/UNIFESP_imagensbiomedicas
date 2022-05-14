#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 13 22:10:52 2022

@author: caiomorozini
"""
import numpy as np
import scipy.signal
import scipy

def fazerMascaraGauss2D(desvio_padrao, media):
    tamanho = 2*media + 1
    IMRI_gauss = scipy.signal.gaussian(tamanho, std=desvio_padrao)
    g1 = np.zeros((tamanho, tamanho), float)
    g1[media,:] = IMRI_gauss
    gtranspose1 = np.transpose(g1)
    w_Gauss2D = scipy.signal.convolve2d(g1, gtranspose1, 'same')
    w_gauss2DNormal = w_Gauss2D/np.sum(w_Gauss2D)
    return w_gauss2DNormal