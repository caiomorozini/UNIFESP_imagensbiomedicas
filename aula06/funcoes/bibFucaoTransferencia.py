#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 20:24:28 2022

@author: caiomorozini
"""
import numpy as np

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


def fazer_mascara_gaussiana_2d(m, n, fc):
    h_gauss = np.zeros((m,n), complex)
    d0 = fc * (m/2)
    for l in range(m):
        for c in range(n):
            dist_c = c - (n/2)
            dist_l = l - (m/2)
            d = np.sqrt((dist_c ** 2) + (dist_l ** 2))
            h_gauss[l,c] = np.exp(-((d**2)/((2*d0) ** 2))) + 0j
    return h_gauss

def fazer_mascara_butterworth_2d(m, n, fc, N):
    h_butter = np.zeros((m,n), complex)
    d0 = fc * (m/2)
    for l in range(m):
        for c in range(n):
            dist_c = c - (n/2)
            dist_l = l - (m/2)
            d = np.sqrt((dist_c ** 2) + (dist_l ** 2))
            h_butter[l,c] = 1/ (1 + (d/d0)**2*n) + 0j
    return h_butter