#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 17:28:23 2018

@author: nephilim
"""
from numba import jit
import numpy as np

@jit(nopython=True)
def Cross_Gradient(m1,m2,dx,dz):
    t_=np.zeros(m1.shape)
    g_m1=np.zeros(m1.shape)
    g_m2=np.zeros(m2.shape)
    for i in range(1,t_.shape[0]-1):
        for j in range(1,t_.shape[1]-1):
            W1=[-(m1[i+1,j]-m1[i-1,j])/4/dx/dz,(m1[i+1,j]-m1[i-1,j])/4/dx/dz,\
                -(m1[i,j+1]-m1[i,j-1])/4/dx/dz,(m1[i,j+1]-m1[i,j-1])/4/dx/dz]
            m2_=[m2[i,j-1],m2[i,j+1],m2[i+1,j],m2[i-1,j]]
            
            W2=[(m2[i+1,j]-m2[i-1,j])/4/dx/dz,-(m2[i,j+1]-m2[i,j-1])/4/dx/dz,\
                (m2[i,j+1]-m2[i,j-1])/4/dx/dz,-(m2[i+1,j]-m2[i-1,j])/4/dx/dz]
            m1_=[m1[i,j-1],m1[i-1,j],m1[i+1,j],m1[i,j+1]]

            t_[i,j]=-np.dot(np.array(W2),np.array(m1_))
            
            tmp_gm1=np.dot(np.array(W2).reshape((4,1)),np.array(W2).reshape((1,4)))
            gm1=-np.dot(tmp_gm1,np.array(m1_))
            g_m1[i,j-1],g_m1[i-1,j],g_m1[i+1,j],g_m1[i,j+1]=[gm1[0],gm1[1],gm1[2],gm1[3]]
            
            tmp_gm2=np.dot(np.array(W1).reshape((4,1)),np.array(W1).reshape((1,4)))
            gm2=-np.dot(tmp_gm2,np.array(m2_))
            g_m2[i,j-1],g_m2[i,j+1],g_m2[i+1,j],g_m2[i-1,j]=[gm2[0],gm2[1],gm2[2],gm2[3]]
    f_CroGra=0.5*np.linalg.norm(t_.flatten(),2)
    return f_CroGra,g_m1.flatten(),g_m2.flatten()