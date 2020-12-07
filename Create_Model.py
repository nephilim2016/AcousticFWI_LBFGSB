#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 17:58:37 2018

@author: nephilim
"""
import numpy as np
from numba import jit
import skimage.transform
from skimage import filters
#Create Abnormal_Model
@jit(nopython=True)
def Abnormal_Model(xl,zl,CPML):
    rho=np.ones((xl+2*CPML+2*8,zl+2*CPML+2*8))*2000
    vp=np.ones((xl+2*CPML+2*8,zl+2*CPML+2*8))*3000
    p=8+34+CPML
    l=14
    w=3
    for i in range(p-l,p+l):
        for j in range(p-w,p+w):
            rho[i][j]=1500
            vp[i][j]=3500
    for i in range(p-w,p+w):
        for j in range(p-l,p+l):
            rho[i][j]=1500
            vp[i][j]=3500
    p=zl+2*8+2*CPML-p
    for i in range(p-l,p+l):
        for j in range(p-w,p+w):
            rho[i][j]=1500
            vp[i][j]=2000
    for i in range(p-w,p+w):
        for j in range(p-l,p+l):
            rho[i][j]=1500
            vp[i][j]=2000
    return rho,vp

#Create Initial_Model
@jit(nopython=True)
def Create_Initial_Model(xl,zl,CPML):
    rho=np.ones((xl+2*CPML+2*8,zl+2*CPML+2*8))*2000
    vp=np.ones((xl+2*CPML+2*8,zl+2*CPML+2*8))*3000
    return rho,vp

#Create Marmousi_ii Model
def Marmousi_ii(xl,zl,CPML):
    rho=np.empty((xl+2*CPML+2*8,zl+2*CPML+2*8))
    vp=np.empty((xl+2*CPML+2*8,zl+2*CPML+2*8))
    rho_data=np.load('rho_marmousi-ii.npy')
    vp_data=np.load('vp_marmousi-ii.npy')
    
    rho_1=rho_data[201:,4600:10300]
    vp_1=vp_data[201:,4600:10300]
    
    rho_8=skimage.transform.resize(rho_1,(325,712),mode='edge')
    vp_8=skimage.transform.resize(vp_1,(325,712),mode='edge')
    rho[CPML+8:CPML+8+xl,CPML+8:CPML+8+zl]=rho_8*1000
    rho[:CPML+8,:]=rho[CPML+8,:]    #top
    rho[CPML+8+xl:,:]=rho[CPML+8+xl-1,:]    #bottom
    rho[:,:8+CPML]=rho[:,8+CPML].reshape((len(rho[:,8+CPML]),-1))    #left
    rho[:,8+CPML+zl:]=rho[:,8+CPML+zl-1].reshape((len(rho[:,8+CPML+zl:]),-1))    #right
    vp[CPML+8:CPML+8+xl,CPML+8:CPML+8+zl]=vp_8*1000
    vp[:CPML+8,:]=vp[CPML+8,:]    #top
    vp[CPML+8+xl:,:]=vp[CPML+8+xl-1,:]    #bottom
    vp[:,:8+CPML]=vp[:,8+CPML].reshape((len(vp[:,8+CPML]),-1))    #left
    vp[:,8+CPML+zl:]=vp[:,8+CPML+zl-1].reshape((len(vp[:,8+CPML+zl:]),-1))    #right
    return rho,vp

#Create Initial_Marmousi_ii Model
def Initial_Marmousi_ii(rho,vp,sig):
    ivp=filters.gaussian(vp,sigma=sig)
    irho=filters.gaussian(rho,sigma=sig)
    return irho,ivp