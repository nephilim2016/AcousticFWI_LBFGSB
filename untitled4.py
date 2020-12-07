#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 17:12:04 2020

@author: nephilim
"""

import numpy as np
from matplotlib import pyplot,cm
import Create_Model

def PlotFigure(data):
    pyplot.figure()
    pyplot.imshow(data,extent=(0,1,0,1),vmin=np.min(data),vmax=np.max(data))
    
def ResultLine(data):
    data_line=[]
    for idx in range(20,121):
        data_line.append(data[idx,idx])
    return data_line

if __name__=='__main__':
    data=np.load('./30Hz_imodel_file/199_imodel.npy')
    rho=data[:int(len(data)/2)].reshape((141,-1))
    vp=data[int(len(data)/2):].reshape((141,-1))
    PlotFigure(rho)
    PlotFigure(vp)
    
    rho_true,vp_true=Create_Model.Abnormal_Model(101,101,12)
    rho_true_line=ResultLine(rho_true)
    vp_true_line=ResultLine(vp_true)
    rho_BFGSB_line=ResultLine(rho)
    vp_BFGSB_line=ResultLine(vp)
    
    pyplot.figure()
    pyplot.plot(rho_true_line)
    pyplot.plot(rho_BFGSB_line)
    
    pyplot.figure()
    pyplot.plot(vp_true_line)
    pyplot.plot(vp_BFGSB_line)