#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 17:54:29 2018

@author: nephilim
"""
from numba import jit
import numpy as np

#Forward modelling ------ update_vw
@jit(nopython=True)            
def update_vw(xl,zl,dx,dz,dt,rho,vp,CPML,a_x,a_z,b_x,b_z,k_x,k_z,v,w,p,memory_dp_dx,memory_dp_dz):
    c1=1.23404
    c2=-0.10665
    c3=0.0230364
    c4=-0.00534239
    c5=0.00107727
    c6=-1.6642e-4
    c7=1.7022e-5
    c8=-8.5235e-7
    x_len=xl+16+CPML*2
    z_len=zl+16+CPML*2
    for j in range(8,z_len-8):
        for i in range(8,x_len-8):
            value_dp_dx=(c1*(p[i+1][j]-p[i-0][j])+c2*(p[i+2][j]-p[i-1][j])+\
                             c3*(p[i+3][j]-p[i-2][j])+c4*(p[i+4][j]-p[i-3][j])+\
                             c5*(p[i+5][j]-p[i-4][j])+c6*(p[i+6][j]-p[i-5][j])+\
                             c7*(p[i+7][j]-p[i-6][j])+c8*(p[i+8][j]-p[i-7][j]))/dx
                         
            if (i>=CPML+8) and (i<x_len-CPML-8):
                v[i][j]+=value_dp_dx*dt/rho[i][j]
                
            elif i<CPML+8:
                memory_dp_dx[i][j]=b_x[i]*memory_dp_dx[i][j]+a_x[i]*value_dp_dx
                value_dp_dx=value_dp_dx/k_x[i]+memory_dp_dx[i][j]
                v[i][j]+=value_dp_dx*dt/rho[i][j]
                
            elif i>=x_len-CPML-8:
                memory_dp_dx[i-xl][j]=b_x[i]*memory_dp_dx[i-xl][j]+a_x[i]*value_dp_dx
                value_dp_dx=value_dp_dx/k_x[i]+memory_dp_dx[i-xl][j]
                v[i][j]+=value_dp_dx*dt/rho[i][j]
                
###############################################################################
    for j in range(8,z_len-8):
        for i in range(8,x_len-8):
            value_dp_dz=(c1*(p[i][j+1]-p[i][j-0])+c2*(p[i][j+2]-p[i][j-1])+\
                         c3*(p[i][j+3]-p[i][j-2])+c4*(p[i][j+4]-p[i][j-3])+\
                         c5*(p[i][j+5]-p[i][j-4])+c6*(p[i][j+6]-p[i][j-5])+\
                         c7*(p[i][j+7]-p[i][j-6])+c8*(p[i][j+8]-p[i][j-7]))/dz
                         
            if (j>=CPML+8) and (j<z_len-CPML-8):
                w[i][j]+=value_dp_dz*dt/rho[i][j]
                
            elif j<CPML+8:
                memory_dp_dz[i][j]=b_z[j]*memory_dp_dz[i][j]+a_z[j]*value_dp_dz
                value_dp_dz=value_dp_dz/k_z[j]+memory_dp_dz[i][j]
                w[i][j]+=value_dp_dz*dt/rho[i][j]
                
            elif j>=z_len-CPML-8:
                memory_dp_dz[i][j-zl]=b_z[j]*memory_dp_dz[i][j-zl]+a_z[j]*value_dp_dz
                value_dp_dz=value_dp_dz/k_z[j]+memory_dp_dz[i][j-zl]
                w[i][j]+=value_dp_dz*dt/rho[i][j]  

    return v,w

#Forward modelling ------ update_p
@jit(nopython=True)            
def update_p(xl,zl,dx,dz,dt,rho,vp,CPML,a_x,a_z,b_x,b_z,k_x,k_z,v,w,p,memory_dv_dx,memory_dw_dz):
    c1=1.23404
    c2=-0.10665
    c3=0.0230364
    c4=-0.00534239
    c5=0.00107727
    c6=-1.6642e-4
    c7=1.7022e-5
    c8=-8.5235e-7
    k_value=rho*vp**2
    x_len=xl+16+CPML*2
    z_len=zl+16+CPML*2
    for j in range(8,z_len-8):
        for i in range(8,x_len-8):
            value_dv_dx=(c1*(v[i+0][j]-v[i-1][j])+c2*(v[i+1][j]-v[i-2][j])+\
                         c3*(v[i+2][j]-v[i-3][j])+c4*(v[i+3][j]-v[i-4][j])+\
                         c5*(v[i+4][j]-v[i-5][j])+c6*(v[i+5][j]-v[i-6][j])+\
                         c7*(v[i+6][j]-v[i-7][j])+c8*(v[i+7][j]-v[i-8][j]))/dx
         
            value_dw_dz=(c1*(w[i][j+0]-w[i][j-1])+c2*(w[i][j+1]-w[i][j-2])+\
                         c3*(w[i][j+2]-w[i][j-3])+c4*(w[i][j+3]-w[i][j-4])+\
                         c5*(w[i][j+4]-w[i][j-5])+c6*(w[i][j+5]-w[i][j-6])+\
                         c7*(w[i][j+6]-w[i][j-7])+c8*(w[i][j+7]-w[i][j-8]))/dz
                         

            if (i>=CPML+8) and (i<x_len-CPML-8) and (j>=CPML+8) and (j<z_len-CPML-8):
                p[i][j]+=k_value[i][j]*(value_dv_dx+value_dw_dz)*dt
                
            elif (i<CPML+8) and (j>=CPML+8) and (j<z_len-CPML-8):
                memory_dv_dx[i][j]=b_x[i]*memory_dv_dx[i][j]+a_x[i]*value_dv_dx
                value_dv_dx=value_dv_dx/k_x[i]+memory_dv_dx[i][j]
                p[i][j]+=k_value[i][j]*(value_dv_dx+value_dw_dz)*dt
                
            elif (i>=x_len-CPML-8) and (j>=CPML+8) and (j<z_len-CPML-8):
                memory_dv_dx[i-xl][j]=b_x[i]*memory_dv_dx[i-xl][j]+a_x[i]*value_dv_dx
                value_dv_dx=value_dv_dx/k_x[i]+memory_dv_dx[i-xl][j]
                p[i][j]+=k_value[i][j]*(value_dv_dx+value_dw_dz)*dt
                
            elif (j<CPML+8) and (i>=CPML+8) and (i<x_len-CPML-8):
                memory_dw_dz[i][j]=b_z[j]*memory_dw_dz[i][j]+a_z[j]*value_dw_dz
                value_dw_dz=value_dw_dz/k_z[j]+memory_dw_dz[i][j]
                p[i][j]+=k_value[i][j]*(value_dv_dx+value_dw_dz)*dt
                
            elif (j>=z_len-CPML-8) and (i>=CPML+8) and (i<x_len-CPML-8):
                memory_dw_dz[i][j-zl]=b_z[j]*memory_dw_dz[i][j-zl]+a_z[j]*value_dw_dz
                value_dw_dz=value_dw_dz/k_z[j]+memory_dw_dz[i][j-zl]
                p[i][j]+=k_value[i][j]*(value_dv_dx+value_dw_dz)*dt
                
            elif (i<CPML+8) and (j<CPML+8):
                memory_dv_dx[i][j]=b_x[i]*memory_dv_dx[i][j]+a_x[i]*value_dv_dx
                value_dv_dx=value_dv_dx/k_x[i]+memory_dv_dx[i][j]
                
                memory_dw_dz[i][j]=b_z[j]*memory_dw_dz[i][j]+a_z[j]*value_dw_dz
                value_dw_dz=value_dw_dz/k_z[j]+memory_dw_dz[i][j]
                
                p[i][j]+=k_value[i][j]*(value_dv_dx+value_dw_dz)*dt
                
            elif (i<CPML+8) and (j>=z_len-CPML-8):
                memory_dv_dx[i][j]=b_x[i]*memory_dv_dx[i][j]+a_x[i]*value_dv_dx
                value_dv_dx=value_dv_dx/k_x[i]+memory_dv_dx[i][j]
                
                memory_dw_dz[i][j-zl]=b_z[j]*memory_dw_dz[i][j-zl]+a_z[j]*value_dw_dz
                value_dw_dz=value_dw_dz/k_z[j]+memory_dw_dz[i][j-zl]
                
                p[i][j]+=k_value[i][j]*(value_dv_dx+value_dw_dz)*dt
                
            elif (i>=x_len-CPML-8) and (j<CPML+8):
                memory_dv_dx[i-xl][j]=b_x[i]*memory_dv_dx[i-xl][j]+a_x[i]*value_dv_dx
                value_dv_dx=value_dv_dx/k_x[i]+memory_dv_dx[i-xl][j]
                
                memory_dw_dz[i][j]=b_z[j]*memory_dw_dz[i][j]+a_z[j]*value_dw_dz
                value_dw_dz=value_dw_dz/k_z[j]+memory_dw_dz[i][j]
                
                p[i][j]+=k_value[i][j]*(value_dv_dx+value_dw_dz)*dt
                
            elif (i>=x_len-CPML-8) and (j>=z_len-CPML-8):
                memory_dv_dx[i-xl][j]=b_x[i]*memory_dv_dx[i-xl][j]+a_x[i]*value_dv_dx
                value_dv_dx=value_dv_dx/k_x[i]+memory_dv_dx[i-xl][j]
                
                memory_dw_dz[i][j-zl]=b_z[j]*memory_dw_dz[i][j-zl]+a_z[j]*value_dw_dz
                value_dw_dz=value_dw_dz/k_z[j]+memory_dw_dz[i][j-zl]
                
                p[i][j]+=k_value[i][j]*(value_dv_dx+value_dw_dz)*dt
    return p

#Forward modelling ------ timeloop
def time_loop(xl,zl,dx,dz,dt,rho,vp,CPML_Params,f,k_max,source_site,ref_pos):
    CPML=CPML_Params.CPML        
    p=np.zeros((xl+2*CPML+16,zl+2*CPML+16))
    v=np.zeros((xl+2*CPML+16,zl+2*CPML+16))
    w=np.zeros((xl+2*CPML+16,zl+2*CPML+16))
        
    memory_dp_dx=np.zeros((2*CPML+16,zl+2*CPML+16))
    memory_dp_dz=np.zeros((xl+2*CPML+16,2*CPML+16))
    memory_dv_dx=np.zeros((2*CPML+16,zl+2*CPML+16))
    memory_dw_dz=np.zeros((xl+2*CPML+16,2*CPML+16))
    
    a_x=CPML_Params.a_x
    b_x=CPML_Params.b_x
    k_x=CPML_Params.k_x
    a_z=CPML_Params.a_z
    b_z=CPML_Params.b_z
    k_z=CPML_Params.k_z
    a_x_half=CPML_Params.a_x_half
    b_x_half=CPML_Params.b_x_half
    k_x_half=CPML_Params.k_x_half
    a_z_half=CPML_Params.a_z_half
    b_z_half=CPML_Params.b_z_half
    k_z_half=CPML_Params.k_z_half
    
    
    for tt in range(k_max):
        v,w=update_vw(xl,zl,dx,dz,dt,rho,vp,CPML,a_x_half,a_z_half,b_x_half,b_z_half,k_x_half,k_z_half,v,w,p,memory_dp_dx,memory_dp_dz)
        p=update_p(xl,zl,dx,dz,dt,rho,vp,CPML,a_x,a_z,b_x,b_z,k_x,k_z,v,w,p,memory_dv_dx,memory_dw_dz)
        p[source_site[0]][source_site[1]]+=f[tt]
        yield p.tolist(),p[ref_pos[:,0],ref_pos[:,1]]
