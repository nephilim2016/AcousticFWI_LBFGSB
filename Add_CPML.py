#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 17:52:32 2018

@author: nephilim
"""
import numpy as np

#Add CPML condition
class Add_CPML():
    def __init__(self,xl,zl,CPML,vp,dx,dz,dt,Npower,k_max_CPML,alpha_max_CPML,Rcoef):
        self.xl=xl+2*CPML
        self.zl=zl+2*CPML
        self.CPML=CPML
      
        self.k_x=np.ones(self.xl+16)
        self.a_x=np.zeros(self.xl+16)
        self.b_x=np.zeros(self.xl+16)
        self.k_x_half=np.ones(self.xl+16)
        self.a_x_half=np.zeros(self.xl+16)
        self.b_x_half=np.zeros(self.xl+16)
        
        self.k_z=np.ones(self.zl+16)
        self.a_z=np.zeros(self.zl+16)
        self.b_z=np.zeros(self.zl+16)
        self.k_z_half=np.ones(self.zl+16)
        self.a_z_half=np.zeros(self.zl+16)
        self.b_z_half=np.zeros(self.zl+16)
        self.add_CPML(vp,dx,dz,dt,Npower,k_max_CPML,alpha_max_CPML,Rcoef)
        
    def add_CPML(self,vp,dx,dz,dt,Npower,k_max_CPML,alpha_max_CPML,Rcoef):
        sig_x_tmp=np.zeros(self.xl)
        k_x_tmp=np.ones(self.xl)
        alpha_x_tmp=np.zeros(self.xl)
        a_x_tmp=np.zeros(self.xl)
        b_x_tmp=np.zeros(self.xl)
        sig_x_half_tmp=np.zeros(self.xl)
        k_x_half_tmp=np.ones(self.xl)
        alpha_x_half_tmp=np.zeros(self.xl)
        a_x_half_tmp=np.zeros(self.xl)
        b_x_half_tmp=np.zeros(self.xl)
        
        sig_z_tmp=np.zeros(self.zl)
        k_z_tmp=np.ones(self.zl)
        alpha_z_tmp=np.zeros(self.zl)
        a_z_tmp=np.zeros(self.zl)
        b_z_tmp=np.zeros(self.zl)
        sig_z_half_tmp=np.zeros(self.zl)
        k_z_half_tmp=np.ones(self.zl)
        alpha_z_half_tmp=np.zeros(self.zl)
        a_z_half_tmp=np.zeros(self.zl)
        b_z_half_tmp=np.zeros(self.zl)
        
        thickness_CPML_x=self.CPML*dx
        thickness_CPML_z=self.CPML*dz
        
        sig0_x=-(Npower+1)*vp*np.log(Rcoef)/(2*thickness_CPML_x)
        sig0_z=-(Npower+1)*vp*np.log(Rcoef)/(2*thickness_CPML_z)
        
        xoriginleft=thickness_CPML_x
        xoriginright=(self.xl-1)*dx-thickness_CPML_x
        for i in range(self.xl):
            xval=dx*i
            abscissa_in_CPML=xoriginleft-xval
            if abscissa_in_CPML>=0:
                abscissa_normalized=abscissa_in_CPML/thickness_CPML_x
                sig_x_tmp[i]=sig0_x*abscissa_normalized**Npower
                k_x_tmp[i]=1+(k_max_CPML-1)*abscissa_normalized**Npower
                alpha_x_tmp[i]=alpha_max_CPML*(1-abscissa_normalized)+0.1*alpha_max_CPML
                
            abscissa_in_CPML=xoriginleft-xval-dx/2
            if abscissa_in_CPML>=0:
                abscissa_normalized=abscissa_in_CPML/thickness_CPML_x
                sig_x_half_tmp[i]=sig0_x*abscissa_normalized**Npower
                k_x_half_tmp[i]=1+(k_max_CPML-1)*abscissa_normalized**Npower
                alpha_x_half_tmp[i]=alpha_max_CPML*(1-abscissa_normalized)+0.1*alpha_max_CPML
                
            abscissa_in_CPML=xval-xoriginright
            if abscissa_in_CPML>=0:
                abscissa_normalized=abscissa_in_CPML/thickness_CPML_x
                sig_x_tmp[i]=sig0_x*abscissa_normalized**Npower
                k_x_tmp[i]=1+(k_max_CPML-1)*abscissa_normalized**Npower
                alpha_x_tmp[i]=alpha_max_CPML*(1-abscissa_normalized)+0.1*alpha_max_CPML
            
            abscissa_in_CPML=xval+dx/2-xoriginright
            if abscissa_in_CPML>=0:
                abscissa_normalized=abscissa_in_CPML/thickness_CPML_x
                sig_x_half_tmp[i]=sig0_x*abscissa_normalized**Npower
                k_x_half_tmp[i]=1+(k_max_CPML-1)*abscissa_normalized**Npower
                alpha_x_half_tmp[i]=alpha_max_CPML*(1-abscissa_normalized)+0.1*alpha_max_CPML
    
            b_x_tmp[i]=np.exp(-(sig_x_tmp[i]/k_x_tmp[i]+alpha_x_tmp[i])*dt)
            b_x_half_tmp[i]=np.exp(-(sig_x_half_tmp[i]/k_x_half_tmp[i]+alpha_x_half_tmp[i])*dt)
            if abs(sig_x_tmp[i]>1e-6):
                a_x_tmp[i]=sig_x_tmp[i]*(b_x_tmp[i]-1)/(k_x_tmp[i]*(sig_x_tmp[i]+k_x_tmp[i]*alpha_x_tmp[i]))
            if abs(sig_x_half_tmp[i]>1e-6):
                a_x_half_tmp[i]=sig_x_half_tmp[i]*(b_x_half_tmp[i]-1)/(k_x_half_tmp[i]*(sig_x_half_tmp[i]+k_x_half_tmp[i]*alpha_x_half_tmp[i]))

        zoriginbottom=thickness_CPML_z
        zorigintop=(self.zl-1)*dz-thickness_CPML_z
        for i in range(self.zl):
            zval=dz*i
            abscissa_in_CPML=zoriginbottom-zval
            if abscissa_in_CPML>=0:
                abscissa_normalized=abscissa_in_CPML/thickness_CPML_z
                sig_z_tmp[i]=sig0_z*abscissa_normalized**Npower
                k_z_tmp[i]=1+(k_max_CPML-1)*abscissa_normalized**Npower
                alpha_z_tmp[i]=alpha_max_CPML*(1-abscissa_normalized)+0.1*alpha_max_CPML
            
            abscissa_in_CPML=zoriginbottom-zval-dz/2
            if abscissa_in_CPML>=0:
                abscissa_normalized=abscissa_in_CPML/thickness_CPML_z
                sig_z_half_tmp[i]=sig0_z*abscissa_normalized**Npower
                k_z_half_tmp[i]=1+(k_max_CPML-1)*abscissa_normalized**Npower
                alpha_z_half_tmp[i]=alpha_max_CPML*(1-abscissa_normalized)+0.1*alpha_max_CPML
    
            abscissa_in_CPML=zval-zorigintop
            if abscissa_in_CPML>=0:
                abscissa_normalized=abscissa_in_CPML/thickness_CPML_z
                sig_z_tmp[i]=sig0_z*abscissa_normalized**Npower
                k_z_tmp[i]=1+(k_max_CPML-1)*abscissa_normalized**Npower
                alpha_z_tmp[i]=alpha_max_CPML*(1-abscissa_normalized)+0.1*alpha_max_CPML
            
            abscissa_in_CPML=zval+dz/2-zorigintop
            if abscissa_in_CPML>=0:
                abscissa_normalized=abscissa_in_CPML/thickness_CPML_z
                sig_z_half_tmp[i]=sig0_z*abscissa_normalized**Npower
                k_z_half_tmp[i]=1+(k_max_CPML-1)*abscissa_normalized**Npower
                alpha_z_half_tmp[i]=alpha_max_CPML*(1-abscissa_normalized)+0.1*alpha_max_CPML
    
            b_z_tmp[i]=np.exp(-(sig_z_tmp[i]/k_z_tmp[i]+alpha_z_tmp[i])*dt)
            b_z_half_tmp[i]=np.exp(-(sig_z_half_tmp[i]/k_z_half_tmp[i]+alpha_z_half_tmp[i])*dt)
            if abs(sig_z_tmp[i]>1e-6):
                a_z_tmp[i]=sig_z_tmp[i]*(b_z_tmp[i]-1)/(k_z_tmp[i]*(sig_z_tmp[i]+k_z_tmp[i]*alpha_z_tmp[i]))
            if abs(sig_z_half_tmp[i]>1e-6):
                a_z_half_tmp[i]=sig_z_half_tmp[i]*(b_z_half_tmp[i]-1)/(k_z_half_tmp[i]*(sig_z_half_tmp[i]+k_z_half_tmp[i]*alpha_z_half_tmp[i]))
        self.a_x[8:-8]=a_x_tmp
        self.b_x[8:-8]=b_x_tmp
        self.k_x[8:-8]=k_x_tmp
        self.a_z[8:-8]=a_z_tmp
        self.b_z[8:-8]=b_z_tmp
        self.k_z[8:-8]=k_z_tmp
        self.a_x_half[8:-8]=a_x_half_tmp
        self.b_x_half[8:-8]=b_x_half_tmp
        self.k_x_half[8:-8]=k_x_half_tmp
        self.a_z_half[8:-8]=a_z_half_tmp
        self.b_z_half[8:-8]=b_z_half_tmp
        self.k_z_half[8:-8]=k_z_half_tmp
