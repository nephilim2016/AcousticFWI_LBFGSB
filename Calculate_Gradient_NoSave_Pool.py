#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 15:38:43 2018

@author: nephilim
"""
from multiprocessing import Pool,Manager,Process
import numpy as np
import time
import Add_CPML
import Wavelet
import Time_loop
import Reverse_time_loop
import Cross_Gradient

def calculate_gradient(rho,vp,index,CPML_Params,para):
    #Get Forward Params
    t=np.arange(para.k_max)*para.dt
    f=Wavelet.ricker(t,para.ricker_freq)
    #True Model Profile Data
    data=para.data[index]
    #Get Forward Data ----> <Generator>
    Forward_data=Time_loop.time_loop(para.xl,para.zl,para.dx,para.dz,para.dt,\
                                                  rho,vp,CPML_Params,f,para.k_max,\
                                                  para.source_site[index],para.ref_pos)
    #Get Generator Data
    V_data=[]
    idata=np.zeros((para.k_max,len(para.ref_pos)))
    for i in range(para.k_max):
        tmp=Forward_data.__next__()
        V_data.append(np.array(tmp[0]))
        idata[i,:]=tmp[1]
    #Get Residual Data
    rhs_data=idata-data
    #Get Reversion Data ----> <Generator>
    Reverse_data=Reverse_time_loop.reverse_time_loop(para.xl,para.zl,para.dx,para.dz,\
                                                     para.dt,rho,vp,CPML_Params,para.k_max,\
                                                     para.ref_pos,rhs_data)
    #Get Generator Data
    RT_P_data=[]
    RT_V_data=[]
    RT_W_data=[]
    for i in range(para.k_max):
        tmp=Reverse_data.__next__()
        RT_P_data.append(np.array(tmp[0]))
        RT_V_data.append(np.array(tmp[1]))
        RT_W_data.append(np.array(tmp[2]))
    RT_P_data.reverse()
    RT_V_data.reverse()
    RT_W_data.reverse()     
    #Calculate Gradient
    time_sum=np.zeros((para.xl+2*8+2*CPML_Params.CPML,para.zl+2*8+2*CPML_Params.CPML))
    dudx_sum=np.zeros((para.xl+2*8+2*CPML_Params.CPML,para.zl+2*8+2*CPML_Params.CPML))
    dudz_sum=np.zeros((para.xl+2*8+2*CPML_Params.CPML,para.zl+2*8+2*CPML_Params.CPML))
    for k in range(1,para.k_max-1):
        u1=V_data[k+1]
        u0=V_data[k-1]
        p1=RT_P_data[k]
        v1=RT_V_data[k]
        w1=RT_W_data[k]
        time_sum+=p1*(u1-u0)/para.dt/2
        
        u=V_data[k]
        dudx_sum[1:u.shape[0]-1,:]+=v1[1:u.shape[0]-1,:]*(u[2:u.shape[0],:]-u[:u.shape[0]-2,:])/para.dx/2
        dudz_sum[:,1:u.shape[1]-1]+=w1[:,1:u.shape[1]-1]*(u[:,2:u.shape[1]]-u[:,:u.shape[1]-2])/para.dz/2
    g_rho=1/(rho**2*vp**2)*time_sum+1/(rho**2)*(dudx_sum+dudz_sum)
    g_vp=2/(rho*vp**3)*time_sum
    return rhs_data.flatten(),g_rho.flatten(),g_vp.flatten()    
  
def calculate_toltal_variation_model(rho,vp,dx,dz):
    normgrad_rho,g_TolVar_rho=cal_toltal_variation(rho,dx,dz)
    normgrad_vp,g_TolVar_vp=cal_toltal_variation(vp,dx,dz)
    f_TolVar_rho=np.sum(normgrad_rho.flatten())
    f_TolVar_vp=np.sum(normgrad_vp.flatten())
    return f_TolVar_rho,f_TolVar_vp,g_TolVar_rho.flatten(),g_TolVar_vp.flatten()

def cal_toltal_variation(u,dx,dz):
    epsilon=1e-16
    dudx=np.zeros(u.shape)
    dudz=np.zeros(u.shape)
    gx=np.zeros(u.shape)
    gz=np.zeros(u.shape)
    dudx[1:u.shape[0]-1,:]=(u[2:u.shape[0],:]-u[:u.shape[0]-2,:])/dx/2
    dudz[:,1:u.shape[0]-1]=(u[:,2:u.shape[0]]-u[:,:u.shape[0]-2])/dz/2
    normgrad=np.sqrt(dudx**2+dudz**2+epsilon)
    fx=dudx/normgrad
    fz=dudz/normgrad
    gx[1:fx.shape[0]-1,:]=(fx[2:fx.shape[0],:]-fx[:fx.shape[0]-2,:])/dx/2
    gz[:,1:fz.shape[0]-1]=(fz[:,2:fz.shape[0]]-fz[:,:fz.shape[0]-2])/dz/2
    div=-(gx+gz)
    return normgrad,div

def misfit(data,para):  
    #Get rho & vp
    rho=data[:int(len(data)/2)].reshape((para.xl+2*para.CPML+16,-1))
    vp=data[int(len(data)/2):].reshape((para.xl+2*para.CPML+16,-1))
    #Get Toltal Variation Params
    lambda_rho=para.lambda_rho
    lambda_vp=para.lambda_vp
    lambda_CroGra=para.lambda_CroGra
    start_time=time.time()
    #Create CPML
    vp_max=max(vp.flatten())
    CPML_Params=Add_CPML.Add_CPML(para.xl,para.zl,para.CPML,vp_max,para.dx,para.dz,para.dt,\
                                  para.Npower,para.k_max_CPML,para.alpha_max_CPML,para.Rcoef)
    #Calculate Gradient
    g_rho=0.0
    g_vp=0.0
    rhs=[]
    pool=Pool(processes=8)
    res_l=[]
    for index,value in enumerate(para.source_site):
        res=pool.apply_async(calculate_gradient,args=(rho,vp,index,CPML_Params,para))
        res_l.append(res)
    pool.close()
    pool.join()
    for res in res_l:
        result=res.get()
        rhs.append(result[0])
        g_rho+=result[1]
        g_vp+=result[2]
        del result
    del res_l
    pool.terminate()
    #Get Profile Data
    rhs=np.array(rhs)
    #Get Function Error
    f=0.5*np.linalg.norm(rhs.flatten(),2)**2
    #Get Toltal Variation
    if para.TolVar_key:
        f_TolVar_rho,f_TolVar_vp,g_TolVar_rho,g_TolVar_vp=calculate_toltal_variation_model(rho,vp,para.dx,para.dz)
        f_TolVar_rho/=10000
        f_TolVar_vp/=10000
    else:
        f_TolVar_rho=0.0
        f_TolVar_vp=0.0
        g_TolVar_rho=np.zeros(g_rho.shape)
        g_TolVar_vp=np.zeros(g_vp.shape)
    
    #Get Cross-Gradient
    if para.CroGra_key:
        f_CroGra,g_CroGra_rho,g_CroGra_vp=Cross_Gradient.Cross_Gradient(rho,vp,para.dx,para.dz)
    else:
        f_CroGra=0.0
        g_CroGra_rho=np.zeros(g_rho.shape)
        g_CroGra_vp=np.zeros(g_vp.shape)
        
    print('''****fd=%s,g_rho=%s,g_vp=%s\n
          f_TolVar_rho=%s,g_TolVar_rho=%s\n
          f_TolVar_vp=%s,g_TolVar_vp=%s\n
          f_CroGra=%s,g_CroGra_rho=%s,g_CroGra_vp=%s'''\
          %(f,np.linalg.norm(g_rho,2),np.linalg.norm(g_vp,2),\
            f_TolVar_rho,np.linalg.norm(g_TolVar_rho,2),\
            f_TolVar_vp,np.linalg.norm(g_TolVar_vp,2),\
            f_CroGra,np.linalg.norm(g_CroGra_rho,2),\
            np.linalg.norm(g_CroGra_vp,2)))
    #Update Lambda
    lambda_rho*=(f_TolVar_rho/(f+f_TolVar_rho))
    lambda_vp*=(f_TolVar_vp/(f+f_TolVar_vp))
    lambda_CroGra*=(f_CroGra/(f+f_CroGra))
    print('***%s,%s,%s***'%(lambda_rho,lambda_vp,lambda_CroGra))
    #Get toltal error
    f+=(lambda_rho*f_TolVar_rho+lambda_vp*f_TolVar_vp+lambda_CroGra*f_CroGra)
    g_rho+=lambda_rho*g_TolVar_rho+lambda_CroGra*g_CroGra_rho
    g_vp+=lambda_vp*g_TolVar_vp+lambda_CroGra*g_CroGra_vp
    g=np.hstack((g_rho,g_vp))
    print('Misfit elapsed time is %s seconds !'%str(time.time()-start_time))
    return f,g


