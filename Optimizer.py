#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 20:46:03 2020

@author: nephilim
"""

import numpy as np
from scipy.optimize import minpack2
from collections import deque
import os
import shutil

class para():
    def __init__(self):
        pass
class options():
    def __init__(self):
        pass
    
class L_BFGS_B(para,options):
    def __init__(self,fh,x0,l,u):
        super().__init__()
        self.fh=fh
        self.x0=x0
        self.l=l
        self.u=u
    def l_bfgs_b(self):
        if not hasattr(options,'m'):
            options.m=5
        if not hasattr(options,'epsg'):
            options.epsg=1e-5
        if not hasattr(options,'epsf'):
            options.epsf=1e7
        if not hasattr(options,'max_iter'):
            options.max_iter=50
        if not hasattr(options,'alpha_linesearch'):
            options.alpha_linesearch=1e-4
        if not hasattr(options,'beta_linesearch'):
            options.beta_linesearch=0.9
        if not hasattr(options,'max_steplength'):
            options.max_steplength=1e8
        if not hasattr(options,'xtol_minpack'):
            options.xtol_minpack=1e-5
        if not hasattr(options,'max_iter_linesearch'):
            options.max_iter_linesearch=10
        if not hasattr(options,'eps_SY'):
            options.eps_SY=np.finfo(float).eps
        
        if not os.path.exists('./%sHz_imodel_file'%para.ricker_freq):
            os.makedirs('./%sHz_imodel_file'%para.ricker_freq)
        else:
            shutil.rmtree('./%sHz_imodel_file'%para.ricker_freq)
            os.makedirs('./%sHz_imodel_file'%para.ricker_freq)
            
        n=self.x0.size
        if self.x0.dtype!=np.float64:
            x=self.x0.astype(np.float64,copy=True)
            x=np.clip(x,self.l,self.u)
        else:
            x=np.clip(self.x0,self.l,self.u)
        S=deque()
        Y=deque()
        W=np.zeros([n,1])
        M=np.zeros([1,1])
        theta=1
        epsmch=np.finfo(1.0).resolution
        
        f0,g=self.fh(x)
        idx=0
        x_history=[]
        while np.max(np.abs(np.clip(x-g,self.l,self.u)-x))>options.epsg and idx<options.max_iter:
            oldf0=f0
            oldx=x.copy()
            oldg=g.copy()
            dictCP=self.compute_Cauchy_point(x,g,W,M,theta)
            dictMinMod=self.minimize_model(x,dictCP['xc'],dictCP['c'],g,W,M,theta)
        
            d=dictMinMod['xbar']-x
            max_stpl=self.max_allowed_steplength(x,d,options.max_steplength)
            steplength=self.line_search(x,f0,g,d,idx,max_stpl,self.fh,options.alpha_linesearch,\
                                        options.beta_linesearch,options.xtol_minpack,\
                                        options.max_iter_linesearch)
        
            if steplength==None:
                if len(S)==0:
                    #Hessian already rebooted: abort.
                    print("Error: can not compute new steplength : abort")
                    return {'x':x, 'f':self.fh(x)[0], 'df':self.fh(x)[1]}
                else:
                    #Reboot BFGS-Hessian:
                    S.clear()
                    Y.clear()
                    W=np.zeros([n,1])
                    M=np.zeros([1,1])
                    theta=1
            else:
                x+=steplength*d
                x_history.append(x.copy())
                
            
                f0,g=self.fh(x)
                W,M,theta=self.update_SY(x-oldx,g-oldg,S,Y,options.m,\
                                         W,M,theta,options.eps_SY)
        
        
                print("Iteration #%d (max: %d): ||x||=%.3e, f(x)=%.3e, ||df(x)||=%.3e, cdt_arret=%.3e (eps=%.3e)"%\
                      (idx,options.max_iter,np.linalg.norm(x,np.inf),f0,np.linalg.norm(g,np.inf),\
                       np.max(np.abs(np.clip(x-g,self.l,self.u)-x)),options.epsg))
                    
                np.save('./%sHz_imodel_file/%d_imodel.npy'%(para.ricker_freq,idx),x)
                np.save('./%sHz_imodel_file/%d_info.npy'%(para.ricker_freq,idx),[idx,f0,np.linalg.norm(g,2)])
                if ((oldf0-f0)/max(abs(oldf0),abs(f0),1)<epsmch*options.epsf):
                    print("Relative reduction of f below tolerence: abort.")
                    break
                idx+=1
            # print(np.max(np.abs(np.clip(x-g,self.l,self.u)-x)))
        
        if idx==options.max_iter:
            print("Maximum iteration reached.")
        
        return {'x_history':x_history,'x':x, 'f':f0, 'df':g}
    
    def compute_Cauchy_point(self,x,g,W,M,theta):
        eps_f_sec=1e-30 
        t=np.empty(x.size)
        d=np.empty(x.size)
        x_cp=x.copy()
        for idx in range(x.size):
            if g[idx]<0:
                t[idx]=(x[idx]-self.u[idx])/g[idx]
            elif g[idx]>0:
                t[idx]=(x[idx]-self.l[idx])/g[idx]
            else:
                t[idx]=np.inf
            if t[idx]==0:
                d[idx]=0
            else:
                d[idx]=-g[idx]

        F=np.argsort(t)
        F=[i for i in F if t[i] >0]
        t_old=0
        F_i=0
        b=F[0]
        t_min=t[b]
        Dt=t_min
    
        p=np.transpose(W).dot(d)
        c=np.zeros(p.size)
        f_prime=-d.dot(d)
        f_second=-theta*f_prime-p.dot(M.dot(p))
        f_sec0=f_second
        Dt_min=-f_prime/f_second

        while Dt_min>=Dt and F_i<len(F):
            if d[b]>0:
                x_cp[b]=self.u[b]
            elif d[b]<0:
                x_cp[b]=self.l[b]
            x_bcp=x_cp[b]
        
            zb=x_bcp-x[b]
            c+=Dt*p
            W_b=W[b,:]
            g_b=g[b]
        
            f_prime+=Dt*f_second+g_b*(g_b+theta*zb-W_b.dot(M.dot(c)))
            f_second-=g_b*(g_b*theta+W_b.dot(M.dot(2*p+g_b*W_b)))
            f_second=min(f_second, eps_f_sec*f_sec0)
        
            Dt_min=-f_prime/(f_second+1e-16)
            p+=g_b*W_b
            d[b]=0
            t_old=t_min
            F_i+=1
        
            if F_i<len(F):
                b=F[F_i]
                t_min=t[b]
                Dt=t_min-t_old
            else:
                t_min=np.inf

        Dt_min=0 if Dt_min<0 else Dt_min
        t_old+=Dt_min
    
        for idx in range(x.size):
            if t[idx]>=t_min:
                x_cp[idx]=x[idx]+t_old*d[idx]
    
        F=[i for i in F if t[i]!=t_min]
            
        c+=Dt_min*p
        return {'xc':x_cp,'c':c,'F':F}
    
    def minimize_model(self,x,xc,c,g,W,M,theta):
        invThet=1.0/theta
     
        Z=list()
        free_vars=list()
        n=xc.size
        unit=np.zeros(n)
        for idx in range(n):
            unit[idx]=1
            if ((xc[idx]!=self.u[idx]) and (xc[idx]!=self.l[idx])):
                free_vars.append(idx)
                Z.append(unit.copy())
            unit[idx]=0
        
        if len(free_vars)==0:
            return {'xbar':xc}
    
        Z=np.asarray(Z).T
        WTZ=W.T.dot(Z)
    
        rHat=[(g+theta*(xc-x)-W.dot(M.dot(c)))[ind] for ind in free_vars]
        v=WTZ.dot(rHat)
        v=M.dot(v)
    
        N=invThet*WTZ.dot(np.transpose(WTZ))
        N=np.eye(N.shape[0])-M.dot(N)
        v=np.linalg.solve(N, v)
    
        dHat=-invThet*(rHat+invThet*np.transpose(WTZ).dot(v))
    
        #Find alpha
        alpha_star=1
        for i in range(len(free_vars)):
            idx=free_vars[i]
            if dHat[i]>0:
                alpha_star=min(alpha_star,(self.u[idx]-xc[idx])/dHat[i])
            elif dHat[i]<0:
                alpha_star=min(alpha_star,(self.l[idx]-xc[idx])/dHat[i])
    
        d_star=alpha_star*dHat;
        xbar=xc;
        for i in range(len(free_vars)):
            idx=free_vars[i];
            xbar[idx]+=d_star[i]

        return {'xbar':xbar}
    
    def max_allowed_steplength(self,x,d,max_steplength):
        max_stpl=max_steplength
        for idx in range(x.size):
            if d[idx]>0:
                max_stpl=min(max_stpl,(self.u[idx]-x[idx])/d[idx])
            elif d[idx]<0 :
                max_stpl=min(max_stpl,(self.l[idx]-x[idx])/d[idx])
    
        return max_stpl

    def line_search(self,x0,f0,g0,d,above_iter,max_steplength,func,alpha=1e-4,\
                    beta=0.9,xtol_minpack=1e-5,max_iter=30):
        steplength_0=1 if max_steplength>1 else 0.5*max_steplength
        f_m1=f0
        dphi=g0.dot(d)
        dphi_m1=dphi
        idx=0

        if(above_iter==0):
            max_steplength=1.0
            steplength_0=min(1.0/np.sqrt(d.dot(d)),1.0)

        isave=np.zeros((2,),np.intc)
        dsave=np.zeros((13,),float)
        task=b'START'
    
        while idx<max_iter:
            steplength,f0,dphi,task=minpack2.dcsrch(steplength_0,f_m1,dphi_m1,\
                                                    alpha,beta,xtol_minpack,task,\
                                                    0,max_steplength,isave,dsave)
            if task[:2] == b'FG':
                steplength_0=steplength
                f_m1,g_m1=func(x0+steplength*d)
                dphi_m1=g_m1.dot(d)
                # print(f_m1)
            else:
                break
        else:
            # max_iter reached, the line search did not converge
            steplength=None
    
    
        if task[:5]==b'ERROR' or task[:4]==b'WARN':
            if task[:21] != b'WARNING: STP = STPMAX':
                print(task)
                steplength = None  # failed
        
        return steplength

    def update_SY(self,sk,yk,S,Y,m,W,M,thet,eps=2.2e-16):
        sTy=sk.dot(yk)
        yTy=yk.dot(yk)
        if (sTy>eps*yTy):
            S.append(sk)
            Y.append(yk)
            if len(S)>m :
                S.popleft()
                Y.popleft()
            Sarray=np.asarray(S).T
            Yarray=np.asarray(Y).T
            STS=np.transpose(Sarray).dot(Sarray)
            L=np.transpose(Sarray).dot(Yarray)
            D=np.diag(-np.diag(L))
            L=np.tril(L,-1)
        
            thet=yTy/sTy
            W=np.hstack([Yarray, thet*Sarray])
            M=np.linalg.inv(np.hstack([np.vstack([D, L]), np.vstack([L.T, thet*STS])]))

        return W, M, thet
