#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 21:38:01 2022

Author: Li Jiangnan
        Kunming University of Science and Technology (KUST)

Email : li-jiangnan@kust.edu.cn

-------------------

MAG2305 : An FDM-FFT micromagnetic simulator

Library version : numpy   1.21.5
                  pytorch 1.13.0

-------------------

"""


__version__ = 'PyTest_2022.11.29'
print('MAG2305 version: {:s}\n'.format(__version__))


import numpy as np
import torch


def DemagCell(D, rv):
    """
    To get the demag matrix of a cell at a certain distance
    
    Parameters
    ----------
    D  : Real(3)
         Cell size : DX,DY,DZ
    rv : Real(3)
         Distance vector from cell center : RX,RY,RZ
    DM : Real(3,3)
         Demag Matrix
    Returns
    -------
    DM
    """
    D  = np.array(D)
    rv = np.array(rv)
    DM = np.zeros((3,3))
    
    pqw_range = [ [i,j,k] for i in [-1,1] for j in [-1,1] for k in [-1,1] ]
    
    for pqw in pqw_range:
        pqw = np.array(pqw)
        R = 0.5*D + pqw*rv
        RR = np.linalg.norm(R)
        
        for i in range(3):
            j = (i+1)%3
            k = (i+2)%3
            DM[i,i] += np.arctan(R[j]*R[k]/R[i]/RR)
            DM[i,j] += 0.5*pqw[i]*pqw[j]*np.log((RR-R[k])/(RR+R[k]))
            DM[j,i] = DM[i,j]
    
    return DM/np.pi/4.


def DemagHalf(D, rv, pqw):
    """
    To get the demag matrix of a boundary cell at a certain distance
    
    Parameters
    ----------
    D   : Real(3)
          Cell size : DX,DY,DZ
    rv  : Real(3)
          Distance vector from cell center : RX,RY,RZ
    pqw : Int(3)
          Sign of boundary state
    DM  : Real(3,3)
          Demag Matrix
    Returns
    -------
    DM
    """
    D  = np.array(D)
    rv = np.array(rv)
    pqw = np.array(pqw)
    R  = np.zeros(3)
    
    DM = DemagCell(D,rv) *np.pi*4.
    
    p = pqw[0]
    if p != 0:
        for q,w in [ [1,1], [1,-1], [-1,1], [-1,-1] ]:
            pqw_vec = np.array([p,q,w])
            R = 0.5*D + rv*pqw_vec
            RR = np.linalg.norm(R)
            DM[0,0] -= abs(p)*np.arctan(R[1]*R[2]/R[0]/RR)
            DM[1,0] -= 0.5*p*q*np.log((RR-R[2])/(RR+R[2]))
            DM[2,0] -= 0.5*w*p*np.log((RR-R[1])/(RR+R[1]))
    
    q = pqw[1]
    if q != 0:
        for p,w in [ [1,1], [1,-1], [-1,1], [-1,-1] ]:
            pqw_vec = np.array([p,q,w])
            R = 0.5*D + rv*pqw_vec
            RR = np.linalg.norm(R)
            DM[1,1] -= abs(q)*np.arctan(R[2]*R[0]/R[1]/RR)
            DM[2,1] -= 0.5*q*w*np.log((RR-R[0])/(RR+R[0]))
            DM[0,1] -= 0.5*p*q*np.log((RR-R[2])/(RR+R[2]))
    
    w = pqw[2]
    if w != 0:
        for p,q in [ [1,1], [1,-1], [-1,1], [-1,-1] ]:
            pqw_vec = np.array([p,q,w])
            R = 0.5*D + rv*pqw_vec
            RR = np.linalg.norm(R)
            DM[2,2] -= abs(w)*np.arctan(R[0]*R[1]/R[2]/RR)
            DM[0,2] -= 0.5*w*p*np.log((RR-R[1])/(RR+R[1]))
            DM[1,2] -= 0.5*q*w*np.log((RR-R[0])/(RR+R[0]))
    
    return DM/np.pi/4.



class mmModel():
    " Define a micromagnetic model "
    
    def __init__(self, types='3DPBC', size=(1,1,1), cell=(1,1,1), 
                 Ms=1, Ax=1.0e-6, device='cpu'):
        """
        Parameters
        ----------
        types : String
                3DPBC : Periodic along all X,Y,Z directions
                film  : Periodic along X,Y directions, Z in-plane
                bulk  : Non-periodic along all X,Y,Z directions
                default = 3DPBC
        size  : Int(3)
                Model size RNX,RNY,RNZ
        cell  : Real(3)
                Cell size DX,DY,DZ [unit nm]
        Ms    : Real
                Saturation magnetization [unit emu/cc]
        Ax    : Real
                Exchange stiffness constant [unit erg/cm]
        Hx0   : Real
                Exchange field constant [unit Oe]
        device: str
                'cuda' or 'cpu'
        
        self.fftsize : Int(3)
                       fft model size FNX,FNY,FNZ
        self.Spin : Real(fftsize,3)
                    Spin direction of each cell
                    Spin = valid values in cell range (RNX,RNY,RNZ)
                    Spin == 0 for fft-only cells (i.e., out of range (RNX,RNY,RNZ))
        self.He   : Real(size,3)
                    Exchange field distribution
        self.Hd   : Real(size,3)
                    Demag field distribution
        self.Heff : Real(size,3)
                    Effective field distribution
        self.FDMW : Complex(FNX,FNY,FNZ//2+1,3,3)
                    DFT of DMW
        """
        self.types = types
        self.size  = np.array(size, dtype=int)
        self.cell  = np.array(cell, dtype=float)
        self.Ms    = float(Ms)
        self.Ax    = float(Ax)
        self.Hx0   = np.array([2.0 * 1.0e14 * self.Ax 
                               / self.Ms / self.cell[i]**2 for i in range(3)])
        self.device = torch.device(device)
        
        # Model size && fft size
        self.fftsize = np.array(size, dtype=int)
        if self.types == 'film':
            self.fftsize[-1] = 2*self.fftsize[-1]
        if self.types == 'bulk':
            self.fftsize = 2*self.fftsize
        self.pbc = (self.size==self.fftsize)
        
        # Tensors; spin, fields, and matrix
        self.Spin = torch.zeros( tuple(self.size) + (3,) ).to(self.device)
        self.He   = torch.zeros( tuple(self.size) + (3,) ).to(self.device)
        self.Hd   = torch.zeros( tuple(self.size) + (3,) ).to(self.device)
        self.Heff = torch.zeros( tuple(self.size) + (3,) ).to(self.device)
        
        self.FDMW = torch.Tensor( np.zeros( ( self.fftsize[0],
                                              self.fftsize[1],
                                              self.fftsize[2]//2+1)
                                              + (3,3) ) ).to(self.device)
        
        return None
    
    
    def NormSpin(self):
        """
        Normalize self.Spin
        
        Parameters
        ----------
        self.Spin : Real(self.size,3)
                    Spin direction of each cell
        """
        norm = torch.sqrt( torch.einsum( 'ijkl,ijkl -> ijk', 
                                         self.Spin, self.Spin ) )
        for l in range(3):
            self.Spin[...,l] /= norm
        return self.Spin
    
    
    def SpinInit(self, Spin_in):
        """
        Update Spin state from input
        
        Parameters
        ----------
        Spin_in   : Real(self.size,3)
                    Input Spin state
        self.Spin : Real(self.size,3)
                    Spin direction of each cell
        stat : Int
               Status of Spin input
               stat=0 successful
               stat=1 failed
        Returns
        -------
        stat
        """
        Spin_in = np.array(Spin_in, dtype=float)
        
        if Spin_in.shape != tuple(self.size) + (3,):
            print(' Spin input error! Size mismatched!')
            stat = 1
        
        else:
            self.Spin[...] = torch.Tensor(Spin_in).to(self.device)
            stat = 0
        
        spin_zero=self.NormSpin()
        
        return stat,spin_zero
    
    
    def DemagInit(self):
        """
        To get the demag matrix of the whole model
        Periodic boudnary conditions applied !
        
        Parameters
        ----------
        FN  : Int(3)
              FN = self.fftsize, model fft size FNX,FNY,FNZ
        RN  : Int(3)
              RN = self.size, model size RNX,RNY,RNZ
        D   : Real(3)
              D = self.cell, cell size DX,DY,DZ
        DMW : Real(fftsize,3,3)
              Demag matrix of whole model
              DMW(0,0,0) , self-demag
        self.FDMW : Complex(FNX,FNY,FNZ//2+1,3,3)
                    DFT of DMW
        """
        FN = self.fftsize
        RN = self.size
        D  = self.cell
        DFN = D*FN
        DMW = np.zeros( tuple(FN) + (3,3) )
        
        # General demagmatrix for each cell
        rvm = np.empty(tuple(FN) + (3,))
        for ijk in np.ndindex(tuple(FN)):
            for l in range(3):
                rvm[ijk][l] = -1.*ijk[l]*D[l]
                rvm[ijk][l] += DFN[l] if ijk[l] > FN[l]//2 else 0.0
        
        pqw_range = [ [i,j,k] for i in [-1,1] for j in [-1,1] for k in [-1,1] ]
        
        for pqw in pqw_range:
            pqw = np.array(pqw)
            R = 0.5*D + pqw*rvm
            RR = np.sqrt( np.einsum( 'ijkl,ijkl -> ijk', R, R ) )
            
            for i in range(3):
                j = (i+1)%3
                k = (i+2)%3
                DMW[...,i,i] += np.arctan(R[...,j]*R[...,k]/R[...,i]/RR)
                DMW[...,i,j] += 0.5*pqw[i]*pqw[j]*np.log((RR-R[...,k])/(RR+R[...,k]))
                DMW[...,j,i]  = DMW[...,i,j]
        
        # Demagmatrix for cells on the facets
        # surfaces x
        D1 = 1.0*D
        D1[0] = 0.5*D[0]
        pqw_bd = np.zeros(tuple(FN))
        for ijk in np.ndindex(tuple(FN)):
            pqw_bd[ijk] = 0
            if ijk[0] == FN[0]//2 and ijk[1] != FN[1]//2 and ijk[2] != FN[2]//2:
                pqw_bd[ijk] = 1
        
        for p in [+1, -1]:
            if p > 0:
                rvm1 = 1.0 * rvm
                rvm1[...,0] -= 0.5 * pqw_bd * D1[0]
            if p < 0:
                rvm1[...,0] += pqw_bd * DFN[0]
            
            for pqw in pqw_range:
                pqw = np.array(pqw)
                R = 0.5*D1 + pqw*rvm1
                RR = np.sqrt( np.einsum( 'ijkl,ijkl -> ijk', R, R ) )
                for i in range(3):
                    j = (i+1)%3
                    k = (i+2)%3
                    DMW[...,i,i] -= p * np.arctan(R[...,j]*R[...,k]/R[...,i]/RR)
                    DMW[...,i,j] -= p * 0.5*pqw[i]*pqw[j]*np.log((RR-R[...,k])/(RR+R[...,k]))
                    DMW[...,j,i] -= p * 0.5*pqw[j]*pqw[i]*np.log((RR-R[...,k])/(RR+R[...,k]))
            
            for q,w in [ [1,1], [1,-1], [-1,1], [-1,-1] ]:
                pqw = np.array([abs(p),q,w])
                R = 0.5*D1 + pqw*rvm1
                RR = np.sqrt( np.einsum( 'ijkl,ijkl -> ijk', R, R ) )
                DMW[...,0,0] += p * np.arctan(R[...,1]*R[...,2]/R[...,0]/RR)
                DMW[...,1,0] += p * 0.5*q*np.log((RR-R[...,2])/(RR+R[...,2]))
                DMW[...,2,0] += p * 0.5*w*np.log((RR-R[...,1])/(RR+R[...,1]))
        
        # surfaces y
        D1 = 1.0*D
        D1[1] = 0.5*D[1]
        pqw_bd = np.zeros(tuple(FN))
        for ijk in np.ndindex(tuple(FN)):
            pqw_bd[ijk] = 0
            if ijk[0] != FN[0]//2 and ijk[1] == FN[1]//2 and ijk[2] != FN[2]//2:
                pqw_bd[ijk] = 1
        
        for q in [+1, -1]:
            if q > 0:
                rvm1 = 1.0 * rvm
                rvm1[...,1] -= 0.5 * pqw_bd * D1[1]
            if q < 0:
                rvm1[...,1] += pqw_bd * DFN[1]
            
            for pqw in pqw_range:
                pqw = np.array(pqw)
                R = 0.5*D1 + pqw*rvm1
                RR = np.sqrt( np.einsum( 'ijkl,ijkl -> ijk', R, R ) )
                for i in range(3):
                    j = (i+1)%3
                    k = (i+2)%3
                    DMW[...,i,i] -= q * np.arctan(R[...,j]*R[...,k]/R[...,i]/RR)
                    DMW[...,i,j] -= q * 0.5*pqw[i]*pqw[j]*np.log((RR-R[...,k])/(RR+R[...,k]))
                    DMW[...,j,i] -= q * 0.5*pqw[j]*pqw[i]*np.log((RR-R[...,k])/(RR+R[...,k]))
            
            for p,w in [ [1,1], [1,-1], [-1,1], [-1,-1] ]:
                pqw = np.array([p,abs(q),w])
                R = 0.5*D1 + pqw*rvm1
                RR = np.sqrt( np.einsum( 'ijkl,ijkl -> ijk', R, R ) )
                DMW[...,1,1] += q * np.arctan(R[...,2]*R[...,0]/R[...,1]/RR)
                DMW[...,2,1] += q * 0.5*w*np.log((RR-R[...,0])/(RR+R[...,0]))
                DMW[...,0,1] += q * 0.5*p*np.log((RR-R[...,2])/(RR+R[...,2]))
        
        # surfaces z
        D1 = 1.0*D
        D1[2] = 0.5*D[2]
        pqw_bd = np.zeros(tuple(FN))
        for ijk in np.ndindex(tuple(FN)):
            pqw_bd[ijk] = 0
            if ijk[0] != FN[0]//2 and ijk[1] != FN[1]//2 and ijk[2] == FN[2]//2:
                pqw_bd[ijk] = 1
        
        for w in [+1, -1]:
            if w > 0:
                rvm1 = 1.0 * rvm
                rvm1[...,2] -= 0.5 * pqw_bd * D1[2]
            if w < 0:
                rvm1[...,2] += pqw_bd * DFN[2]
            
            for pqw in pqw_range:
                pqw = np.array(pqw)
                R = 0.5*D1 + pqw*rvm1
                RR = np.sqrt( np.einsum( 'ijkl,ijkl -> ijk', R, R ) )
                for i in range(3):
                    j = (i+1)%3
                    k = (i+2)%3
                    DMW[...,i,i] -= w * np.arctan(R[...,j]*R[...,k]/R[...,i]/RR)
                    DMW[...,i,j] -= w * 0.5*pqw[i]*pqw[j]*np.log((RR-R[...,k])/(RR+R[...,k]))
                    DMW[...,j,i] -= w * 0.5*pqw[j]*pqw[i]*np.log((RR-R[...,k])/(RR+R[...,k]))
            
            for p,q in [ [1,1], [1,-1], [-1,1], [-1,-1] ]:
                pqw = np.array([p,q,abs(w)])
                R = 0.5*D1 + pqw*rvm1
                RR = np.sqrt( np.einsum( 'ijkl,ijkl -> ijk', R, R ) )
                DMW[...,2,2] += w * np.arctan(R[...,0]*R[...,1]/R[...,2]/RR)
                DMW[...,0,2] += w * 0.5*p*np.log((RR-R[...,1])/(RR+R[...,1]))
                DMW[...,1,2] += w * 0.5*q*np.log((RR-R[...,0])/(RR+R[...,0]))
        
        
        DMW[...] = DMW[...]/np.pi/4.0
        
        
        # Demagmatrix modification for boundary cells
        for ijk in np.ndindex(tuple(FN)):
            ijk = np.array(ijk)
            rv = -1.*ijk*D
            
            for l in range(3):
                rv[l] += DFN[l] if ijk[l] > FN[l]//2 else 0.0
            
# =============================================================================
#             DM = DemagCell(D,rv)
# =============================================================================
            DM = np.zeros((3,3))
            
            pqw = np.array([0, 0, 0])
            ipp = [0, 0, 0]
            for l in range(3):
                ipp[l] = 1 if ijk[l]==FN[l]//2 else 0
            n_case = np.sum(ipp)
            
            # Cells on the facets
            if n_case == 1:
# =============================================================================
#                 D1 = 1.*D
#                 ll = 0
#                 for l in range(3):
#                     ll = l if ipp[l]==1 else ll
#                 D1[ll] = 0.5*D1[ll]
#                 
#                 pqw[ll] = 1
#                 rv1 = rv - 0.5*pqw*D1
#                 DM -= DemagHalf(D1, rv1, pqw)
#                 rv1[ll] = rv1[ll] + DFN[ll]
#                 DM += DemagHalf(D1, rv1, pqw)
#                 
#                 DMW[tuple(ijk)] += DM
# =============================================================================
                pass
            
            # Cells on the edges
            elif n_case == 2:
                D2 = 1.*D
                ll = 0
                for l in range(3):
                    ll = l if ipp[l]==0 else ll
                ii = (ll+1)%3
                jj = (ll+2)%3
                D2[ii] = 0.5*D2[ii]
                D2[jj] = 0.5*D2[jj]
                
                for m,n in [ [1,1], [1,-1], [-1,1], [-1,-1] ]:
                    pqw[ii] = m
                    pqw[jj] = n
                    rv2 = rv - 0.5*pqw*D2
                    DM -= DemagHalf(D2, rv2, pqw)
                    if pqw[ii]==1 :
                        rv2[ii] = rv2[ii] + DFN[ii]
                    if pqw[jj]==1 :
                        rv2[jj] = rv2[jj] + DFN[jj]
                    DM += DemagHalf(D2, rv2, pqw)
                    
                DMW[tuple(ijk)] += DM
            
            # Cell at the corner
            elif n_case == 3:
                D3 = 0.5*D
                
                for l,m,n in [ [1,1,1], [1,-1,1], [-1,1,1], [-1,-1,1],
                               [1,1,-1], [1,-1,-1], [-1,1,-1], [-1,-1,-1] ]:
                    pqw = np.array([l,m,n])
                    rv3 = rv - 0.5*pqw*D3
                    DM -= DemagHalf(D3, rv3, pqw)
                    for lmn in range(3):
                        if pqw[lmn]==1 :
                            rv3[lmn] = rv3[lmn] + DFN[lmn]
                    DM += DemagHalf(D3, rv3, pqw)
                    
                DMW[tuple(ijk)] += DM
            
            else:
                pass
        
        
        # Demag Matrix SumTest
        sum = np.zeros((3,3))
        for ijk in np.ndindex(tuple(RN)):
            sum += DMW[ijk]
        
        print("Demag Matrix SumTest:")
        for i in range(3):
            print(sum[i,:])
        print("")
        
        # FFT of Demag Matrix
        for m in range(3):
            for n in range(3):
                DMW_device = torch.Tensor(DMW[...,m,n]).to(self.device)
                self.FDMW[...,m,n] = torch.fft.rfftn( DMW_device )
        
        return None
    
    
    def DemagField_FFT(self):
        """
        To get demagfield distribution of the whole model
        
        Parameters
        ----------
        FN  : Int(3)
              FN = self.fftsize, model fft size FNX,FNY,FNZ
        RN  : Int(3)
              RN = self.size, model size RNX,RNY,RNZ
        H   : Real(FN,3)
              Demag field for whole fft model
        FH  : Complex(FNX,FNY,FNZ//2+1,3)
              DFT of H
        self.FDMW : Complex(FNX,FNY,FNZ//2+1,3,3)
                    DFT of DMW
        self.Hd   : Real(RN,3)
                    Demag field distribuition
        Returns
        -------
        self.Hd
        """
        FN = self.fftsize
        RN = self.size
        Ms = self.Ms
        
        shape_out = (FN[0], FN[1], FN[2]//2+1)
        
        FM = torch.Tensor(np.empty( shape_out + (3,), dtype=complex )).to(self.device)
        for l in range(3):
            M_tmp = torch.zeros( tuple(FN) ).to(self.device)
            M_tmp[:RN[0], :RN[1], :RN[2]] = self.Spin[...,l].to(self.device)
            FM[...,l] = torch.fft.rfftn( M_tmp )
        
        H = torch.empty( tuple(FN) + (3,) ).to(self.device)
        FH = torch.Tensor(np.zeros( shape_out + (3,), dtype=complex )).to(self.device)
        for m in range(3):
            for n in range(3):
                FH[...,m] -= self.FDMW[...,m,n] * FM[...,n]
            H[...,m] = torch.fft.irfftn( FH[...,m] )
        
        self.Hd[...] = H[:RN[0], :RN[1], :RN[2], :] * Ms *torch.pi*4.0
        
        return self.Hd.cpu()
    
    
    def ExchangeField(self):
        """
        To get exchange field distribution of the whole model
        
        Parameters
        ----------
        Hecst : Real
                Exchange field constant
        He  : Real(RN,3)
              Exchange field distribuition
        Returns
        -------
        He
        """
        Hecst = self.Hx0[0]
        
        Spin_nb  = torch.roll(self.Spin, 1, 0).to(self.device)
        if not self.pbc[0]:
            Spin_nb[0,:,:] = self.Spin[0,:,:]
        self.He[...] = Hecst * (Spin_nb - self.Spin)
        
        Spin_nb  = torch.roll(self.Spin,-1, 0).to(self.device)
        if not self.pbc[0]:
            Spin_nb[-1,:,:] = self.Spin[-1,:,:]
        self.He += Hecst * (Spin_nb - self.Spin)
        
        Hecst = self.Hx0[1]
        
        Spin_nb  = torch.roll(self.Spin, 1, 1).to(self.device)
        if not self.pbc[1]:
            Spin_nb[:,0,:] = self.Spin[:,0,:]
        self.He += Hecst * (Spin_nb - self.Spin)
        
        Spin_nb  = torch.roll(self.Spin,-1, 1).to(self.device)
        if not self.pbc[1]:
            Spin_nb[:,-1,:] = self.Spin[:,-1,:]
        self.He += Hecst * (Spin_nb - self.Spin)
        
        Hecst = self.Hx0[2]
        
        Spin_nb  = torch.roll(self.Spin, 1, 2).to(self.device)
        if not self.pbc[2]:
            Spin_nb[:,:,0] = self.Spin[:,:,0]
        self.He += Hecst * (Spin_nb - self.Spin)
        
        Spin_nb  = torch.roll(self.Spin,-1, 2).to(self.device)
        if not self.pbc[2]:
            Spin_nb[:,:,-1] = self.Spin[:,:,-1]
        self.He += Hecst * (Spin_nb - self.Spin)
        
        return self.He.cpu()
    
    
    def SpinDescent(self, Hext=torch.zeros(3), dtime=1.0e-6):
        """
        Update Spin state based on energy descent direction (Heff)
        
        Parameters
        ----------
        Hext  : Real(3)
                External field
        dtime : Real
                Pseudo time step for Spin update
        GSpin : Real(self.size,3)
                GSpin = Heff - Spin * (Spin .dot. Heff)
        DSpin : Real(self.size,3)
                Delta Spin; DSpin = dtime * GSpin
        Returns
        -------
        self.Spin
        """
        Hext = torch.Tensor(Hext).to(self.device)
        
        #self.DemagField_FFT()
        
        self.ExchangeField()
        
        self.Heff[...] = self.Hd + self.He + Hext
        
        GSpin = torch.empty( tuple(self.size) + (3,) ).to(self.device)
        SHdot = torch.einsum('ijkl,ijkl -> ijk', self.Spin, self.Heff)
        for l in range(3):
            GSpin[...,l] = self.Heff[...,l] - self.Spin[...,l] * SHdot
        
        DSpin = dtime * GSpin
        
        error = torch.sqrt( torch.einsum('ijkl,ijkl -> ijk', DSpin, DSpin).max() )
        
        self.Spin += DSpin
        
        self.NormSpin()
        


        return self.Spin.cpu(), float(error)