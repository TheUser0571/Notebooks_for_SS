#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 17:21:48 2021

@author: diego
"""
import numpy as np

class discrete:
    
    #Elementary signal definition
    #########################################
    
    #Kronecker impulsion
    def Kroneckerimp (n):
        if n==0: return 1
        else: return 0
        
    #Unit jump
    def unitjump(n):
        if n>=0: return 1
        else: return 0
     
    #Casual polinom of degree N>=1. First variable (N)
    #is the degree and second (n) is the value to evaluate the pol
    def casualpol(N,n):
        N=int(N)
        if N>=1:
            if n>=0:
                prod = 1.
                for i in range(N):
                    prod*=n+i+1
                
                return (prod/np.math.factorial(N))
            else: return 0.
                    
                    
        else: return 0.
    
    #Rectangular signal
    def rectsignal(n1,n2,n):
        if n>=n1 and n<=n2: return 1
        else: return 0 
        
    
    #Discrete signal transformations/operations
    ####################################################
    
    #Discrete convolution. First argument: array of length N. Second argument: array of lenght M
    #Third argument, mode:
        #1) "full", it returns the convolution at each point of overlap, with an output shape of N+M-1. 
        #At the end-points of the convolution, the signals do not overlap completely, and boundary effects may be seen
        #2) "same" returns output of length max(M, N). Boundary effects are still visible.
        #3) "valid" returns output of length max(M, N) - min(M, N) + 1. The convolution product is only given for points where the signals overlap completely. 
        #Values outside the signal boundary have no effect.
    
    discreteconv = np.convolve
    
    #Circular convolution. Use of the convolution theorem to perform a circular convolution
    #The two arrays must have the same length N and be N-periodic
    
    def circularconv(signal, ker):
        
        if len(signal)==len(ker): return np.real(np.fft.ifft( np.fft.fft(signal)*np.fft.fft(ker) ))
        else: return 0
        
    