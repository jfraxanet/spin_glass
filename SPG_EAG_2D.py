# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 14:45:12 2018

@author: jfraxanet
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import copy
import timeit
import itertools as it

def Lattice(N):
    '''Square 2D lattice within sqrtN'''
    Position = np.zeros((N,2))
    SqrtN = np.ceil(np.sqrt(N)) 
    for i in range(N):
        Position[i][0] = (i%SqrtN)
        Position[i][1] = (i//SqrtN)  
    SpinX = Position[:,0]
    SpinY = Position[:,1]
    return SpinX, SpinY

def NN_Random_couplings(N, side, SpinX, SpinY):
    '''Computes the symmetric matrix with all random couplings for nearest neighbours PERIODIC BOUNDARY CONDITIONS'''
    Matrix = np.zeros((N,N))
    for i in range(N):
        for j in range(i+1,N):
            dist_vector = np.array([SpinX[i]-SpinX[j],SpinY[i]-SpinY[j]])
            dist_vector_PBC = (dist_vector + side/2)%side-side/2
            if np.linalg.norm(dist_vector_PBC) == 1:
                #Matrix[i][j] = 1
                Matrix[i][j] = np.random.normal(0,1)
                Matrix[j][i] = Matrix[i][j]
    return Matrix         
   
def Hamiltonian(Spins, Matrix, N):
    '''Computes the cost energy of a given state'''
    Energy = 0
    Spins = np.array(Spins, float)
    for i in range(N):
        for j in range(N):
            Energy += -Matrix[i][j]*Spins[i]*Spins[j]/float(2)
    return Energy/float(N)

def Swap(State, Energy, Matrix, i):
    '''Computes the new energy obtained by swapping spin i'''
    NewEnergy = copy.deepcopy(Energy)
    State = np.array(State, float)
    for j in range(N):
        NewEnergy += float(2)*Matrix[i][j]*State[i]*State[j]/float(N) 
    return NewEnergy
    
def PlotState(title, State, SpinX, SpinY):
    '''Plots the spin state'''
    plt.figure()
    plt.title(title)
    plt.plot(SpinX[State>0],SpinY[State>0],color='red',marker='^',linestyle='None',label='Spin +1')
    plt.plot(SpinX[State<0],SpinY[State<0],color='green',marker='v',linestyle='None',label='Spin -1')
    plt.legend()
            
def Metropolis_sweep(dummy, State, Energy, BestEnergy, BestState, Matrix, T, N):
    '''Performs a step in the Metropolis algorithm and decides whether to accept it in a given T'''    
    for i in range(N):
        spin = i
        NewEnergy = Swap(State, Energy, Matrix, spin)
        Dif = NewEnergy - Energy
        if Dif < 0: 
            State[spin] = -State[spin]
            Energy = NewEnergy
            if Energy < BestEnergy:
                BestState = copy.deepcopy(State)
                BestEnergy = Energy 
            Prob = 1
            Accept = 1
        else:
            Prob = np.exp(-Dif/T)
            if dummy < Prob:
                State[spin] = -State[spin]
                Energy = NewEnergy  
                Accept = 1
            else:
                Accept = 0   
    return State, Energy, BestEnergy, BestState, Prob, Accept

def GS_Brute_Force(N, Matrix):    
    print('\n\nSTART COMPUTATION OF GS BY BRUTE FORCE')
    start_BF = timeit.default_timer()
    degeneracy = 0
    BestEnergy = 0
    BestState = np.zeros(9)
    It = 0    
    for State in it.product([-1,1], repeat = N):        
        It += 1
        if It%100000 == 0:
            print(It)        
        State = np.array(list(State))        
        Energy = Hamiltonian(State,Matrix,N)        
        if Energy <= BestEnergy:
            if Energy == BestEnergy:
                degeneracy += 1
                BestState = np.vstack((BestState,State))
            else:
                degeneracy = 0
                BestState = State
            BestEnergy = Energy   
    end_BF = timeit.default_timer()
    print('TIME: ', end_BF-start_BF)    
    print('MINIMUM ENERGY: ', BestEnergy)
    print('DEGENERACY: ', degeneracy)
    print('STATES WITH MINIMUM ENERGY: \n', BestState)
    return BestEnergy

for ex in range(1):
    
    print('\n -----COMPUTING EXAMPLE ', ex, '-----\n')
    
    #Lattice
    size = 'big' #or small
    side = 16
    N = side*side
    SpinX,SpinY = Lattice(N)
    Matrix = NN_Random_couplings(N,side,SpinX,SpinY)
    np.save('Matrix'+str(ex), Matrix)
    #Matrix = np.load('Matrix'+str(ex)+'.npy')
    
    #SA
    EndT = 1e-3
    StartT = 1e3
    CoolingFactor = 0.95
    Equilibration = 20
    Cycles = 100
    
    State_init = np.sign(np.random.rand(N)*2-1)
    np.save('State_init'+str(ex), State_init)
    #State_init = np.load('State_init'+str(ex)+'.npy')
    
    #PlotState('Initial configuration', State_init, SpinX, SpinY)  
    
    start_program = timeit.default_timer()
    
    if size == 'small':
        BF_BestEnergy = GS_Brute_Force(N,Matrix)
    else:
        BF_BestEnergy = 0
        
    for method in ['Random', 'SysRandom']:
        print('\n\n---START SA WITH METHOD ', method, '---')    
        State = State_init    
        Energy = Hamiltonian(State,Matrix,N)
        FinalEnergies = []
        
        for cycle in range(Cycles):
            start_cycle = timeit.default_timer()
            Energy = Hamiltonian(State,Matrix,N)
            BestEnergy = Energy
            BestState = copy.deepcopy(State)
            T = StartT*0.9
            EndT = EndT*0.9
            Sweeps = 0  
            print('\nSTART SIMULATED ANNEALING CYCLE ', cycle+1, ' ', method)
            while T > EndT:
                for i in range(Equilibration):
                    Sweeps += 1
                    if Sweeps%10000 == 0:
                        print('Sweeps: ', Sweeps)
                    if method is 'Random':
                        dummy = random.random()
                    if method is 'SysRandom':
                        rng = random.SystemRandom()
                        dummy = rng.random()
                    State,Energy,BestEnergy,BestState,Prob,Accept = Metropolis_sweep(dummy,State,Energy,BestEnergy,BestState,Matrix,T,N)
                T = T*CoolingFactor
                if Energy == BF_BestEnergy:
                    print('Ground State Energy was reached in cycle ', cycle, ' and sweep ', Sweeps, '\n')
                    break
            
            State = BestState
            Energy = Hamiltonian(BestState,Matrix,N)
            FinalEnergies = np.append(FinalEnergies, Energy)
            end_cycle = timeit.default_timer()
            print('MINIMUM REACHED: ', Energy)
            print('TIME: ', end_cycle-start_cycle)
            if Energy == BF_BestEnergy:
                print('Ground State Energy was reached in cycle ', cycle, ' and sweep ', Sweeps, '\n')
                break
    
            if (cycle+1)%10 == 0:
                print('(Save data...)')
                plt.figure()
                plt.plot(FinalEnergies, label = method)
                plt.legend()
                plt.title('Simulated annealing cycles on 10x10 Gaussian EA Spin Glass')
                plt.xlabel('Cycles')
                plt.ylabel('Optimal energy')
                plt.savefig('FinalEnergies_'+method+'.png')
                np.save('Evolution_'+method+str(ex), FinalEnergies)
                
        #PlotState('Final configuration', State, SpinX, SpinY)
        GroundState = min(FinalEnergies)
        print('\nTHE MINIMUM ENERGY FOUND IS: ', GroundState)
        np.save('Evolution_'+method+str(ex), FinalEnergies)
                
    Ev_random1 = np.load('Evolution_Random'+str(ex)+'.npy')
    Ev_random2 = np.load('Evolution_SysRandom'+str(ex)+'.npy')
    
    plt.figure()
    plt.plot(Ev_random1, label='random.random')
    plt.plot(Ev_random2, label='random.SystemRandom')
    if size == 'small':
        plt.axhline(y=BF_BestEnergy, linestyle='--', label='Ground State Energy')
    plt.legend()
    plt.xlim(0,Cycles-1)
    plt.title('Simulated annealing cycles on 16x16 Gaussian EA Spin Glass')
    plt.xlabel('Cycles')
    plt.ylabel('Optimal energy')
    plt.savefig('16x16_'+str(ex)+'.png')
    
    end_program = timeit.default_timer()
    print('TOTAL RUNNING TIME: ', end_program-start_program)
