# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 11:26:22 2018

@author: jfraxanet
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import copy
import timeit
import itertools as it

Random1 = []
Random2 = []

for ex in range(5):
    Ev_random1 = np.load('Evolution_Random'+str(ex)+'.npy')
    Ev_random2 = np.load('Evolution_SysRandom'+str(ex)+'.npy')
    
    if len(Random1) == 0:
        Random1 = Ev_random1
    else:
        Random1 = np.vstack((Random1, Ev_random1))
    
    if len(Random2) == 0:
        Random2 = Ev_random2
    else:
        Random2 = np.vstack((Random2, Ev_random2))


plt.figure()
plt.plot(np.mean(Random1, axis= 0), 'r-', label='Random mean')
plt.plot(np.mean(Random2, axis= 0), 'b-', label='RandomSystem mean')
plt.legend()
plt.xlabel('Cycles')
plt.ylabel('Optimal energy')
