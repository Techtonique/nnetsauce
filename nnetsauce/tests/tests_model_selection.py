#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 16:37:42 2019

@author: moudiki
"""

from nnetsauce.utils.model_selection import TimeSeriesSplit
import numpy as np
import unittest as ut    

class TS_Split(ut.TestCase):
    def test_tsplit(self):
        
        X = np.array([[4, 5, 6, 1, 0, 2, 7, 9, 3, 8], 
              [3.1, 3.5, 1.0, 2.1, 8.3, 1.1, 8.9, 0.3, 0.2, 8.7]]).T
        
        tscv = TimeSeriesSplit()
        
        for train_index, test_index in tscv.split(X, initial_window=5, horizon=3, 
                                           fixed_window=False):
            0
            #print("TRAIN:", train_index, "TEST:", test_index)
    
        for train_index2, test_index2 in tscv.split(X, initial_window=5, horizon=3, 
                                                   fixed_window=True):
            0
            #print("TRAIN:", train_index2, "TEST:", test_index2)    
        
        self.assertTrue(
            np.allclose(train_index, [0, 1, 2, 3, 4, 5, 6]) # with fixed_window=False
            & np.allclose(test_index, [7, 8, 9]) # with fixed_window=False
            & np.allclose(train_index2, [2, 3, 4, 5, 6]) # with fixed_window=True
            & np.allclose(test_index2, [7, 8, 9])) # with fixed_window=True
            
