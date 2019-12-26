import numpy as np
from sklearn import datasets
import unittest as ut
import nnetsauce as ns


# Basic tests


class TestTimeSeriesSplit(ut.TestCase):
    def test_TimeSeriesSplit(self):

        X = [[1, 2], [3, 4], [5, 6], [7, 8], 
             [9, 10], [11, 12], [13, 14]]
        
        tscv2 = ns.utils.TimeSeriesSplit()
        
        splits2 = tscv2.split(X[:-1], initial_window=3,
                             horizon=2, fixed_window=False)
        
        train, test = next(splits2)
        self.assertTrue(
            np.allclose(train[2], 2)
        )
        self.assertTrue(
            np.allclose(test[1], 4)
        )
        #assert_array_equal(train, [0, 1, 2])
        #assert_array_equal(test, [3, 4])
        
        train, test = next(splits2)
        self.assertTrue(
            np.allclose(train[3], 3)
        )
        self.assertTrue(
            np.allclose(test[1], 5)
        )        
        #assert_array_equal(train, [0, 1, 2, 3])
        #assert_array_equal(test, [4, 5)                
        
        tscv3 = ns.utils.TimeSeriesSplit()
        
        splits3 = tscv3.split(X[:-1], initial_window=3,
                             horizon=2, fixed_window=True)
        
        train, test = next(splits3)
        self.assertTrue(
            np.allclose(train[2], 2)
        )
        self.assertTrue(
            np.allclose(test[1], 4)
        )        
        #assert_array_equal(train, [0, 1, 2])
        #assert_array_equal(test, [3, 4])
        
        train, test = next(splits3)
        self.assertTrue(
            np.allclose(train[2], 3)
        )
        self.assertTrue(
            np.allclose(test[1], 5)
        )        
        #assert_array_equal(train, [1, 2, 3])
        #assert_array_equal(test, [4, 5)   
        assert tscv2.get_n_splits() == 3            
        assert tscv3.get_n_splits() == 3            
       
if __name__ == "__main__":
    ut.main()
