import unittest

import numpy as np
import pandas as pd

import utilities


class TestUtilities(unittest.TestCase):
    def setUp(self):
        self.data = np.array(
            [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
        )
        self.df = pd.DataFrame(self.data, columns=[pd.Period(f'2019-0{i}') for i in range(1,5)], index=[1, 7, 12])
    
    
    def test_nothing(self):
        self.assertTrue(True)
        
    def test_create_windowed(self):
        X,Y = utilities.create_windowed_XY(self.df, 2)
        expected_X = pd.DataFrame(np.array(
            [[1,2], [5,6], [9,10], [2,3], [6,7], [10, 11]]
        ), columns=[2,1])
        expected_Y = pd.Series(np.array([3, 7, 11, 4, 8, 12]), name='Y')
        np.testing.assert_array_equal(X, expected_X)
        np.testing.assert_array_equal(Y, expected_Y)
        
    def test_create_windowed_append_month(self):
        X,Y = utilities.create_windowed_XY(self.df, 2, append_pred_month=True)
        expected_X = pd.DataFrame(np.array(
            [[1,2], [5,6], [9,10], [2,3], [6,7], [10, 11]]
        ), columns=[2,1])
        expected_X['pred_month'] = [3, 3, 3, 4, 4, 4]
        expected_Y = pd.Series(np.array([3, 7, 11, 4, 8, 12]), name='Y')
        np.testing.assert_array_equal(X, expected_X)
        np.testing.assert_array_equal(Y, expected_Y)
        
    @unittest.skip('Not written yet.')
    def test_cross_val_score_stride_1(self):
        pass
        
        
        
        
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)