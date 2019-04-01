import numpy as np
import scipy.io as sio
import argparse

"""
Script containing helper functions to check that two datasets are equal to
within the allowable tolerance. 

Used for when we generate the same dataset via different code and want to
check that the resulting dataset is the same. (Potentially could be 
written as a test.)
"""

def assert_shape(name, a1, a2):
    s1 = np.shape(a1)
    s2 = np.shape(a2)
    assert s1==s2, "{} shapes {} and {} do not match.".format(name, s1, s2)
    
def assert_almost_equal(name, a1, a2, tolerance):
    d = np.abs(a1-a2)
    d_max = np.max(d)
    print("Maximum elementwise difference for {} is {}".format(name, d_max))
    assert d_max <= tolerance, "Maximum elementwise difference for {}, {} exceeds tolerance {}".format(name, d_max, tolerance)
    
def check_dataset_equality(filename1, filename2):
    m1 = sio.loadmat(filename1)
    m2 = sio.loadmat(filename2)
    
    i1 = m1["images"]
    i2 = m2["images"]
    
    l1 = m1["labels"]
    l2 = m2["labels"]
    
    assert_shape("images", i1, i2)
    assert_shape("labels", l1, l2)
    assert_almost_equal("images", i1, i2, tolerance=1e-6)
    assert_almost_equal("labels", l1, l2, tolerance=0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--f1', required=True, type=str, help='Original datafile name.')
    parser.add_argument('--f2', required=True, type=str, help='New datafile name.')
    FLAGS = parser.parse_args()
    check_dataset_equality(FLAGS.f1, FLAGS.f2)
	
