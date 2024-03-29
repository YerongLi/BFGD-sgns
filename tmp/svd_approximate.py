import numpy as np

from scipy.sparse.linalg import svds
from numpy.linalg import norm, svd

def low_rank_approx(SVD=None, A=None, r=1):
    """
    Computes an r-rank approximation of a matrix
    given the component u, s, and v of it's SVD

    Requires: numpy
    """
    if not SVD:
        SVD = np.linalg.svd(A, full_matrices=False)
    u, s, v = SVD
    Ar = np.zeros((len(u), len(v)))
    for i in range(r):
        Ar += s[i] * np.outer(u.T[i], v[i])
    return Ar

if __name__ == "__main__":
    """
    Test: visualize an r-rank approximation of `lena`
    for increasing values of r

    Requires: scipy, matplotlib
    """
    from scipy.misc import lena
    import pylab
    x = np.array([[1.,0,1],[0,3.,0],[0,0,5.]])
    
    u, s, v = np.linalg.svd(x, full_matrices=False)
    i = 2
    
    y = low_rank_approx((u, s, v), r=i)
    
    print(y)
    
    
    u, s, vt = svds(x, k=i)
    C0 = u.dot(np.sqrt(np.diag(s))).T
    W0 = np.sqrt(np.diag(s)).dot(vt)
    print(C0.T.dot(W0), 'product')
    pylab.figure()
    pylab.ion()
    while i < len(x) - 1:
        y = low_rank_approx((u, s, v), r=i)
        pylab.imshow(y, cmap=pylab.cm.gray)
        pylab.draw()
        i += 1
        #print percentage of singular spectrum used in approximation
        print("%0.2f %%" % (100 * i / 512.))
