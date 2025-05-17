import numpy as np

def fre_single(t, y, J, eta, delta, I = lambda t: 0):
    '''
        firing-rate equation for single population.
    '''
    r, v = y
    dr = 2 * r * v + delta / np.pi
    dv = v**2 + J * r + eta - (np.pi * r) ** 2 + I(t)
    return np.array([dr, dv])
def fre_single_inverse(t, y, J, eta, delta, I = lambda t: 0):
    return -fre_single(t, y, J, eta, delta, I)

def fre_single_jacobian(t, y, J, eta, delta):
    '''
        Jacobian matrix of fre_single.
    '''
    r, v = y
    A = np.array([[2 * v, 2 * r], [J - 2 * np.pi**2 * r, 2 * v]])
    return A

def fre_network(t, y, w, eta, delta, P, I = lambda t: 0):
    '''
        firing-rate equation for network of P populations.
    '''
    dy = np.zeros(P * 2)
    dy[:P] = 2 * y[:P] * y[P:] + delta / np.pi # r_k
    dy[P:] = y[P:]**2 + w @ y[:P] + eta - (np.pi * y[:P]) ** 2 + I(t)# v_k
    return dy

def fre_seq(t, y, J1, J2, eta, delta, P, I = lambda t: 0, global_inhibition = True):
    '''
        firing-rate equation for sequential memory.
        J1 : connection between (k-1) -> k
        J2 : recurrent connection between k -> k
        note that J2 corresponds to J in fre_single.
    '''
    w = J1 * np.roll(np.eye(P), 1, axis = 0)  + J2 * np.eye(P) - (J1/P) * global_inhibition * np.ones((P, P), dtype = 'float')

    dy = fre_network(t, y, w, eta, delta, P, I)
    return dy
