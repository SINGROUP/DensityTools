import numpy as np


def convolve_fast(arr, ker):
    ''' Convolution via FFT and inverse-FFT '''
    #'''
    in1 = np.asarray(arr)
    in2 = np.asarray(ker)

    s1 = np.array(in1.shape)
    s2 = np.array(in2.shape)

    if len(s1) != len(s2):
        raise RuntimeError("Dimensions of array"
                           " and kernel don't match")

    sp1 = np.fft.rfftn(in1, s1)
    sp2 = np.fft.rfftn(in2, s1)
    ret = np.fft.irfftn(sp1 * sp2, s1).copy()
    ret = ret.real
    shift_ = (s2 - 1) // 2
    return shift(ret, shift_)

def gauss(data, sigma=1, voxl_size=0.5):
    if sigma < 1e-4:
        import copy
        return copy.deepcopy(data)
    from scipy.signal import gaussian
    sigma /= voxl_size
    sig = gaussian(data.shape[0], sigma) / voxl_size
    ker = sig[None, :, None] * sig[:, None, None] * sig[None, None, :]
    ker /= np.sum(ker)
    return convolve_fast(data, ker)


def shift(arr, shift):
    '''
    Shifts shift array coordinate to origin
    '''
    shift = np.asarray(shift, dtype=int)
    arr_out = np.asarray(arr.copy())
    shift %= arr_out.shape
    dim = len(shift)
    dim_ind = list(range(dim))
    hold = dim_ind.pop(0)
    dim_ind.insert(dim - 1, hold)
    for i in range(dim):
        if shift[i] == 0:
            arr_out = arr_out.transpose(dim_ind)
            continue
        arr1 = np.zeros_like(arr_out)
        arr1[-shift[i]:] = arr_out[:shift[i]]
        arr1[:-shift[i]] = arr_out[shift[i]:]
        arr_out = arr1.transpose(dim_ind)
    return arr_out

