import numpy as np

def corr2_stress(stress):
    """
    Two-point transient time stress correlation 
    for all of the tensor component pairs
    """
    if stress.ndim != 3:
        raise ValueError("Expected shape is (m,n,n)")
    m, n, _ = stress.shape
    if n != stress.shape[-1]:
        raise ValueError("Stress should be a square matrix")
    res = np.zeros((n,)*4 + (2*m,), dtype=complex)
    for i in range(n):
        for j in range(i,n):
            for k in range(n):
                for l in range(k,n):
                    res[i,j,k,l] = res[j,i,k,l] = res[i,j,l,k] = res[j,i,l,k] = \
                    corr2_fast(stress[:,i,j], stress[:,k,l])
                    res[i,j,k,l][m:] = 0
    return res

def corr2_simple(data1, data2=None):
    """
    Find correlation between the data by padding it with zeros and 
    performing convolution (slow method O(N^2)).
    """
    data1 = np.asarray(data1)
    N = data1.shape[0]
    if data2 is None:
        data2 = data1
    if data1.shape != data2.shape:
        raise ValueError("Arrays of unequal lengths (shapes)")
    if data1.ndim > 1:
        raise ValueError("Too many axes in the input arrays (expected 1)")
    res = np.zeros(2*N, dtype=data1.dtype)
    for i in range(1-N, N):
        # it is like padding the data with zeros outside the range
        if i >= 0:
            res[i] = data1[:N-i].conj()@data2[i:]/N
        else:
            res[i] = data1[-i:].conj()@data2[:N+i]/N
    return res


def corr2_fast(data1, data2=None):
    """
    Find correlation between the data by padding it with zeros and 
    performing convolution (fast method using FFT, O(NlogN)).
    """
    data1 = np.asarray(data1)
    N = data1.shape[0]
    if data2 is None:
        data2 = data1
    if data1.shape != data2.shape:
        raise ValueError("Arrays of unequal lengths (shapes)")
    if data1.ndim > 1:
        raise ValueError("Too many axes in the input arrays (expected 1)")
    data1 = np.pad(data1, (0, N))
    data2 = np.pad(data2, (0, N))
    data1_fft = np.fft.fft(data1)
    data2_fft = np.fft.fft(data2)
    prod = data1_fft.conj()*data2_fft/N    
    return np.fft.ifft(prod)


def corr3_simple(data1, data2=None, data3=None):
    """
    Find correlation between the data by padding it with zeros and 
    performing convolution (slow method O(N^3)).
    """
    data1 = np.asarray(data1)
    N = data1.shape[0]
    if data2 is None:
        data2 = data1
    if data3 is None:
        data3 = data1
    if not(data1.shape == data2.shape == data3.shape):
        raise ValueError("Arrays of unequal lengths (shapes)")
    if data1.ndim > 1:
        raise ValueError("Too many axes in the input arrays (expected 1)")
    res = np.zeros((2*N, 2*N), dtype=data1.dtype)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                res[i-j,i-k] += data1[i] * data2[j] * data3[k] / N
    return res


def corr3_fast(data1, data2=None, data3=None):
    """
    Find correlation between the data by padding it with zeros and
    performing convolution (fast method using FFT, O(N^2 log N)).
    """
    data1 = np.asarray(data1)
    N = data1.shape[0]
    if data2 is None:
        data2 = data1
    if data3 is None:
        data3 = data1
    if not(data1.shape == data2.shape == data3.shape):
        raise ValueError("Arrays of unequal lengths (shapes)")
    if data1.ndim > 1:
        raise ValueError("Too many axes in the input arrays (expected 1)")

    L = 2 * N
    x = np.pad(data1, (0, N))
    y = np.pad(data2, (0, N))
    z = np.pad(data3, (0, N))

    fft_x = np.fft.fft(x)
    fft_y = np.fft.fft(y)
    fft_z = np.fft.fft(z)

    # Build 2D frequency-domain result:
    #   M_hat[a, b] = fft_x[(a+b) % L] * conj(fft_y[a]) * conj(fft_z[b]) / N^2
    a_idx = np.arange(L)
    b_idx = np.arange(L)
    sum_idx = (a_idx[:, None] + b_idx[None, :]) % L

    M_hat = fft_x[sum_idx] * fft_y[(-a_idx) % L][:, None] * fft_z[(-b_idx) % L][None, :]
    M_hat /= N

    return np.fft.ifft2(M_hat)