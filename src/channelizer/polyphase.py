import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.special import erfc
from scipy.fft import fft, ifft


class PolyphaseChannelizer:
    def __init__(self, nChannel=2, fs=1, kBeta=None, nProto=64):
        self.nChannel = nChannel
        self.fs = fs
        filter1D, self.kBeta = nprCoeff(self.nChannel, nProto, kBeta, -0.25 * self.nChannel - 0.5)
        coefficients_ = np.reshape(filter1D, (self.nChannel // 2, -1), order="F")
        coefficients_ = coefficients_ / coefficients_.sum()
        self.coefficients = np.fliplr(coefficients_).T

        nChannelHalf = self.nChannel // 2
        self.fftAndInterlaceMatrix = self.calcFftMatrixAndInterlacer(nChannelHalf)

    def process(self, data):
        out = self.coreFunction(self.coefficients, data.numpy(), self.fftAndInterlaceMatrix)
        return torch.from_numpy(out);

    def coreFunction(self, coeff, dataIn, fftAndInterlaceMatrix):
        dimOut = firstNonSingletonDimension(dataIn)
        nChannel = coeff.shape[1]
        nSlice = dataIn.shape[0] // nChannel
        dataIn = dataIn[:nChannel * (len(dataIn) // nChannel)]
        
        dataIn = np.reshape(dataIn, (nSlice, nChannel))
        A = dataIn
        B = ((2 * np.mod(np.arange(1, nSlice+1), 2) - 1)[:, np.newaxis] * dataIn *
             np.exp(1j * np.pi * np.arange(nChannel) / nChannel))
        S = A.shape
        # S = (S[0] + coeff.shape[0] - 1, S[1], 2)
        coeff = coeff.astype(A.dtype) * nChannel
        dataIn = np.concatenate([A[..., np.newaxis], B[..., np.newaxis]], axis=2)
        dataIn = convnfft(dataIn, coeff, 'same', 0);
        dataOut = dataIn
        dataOut = np.fft.ifft(dataOut, axis=1)
        ind = np.roll(np.fliplr(np.arange(0, nChannel*2).reshape((-1, 2))), 1, axis=1).ravel() - 1
        out = np.reshape(dataOut,(S[0],-1));
        out = out[:,ind];
        # S = list(np.ones(dimOut+1, dtype=np.int64))
        # S[dimOut] = out.shape[0]
        # S[dimOut+1] = out.shape[1]
        # out = np.reshape(out, tuple(S), order="F")
        return out

    # def calcFftMatrixAndInterlacer(self, nChannelOver2):
    #     colVector = np.arange(nChannelOver2)
    #     rowVector = np.roll(np.arange(nChannelOver2), -1)
    #     rowMatrix = np.zeros((nChannelOver2, nChannelOver2), dtype=np.int64)
    #     rowMatrix[colVector, rowVector] = 1
    #     matrixTop = np.concatenate((np.eye(nChannelOver2), rowMatrix), axis=1)
    #     matrixBottom = np.concatenate((rowMatrix, np.eye(nChannelOver2)), axis=1)
    #     matrixFull = np.concatenate
    
    def calcFftMatrixAndInterlacer(self, nChannelOver2):
        # calculate FFT matrix
        m = np.fft.fft(np.eye(nChannelOver2))

        # calculate interlacer matrix
        l = np.zeros((nChannelOver2, nChannelOver2))
        for i in range(nChannelOver2):
            for j in range(nChannelOver2):
                l[i, j] = ((-1)**(i+j)) * np.exp(-2j*np.pi*(i*j)/(2*nChannelOver2))

        # combine the two matrices
        fftAndInterlaceMatrix = np.dot(l, m)

        return fftAndInterlaceMatrix


def convnfft(A, B, shape='full', dims=None):
    
    nd = max(np.ndim(A), np.ndim(B))
    # work on all dimensions by default
    if dims is None:
        dims = np.arange(1, nd+1)
    dims = np.reshape(dims, (1, -1)) # row (needed for for-loop index)
    
    # IFUN function will be used later to truncate the result
    # M and N are respectively the length of A and B in some dimension
    if shape.lower() == 'full':
        ifun1 = lambda m, n : 1
        ifun2 = lambda m, n : m + n - 1
        sizeFunc = lambda m, n : max([n+m-1,n,m])
    elif shape.lower() == 'same':
        ifun1 = lambda m, n : np.ceil((n-1)/2)+1
        ifun2 = lambda m, n : np.ceil((n-1)/2)+m
        sizeFunc = lambda m, n : m
    elif shape.lower() == 'valid':
        ifun1 = lambda m, n : n
        ifun2 = lambda m, n : m
        sizeFunc = lambda m, n : max(m - max(0, n - 1), 0)
    else:
        raise ValueError('convnfft: unknown shape %s' % shape)
    
    ABreal = np.isreal(A).all() and np.isreal(B).all()
    
    # Special case, empty convolution, try to follow MATLAB CONVN convention
    if np.any(np.array(A.shape) == 0) or np.any(np.array(B.shape) == 0):
        szA = np.zeros(nd)
        szA[0:np.ndim(A)] = A.shape
    
        szC = szA.copy()
        for dim in dims[0]:
            szC[dim-1] = sizeFunc(A.shape[dim-1], B.shape[dim-1])
        A = np.zeros(szC, dtype=A.dtype) # empty -> return zeros
        return A
    
    # slower, but smaller temporary arrays
    lfftfun = lambda x : x
    
    # # no need to convolve on dims of 1
    nn = max([np.max(dims), np.ndim(A), np.ndim(B)])
    s1 = sizeMinDim(A, nn)
    s2 = sizeMinDim(B, nn)
    kSkip = (s1[dims-1] == 1) & (s2[dims-1] == 1)
    dims = np.delete(dims[0], np.where(kSkip))
    
    # Do the FFT
    szA = sizeMinDim(A, max(dims))
    szB = sizeMinDim(B, max(dims))
    for dim in dims:
        # compute the FFT length
        x = lfftfun(szA[dim] + szB[dim] - 1)
        A = np.fft.fft(A, x, dim)
        B = np.fft.fft(B, x, dim)

    # the product in fft space
    for di in range(dim):
        A[..., di] = A[..., di] * B
    B = None

    # Back to the non-Fourier space
    for dim in dims:
        A = np.fft.ifft(A, None, dim)

        # Truncate the results
        m = szA[dim]
        n = szB[dim]
        A = indexAtDimRegular(A, dim, int(ifun1(m, n)), int(ifun2(m, n)))

    if ABreal:
        # Make sure the result is real
        A = np.real(A)

    return A

def sizeMinDim(array, minDim=None):
    if minDim is None:
        minDim = array.ndim
    sizeArray = np.array(array.shape)
    if sizeArray.size < minDim:
        sizeArray = np.pad(sizeArray, (0, minDim - sizeArray.size), 'constant', constant_values=1)
    return sizeArray

def indexAtDimRegular(data, dim, start, stop, step=1):

    S = np.shape(data)

    data = data[start-1:stop:step, :]
    S = list(S)
    S[0] = np.shape(data)[0]

    data = np.reshape(data, S)
    return data

def rrerf(F, K, M):
    x = K * (2 * M * F - 0.5)
    A = np.sqrt(0.5 * erfc(x))
    return A

def nprCoeff(N, L=128, K=None, shiftPixels=0):
    """
    Nonparametric Reconstruction (NPR) Coefficients
    """
    if K is None:
        # if the K value is not specified us a default value.
        # these values minimize the reconstruction error.
        lookupTable = np.array([[8, 4.853], [10, 4.775], [12, 5.257], [14, 5.736], [16, 5.856], [18, 7.037],
                                [20, 6.499], [22, 6.483], [24, 7.410], [26, 7.022], [28, 7.097], [30, 7.755],
                                [32, 7.452], [48, 8.522], [64, 9.396], [96, 10.785], [128, 11.5], [192, 11.5], [256, 11.5]])

        K = np.interp(np.log(L), np.log(lookupTable[:, 0]), lookupTable[:, 1]) * 2.5

    M = N // 2
    F = np.arange(L * M) / (L * M)

    # The prototype is based on a root raised error function
    A = rrerf(F, K, M)
    N = len(A)

    n = np.arange(N // 2)
    A[N-n-1] = np.conj(A[2+n])
    A[N // 2] = 0

    A = A.astype(np.complex128);
    # shift
    i,_ = getIJVector(len(A));
    arr = np.exp(-2j * np.pi / len(A) * shiftPixels * i);
    A *= np.fft.fftshift(arr);

    B = np.real(np.fft.ifft(A))
    B = np.fft.fftshift(B)

    B /= np.sum(B)
    coeff = np.reshape(B, (M, L))

    return coeff, K

def firstNonSingletonDimension(x):
    """
    Find the first non-singleton dimension in array, or output 1 if a scalar.

    Args:
        x: array_like, input array.

    Returns:
        dim: int, the index of the first non-singleton dimension, or 1 if a scalar.

    """
    sizeX = np.asarray(x).shape
    dim = np.where(sizeX != 1)[0]
    if len(dim) == 0:
        dim = 1
    else:
        dim = dim[0]
    return dim


def getIJVector(H, W=None, pixelSize=None):
    if W is None:
        W = H
    if pixelSize is None:
        pixelSize = np.ones((2,))
    elif np.isscalar(pixelSize):
        pixelSize = np.array([pixelSize, pixelSize])
    else:
        pixelSize = np.asarray(pixelSize)
        if pixelSize.size == 1:
            pixelSize = np.array([pixelSize, pixelSize])
    
    I = getVector(H)
    J = getVector(W)

    I = I * pixelSize[0]
    J = J * pixelSize[1]

    I = I.ravel()
    J = J.ravel()

    return I, J

def getVector(n):
    nIn = n
    if n > 3e7 and np.float32(n) != np.float64(n):
        print('data is too large for single precision, array with not be accurate')
    if n > 1e7:
        n = np.double(n)
    if n % 2 == 1:
        x = -1 * ((n-1)/2) + np.arange(n)
    else:
        x = np.arange(n) - n/2
    return x