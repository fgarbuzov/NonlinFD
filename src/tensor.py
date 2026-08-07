import numpy as np

def sym_dyad(tens1, tens2, dim=3):
    """
    R_ijkl = 1/4 (tens1_ik tens2_jl + tens1_il tens2_jk
                + tens1_jk tens2_il + tens1_jl tens2_ik)
    For symmetric unity 4th order tensor run sym_dyad(np.eye(3), np.eye(3))
    """
    if tens1.shape[-2:] != tens2.shape[-2:]:
        raise ValueError("Incompatible tensor shapes")
    
    batch_shape = tens1.shape[:-2] + tens2.shape[:-2]
    res = np.zeros(batch_shape + (dim,)*4)
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                for l in range(dim):
                    res[...,i,j,k,l] = (tens1[...,i,k]*tens2[...,j,l] 
                                        + tens1[...,i,l]*tens2[...,j,k]
                                        + tens1[...,j,k]*tens2[...,i,l] 
                                        + tens1[...,j,l]*tens2[...,i,k]) / 4
    return res

def get_I4(dim=3):
    return sym_dyad(np.eye(dim), np.eye(dim), dim)

def isotropic3(moduli:dict):
    if all(key in moduli for key in ('l', 'm', 'n')):
        A = moduli['l'] - moduli['m'] + moduli['n']/2
        B = moduli['m'] - moduli['n']/2
        C = moduli['n']
    elif all(key in moduli for key in ('A', 'B', 'C')):
        A = moduli['A']
        B = moduli['B']
        C = moduli['C']
    else:
        raise ValueError("Unknown moduli type.")
    dim = 3
    res = np.zeros((dim,)*6)
    I = np.eye(3)
    I4 = get_I4()
    # for i in range(dim):
    #     for j in range(dim):
    #         for k in range(dim):
    #             for l in range(dim):
    #                 for m in range(dim):
    #                     for n in range(dim):
    #                         res[i,j,k,l,m,n] = (A*(i==j)*(k==l)*(m==n) + 
    #                                             B*((i==j)*I4[k,l,m,n] + 
    #                                                (k==l)*I4[i,j,m,n] + 
    #                                                (m==n)*I4[i,j,k,l]
    #                                               ) + 
    #                                             C*((i==k)*I4[j,l,m,n] + 
    #                                                (j==k)*I4[i,l,m,n] +
    #                                                (i==l)*I4[j,k,m,n] +
    #                                                (j==l)*I4[i,k,m,n]
    #                                               ) / 4)
    res += A * np.einsum('ij,kl,mn->ijklmn', I, I, I)
    
    # Term 2: δ_ij I4_klmn + δ_kl I4_ijmn + δ_mn I4_ijkl
    res += B * (np.einsum('ij,klmn->ijklmn', I, I4) +
                np.einsum('kl,ijmn->ijklmn', I, I4) +
                np.einsum('mn,ijkl->ijklmn', I, I4))
    
    # Term 3: (δ_ik I4_jlmn + δ_jk I4_ilmn + δ_il I4_jkmn + δ_jl I4_ikmn) / 4
    res += (C/4) * (np.einsum('ik,jlmn->ijklmn', I, I4) +
                    np.einsum('jk,ilmn->ijklmn', I, I4) +
                    np.einsum('il,jkmn->ijklmn', I, I4) +
                    np.einsum('jl,ikmn->ijklmn', I, I4))
    return res

def fit_isotropic3(target_tensor: np.ndarray):
    basis_A = isotropic3({'A':1, 'B':0, 'C':0})
    basis_B = isotropic3({'A':0, 'B':1, 'C':0})
    basis_C = isotropic3({'A':0, 'B':0, 'C':1})
    
    target_flat = target_tensor.flatten()
    X = np.column_stack([basis_A.flatten(), basis_B.flatten(), basis_C.flatten()])
    params, _, _, _ = np.linalg.lstsq(X, target_flat, rcond=None)
    
    moduli = {'A': params[0],
              'B': params[1],
              'C': params[2]}
    re = np.linalg.norm(isotropic3(moduli) - target_tensor) 
    re/= np.linalg.norm(target_tensor)
    return moduli, re


def lammps_array_to_matrix(arr):
    """
    Convert a LAMMPS-style packed array into a full symmetric matrix.

    Parameters
    ----------
    arr : array_like
        Input array of shape (..., L), where L = n*(n+1)/2 for some integer n.
        The last dimension contains the diagonal elements (first n values)
        followed by the upper-triangular off-diagonal elements.

    Returns
    -------
    result : numpy.ndarray
        Array of shape (..., n, n) with the reconstructed symmetric matrix.
        Has the same dtype as the input.

    Raises
    ------
    ValueError
        If the size of the last dimension does not correspond to a valid
        upper-triangular matrix size n*(n+1)/2.
    """
    
    arr = np.asarray(arr)
    
    # determine the matrix size n
    arr_len = arr.shape[-1]
    n = int((-1 + np.sqrt(1 + 8 * arr_len)) / 2)
    if n * (n + 1) // 2 != arr_len:
        raise ValueError("Array length does not match an upper-triangular matrix size")

    output_shape = arr.shape[:-1] + (n, n)
    result = np.zeros(output_shape, dtype=arr.dtype)
    result[..., *np.triu_indices(n, k=1)] = arr[...,n:]
    result = result + np.swapaxes(result, -2, -1)
    result[..., *np.diag_indices(n)] = arr[..., :n]
    
    return result



def to_traceless_upper_triangular(arr, axis=-1):
    """
    Convert upper triangular elements to "traceless" (zero-diagonal) 
    upper triangular matrices.

    Args:
        arr: ndarray where the specified axis contains triangular elements
        axis: axis along which the triangular elements are stored (default: -1)
    
    Returns:
        (N+1)D array for N-dimensional input array with same dimensions 
        except the specified axis is replaced by two new axes of the traceless
        upper triangular matrix
    
    Raises:
        ValueError: If input array length isn't a triangular number (1,3,6,10,15...)
    
    Notes:
        - Diagonal and lower triangle are filled with zeros
        - Elements fill upper triangle in row-major order (lammps style)
    
    Example:
        >>> to_traceless_upper_triangular([1,2,3])  # 3x3 matrix
        array([[0,1,2],
               [0,0,3],
               [0,0,0]])
    """
    
    arr = np.asarray(arr)
    if arr.ndim > 1:
        arr = np.moveaxis(arr, axis, -1)
    
    # determine the matrix size n
    arr_len = arr.shape[-1]
    n = int((1 + np.sqrt(1 + 8 * arr_len)) / 2)
    if n * (n - 1) // 2 != arr_len:
        raise ValueError("Array length does not match an upper-triangular matrix size")

    output_shape = arr.shape[:-1] + (n, n)
    result = np.zeros(output_shape, dtype=arr.dtype)
    result[..., *np.triu_indices(n, k=1)] = arr
    result = np.moveaxis(result, [-2, -1], [axis-1, axis])
    
    return result


def voigt_to_tensor(C_voigt, order=2):
    """
    Convert a stiffness tensor in Voigt notation to a full tensor.
    
    Args:
        C_voigt: ndarray
            Input array in Voigt notation. Last `order` dimensions should be square
            with size 3 (2D) or 6 (3D).
        order: int, optional
            Number of Voigt dimensions. Default: 2
    
    Returns:
        C_tensor: ndarray
            Full tensor
    """
    
    batch_shape = C_voigt.shape[:-order]
    voigt_shape = C_voigt.shape[-order:]
    voigt_n = voigt_shape[-1]
    if all(voigt_n != s for s in voigt_shape):
        raise ValueError("Input array is not a square matrix/tensor")
    
    if voigt_n == 3:
        map_ij = np.array([[0,0], [1,1], [0,1]])
        dim = 2
    elif voigt_n == 6:
        map_ij = np.array([[0,0], [1,1], [2,2], [1,2], [0,2], [0,1]])
        dim = 3
    else:
        raise ValueError("Input array is not of Voigt shape (3 or 6)")

    C_tensor = np.zeros(batch_shape + (dim,) * (2 * order), dtype=C_voigt.dtype)
    voigt_indices = np.indices((voigt_n,) * order).reshape(order, -1)
    for idx in range(voigt_indices.shape[1]):
        voigt_idx = tuple(voigt_indices[:, idx])
        spatial_idx = ()
        for v_idx in voigt_idx:
            spatial_idx += (map_ij[v_idx, 0], map_ij[v_idx, 1])
        C_tensor[(...,) + spatial_idx] = C_voigt[(...,) + voigt_idx]

    # symmetrization
    for a in range(order):
        C_tensor = np.moveaxis(C_tensor, [-2*a-2, -2*a-1], [-2, -1])
        for i,j in map_ij[dim:]:
            C_tensor[..., j, i] = C_tensor[..., i, j]
        C_tensor = np.moveaxis(C_tensor, [-2, -1], [-2*a-2, -2*a-1])
    
    return C_tensor