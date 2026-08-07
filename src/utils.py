import numpy as np

def landau2murn(*args):
    if len(args) == 1:
        moduli = args[0]
        if isinstance(moduli, (list, np.ndarray)):
            A, B, C = moduli
            return A+B, B-C/2, C
        if isinstance(moduli, dict):
            A, B, C = moduli['A'], moduli['B'], moduli['C']
            return {'l':A+B, 'm':B-C/2, 'n':C}
    if len(args) == 3:
        A, B, C = args
        return A+B, B-C/2, C
    raise ValueError("Incorrect arguments.")

def murn2landau2(*args):
    if len(args) == 1:
        moduli = args[0]
        if isinstance(moduli, (list, np.ndarray)):
            l, m, n = moduli
            return l - m + n/2, m - n/2, n
        if isinstance(moduli, dict):
            l, m, n = moduli['l'], moduli['m'], moduli['n']
            return {'A':l - m + n/2, 
                    'B':m - n/2, 
                    'C':n}
    if len(args) == 3:
        l, m, n = moduli
        return l - m + n/2, m - n/2, n
    raise ValueError("Incorrect arguments.")



def voigt_to_tensor_index_map():
    return np.array([[0,0], [1,1], [2,2], [1,2], [0,2], [0,1]])

def probe_deformations(eps):
    map_ij = voigt_to_tensor_index_map()
    deforms = np.zeros((len(map_ij), 3, 3))
    for k, (i, j) in enumerate(map_ij):
        deforms[k, i, j] = eps
    return deforms, map_ij

def probe_deformations_sym(eps):
    map_ij = voigt_to_tensor_index_map()
    deforms = np.zeros((len(map_ij), 3, 3))
    for k, (i, j) in enumerate(map_ij):
        deforms[k, i, j] += eps/2
        deforms[k, j, i] += eps/2
    return deforms, map_ij


def get_cell_matrix(lmp):
    box_info = lmp.extract_box()
    boxlo, boxhi, xy, yz, xz, *_ = lmp.extract_box()
    lx, ly, lz = np.subtract(boxhi, boxlo)
    return np.array(
        [[lx, xy, xz],
         [ 0, ly, yz],
         [ 0,  0, lz]])


def set_cell_matrix(lmp, cell_matrix, origin=(0,0,0), remap=False, convert=False):
    """
    Deform cell matrix
    
    Args:
        lmp : lammps instance
        cell_matrix : 2d array
            Cell matrix with box lengths on diagonal and tilt factors off the diagonal
        origin : array-like, optional
            Default: (0, 0, 0)
        remap : bool, optional
            Controls if atoms should be displaced according to the affine deformation
            of the cell. Default: False
        convert : bool, optional
            Controls if the passed cell matrix needs to be converted from general
            to restricted triclinic box. Default: False
    """
    ((lx, xy, xz),
     (yx, ly, yz),
     (zx, zy, lz)) = cell_matrix
    if abs(yx) + abs(zx) + abs(zy) > 1e-14:
        if convert: 
            # see https://docs.lammps.org/Howto_triclinic.html#transformation-from-general-to-restricted-triclinic-boxes
            A, B, C = cell_matrix.T
            lx = np.linalg.norm(A)
            xy = B @ A / lx
            ly = np.sqrt(B@B - xy**2)
            xz = C @ A / lx
            yz = (B @ C - xy * xz) / ly
            lz = np.sqrt(C@C - xz**2 - yz**2)
        else:
            raise ValueError("Cell matrix is not of restricted triclinic box")
    boxlo = origin
    boxhi = np.add(origin, (lx, ly, lz))
    lmp.command(f"change_box all x final {boxlo[0]} {boxhi[0]} "
            f"y final {boxlo[1]} {boxhi[1]} "
            f"z final {boxlo[2]} {boxhi[2]} "
            f"xy final {xy} yz final {yz} xz final {xz} {'remap' if remap else ''} units box")
    lmp.command("run 0")


def avg_log(x, func, num_bins=50, thresh=8):
    """
    Averaging of the function in the logarithmic domain 
    excluding bins with less than 'thresh' points
    """
    x = x[x>0]
    x_bins = np.geomspace(x[0], x[-1], num=num_bins)
    ind = np.cumsum(np.histogram(x, x_bins)[0])
    ind_diff = np.diff(ind)
    mask = ind_diff>=thresh
    func_avg = np.zeros(len(ind_diff), dtype=func.dtype)
    for i, diff in enumerate(ind_diff):
        if mask[i]:
            func_avg[i] = func[ind[i]:ind[i+1]].mean()
    log_x = (x_bins[1:] + x_bins[:-1])[1:] / 2
    return log_x[mask], func_avg[mask]


def trapz_cumsum(x, xp, fp):
    """
    Smoothing of function fp of xp over the points x
    """
    bins = np.concatenate(([-np.inf], x))
    h = np.histogram(xp, bins)[0]
    ind = np.cumsum(h)
    dxp = np.diff(xp)
    assert np.all(dxp > 0)
    Np = len(xp)
    w = np.zeros(Np)
    w[0:-1] += dxp/2
    w[1:] += dxp/2
    mask = (ind != 0)*(ind != Np)
    ind_l = np.maximum(ind - 1, 0)
    ind_r = np.minimum(ind, Np - 1)
    xp_l = xp[ind_l[mask]]
    xp_r = xp[ind_r[mask]]
    w_l = np.zeros(len(x))
    w_l[ind==0] = (xp[1] - xp[0])/2
    w_l[mask] = (x[mask] - xp_r)**2/2/(xp_r - xp_l)
    w_r = np.zeros(len(x))
    w_r[mask] = (x[mask] - xp_l)**2/2/(xp_r - xp_l)
    assert fp.shape[-1] == Np
    fp_sum = np.cumsum(fp*w, axis=-1)
    return fp_sum[..., ind_l] + fp[..., ind_r]*w_r - fp[..., ind_l]*w_l
