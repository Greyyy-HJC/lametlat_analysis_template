# %%
import h5py as h5
import numpy as np
import gvar as gv
from lametlat.utils.resampling import jackknife, bootstrap

a = 0.06 # fm
mpi = 0.3 # GeV
Ls = 48
Lt = 64
N_conf = 553
tsep_ls = [6, 8, 10, 12]
bin = 5
N_bs_samp = 200

def bad_point_filter(data, threshold=1):
    """
    Filter out the bad points in the data.

    Args:
        data (ndarray): The input data.
        threshold (float, optional): The threshold value. Defaults to 1.

    Returns:
        ndarray: The filtered data.
    """
    mask = np.abs(data) > threshold
    bad_loc = np.argwhere(mask)

    for loc in bad_loc:
        data[tuple(loc)] = np.random.choice([-1, 1])

    return data

def get_2pt_data(ss_sp, px, py, pz, jk_bs=None):
    """
    Get the 2-point data for a given set of parameters.

    Parameters:
    - ss_sp (str): The value of ss_sp parameter.
    - px (int): The value of px parameter.
    - py (int): The value of py parameter.
    - pz (int): The value of pz parameter.
    - jk_bs (str or None): The type of analysis to perform. Options are "jk" for jackknife, "bs" for bootstrap, or None for no analysis.

    Returns:
    - pt2_real (ndarray): The real part of the 2-point data.
    - pt2_imag (ndarray): The imaginary part of the 2-point data.

    """

    if px == 3:
        pt2_file = f"../data/c2pt_comb/c2pt.CG52bxyp20_CG52bxyp20.{ss_sp}.meson_g15.PX{px}_PY{py}_PZ{pz}"
    else:
        pt2_file = f"../data/c2pt_comb/c2pt.CG52bxyp30_CG52bxyp30.{ss_sp}.meson_g15.PX{px}_PY{py}_PZ{pz}"

    # read csv file
    pt2_real = np.loadtxt(pt2_file + ".real", skiprows=1, delimiter=",")
    pt2_imag = np.loadtxt(pt2_file + ".imag", skiprows=1, delimiter=",")


    # the idx 0 on axis 1 is the time slice, so we remove it and swap axes to make configurations on axis 0
    pt2_real = np.swapaxes(pt2_real[:, 1:], 0, 1)
    pt2_imag = np.swapaxes(pt2_imag[:, 1:], 0, 1)
    pt2_real = bad_point_filter(pt2_real)
    pt2_imag = bad_point_filter(pt2_imag)

    if jk_bs == None:
        return pt2_real, pt2_imag
    
    elif jk_bs == "jk":
        pt2_real_jk = jackknife(pt2_real)
        pt2_imag_jk = jackknife(pt2_imag)

        return pt2_real_jk, pt2_imag_jk
    
    elif jk_bs == "bs":
        pt2_real_bs, _ = bootstrap(pt2_real, samp_times=N_bs_samp, bin=bin)
        pt2_imag_bs, _ = bootstrap(pt2_imag, samp_times=N_bs_samp, bin=bin)

        return pt2_real_bs, pt2_imag_bs

def get_3pt_data(px, py, pz, b, z, tsep_ls, jk_bs=None):
    """
    Retrieve 3-point correlation function data from a file.

    Args:
        px (int): Momentum component in the x-direction.
        py (int): Momentum component in the y-direction.
        pz (int): Momentum component in the z-direction.
        b (int): Binning index.
        z (int): Z index.
        tsep_ls (list): List of time separations.
        jk_bs (str, optional): Jackknife or bootstrap method. Defaults to None.

    Returns:
        tuple: A tuple containing the real and imaginary parts of the 3-point correlation function data.
            If jk_bs is None, returns (pt3_real, pt3_imag).
            If jk_bs is "jk", returns (pt3_real_jk, pt3_imag_jk).
            If jk_bs is "bs", returns (pt3_real_bs, pt3_imag_bs).
    """
    
    if px == 4 or px == 5:
        pt3_file = f"../data/c3pt_h5/qpdf.SS.meson.ama.CG52bxyp30_CG52bxyp30.PX{px}_PY{py}_PZ{pz}.Z0-24.XY0-24.g8.qx0_qy0_qz0.h5"
    elif px == 3:
        pt3_file = f"../data/c3pt_h5/qpdf.SS.meson.ama.CG52bxyp20_CG52bxyp20.PX{px}_PY{py}_PZ{pz}.Z0-24.XY0-24.g8.qx0_qy0_qz0.h5"

    else:
        print(">>> We only have PX=PY=3, PX=PY=4 and PX=PY=5 data now.")

    pt3_data = []
    for tsep in tsep_ls:
        data = h5.File(pt3_file, "r")[f"dt{tsep}"][f"Z{b}"][f"XY{z}"][:] # * Note here z = 1 means x = y = 1, i.e. the real separation is np.sqrt(2)
        pt3_data.append(data)

    pt3_data = np.array(pt3_data)
    pt3_data = np.swapaxes(pt3_data, 0, 2)
    pt3_data = np.swapaxes(pt3_data, 1, 2)

    pt3_real = np.real(pt3_data)
    pt3_imag = np.imag(pt3_data)
    pt3_real = bad_point_filter(pt3_real)
    pt3_imag = bad_point_filter(pt3_imag)

    if jk_bs == None:
        return pt3_real, pt3_imag
    
    elif jk_bs == "jk":
        pt3_real_jk = jackknife(pt3_real)
        pt3_imag_jk = jackknife(pt3_imag)

        return pt3_real_jk, pt3_imag_jk
    
    elif jk_bs == "bs":
        pt3_real_bs, _ = bootstrap(pt3_real, samp_times=N_bs_samp, bin=bin)
        pt3_imag_bs, _ = bootstrap(pt3_imag, samp_times=N_bs_samp, bin=bin)

        return pt3_real_bs, pt3_imag_bs

def get_ratio_data(px, py, pz, b, z, tsep_ls, jk_bs="bs"):
    """
    Calculate the ratio of 3pt correlators to 2pt correlators.

    Parameters:
    px (float): Momentum component x.
    py (float): Momentum component y.
    pz (float): Momentum component z.
    b (float): Impact parameter.
    z (float): Light-cone momentum fraction.
    tsep_ls (list): List of time separations.
    jk_bs (str, optional): Jackknife or bootstrap method. Defaults to "jk".

    Returns:
    np.array: Array of real parts of the ratio.
    np.array: Array of imaginary parts of the ratio.
    """
    # * take 2pt_ss as the denominator, do the ratio on each sample
    if jk_bs == "jk":
        pt2_real, pt2_imag = get_2pt_data("SS", px, py, pz, jk_bs="jk")
        pt3_real, pt3_imag = get_3pt_data(px, py, pz, b, z, tsep_ls, jk_bs="jk")
    elif jk_bs == "bs":
        pt2_real, pt2_imag = get_2pt_data("SS", px, py, pz, jk_bs="bs")
        pt3_real, pt3_imag = get_3pt_data(px, py, pz, b, z, tsep_ls, jk_bs="bs")

    ra_real = []
    ra_imag = []
    N_samp = len(pt2_real)

    for n in range(N_samp):
        ra_real.append([])
        ra_imag.append([])
        for id in range(len(tsep_ls)):
            tsep = tsep_ls[id]

            pt2_complex = pt2_real[n][tsep] + 1j * pt2_imag[n][tsep]
            pt3_complex = pt3_real[n][id] + 1j * pt3_imag[n][id]
            # * use complex divide complex to get ratio
            ra_complex = pt3_complex / pt2_complex

            ra_real[n].append(np.real(ra_complex))
            ra_imag[n].append(np.imag(ra_complex))
            # * here includes all 16 tau values from 0 to 15

    return np.array(ra_real), np.array(ra_imag)
    # * shape = ( N_samp, len(tsep_ls), 16 )

def get_sum_data(px, py, pz, b, z, tsep_ls, jk_bs="bs", tau_cut=1):
    """
    Calculate the sum of the ratio data over the tau axis.

    Args:
        px (array-like): x-component of momentum.
        py (array-like): y-component of momentum.
        pz (array-like): z-component of momentum.
        b (array-like): impact parameter.
        z (array-like): light-cone momentum fraction.
        tsep_ls (array-like): list of time separations.
        jk_bs (str, optional): jackknife binning scheme. Defaults to "jk".
        tau_cut (int, optional): number of contact points to cut. Defaults to 1.

    Returns:
        tuple: A tuple containing the sum of the real and imaginary parts of the ratio data.
               The shape of each element in the tuple is (N_samp, len(tsep_ls)).
    """
    ra_real, ra_imag = get_ratio_data(px, py, pz, b, z, tsep_ls, jk_bs=jk_bs)

    sum_real, sum_imag = [], []

    for id in range(len(tsep_ls)):
        tsep = tsep_ls[id]
        cutted_real = ra_real[:, id, tau_cut : tsep - tau_cut + 1]
        cutted_imag = ra_imag[:, id, tau_cut : tsep - tau_cut + 1]

        sum_real.append(np.sum(cutted_real, axis=1))
        sum_imag.append(np.sum(cutted_imag, axis=1))

    sum_real = np.swapaxes(
        np.array(sum_real), 0, 1
    )  # * swap the sample axis to the 0-th axis
    sum_imag = np.swapaxes(np.array(sum_imag), 0, 1)

    return sum_real, sum_imag

def get_fh_data(px, py, pz, b, z, tsep_ls, jk_bs="jk", tau_cut=1):
    """
    Calculate the FH data, FH = sum(t + 1) - sum(t)
    """
    sum_real, sum_imag = get_sum_data(px, py, pz, b, z, tsep_ls, jk_bs=jk_bs, tau_cut=tau_cut)
    
    tsep_gap = tsep_ls[1] - tsep_ls[0]
    
    fh_real = (sum_real[:, 1:] - sum_real[:, :-1]) / tsep_gap
    fh_imag = (sum_imag[:, 1:] - sum_imag[:, :-1]) / tsep_gap

    return fh_real, fh_imag

# %%