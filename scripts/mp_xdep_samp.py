# %%

import os
import glob
import numpy as np
import gvar as gv
import logging
from tqdm.auto import tqdm
import multiprocessing as mp
from lametlat.utils.log import set_up_log
from lametlat.utils.funcs import add_error_to_sample
from lametlat.extrapolation.lambda_extrapolation import (
    extrapolate_no_fit,
    bf_aft_extrapolation_plot,
)
from lametlat.utils.fourier_transform import sum_ft_re_im
from lametlat.utils.constants import *


a = 0.06
Ls = 48
Lt = 64
N_samp = 553
jk_bs = "jk"
mom_unit = 0.6085


[
    os.remove(f)
    for f in [
        *glob.glob(f"../log/extrapolation/main_{jk_bs}_process*.log"),
        f"../log/extrapolation/main_{jk_bs}.log",
    ]
    if os.path.exists(f)
]
set_up_log(f"../log/extrapolation/main_{jk_bs}.log")


def extrap_ft(
    zdep_re_bs_ls,
    zdep_im_bs_ls,
    lam_ls,
    fit_idx_range,
    x_ls,
    extrapolated_length,
    id_label,
    ifextrapolate=True,
    ifinterpolate=False,
):
    px = id_label["px"]
    py = id_label["py"]
    pz = id_label["pz"]
    b = id_label["b"]
    
    
    
    P_mom = round(px * mom_unit, 2)

    re_for_ft = []
    im_for_ft = []
    if ifextrapolate:
        zdep_re_gv = add_error_to_sample(zdep_re_bs_ls, jk_bs=jk_bs)
        zdep_im_gv = add_error_to_sample(zdep_im_bs_ls, jk_bs=jk_bs)

        for n in tqdm(
            range(N_samp), desc=f"Loop in samples for extrapolation of P{px}, b={b}"
        ):
            re_gv = zdep_re_gv[n]
            im_gv = zdep_im_gv[n]

            # todo: z=0 is real
            im_gv[0] = gv.gvar(0, 0)

            (
                extrapolated_lam_ls,
                extrapolated_re_gv,
                extrapolated_im_gv,
                fit_result_re,
                fit_result_im,
            ) = extrapolate_no_fit(  # TODO: extrapolation form
                lam_ls, re_gv, im_gv, fit_idx_range, extrapolated_length, weight_ini=0.
            )

            re_for_ft.append(gv.mean(extrapolated_re_gv))
            im_for_ft.append(gv.mean(extrapolated_im_gv))

            if n == 0:
                bf_aft_extrapolation_plot(
                    lam_ls,
                    re_gv,
                    im_gv,
                    extrapolated_lam_ls,
                    extrapolated_re_gv,
                    extrapolated_im_gv,
                    fit_idx_range,
                    title=f"P{px}_b{b}",
                    xlim=[-1, lam_ls[-1] + 4],
                    ylabel_re=r"$\tilde{h}(\lambda,~P^z=$" + f"{P_mom}" + " GeV, " + r"$b_\perp=$" + f"{b} a)",
                    ylabel_im=r"$\tilde{h}(\lambda,~P^z=$" + f"{P_mom}" + " GeV, " + r"$b_\perp=$" + f"{b} a)",
                    save_path="../log/extrapolation/",
                )

                if fit_result_re is not None:
                    logging.info("\n>>> Extrapolation fit result for real part:")
                    logging.info(fit_result_re.format(100))
                if fit_result_im is not None:
                    logging.info("\n>>> Extrapolation fit result for imag part:")
                    logging.info(fit_result_im.format(100))

        lam_for_ft = extrapolated_lam_ls

    else:
        lam_for_ft = lam_ls
        re_for_ft = zdep_re_bs_ls
        im_for_ft = zdep_im_bs_ls

    #! interpolate the lambda-dependence
    if ifinterpolate:
        # Interpolate lambda-dependence
        from scipy import interpolate

        # Define a finer lambda grid for interpolation
        lam_interp = np.linspace(min(lam_for_ft), max(lam_for_ft), len(lam_for_ft) * 10)

        re_for_ft_interp = []
        # im_for_ft_interp = []

        for n in range(len(re_for_ft)):
            # Create interpolation functions for real and imaginary parts
            f_re = interpolate.interp1d(lam_for_ft, re_for_ft[n], kind="linear")
            # f_im = interpolate.interp1d(lam_for_ft, im_for_ft[n], kind="linear")

            # Interpolate on the finer grid
            re_for_ft_interp.append(f_re(lam_interp))
            # im_for_ft_interp.append(f_im(lam_interp))

        # Update variables with interpolated values
        lam_for_ft = lam_interp
        re_for_ft = re_for_ft_interp
        im_for_ft = np.zeros_like(re_for_ft)

    xdep_re_bs_ls = []
    xdep_im_bs_ls = []

    im_for_ft = [-ls for ls in im_for_ft]  #! for convention of FT
    for n in tqdm(range(N_samp), desc=f"Loop in samples for FT of P{px}, b={b}"):

        # complete the z < 0 part
        lam_array = np.array(lam_for_ft)
        re_array = np.array(re_for_ft[n])
        im_array = np.array(im_for_ft[n])
        lam_array = np.concatenate([-lam_array[::-1][:-1], lam_array])
        re_array = np.concatenate([re_array[::-1][:-1], re_array])
        im_array = np.concatenate([-im_array[::-1][:-1], im_array])
        
        # * set imag to be zero
        im_array = np.zeros_like(re_array)

        xdep_re_samp = []
        xdep_im_samp = []
        for x in x_ls:
            temp_re, temp_im = sum_ft_re_im(
                lam_array, re_array, im_array, lam_array[1] - lam_array[0], x
            )
            xdep_re_samp.append(temp_re)
            xdep_im_samp.append(temp_im)

        xdep_re_bs_ls.append(xdep_re_samp)
        xdep_im_bs_ls.append(xdep_im_samp)

    return xdep_re_bs_ls, xdep_im_bs_ls



# %%
if __name__ == "__main__":
    # b_ls = [0, 2, 4, 6, 8, 10, 12, 14]
    b_ls = np.arange(2, 19)
    p_ls = [3, 4, 5]
    zmax = 21
    n_jobs = 8
    ifextrapolate = True
    
    extraz = 13
    
    fit_type = "ratio"

    x_ls = np.linspace(-2, 2, 1000)

    def wrapper_extrap_ft(px, b):
        bare_quasi = np.load(
            # f"../output/dump/bare_quasi_zdep_p{px}_b{b}_{fit_type}.{jk_bs}.npy",
            f"../cache/bare_quasi_zdep_p{px}_b{b}_{fit_type}.{jk_bs}.npy",
            allow_pickle=True
        ).item()  # note the shape of the data is (zmax, N_samp)

        # * p0 qTMDWF for ratio scheme
        p0_quasi = gv.load("../cache/bare_qtmdwf_zdep_p0_1st.gv")
        re_samp = np.array(bare_quasi[f"re"]).swapaxes(0, 1)
        im_samp = np.array(bare_quasi[f"im"]).swapaxes(0, 1)

        norm_factor = p0_quasi[f"b{b}_re"][0]

        # * renormalization: same as the qTMDWF
        re_samp = re_samp / gv.mean(norm_factor) 
        im_samp = np.zeros_like(re_samp)

        lam_ls = (
            np.arange(zmax) * a * lat_unit_convert(px, a, Ls, "P") / GEV_FM * 2
        )

        # TODO: adjust para for extrapolation
        if px == 3:
            fit_idx_range = [extraz, 21]
        elif px == 4:
            fit_idx_range = [extraz, 21]
        elif px == 5:
            fit_idx_range = [extraz, 21]
            
        extrapolated_length = 100
        id_label = {"px": px, "py": px, "pz": 0, "b": b}

        xdep_re_jk_ls, xdep_im_jk_ls = extrap_ft(
            re_samp,
            im_samp,
            lam_ls,
            fit_idx_range,
            x_ls,
            extrapolated_length,
            id_label,
            ifextrapolate=ifextrapolate, #TODO
            ifinterpolate=False,
        )

        result = {"re": xdep_re_jk_ls, "im": xdep_im_jk_ls}

        return result

    for px in p_ls:
        loop_params = [(px, b) for b in b_ls]
    
        with mp.Pool(processes=n_jobs) as pool:
            results = list(tqdm(
                pool.starmap(wrapper_extrap_ft, loop_params),
                desc=f"Loop in px={px}",
                total=len(loop_params)
            ))
        
        xdep_collection = {}
        for (px, b), result in zip(loop_params, results):
            xdep_collection[f"b{b}_re"] = result["re"]

        xdep_collection["x_ls"] = x_ls
        np.save(f"../output/dump/renorm_quasi_xdep_p{px}_{fit_type}_extraz={extraz}.{jk_bs}.npy", xdep_collection)




# %%
