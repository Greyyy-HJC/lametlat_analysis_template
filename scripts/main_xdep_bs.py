# %%

import os
import glob
import numpy as np
import gvar as gv
import pandas as pd
from multiprocessing import current_process
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from lametlat.utils.log import set_up_log
from lametlat.utils.funcs import add_error_to_sample
from lametlat.extrapolation.lambda_extrapolation import (
    extrapolate_exp,
    extrapolate_Regge,
    bf_aft_extrapolation_plot,
)
from lametlat.utils.fourier_transform import sum_ft_re_im
from lametlat.utils.constants import *


a = 0.06
Ls = 48
Lt = 64
data_tsep_ls = [4, 6, 8, 10, 12]
N_samp = 200


[
    os.remove(f)
    for f in [
        *glob.glob("../log/extrapolation/main_bs_process*.log"),
        "../log/extrapolation/main_bs.log",
    ]
    if os.path.exists(f)
]
set_up_log("../log/extrapolation/main_bs.log")


def extrap_ft(
    zdep_re_bs_ls,
    zdep_im_bs_ls,
    lam_ls,
    fit_idx_range,
    x_ls,
    extrapolated_length,
    id_label,
    ifextrapolate=True,
    ifsep_reim=True,
    ifinterpolate=False,
):
    px = id_label["px"]
    py = id_label["py"]
    pz = id_label["pz"]
    b = id_label["b"]
    zs = id_label["zs"]

    re_for_ft = []
    im_for_ft = []
    if ifextrapolate:
        zdep_re_gv = add_error_to_sample(zdep_re_bs_ls)
        zdep_im_gv = add_error_to_sample(zdep_im_bs_ls)

        for n in tqdm(
            range(N_samp), desc=f"Loop in samples for extrapolation of P{px}, b={b}"
        ):
            re_gv = zdep_re_gv[n]
            im_gv = zdep_im_gv[n]

            (
                extrapolated_lam_ls,
                extrapolated_re_gv,
                extrapolated_im_gv,
                fit_result_re,
                fit_result_im,
            ) = extrapolate_exp(
                lam_ls, re_gv, im_gv, fit_idx_range, extrapolated_length
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
                    title=f"P{px}_b{b}_zs{zs}",
                    xlim=[-1, 110],
                    save_path="/home/jinchen/git/anl/cg_nn_pdf/log/extrapolation/",
                )

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
        im_for_ft_interp = []
        
        for n in range(len(re_for_ft)):
            # Create interpolation functions for real and imaginary parts
            f_re = interpolate.interp1d(lam_for_ft, re_for_ft[n], kind='linear')
            f_im = interpolate.interp1d(lam_for_ft, im_for_ft[n], kind='linear')
            
            # Interpolate on the finer grid
            re_for_ft_interp.append(f_re(lam_interp))
            im_for_ft_interp.append(f_im(lam_interp))
        
        # Update variables with interpolated values
        lam_for_ft = lam_interp
        re_for_ft = re_for_ft_interp
        im_for_ft = im_for_ft_interp
        

    xdep_re_bs_ls = []
    xdep_im_bs_ls = []

    im_for_ft = [-ls for ls in im_for_ft] #! for convention of FT
    for n in tqdm(range(N_samp), desc=f"Loop in samples for FT of P{px}, b={b}"):
        
        # complete the z < 0 part
        lam_array = np.array(lam_for_ft)
        re_array = np.array(re_for_ft[n])
        im_array = np.array(im_for_ft[n])
        lam_array = np.concatenate([-lam_array[::-1][:-1], lam_array])
        re_array = np.concatenate([re_array[::-1][:-1], re_array])
        im_array = np.concatenate([-im_array[::-1][:-1], im_array])
        
        xdep_re_samp = []
        xdep_im_samp = []
        for x in x_ls:
            #TODO: check ifsep_reim
            if not ifsep_reim:
                temp_re, temp_im = sum_ft_re_im(
                    lam_array, re_array, im_array, lam_array[1] - lam_array[0], x
                )
                xdep_re_samp.append(temp_re)
                xdep_im_samp.append(temp_im)
            else:
                temp_re, _ = sum_ft_re_im(
                    lam_array,
                    re_array,
                    np.zeros_like(im_array),
                    lam_array[1] - lam_array[0],
                    x,
                )
                temp_im, _ = sum_ft_re_im(
                    lam_array,
                    np.zeros_like(re_array),
                    im_array,
                    lam_array[1] - lam_array[0],
                    x,
                )
                xdep_re_samp.append(temp_re)
                xdep_im_samp.append(temp_im)

        xdep_re_bs_ls.append(xdep_re_samp)
        xdep_im_bs_ls.append(xdep_im_samp)

    return xdep_re_bs_ls, xdep_im_bs_ls


# %%
if __name__ == "__main__":
    b = 0
    n_jobs = 6
    p_ls = [4, 5]
    zs_ls = [2, 3, 4]
    zmax = 21
    fit_method = "sum_81012"  # "sum_81012" or "sum_two_state_681012"
    ifextrapolate = True
    re_im_sep = False

    nn = 4000
    xmin = -1.99951
    xmax = 2.00049
    x_ls = np.linspace(xmin, xmax, nn)

    def wrapper_extrap_ft(px, b, zs, n_jobs):
        if n_jobs > 1:
            process_id = current_process()._identity[0] - 1  # Process IDs start from 1
            set_up_log(f"../log/extrapolation/main_bs_process{process_id}.log")

        renorm_quasi = gv.load(
            f"../cache/renorm_quasi_p45_b0_zs{zs}_{fit_method}.dat"
        )  # note the shape of the data is (zmax, N_samp)

        lam_ls = (
            np.arange(zmax) * np.sqrt(2) * a * lat_unit_convert(px * np.sqrt(2), a, Ls, "P") / GEV_FM
        )

        # TODO: adjust para for extrapolation
        fit_idx_range = [12, 21]
        extrapolated_length = 100
        id_label = {"px": px, "py": px, "pz": 0, "b": b, "zs": zs}

        xdep_re_bs_ls, xdep_im_bs_ls = extrap_ft(
            renorm_quasi[f"re_p{px}"].swapaxes(0, 1),
            renorm_quasi[f"im_p{px}"].swapaxes(0, 1),
            lam_ls,
            fit_idx_range,
            x_ls,
            extrapolated_length,
            id_label,
            ifextrapolate=ifextrapolate, #TODO
            ifsep_reim=re_im_sep, #TODO: check ifsep_reim
        )

        result = {"re": xdep_re_bs_ls, "im": xdep_im_bs_ls}

        return result

    loop_params = [(px, b, zs) for px in p_ls for zs in zs_ls]
    results = Parallel(n_jobs=n_jobs)(
        delayed(wrapper_extrap_ft)(px, b, zs, n_jobs) for px, b, zs in loop_params
    )

    xdep_collection = {}
    for (px, b, zs), result in zip(loop_params, results):
        xdep_collection[f"re_p{px}_b{b}_zs{zs}"] = result["re"]
        xdep_collection[f"im_p{px}_b{b}_zs{zs}"] = result["im"]
        
    if re_im_sep:
        gv.dump(xdep_collection, f"../output/dump/xdep_collection_p45_b0_zsmix_{fit_method}_re_im_sep.dat")
    else:
        gv.dump(xdep_collection, f"../output/dump/xdep_collection_p45_b0_zsmix_{fit_method}.dat")

    #! Output xdep_collection to txt file
    if False:
        for zs in zs_ls:
            output_data = []
            for i, x in enumerate(x_ls):
                for n_conf in range(len(xdep_collection[f"re_p4_b0_zs{zs}"])):
                    output_data.append({
                        "n_conf": n_conf,
                        "x": x,
                        "p4_real": xdep_collection[f"re_p4_b0_zs{zs}"][n_conf][i],
                        "p4_imag": xdep_collection[f"im_p4_b0_zs{zs}"][n_conf][i],
                        "p5_real": xdep_collection[f"re_p5_b0_zs{zs}"][n_conf][i],
                        "p5_imag": xdep_collection[f"im_p5_b0_zs{zs}"][n_conf][i]
                    })

            df = pd.DataFrame(output_data)
            filename = f"../output/else/unpol_quasi_tmd_b0_xdep_p4p5_zs{zs}_{fit_method}_re_im_sep.txt"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            df.to_csv(filename, sep="\t", index=False)
            print(f"Data has been saved to '{filename}'")


# %%
