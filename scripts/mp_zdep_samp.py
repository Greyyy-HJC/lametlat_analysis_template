# %%
import os
import glob
import itertools
import numpy as np
import gvar as gv
import matplotlib.pyplot as plt
import multiprocessing as mp
from tqdm import tqdm
from read_data import get_2pt_data, get_ratio_data, get_fh_data

from lametlat.utils.plot_settings import fs_p
from lametlat.utils.funcs import add_error_to_sample
from lametlat.utils.log import set_up_log
from lametlat.utils.resampling import jk_ls_avg, bs_ls_avg
from lametlat.gsfit.pt2_fit import pt2_two_state_fit
from lametlat.gsfit.ratio_fit import ra_two_state_fit, plot_ra_fit_on_data
from lametlat.gsfit.fh_fit import fh_one_state_fit, plot_fh_fit_on_data


a = 0.06
Ls = 48
Lt = 64
jk_bs = "jk"
N_samp = 553
data_tsep_ls = [6, 8, 10, 12]


[os.remove(f) for f in [*glob.glob(f'../log/gsfit/main_{jk_bs}_process*.log'), f'../log/gsfit/main_{jk_bs}.log'] if os.path.exists(f)]
set_up_log(f"../log/gsfit/main_{jk_bs}.log")


def wrapper_ratio_fit(px, b, z, fit_tsep_ls, tau_cut):
    
    py = px
    pz = 0
    
    id_label = {"px": px, "py": py, "pz": pz, "b": b, "z": z}

    # * 2pt data
    pt2_ss_re, _ = get_2pt_data('SS', px, py, pz, jk_bs)
    pt2_ss_re_gv = add_error_to_sample(pt2_ss_re, jk_bs)

    # * ratio data
    ra_re, ra_im = get_ratio_data(px=px, py=py, pz=pz, b=b, z=z, tsep_ls=data_tsep_ls, jk_bs=jk_bs) # shape = (N_samp, len(tsep_ls), 16)
    
    # Add error to each tsep separately
    ra_re_gv_dict = {f'tsep_{tsep}': add_error_to_sample(ra_re[:, i, :], jk_bs) for i, tsep in enumerate(data_tsep_ls)}
    ra_im_gv_dict = {f'tsep_{tsep}': add_error_to_sample(ra_im[:, i, :], jk_bs) for i, tsep in enumerate(data_tsep_ls)}

    # * do the fit
    temp_re = []
    temp_im = []
    for n in tqdm( range(N_samp), desc=f"Loop in samples for ratio gs fit of P{px}, b={b}, z={z}" ):
        #! note different px may need different 2pt range
        pt2_fit_res = pt2_two_state_fit(pt2_ss_re_gv[n], tmin=3, tmax=13, Lt=Lt, label=f"P = {px}")
        
        ra_re_avg_dic = {f'tsep_{tsep}': ra_re_gv_dict[f'tsep_{tsep}'][n] for tsep in fit_tsep_ls}
        ra_im_avg_dic = {f'tsep_{tsep}': ra_im_gv_dict[f'tsep_{tsep}'][n] for tsep in fit_tsep_ls}
    
        ra_fit_res = ra_two_state_fit(ra_re_avg_dic, ra_im_avg_dic, fit_tsep_ls, tau_cut, Lt, id_label, pt2_fit_res=pt2_fit_res)

        temp_re.append(gv.mean(ra_fit_res.p['pdf_re']))
        temp_im.append(gv.mean(ra_fit_res.p['pdf_im']))
        
        # plot the fit on data of the first sample in log
        if n == 0:
            err_tsep_ls = data_tsep_ls
            fill_tsep_ls = fit_tsep_ls
            
            ra_re_avg = []
            ra_im_avg = []
            for tsep in data_tsep_ls:
                ra_re_avg.append(ra_re_gv_dict[f'tsep_{tsep}'][n])
                ra_im_avg.append(ra_im_gv_dict[f'tsep_{tsep}'][n])
                
            ra_re_avg = np.array(ra_re_avg)
            ra_im_avg = np.array(ra_im_avg)
            
            fig_re, fig_im, ax_re, ax_im = plot_ra_fit_on_data(ra_re_avg, ra_im_avg, ra_fit_res, err_tsep_ls, fill_tsep_ls, Lt, id_label, err_tau_cut=1, fill_tau_cut=1) 
            ax_re.set_title(r"$n^x=n^y=$"+f"{px}"+r", $b_\perp=$"+f"{b}"+r" $a$, $z=$"+f"{z}"+r" $a$", **fs_p) # * fix the title
            fig_re.savefig(f"../log/gsfit/ratio_fit_re_p{px}_b{b}_z{z}.pdf", transparent=True)
            fig_im.savefig(f"../log/gsfit/ratio_fit_im_p{px}_b{b}_z{z}.pdf", transparent=True)
            plt.close()
            plt.close()
            
            with open(f"../log/gsfit/ratio_fit_p{px}_b{b}_z{z}.txt", "w") as f:
                f.write(f"Ratio fit results for px={px}, b={b}, z={z}\n")
                f.write(ra_fit_res.format(100))
                f.close()
    
    bare_quasi_dic = {'re': temp_re, 'im': temp_im}
    
    return bare_quasi_dic


def wrapper_fh_fit(px, b, z, fit_tsep_ls, tau_cut):
    py = px
    pz = 0
    
    id_label = {"px": px, "py": py, "pz": pz, "b": b, "z": z}
    
    # * sum data
    fh_re, fh_im = get_fh_data(px, py, pz, b, z, fit_tsep_ls, jk_bs=jk_bs, tau_cut=tau_cut)
    
    # * add error to each conf
    fh_re_gv = add_error_to_sample(fh_re, jk_bs)
    fh_im_gv = add_error_to_sample(fh_im, jk_bs)
    
    # * do the fit
    temp_re = []
    temp_im = []
    for n in tqdm( range(N_samp), desc=f"Loop in samples for fh gs fit of P{px}, b={b}, z={z}" ):
        fh_fit_res = fh_one_state_fit(fh_re_gv[n], fh_im_gv[n], fit_tsep_ls, id_label)
        temp_re.append(gv.mean(fh_fit_res.p['pdf_re']))
        temp_im.append(gv.mean(fh_fit_res.p['pdf_im']))
        
        # plot the fit on data of the first sample in log
        if n == 0:
            # err_tsep_ls = data_tsep_ls
            err_tsep_ls = fit_tsep_ls
            fill_tsep_ls = fit_tsep_ls
            
            fig_re, fig_im, ax_re, ax_im = plot_fh_fit_on_data(fh_re_gv[n], fh_im_gv[n], fh_fit_res, err_tsep_ls, fill_tsep_ls, id_label)
            
            fig_re.savefig(f"../log/gsfit/fh_fit_re_p{px}_b{b}_z{z}.pdf", transparent=True)
            fig_im.savefig(f"../log/gsfit/fh_fit_im_p{px}_b{b}_z{z}.pdf", transparent=True)
            plt.close()
            plt.close()
            
            with open(f"../log/gsfit/fh_fit_p{px}_b{b}_z{z}.txt", "w") as f:
                f.write(f"FH fit results for px={px}, b={b}, z={z}\n")
                f.write(fh_fit_res.format(100))
                f.close()
    
    bare_quasi_dic = {'re': temp_re, 'im': temp_im}

    return bare_quasi_dic


# %%
#! Employ gs fit to get bare quasi
if __name__ == "__main__":
    p_ls = [3, 4, 5]
    b_range = np.arange(1, 4)
    z_range = np.arange(21)
    
    n_jobs = 8

    #! ratio 2st fit
    if True:  
        fit_tsep_ls = [6, 8, 10, 12]
        tau_cut = 3
        for px, b in itertools.product(p_ls, b_range):
            loop_params = [(px, b, z, fit_tsep_ls, tau_cut) for z in z_range]
            
            with mp.Pool(processes=n_jobs) as pool:
                results = list(tqdm(
                    pool.starmap(wrapper_ratio_fit, loop_params),
                    desc=f"Loop in px={px}, b={b}",
                    total=len(loop_params)
                ))
                
            bare_quasi_re = []
            bare_quasi_im = []
            for result in results:
                bare_quasi_re.append(result["re"])
                bare_quasi_im.append(result["im"])
            
            np.save(f'../output/dump/bare_quasi_zdep_p{px}_b{b}_ratio.{jk_bs}', {'re': bare_quasi_re, 'im': bare_quasi_im})
    
            
    #! fh fit
    if False:
        fit_tsep_ls = [8, 10, 12]
        tau_cut = 3
        
        for px, b in itertools.product(p_ls, b_range):
            loop_params = [(px, b, z, fit_tsep_ls, tau_cut) for z in z_range]
            
            with mp.Pool(processes=n_jobs) as pool:
                results = list(tqdm(
                    pool.starmap(wrapper_fh_fit, loop_params),
                    desc=f"Loop in px={px}, b={b}",
                    total=len(loop_params)
                ))
                
            bare_quasi_re = []
            bare_quasi_im = []
            for result in results:
                bare_quasi_re.append(result["re"])
                bare_quasi_im.append(result["im"])
            
            np.save(f'../output/dump/bare_quasi_zdep_p{px}_b{b}_fh.{jk_bs}', {'re': bare_quasi_re, 'im': bare_quasi_im})

    
    