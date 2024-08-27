# %%
import os
import glob
import numpy as np
import gvar as gv
from joblib import Parallel, delayed
from read_data import get_2pt_data, get_ratio_data, get_sum_data, get_fh_data
from lametlat.gsfit.pt2_fit import pt2_two_state_fit
from lametlat.gsfit.ratio_fit import ra_two_state_fit
from lametlat.gsfit.sum_fit import sum_one_state_fit, sum_two_state_fit
from lametlat.gsfit.fh_fit import fh_one_state_fit
from lametlat.utils.funcs import add_error_to_sample
from lametlat.utils.log import set_up_log
from tqdm.auto import tqdm
from multiprocessing import current_process


a = 0.06
Ls = 48
Lt = 64
data_tsep_ls = [4, 6, 8, 10, 12]
N_samp = 200


[os.remove(f) for f in [*glob.glob('../log/gsfit/main_bs_process*.log'), '../log/gsfit/main_bs.log'] if os.path.exists(f)]
set_up_log("../log/gsfit/main_bs.log")


def wrapper_ratio_fit(px, b, z, fit_tsep_ls, tau_cut, n_jobs):
    if n_jobs > 1:
        process_id = current_process()._identity[0] - 1 # Process IDs start from 1
        set_up_log(f"../log/gsfit/main_bs_process{process_id}.log")
    
    py = px
    pz = 0
    
    id_label = {"px": px, "py": py, "pz": pz, "b": b, "z": z}

    # * 2pt data
    pt2_ss_re, _ = get_2pt_data('SS', px, py, pz, jk_bs="bs")
    pt2_ss_re_gv = add_error_to_sample(pt2_ss_re)

    # * ratio data
    ra_re, ra_im = get_ratio_data(px=px, py=py, pz=pz, b=b, z=z, tsep_ls=data_tsep_ls, jk_bs="bs") # shape = (N_samp, len(tsep_ls), 16)
    
    # Add error to each tsep separately
    ra_re_gv_dict = {f'tsep_{tsep}': add_error_to_sample(ra_re[:, i, :]) for i, tsep in enumerate(data_tsep_ls)}
    ra_im_gv_dict = {f'tsep_{tsep}': add_error_to_sample(ra_im[:, i, :]) for i, tsep in enumerate(data_tsep_ls)}

    
    # * do the fit
    temp_re = []
    temp_im = []
    for n in tqdm( range(N_samp), desc=f"Loop in samples for gs fit of P{px}, b={b}, z={z}" ):
        #! note different px may need different 2pt range
        pt2_fit_res = pt2_two_state_fit(pt2_ss_re_gv[n], tmin=3, tmax=11, Lt=Lt, label=f"P = {px}")
        
        ra_re_avg_dic = {f'tsep_{tsep}': ra_re_gv_dict[f'tsep_{tsep}'][n] for tsep in fit_tsep_ls}
        ra_im_avg_dic = {f'tsep_{tsep}': ra_im_gv_dict[f'tsep_{tsep}'][n] for tsep in fit_tsep_ls}
    
        ra_fit_res = ra_two_state_fit(ra_re_avg_dic, ra_im_avg_dic, fit_tsep_ls, tau_cut, Lt, id_label, pt2_fit_res=pt2_fit_res)

        temp_re.append(gv.mean(ra_fit_res.p['pdf_re']))
        temp_im.append(gv.mean(ra_fit_res.p['pdf_im']))
    
    bare_quasi_dic = {'re': temp_re, 'im': temp_im}
    
    return bare_quasi_dic

def wrapper_sum_fit(px, b, z, fit_tsep_ls, tau_cut, n_jobs):
    # Get the current process ID
    if n_jobs > 1:
        process_id = current_process()._identity[0] - 1 # Process IDs start from 1
        set_up_log(f"../log/gsfit/main_bs_process{process_id}.log")
    
    py = px
    pz = 0
    
    id_label = {"px": px, "py": py, "pz": pz, "b": b, "z": z}
    
    # * sum data
    sum_re, sum_im = get_sum_data(px, py, pz, b, z, fit_tsep_ls, jk_bs="bs", tau_cut=tau_cut)
    
    # * add error to each conf
    sum_re_gv = add_error_to_sample(sum_re)
    sum_im_gv = add_error_to_sample(sum_im)
    
    #TODO: Reduce the error for tsep by half
    # selected_tsep = 12
    # if selected_tsep in fit_tsep_ls:
    #     tsep_index = fit_tsep_ls.index(selected_tsep)
    #     sum_re_gv[:, tsep_index] = gv.gvar(gv.mean(sum_re_gv[:, tsep_index]), gv.sdev(sum_re_gv[:, tsep_index]) / np.sqrt(2))
    #     sum_im_gv[:, tsep_index] = gv.gvar(gv.mean(sum_im_gv[:, tsep_index]), gv.sdev(sum_im_gv[:, tsep_index]) / np.sqrt(2))
    
    # * do the fit
    temp_re = []
    temp_im = []
    for n in tqdm( range(N_samp), desc=f"Loop in samples for gs fit of P{px}, b={b}, z={z}" ):
        sum_fit_res = sum_one_state_fit(sum_re_gv[n], sum_im_gv[n], fit_tsep_ls, tau_cut, id_label)
        temp_re.append(gv.mean(sum_fit_res.p['pdf_re']))
        temp_im.append(gv.mean(sum_fit_res.p['pdf_im']))
    
    bare_quasi_dic = {'re': temp_re, 'im': temp_im}

    return bare_quasi_dic
    
def wrapper_sum_two_state_fit(px, b, z, fit_tsep_ls, tau_cut, n_jobs):
    # Get the current process ID
    if n_jobs > 1:
        process_id = current_process()._identity[0] - 1 # Process IDs start from 1
        set_up_log(f"../log/gsfit/main_bs_process{process_id}.log")
    
    py = px
    pz = 0
    
    id_label = {"px": px, "py": py, "pz": pz, "b": b, "z": z}
    
    # * 2pt data
    pt2_ss_re, _ = get_2pt_data('SS', px, py, pz, jk_bs="bs")
    pt2_ss_re_gv = add_error_to_sample(pt2_ss_re)

    # * sum data
    sum_re, sum_im = get_sum_data(px, py, pz, b, z, fit_tsep_ls, jk_bs="bs", tau_cut=tau_cut)
    
    # Add error to each tsep separately
    sum_re_gv = add_error_to_sample(sum_re)
    sum_im_gv = add_error_to_sample(sum_im)
    
    #TODO: Reduce the error for tsep by half
    # selected_tsep = 10
    # if selected_tsep in fit_tsep_ls:
    #     tsep_index = fit_tsep_ls.index(selected_tsep)
    #     sum_re_gv[:, tsep_index] = gv.gvar(gv.mean(sum_re_gv[:, tsep_index]), gv.sdev(sum_re_gv[:, tsep_index]) / np.sqrt(2))
    #     sum_im_gv[:, tsep_index] = gv.gvar(gv.mean(sum_im_gv[:, tsep_index]), gv.sdev(sum_im_gv[:, tsep_index]) / np.sqrt(2))
    
     # * do the fit
    temp_re = []
    temp_im = []
    for n in tqdm( range(N_samp), desc=f"Loop in samples for gs fit of P{px}, b={b}, z={z}" ):
        #! note different px may need different 2pt range
        pt2_fit_res = pt2_two_state_fit(pt2_ss_re_gv[n], tmin=3, tmax=11, Lt=Lt, label=f"P = {px}")
        
        sum_fit_res = sum_two_state_fit(sum_re_gv[n], sum_im_gv[n], fit_tsep_ls, tau_cut, id_label, pt2_fit_res=pt2_fit_res)
        temp_re.append(gv.mean(sum_fit_res.p['pdf_re']))
        temp_im.append(gv.mean(sum_fit_res.p['pdf_im']))
    
    bare_quasi_dic = {'re': temp_re, 'im': temp_im}

    return bare_quasi_dic
    
def wrapper_fh_fit(px, b, z, fit_tsep_ls, tau_cut, n_jobs):
    # Get the current process ID
    if n_jobs > 1:
        process_id = current_process()._identity[0] - 1 # Process IDs start from 1
        set_up_log(f"../log/gsfit/main_bs_process{process_id}.log")
    
    py = px
    pz = 0
    
    id_label = {"px": px, "py": py, "pz": pz, "b": b, "z": z}
    
    # * sum data
    fh_re, fh_im = get_fh_data(px, py, pz, b, z, fit_tsep_ls, jk_bs="bs", tau_cut=tau_cut)
    
    # * add error to each conf
    fh_re_gv = add_error_to_sample(fh_re)
    fh_im_gv = add_error_to_sample(fh_im)
    
    # * do the fit
    temp_re = []
    temp_im = []
    for n in tqdm( range(N_samp), desc=f"Loop in samples for gs fit of P{px}, b={b}, z={z}" ):
        fh_fit_res = fh_one_state_fit(fh_re_gv[n], fh_im_gv[n], fit_tsep_ls, id_label)
        temp_re.append(gv.mean(fh_fit_res.p['pdf_re']))
        temp_im.append(gv.mean(fh_fit_res.p['pdf_im']))
    
    bare_quasi_dic = {'re': temp_re, 'im': temp_im}

    return bare_quasi_dic
    
    

# %%
if __name__ == "__main__":
    b = 0
    zmax = 21
    zs = 3
    n_jobs = 8
    p_ls = [4, 5]
    fit_method = "sum_81012"  # "sum_81012" or "sum_two_state_681012"
    
    # * Zero momentum
    if False:
        loop_params = [(b, z) for z in range(zmax)]
        
        results = Parallel(n_jobs=n_jobs)(delayed(wrapper_sum_fit)(px=0, b=b, z=z, fit_tsep_ls=[8, 10, 12], tau_cut=2, n_jobs=n_jobs) for b, z in tqdm(loop_params, desc="Loop in b, z"))
        
        bare_p0 = {"re": [], "im": []}
        for (b, z), (result) in zip(loop_params, results):
            bare_p0["re"].append(result["re"])
            bare_p0["im"].append(result["im"])
            
        # ! normalization is f(P=0, b, z) / f(P=0, b, z=0)
        bare_p0["re"] = np.array(bare_p0["re"]) / np.array(bare_p0["re"])[0]
        bare_p0["im"] = np.array(bare_p0["im"]) / np.array(bare_p0["re"])[0]

        gv.dump(bare_p0, f"../output/dump/bare_quasi_p0_b{b}_zdep.pkl")
    else:
        bare_p0 = gv.load(f"../output/dump/bare_quasi_p0_b{b}_zdep.pkl")
        
    
    # * Non-zero momentum
    if True:
        loop_params = [(px, b, z) for px in p_ls for z in range(zmax)]
        
        if fit_method == "sum_81012":
            results = Parallel(n_jobs=n_jobs)(delayed(wrapper_sum_fit)(px, b, z, fit_tsep_ls=[8, 10, 12], tau_cut=2, n_jobs=n_jobs) for px, b, z in tqdm(loop_params, desc="Loop in px, b, z"))
        elif fit_method == "sum_two_state_681012":
            results = Parallel(n_jobs=n_jobs)(delayed(wrapper_sum_two_state_fit)(px, b, z, fit_tsep_ls=[6, 8, 10, 12], tau_cut=2, n_jobs=n_jobs) for px, b, z in tqdm(loop_params, desc="Loop in px, b, z"))
        else:
            raise ValueError(f"Unknown fit method: {fit_method}")
        
        bare_quasi = {f"{part}_p{px}": [] for px in p_ls for part in ["re", "im"]}
        bare_quasi_norm = {f"{part}_p{px}": [] for px in p_ls for part in ["re", "im"]}
        for (px, b, z), (result) in zip(loop_params, results):
            bare_quasi[f"re_p{px}"].append(result["re"])
            bare_quasi[f"im_p{px}"].append(result["im"])
            
        # ! normalization is f(P, b, z) / f(P, b, z=0)
        for px in p_ls:
            bare_quasi_norm[f"re_p{px}"] = np.array(bare_quasi[f"re_p{px}"]) / np.array(bare_quasi[f"re_p{px}"])[0]
            bare_quasi_norm[f"im_p{px}"] = np.array(bare_quasi[f"im_p{px}"]) / np.array(bare_quasi[f"re_p{px}"])[0]

        gv.dump(bare_quasi_norm, f"../output/dump/bare_quasi_p45_b{b}_zdep.pkl")
    else:
        bare_quasi_norm = gv.load(f"../output/dump/bare_quasi_p45_b{b}_zdep.pkl")
        
    # * Renormalization
    # ! Ratio scheme: when z < zs, take f(P, b, z) / f(P=0, b, z), when z >= zs, take f(P, b, z) / f(P=0, b, z=zs)
    
    denominator = np.array(bare_p0["re"]) # shape = (zmax, N_samp)
    denominator[zs:, :] = denominator[zs:zs+1, :]
    
    renorm_quasi_re = {
        f"re_p{px}": np.array(bare_quasi_norm[f"re_p{px}"]) / denominator
        for px in p_ls
    }
    renorm_quasi_im = {
        f"im_p{px}": np.array(bare_quasi_norm[f"im_p{px}"]) / denominator
        for px in p_ls
    }
    renorm_quasi = {
        f"{part}_p{px}": np.array(bare_quasi_norm[f"{part}_p{px}"]) / denominator
        for px in p_ls
        for part in ["re", "im"]
    }
    
    gv.dump(renorm_quasi, f"../output/dump/renorm_quasi_p45_b{b}_zdep_{fit_method}.pkl")
    
    

# %%
