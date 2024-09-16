# lametlat_analysis_template
This is a template repo for lattice analysis, especially under LaMET, use the Python package lametlat as the main tool.

## Folder structure
```
data folder: [keep at local only] raw data files, such as the data from lattice QCD simulations, usually in .h5 format

cache folder: cache files, such as the data after pre-processing

output folder: [keep at local only] output files, including the plots and dump files
    dump folder
    plots folder
    else folder

log folder: log files and the plots, such as ground state fit result and the plots

scripts folder: scripts for specific project, not for general template

debug folder: debug and cross-check files

test_data folder: test data for the liblattice library.
```
