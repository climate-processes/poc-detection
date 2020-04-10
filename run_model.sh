
# Sometimes this works and sometimes it doesn't... It may be to do if it's busy or not
bsub -q lotus_gpu -Is /bin/bash
# Load the env
conda activate keras
# Run on the two GPUs
./poc_model.py /gws/nopw/j04/eo_shared_data_vol2/scratch/POC_analysis/peru/np_cache_peruvian_sc_downsampled_.npy /home/users/dwatsonparris/poc-detection/ -n 2 -v -o peruvian_poc_mask.npz
