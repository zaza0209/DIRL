#!/bin/bash
cd bash_files

# 1: init 2: N, 3: T, 4: setting, 5: nthread
write_slurm() {
    # Save the original IFS
    local old_ifs="$IFS"
    # Set IFS to underscore for concatenation
    IFS='_'
    
    # Construct the job name from the function arguments
    local job_name="online_$*"
    
    # Restore the original IFS
    IFS="$old_ifs"
    
    # Ensure the job name is not empty
    if [ -z "$job_name" ]; then
        echo "Error: Job name is empty"
        return
    fi
    
    # Construct the slurm script filename
    local slurm_file="job_${job_name}.slurm"
    
    # Check for any potential errors in filename
    if [ -z "$slurm_file" ]; then
        echo "Error: Slurm file name is empty"
        return
    fi

    echo "#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --partition=medium
#SBATCH --mem=3g
#SBATCH --cpus-per-task=${nthread}
#SBATCH --array=0-19
#SBATCH -o ./reports/%x_%A_%a.out 

cd ..
python3 run_value.py \$SLURM_ARRAY_TASK_ID $@
" > "$slurm_file"

    # Submit the job if running is enabled
    if $run; then
       sbatch "$slurm_file"
    fi
    
}

Ns=(50)
T_new=250 
cov=0.25
nthread=8
cp_detect_interval=25
is_tune_parallel=0
is_cp_parallel=0
run=true
max_iter=2
for N in "${Ns[@]}"; do
    for type in  "proposed"; do #"only_clusters" "overall" "oracle"   "only_cp" ; do  
        for setting in "pwconst2"; do #"smooth" 
             write_slurm ${type} ${N} ${T_new} ${setting} ${nthread} ${cov} ${cp_detect_interval} ${is_tune_parallel} ${is_cp_parallel} $max_iter 
         done
    done 
done
	

