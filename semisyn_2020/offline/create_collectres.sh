#!/bin/bash
cd bash_files
write_slurm(){
    # Save the original IFS
    local old_ifs="$IFS"
    # Set IFS to underscore for concatenation
    IFS='_'
    
    # Construct the job name from the function arguments
    local job_name="collectres_$*"
    
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
#SBATCH --partition=small
#SBATCH --mem=1g
#SBATCH --cpus-per-task=1
#SBATCH --array=0
#SBATCH -o ./reports/%x_%A_%a.out 

cd ..
python collect_res.py $@
" > "$slurm_file"

    # Submit the job if running is enabled
    if $run; then
       sbatch "$slurm_file"
    fi
    
}

run=true

T=101
effect_size_factor="1 2"
max_iter=2
K_true=3
kappa_interval_step=50
env_type='randomstationary' #"original" #"split_firstcluster"
cov=0.25
stationary_changeactionsign=0
for stationary_team_dynamic in -1; do
    for transform_s in 1; do
        for N_per_cluster in 20; do
            effect_size_factor_input=($effect_size_factor)
            write_slurm $max_iter $T $N_per_cluster $K_true $stationary_team_dynamic $effect_size_factor_list $kappa_interval_step $env_type $cov $transform_s $stationary_changeactionsign $"${effect_size_factor_input[@]}"
        done
    done
done
