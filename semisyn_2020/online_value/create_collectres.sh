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
python collect_results.py $@
" > "$slurm_file"

    # Submit the job if running is enabled
    if $run; then
       sbatch "$slurm_file"
    fi
    
}

run=true

max_iter=2

fixed_kappamin=0.1

cp_detect_interval=25
true_cp_interval=60
T_new=401

cov=0.25
K_true=3

is_r_orignial_scale=1

effect_size_factor="0.6 1 5"
effect_size_factor_input=($effect_size_factor)
for N_per_cluster in 20; do
    for stationary_team_dynamic in 1; do
        for is_transform in 1; do
            for stationary_changeactionsign in 1; do
                for env_type in 'randomstationary' ; do #'randomstationary3_changesign' 'randomstationary3_changeactionsign'
                    for reward_type in  "next_state"; do #"current_state"
                        write_slurm $max_iter $N_per_cluster $fixed_kappamin $cp_detect_interval $true_cp_interval $T_new $env_type $reward_type $is_transform $K_true $stationary_team_dynamic $stationary_changeactionsign $is_r_orignial_scale $"${effect_size_factor_input[@]}"
                    done
                done
            done
        done
    done
done
