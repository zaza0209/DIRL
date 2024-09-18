#!/bin/bash
cd bash_files

# 1: init 2: N, 3: T, 4: setting, 5: nthread
write_slurm() {
    # Save the original IFS
    local old_ifs="$IFS"
    # Set IFS to underscore for concatenation
    IFS='_'
    
    # Construct the job name from the function arguments
    local job_name="on_$*"
    
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
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --partition=xlargecpu
#SBATCH --mem=5g #25 if cpu=32
#SBATCH --cpus-per-task=${nthread}
#SBATCH --array=0
#SBATCH -o ./reports/%x_%A_%a.out 

cd ..
python3 run_value.py \$SLURM_ARRAY_TASK_ID $@
" > "$slurm_file"

    # Submit the job if running is enabled
    if $run; then
       sbatch "$slurm_file"
    fi
    
}


cov=0.25
nthread=8

is_tune_parallel=0
is_cp_parallel=0
run=true
max_iter=2 

K_true=3
T_initial=101
fixed_kappamin=0.1

cp_detect_interval=25
true_cp_interval=60
T_new=401

effect_size_factor=1
refit=0
stationary_changeactionsign=0
is_transform=1

is_r_orignial_scale=1

for N_per_cluster in 20; do
    for stationary_team_dynamic in 1; do
        for effect_size_factor in 1; do #0.6 1 5 0.6
            for type_est in "oracle" ; do  # "overall" "only_clusters" "proposed" "only_cp" 
               	for env_type in 'randomstationary'; do # 'randomstationary3_changeactionsign''randomstationary3_changesign'  
                   	for reward_type in "next_state"; do #"current_state"
                           for trans_setting in "pwconst2"; do #"smooth" 
                                write_slurm $type_est $N_per_cluster $T_new $trans_setting $nthread $cov $cp_detect_interval $is_tune_parallel $is_cp_parallel $max_iter $effect_size_factor $env_type $K_true $stationary_team_dynamic $T_initial $fixed_kappamin $true_cp_interval $reward_type $is_transform $refit $stationary_changeactionsign $is_r_orignial_scale
                            done
                       done
                done
            done
        done
    done 
done
	

