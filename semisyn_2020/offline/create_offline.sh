#!/bin/bash
cd bash_files
write_slurm() {
    # Save the original IFS
    local old_ifs="$IFS"
    # Set IFS to underscore for concatenation
    IFS='_'
    
    # Construct the job name from the function arguments
    local job_name="offline_$*"
    
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
#SBATCH --partition=xlargecpu
#SBATCH --mem=5g
#SBATCH --cpus-per-task=${nthread}
#SBATCH --array=20
#SBATCH -o ./reports/%x_%A_%a.out 

cd ..
python3 offline.py \$SLURM_ARRAY_TASK_ID $@
" > "$slurm_file"

    # Submit the job if running is enabled
    if $run; then
       sbatch "$slurm_file"
    fi
    
}

nthread=8
T=101
run=true
K_list="1 2 3 4"
max_iter=2
refit=1
K_true=3
kappa_interval_step=50
cov=0.25
env_type='randomstationary' #"original" #"split_firstcluster"
stationary_changeactionsign=0

for stationary_team_dynamic in -1; do
    for effect_size_factor in 1 2; do
        for N_per_cluster in 20; do
            for transform_s in 1; do
                for trans_setting in "pwconst2" ; do #"smooth"    
                    for init in "tuneK_iter" ; do #"only_cp" "only_clusters"  "kmeans" "true_clustering" "no_clusters" "random_cp""true_change_points" "no_change_points" "random_clustering" ; do # #
                        if [[ "${init}" == "kmeans" ]]; then
                            for K in 1 2 3 4;  do 
                                write_slurm $effect_size_factor $init $refit $max_iter $nthread $T $N_per_cluster $trans_setting $K_true $stationary_team_dynamic $env_type $cov $transform_s $stationary_changeactionsign ${K} 
                            done
                        else              
                            K_list_input=($K_list)
                            write_slurm $effect_size_factor $init $refit $max_iter $nthread $T $N_per_cluster $trans_setting $K_true $stationary_team_dynamic $kappa_interval_step $env_type $cov $transform_s $stationary_changeactionsign $"${K_list_input[@]}"
                        fi
                    done
                done
            done
        done
    done
done

	

