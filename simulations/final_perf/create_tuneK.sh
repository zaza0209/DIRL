#!/bin/bash
cd bash_files

# 1: init 2: N, 3: T, 4: setting, 5: nthread
write_slurm() {
    # Save the original IFS
    local old_ifs="$IFS"
    # Set IFS to underscore for concatenation
    IFS='_'
    
    # Construct the job name from the function arguments
    local job_name="tuneK_$*"
    
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
#SBATCH --array=0
#SBATCH -o ./reports/%x_%A_%a.out 

cd ..
python3 tune_K.py \$SLURM_ARRAY_TASK_ID $@
" > "$slurm_file"

    # Submit the job if running is enabled
    if $run; then
       sbatch "$slurm_file"
    fi
    
}

Ns=(50)
nthread=8
T=50
run=true
effect_size="strong"
K_list="1 2 3 4"
max_iter=2
for N in "${Ns[@]}"; do
    for setting in "pwconst2" ; do #"smooth"    
        for init in "tuneK_iter" ; do # "kmeans" "true_clustering" "no_clusters" "random_cp""true_change_points" "no_change_points" "random_clustering" ; do # #
            if [[ "${init}" == "kmeans" ]]; then
                for K in 1 2 3 4;  do 
                    write_slurm ${init} ${N} ${T} ${setting} ${nthread} ${effect_size} $max_iter ${K} 
                done
            else              
                K_list_input=($K_list)
                write_slurm ${init} ${N} ${T} ${setting} ${nthread} ${effect_size} $max_iter $"${K_list_input[@]}"
            fi
        done
    
    done
done

	

