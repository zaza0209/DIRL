#!/bin/bash
cd bash_files
write_slurm(){
    # Save the original IFS
    local old_ifs="$IFS"
    # Set IFS to underscore for concatenation
    IFS='_'
    
    # Construct the job name from the function arguments
    local job_name="realdata_$*"
    
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
#SBATCH --partition=$partition
#SBATCH --mem=${mem}g
#SBATCH --cpus-per-task=$nthread
#SBATCH --array=1
#SBATCH -o ./reports/%x_%A_%a.out 

cd ..
python realdata.py \$SLURM_ARRAY_TASK_ID $@
" > "$slurm_file"

    # Submit the job if running is enabled
    if $run; then
       sbatch "$slurm_file"
    fi
    
}

run=true

state_transform=1
refit=0
early_stopping=1

threshold_type="maxcusum"
training_phrase="evaluation"
ic_T_dynamic=0
is_cv=1
is_r_orignial_scale=1

for method in   "proposed" "only_cp"  "only_clusters" "overall"; do #       
    case "$method" in
        "proposed"|"only_cp")
            nthread=5
            partition="medium" #"xlargecpu"
            mem=5
            ;;
        *)
            nthread=5
            partition="medium" #"xlargecpu""small" #
            mem=5
            ;;
    esac
    for max_iter in 2; do
        for test_cluster_type in "proposed_g_index" ; do #"test_set_only""g_index"  
            for C in 1; do
                for T_train in {82..82..1}; do
                    write_slurm $state_transform $method $nthread $refit $early_stopping $max_iter $threshold_type $training_phrase $T_train $ic_T_dynamic $test_cluster_type $is_cv $C $is_r_orignial_scale
                done
        	done
        done
    done
done
