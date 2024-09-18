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
python collect_res.py \$SLURM_ARRAY_TASK_ID $@
" > "$slurm_file"

    # Submit the job if running is enabled
    if $run; then
       sbatch "$slurm_file"
    fi
    
}

run=true

state_transform=1
early_stopping=1

threshold_type="maxcusum"
ic_T_dynamic=0
is_cv=1

max_iter=10
is_r_orignial_scale=1

for test_cluster_type in "proposed_g_index" ; do #"test_set_only""g_index"  
    for C in 1; do
        for max_iter in 2; do
            # Initialize indices as an empty string
            T_train_list=""
            T_train_len=0
            # Generate indices based on the value of index_range
            for T_train in {62..82..1}; do
                T_train_list+="$T_train "
                ((T_train_len++))
            done
            
            # Trim the trailing space
            T_train_list=${T_train_list%}
            write_slurm $state_transform $early_stopping $max_iter $threshold_type $ic_T_dynamic $test_cluster_type $is_cv $C $T_train_len $is_r_orignial_scale $T_train_list
    	done
	done
done
