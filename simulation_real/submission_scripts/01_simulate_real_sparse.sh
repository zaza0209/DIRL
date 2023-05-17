#!/bin/bash
cd /home/mengbing/research/HeterRL/simulation_real/submission_scripts
effect_sizes=("weak" "moderate" "strong") 
# effect_sizes=("weak" "strong") 
Ks=(2 3 4)
inits=(15 20)
# threshold_types=("Chi2" "permutation")

# ${!array[@]} is the list of all the indexes set in the array
time=04:00:00
memory=6
for effect_size in ${effect_sizes[@]}; do
	for K in ${Ks[@]}; do
		for init in ${inits[@]}; do
    # effect_size=${effect_sizes[$i]}
    # for threshold_type in "${threshold_types[@]}"; do

	echo "#!/bin/bash
##SBATCH --account=zhenkewu0
#SBATCH --partition=standard
#SBATCH --job-name=${effect_size}_K${K}_init${init}
#SBATCH --time=${time}
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --mem=${memory}g
#SBATCH --cpus-per-task=1
#SBATCH --array=11-20
#SBATCH -o ./reports/%x_%A_%a.out 
##SBATCH --constraint=E5-2650v4

#module load python
cd /home/mengbing/research/HeterRL/simulation_real/
python3 01_simulate_real_sparse_run.py \$SLURM_ARRAY_TASK_ID ${effect_size} ${K} ${init}" > 01_simulate_real_${effect_size}_K${K}_init${init}_run.slurm
	 		sbatch 01_simulate_real_${effect_size}_K${K}_init${init}_run.slurm
		done
	done
done
