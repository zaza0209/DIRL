#!/bin/bash
cd /home/mengbing/research/HeterRL/simulation_real/submission_scripts
effect_sizes=("weak" "moderate" "strong") 
# threshold_types=("Chi2" "permutation")

# ${!array[@]} is the list of all the indexes set in the array
time=01:00:00
memory=6
for i in ${!effect_sizes[@]}; do
    effect_size=${effect_sizes[$i]}
    # for threshold_type in "${threshold_types[@]}"; do

	echo "#!/bin/bash
##SBATCH --account=zhenkewu0
#SBATCH --partition=standard
#SBATCH --job-name=real_${effect_size}
#SBATCH --time=${time}
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --mem=${memory}g
#SBATCH --cpus-per-task=1
#SBATCH --array=1-30
#SBATCH -o ./reports/%x_%A_%a.out 
##SBATCH --constraint=E5-2650v4

#module load python
cd /home/mengbing/research/HeterRL/simulation_real/
python3 01_simulate_real_run.py \$SLURM_ARRAY_TASK_ID ${effect_size} " > 01_simulate_real_${effect_size}_run.slurm
	 sbatch 01_simulate_real_${effect_size}_run.slurm
	# done
done
