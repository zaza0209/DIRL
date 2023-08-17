#!/bin/bash
cd /home/xxx/research/HeterRL/simulation_real/submission_scripts

effect_sizes=("weak" "moderate" "strong") 
# type_ests=("proposed" "oracle" "cluster_only" "changepoint_only") 
type_ests=("overall") 

# ${!array[@]} is the list of all the indexes set in the array

for j in ${!effect_sizes[@]}; do
    effect_size=${effect_sizes[$j]}
    for type_est in "${type_ests[@]}"; do
    	time=01:00:00
    	memory=6
		if [[ ${type_est} == "proposed" ]]
		then
		  time=10:00:00
		  memory=6
		# elif [[ ${N} -ge 50 ]]
		# then
		#   time=01:00:00
		#   memory=6
		fi

	echo "#!/bin/bash
##SBATCH --account=zhenkewu0
#SBATCH --partition=standard
#SBATCH --job-name=eval_${effect_size}_${type_est}
#SBATCH --time=${time}
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --mem=${memory}g
#SBATCH --cpus-per-task=1
#SBATCH --array=1-20
#SBATCH -o ./reports/%x_%A_%a.out 
##SBATCH --constraint=E5-2650v4

module load python
cd /home/xxx/research/HeterRL/simulation_real/
python3 03_evaluate_real.py \$SLURM_ARRAY_TASK_ID ${effect_size} ${type_est}" > 03_evaluate_${effect_size}_${type_est}_run.slurm
	 sbatch 03_evaluate_${effect_size}_${type_est}_run.slurm

	done
done
