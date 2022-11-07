#!/bin/bash
cd /home/mengbing/research/HeterRL/simulation_real/submission_scripts

gammas=(0.9)
gamma_names=(09)
type_ests=("overall" "proposed" "truecluster" "estimatedcluster" "oracle") 

# ${!array[@]} is the list of all the indexes set in the array
for i in ${!gammas[@]}; do
    gamma=${gammas[$i]}
    for type_est in "${type_ests[@]}"; do

#		for N in "${Ns[@]}"; do
#				if [[ ${N} -le 25 ]]
#				then
#				  time=02:00:00
#				  memory=6
#				elif [[ ${N} -ge 50 ]]
#				then
				  time=01:00:00
				  memory=6
#				fi

	echo "#!/bin/bash
##SBATCH --account=zhenkewu0
#SBATCH --partition=standard
#SBATCH --job-name=real_${type_est}
#SBATCH --time=${time}
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --mem=${memory}g
#SBATCH --cpus-per-task=1
#SBATCH --array=1-30
#SBATCH -o ./reports/%x_%A_%a.out 
##SBATCH --constraint=E5-2650v4

#module load python
cd /home/mengbing/research/HeterRL/simulation_real/
python3 02_evaluate_real.py \$SLURM_ARRAY_TASK_ID ${type_est}" > 02_evaluate_${type_est}_run.slurm
	 sbatch 02_evaluate_${type_est}_run.slurm


#		done
	done
done
