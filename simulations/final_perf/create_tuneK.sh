#!/bin/bash
cd /home/huly0209_gmail_com/heterRL/toyexample/final_2

Ns=(50)
nthread=8
T=50
run=true
threshold_type="maxcusum"
# 1: init 2: N, 3: T, 4: setting, 5: nthread
write_slurm() {
    echo "#!/bin/bash
#SBATCH --job-name=$1_$4
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --mem=3g
#SBATCH --cpus-per-task=$5
#SBATCH --array=0-19
#SBATCH -o ./reports/%x_%A_%a.out 

cd /home/huly0209_gmail_com/heterRL/toyexample/final_2

python3 tune_K.py \$SLURM_ARRAY_TASK_ID $1 $2 $3 $4 $5
" > maxiter_init$1_N$2_T$3_set$4.slurm

if $run
then
    sbatch axiter_init$1_N$2_T$3_set$4.slurm
fi
}

for N in "${Ns[@]}"; do
    for setting in "pwconst2" "smooth"; do #    
        for init in "tuneK_iter"; do # "kmeans" "true_clustering" "no_clusters" "random_cp""true_change_points" "no_change_points" "random_clustering" ; do # #
                if [[ "${init}" == "kmeans" ]]; then
                    for K in "2" "1" "3" '4';  do 
                        write_slurm ${init} ${N} ${T} ${setting} ${nthread} ${cov} ${threshold_type} ${K} ${max_iter} 0 
                    done
                else
                    write_slurm ${init} ${N} ${T} ${setting} ${nthread} ${cov} ${threshold_type} 2 ${max_iter} 0 
                fi
            done
    
    done
done

	

