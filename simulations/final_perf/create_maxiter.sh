#!/bin/bash
cd /home/xx/heterRL/toyexample/final_2

Ns=(50)
nthread=8
T=50
cov=0.25
max_iter=10
run=true
threshold_type="maxcusum"
# 1: init 2: N, 3: T, 4: setting, 5: nthread, 6:cov, 7: threshold_type, 8: K, 9: max_iter, 10:random_cp_index
write_slurm() {
    echo "#!/bin/bash
#SBATCH --job-name=$1_K${8}_$6_$4
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --mem=3g
#SBATCH --cpus-per-task=$5
#SBATCH --array=0-19
#SBATCH -o ./reports/%x_%A_%a.out 

cd /home/xx/heterRL/toyexample/final_2

python3 run_maxiter.py \$SLURM_ARRAY_TASK_ID $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10}
" > maxiter_init$1_N$2_T$3_set$4_cov$6_cp$7_K${8}_cp${10}_run.slurm

if $run
then
    sbatch maxiter_init$1_N$2_T$3_set$4_cov$6_cp$7_K${8}_cp${10}_run.slurm 
fi
}

for N in "${Ns[@]}"; do
    for setting in "pwconst2" "smooth"; do #    
        for init in "tuneK_iter" "no_change_points" "random_cp"; do # "kmeans" "true_clustering" "no_clusters" "true_change_points"  "random_clustering" ; do # #
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

	

