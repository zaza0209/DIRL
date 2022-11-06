#!/bin/bash
cd /home/huly0209_gmail_com/heterRL/toyexample/final_2

Ns=(50)
nthread=8
T=50
cov=0.25
max_iter=10
run=true
threshold_type="maxcusum"
# 1: init 2: N, 3: T, 4: setting, 5: nthread, 6:cov, 7: threshold_type, 8: K, 9: max_iter, 10:random_cp_index, 11: effect_size
write_slurm() {
    echo "#!/bin/bash
#SBATCH --job-name=$1_K${8}_$6_$4_${11}
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --mem=3g
#SBATCH --cpus-per-task=$5
#SBATCH --array=0-19
#SBATCH -o ./reports/%x_%A_%a.out 

cd /home/huly0209_gmail_com/heterRL/toyexample/final_2

python3 run_maxiter_samesign.py \$SLURM_ARRAY_TASK_ID $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11}
" > maxiter_init$1_N$2_T$3_set$4_cov$6_cp$7_K${8}_cp${10}_${11}_run.slurm

if $run
then
    sbatch maxiter_init$1_N$2_T$3_set$4_cov$6_cp$7_K${8}_cp${10}_${11}_run.slurm 
fi
}

random_cp_list=($(seq 1 1 5))

for N in "${Ns[@]}"; do
    for setting in "smooth"  ; do #  "pwconst2" 
        for init in "kmeans"; do #  "tuneK_iter""true_clustering" "no_clusters" "random_cp""true_change_points" "no_change_points" "random_clustering" ; do # #
            for effect_size in "strong"; do #"0.4""moderate" "weak"
                if [[ "${init}" == "kmeans" ]]; then
                    for K in "2" "3" "1" '4';  do  #  
                        write_slurm ${init} ${N} ${T} ${setting} ${nthread} ${cov} ${threshold_type} ${K} ${max_iter} 0 ${effect_size}
                    done
                else
                    write_slurm ${init} ${N} ${T} ${setting} ${nthread} ${cov} ${threshold_type} 2 ${max_iter} 0 ${effect_size}
                fi
            done
        done
    done
done

	

