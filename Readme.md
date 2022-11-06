# Doubly Non-homogeneous Reinforcement Learning

This repository contains the implementation for the paper "Doubly Non-homogeneous Reinforcement Learning" in Python (and R for plotting).

## File Overview

- Folder `functions/`: This folder contains all utility Python functions used in numerical experiments including simulation and data analysis
    - `simu_mean_detect` implements the proposed change point and cluster detection method for Non-homogeneous environment.
    - `compute_test_statistics_separateA.py` computes the optimal policy.
    - `evaluation.py` implements the evaluation procedure. Specifically, it contains functions for estimating the optimal policy and estimating the value of the policy using fitted-Q evaluation.
    - `simulate_data_1d.py` generates data in 1-dimensional simulation. It contains functions to simulate data in s scenarios of different transition.
    - `simulate_data_1d_flexible.py` generated data in 1-dimensional simulation with more flexibility.


- Folder `simulations/`: This folder contains the platform that realizes the 1-dimensional simulation in the paper. Files starting with `plot` in their names contain codes to generate plots in the paper. 
    - Folder `final_perf/`: This folder contains the code for offline estimation in Section 5.1.1 and Section 5.1.2.
        - `run_maxiter.py` simulates 1-dimensional data with different-sign transition functions and test for double non-homogeneity. Usage:
        ```console
        python 01_sim_1d_run.py {init} {N} {T} {setting} {nthread} {cov} {threshold_type} {K} {max_iter} {random_cp}
        ```
        See the annotation in the script for the meanings of arguments. Example:
        ```console
        python 01_sim_1d_run.py kmeans 50 50 pwconst2 8 0.25 maxcusum 2 10 0
        ```
        - `run_maxiter_samesign.py` simulates 1-dimensional data with same-sign transition functions and test for double non-homogeneity.
        - `collect_results.py` collect simulation results of the performance of the proposed method to generate Figure 3 and Figure 4 in the paper.
        - `collect_results.py` collect simulation results of the performance of the proposed method given numbers of clusters when the transition functions of the two clusters have the same sign on the interaction term.
        - `plot_fig3_cp.py` creates Figure 3 of the estimation performance the proposed method given different initial change point locations. 
        - `plot_fig4_cp.py` creates Figure 4 of the estimation performance the proposed method given different numbers of clusters. 
        - To run the 1-dimensional simulation in sequence, 
        ```sh
        bash create_maxiter.sh
        bash create_maxiter_samesign.sh
        ```
    - Folder `value/`: This folder contains the code for online evaluation in Section 5.1.2.
        - `run_value.py` estimates the value of different policies in environment where different clusters have different signs on the interaction term in the transition functions.
        - `run_value_samesign.py` estimates the value of different policies in environment where different clusters have the same sign in the interaction term in the transition functions.
        - `collect_results.py` collect simulation results of the value of the proposed method when the transition functions of the two clusters have different signs on the interaction term. 
        - `collect_results_samesign.py` collect simulation results of the value of the proposed method when the transition functions of the two clusters have the same sign on the interaction term. 
        - `plot_fig5_supp1_value.py` creates Figure 5 in the paper and the Figure 1 in the supplement of value difference distribution. 
        - To run the evaluation in sequence, 
        ```sh
        bash create_value.sh
        bash create_value.samesign.sh
        ```
   
    - Folder `output` contains raw results and corresponding figures of the simulation in the paper.
