#!/usr/bin/python
import platform, sys, os
import numpy as np
plat = platform.platform()
if plat == 'macOS-12.5-x86_64-i386-64bit': ##local
    os.chdir("/Users/mengbing/Documents/research/change_point_clustering/HeterRL_private/simulation_real")
    sys.path.append("/Users/mengbing/Documents/research/change_point_clustering/HeterRL_private")
elif plat == 'Linux-3.10.0-1160.42.2.el7.x86_64-x86_64-with-centos-7.6.1810-Core': # biostat cluster
    os.chdir("/home/mengbing/research/HeterRL/simulation_real")
    sys.path.append("/home/mengbing/research/HeterRL")
elif plat == 'Linux-4.18.0-305.45.1.el8_4.x86_64-x86_64-with-glibc2.28':  # greatlakes
    os.chdir("/home/mengbing/research/HeterRL/simulation_real/submission_scripts")
    sys.path.append("/home/mengbing/research/HeterRL/simulation_real/submission_scripts")

import subprocess
import time

#%% create slurm scripts
os.system("bash 01_simulate_real.sh")
effect_sizes = ["weak", "moderate", "strong"]
Ks = [2,3,4,5]
inits = [10,15,20]

njobs = 0
n_jobs_running = 0
cmd_job = ["squeue", "-u", "mengbing"]
error_handling1 = """exitStatus=1; while [[ "$exitStatus" == "1" ]]; do sleep 10; """
error_handling2 = """exitStatus=$?; done"""
for effect_size in effect_sizes:
    for K in Ks:
        for init in inits:
            # if ((setting == ["homo","pwconst2"] and gamma == 0.9 and kappa >= 60)):
            #     continue
            job_name = "01_simulate_real_" + effect_size + "_K" + str(K) + "_init" + str(init) + "_run.slurm"
            cmd = error_handling1 + "sbatch " + job_name + "; " + error_handling2
            # # submit the first job
            # if njobs == 0 or n_jobs_running < 500:
            #     # cmd = ["sbatch", job_name]
            #     cmd = error_handling1 + "sbatch " + job_name + "; " + error_handling2
            #
            # else:
            #     # cmd = ["sbatch", "--depend=afterany:"+str(jobnum), job_name]
            #     cmd = error_handling1 + "sbatch --depend=afterany:" + str(jobnum) + " " + job_name + "; " + error_handling2

            print("Submitting Job with command: %s" % cmd)
            status = subprocess.check_output(cmd, shell=True)
            time.sleep(5)
            jobnum = [int(s) for s in status.split() if s.isdigit()][0]
            print("Job number is %s" % jobnum)


            # check the number of running jobs
            # cmd_job = ["squeue", "-u", "mengbing"]
            job_status = subprocess.check_output(cmd_job)
            time.sleep(5)
            # n_jobs_running = len(job_status.split(b'\n')) - 2

            jobs_running = job_status.split(b'\n')[1:-1]
            n_jobs_running = len(jobs_running)
            # find array jobs
            array_jobs = [len(job_running.split(b'[')) for job_running in jobs_running]
            n_array_jobs = sum(array_job == 2 for array_job in array_jobs)

            while n_array_jobs > 4 and n_jobs_running >= 1300:
                time.sleep(20)
                job_status = subprocess.check_output(cmd_job)
                # n_jobs_running = len(job_status.split(b'\n')) - 2
                jobs_running = job_status.split(b'\n')[1:-1]
                n_jobs_running = len(jobs_running)
                array_jobs = [len(job_running.split(b'[')) for job_running in jobs_running]
                n_array_jobs = sum(array_job == 2 for array_job in array_jobs)

            njobs += 1

    print("\nCurrent status:\n")
    #show the current status
    os.system("squeue -u mengbing")



# os.system("python3 ../summary_nonstationary_rbf_1d.py")

# job_status = b'  JOBID PARTITION     NAME     USER  ACCOUNT ST       TIME  NODES NODELIST(REASON)\n 19399394_[157-500]  standard poly_tra mengbing zhenkewu PD       0:00      1 (None)\n19399394_101  standard poly_tra mengbing zhenkewu PD       0:00      1 (None)\n'
# jobs_running = job_status.split(b'\n')[1:-1]
# n_jobs_running = len(jobs_running)
# array_jobs = [len(job_running.split(b'[')) for job_running in jobs_running]
# n_array_jobs = sum(array_job == 2 for array_job in array_jobs)
#
# test1 = b'19399394_[157-500]  standard poly_tra mengbing zhenkewu PD       0:00      1 (None)\n'
# test2 = b'19399394_101  standard poly_tra mengbing zhenkewu PD       0:00      1 (None)\n'
# len(test1.split(b'['))
# len(test2.split(b'['))

# success = False
# while not success:
#     print(success)
#     time.sleep(1)
#     try:
#         status = subprocess.check_output(cmd)
#         print(status)
#         jobnum = [int(s) for s in status.split() if s.isdigit()][0]
#         if type(jobnum) == int or type(jobnum) == float:
#             success = True
#             print("Job number is %s" % jobnum)
#             # check the number of running jobs
#             job_status = subprocess.check_output(cmd_job)
#         else:
#             continue
#
#         # jobnum = [int(s) for s in status.split() if s.isdigit()][0]
#
#     except subprocess.CalledProcessError as e:
#         print(e.output)
#         raise RuntimeError(
#             "command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))