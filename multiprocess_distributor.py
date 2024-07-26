import time as time
import numpy as np
import os as os
import multiprocessing as mp
import itertools
import psutil as pst

def process_distributor(target_num):
    num_process = pst.cpu_count(logical=False) # Number of PHYSICAL CPUs
    numbers = [1, 2, 3, 4, 5, 6, 8] # As the maximun order in input file is 30, the function make the combination of these numbers
    process_distr_list = np.zeros((num_process, 2))
    initializer = np.zeros(num_process)
    initializer[0] = target_num
    all_results = np.array([initializer])
    pr_max = target_num
    pr_min = 0
    for combination in itertools.product(numbers, repeat=num_process):
        if sum(combination) == target_num:
            all_results = np.append(all_results, np.array([combination]), axis=0)
            if np.max(combination) <= pr_max and np.min(combination) >= pr_min :
                best_result = combination
                pr_max = np.max(combination)
                pr_min = np.min(combination)
    for i in range(num_process):
        if i == 0:
            process_distr_list[i, 1] = best_result[0]
        else:
            process_distr_list[i, 0] = process_distr_list[i - 1, 1]
            process_distr_list[i, 1] = process_distr_list[i, 0] + best_result[i]
            
    process_distr_list = process_distr_list.astype(int)       
    return  num_process, all_results, best_result, process_distr_list

 # Define your numbers here. These numbers 
 
if __name__ == "__main__":
    target = 13
    num_process, permutations, best_result, process_dist_list = process_distributor(target)

    print(f"Permutations of 4 numbers that sum to {target}:")
    for permutation in permutations:
        print(permutation, "\n")
    print("\n\n The best result is: ", best_result)
    print("\n\n The best result is: ", process_dist_list)

    num_processes = mp.cpu_count()
    print("number of CPU core(s) are:", num_processes)

    num_physical_cores = pst.cpu_count(logical=False)

    print("number of Physical CPU(s) are: ", num_physical_cores)

