# %% [markdown]
# # ARC-AGI Without Pretraining - Official Competition Template Version
# This file interfaces between the kaggle competition website and the rest of the solution code, which is included in the input files.
# 
# The main differences between this notebook and the method in the ARC-AGI Without Pretraining blog post aim to parallelize the solving of many puzzles at once using all the CPUs and GPUs that are offered in this competition. In the blog post, we solved puzzles in series, vastly underutilized one RTX 4070 GPU, and blew past the time budget. Instead, what we do in this notebook is:
# - We run 2 steps of every puzzle to determine how much memory each puzzle uses.
# - We run 10 steps of every puzzle at optimal puzzle parallelization under memory constraint to determine how much time per step we need to solve the puzzles in bulk.
# - We run as many steps as we can at optimal puzzle parallelization under memory constraint to fit a 12 hour budget.
# - We have changed layers.direction_share() to make it run faster, and got something like a 5-10% speedup.
# 
# If the dataset size is 120 puzzles, we should expect this to get ~2300 steps in per puzzle.

# %% [markdown]
# ### Imports

# %% [code] {"execution":{"iopub.status.busy":"2025-03-30T04:05:19.678216Z","iopub.execute_input":"2025-03-30T04:05:19.678502Z","iopub.status.idle":"2025-03-30T04:05:23.186962Z","shell.execute_reply.started":"2025-03-30T04:05:19.67848Z","shell.execute_reply":"2025-03-30T04:05:23.18631Z"}}
import os
import sys
import time
import json
import importlib
import multiprocessing
from multiprocessing import Pool

import numpy as np
import torch

sys.path.append('/home/atgu/Desktop/ARCCompressor')

# This little block of code does "import preprocessing" but avoids a name collision with another module
module_path = "/home/atgu/Desktop/ARCCompressor/preprocessing.py"
module_name = "preprocessing"
spec = importlib.util.spec_from_file_location(module_name, module_path)
preprocessing = importlib.util.module_from_spec(spec)
sys.modules[module_name] = preprocessing
spec.loader.exec_module(preprocessing)
import train
import arc_compressor
import initializers
import multitensor_systems
import layers
import solution_selection
import visualization
import solve_task

# %% [markdown]
# ### Getting all the task names, setting defaults and constants

# %% [code] {"execution":{"iopub.status.busy":"2025-03-30T04:05:23.187842Z","iopub.execute_input":"2025-03-30T04:05:23.188172Z","iopub.status.idle":"2025-03-30T04:05:23.339212Z","shell.execute_reply.started":"2025-03-30T04:05:23.188145Z","shell.execute_reply":"2025-03-30T04:05:23.338574Z"}}
multiprocessing.set_start_method('spawn', force=True)
torch.set_default_dtype(torch.float32)
torch.set_default_device('cuda')
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

if __name__ == '__main__':

    start_time = time.time()
    end_time = start_time + 12*3600 - 300

    n_cpus = multiprocessing.cpu_count()
    n_gpus = torch.cuda.device_count()

    # Find all the puzzle names
    split = "test"
    with open(f'/home/atgu/Desktop/ARCCompressor/2025data/arc-agi_{split}_challenges.json', 'r') as f:
        problems = json.load(f)
    task_names = list(problems.keys())
    del problems
    n_tasks = len(task_names)

# %% [markdown]
# ### Function that can spawn processes and schedule them on GPUs to take up each GPUs quota

# %% [code] {"execution":{"iopub.status.busy":"2025-03-30T04:05:23.340363Z","iopub.execute_input":"2025-03-30T04:05:23.340573Z","iopub.status.idle":"2025-03-30T04:05:23.349348Z","shell.execute_reply.started":"2025-03-30T04:05:23.340555Z","shell.execute_reply":"2025-03-30T04:05:23.348749Z"}}
def parallelize_runs(gpu_quotas, task_usages, n_iterations, verbose=False):
    gpu_quotas = gpu_quotas[:]
    # Schedule the tasks greedily to max out memory usage
    t = time.time()
    tasks_started = [False for i in range(n_tasks)]
    tasks_finished = [False for i in range(n_tasks)]
    processes = [None for i in range(n_tasks)]
    process_gpu_ids = [None for i in range(n_tasks)]
    with multiprocessing.Manager() as manager:
        memory_dict = manager.dict()
        solutions_dict = manager.dict()
        error_queue = manager.Queue()
        while not all(tasks_finished):
            if not error_queue.empty():
                raise ValueError(error_queue.get())
            for i in range(n_tasks):
                if tasks_started[i] and not tasks_finished[i]:
                    processes[i].join(timeout=0)
                    if not processes[i].is_alive():
                        tasks_finished[i] = True
                        gpu_quotas[process_gpu_ids[i]] += task_usages[i]
                        if verbose:
                            print(task_names[i], 'finished on gpu', process_gpu_ids[i],
                                  'New quota is', gpu_quotas[process_gpu_ids[i]])
            for gpu_id in range(n_gpus):
                for i in range(n_tasks):
                    enough_quota = gpu_quotas[gpu_id] > task_usages[i]
                    enough_cpus = sum(map(int, tasks_started)) - sum(map(int, tasks_finished)) < n_cpus
                    if not tasks_started[i] and enough_quota and enough_cpus:
                        gpu_quotas[gpu_id] -= task_usages[i]
                        args = (task_names[i], split, end_time, n_iterations, gpu_id, memory_dict, solutions_dict, error_queue)
                        p = multiprocessing.Process(target=solve_task.solve_task, args=args)
                        p.start()
                        processes[i] = p
                        tasks_started[i] = True
                        process_gpu_ids[i] = gpu_id
                        if verbose:
                            print(task_names[i], 'started on gpu', process_gpu_ids[i],
                                  'New quota is', gpu_quotas[process_gpu_ids[i]])
            time.sleep(1)
        if not error_queue.empty():
            raise ValueError(error_queue.get())
        memory_dict = dict(memory_dict)
        solutions_dict = dict(solutions_dict)
    time_taken = time.time() - t
    if verbose:
        print('All jobs finished in', time_taken, 'seconds.')
    return memory_dict, solutions_dict, time_taken

# %% [markdown]
# ### Measuring the amount of memory used for every task

# %% [code] {"execution":{"iopub.status.busy":"2025-03-30T04:05:23.350251Z","iopub.execute_input":"2025-03-30T04:05:23.350448Z","iopub.status.idle":"2025-03-30T04:07:09.252642Z","shell.execute_reply.started":"2025-03-30T04:05:23.350432Z","shell.execute_reply":"2025-03-30T04:07:09.251912Z"}}
if __name__ == '__main__':
    gpu_memory_quotas = [torch.cuda.mem_get_info(i)[0] for i in range(n_gpus)]

    gpu_task_quotas = [int(gpu_memory_quota // (4 * 1024**3)) for gpu_memory_quota in gpu_memory_quotas]
    task_usages = [1 for i in range(n_tasks)]
    memory_dict, _, _ = parallelize_runs(gpu_task_quotas, task_usages, 2, verbose=False)
    
    # Sort the tasks by decreasing memory usage
    tasks = sorted(memory_dict.items(), key=lambda x: x[1], reverse=True)
    task_names, task_memory_usages = zip(*tasks)

# %% [markdown]
# ### Computing the time taken, while saturating memory

# %% [code] {"execution":{"iopub.status.busy":"2025-03-30T04:07:09.253285Z","iopub.execute_input":"2025-03-30T04:07:09.253488Z","iopub.status.idle":"2025-03-30T04:10:47.618982Z","shell.execute_reply.started":"2025-03-30T04:07:09.253471Z","shell.execute_reply":"2025-03-30T04:10:47.618304Z"}}
if __name__ == '__main__':
    test_steps = 20
    safe_gpu_memory_quotas = [memory_quota - 6 * 1024**3 for memory_quota in gpu_memory_quotas]
    _, _, time_taken = parallelize_runs(safe_gpu_memory_quotas, task_memory_usages, test_steps, verbose=False)

# %% [markdown]
# ### Computing the solution for every task, while saturating memory and time

# %% [code] {"execution":{"iopub.status.busy":"2025-03-30T04:10:47.619663Z","iopub.execute_input":"2025-03-30T04:10:47.619864Z"}}
if __name__ == '__main__':
    time_per_step = time_taken / test_steps
    time_left = end_time - time.time()
    n_steps = int(time_left // time_per_step)
    _, solutions_dict, time_taken = parallelize_runs(safe_gpu_memory_quotas, task_memory_usages, n_steps, verbose=True)
    
    # Format the solutions and put into submission file
    with open('submission.json', 'w') as f:
        json.dump(solutions_dict, f, indent=4)
        
    print(n_tasks, 'tasks solved.')
    print(n_steps, 'steps taken.')
    print(time_taken, 'seconds taken.')