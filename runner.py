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
import os
import signal
import psutil

def kill_python3_processes():
    current_pid = os.getpid()
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        pid   = proc.info['pid']
        name  = proc.info['name'] or ''
        cmd   = proc.info['cmdline'] or []

        # Check if this is a python3 process (by name or command-line)
        if pid != current_pid and ('python3' in name or (cmd and 'python3' in cmd[0])):
            try:
                os.kill(pid, signal.SIGTERM)
            except (psutil.NoSuchProcess, PermissionError) as e:
                pass
# %% [markdown]
# ### Getting all the task names, setting defaults and constants

# %% [code] {"execution":{"iopub.status.busy":"2025-03-30T04:05:23.187842Z","iopub.execute_input":"2025-03-30T04:05:23.188172Z","iopub.status.idle":"2025-03-30T04:05:23.339212Z","shell.execute_reply.started":"2025-03-30T04:05:23.188145Z","shell.execute_reply":"2025-03-30T04:05:23.338574Z"}}
multiprocessing.set_start_method('spawn', force=True)
torch.set_default_dtype(torch.float32)
torch.set_default_device('cuda')
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

if __name__ == '__main__':
    kill_python3_processes()

    start_time = time.time()
    end_time = int(start_time + 1.0*3600 - 200)

    n_cpus = multiprocessing.cpu_count()
    n_gpus = torch.cuda.device_count()

    # Find all the puzzle names
    split = "test"
    with open(f'/home/atgu/Desktop/ARCCompressor/2025data/arc-agi_{split}_challenges.json', 'r') as f:
        problems = json.load(f)
    task_names = list(problems.keys())
    del problems
    n_tasks = 10#len(task_names)

# %% [markdown]
# ### Function that can spawn processes and schedule them on GPUs to take up each GPUs quota

# %% [code] {"execution":{"iopub.status.busy":"2025-03-30T04:05:23.340363Z","iopub.execute_input":"2025-03-30T04:05:23.340573Z","iopub.status.idle":"2025-03-30T04:05:23.349348Z","shell.execute_reply.started":"2025-03-30T04:05:23.340555Z","shell.execute_reply":"2025-03-30T04:05:23.348749Z"}}
def parallelize_runs(gpu_quotas, task_usages, n_iterations, verbose=False, calibrate=False):
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
                        args = (task_names[i], split, end_time, n_iterations, gpu_id, memory_dict, solutions_dict, error_queue, i, calibrate)
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

    print("Measuring the amount of memory used for every task")
    gpu_memory_quotas = [torch.cuda.mem_get_info(i)[0] for i in range(n_gpus)]

    gpu_task_quotas = [int(gpu_memory_quota // (1.8 * 1024**3)) for gpu_memory_quota in gpu_memory_quotas] #4
    task_usages = [1 for i in range(n_tasks)]
    memory_dict, _, _ = parallelize_runs(gpu_task_quotas, task_usages, 2, verbose=False, calibrate=True)
    
    # Sort the tasks by decreasing memory usage
    tasks = sorted(memory_dict.items(), key=lambda x: x[1], reverse=True)
    task_names, task_memory_usages = zip(*tasks)

# %% [markdown]
# ### Computing the time taken, while saturating memory

# %% [code] {"execution":{"iopub.status.busy":"2025-03-30T04:07:09.253285Z","iopub.execute_input":"2025-03-30T04:07:09.253488Z","iopub.status.idle":"2025-03-30T04:10:47.618982Z","shell.execute_reply.started":"2025-03-30T04:07:09.253471Z","shell.execute_reply":"2025-03-30T04:10:47.618304Z"}}
if __name__ == '__main__':
    test_steps = 20
    safe_gpu_memory_quotas = [memory_quota - 1 * 1024**3 for memory_quota in gpu_memory_quotas] #6
    _, _, time_taken = parallelize_runs(safe_gpu_memory_quotas, task_memory_usages, test_steps, verbose=False, calibrate=True)

# %% [markdown]
# ### Computing the solution for every task, while saturating memory and time

# %% [code] {"execution":{"iopub.status.busy":"2025-03-30T04:10:47.619663Z","iopub.execute_input":"2025-03-30T04:10:47.619864Z"}}
if __name__ == '__main__':
    time_per_step = time_taken / test_steps
    time_left = end_time - time.time()
    n_steps = int(time_left // time_per_step)
    _, solutions_dict, time_taken = parallelize_runs(safe_gpu_memory_quotas, task_memory_usages, n_steps, verbose=True, calibrate=False)
    
    # Format the solutions and put into submission file
    with open('submission.json', 'w') as f:
        json.dump(solutions_dict, f, indent=4)
        
    print(n_tasks, 'tasks solved.')
    print(n_steps, 'steps taken.')
    print(time_taken, 'seconds taken.')


    # %% [code]
    # Evaluate against the training‑split ground truth

    # Load your predictions
    with open('submission.json', 'r') as f:
        preds = json.load(f)

    # Load the true solutions for the training split
    # (adjust the path to wherever your arc‑agi_training_solutions.json lives)
    with open('/home/atgu/Desktop/ARCCompressor/2025data/arc-agi_training_solutions.json', 'r') as f:
        truths = json.load(f)

    perfect = 0
    total   = len(preds)

    for task_name, pred_examples in preds.items():
        attempt_1_match = True
        attempt_2_match = True
        true_examples = truths.get(task_name)
        if true_examples is None:
            # no ground truth for this task
            continue

        # require that for every example, attempt_1 OR attempt_2 equals the truth
        all_match = True
        for i, true_grid in enumerate(true_examples):
            a1 = pred_examples[i]['attempt_1']
            a2 = pred_examples[i]['attempt_2']
            if a1 != true_grid:
                attempt_1_match = False
            if a2 != true_grid:
                attempt_2_match = False
            if a1 != true_grid and a2 != true_grid:
                all_match = False
                break

        if all_match or attempt_1_match or attempt_2_match:
            perfect += 1

    print(f"Predicted {perfect}/{total} tasks.")

    # %% [code]
    import matplotlib.pyplot as plt

    color_list = np.array([
        [0, 0, 0],  # black
        [30, 147, 255],  # blue
        [249, 60, 49],  # red
        [79, 204, 48],  # green
        [255, 220, 0],  # yellow
        [153, 153, 153],  # gray
        [229, 58, 163],  # magenta
        [255, 133, 27],  # orange
        [135, 216, 241],  # light blue
        [146, 18, 49],  # brown
    ])

    def convert_color(grid):  # grid dims must end in c
        return np.clip(np.matmul(grid, color_list), 0, 255).astype(np.uint8)

    def plot_problem(logger):
        n_train = logger.task.n_train
        n_test  = logger.task.n_test
        n_x, n_y = logger.task.n_x, logger.task.n_y

        pixels = 255 + np.zeros([n_train+n_test, 2*n_x+2, 2, 2*n_y+8, 3], dtype=np.uint8)
        for ex in range(n_train+n_test):
            subsplit = 'train' if ex < n_train else 'test'
            idx      = ex if ex < n_train else ex - n_train
            for mode_num, mode in enumerate(('input','output')):
                if subsplit=='test' and mode=='output': continue
                grid = np.array(logger.task.unprocessed_problem[subsplit][idx][mode])
                grid = (np.arange(10)==grid[:,:,None]).astype(np.float32)
                grid = convert_color(grid)
                rg = np.repeat(np.repeat(grid,2,0),2,1)
                pixels[ex, n_x+1-grid.shape[0]:n_x+1+grid.shape[0],
                        mode_num,
                        n_y+4-grid.shape[1]:n_y+4+grid.shape[1]] = rg

        img = pixels.reshape([(n_train+n_test)*(2*n_x+2), 2*(2*n_y+8), 3])
        os.makedirs("plots", exist_ok=True)
        fig,ax = plt.subplots(figsize=(6,6))
        ax.imshow(img,interpolation='none',aspect='equal')
        ax.axis('off')
        fig.savefig(f"/home/atgu/Desktop/ARCCompressor/plots/{logger.task.task_name}_problem.png", bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    def plot_solution(logger, fname=None):
        n_train = logger.task.n_train
        n_test  = logger.task.n_test
        n_x, n_y = logger.task.n_x, logger.task.n_y

        sols = [
        torch.softmax(logger.current_logits,dim=1).cpu().numpy(),
        torch.softmax(logger.ema_logits,dim=1).cpu().numpy(),
        logger.solution_most_frequent,
        logger.solution_second_most_frequent,
        ]
        masks = [
        (logger.current_x_mask, logger.current_y_mask),
        (logger.ema_x_mask,     logger.ema_y_mask),
        None, None
        ]
        labels = ['sample','sample average','guess 1','guess 2']
        P = len(sols)

        pixels = 255 + np.zeros([n_test, 2*n_x+2, P, 2*n_y+8, 3], dtype=np.uint8)
        shapes = []

        for i in range(n_test):
            shapes.append([])
            for j,(sol,msk,label) in enumerate(zip(sols,masks,labels)):
                grid = np.array(sol[i])  # either cxy or xy
                if 'sample' in label:
                    grid = np.einsum('dxy,dc->xyc', grid, color_list[logger.task.colors])
                    xl, yl = (None,None) if not (logger.task.in_out_same_size or logger.task.all_out_same_size) \
                            else (logger.task.shapes[n_train+i][1])
                    x0,x1 = logger._best_slice_point(msk[0][i], xl)
                    y0,y1 = logger._best_slice_point(msk[1][i], yl)
                    grid = grid[x0:x1,y0:y1,:]
                    grid = np.clip(grid,0,255).astype(np.uint8)
                else:
                    grid = (np.arange(10)==grid[:,:,None]).astype(np.float32)
                    grid = convert_color(grid)

                shapes[i].append(grid.shape[:2])
                rg = np.repeat(np.repeat(grid,2,0),2,1)
                pixels[i, n_x+1-grid.shape[0]:n_x+1+grid.shape[0],
                        j,
                        n_y+4-grid.shape[1]:n_y+4+grid.shape[1]] = rg

        img = pixels.reshape([n_test*(2*n_x+2), P*(2*n_y+8), 3])
        fig,ax = plt.subplots(figsize=(6,6))
        ax.imshow(img,interpolation='none',aspect='equal')
        ax.axis('off')
        if fname is None:
            fname = f"/home/atgu/Desktop/ARCCompressor/plots/{logger.task.task_name}_solutions.png"
        fig.savefig(fname, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    # %% [code]
    from preprocessing import Task
    from solution_selection import Logger

    # load your preds & the challenge definitions for whichever split you're visualizing
    with open('submission.json') as f: preds = json.load(f)
    with open('/home/atgu/Desktop/ARCCompressor/2025data/arc-agi_test_challenges.json') as f:
        problems = json.load(f)

    for name, ex_list in preds.items():
        task   = Task(name, problems[name], None)
        logger = Logger(task)
        # override the two guesses
        logger.solution_most_frequent = tuple(
            tuple(tuple(row) for row in ex['attempt_1'])
            for ex in ex_list
        )
        logger.solution_second_most_frequent = tuple(
            tuple(tuple(row) for row in ex['attempt_2'])
            for ex in ex_list
        )
        # dump plots
        plot_problem(logger)
        plot_solution(logger)
        print("Plotted solution for ", name)
