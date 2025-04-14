import os
import sys
import time
import json
import importlib
import gc
import multiprocessing
import tqdm
import traceback

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
from bitsandbytes.optim import AdamW8bit

def solve_task(task_name, split, time_limit, n_train_iterations, gpu_id, memory_dict, solutions_dict, error_queue):
    try:
        torch.set_default_device('cuda')
        torch.cuda.set_device(gpu_id)
        torch.cuda.reset_peak_memory_stats()

        with open(f'/home/atgu/Desktop/ARCCompressor/2025data/arc-agi_{split}_challenges.json', 'r') as f:
            problems = json.load(f)
        task = preprocessing.Task(task_name, problems[task_name], None)
        del problems

        model = arc_compressor.ARCCompressor(task)
        optimizer = optimizer = AdamW8bit(model.weights_list, lr=0.01, betas=(0.5, 0.9), weight_decay=0.0001)
        scaler = torch.GradScaler("cuda") 
        train_history_logger = solution_selection.Logger(task)
        train_history_logger.solution_most_frequent = tuple(((0, 0), (0, 0)) for example_num in range(task.n_test))
        train_history_logger.solution_second_most_frequent = tuple(((0, 0), (0, 0)) for example_num in range(task.n_test))

        for train_step in range(n_train_iterations):
            train.take_step(task, model, optimizer, train_step, train_history_logger, scaler)
            if time.time() > time_limit:
                break

        example_list = []
        for example_num in range(task.n_test):
            attempt_1 = [list(row) for row in train_history_logger.solution_most_frequent[example_num]]
            attempt_2 = [list(row) for row in train_history_logger.solution_second_most_frequent[example_num]]
            example_list.append({'attempt_1': attempt_1, 'attempt_2': attempt_2})
        del task
        del model
        del optimizer
        del train_history_logger
        torch.cuda.empty_cache()
        gc.collect()

        memory_dict[task_name] = torch.cuda.max_memory_allocated()
        solutions_dict[task_name] = example_list
    except Exception as e:
        error_queue.put(traceback.format_exc())
