import torch
import copy
import random
from preprocessing import Task
from arc_compressor import ARCCompressor
from train import take_step
from solution_selection import Logger

def get_task_sample(task_name, problem_data):
    task = Task(task_name, problem_data[task_name], None)
    return task

def get_weights(model):
    return [w.detach().clone() for w in model.weights_list]

def set_weights(model, weights):
    for param, new_val in zip(model.weights_list, weights):
        param.data.copy_(new_val.data)
def set_weights_safely(model, weights_ref):
    for param, ref in zip(model.weights_list, weights_ref):
        if param.shape == ref.shape:
            param.data.copy_(ref.data)
        else:
            # Optional: zero out or skip unmatched
            print(f"[WARN] Shape mismatch: skipping param {param.shape} != {ref.shape}")
def average_weights_safely(meta_weights, task_weights, alpha):
    for mw, tw in zip(meta_weights, task_weights):
        if mw.shape == tw.shape:
            mw.data += alpha * (tw.data - mw.data)
        else:
            print(f"[WARN] Shape mismatch during meta-update: {mw.shape} != {tw.shape}")

def average_weights(meta_weights, task_weights, alpha):
    for mw, tw in zip(meta_weights, task_weights):
        mw.data += alpha * (tw.data - mw.data)

def train_reptile(problem_data, task_names, meta_iters=1000, inner_iters=5, meta_lr=0.1):
    # Choose one task to initialize the meta-model
    meta_task = get_task_sample(random.choice(task_names), problem_data)
    meta_model = ARCCompressor(meta_task)
    meta_weights = get_weights(meta_model)

    for meta_step in range(meta_iters):
        task_name = random.choice(task_names)
        task = get_task_sample(task_name, problem_data)
        model = ARCCompressor(task)
        set_weights_safely(model, meta_weights)

        optimizer = torch.optim.Adam(model.weights_list, lr=0.01, betas=(0.5, 0.9))
        logger = Logger(task)

        for inner_step in range(inner_iters):
            take_step(task, model, optimizer, inner_step, logger)

        task_weights = get_weights(model)
        average_weights_safely(meta_weights, task_weights, alpha=meta_lr)

        if meta_step % 10 == 0:
            print(f"Meta Step {meta_step}: Updated meta-weights.")

    # Save the trained meta-model
    set_weights_safely(meta_model, meta_weights)
    return meta_model
