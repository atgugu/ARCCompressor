import time
import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler

# Import Adam8bit from bitsandbytes.
from bitsandbytes.optim import Adam8bit

import preprocessing
import arc_compressor
import initializers
import multitensor_systems
import layers
import solution_selection
import visualization


"""
This file trains a model for every ARC-AGI task in a split using Adam8bit.
"""

np.random.seed(0)
torch.manual_seed(0)


def mask_select_logprobs(mask, length):
    """
    Figure out the unnormalized log probability of taking each slice given the output mask.
    """
    logprobs = []
    for offset in range(mask.shape[0] - length + 1):
        logprob = -torch.sum(mask[:offset])
        logprob = logprob + torch.sum(mask[offset:offset + length])
        logprob = logprob - torch.sum(mask[offset + length:])
        logprobs.append(logprob)
    logprobs = torch.stack(logprobs, dim=0)
    log_partition = torch.logsumexp(logprobs, dim=0)
    return log_partition, logprobs


def take_step(task, model, optimizer, train_step, train_history_logger, scaler):
    """
    Runs a forward pass of the model on the ARC-AGI task with mixed precision.
    """
    optimizer.zero_grad()
    
    # Run the forward pass in an AMP context.
    with torch.autocast("cuda"):
        logits, x_mask, y_mask, KL_amounts, KL_names = model.forward()
        logits = torch.cat([torch.zeros_like(logits[:, :1, :, :]), logits], dim=1)  # add black color

        total_KL = 0
        for KL_amount in KL_amounts:
            total_KL += torch.sum(KL_amount)

        reconstruction_error = 0
        for example_num in range(task.n_examples):
            for in_out_mode in range(2):
                if example_num >= task.n_train and in_out_mode == 1:
                    continue

                grid_size_uncertain = not (task.in_out_same_size or 
                                           (task.all_out_same_size and in_out_mode == 1) or 
                                           (task.all_in_same_size and in_out_mode == 0))
                coefficient = (0.01 ** max(0, 1 - train_step / 100)) if grid_size_uncertain else 1

                logits_slice = logits[example_num, :, :, :, in_out_mode]
                problem_slice = task.problem[example_num, :, :, in_out_mode]
                output_shape = task.shapes[example_num][in_out_mode]
                x_log_partition, x_logprobs = mask_select_logprobs(coefficient * x_mask[example_num, :, in_out_mode], output_shape[0])
                y_log_partition, y_logprobs = mask_select_logprobs(coefficient * y_mask[example_num, :, in_out_mode], output_shape[1])
                
                if grid_size_uncertain:
                    x_log_partitions = []
                    y_log_partitions = []
                    for length in range(1, x_mask.shape[1] + 1):
                        x_log_partitions.append(mask_select_logprobs(coefficient * x_mask[example_num, :, in_out_mode], length)[0])
                    for length in range(1, y_mask.shape[1] + 1):
                        y_log_partitions.append(mask_select_logprobs(coefficient * y_mask[example_num, :, in_out_mode], length)[0])
                    x_log_partition = torch.logsumexp(torch.stack(x_log_partitions, dim=0), dim=0)
                    y_log_partition = torch.logsumexp(torch.stack(y_log_partitions, dim=0), dim=0)

                logprobs = [[] for _ in range(x_logprobs.shape[0])]
                for x_offset in range(x_logprobs.shape[0]):
                    for y_offset in range(y_logprobs.shape[0]):
                        logprob = (x_logprobs[x_offset] - x_log_partition +
                                   y_logprobs[y_offset] - y_log_partition)
                        logits_crop = logits_slice[:, x_offset:x_offset + output_shape[0],
                                                        y_offset:y_offset + output_shape[1]]
                        target_crop = problem_slice[:output_shape[0], :output_shape[1]]
                        logprob -= torch.nn.functional.cross_entropy(
                            logits_crop[None, ...], target_crop[None, ...], reduction='sum'
                        )
                        logprobs[x_offset].append(logprob)
                logprobs = torch.stack([torch.stack(lp, dim=0) for lp in logprobs], dim=0)
                coefficient = (0.1 ** max(0, 1 - train_step / 100)) if grid_size_uncertain else 1
                logprob = torch.logsumexp(coefficient * logprobs, dim=(0, 1)) / coefficient
                reconstruction_error -= logprob

        loss = total_KL + 10 * reconstruction_error

    # Use GradScaler for mixed precision backward and optimizer step
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()

    # Log performance metrics.
    train_history_logger.log(
        train_step,
        logits,
        x_mask,
        y_mask,
        KL_amounts,
        KL_names,
        total_KL,
        reconstruction_error,
        loss
    )


if __name__ == "__main__":
    start_time = time.time()

    task_nums = list(range(400))
    split = "training"  # "training", "evaluation", or "test"

    tasks = preprocessing.preprocess_tasks(split, task_nums)
    models = []
    optimizers = []
    scalers = []          # One GradScaler per model for AMP
    train_history_loggers = []

    # Initialize models, optimizers (using Adam8bit), scalers, and loggers.
    for task in tasks:
        model = arc_compressor.ARCCompressor(task)
        models.append(model)
        optimizer = Adam8bit(model.weights_list, lr=0.01, betas=(0.5, 0.9))
        optimizers.append(optimizer)
        scaler = GradScaler()
        scalers.append(scaler)
        train_history_logger = solution_selection.Logger(task)
        visualization.plot_problem(train_history_logger)
        train_history_loggers.append(train_history_logger)

    true_solution_hashes = [task.solution_hash for task in tasks]

    # Training loop: train each model
    for i, (task, model, optimizer, scaler, train_history_logger) in enumerate(
        zip(tasks, models, optimizers, scalers, train_history_loggers)
    ):
        n_iterations = 2000
        for train_step in range(n_iterations):
            take_step(task, model, optimizer, train_step, train_history_logger, scaler)
        visualization.plot_solution(train_history_logger)
        solution_selection.save_predictions(train_history_loggers[:i + 1])
        solution_selection.plot_accuracy(true_solution_hashes)

    # Save timing information.
    with open('timing_result.txt', 'w') as f:
        f.write("Time elapsed in seconds: " + str(time.time() - start_time))
