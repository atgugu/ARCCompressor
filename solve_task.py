import torch
from tqdm import trange
from torch.optim.lr_scheduler import ReduceLROnPlateau
from bitsandbytes.optim import AdamW8bit
import torch.cuda.amp as amp
import gc, traceback, json, time
import preprocessing
import train
import arc_compressor
import solution_selection
import sys, importlib
sys.path.append('/home/atgu/Desktop/ARCCompressor')

# This little block of code does "import preprocessing" but avoids a name collision with another module
module_path = "/home/atgu/Desktop/ARCCompressor/preprocessing.py"
module_name = "preprocessing"
spec = importlib.util.spec_from_file_location(module_name, module_path)
preprocessing = importlib.util.module_from_spec(spec)
sys.modules[module_name] = preprocessing
spec.loader.exec_module(preprocessing)

def solve_task(task_name,
               split,
               time_limit,
               n_train_iterations,
               gpu_id,
               memory_dict,
               solutions_dict,
               error_queue,
               position,
               calibrate=False):
    try:
        # --- GPU setup & clear old state ---
        torch.cuda.empty_cache()
        gc.collect()
        torch.set_default_device('cuda')
        torch.cuda.set_device(gpu_id)
        torch.cuda.reset_peak_memory_stats()

        # --- Load task spec ---
        path = f'/home/atgu/Desktop/ARCCompressor/2025data/arc-agi_{split}_challenges.json'
        with open(path, 'r') as f:
            problems = json.load(f)

        task = preprocessing.Task(task_name, problems[task_name], None, transforms='all')

        # --- Build model / optimizer / scaler / scheduler / logger ---
        model     = arc_compressor.ARCCompressor(task)
        model.load_invariant(f'/home/atgu/Desktop/ARCCompressor/invariants/{task_name}.pt')    
        train_history_logger = solution_selection.Logger(task)
        train_history_logger.solution_most_frequent = tuple(((0, 0), (0, 0)) for example_num in range(task.n_test))
        train_history_logger.solution_second_most_frequent = tuple(((0, 0), (0, 0)) for example_num in range(task.n_test))

        optimizer = AdamW8bit(
            model.weights_list,
            lr=3e-3,
            betas=(0.7, 0.95),
            weight_decay=1e-6
        )
        scaler    = torch.amp.GradScaler('cuda')
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.8,
            patience=20,
            threshold=1e-3,
            min_lr=1e-4
        )
        logger    = solution_selection.Logger(task)

        # --- Training loop with TQDM & early stop ---
        pbar     = trange(
            n_train_iterations,
            desc=task_name,
            unit='step',
            position=position,
            leave=True,
            dynamic_ncols=True
        )
        best_loss = float('inf')

        for step in pbar:
            if time.time() > time_limit:
                break
            # one training step returns the scalar loss
            loss = train.take_step(
                task, task_name, model, optimizer,
                step, logger, scaler, scheduler
            )

            # update tqdm
            if loss < best_loss:
                best_loss = loss
                model.save_invariant(f'/home/atgu/Desktop/ARCCompressor/invariants/{task_name}.pt')
            pbar.set_postfix(loss=f"{loss:.4f}", best=f"{best_loss:.4f}")

            # early‑stop if fully reconstructing train set
            if step > 30000000:
                with torch.no_grad():
                    logits, x_mask, y_mask, _, _ = model.forward()
                    # add black channel
                    logits = torch.cat(
                        [torch.zeros_like(logits[:, :1, :, :]), logits],
                        dim=1
                    )
                    # predicted color index
                    preds = torch.argmax(logits, dim=1)  # [examples, x, y]

                    # true train outputs
                    true_train = task.problem[:task.n_train, :, :, 1]
                    if torch.equal(preds[:task.n_train], true_train):
                        print("Early stop at step", step, " for task", task_name, "with loss", loss, "and best loss", best_loss, " fully reconstructed train outputs.")
                        break
                # scheduler.step(loss)

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