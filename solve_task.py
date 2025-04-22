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
               position):
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
        task = preprocessing.Task(task_name, problems[task_name], None)
        del problems

        # --- Build model / optimizer / scaler / scheduler / logger ---
        model     = arc_compressor.ARCCompressor(task)
        # if(task_name == '007bbfb7'):
        #     model.load_invariant(f'/home/atgu/Desktop/ARCCompressor/invariants/00576224.pt')    
        # if(task_name == '00576224'):
        #     model.load_invariant(f'/home/atgu/Desktop/ARCCompressor/invariants/007bbfb7.pt')   
        optimizer = AdamW8bit(
            model.weights_list,
            lr=1e-2,
            betas=(0.5, 0.9),
            weight_decay=1e-4
        )
        scaler    = torch.amp.GradScaler('cuda')
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.99,
            patience=30,
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

                # LR schedule and time check
                scheduler.step(loss)
                if time.time() > time_limit:
                    break

        # --- Inference on test split ---
        with torch.no_grad():
            # raw model output has shape [N, C, X, Y, 2]
            out, x_mask, y_mask, _, _ = model.forward()

            # select the “output” channel = index 1 of that last dim → now [N, C, X, Y]
            logits_pred = out[..., 1]

            # add a zero‐logit “black” class at color‐index 0 → [N, C+1, X, Y]
            logits_pred = torch.cat(
                [torch.zeros_like(logits_pred[:, :1, :, :]), logits_pred],
                dim=1
            )

            # split off only the test examples
            test_logits = logits_pred[task.n_train:]      # [n_test, C+1, X, Y]
            test_xmask  = x_mask[task.n_train:, :, 1]     # [n_test, X]
            test_ymask  = y_mask[task.n_train:, :, 1]     # [n_test, Y]

            # now these have exactly the right shapes for Logger._postprocess_solution
            solutions, _ = logger._postprocess_solution(
                test_logits, test_xmask, test_ymask
            )

        # turn into the two‑attempt format
        example_list = []
        for grid in solutions:
            sol = [list(row) for row in grid]
            example_list.append({
                'attempt_1': sol,
                'attempt_2': sol
            })

        # … cleanup & memory report unchanged …
        memory_dict[task_name]    = torch.cuda.max_memory_allocated()
        solutions_dict[task_name] = example_list

    except Exception:
        error_queue.put(traceback.format_exc())