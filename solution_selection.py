import matplotlib.pyplot as plt
import numpy as np
import torch

np.random.seed(0)
torch.manual_seed(0)

class Logger:
    """
    Record model outputs, accumulate candidate solutions, and track top-K hypotheses.
    """
    ema_decay = 0.97

    def __init__(self, task, top_k=5):
        self.task = task
        self.top_k = top_k
        # Curves
        self.KL_curves = {}
        self.total_KL_curve = []
        self.reconstruction_error_curve = []
        self.loss_curve = []
        self.solution_most_frequent       = tuple([(0, 0), (0, 0)])
        self.solution_second_most_frequent = tuple([(0, 0), (0, 0)])

        # Buffers for logits and masks
        n_test, n_colors, n_x, n_y = task.n_test, task.n_colors, task.n_x, task.n_y
        shape = (n_test, n_colors + 1, n_x, n_y)
        self.current_logits = torch.zeros(shape)
        self.current_x_mask = torch.zeros((n_test, n_x))
        self.current_y_mask = torch.zeros((n_test, n_y))
        self.ema_logits = torch.zeros(shape)
        self.ema_x_mask = torch.zeros((n_test, n_x))
        self.ema_y_mask = torch.zeros((n_test, n_y))

        # Tracking solution scores
        self.solution_hashes_count = {}        # {hash: aggregated_score}
        self.hash_to_solution = {}             # {hash: grid}
        self.top_solutions = []                # list of grids for top-K hashes
        self.solution_contributions_log = []
        self.solution_picks_history = []       # list of lists of top-K hashes per step

    def log(self, train_step, logits, x_mask, y_mask,
            KL_amounts, KL_names, total_KL, reconstruction_error, loss):
        if train_step == 0:
            self.KL_curves = {name: [] for name in KL_names}
        # record KL, losses
        for amount,name in zip(KL_amounts, KL_names):
            self.KL_curves[name].append(float(amount.sum().cpu()))
        self.total_KL_curve.append(float(total_KL.cpu()))
        self.reconstruction_error_curve.append(float(reconstruction_error.cpu()))
        self.loss_curve.append(float(loss.cpu()))

        # update candidate solutions
        self._track_solution(train_step, logits.detach(), x_mask.detach(), y_mask.detach())

    def _track_solution(self, train_step, logits, x_mask, y_mask):
        # extract test logits / masks
        self.current_logits = logits[self.task.n_train:,:,:,:,1]
        self.current_x_mask = x_mask[self.task.n_train:,:,1]
        self.current_y_mask = y_mask[self.task.n_train:,:,1]
        # update EMAs
        self.ema_logits = self.ema_decay*self.ema_logits + (1-self.ema_decay)*self.current_logits
        self.ema_x_mask = self.ema_decay*self.ema_x_mask + (1-self.ema_decay)*self.current_x_mask
        self.ema_y_mask = self.ema_decay*self.ema_y_mask + (1-self.ema_decay)*self.current_y_mask

        # generate two candidate grids: sample & EMA
        candidates = [
            (self.current_logits, self.current_x_mask, self.current_y_mask),
            (self.ema_logits, self.ema_x_mask, self.ema_y_mask)
        ]
        contributions = []
        for logits_c, xm, ym in candidates:
            grid, uncert = self._postprocess_solution(logits_c, xm, ym)
            h = hash(grid)
            # map hash→grid once
            if h not in self.hash_to_solution:
                self.hash_to_solution[h] = grid
            # compute score
            score = -10 * uncert
            if train_step < 150:
                score -= 10
            if logits_c is self.ema_logits:
                score -= 4
            # aggregate via logsumexp
            prev = self.solution_hashes_count.get(h, -np.inf)
            self.solution_hashes_count[h] = float(np.logaddexp(prev, score))
            contributions.append((h, score))

        # record contributions
        self.solution_contributions_log.append(contributions)
        # update top-K solutions by aggregated count
        self._update_top_k()
        # store history of top-K hashes
        self.solution_picks_history.append([h for h,_ in self.solution_hashes_count.items()]
                                           [:self.top_k])

    def _update_top_k(self):
        # sort hashes by descending aggregated score
        sorted_items = sorted(self.solution_hashes_count.items(),
                              key=lambda kv: kv[1], reverse=True)
        top = sorted_items[:self.top_k]
        # update grid list
        self.top_solutions = [self.hash_to_solution[h] for h,_ in top]
        if len(self.top_solutions) > 0:
            self.solution_most_frequent = self.top_solutions[0]
        if len(self.top_solutions) > 1:
            self.solution_second_most_frequent = self.top_solutions[1]

    def best_crop(self, prediction, x_mask, x_length, y_mask, y_length):
        x_start, x_end = self._best_slice_point(x_mask, x_length)
        y_start, y_end = self._best_slice_point(y_mask, y_length)
        return prediction[..., x_start:x_end, y_start:y_end]

    def _best_slice_point(self, mask, length):
        if self.task.in_out_same_size or self.task.all_out_same_size:
            search_lengths = [length]
        else:
            search_lengths = list(range(1, mask.shape[0]+1))
        max_logprob, best_slice_start, best_slice_end = None, None, None

        for length in search_lengths:
            logprobs = torch.stack([
                -torch.sum(mask[:offset]) + torch.sum(mask[offset:offset + length]) - torch.sum(mask[offset + length:])
                for offset in range(mask.shape[0] - length + 1)
            ])
            if max_logprob is None or torch.max(logprobs) > max_logprob:
                max_logprob = torch.max(logprobs)
                best_slice_start = torch.argmax(logprobs).item()
                best_slice_end = best_slice_start + length

        return best_slice_start, best_slice_end

    def _postprocess_solution(self, prediction, x_mask, y_mask):  # prediction must be example, color, x, y
        """Postprocess a solution and compute some variables that are used to calculate the score."""
        colors = torch.argmax(prediction, dim=1)  # example, x, y
        uncertainties = torch.logsumexp(prediction, dim=1) - torch.amax(prediction, dim=1)  # example, x, y
        solution_slices, uncertainty_values = [], []  # example, x, y; example

        for example_num in range(self.task.n_test):
            x_length = None
            y_length = None
            if self.task.in_out_same_size or self.task.all_out_same_size:
                x_length = self.task.shapes[self.task.n_train+example_num][1][0]
                y_length = self.task.shapes[self.task.n_train+example_num][1][1]
            solution_slice = self.best_crop(colors[example_num],
                                            x_mask[example_num],
                                            x_length,
                                            y_mask[example_num],
                                            y_length)  # x, y
            uncertainty_slice = self.best_crop(uncertainties[example_num],
                                               x_mask[example_num],
                                               x_length,
                                               y_mask[example_num],
                                               y_length)  # x, y

            solution_slices.append(solution_slice.cpu().numpy().tolist())
            uncertainty_values.append(float(np.mean(uncertainty_slice.cpu().numpy())))

        for example in solution_slices:
            for row in example:
                for i, val in enumerate(row):
                    row[i] = self.task.colors[val]

        solution_slices = tuple(tuple(tuple(row) for row in example) for example in solution_slices)
        return solution_slices, np.mean(uncertainty_values)


def save_predictions(loggers, fname='predictions.npz'):
    """Saves solution score contributions and history of chosen solutions."""
    np.savez(fname,
             solution_contribution_logs=[logger.solution_contributions_log for logger in loggers],
             solution_picks_histories=[logger.solution_picks_history for logger in loggers])


def plot_accuracy(true_solution_hashes, fname='predictions.npz'):
    """Plots accuracy curve over training iterations."""
    stored_data = np.load(fname, allow_pickle=True)
    solution_picks_histories = stored_data['solution_picks_histories']

    n_tasks = len(solution_picks_histories)
    n_iterations = len(solution_picks_histories[0])

    correct = np.array([[
        int(any(hash_ == true_solution_hashes[task_num] for hash_ in solution_pair))
        for solution_pair in task_history
    ] for task_num, task_history in enumerate(solution_picks_histories)])

    accuracy_curve = correct.mean(axis=0)

    plt.figure()
    plt.plot(np.arange(n_iterations), accuracy_curve, 'k-')
    plt.savefig('accuracy_curve.pdf', bbox_inches='tight')
    plt.close()