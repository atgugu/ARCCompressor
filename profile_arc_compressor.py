import torch
import cProfile
import pstats
import io

from arc_compressor import ARCCompressor
from multitensor_systems import MultiTensorSystem

# Set default device to cuda if available for ARCCompressor initialization
if torch.cuda.is_available():
    torch.set_default_device('cuda')
else:
    torch.set_default_device('cpu')
torch.manual_seed(0)

class DummyTask:
    def __init__(self, problem_tensor):
        self.problem = problem_tensor
        self.n_examples = problem_tensor.shape[0]
        self.n_x = problem_tensor.shape[1]
        self.n_y = problem_tensor.shape[2]

        # n_colors should be the max value in the problem tensor
        # Add 1 because colors are 0-indexed but n_colors is a count
        self.n_colors = problem_tensor.max().item() + 1

        self.multitensor_system = MultiTensorSystem(
            n_examples=self.n_examples,
            n_colors=self.n_colors,
            n_x=self.n_x,
            n_y=self.n_y,
            task=self
        )
        # Create dummy masks. Based on layers.py, masks seem to be boolean tensors.
        # The exact shape and meaning might need further investigation if this causes issues.
        # For now, let's assume they are related to problem dimensions.
        # Example: masks indicating valid areas, perhaps all True for a dummy task.
        # From layers.cummax and layers.shift, task.masks is used.
        # Let's create masks that allow all operations.
        # The shapes are (n_examples, n_x, 2) and (n_examples, n_y, 2)
        self.masks = {
            'x': torch.ones(self.n_examples, self.n_x, 2, dtype=torch.bool, device=self.problem.device),
            'y': torch.ones(self.n_examples, self.n_y, 2, dtype=torch.bool, device=self.problem.device)
        }
        # ARCCompressor also expects task.n_train for color permutation logic
        self.n_train = self.n_examples


def main():
    # Create a minimal dummy ARC task
    # problem shape: [n_examples, X, Y, 2] (input and output pairs)
    # n_colors derived from max value in problem
    dummy_problem_tensor = torch.randint(0, 3, (1, 5, 5, 2), device=torch.get_default_device())
    dummy_task = DummyTask(dummy_problem_tensor)

    # Initialize ARCCompressor
    arc_compressor = ARCCompressor(dummy_task)

    # Create profiler object
    profiler = cProfile.Profile()

    # Enable profiler
    profiler.enable()

    # Run the arc_compressor.forward() method a few times
    for _ in range(10):
        arc_compressor.forward()

    # Disable profiler
    profiler.disable()

    # Create string stream for stats
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)

    # Print the stats
    ps.print_stats()
    print(s.getvalue())

if __name__ == '__main__':
    main()
