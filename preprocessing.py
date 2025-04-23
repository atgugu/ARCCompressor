import copy
import json
import numpy as np
import torch
import multitensor_systems

np.random.seed(0)
torch.manual_seed(0)

class Task:
    """
    A class that helps deal with ARC tasks: preprocessing, shape handling, solution processing.
    Supports optional data augmentations (rotations, flips, and their combinations) applied to
    training examples by concatenation and deduplication.
    Initializes the multitensor system for model construction.
    """
    def __init__(self, task_name, problem, solution=None, transforms='all'):
        """
        Args:
            task_name (str): ARC task identifier.
            problem (dict): 'train'/'test' splits with examples.
            solution (list of grids, optional): True outputs for test grids.
            transforms ("all" or callable or list): "all" for full symmetry group (rotations,
                flips, combinations), or custom transform(s) to augment training data.
        """
        self.task_name = task_name
        problem = copy.deepcopy(problem)

        # Determine transform functions
        if transforms == 'all':
            funcs = Task.get_all_transforms()
        elif transforms is None:
            funcs = []
        else:
            funcs = transforms if isinstance(transforms, (list, tuple)) else [transforms]

        # Augment training examples
        original_train = problem['train']
        augmented = []
        for ex in original_train:
            if 'input' in ex and 'output' in ex:
                for tf in funcs:
                        new_ex = copy.deepcopy(ex)
                        new_ex['input'] = tf(new_ex['input'])
                        new_ex['output'] = tf(new_ex['output'])
                        augmented.append(new_ex)
        train_all = original_train + augmented
        
        # Deduplicate examples
        unique, seen = [], set()
        for ex in train_all:
            inp_key = tuple(tuple(row) for row in ex['input'])
            out_key = tuple(tuple(row) for row in ex.get('output', [])) if 'output' in ex else None
            key = (inp_key, out_key)
            if key not in seen:
                seen.add(key)
                unique.append(ex)
        problem['train'] = unique

        # Counts
        self.n_train = len(problem['train'])
        self.n_test = len(problem['test'])
        self.n_examples = self.n_train + self.n_test
        self.unprocessed_problem = problem

        # Shapes and predictions
        self.shapes = self._collect_problem_shapes(problem)
        self._predict_solution_shapes()

        # Build system
        self._construct_multitensor_system(problem)
        self._compute_mask()
        self._create_problem_tensor(problem)

        # Solutions
        self.solution = self._create_solution_tensor(solution) if solution is not None else None

    @staticmethod
    def get_all_transforms():
        """
        Return a list of all non-identity symmetry transforms: rotations (90,180,270),
        flips (diag, anti-diag), and their combinations (rotate+flip).
        """
        transforms = []
        # Rotations
        for k in [1, 2, 3]:
            transforms.append(lambda g, k=k: Task.rotate_grid(g, k))
        # Flips
        transforms.append(Task.flip_diagonal)
        transforms.append(Task.flip_antidiagonal)
        # Combination: rotate then flip
        for k in [1, 2, 3]:
            transforms.append(lambda g, k=k: Task.flip_diagonal(Task.rotate_grid(g, k)))
            transforms.append(lambda g, k=k: Task.flip_antidiagonal(Task.rotate_grid(g, k)))
        return transforms

    @staticmethod
    def rotate_grid(grid, k=1):
        """Rotate a 2D grid by 90*k degrees clockwise."""
        arr = np.array(grid)
        return np.rot90(arr, -k).tolist()

    @staticmethod
    def flip_diagonal(grid):
        """Flip a 2D grid over its main diagonal (transpose)."""
        arr = np.array(grid)
        return arr.T.tolist()

    @staticmethod
    def flip_antidiagonal(grid):
        """Flip a 2D grid over its anti-diagonal."""
        arr = np.array(grid)
        return np.rot90(arr, 2).T.tolist()
    
    @staticmethod
    def get_random_color_transforms(colors, n_permutations=1):
        """
        Return a list of callable transforms that permute the provided colors randomly.

        Args:
            colors (list[int]): The palette of colors in the task (e.g., task.colors).
            n_permutations (int): Number of random permutations to generate.

        Returns:
            List[Callable[[List[List[int]]], List[List[int]]]]: A list of functions mapping grids.
        """
        transforms = []
        for _ in range(n_permutations):
            permuted = list(colors)
            np.random.shuffle(permuted)
            mapping = {orig: new for orig, new in zip(colors, permuted)}
            # each transform applies the same mapping to input and output grids
            transforms.append(lambda grid, mapping=mapping: [[mapping[c] for c in row] for row in grid])
        return transforms

    def augment_color_examples(self, examples, n_permutations=1):
        """
        Generate new examples by applying random color permutations.

        Args:
            examples (list of dict): A list of examples, each with 'input' and 'output' grids.
            n_permutations (int): Number of random color mappings to apply per example.

        Returns:
            List[dict]: Augmented examples with permuted colors.
        """
        augmented = []
        # Create transform functions based on this task's color palette
        transforms = Task.get_random_color_transforms(self.colors, n_permutations)
        for ex in examples:
            for tf in transforms:
                if 'input' in ex and 'output' in ex:
                    new_ex = {
                        'input': tf(ex['input']),
                        'output': tf(ex['output'])
                    }
                augmented.append(new_ex)
        return augmented
    def _collect_problem_shapes(self, problem):
        shapes = []
        for split in ['train', 'test']:
            for ex in problem[split]:
                in_shape = list(np.array(ex['input']).shape)
                out_shape = list(np.array(ex['output']).shape) if 'output' in ex else None
                shapes.append([in_shape, out_shape])
        return shapes

    def _predict_solution_shapes(self):
        self.in_out_same_size = all(
            tuple(inp) == tuple(out) for inp, out in self.shapes[:self.n_train]
        )
        self.all_in_same_size = len({tuple(s[0]) for s in self.shapes}) == 1
        self.all_out_same_size = len({tuple(s[1]) for s in self.shapes if s[1]}) == 1

        if self.in_out_same_size:
            for s in self.shapes[self.n_train:]: s[1] = s[0]
        elif self.all_out_same_size:
            default = self.shapes[0][1]
            for s in self.shapes[self.n_train:]: s[1] = default
        else:
            mx, my = self._get_max_dimensions()
            for s in self.shapes[self.n_train:]: s[1] = [mx, my]

    def _get_max_dimensions(self):
        mx = my = 0
        for in_out in self.shapes:
            for shape in in_out:
                if shape:
                    mx = max(mx, shape[0]); my = max(my, shape[1])
        return mx, my

    def _construct_multitensor_system(self, problem):
        self.n_x = max(shape[i][0] for shape in self.shapes for i in range(2))
        self.n_y = max(shape[i][1] for shape in self.shapes for i in range(2))
        colors = set()
        for split in ['train', 'test']:
            for ex in problem[split]:
                for grid in [ex['input'], ex.get('output', [])]:
                    for row in grid: colors.update(row)
        colors.add(0)
        self.colors = sorted(colors)
        self.n_colors = len(self.colors) - 1
        self.multitensor_system = multitensor_systems.MultiTensorSystem(
            self.n_examples, self.n_colors, self.n_x, self.n_y, self
        )

    def _compute_mask(self):
        m = np.zeros((self.n_examples, self.n_x, self.n_y, 2))
        for i, (inp_shape, out_shape) in enumerate(self.shapes):
            for mode, shape in enumerate([inp_shape, out_shape]):
                if shape:
                    xm = np.arange(self.n_x) < shape[0]
                    ym = np.arange(self.n_y) < shape[1]
                    m[i, :, :, mode] = np.outer(xm, ym)
        self.masks = torch.tensor(m, dtype=torch.float32)

    def _create_problem_tensor(self, problem):
        arr = np.zeros((self.n_examples, self.n_colors+1, self.n_x, self.n_y, 2))
        for split, cnt in [('train', self.n_train), ('test', self.n_test)]:
            for i, ex in enumerate(problem[split]):
                idx = i if split=='train' else self.n_train+i
                for mode in ['input','output']:
                    if split=='test' and mode=='output': continue
                    grid = ex.get(mode, np.zeros(self.shapes[idx][1]))
                    gt = self._create_grid_tensor(grid)
                    m = 0 if mode=='input' else 1
                    arr[idx, :, :gt.shape[1], :gt.shape[2], m] = gt
        self.problem = torch.from_numpy(np.argmax(arr,axis=1)).to(torch.get_default_device())

    def _create_grid_tensor(self, grid):
        arr = np.array(grid)
        tensor = np.zeros((self.n_colors+1, arr.shape[0], arr.shape[1]))
        for ci, col in enumerate(self.colors): tensor[ci] = (arr==col)
        return tensor

    def _create_solution_tensor(self, solution):
        tensor = np.zeros((self.n_test, self.n_colors+1, self.n_x, self.n_y))
        hashes = []
        for i, grid in enumerate(solution):
            hashes.append(tuple(map(tuple,grid)))
            gt = self._create_grid_tensor(grid)
            mx = min(gt.shape[1], self.n_x)
            my = min(gt.shape[2], self.n_y)
            tensor[i,:,:mx,:my] = gt[:,:mx,:my]
        self.solution_hash = hash(tuple(hashes))
        return torch.from_numpy(np.argmax(tensor,axis=1)).to(torch.get_default_device())


def preprocess_tasks(split, task_ids, transforms=None):
    """
    Load ARC tasks by split and apply optional augmentations.
    """
    with open(f'dataset/arc-agi_{split}_challenges.json') as f:
        problems = json.load(f)
    sols = None
    if split!='test':
        with open(f'dataset/arc-agi_{split}_solutions.json') as f:
            sols = json.load(f)
    tasks = []
    for idx,name in enumerate(problems):
        if name in task_ids or idx in task_ids:
            sol = sols.get(name) if sols else None
            tasks.append(Task(name, problems[name], sol, transforms))
    return tasks
