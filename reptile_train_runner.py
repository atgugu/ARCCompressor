import json
import os
import torch
from reptile_train import train_reptile

def main():
    # Set reproducibility
    torch.manual_seed(0)

    # Path to ARC-AGI training data
        # Find all the puzzle names
    split = "training"
    dataset_path = f'/home/atgu/Desktop/ARCCompressor/2025data/arc-agi_{split}_challenges.json'
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Could not find training dataset at {dataset_path}")

    # Load ARC-AGI training problems
    with open(dataset_path, "r") as f:
        problem_data = json.load(f)

    task_names = list(problem_data.keys())
    print(f"Loaded {len(task_names)} training tasks.")

    # Train meta-model using Reptile
    print("Starting Reptile meta-training...")
    trained_meta_model = train_reptile(
        problem_data=problem_data,
        task_names=task_names,
        meta_iters=1000,
        inner_iters=5,
        meta_lr=0.1
    )

    # Save meta weights
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    weights_path = os.path.join(save_dir, "meta_weights.pt")
    torch.save([w.detach().cpu() for w in trained_meta_model.weights_list], weights_path)
    print(f"Saved trained meta-weights to {weights_path}")

if __name__ == "__main__":
    main()
