import argparse
import os

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset


# Each worker is a separate Python process with its own copy of the model.
# Start the workers together using torchrun, as shown in the README.
def main():
    # Read where to train and how many passes to make over the training data.
    parser = argparse.ArgumentParser(description="Train a tiny model using CPU/Gloo or GPU/NCCL.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()
    # Reject invalid settings before starting communication between workers.
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if "LOCAL_RANK" not in os.environ:
        parser.error("Launch with torchrun --standalone --nproc-per-node=2 distributed_training.py")

    # LOCAL_RANK is this worker's number on its own computer: 0, 1, and so on.
    # torchrun sets this value automatically when it starts the worker processes.
    local_rank = int(os.environ["LOCAL_RANK"])
    use_cuda = args.device == "cuda"
    if use_cuda:
        # Give each worker a different GPU on the same computer.
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        # Use one CPU compute thread per worker to avoid competing for every core.
        torch.set_num_threads(1)
        device = torch.device("cpu")
    # Connect all workers. NCCL communicates GPU tensors; Gloo communicates CPU tensors.
    dist.init_process_group(backend="nccl" if use_cuda else "gloo")

    try:
        # rank identifies this worker across all computers, starting from 0.
        # world_size is the total number of workers across all computers.
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        # The same seed makes every worker generate the same starting dataset.
        torch.manual_seed(42)

        # Create 1024 examples, each with four input values and one expected answer.
        x = torch.randn(1024, 4)
        true_weights = torch.tensor([[1.0], [-2.0], [0.5], [3.0]])
        # @ multiplies matrices: the model will learn these weights and the bias 0.7.
        y = x @ true_weights + 0.7
        # Keep each input row paired with its expected answer.
        dataset = TensorDataset(x, y)
        # Give each worker a different part of the same dataset.
        # Drop leftover examples so all workers receive equally sized parts.
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                     shuffle=True, drop_last=True)
        # Read 32 examples at a time. Pin CPU memory when copying batches to a GPU.
        loader = DataLoader(dataset, batch_size=32, sampler=sampler, pin_memory=use_cuda)
        if len(loader) == 0:
            raise ValueError("There must be at least one training sample per worker")

        # A linear layer turns four inputs into one prediction using weights and a bias.
        model = nn.Linear(4, 1).to(device)
        # DDP starts model copies with the same parameters and combines their gradients.
        model = DistributedDataParallel(model, device_ids=[local_rank] if use_cuda else None)
        # SGD adjusts the parameters; lr controls the size of each adjustment.
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        # Measure prediction error by averaging the squared differences from the answers.
        loss_function = nn.MSELoss()

        for epoch in range(args.epochs):
            # Use a new shuffle each epoch while keeping the workers' data splits coordinated.
            sampler.set_epoch(epoch)
            # statistics[0] stores total error; statistics[1] counts processed examples.
            statistics = torch.zeros(2, device=device)
            for features, targets in loader:
                # Place this batch on the chosen device. Pinned GPU copies can be asynchronous.
                features = features.to(device, non_blocking=use_cuda)
                targets = targets.to(device, non_blocking=use_cuda)
                # Clear the previous batch's gradients so they do not accumulate.
                optimizer.zero_grad(set_to_none=True)
                # Predict the answers and calculate how far they are from the targets.
                loss = loss_function(model(features), targets)
                # Compute gradients: how each parameter affects the error.
                loss.backward()  # DDP also averages these gradients across workers.
                optimizer.step()  # All workers then apply the same parameter update.
                # detach keeps these logging values out of the gradient calculation.
                # Convert the batch's average error into a total before adding it.
                statistics[0] += loss.detach() * features.size(0)
                statistics[1] += features.size(0)

            # Add error totals and example counts from all workers for reporting.
            # Gradient averaging was already handled by DDP during backward().
            dist.all_reduce(statistics, op=dist.ReduceOp.SUM)
            # Print once, from worker 0, instead of repeating the same message everywhere.
            if rank == 0:
                mean_loss = (statistics[0] / statistics[1]).item()
                print("Epoch", epoch + 1, "global mean loss:", mean_loss, flush=True)

        if rank == 0:
            # .module accesses the model inside DDP; .cpu().tolist() makes values printable.
            print("Learned weights:", model.module.weight.detach().cpu().tolist())
            print("Learned bias:", model.module.bias.detach().cpu().tolist())
    finally:
        # Release the workers' communication resources, including when an error occurs.
        dist.destroy_process_group()


# Run training when this file is executed, rather than when it is imported.
if __name__ == "__main__":
    main()