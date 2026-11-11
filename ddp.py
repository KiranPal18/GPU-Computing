# Run with one process per GPU:  torchrun --nproc_per_node=<num_gpus> ddp.py
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset


def main():
    # torchrun sets these for every process it starts.
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Each process drives one GPU, like each std::thread in multi_gpu.cu.
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)

    # Toy data: learn y = 3x + 2.
    n = 1024
    x = torch.linspace(-1, 1, n).unsqueeze(1)
    y = 3 * x + 2
    dataset = TensorDataset(x, y)

    # DistributedSampler gives each rank a different slice of the dataset.
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(dataset, batch_size=64, sampler=sampler)
    print(f"[rank {rank}] GPU {local_rank} gets {len(sampler)} of {n} samples")

    # DDP keeps a model copy per GPU and all-reduces gradients during backward.
    model = DDP(nn.Linear(1, 1).to(device), device_ids=[local_rank])
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()

    for epoch in range(20):
        sampler.set_epoch(epoch)  # reshuffle differently every epoch
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()  # gradients are averaged across all GPUs here
            opt.step()

        # Average the loss over all ranks just for printing.
        avg = loss.detach().clone()
        dist.all_reduce(avg, op=dist.ReduceOp.AVG)
        if rank == 0:
            print(f"epoch {epoch:2d}  loss {avg.item():.5f}")

    # Every rank holds identical weights because gradients were synced.
    w, b = model.module.weight.item(), model.module.bias.item()
    print(f"[rank {rank}] w={w:.4f} b={b:.4f} (target 3, 2)")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
