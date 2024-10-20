import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch.distributed as dist

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from torch.distributed.fsdp.api import (
    ShardingStrategy,
)

def print_rank_0(str):
    if dist.get_rank() == 0:
        print(str)

class Net(nn.Module):
    def __init__(self, H):
        super(Net, self).__init__()
        self.fc0 = nn.Linear(H, H, bias=False)
        self.fc1 = nn.Linear(H, H, bias=False)
        self.fc2 = nn.Linear(H, H, bias=False)
        self.fc3 = nn.Linear(H, H, bias=False)
        self.fc4 = nn.Linear(H, H, bias=False)


        
    def forward(self, x):
        x = self.fc0(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x
    




def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12354'
    os.environ['RANK'] = str(rank)
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def traverse_model(model):
    print_rank_0(f"==== start traversing model: {model.__class__}")
    for name, param in model.named_parameters():
        print_rank_0(f"Parameter {name}: shape={param.shape}")

def fsdp_main(rank, world_size,):
    setup(rank, world_size)
    
    model = Net(256)
    
    print_rank_0("====== use_orig_params=False ======")
    fsdp_model = FSDP(
        model, use_orig_params=False, device_id=torch.cuda.current_device(),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
    )
    traverse_model(fsdp_model)
    
    print_rank_0("\n\n====== use_orig_params=False ======")
    model_2 = Net(256)
    fsdp_model_2 = FSDP(
        model_2, use_orig_params=True, device_id=torch.cuda.current_device(),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
    )
    traverse_model(fsdp_model_2)
    
    

if __name__ == '__main__':
    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(fsdp_main,
        args=(WORLD_SIZE,),
        nprocs=WORLD_SIZE,
        join=True)