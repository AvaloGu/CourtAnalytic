import time
from Models.ConvNeXt import ConvNeXt
from Models.Gru import EfficientGRUModel
from Models.ResNet_RS import ResNet_RS
from Classifiers.FrameClassifier import FrameClassifier
import torch
import os
import math
from Loaders.Dataloader import DataLoaderLite
from timm.utils import ModelEma


# -----------------------------------------------------------------------------
# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py

# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK']) # rank globally, 0 to world_size-1
    ddp_local_rank = int(os.environ['LOCAL_RANK']) # rank on this node, 0 to num_gpu_per_node-1
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}' # tells the process which GPU to use
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    # elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    #     device = "mps"
    print(f"using device: {device}")

device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(6216)
if torch.cuda.is_available():
    torch.cuda.manual_seed(6216)


batch_size = 16

# print("I am GPUv ", ddp_rank)
# import sys; sys.exit(0)

train_loader = DataLoaderLite(batch_size, process_rank=ddp_rank, num_processes=ddp_world_size)

# use tensor float32 precision for all matrix multiplications
# faster than the normal float32 matmul. But all variables (tensors) are still stored in float32
torch.set_float32_matmul_precision('high') 

ConvNet = ConvNeXt()
rnn = EfficientGRUModel()

# ConvNet = ResNet_RS()
# rnn = EfficientGRUModel(input_size=2048)

model = FrameClassifier(ConvNet, rnn)
model.to(device)

# TODO: turn it off for debugging and tunning
model = torch.compile(model) # compiler for Neural networks, compile the model to make it faster

optimizer = model.configure_optimizers(weight_decay = 0.05, learning_rate = 1e-2, device_type = device_type, master_process = master_process)

# create EMA wrapper, use it for validation
model_ema = ModelEma(model, decay=0.999, device='cpu')

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank]) # wrap the model into DDP container
# raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

# TODO
max_lr = 4e-3
min_lr = max_lr * 0.1
warmup_steps = 100
max_steps = 1000 # around 10 epochs

def get_lr(iter):
    # 1) linear warmup for warmup_iters steps
    if iter < warmup_steps:
        return max_lr * (iter+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if iter > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (iter - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

model.train()

gru_hidden = None

for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    optimizer.zero_grad()
    loss_accum = 0.0

    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
        
    # autocast context manager, it should only surround forward pass and loss calculation
    # the backward() call and optimizer step should be outside the autocast scope.
    # We use bfloat16 precision.
    # The mix precison of bfloat16 and float32 is handled automatically by PyTorch.
    # Not all operations are safe in bfloat16, e.g. softmax, so PyTorch will
    # automatically use float32 for those operations.
    # But the logits (activation before softmax), and matrix multiplies will be converted to bfloat16.
    # TLDR: only some layers selective will be running in BFloat16, other remain in Float32.
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        logits, h, loss = model(x, y, gru_hidden)
    # we have to scale the loss to account for gradient accumulation,
    # because the gradients just add on each successive backward().
    # addition of gradients corresponds to a SUM in the objective, but
    # instead of a SUM we want MEAN. Scale the loss here so it comes out right
    # loss = loss / grad_accum_steps
    # for now, we won't use gradient accumulation, because we make sure the batch size fits in memory
    gru_hidden = h.detach()
    loss_accum += loss.detach()
    loss.backward() 
    
    if ddp: 
        # average the loss across all processes for reporting
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG) 

    # gradient clipping, clip the global norm of the gradients at 1.0, prevent exploding gradients
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    lr = get_lr(step) 
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()

    raw_model = model.module if ddp else model # always contains the "raw" unwrapped model
    model_ema.update(raw_model)

    if device_type == "cuda":
        torch.cuda.synchronize() # wait for the GPU to finish work, measure time accurately

    t1 = time.time()
    dt = t1 - t0 # time differenece in seconds
    examples_processed = train_loader.B * ddp_world_size
    examples_per_sec = examples_processed / dt
    if master_process:
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | img/sec: {examples_per_sec:.2f}")
        # with open(log_file, "a") as f:
        #     f.write(f"{step} train {loss_accum.item():.6f}\n")

torch.save(model_ema.ema.state_dict(), "model_ema.pth")

if ddp:
    destroy_process_group()
        






