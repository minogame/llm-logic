"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # specify which GPUs to use, e.g. '0,1' for two GPUs
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model_rope import GPTConfig, GPT
from data_generator_bool import BoolLogic, BoolLogicTokenizer
from scipy.stats import norm

exp_name = 'rope_l2_grpo_3_rs_version1_a'
# open log file in append mode so existing logs are preserved; we'll write a run timestamp header once per run
log_file = open(f'train_log/{exp_name}.log', 'a')
import sys
from datetime import datetime

# unified logger: write to both stdout and the log file
def logger(msg='', end='\n', to_stdout=True, file=log_file):
    text = str(msg)
    if to_stdout:
        sys.stdout.write(text + end)
        try:
            sys.stdout.flush()
        except Exception:
            pass
    if file is not None:
        try:
            file.write(text + end)
            file.flush()
        except Exception:
            pass

# alias for convenience
log = logger

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
ckpt_path = os.path.join(out_dir, 'ckpt_rope_l2_init_3_loss_mask_10_version1_20.pt')
eval_interval = 50
save_interval = 50
log_interval = 1
eval_iters = 10000 // 64
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 12 # used to simulate larger batch sizes
batch_size = 24 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1536
num_answers = 16
max_new_tokens = 512
temperature = 1.0 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 10 # retain only the top_k most likely tokens, clamp others to have 0 probability
# model
n_layer = 2
n_head = 6
n_embd = 384
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
attn_T = False
# adamw optimizer
learning_rate = 6e-6 # max learning rate
max_iters = 5000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 0.1 # clip gradients at this value, or disable if == 0.0
kl_beta = 0 # weight of kl divergence loss
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 0 # how many steps to warm up for
lr_decay_iters = 5000 # should be ~= max_iters per Chinchilla
min_lr = 6e-7 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size

# write a single timestamp header at the start of this run (only master process should log it)
try:
    if master_process:
        run_ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # use the unified logger so the header goes to both stdout and the log file
        logger(f"\n=== RUN START: {run_ts} ===")
except Exception:
    # non-fatal: if writing header fails, continue
    pass

logger(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(int(time.time() * 1000) + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# get_batch = None # set in data_loader.py
# exec(open('data_loader.py').read())
# poor man's data loader
from data_generator_bool import BoolLogic as Dataset, BoolLogicTokenizer as Tokenizer
tokenizer = Tokenizer()
data_train = Dataset.load_dataset(filename='datasets/bool_logic_dataset_train.pkl') #
# data_train = Dataset.load_dataset(filename='datasets_sampling/bool_logic_dataset_train_345_grpo_sampling_merged.pkl') #
data_val = Dataset.load_dataset(filename='datasets/bool_logic_dataset_val_d6_v1.pkl')
data_train['offset'] = 0
data_val['offset'] = 0

def get_batch_val():
    data = data_val

    x_to_stack = []
    y_to_stack = []
    loss_mask_to_stack = []
    for idx in range(batch_size):
        expression_idx = data['offset'] + idx
        tokenized_expression = data['tokenized_expressions'][expression_idx][:]
        if len(tokenized_expression) > block_size:
            # truncate the expression to fit into the block size
            tokenized_expression = tokenized_expression[-block_size:]
            target_expression = tokenized_expression[1:] + [tokenizer.tokens[' ']]  # shift by one for target
        else:
            # pad the expression to fit into the block size
            tokenized_expression += [tokenizer.tokens[' ']] * (block_size - len(tokenized_expression))
            target_expression = tokenized_expression[1:] + [tokenizer.tokens[' ']]  # shift by one for target

        loss_mask = [0] * data['pos_implies'][expression_idx] + \
                    [0] * (len(data['tokenized_expressions'][expression_idx]) - data['pos_implies'][expression_idx] - 4) + \
                    [1] * 3 + \
                    [0] * (block_size - len(data['tokenized_expressions'][expression_idx]) + 1)
        loss_mask = loss_mask[:block_size]  # make sure loss_mask is of length block_size

        loss_mask_to_stack.append(torch.tensor(loss_mask, dtype=torch.float32))
        x_to_stack.append(torch.from_numpy(np.array(tokenized_expression, dtype=np.int64)))
        y_to_stack.append(torch.from_numpy(np.array(target_expression, dtype=np.int64)))

    x = torch.stack(x_to_stack)
    y = torch.stack(y_to_stack)
    loss_mask = torch.stack(loss_mask_to_stack)
    x, y, loss_mask = x.to(device), y.to(device), loss_mask.to(device)

    data['offset'] += batch_size
    if data['offset'] + batch_size > len(data['tokenized_expressions']):
        # reset the offset to 0, so we can loop over the dataset
        data['offset'] = 0
        logger(f"Resetting dataset offset to 0")

    return x, y, loss_mask

def get_batch_train_ori():
    data = data_train

    len_data = len(data['tokenized_expressions'])

    x_to_stack = []
    y_to_stack = []
    loss_mask_to_stack = []
    for idx in range(batch_size):
        expression_idx = (data['offset'] + idx) % len_data
        tokenized_expression = data['tokenized_expressions'][expression_idx][:]
        if len(tokenized_expression) > block_size:
            # truncate the expression to fit into the block size
            tokenized_expression = tokenized_expression[-block_size:]
            target_expression = tokenized_expression[1:] + [tokenizer.tokens[' ']]  # shift by one for target
        else:
            # pad the expression to fit into the block size
            tokenized_expression += [tokenizer.tokens[' ']] * (block_size - len(tokenized_expression))
            target_expression = tokenized_expression[1:] + [tokenizer.tokens[' ']]  # shift by one for target

        loss_mask = [0] * data['pos_implies'][expression_idx] + \
                    [0] * (len(data['tokenized_expressions'][expression_idx]) - data['pos_implies'][expression_idx] - 4) + \
                    [1] * 3 + \
                    [0] * (block_size - len(data['tokenized_expressions'][expression_idx]) + 1)

        loss_mask_to_stack.append(torch.tensor(loss_mask, dtype=torch.float32))
        x_to_stack.append(torch.from_numpy(np.array(tokenized_expression, dtype=np.int64)))
        y_to_stack.append(torch.from_numpy(np.array(target_expression, dtype=np.int64)))

    x = torch.stack(x_to_stack)
    y = torch.stack(y_to_stack)
    loss_mask = torch.stack(loss_mask_to_stack)
    x, y, loss_mask = x.to(device), y.to(device), loss_mask.to(device)

    return x, y, loss_mask

def get_batch_train(actor_model, micro_step=0):
    data = data_train

    x_to_stack = []
    y_to_stack = []
    mask_to_stack = []
    score_to_stack = []
    label_to_stack = []

    for idx in range(batch_size):
        expression_idx = data['offset'] + idx

        prompt = data['expressions'][expression_idx][:]
        label = data['expressions'][expression_idx][-1]
        start_ids = tokenizer.tokenize(prompt.split(Dataset.IMPLIES)[0] + Dataset.IMPLIES)
        x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

        # duplicate x to have batch size of num_answers
        x = x.repeat(num_answers, 1)
        y = actor_model.batch_generate(x, max_new_tokens, temperature=temperature, top_k=top_k).detach().cpu().numpy()

        sampled_exprs = tuple([ tokenizer.detokenize(y[i].tolist()).split('E')[0] for i in range(num_answers) ])
        pos_sampled_exprs_answer = tuple([ (len(expr)-2, tokenizer.tokens[label]) for expr in sampled_exprs ])  # position of the answer (last character before 'E')

        if master_process and micro_step == 0 and idx % batch_size == 0:
            logger(f"Generating samples for expression {expression_idx}: \n{prompt.split(Dataset.IMPLIES)[0] + Dataset.IMPLIES} -> \n", to_stdout=False)
            logger(f"    ground truth: {data['expressions'][expression_idx]}", to_stdout=False)
            for expr in sampled_exprs:
                logger(f"    {expr}", to_stdout=False)
            logger("", to_stdout=False)
            logger(f"    sample diversity: {[len(set(expr)) for expr in sampled_exprs]}", to_stdout=False)
            logger("", to_stdout=False)
            logger(f"    positions: {pos_sampled_exprs_answer}", to_stdout=False)
            logger("", to_stdout=False)

        score = tuple([ Dataset.evaluate_expression(expr, data['expressions'][expression_idx], version=2) for expr in sampled_exprs ])
        first_implies = y[0].tolist().index(tokenizer.tokens[BoolLogic.IMPLIES])
        # add EOS
        sampled_exprs = tuple([ expr + 'E' if not expr.endswith('E') else expr for expr in sampled_exprs ])
        tokenized_sampled_expressions = tuple([ tokenizer.tokenize(expr) for expr in sampled_exprs ])

        exprs, targets, masks = [], [], []
        for expr_idx, expr in enumerate(tokenized_sampled_expressions):
            if len(expr) > block_size:
                # truncate the expression to fit into the block size
                expr = expr[-block_size:]
            else:
                # pad the expression to fit into the block size
                expr += [tokenizer.tokens[' ']] * (block_size - len(expr))

            target = expr[1:] + [tokenizer.tokens[' ']]  # shift by one for target

            mask =  [0] * first_implies + \
                    [0.9] * (len(expr) - first_implies - 4) + \
                    [1] * 3 + \
                    [0] * (block_size - len(expr) + 1)
            
            mask = mask[:block_size]  # make sure loss_mask is of length block_size

            exprs.append(torch.from_numpy(np.array(expr, dtype=np.int64)))
            targets.append(torch.from_numpy(np.array(target, dtype=np.int64)))
            masks.append(torch.tensor(mask, dtype=torch.float32))

        x_to_stack.append(torch.stack(exprs))
        y_to_stack.append(torch.stack(targets))
        mask_to_stack.append(torch.stack(masks))
        score_to_stack.append(score)
        label_to_stack.append(pos_sampled_exprs_answer)

    x_to_stack = torch.stack(x_to_stack) # (B, G, T)
    y_to_stack = torch.stack(y_to_stack) # (B, G, T)
    mask_to_stack = torch.stack(mask_to_stack) # (B, G, T)
    label_to_stack = torch.tensor(label_to_stack, dtype=torch.long) # (B, G, 2)

    data['offset'] += batch_size
    if data['offset'] + batch_size > len(data['tokenized_expressions']):
        # reset the offset to 0, so we can loop over the dataset
        data['offset'] = 0

    return x_to_stack, y_to_stack, mask_to_stack, score_to_stack, label_to_stack

def get_batch(split, actor_model, micro_step=0):

    if split == 'train':
        return get_batch_train(actor_model, micro_step=micro_step)
    elif split == 'val':
        return get_batch_val()
    elif split == 'train_ori':
        return get_batch_train_ori()


iter_num = 0
best_val_loss = 1e9

# model init: create an actor (trainable) and an actor_ref (frozen, not trained) copy
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout, attn_T=attn_T) # start with model_args from command line
logger(f"Loading actor from {ckpt_path}")
checkpoint = torch.load(ckpt_path, map_location=device)
checkpoint_model_args = checkpoint['model_args']
# force these config attributes to be equal otherwise we can't even resume training
for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
    model_args[k] = checkpoint_model_args[k]
# create the actor and load weights
gptconf = GPTConfig(**model_args)
actor = GPT(gptconf)
state_dict = checkpoint['model']
# fix the keys of the state dictionary if needed
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
actor.load_state_dict(state_dict)

# use 'model' name for the training actor to keep the rest of the script unchanged
model = actor
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
# optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    logger("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train_ori', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y, loss_mask = get_batch(split)
            with ctx:
                logits, loss, _ = model(X, Y, loss_mask=loss_mask, mode='pretrain')
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# training loop
X, Y, masks, scores, label = get_batch('train', actor) # fetch the very first batch

t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0

while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num > 0 and iter_num % eval_interval == 0 and master_process:
        # losses = estimate_loss()
        # logger(f"step {iter_num}: train_ori loss {losses['train_ori']:.4f}, val loss {losses['val']:.4f}")

        checkpoint = {
            'model': raw_model.state_dict(), 'optimizer': optimizer.state_dict(),
            'model_args': model_args, 'iter_num': iter_num,
            'best_val_loss': best_val_loss, 'config': config,
        }
        if (iter_num) % save_interval == 0:
            logger(f"saving checkpoint to {out_dir}")
            torch.save(checkpoint, os.path.join(out_dir, f'ckpt_{exp_name}_{iter_num}.pt'))
            del checkpoint 


    if iter_num == 0 and eval_only: break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            # group_sequences, group_rewards, actor_ref, kl_beta, behavior_log_probs=None, use_is=False
            loss, policy_loss, kl_divergence = model(X, Y, masks, group_rewards=scores, group_labels=label, mode='reject_sampling')
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y, masks, scores, label = get_batch('train', actor, micro_step)
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps

        if isinstance(policy_loss, torch.Tensor):
            policy_loss_raw = float(policy_loss.item())
            kl_div_raw = float(kl_divergence.item())
        else:
            policy_loss_raw = 0.0
            kl_div_raw = 0.0

        # initialize EMA accumulators on first use
        if 'ema_policy_loss' not in globals() or ema_policy_loss is None:
            ema_policy_loss = policy_loss_raw
        if 'ema_kl_div' not in globals() or ema_kl_div is None:
            ema_kl_div = kl_div_raw

        # EMA smoothing factor (alpha): higher -> more weight to recent values
        ema_alpha = 0.1

        # update exponential moving averages
        ema_policy_loss = ema_alpha * policy_loss_raw + (1.0 - ema_alpha) * ema_policy_loss
        ema_kl_div = ema_alpha * kl_div_raw + (1.0 - ema_alpha) * ema_kl_div

        # use EMA values for logging / downstream computations
        policy_lossf = ema_policy_loss
        kl_divf = ema_kl_div

        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu

        mu = batch_size*num_answers * 0.5
        sigma = math.sqrt(batch_size*num_answers*0.25)
        x_corrected = kl_divf - 0.5
        z_score = (x_corrected - mu) / sigma
        p_value = norm.sf(z_score)

        acc_raw = kl_div_raw / batch_size / num_answers
        acc = kl_divf / batch_size / num_answers

        log_info = f"iter {iter_num}: loss {lossf:.4f}, lr {lr:.2e}, "
        log_info += f"rewards {policy_loss_raw:.4f} (ema {policy_lossf:.4f}), "
        log_info += f"acc {acc_raw:.4f} (ema {acc:.4f}), "
        log_info += f"p_value {p_value:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%"
        logger(log_info)

    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()

k_s = list(globals().keys())

for k in k_s:
    if k in globals().keys():
        if not k.startswith('_') and isinstance(globals()[k], (int, float, str, bool)):
            logger(f"{k}: {globals()[k]}")

log_file.close()