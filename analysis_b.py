import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # specify which GPUs to use, e.g. '0,1' for two GPUs
import pickle
from contextlib import nullcontext
import torch
import time
import tiktoken
from model_rope_attn import GPTConfig, GPT
from data_generator_bool import BoolLogic as Dataset, BoolLogicTokenizer as Tokenizer
import math
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize, linewidth=2000, precision=4, suppress=True)
import matplotlib.pyplot as plt


init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 1 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 10 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337 + int(time.time())
block_size = 1536
device = 'cuda:0' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float32' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

model_name = 'ckpt_rope_l2_grpo_345_version3_b_50.pt'
# model_name = 'ckpt_rope_l1_init_345_loss_mask_10_version3_100.pt'
# model_name = 'ckpt_rope_l1_mixed_x6_1000.pt'
# init from a model saved in a specific directory
ckpt_path = os.path.join(out_dir, model_name)
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict, strict=False)

model.eval()
model.to(device)

# ---- 新增：注册 attention hook 并收集注意力（每个 transformer block 一个列表） ----
attn_records = {}
try:
    n_blocks = len(model.transformer.h)
except Exception:
    n_blocks = 0

for i in range(n_blocks):
    attn_records[i] = []

def make_hook(layer_idx):
    def hook(att):
        # att: Tensor with shape (B, nh, T, T) (detached in module)
        try:
            att_cpu = att.detach().cpu()  # 保持 tensor，后续可以分析或转换为 numpy
        except Exception:
            att_cpu = att.clone().detach().cpu()
        attn_records[layer_idx].append(att_cpu)
    return hook

for i in range(n_blocks):
    try:
        model.transformer.h[i].attn.register_attn_hook(make_hook(i))
    except Exception:
        # 如果某个 block 不存在 attn 属性则跳过
        pass
# ---------------------------------------------------------------------

tokenizer = Tokenizer()

# 'bool_logic_dataset_val_d1_v1.pkl'
# 'bool_logic_dataset_val_d2_v1.pkl'
# 'bool_logic_dataset_val_d3_v1.pkl'
# 'bool_logic_dataset_val_d4_v1.pkl'
# 'bool_logic_dataset_val_d5_v1.pkl'
# 'bool_logic_dataset_val_d6_v1.pkl'

dataset_val = 'bool_logic_dataset_val_d4_v1.pkl'
data_val = Dataset.load_dataset(filename=f'datasets/{dataset_val}')

outfile = open(f'analysis/analysis_b_{model_name}_on_{dataset_val}_samples_{num_samples}.txt', 'w')
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            test_expression = data_val['expressions'][k]
            outfile.write(f"Original expression: {test_expression}\n")
            test_expression = test_expression.split(Dataset.IMPLIES)[0] + Dataset.IMPLIES
            start_ids = tokenizer.tokenize(test_expression)

            x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            output = tokenizer.detokenize(y[0].tolist()).split('E')[0]
            outfile.write(f"Detokenized outputs: {output}\n")

            outfile.write('---------------\n')


def plot_softmax_heatmap(softmax_logits, out_png):
    # convert to numpy if needed
    if isinstance(softmax_logits, np.ndarray):
        soft = softmax_logits
    else:
        try:
            soft = softmax_logits.numpy()
        except Exception:
            soft = softmax_logits.detach().cpu().numpy()

    # soft is (T, V)
    T, V = soft.shape

    # if vocab is huge, truncate for a simple plot to keep image readable
    max_cols = 400
    truncated = False
    if V > max_cols:
        soft_plot = soft[:, :max_cols]  # keep first max_cols vocab cols
        truncated = True
        V_display = max_cols
    else:
        soft_plot = soft
        V_display = V

    # we want x-axis = time step (T), y-axis = vocab index (V)
    # imshow expects array with shape (rows, cols) -> (y, x) => transpose soft_plot
    img = soft_plot.T  # now shape is (V_display, T) => rows=vocab, cols=time

    # figure sizing: width depends on time steps, height on vocab indices
    fig_w = min(20, max(6, T / 40))
    fig_h = min(10, max(3, V_display / 40))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(img, cmap='viridis', aspect='auto', origin='lower')
    ax.set_xlabel('time step')
    ax.set_ylabel('vocab index')
    title = f"Softmax heatmap (vocab x time = {V_display}x{T})"
    if truncated:
        title += f" - vocab truncated to first {max_cols} cols"
    ax.set_title(title, fontsize=10)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

def plot_attention_heatmap(last_att, out_png):

    att_np = last_att.numpy() if not isinstance(last_att, np.ndarray) else last_att
    B, nh, T, _ = att_np.shape

    # choose batch 0 for plotting
    att_batch = att_np[0]

    ncols = min(4, nh)
    nrows = math.ceil(nh / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.2))
    axes = np.array(axes).reshape(-1)

    vmax = float(att_batch.max())
    vmin = float(att_batch.min())

    for h in range(nh):
        ax = axes[h]
        im = ax.imshow(att_batch[h], cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_title(f"head {h}", fontsize=8)
        ax.set_xlabel("key")
        ax.set_ylabel("query")
        ax.tick_params(axis='both', which='major', labelsize=6)
        # small colorbar per head (optional)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    # hide any unused subplots
    for j in range(nh, axes.size):
        axes[j].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# 清空本次 sample 的 attention 记录
for lst in attn_records.values():
    lst.clear()

X = tokenizer.tokenize(output)
# Y = X[1:] + [tokenizer.tokens[' ']]
X = torch.tensor(X, dtype=torch.long, device=device)[None, ...]
# Y = torch.tensor(Y, dtype=torch.long, device=device)[None, ...]

with ctx:
    logits, softmax_logits = model(X, bias=None, mode='logits')

# 将 hook 收集到的 attention 信息写入文件（简要摘要）
for layer_idx, recs in attn_records.items():
    if len(recs) == 0:
        outfile.write(f"Layer {layer_idx}: no attn records\n")
    else:
        # recs 可能包含多次 forward 的 attention，这里写入最后一次的 shape 和简单统计
        last_att = recs[-1]  # Tensor on CPU
        outfile.write(f"Layer {layer_idx}: records={len(recs)}, last_shape={tuple(last_att.shape)}, last_mean={float(last_att.mean()):.6f}\n")

        attn_png = f"analysis/analysis_b_{model_name}_on_{dataset_val}_layer_{layer_idx}_heads.png"
        plot_attention_heatmap(last_att, attn_png)

softmax_png = f"analysis/analysis_b_{model_name}_on_{dataset_val}_softmax_heatmap.png"
plot_softmax_heatmap(softmax_logits[0].detach().cpu(), softmax_png)
outfile.write(f"-2 token softmax bias {softmax_logits[0, -5]}\n")
outfile.write(f"-1 token softmax bias {softmax_logits[0, -4]}\n")
outfile.write(f"IMPLIES softmax {softmax_logits[0, -3]}\n")
outfile.write(f"ANSWER softmax {softmax_logits[0, -2]}\n")
outfile.write(f"EOS softmax {softmax_logits[0, -1]}\n")

# 清空本次 sample 的 attention 记录
for lst in attn_records.values():
    lst.clear()

bias = torch.tril(torch.ones(block_size, block_size))
len_prompt = len(start_ids)
eos_pos = len(X[0]) - 1  # position of EOS token in the sequence
bias[eos_pos-1, :len_prompt] = 0  # remove prompt attention from the answer token
bias = bias.view(1, 1, block_size, block_size).to(device=device)

with ctx:
    logits, softmax_logits = model(X, bias=bias, mode='logits')


# 将 hook 收集到的 attention 信息写入文件（简要摘要）
for layer_idx, recs in attn_records.items():
    if len(recs) == 0:
        outfile.write(f"Layer {layer_idx}: no attn records\n")
    else:
        # recs 可能包含多次 forward 的 attention，这里写入最后一次的 shape 和简单统计
        last_att = recs[-1]  # Tensor on CPU
        outfile.write(f"Layer {layer_idx}: records={len(recs)}, last_shape={tuple(last_att.shape)}, last_mean={float(last_att.mean()):.6f}\n")

        attn_png = f"analysis/analysis_b_{model_name}_on_{dataset_val}_layer_{layer_idx}_heads_bias.png"
        plot_attention_heatmap(last_att, attn_png)

softmax_png = f"analysis/analysis_b_{model_name}_on_{dataset_val}_softmax_heatmap_bias.png"
plot_softmax_heatmap(softmax_logits[0].detach().cpu(), softmax_png)
outfile.write(f"-2 token softmax bias {softmax_logits[0, -5]}\n")
outfile.write(f"-1 token softmax bias {softmax_logits[0, -4]}\n")
outfile.write(f"IMPLIES softmax bias {softmax_logits[0, -3]}\n")
outfile.write(f"ANSWER softmax bias {softmax_logits[0, -2]}\n")
outfile.write(f"EOS softmax bias {softmax_logits[0, -1]}\n")


outfile.close()

