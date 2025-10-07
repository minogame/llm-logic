import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # specify which GPUs to use, e.g. '0,1' for two GPUs
import pickle
from contextlib import nullcontext
import torch
import time
import tiktoken
from model_rope import GPTConfig, GPT
from data_generator_bool import BoolLogic as Dataset, BoolLogicTokenizer as Tokenizer

init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 1 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 10 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337 + int(time.time())
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

model_name = 'ckpt_rope_l1_grpo_345_version2_a_100.pt'
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
model.load_state_dict(state_dict)

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

outfile = open(f'analysis_a_{model_name}_on_{dataset_val}_samples_{num_samples}.txt', 'w')
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

# 清空本次 sample 的 attention 记录
for lst in attn_records.values():
    lst.clear()

# 评估模型在验证集上的 PPL
model.eval()

X = tokenizer.tokenize(output)
Y = X[1:] + [tokenizer.tokens[' ']]
X = torch.tensor(X, dtype=torch.long, device=device)[None, ...]
Y = torch.tensor(Y, dtype=torch.long, device=device)[None, ...]

with ctx:
    logits, loss, _ = model(X, Y, loss_mask=None, mode='pretrain')


# 将 hook 收集到的 attention 信息写入文件（简要摘要）
for layer_idx, recs in attn_records.items():
    if len(recs) == 0:
        outfile.write(f"Layer {layer_idx}: no attn records\n")
    else:
        # recs 可能包含多次 forward 的 attention，这里写入最后一次的 shape 和简单统计
        last_att = recs[-1]  # Tensor on CPU
        outfile.write(f"Layer {layer_idx}: records={len(recs)}, last_shape={tuple(last_att.shape)}, last_mean={float(last_att.mean()):.6f}\n")


outfile.close()

