import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'  # specify which GPUs to use, e.g. '0,1' for two GPUs
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
num_samples = 100 # number of samples to draw
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

model_name = 'ckpt_rope_l1_init_345_loss_mask_400.pt'
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

tokenizer = Tokenizer()
outfile = open(f'log/test_acc_{model_name}.txt', 'w')

for dataset_val, num_samples in zip(['bool_logic_dataset_val_d1_v1.pkl', 'bool_logic_dataset_val_d2_v1.pkl', 'bool_logic_dataset_val_d3_v1.pkl',
                    'bool_logic_dataset_val_d4_v1.pkl', 'bool_logic_dataset_val_d5_v1.pkl', 'bool_logic_dataset_val_d6_v1.pkl'],
                    [16, 100, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]):
    print(f"Loading dataset: {dataset_val}")
    data_val = Dataset.load_dataset(filename=f'datasets/{dataset_val}')

    accurate = 0
    
    with torch.no_grad():
        with ctx:
            for k in range(num_samples):
                test_expression = data_val['expressions'][k]
                test_label = test_expression[-2:] # ->T or ->F

                outfile.write(f"Original expression: {test_expression}\n")
                test_expression = test_expression.split(Dataset.IMPLIES)[0] + Dataset.IMPLIES
                start_ids = tokenizer.tokenize(test_expression)

                x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                output = tokenizer.detokenize(y[0].tolist()).split('E')[0]

                pred_label = output[-2:] # ->T or ->F
                if pred_label == test_label:
                    accurate += 1

    outfile.write(f"Dataset: {dataset_val}\n")
    outfile.write(f"Test accuracy: {accurate}/{num_samples} = {accurate/num_samples*100:.2f}%\n")
    print(f"Dataset: {dataset_val}")
    print(f"Test accuracy: {accurate}/{num_samples} = {accurate/num_samples*100:.2f}%")

outfile.close()

