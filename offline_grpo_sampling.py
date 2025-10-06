import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # specify which GPUs to use, e.g. '0,1' for two GPUs
import pickle
from contextlib import nullcontext
import torch
import time
import tiktoken
from model_rope import GPTConfig, GPT
from data_generator_bool import BoolLogic as Dataset, BoolLogicTokenizer as Tokenizer

model_name = 'ckpt_rope_l1_init_345_loss_mask_400.pt'
model_name = 'ckpt_rope_l1_offline_grpo_345_c_20.pt'
dataset_sampling_id = 0 # which dataset to use for sampling, from 0 to 99
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_answers = 8 # number of answers to generate for each sample
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 1.2 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 10 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337 + int(time.time())
device = 'cuda:0' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float32' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file

dataset_val = f'datasets_sampling/bool_logic_dataset_train_345_grpo_sampling_{dataset_sampling_id}.pkl'
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

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

print(f"Processing dataset: {dataset_val} :", end=' ', flush=True)
data_val = Dataset.load_dataset(filename=dataset_val)

# add new keys to data_val
data_val['sampled_expressions'] = []
data_val['tokenized_sampled_expressions'] = []
data_val['score'] = []
data_val['statistics'] = []

# Print all of those at the end
def print_results(data_val):

    print("Sampled Expressions:")
    for d in data_val['sampled_expressions']:
        print(d)
    # print(data_val['sampled_expressions'])
    # print("Tokenized Sampled Expressions:")
    # print(data_val['tokenized_sampled_expressions'])
    print("Score:")
    print(data_val['score'])
    print("Statistics:")
    print(data_val['statistics'])

# Call this after your sampling loop

num_samples = 1 # len(data_val['expressions'])

with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            test_expression = data_val['expressions'][k][:]
            test_expression = test_expression.split(Dataset.IMPLIES)[0] + Dataset.IMPLIES
            start_ids = tokenizer.tokenize(test_expression)

            x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
            # duplicate x to have batch size of num_answers
            x = x.repeat(num_answers, 1)
            y = model.batch_generate(x, max_new_tokens, temperature=temperature, top_k=top_k)

            sampled_exprs = tuple([ tokenizer.detokenize(y[i].tolist()).split('E')[0] for i in range(num_answers) ])
            data_val['sampled_expressions'].append(sampled_exprs)

            data_val['tokenized_sampled_expressions'].append(tuple([ tokenizer.tokenize(expr) for expr in sampled_exprs ]))

            score = tuple([ Dataset.evaluate_expression(expr, data_val['expressions'][k]) for expr in sampled_exprs ])
            data_val['score'].append(score)
            data_val['statistics'].append(Dataset.compute_statistics(score))

            if (k + 1) % 20 == 0:
                print(f".", end=' ', flush=True)

print()
print_results(data_val)


# dump_path = f'datasets_sampling/bool_logic_dataset_train_345_grpo_sampling_{dataset_sampling_id}_sampled.pkl'
# with open(dump_path, 'wb') as f:
#     pickle.dump(data_val, f)