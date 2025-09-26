import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
import inspect
from mamba_ssm import Mamba

# ------------------------------------------------------------------------------------
# 第一部分：首先，包含我们之前修正好的 MambaBlock
# ------------------------------------------------------------------------------------
# class MambaBlock(nn.Module):
#     """
#     一个精简且修正后的 Mamba 模块实现。
#     """
#     def __init__(self, n_embd, d_state=16, d_conv=4, expand=2):
#         super().__init__()
        
#         self.n_embd = n_embd
#         self.d_state = d_state
#         self.d_conv = d_conv
#         self.expand = expand
        
#         self.d_inner = int(self.expand * self.n_embd)
        
#         self.in_proj = nn.Linear(self.n_embd, 2 * self.d_inner, bias=False)
        
#         self.conv1d = nn.Conv1d(
#             in_channels=self.d_inner,
#             out_channels=self.d_inner,
#             kernel_size=d_conv,
#             bias=True,
#             groups=self.d_inner,
#             padding=d_conv - 1
#         )
        
#         self.x_proj = nn.Linear(self.d_inner, self.d_inner + 2 * self.d_state, bias=False)
#         self.out_proj = nn.Linear(self.d_inner, self.n_embd, bias=False)
        
#         A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
#         self.A_log = nn.Parameter(torch.log(A))
        
#     def forward(self, x):
#         B, L, D = x.shape
        
#         xz = self.in_proj(x)
#         x, z = xz.chunk(2, dim=-1)
        
#         x = x.transpose(1, 2)
#         x = self.conv1d(x)[:, :, :L]
#         x = x.transpose(1, 2)
        
#         x = F.silu(x)
#         y = self.ssm(x)
        
#         y = y * F.silu(z)
#         output = self.out_proj(y)
        
#         return output

#     def ssm(self, x):
#         B, L, d_inner = x.shape
        
#         A = -torch.exp(self.A_log.float())

#         x_proj_out = self.x_proj(x)
#         delta, B_param, C_param = torch.split(
#             x_proj_out,
#             [self.d_inner, self.d_state, self.d_state],
#             dim=-1
#         )
        
#         delta = F.softplus(delta)
        
#         delta_A = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
#         delta_B = delta.unsqueeze(-1) * B_param.unsqueeze(2)
#         delta_B_x = delta_B * x.unsqueeze(-1)
        
#         h = torch.zeros(B, d_inner, self.d_state, device=x.device)
#         ys = []

#         for i in range(L):
#             h = delta_A[:, i] * h + delta_B_x[:, i]
#             y = (h * C_param[:, i].unsqueeze(1)).sum(dim=-1)
#             ys.append(y)
        
#         return torch.stack(ys, dim=1)

# ------------------------------------------------------------------------------------
# 第二部分：定义完整的语言模型 MambaLM
# ------------------------------------------------------------------------------------
# model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
#                   bias=bias, vocab_size=None, dropout=dropout) 
class MambaLM(nn.Module):
    def __init__(self, config):
        """
        参数:
        vocab_size: 词汇表大小
        n_embd: 模型维度
        n_layer: 堆叠的 MambaBlock 数量
        d_state, d_conv, expand: MambaBlock 的内部参数
        """
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layer = config.n_layer
        self.n_embd = config.n_embd
        self.d_state = config.d_state
        self.d_conv = config.d_conv
        self.expand = config.expand 
        
        # 1. 嵌入层
        # print(f'初始化词嵌入层，词汇表大小: {vocab_size}, 嵌入维度: {n_embd}')
        # exit()
        self.embedding = nn.Embedding(self.vocab_size, self.n_embd)
        
        # 2. 核心处理层：堆叠多个 MambaBlock
        self.layers = nn.ModuleList([
            # 为了加入残差连接，我们创建一个包含 Norm 和 MambaBlock 的 "层"
            nn.Sequential(
                nn.LayerNorm(self.n_embd),
                Mamba(d_model=self.n_embd, d_state=self.d_state, d_conv=self.d_conv, expand=self.expand)
            )
            for _ in range(self.n_layer)
        ])
        
        # 3. 输出层
        self.final_norm = nn.LayerNorm(self.n_embd)
        self.lm_head = nn.Linear(self.n_embd, self.vocab_size, bias=False)

        # 权重共享 (可选但推荐): 嵌入层和输出层的权重可以共享
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids, targets=None, loss_mask=None):
        """
        input_ids: 输入的词元 ID 张量，形状为 (B, L)
        """
        # (B, L) -> (B, L, D)
        x = self.embedding(input_ids)
        
        # 依次通过每一个 Mamba 层
        for layer in self.layers:
            # 应用残差连接
            # 结构为: output = x + MambaBlock(LayerNorm(x))
            # 注意：这里的 'layer' 已经是 Sequential(Norm, MambaBlock)
            residual = x
            x = layer(x)
            x = residual + x
            
        # 通过最终的归一化层
        x = self.final_norm(x)
        
        # 映射到词汇表，得到 logits
        # (B, L, D) -> (B, L, vocab_size)
        logits = self.lm_head(x)
        

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction='none')
            if loss_mask is not None:
                loss = loss * loss_mask.view(-1)

            loss = loss.mean()
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


@dataclass
class ManbaConfig:
    block_size: int = 512
    vocab_size: int = 10
    n_layer: int = 1
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


# --- 示例用法 ---
if __name__ == '__main__':
    # 模型超参数
    vocab_size = 10000 # 假设词汇表大小为 10000
    n_embd = 128     # 模型维度
    n_layer = 4    # 堆叠 4 层 MambaBlock
    seq_len = 256     # 序列长度
    batch_size = 16   # 批次大小

    # 创建模型实例
    model = MambaLM(
        vocab_size=vocab_size,
        n_embd=n_embd,
        n_layer=n_layer
    )
    print(model)
    print(f"\n模型总参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f} M")


    # 创建一个随机的输入张量 (代表一个批次的文本 token IDs)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    # 前向传播
    logits = model(input_ids)

    # 打印输入和输出的形状
    print(f"\n输入形状 (input_ids): {input_ids.shape}")
    print(f"输出形状 (logits):    {logits.shape}")

    # 验证输出形状是否正确
    assert logits.shape == (batch_size, seq_len, vocab_size)
    print("\n语言模型运行成功，输出形状正确！")

    # 在实际训练中，你会将 logits 和目标 labels 传入损失函数
    # 例如：loss_fn = nn.CrossEntropyLoss()
    # labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    # loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1))
    # loss.backward()