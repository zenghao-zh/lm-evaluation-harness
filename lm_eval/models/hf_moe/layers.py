import torch
import math
from torch import nn

from collections import OrderedDict
from transformers.activations import ACT2FN
from .configuration_moe import MoEConfig

class ClassInstantier(OrderedDict):
    def __getitem__(self, key):
        content = super().__getitem__(key)
        return content

class Predictor(nn.Module):
    def __init__(self, in_features: int, mid_features:int, out_features: int, bias: bool = False) -> None:
        super().__init__()
        self.fc_1 = nn.Linear(in_features, mid_features, bias)
        self.fc_2 = nn.Linear(mid_features, out_features, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_2(self.fc_1(x))

class MoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts

        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux

        # topk selection algorithm
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init  as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape        
        ### compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = torch.nn.functional.linear(hidden_states, self.weight, None)
        topk_weight, topk_idx = torch.topk(logits, k=self.top_k, dim=-1, sorted=False)
        if self.scoring_func == 'softmax':
            # scores = logits.softmax(dim=-1)
            topk_weight = topk_weight.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')
        
        ### select top-k experts
        # topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        
        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        ### expert-level computation auxiliary loss
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            # always compute aux loss based on the naive greedy topk method
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss, torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim = 1)).sum(dim = 1).mean() * self.alpha
            else:
                mask_ce = torch.nn.functional.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = None
        return topk_idx, topk_weight, aux_loss

class MoELayer(nn.Module):
    """
    A mixed expert module containing shared experts.
    """
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.expert_mlp_class = EXPERTMLP2MODULE[config.moe_expert_class]
        self.experts = nn.ModuleList([self.expert_mlp_class(config, intermediate_size=config.moe_intermediate_size) for i in range(config.n_routed_experts)])
        self.gate = MoEGate(config)
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = MLP(config=config, intermediate_size=intermediate_size)
    
    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            hidden_states = hidden_states.repeat_interleave(self.num_experts_per_tok, dim=0)
            y = torch.empty_like(hidden_states)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i]).to(y.dtype)
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y =  y.view(*orig_shape)
            y = AddAuxiliaryLoss.apply(y, aux_loss)
        else:
            y = self.moe_infer(hidden_states, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y
    
    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.num_experts_per_tok
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i-1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_reduce_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out.to(expert_cache.dtype), reduce='sum')
        return expert_cache

class MLP(nn.Module):
    def __init__(self, config, intermediate_size = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class BalancedTopkModule(nn.Module):
    def __init__(self, hidden_size, topk, bank_size):
        super().__init__()
        self.register_buffer('balanced_bias', torch.zeros(hidden_size, dtype = torch.float32))
        self.register_buffer('num_assigned_tokens', torch.zeros(hidden_size))

        self.topk = topk
        self.bank_size = bank_size
        self.hidden_size = hidden_size
    
    def forward(self, x):
        mask_ = BalancedTopkFunction.apply(x.view(-1, self.hidden_size//self.bank_size, self.bank_size), self.topk, self.balanced_bias.view(-1, self.bank_size))
        mask = mask_.view_as(x)
        if self.training:
            with torch.no_grad():
                self.num_assigned_tokens += (mask != 0).sum(dim = tuple(range(x.ndim - 1)))
        return mask, self.num_assigned_tokens
    
    def reset_num_assigned_tokens(self):
        self.num_assigned_tokens.zero_()

class BalancedTopkFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, k, bias):
        if ctx.needs_input_grad[0]:
            _, topk_indices = (input.abs()+bias).topk(k, dim=-1)
        else:
            _, topk_indices = (input.abs()+bias).topk(k, dim=-1)
        mask = torch.zeros_like(input, dtype= input.dtype)
        mask.scatter_(-1, topk_indices, 1)
        output = input*mask

        # 保存掩码用于反向传播
        ctx.save_for_backward(mask)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors  # 恢复前向传播保存的掩码

        grad_input = grad_output*mask
        return grad_input, None, None  # 返回梯度（grad_input）以及k的None（无梯度）

class BalancedTopkMLP(nn.Module):
    def __init__(self, config, intermediate_size = None) -> None:
        super().__init__()

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

        self.topk = config.predictor_topk
        self.bank_size = config.predictor_bank_size
        # # fc1o topk
        self.predictor = Predictor(self.hidden_size, config.predictor_hidden_size, self.intermediate_size, bias=config.mlp_bias)
        # fc1o/fc2i topk
        self.topk_model = BalancedTopkModule(self.intermediate_size, self.topk, self.bank_size)
        
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # fc1o topk
        pred_x = torch.nn.functional.sigmoid(self.predictor(x))

        ## fc1o topk
        mask, *_ = self.topk_model(pred_x)
        x_fc_1 = self.gate_proj(x)
        x_fc_2 = self.up_proj(x)
        x = mask*self.act_fn(x_fc_1)*x_fc_2

        return self.down_proj(x)

EXPERTMLP2CLS = {
    "MLP": MLP,
    "BalancedTopkMLP": BalancedTopkMLP
    
}
EXPERTMLP2MODULE = ClassInstantier(EXPERTMLP2CLS)

MLP2CLS = {
    "MoELayer": MoELayer,
    "MLP": MLP
}
MLP2MODULE = ClassInstantier(MLP2CLS)
