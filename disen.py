import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, feature):
        query = self.query(feature)
        key = self.key(feature)
        value = self.value(feature)

        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (feature.size(-1) ** 0.5)
        attention_probs = self.softmax(attention_scores)

        # Apply attention to the value
        attention_output = torch.matmul(attention_probs, value)

        return attention_output


class CrossAttention(nn.Module):
    def __init__(self, hidden_size):
        super(CrossAttention, self).__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query_feature, key_value_feature):
        query = self.query(query_feature)
        key = self.key(key_value_feature)
        value = self.value(key_value_feature)

        # Compute cross-attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (key_value_feature.size(-1) ** 0.5)
        attention_probs = self.softmax(attention_scores)

        # Apply attention to the value
        attention_output = torch.matmul(attention_probs, value)

        return attention_output


class GatedMixtureOfExperts(nn.Module):
    def __init__(self, hidden_size, num_experts=3):
        super(GatedMixtureOfExperts, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size)
            ) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(hidden_size, num_experts)

    def forward(self, x):
        gate_logits = self.gate(x)  # [B, L, num_experts]
        gate_weights = torch.softmax(gate_logits, dim=-1)
        expert_outputs = [expert(x) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=-1)  # [B, L, hidden_size, num_experts]
        gate_weights = gate_weights.unsqueeze(2)  # [B, L, 1, num_experts]
        output = torch.sum(expert_outputs * gate_weights, dim=-1)  # [B, L, hidden_size]
        return output


# -------------------------------
# Conditional Convolution
# -------------------------------
class ConditionalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, cond_dim, stride=1, padding=0):
        super(ConditionalConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        num_params = in_channels * out_channels * kernel_size * kernel_size
        hidden_dim = num_params // 2
        self.hyper = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_params)
        )

    def forward(self, x, cond):
        # x: [B, in_channels, H, W]��cond: [B, cond_dim]
        batch_size = x.size(0)
        conv_weights = self.hyper(cond)  # [B, num_params]
        conv_weights = conv_weights.view(batch_size, self.out_channels, self.in_channels, self.kernel_size,
                                         self.kernel_size)
        outputs = []
        for i in range(batch_size):
            weight = conv_weights[i]  # [out_channels, in_channels, kernel, kernel]
            out = F.conv2d(x[i].unsqueeze(0), weight, stride=self.stride, padding=self.padding)
            outputs.append(out)
        return torch.cat(outputs, dim=0)


# -------------------------------
# ImageAdapter
# -------------------------------
class Image_adapter(nn.Module):
    def __init__(self, hidden_size=1024, mid_channels=128, cond_dim=128,num_attention_layers=2, num_experts=3):
        super().__init__()
        # �������� 1024 �������� 128
        self.down_proj = nn.Conv2d(hidden_size, mid_channels, kernel_size=1, stride=1, padding=0)
        self.up_proj = nn.Conv2d(mid_channels, hidden_size, kernel_size=1, stride=1, padding=0)

        self.cond_conv = ConditionalConv(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=1,
            cond_dim=cond_dim,
            stride=1,
            padding=0
        )
        self.self_attention_layers = nn.ModuleList(
            [SelfAttention(hidden_size) for _ in range(num_attention_layers)]
        )
        self.moe = GatedMixtureOfExperts(hidden_size, num_experts=num_experts)

        self.gate = nn.Sigmoid()

    def forward(self, feature, cond_input=None):

        original_dim = feature.dim()

        if original_dim == 4:
            B, C, H, W = feature.shape
            feature_seq = feature.view(B, C, H * W).transpose(1, 2)
        else:
            feature_seq = feature

        for attn in self.self_attention_layers:
            feature_seq = feature_seq + attn(feature_seq)

        moe_output = self.moe(feature_seq)

        gate_value = self.gate(moe_output)
        combined = gate_value * moe_output + (1 - gate_value) * feature_seq

        if original_dim == 4:
            combined = combined.transpose(1, 2).view(B, C, H, W)

            if cond_input is not None:
                combined = combined + self.cond_conv(combined, cond_input)
            return combined
        else:
            return combined


def self_attention_params(hidden_size):
    return 3 * hidden_size * hidden_size


def gated_mixture_of_experts_params(hidden_size, num_experts):
    expert_params = 2 * hidden_size * hidden_size * num_experts
    gate_params = hidden_size * num_experts
    return expert_params + gate_params

def conditional_conv_params(in_channels, out_channels, kernel_size, cond_dim):
    num_params = in_channels * out_channels * kernel_size * kernel_size
    hidden_dim = num_params // 2
    hyper_params = cond_dim * hidden_dim + hidden_dim * num_params
    conv_params = num_params
    return hyper_params + conv_params

def compute_image_adapter_params(hidden_size, mid_channels, cond_dim, num_attention_layers, num_experts):
    attention_params = num_attention_layers * self_attention_params(hidden_size)

    moe_params = gated_mixture_of_experts_params(hidden_size, num_experts)

    cond_conv_params = conditional_conv_params(mid_channels, mid_channels, 1, cond_dim)

    down_proj_up_proj_params = 2 * hidden_size * mid_channels

    total_params = (down_proj_up_proj_params + cond_conv_params + attention_params + moe_params)
    return total_params

def print_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model total parameters: {total_params}")

if __name__ == '__main__':
    hidden_size = 1024
    mid_channels = 48
    cond_dim = 1024
    num_attention_layers = 2
    num_experts = 3

    total_params = compute_image_adapter_params(hidden_size, mid_channels, cond_dim, num_attention_layers, num_experts)
    print(f"AD-MOE: {total_params}")

    model = Image_adapter(hidden_size=hidden_size, mid_channels=mid_channels, cond_dim=cond_dim,
                          num_attention_layers=num_attention_layers, num_experts=num_experts)

    print_model_params(model)





