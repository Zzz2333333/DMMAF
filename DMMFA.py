import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.init as init
import torch.nn.functional as F
import random
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from timm.models.vision_transformer import PatchEmbed



class SELayer(nn.Module):
    def __init__(self, channel, reduction = 8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(channel // reduction, channel // reduction, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(channel // reduction, channel, bias = False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class RDPBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, dilation=1):
        super(RDPBlock, self).__init__()

        reflect_padding = int(dilation * (kernel_size - 1) / 2)
        self.reflection_pad = nn.ReflectionPad2d(reflect_padding)

        # 深度卷积
        self.depthwise = nn.Conv2d(in_dim, in_dim, kernel_size, stride, dilation=dilation, groups=in_dim, bias=False)
        # 逐点卷积
        self.pointwise = nn.Conv2d(in_dim, out_dim, 1, 1, bias=False)

    @torch.no_grad()
    def fuse(self):
        # 融合 BN
        w = self.bn.weight / (self.bn.running_var + self.bn.eps) ** 0.5
        w_depthwise = self.depthwise.weight * w[:, None, None, None]
        w_pointwise = self.pointwise.weight * w[:, None, None, None]
        b = self.bn.bias - self.bn.running_mean * self.bn.weight / (self.bn.running_var + self.bn.eps) ** 0.5

        fused_depthwise = nn.Conv2d(self.depthwise.in_channels, self.depthwise.out_channels, self.depthwise.kernel_size,
                                    self.depthwise.stride, self.depthwise.padding, groups=self.depthwise.groups,
                                    bias=True)
        fused_pointwise = nn.Conv2d(self.pointwise.in_channels, self.pointwise.out_channels, self.pointwise.kernel_size,
                                    self.pointwise.stride, self.pointwise.padding, bias=True)

        fused_depthwise.weight.data.copy_(w_depthwise)
        fused_pointwise.weight.data.copy_(w_pointwise)
        fused_pointwise.bias.data.copy_(b)

        return fused_depthwise, fused_pointwise
    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.depthwise(out)
        out = self.pointwise(out)
        return out

# -----------------Transformer-----------------

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # b, c, h, w = x.shape
        # x = x.reshape(b,c,h*w)
        x = self.net(x)
        # x = x.reshape(b,c,h,w)
        return x


# Light transformer
class DeformableAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=32, dropout=0.,layer_scale_init_value=1e-2):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.proj = nn.Linear(inner_dim, dim)

        self.offset_conv = nn.Conv2d(dim, 2 * heads, 3, padding=1, bias=True)
        self.dropout = nn.Dropout(dropout)

        # 加入 LayerScale
        self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x_in):
        """
        x_in: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.view(b, h*w, c)

        q = self.to_q(x)  # [b, hw, heads * dim_head]
        kv = self.to_kv(x).chunk(2, dim=-1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h d n', h=self.num_heads), kv)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)

        # 学习偏移
        offset = self.offset_conv(x_in.permute(0,3,1,2))  # [b, heads*2, h, w]
        offset = rearrange(offset, 'b (h2 c2) h w -> b h2 c2 h w', h2=self.num_heads)

        # 生成采样网格 grid
        # 注意grid_sample要求 grid ∈ [-1,1]
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, h, device=x.device),
            torch.linspace(-1, 1, w, device=x.device)
        )
        base_grid = torch.stack((grid_x, grid_y), dim=-1)  # 注意这里是先x再y
        base_grid = torch.stack((grid_x, grid_y), dim=-1)  # [h,w,2]

        base_grid = base_grid.unsqueeze(0).unsqueeze(0).repeat(b,self.num_heads,1,1,1)  # [b,heads,h,w,2]
        sampling_grid = base_grid + offset.permute(0,1,3,4,2)  # [b,heads,h,w,2]

        # 采样 k,v，grid_sample要求[N,C,H,W]
        k_map = k.reshape(b * self.num_heads, self.dim_head, h, w)
        v_map = v.reshape(b * self.num_heads, self.dim_head, h, w)
        sampling_grid = sampling_grid.view(b*self.num_heads, h, w, 2)

        sampled_k = F.grid_sample(k_map, sampling_grid, mode='bilinear', padding_mode='border', align_corners=True)
        sampled_v = F.grid_sample(v_map, sampling_grid, mode='bilinear', padding_mode='border', align_corners=True)

        # flatten
        sampled_k = sampled_k.view(b, self.num_heads, self.dim_head, h*w).transpose(2,3)  # [b,heads,hw,dim_head]
        sampled_v = sampled_v.view(b, self.num_heads, self.dim_head, h*w).transpose(2,3)

        # attention
        q = F.normalize(q, dim=-1)
        sampled_k = F.normalize(sampled_k, dim=-1)

        attn = (q * sampled_k).sum(dim=-1)  # [b,heads,hw]
        attn = F.softmax(attn, dim=-1)

        out = (attn.unsqueeze(-1) * sampled_v).sum(dim=2)  # [b,heads,dim_head]
        out = rearrange(out, 'b h d -> b 1 (h d)').expand(-1, h*w, -1)

        out = self.proj(out)
        out = self.dropout(out)

        # 加 LayerScale
        out = out * self.layer_scale
        out = out.view(b, h, w, c)
        return out

class MambaSSM(nn.Module):
    def __init__(self, dim, ssm_kernel_size=16, mlp_dim=None, dropout=0.):
        super(MambaSSM, self).__init__()

        # 参数
        self.dim = dim
        self.kernel_size = ssm_kernel_size

        # 状态空间模型（SSM）内核，用于建模时序依赖
        self.ssm_kernel = nn.Parameter(torch.randn(dim, 1, ssm_kernel_size))  # [C,1,K]

        # 输入的值和门控投影
        self.value_proj = nn.Linear(dim, dim)
        self.gate_proj = nn.Linear(dim, dim)

        # 门控激活（SiLU）
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU()  # 使用 SiLU 激活函数
        )

        # Dropout 和 LayerNorm
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        x: [B, H, W, C] - 输入张量
        """
        # LayerNorm 处理
        x_norm = self.norm(x)

        # 展平输入以进行 SSM 处理
        b, h, w, c = x.shape
        x_flat = x.view(b, h * w, c)  # [B, L, C]

        # 计算门控（Gate）机制
        v = self.value_proj(x_flat)  # [B, L, C] 计算值
        g = F.silu(self.gate_proj(x_flat))  # [B, L, C] 计算门控
        x_proj = v * g  # [B, L, C]  门控后的输入

        # 转换为卷积格式
        x_proj = x_proj.permute(0, 2, 1)  # [B, C, L]

        # 进行 causal padding（保持时序依赖）
        x_proj = F.pad(x_proj, (self.kernel_size - 1, 0))  # [B, C, L+K-1]

        # 状态空间卷积（SSM 卷积核）
        ssm_out = F.conv1d(x_proj, self.ssm_kernel, groups=self.dim)  # [B, C, L]

        # 将卷积结果转回图像格式
        ssm_out = ssm_out.permute(0, 2, 1).view(b, h, w, c)  # [B, L, C] → [B, H, W, C]

        # 再加一个门控机制
        gated = self.gate(x_norm)  # [B, H, W, C] 门控加权输入

        # 将 SSM 输出与门控结合
        out = ssm_out * gated

        # Dropout
        out = self.dropout(out)

        return out

class Transformer_E(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(depth):
            if i % 2 == 0:
                self.layers.append(nn.ModuleList([
                    Residual(PreNorm(dim, DeformableAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                    Residual(PreNorm(dim, Mamba(dim, mlp_dim, dropout=dropout))),
                    Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
                ]))
            else:
                self.layers.append(nn.ModuleList([
                    Residual(PreNorm(dim, Mamba(dim, mlp_dim, dropout=dropout))),
                    Residual(PreNorm(dim, Mamba(dim, mlp_dim, dropout=dropout))),
                    Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
                ]))

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        for layer in self.layers:
            attn, ffn = layer
            x = attn(x)
            x = ffn(x)
        x = x.permute(0, 3, 1, 2)
        return x

class Transformer_D(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, DeformableAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, DeformableAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # [B, C, H, W] → [B, H, W, C]

        for attn1, attn2, ffn in self.layers:
            x = attn1(x)
            x = attn2(x)
            x = ffn(x)

        x = x.permute(0, 3, 1, 2)  # [B, H, W, C] → [B, C, H, W]
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class ADNMask(nn.Module):


    def __init__(
        self,
        mask_ratio=0.3,
        noise_strength=1.0,
        use_channel_mask=True,
        adaptive_masking=False
    ):

        super().__init__()
        self.mask_ratio = mask_ratio
        self.noise_strength = noise_strength
        self.use_channel_mask = use_channel_mask
        self.adaptive_masking = adaptive_masking

    def forward(self, x):
        """
        x: [b, c, h, w] 的特征图
        """
        b, c, h, w = x.shape
        # 将特征图展平到 L = c*h*w
        x_reshape = x.view(b, -1)  # [b, L]

        # 1) 生成随机噪声并排序 (MAE中的做法)
        #   注意：如果你更倾向于在通道维度排序，可以自行 reshape 调整，这里简化为对全部像素展开排序
        N, L = x_reshape.shape
        noise = torch.rand(N, L, device=x.device)  # 噪声分布在 [0,1]
        # 按噪声升序排序，得到每个样本的排序索引
        ids_shuffle = torch.argsort(noise, dim=1)  # [b, L]
        # 反向索引
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # 2) 计算需要保留的长度 (MAE中的做法)
        len_keep = int(L * (1 - self.mask_ratio))
        ids_keep = ids_shuffle[:, :len_keep]

        # 3) 仅保留前 len_keep 的索引
        x_kept = torch.gather(x_reshape, dim=1, index=ids_keep)

        # 4) 生成二进制 mask：0表示保留, 1表示被mask
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)  # [b, L]

        # 5) 这里，你可以决定如何处理被遮掩的部分
        #    比如像 MAE 一样，把被遮掩部分替换为 0 向量，
        #    或者像你原先的 random_masking2 一样，加噪声再置零
        x_masked = x_reshape.clone()
        # 根据 mask = 1 的位置进行操作
        # 先加噪声
        noise_mask = (mask == 1).float().unsqueeze(2)  # [b, L, 1]
        # 对被mask部分加噪
        x_masked = x_masked.unsqueeze(-1)  # [b, L, 1]
        random_noise = torch.randn_like(x_masked) * self.noise_strength
        x_masked = x_masked + noise_mask * random_noise
        # 再置零（只对被遮掩位置）
        x_masked = x_masked * (1 - noise_mask)
        x_masked = x_masked.squeeze(-1)  # [b, L]

        # 6) 如果需要对通道维度额外做操作 (use_channel_mask=True)
        #    比如对被遮掩的通道再次加噪声，或者只对通道级别进行遮掩
        if self.use_channel_mask:
            # 这里演示一个简化版本：随机选择部分通道，将其整体置零/加噪
            # 具体策略可自行改进
            channel_num = c
            masked_channel_num = int(channel_num * 0.2)  # 比如额外随机 20% 的通道
            channel_indices = random.sample(range(channel_num), masked_channel_num)
            # 在这些通道上，我们对全部像素位置做一次噪声 + 置零
            x_masked_view = x_masked.view(b, c, h*w)
            for ch in channel_indices:
                # ch 通道的特征加噪
                x_masked_view[:, ch, :] += (
                    self.noise_strength * torch.randn(b, h*w, device=x.device).abs_()
                )
                # 再置零
                x_masked_view[:, ch, :] = 0.

            x_masked = x_masked_view.view(b, -1)

        # 将处理后结果 reshape 回 [b, c, h, w]
        x_out = x_masked.view(b, c, h, w)

        # 如果开启自适应遮掩比例，可在此处随训练步数迭代地修改 self.mask_ratio
        # 例如 self.mask_ratio = min(1.0, self.mask_ratio + 0.01) 等
        return x_out

    def update_mask_ratio(self, epoch, total_epochs):

            if self.adaptive_masking:
                # 例如让 mask_ratio 随 epoch 缓慢增大
                increment = 0.5 * (epoch / float(total_epochs))  # 线性增长到 0.5
                new_ratio = self.mask_ratio + increment
                self.mask_ratio = min(1.0, new_ratio)


class DMMFA(nn.Module):
    def __init__(self, inplanes=3, planes=31, channels=200, n_DRBs=8):
        super(DMMFA, self).__init__()
        # 2D Nets
        num_feature = channels
        self.reshape0 = nn.Sequential(
            RDPBlock(inplanes, 31, 3, 1),
            nn.PReLU(),
            RDPBlock(31, 31, 3, 1)
        )

        self.mask = nn.Sequential(
            ADNMask(mask_ratio=0.3),
            RDPBlock(31, 31, 3, 1),
            nn.PReLU(),
            RDPBlock(31, 31, 3, 1)
        )

        self.reshape1 = nn.Sequential(
            RDPBlock(31, channels, 3, 1),
            nn.PReLU(),
            RDPBlock(channels, channels, 3, 1)
        )

        self.TE0 = Transformer_E(channels, 2, 3, channels // 3, channels)
        self.SE0 = SELayer(channels)

        self.down_sample_1 = nn.Conv2d(channels, channels * 2, 4, 2, 1, bias=False)
        self.TE1 = Transformer_E(channels * 2, 2, 3, channels * 2 // 3, channels * 2)
        self.up_sample_1 = nn.ConvTranspose2d(channels * 2, channels, 4, 2, 1)

        self.down_sample_2 = nn.Conv2d(channels * 2, channels * 4, 4, 2, 1, bias=False)
        self.TE2 = Transformer_E(channels * 4, 2, 3, channels * 4 // 3, channels * 4)
        self.up_sample_2 = nn.ConvTranspose2d(channels * 4, channels * 2, 4, 2, 1)
        self.reshape2 = nn.Sequential(
            RDPBlock(channels * 3, channels, 3, 1),
            nn.PReLU(),
            RDPBlock(channels, channels, 3, 1)
        )

        self.TD1 = Transformer_D(channels, 2, 3, channels // 3, channels)
        self.TD2 = Transformer_D(channels, 2, 3, channels // 3, channels)

        self.refine1 = nn.Sequential(
            RDPBlock(channels, channels, 3, 1),
            nn.PReLU(),
            RDPBlock(channels, channels, 3, 1)
        )
        self.reshape3 = nn.Sequential(
            RDPBlock(channels, channels, 3, 1),
            nn.PReLU(),
            RDPBlock(channels, planes, 3, 1)
        )

    def forward(self, x):
        out = self.DRN2D(x)
        return out

    @torch.no_grad()
    def fuse(self):
        # 融合 Repvit
        self.reshape0 = nn.Sequential(*[m.fuse() if isinstance(m, RDPBlock) else m for m in self.reshape0])
        self.mask = nn.Sequential(*[m.fuse() if isinstance(m, RDPBlock) else m for m in self.mask])
        self.reshape1 = nn.Sequential(*[m.fuse() if isinstance(m, RDPBlock) else m for m in self.reshape1])
        return self

    def DRN2D(self, x):
        out = self.reshape0(x)
        before = out.clone()
        out = self.mask(out)
        out = before + 0.1 * out

        after = out.clone()  # 用于画特征图2
        out = self.reshape1(out)

        y_0 = out
        y_1 = self.down_sample_1(y_0)
        y_2 = self.down_sample_2(y_1)

        out_2 = self.TE2(y_2)
        out_2 = self.up_sample_2(out_2)
        con1 = self.up_sample_1(out_2)

        y_1 = y_1 + out_2
        out_1 = self.TE1(y_1)
        out_1 = self.up_sample_1(out_1)
        con0 = out_1

        y_0 = self.SE0(y_0) + out_1
        out_0 = self.TE0(y_0)
        out = out + out_0

        out = torch.cat((con1, con0, out), 1)
        out = self.reshape2(out)

        out = self.TD1(out)
        out = self.TD2(out)

        out = out + y_0

        out = self.refine1(out)
        out = self.reshape3(out)

        return out, before, after


class Conv2D(nn.Module):
    def __init__(self, in_channel=256, out_channel=8):
        super(Conv2D, self).__init__()
        self.guide_conv2D = nn.Conv2d(in_channel, out_channel, 3, 1, 1)

    def forward(self, x):
        spatial_guidance = self.guide_conv2D(x)
        return spatial_guidance



if __name__ == "__main__":
    # import os
    # os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    input_tensor = torch.rand(1, 3, 64, 64)

    model = DMMFA(3, 31, 48, 1)
    # model = nn.DataParallel(model).cuda()
    model.fuse()
    with torch.no_grad():
        output_tensor = model(input_tensor)
    # print(output_tensor.shape)
    print('Parameters number is ', sum(param.numel() for param in model.parameters()))