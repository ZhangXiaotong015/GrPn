import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from torch.nn.modules.utils import _pair
import math
from modules.dgmn.dcn import DeformUnfold, DeformUnfold3D


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class RelPosEmb(nn.Module):
    def __init__(self, fmap_size, dim_head, num_samples):
        super().__init__()
        height, width = fmap_size
        scale = dim_head ** -0.5
        self.num_samples = num_samples
        self.rel_height = nn.Parameter(torch.randn(height + num_samples - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(width + num_samples - 1, dim_head) * scale)

    def rel_to_abs(self, x):
        b, h, l, c = x.shape # (B*num_heads, H, W, W+9-1)
        x = torch.cat((x, torch.zeros((b, h, l, 1), dtype=x.dtype, device=x.device)), dim=3) # (B*num_heads, H, W, W+9)
        x = x.reshape(b, h, l * (c + 1)) # (B*num_heads, H, W*(W+9))
        x = torch.cat((x, torch.zeros((b, h, self.num_samples - 1), dtype=x.dtype, device=x.device)), dim=2) # (B*num_heads, H, l*(l+num_samples)+num_samples-1)
        x = x.reshape(b, h, l + 1, self.num_samples + l - 1)
        x = x[:, :, :l, (l - 1):] # (B*num_heads, H, W, num_samples)
        return x

    def relative_logits_1d(self, q, rel_k):
        logits = torch.matmul(q, rel_k.transpose(0, 1)) # (B*num_heads, 1, H, W, W+9-1)
        b, h, x, y, r = logits.shape # (B*num_heads, 1, H, W, W+9-1)
        logits = logits.reshape(b, h * x, y, r) # (B*num_heads, H, W, W+9-1)
        logits = self.rel_to_abs(logits) # (B*num_heads, H, W, num_samples)
        return logits

    def forward(self, q, H, W):
        rel_width = F.interpolate(
            self.rel_width.unsqueeze(0).unsqueeze(0),
            size=(W + 9 - 1, self.rel_width.shape[1]), mode="bilinear").squeeze(0).squeeze(0)
        rel_height = F.interpolate(
            self.rel_height.unsqueeze(0).unsqueeze(0),
            size=(H + 9 - 1, self.rel_height.shape[1]), mode="bilinear").squeeze(0).squeeze(0)
        # 'q: (B * num_heads, 1, H, W, head_dim)'
        rel_logits_w = self.relative_logits_1d(q, rel_width) # (B*num_heads, H, W, num_samples)
        q = q.transpose(2, 3) # (B * num_heads, 1, W, H, head_dim)
        rel_logits_h = self.relative_logits_1d(q, rel_height) # (B*num_heads, W, H, num_samples)

        rel_logits_h = F.interpolate(
            rel_logits_h.permute(0, 3, 1, 2),
            size=rel_logits_w.shape[1:3], mode="bilinear").permute(0, 2, 3, 1) # (B*num_heads, H, W, num_samples)

        return rel_logits_w + rel_logits_h

class RelPosEmb_3D(nn.Module):
    def __init__(self, fmap_size, dim_head, num_samples):
        super().__init__()
        depth, height, width = fmap_size  # Expecting 3D input (depth, height, width)
        scale = dim_head ** -0.5
        self.num_samples = num_samples
        
        # Initialize parameters for relative positional embeddings in 3D
        self.rel_depth = nn.Parameter(torch.randn(depth + num_samples - 1, dim_head) * scale)
        self.rel_height = nn.Parameter(torch.randn(height + num_samples - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(width + num_samples - 1, dim_head) * scale)

    def rel_to_abs(self, x):
        # Adjusted for 3D dimensions (depth, height, width)
        b, d, h, w, c = x.shape  # (B*num_heads, D, H, W, W+27-1)
        x = torch.cat((x, torch.zeros((b, d, h, w, 1), dtype=x.dtype, device=x.device)), dim=4) # (B*num_heads, D, H, W, W+27)
        x = x.reshape(b, d, h, w * (c + 1)) # (B*num_heads, D, H, W*(W+27)
        x = torch.cat((x, torch.zeros((b, d, h, self.num_samples - 1), dtype=x.dtype, device=x.device)), dim=3) # (B*num_heads, D, H, num_samples-1+W*(W+num_samples) )
        x = x.reshape(b, d, h, w + 1, self.num_samples + w - 1)
        x = x[:, :, :, :w, (w - 1):] # (B*num_heads, D, H, W, num_samples)
        return x

    def relative_logits_1d(self, q, rel_k):
        logits = torch.matmul(q, rel_k.transpose(0, 1)) # (B*num_heads, 1, D, H, W, W+27-1)
        b, h, z, x, y, r = logits.shape
        logits = logits.reshape(b, h * z, x, y, r) # (B*num_heads, D, H, W, W+27-1)
        logits = self.rel_to_abs(logits) # (B*num_heads, D, H, W, num_samples)
        return logits

    def forward(self, q, D, H, W):
        'q: (B * num_heads, 1, D, H, W, head_dim)'
        # Interpolate the relative positional embeddings for depth, height, and width
        rel_width = F.interpolate(
            self.rel_width.unsqueeze(0).unsqueeze(0),
            size=(W + 27 - 1, self.rel_width.shape[1]), mode="bilinear").squeeze(0).squeeze(0)
        
        rel_height = F.interpolate(
            self.rel_height.unsqueeze(0).unsqueeze(0),
            size=(H + 27 - 1, self.rel_height.shape[1]), mode="bilinear").squeeze(0).squeeze(0)
        
        rel_depth = F.interpolate(
            self.rel_depth.unsqueeze(0).unsqueeze(0),
            size=(D + 27 - 1, self.rel_depth.shape[1]), mode="bilinear").squeeze(0).squeeze(0)

        # Get the relative logits for each of the 3 dimensions
        # q: (B * num_heads, 1, D, H, W, head_dim)
        rel_logits_w = self.relative_logits_1d(q, rel_width) # (B*num_heads, D, H, W, num_samples)

        q = q.transpose(3, 4) # (B * num_heads, 1, D, W, H, head_dim)
        rel_logits_h = self.relative_logits_1d(q, rel_height) #(B*num_heads, D, W, H, num_samples)

        q = q.transpose(3, 4) # (B * num_heads, 1, D, H, W, head_dim)
        q = q.transpose(2, 4) # (B * num_heads, 1, W, H, D, head_dim)
        rel_logits_d = self.relative_logits_1d(q, rel_depth) # (B*num_heads, W, H, D, num_samples)

        # Adjusting dimensions for interpolation to match the final output shape
        rel_logits_h = F.interpolate(
            rel_logits_h.permute(0, 4, 1, 2, 3),
            size=rel_logits_w.shape[1:4], mode="trilinear").permute(0, 2, 3, 4, 1) # (B*num_heads,D,H,W,num_samples)

        rel_logits_d = F.interpolate(
            rel_logits_d.permute(0, 4, 1, 2, 3),
            size=rel_logits_w.shape[1:4], mode="trilinear").permute(0, 2, 3, 4, 1) # (B*num_heads,D,H,W,num_samples)

        # Combine the 3D relative logits
        return rel_logits_w + rel_logits_h + rel_logits_d # (B*num_heads,D,H,W,num_samples)


class DGMN2Attention_3D(nn.Module):
    def __init__(self, dim, num_heads=8, fea_size=(224, 224, 224), qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # sample
        self.num_samples = 27
        self.conv_offset = nn.Linear(self.head_dim, self.num_samples * 3, bias=qkv_bias)
        self.unfold = DeformUnfold3D(kernel_size=3, padding=1, dilation=1)

        # relative position
        self.pos_emb = RelPosEmb_3D(fea_size, self.head_dim, self.num_samples)

    def forward(self, x, D, H, W):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4) # (3,B,num_heads,N,head_dim)
        q, k, v = qkv.unbind(0) # (B,num_heads,N,head_dim)

        offset = self.conv_offset(x.reshape(B, N, self.num_heads, self.head_dim)).permute(0, 2, 3, 1).reshape(B * self.num_heads, self.num_samples * 3, D, H, W)

        k = k.transpose(2, 3).reshape(B * self.num_heads, self.head_dim, D, H, W) 
        v = v.transpose(2, 3).reshape(B * self.num_heads, self.head_dim, D, H, W)
        
        ### self.unfold(k, offset): (B*num_heads, head_dim*num_samples, N)
        k = self.unfold(k, offset).transpose(1, 2).reshape(B, self.num_heads, N, self.head_dim, self.num_samples)
        v = self.unfold(v, offset).reshape(B, self.num_heads, self.head_dim, self.num_samples, N).permute(0, 1, 4, 3, 2) # (B, num_heads, N, num_samples, head_dim)

        attn = torch.matmul(q.unsqueeze(3), k) * self.scale # (B,num_heads,N,1,num_samples)

        attn_pos = self.pos_emb(q.reshape(B * self.num_heads, 1, D, H, W, self.head_dim), D, H, W).reshape(B,self.num_heads, N, 1, self.num_samples)
        attn = attn + attn_pos

        attn = attn.softmax(dim=-1) # (B,num_heads, N, 1, num_samples)
        attn = self.attn_drop(attn)

        x = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, C) #(B,num_heads,N,1,head_dim)->(B,N,num_heads,1,head_dim)->(B,N,C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class DGMN2Block_3D(nn.Module):
    def __init__(
        self, dim, num_heads,fea_size=(224, 224, 224), mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
        drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = DGMN2Attention_3D(
            dim, num_heads=num_heads, fea_size=fea_size, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
            proj_drop=drop
        )

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, D, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), D, H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x