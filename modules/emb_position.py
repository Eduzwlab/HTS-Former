import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

class SINCOS(nn.Module):
    def __init__(self,embed_dim=512):
        super(SINCOS, self).__init__()
        self.embed_dim = embed_dim
        self.pos_embed = self.get_2d_sincos_pos_embed(embed_dim, 8)
    def get_1d_sincos_pos_embed_from_grid(self,embed_dim, pos):
        """
        embed_dim: output dimension for each position
        pos: a list of positions to be encoded: size (M,)
        out: (M, D)
        """
        assert embed_dim % 2 == 0
        omega = np.arange(embed_dim // 2, dtype=np.float)
        omega /= embed_dim / 2.
        omega = 1. / 10000**omega  # (D/2,)

        pos = pos.reshape(-1)  # (M,)
        out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

        emb_sin = np.sin(out) # (M, D/2)
        emb_cos = np.cos(out) # (M, D/2)

        emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
        return emb

    def get_2d_sincos_pos_embed_from_grid(self,embed_dim, grid):
        assert embed_dim % 2 == 0

        # use half of dimensions to encode grid_h
        emb_h = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
        emb_w = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

        emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
        return emb

    def get_2d_sincos_pos_embed(self,embed_dim, grid_size, cls_token=False):
        """
        grid_size: int of the grid height and width
        return:
        pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
        """
        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)  # here w goes first
        grid = np.stack(grid, axis=0)

        grid = grid.reshape([2, 1, grid_size, grid_size])
        pos_embed = self.get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
        if cls_token:
            pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
        return pos_embed

    def forward(self, x):
        B,H,W,C = x.shape
        pos_embed = torch.from_numpy(self.pos_embed).float().to(x.device)
        x = x + pos_embed.unsqueeze(1).unsqueeze(1).repeat(1,H,W,1)
        return x

class CATA(nn.Module):
    def __init__(self, in_dim, dim=512,k=7,conv_1d=False,bias=True):
        super(CATA, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.query_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma1 = nn.Parameter(torch.zeros(1))

        self.proj = nn.Conv2d(dim, dim, k, 1, k // 2, groups=dim, bias=bias) if not conv_1d else nn.Conv2d(dim, dim, (k, 1),
                                                                                                           1, (k // 2, 0),
                                                                                                           groups=dim,
                                                                                                           bias=bias)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim, bias=bias) if not conv_1d else nn.Conv2d(dim, dim,
                                                                                                            (5, 1), 1,
                                                                                                            (5 // 2, 0),
                                                                                                            groups=dim,
                                                                                                            bias=bias)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim, bias=bias) if not conv_1d else nn.Conv2d(dim, dim,
                                                                                                            (3, 1), 1,
                                                                                                            (3 // 2, 0),
                                                                                                            groups=dim,
                                                                                                            bias=bias)

    def forward(self, x):
        B, N, C = x.shape
        H1 = x.shape[1]
        H, W = int(np.ceil(np.sqrt(H1))), int(np.ceil(np.sqrt(H1)))
        add_length = H * W - N
        x1 = torch.cat([x, x[:, :add_length, :]], dim=1)


        if H < 7:
            H, W = 7, 7
            zero_pad = H * W - (N + add_length)
            x1 = torch.cat([x1, torch.zeros((B, zero_pad, C), device=x1.device)], dim=1)
            add_length += zero_pad
        cnn_feat = x1.transpose(1, 2).view(B, C, H, W)


        proj_query = self.query_conv(cnn_feat)
        proj_key = self.key_conv(cnn_feat)
        proj_value = self.value_conv(cnn_feat)

        energy = ca_weight(proj_query, proj_key)
        attention = F.softmax(energy, 1)

        out = ca_map(attention, proj_value)
        out = self.gamma * out + cnn_feat

        cnn_feat1 = self.proj(cnn_feat) + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        out = out + cnn_feat1

        proj_query1 = self.query_conv1(out)
        proj_key1 = self.key_conv1(out)
        proj_value1 = self.value_conv1(out)

        energy1 = ca_weight(proj_query1, proj_key1)
        attention1 = F.softmax(energy1, 1)
        out1 = ca_map(attention1, proj_value1)
        out1 = self.gamma1 * out1 + out
        out2 = out1.flatten(2).transpose(1, 2)

        if add_length > 0:
            out2 = out2[:, :-add_length,:]

        return out2


def ca_weight(proj_query, proj_key):
    [b, c, h, w] = proj_query.shape
    proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h).permute(0, 2, 1)
    proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w).permute(0, 2, 1)
    proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)
    proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)
    energy_H = torch.bmm(proj_query_H, proj_key_H).view(b, w, h, h).permute(0, 2, 1, 3)
    energy_W = torch.bmm(proj_query_W, proj_key_W).view(b, h, w, w)
    concate = torch.cat([energy_H, energy_W], 3)

    return concate


def ca_map(attention, proj_value):
    [b, c, h, w] = proj_value.shape
    proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)
    proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)
    att_H = attention[:, :, :, 0:h].permute(0, 2, 1, 3).contiguous().view(b * w, h, h)
    att_W = attention[:, :, :, h:h + w].contiguous().view(b * h, w, w)
    out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(b, w, -1, h).permute(0, 2, 3, 1)
    out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(b, h, -1, w).permute(0, 2, 1, 3)
    out = out_H + out_W
    return out


