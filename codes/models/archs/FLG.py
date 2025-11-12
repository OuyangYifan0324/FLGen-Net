import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from entmax import entmax15

class ColorRestoreBlockV3(nn.Module):
    """
    C -> (C*width) -> (C*width) -> C
    先扩张再压缩，配合 SE 通道注意力
    """
    def __init__(self, channels: int = 3, width: int = 16, r: int = 4):
        super().__init__()
        hidden = channels * width          # 扩张后的通道数

        # ① 1×1 扩张
        self.expand = nn.Conv2d(channels, hidden, 1, bias=False)
        # ② 3×3 深度卷积
        self.dw = nn.Conv2d(hidden, hidden, 3, padding=1,
                             groups=hidden, bias=False)
        # ③ 1×1 压缩
        self.reduce = nn.Conv2d(hidden, channels, 1, bias=False)

        # ④ SE 注意力（仍放在压缩前后均可，这里放压缩后）
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, 8, 1, bias=True)
        self.fc2 = nn.Conv2d(8, channels, 1, bias=True)

        self.relu = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        y = self.relu(self.expand(x))
        y = self.relu(self.dw(y))
        y = self.reduce(y)

        w = self.avg(y)
        w = self.relu(self.fc1(w))
        w = self.sig(self.fc2(w))
        y = y * w                          # 调色
        return x + y                       # 残差

class Lap_Pyramid_Conv(nn.Module):
    def __init__(self, num_high=3):
        super(Lap_Pyramid_Conv, self).__init__()

        self.num_high = num_high
        self.kernel = self.gauss_kernel()
        self.color_rest = nn.ModuleList([
            ColorRestoreBlockV3(3, 16, 4)
            for _ in range(num_high-1)
        ])

    def gauss_kernel(self, device=torch.device('cuda'), channels=3):
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                               [4., 16., 24., 16., 4.],
                               [6., 24., 36., 24., 6.],
                               [4., 16., 24., 16., 4.],
                               [1., 4., 6., 4., 1.]])
        kernel /= 256.
        kernel = kernel.repeat(channels, 1, 1, 1)
        kernel = kernel.to(device)
        return kernel

    def downsample(self, x):
        return x[:, :, ::2, ::2]

    def upsample(self, x):
        cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
        cc = cc.permute(0, 1, 3, 2)
        cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
        x_up = cc.permute(0, 1, 3, 2)
        return self.conv_gauss(x_up, 4 * self.kernel)

    def conv_gauss(self, img, kernel):
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
        out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
        return out

    def pyramid_decom(self, img):
        current = img
        pyr = []
        for i in range(self.num_high):
            filtered = self.conv_gauss(current, self.kernel)
            down = self.downsample(filtered)
            up = self.upsample(down)
            if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
                up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
            diff = current - up
            pyr.append(diff)
            current = down
        pyr.append(current)
        return pyr

    def pyramid_recons(self, pyr):
        image = pyr[-1]
        i=0
        for level in reversed(pyr[:-1]):
            up = self.upsample(image)
            if up.shape[2] != level.shape[2] or up.shape[3] != level.shape[3]:
                up = nn.functional.interpolate(up, size=(level.shape[2], level.shape[3]))
            image = up + level
            if i<= 1:
                image = self.color_rest[i](image)
            i+=1
        return image

pi = math.pi

class RGB_HVI(nn.Module):
    def __init__(self):
        super(RGB_HVI, self).__init__()
        self.density_k = torch.nn.Parameter(torch.full([1], 0.2))  # k is reciprocal to the paper mentioned
        self.gated = False
        self.gated2 = False
        self.alpha = 1.0
        self.alpha_s = 1.3
        self.this_k = 0

    def to_hvi(self, img):
        eps = 1e-8
        device = img.device
        dtypes = img.dtype
        hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(device).to(dtypes)
        value = img.max(1)[0].to(dtypes)
        img_min = img.min(1)[0].to(dtypes)
        hue[img[:, 2] == value] = 4.0 + ((img[:, 0] - img[:, 1]) / (value - img_min + eps))[img[:, 2] == value]
        hue[img[:, 1] == value] = 2.0 + ((img[:, 2] - img[:, 0]) / (value - img_min + eps))[img[:, 1] == value]
        hue[img[:, 0] == value] = (0.0 + ((img[:, 1] - img[:, 2]) / (value - img_min + eps))[img[:, 0] == value]) % 6

        hue[img.min(1)[0] == value] = 0.0
        hue = hue / 6.0

        saturation = (value - img_min) / (value + eps)
        saturation[value == 0] = 0

        hue = hue.unsqueeze(1)
        saturation = saturation.unsqueeze(1)
        value = value.unsqueeze(1)

        k = self.density_k
        self.this_k = k.item()

        color_sensitive = ((value * 0.5 * pi).sin() + eps).pow(k)
        ch = (2.0 * pi * hue).cos()
        cv = (2.0 * pi * hue).sin()
        H = color_sensitive * saturation * ch
        V = color_sensitive * saturation * cv
        I = value
        xyz = torch.cat([H, V, I], dim=1)
        return xyz

    def to_rgb(self, img):
        eps = 1e-8
        H, V, I = img[:, 0, :, :], img[:, 1, :, :], img[:, 2, :, :]

        # clip
        H = torch.clamp(H, -1, 1)
        V = torch.clamp(V, -1, 1)
        I = torch.clamp(I, 0, 1)

        v = I
        k = self.this_k
        color_sensitive = ((v * 0.5 * pi).sin() + eps).pow(k)
        H = (H) / (color_sensitive + eps)
        V = (V) / (color_sensitive + eps)
        H = torch.clamp(H, -1, 1)
        V = torch.clamp(V, -1, 1)
        h = torch.atan2(V + eps, H + eps) / (2 * pi)
        h = h % 1
        s = torch.sqrt(H ** 2 + V ** 2 + eps)

        if self.gated:
            s = s * self.alpha_s

        s = torch.clamp(s, 0, 1)
        v = torch.clamp(v, 0, 1)

        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)

        hi = torch.floor(h * 6.0)
        f = h * 6.0 - hi
        p = v * (1. - s)
        q = v * (1. - (f * s))
        t = v * (1. - ((1. - f) * s))

        hi0 = hi == 0
        hi1 = hi == 1
        hi2 = hi == 2
        hi3 = hi == 3
        hi4 = hi == 4
        hi5 = hi == 5

        r[hi0] = v[hi0]
        g[hi0] = t[hi0]
        b[hi0] = p[hi0]

        r[hi1] = q[hi1]
        g[hi1] = v[hi1]
        b[hi1] = p[hi1]

        r[hi2] = p[hi2]
        g[hi2] = v[hi2]
        b[hi2] = t[hi2]

        r[hi3] = p[hi3]
        g[hi3] = q[hi3]
        b[hi3] = v[hi3]

        r[hi4] = t[hi4]
        g[hi4] = p[hi4]
        b[hi4] = v[hi4]

        r[hi5] = v[hi5]
        g[hi5] = p[hi5]
        b[hi5] = q[hi5]

        r = r.unsqueeze(1)
        g = g.unsqueeze(1)
        b = b.unsqueeze(1)
        rgb = torch.cat([r, g, b], dim=1)
        if self.gated2:
            rgb = rgb * self.alpha
        return rgb

class SEBlock(nn.Module):
    """标准 SE 注意力"""
    def __init__(self, ch, r=8):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(ch, ch // r, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // r, ch, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.avg(x))

class SoftPartition(nn.Module):
    """
    使用卷积 + SE 通道注意力 + Entmax 分区
    """
    def __init__(self, hvi: int, n_partitions: int = 32):
        super().__init__()
        c1, c2 = 4 * hvi, 16 * hvi

        self.extract = nn.Sequential(
            nn.Conv2d(hvi, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),

            nn.Conv2d(c1, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),

            SEBlock(c2, r=2*hvi),

            nn.Conv2d(c2, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Conv2d(c2, n_partitions, kernel_size=1)

    def forward(self, x):  # x: [B,hvi,H,W]
        feat = self.extract(x)
        logits = self.classifier(feat)  # [B, P, H, W]
        return entmax15(logits, dim=1)

def soft_clamp(x, min_val, max_val, k=6.0):
    """平滑地限制 x 在 [min_val, max_val] 范围内"""
    x_norm = (x - min_val) / (max_val - min_val)
    x_smooth = torch.sigmoid(k * (x_norm - 0.5))  # S型曲线，中心值映射为0.5
    return x_smooth * (max_val - min_val) + min_val

class ScaledTanh(nn.Module):
    def __init__(self, scale=0.5, slope=2.5):  # slope 控制斜率
        super().__init__()
        self.scale = scale
        self.slope = slope

    def forward(self, x):
        return self.scale * torch.tanh(self.slope * x)

class fenquEnhancer(nn.Module):
    """
       使用共享 Transformer 进行单通道分区增强：
       - 输入 x: [B,1,H,W] 单通道
       - masks: [B,C,H,W] 分区 softmax 分数
       - 输出: [B,1,H,W] 增强后通道
       """

    def __init__(self,
                 n_parts: int,
                 d_model: int = 32,
                 num_heads: int = 2,
                 ffn_exp: float = 2.66,
                 num_layers: int = 1,
                 scale: float = 0.5,
                 slope: float = 2.5):
        super().__init__()
        self.n_parts = n_parts
        # 将 masks + x 叠加后升维到 d_model
        self.input_proj = nn.Conv2d(n_parts + 1, d_model, kernel_size=1)
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=int(d_model * ffn_exp),
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 输出预测 delta
        self.to_delta = nn.Linear(d_model, 1)
        self.act = nn.Tanh()
        self.scale = scale
        self.slope = slope

    def forward(self, x: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        # 拼接 masks 与 x
        feat = torch.cat([masks, x], dim=1)  # [B, C+1, H, W]
        # 投影至 d_model
        feat = self.input_proj(feat)  # [B, d_model, H, W]
        # flatten 空间
        N = H * W
        tokens = feat.view(B, feat.shape[1], N).transpose(1, 2)  # [B, N, d_model]
        # Transformer 编码
        encoded = self.transformer(tokens)  # [B, N, d_model]
        # 预测 delta
        delta = self.to_delta(encoded)  # [B, N, 1]
        # reshape 回图像
        delta = delta.transpose(1, 2).view(B, 1, H, W)
        # 激活并 scale
        delta = self.act(self.slope * delta) * self.scale
        # 残差增强
        return x + delta

class fenquEnhancerHV(nn.Module):
    """
        共享 Transformer 的 HV 通道增强器：
        - 输入 hv 分区特征 feat: [B, C, H, W]
        - 输出对 H/V 两通道的增量 dh, dv: [B,1,H,W] each
        """

    def __init__(self,
                 feat_dim: int,  # 输入特征通道数 C
                 d_model: int = 128,  # Transformer embed dim
                 num_heads: int = 8,
                 ffn_exp: float = 2.66,
                 num_layers: int = 1,
                 scale: float = 0.5,
                 slope: float = 2.5):
        super().__init__()
        # 1x1 升维至 transformer d_model
        self.input_proj = nn.Conv2d(feat_dim, d_model, kernel_size=1)
        # Positional encoding (可选简化)
        # Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=int(d_model * ffn_exp),
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                 num_layers=num_layers)
        # 输出层：预测 dh, dv
        self.to_deltas = nn.Linear(d_model, 2)
        self.act = nn.Tanh()
        self.scale = scale
        self.slope = slope

    def forward(self, h: torch.Tensor, v: torch.Tensor, feat: torch.Tensor):
        B, C, H, W = feat.shape
        # 投影
        x = self.input_proj(feat)  # [B, d_model, H, W]
        # flatten 空间
        x_flat = x.view(B, x.shape[1], -1).transpose(1, 2)  # [B, N, d_model], N=H*W
        # Transformer 编码
        x_enc = self.transformer(x_flat)  # [B, N, d_model]
        # 预测 delta H/V
        deltas = self.to_deltas(x_enc)  # [B, N, 2]
        # reshape
        deltas = deltas.transpose(1, 2).view(B, 2, H, W)  # [B,2,H,W]
        dh = self.act(self.slope * deltas[:, 0:1]) * self.scale
        dv = self.act(self.slope * deltas[:, 1:2]) * self.scale
        # 返回增强后的 H, V
        return h + dh, v + dv

class HVIEnhancer(nn.Module):
    def __init__(
        self,
        p_hv: int = 32,  # HV 轴分区数
        p_i: int = 16,  # I 轴分区数
    ):
        super().__init__()
        self.hvi     = RGB_HVI()          # 颜色空间转换器
        self.part_hv = SoftPartition(hvi=2, n_partitions=p_hv)
        self.part_i = SoftPartition(hvi=1, n_partitions=p_i)

        self.hv_enhance = fenquEnhancerHV(feat_dim=p_hv, d_model=128, num_heads=8)
        self.i_enhance = fenquEnhancer(n_parts=p_i, d_model=64, num_heads=4)
        self.cross = HV_I_CrossCAB()

    # ---------------------------------------------------------
    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        rgb : [B,3,H,W], 值域 0‑1
        返回增强后 RGB
        """
        hvi = self.hvi.to_hvi(rgb)  # [B,3,H,W]  (H,V,I)
        H, V, I = hvi[:, 0:1], hvi[:, 1:1 + 1], hvi[:, 2:3]
        hv = torch.cat([H, V], dim=1)


        mI = self.part_i(I)
        I_raw = self.i_enhance(I, mI)
        mHV = self.part_hv(hv)
        H_raw, V_raw = self.hv_enhance(H, V, mHV)

        H_enh = soft_clamp(H_raw, -1.0, 1.0, k=6.0)
        V_enh = soft_clamp(V_raw, -1.0, 1.0, k=6.0)
        I_enh = soft_clamp(I_raw, 0.0, 1.0, k=6.0)

        hvi_out = torch.cat([H_enh, V_enh, I_enh], dim=1)
        rgb_out = self.hvi.to_rgb(hvi_out)  # [B,3,H,W]
        return rgb_out

class ALGA(nn.Module):
    """
    Adaptive Luma-Guided Attention
    """
    def __init__(self, in_ch=3, heads=3):
        super().__init__()
        self.heads = heads
        self.scale = (in_ch // heads) ** -0.5

        # 1×1 conv 做 Q/K/V 投影
        self.q_conv = nn.Conv2d(in_ch, in_ch, 1, bias=False)
        self.k_conv = nn.Conv2d(in_ch, in_ch, 1, bias=False)
        self.v_conv = nn.Conv2d(in_ch, in_ch, 1, bias=False)

        # 输出映射
        self.out_conv = nn.Conv2d(in_ch, in_ch, 1, bias=False)

    def forward(self, rgb):
        """
        rgb : [B,3,H,W]   (可直接接 32×32 低频图或深层 feature)
        """
        B, C, H, W = rgb.shape
        h = self.heads
        c_h = C // h

        # --- 亮度偏差 ΔL  ---
        L = 0.299 * rgb[:,0:1] + 0.587 * rgb[:,1:2] + 0.114 * rgb[:,2:3]             # [B,1,H,W]
        mu = L.mean(dim=[2,3], keepdim=True)      # 均值亮度
        dL = torch.abs(L - mu)                    # 偏差
        # 归一化到 0~1
        std = dL.std(dim=[2, 3], keepdim=True) + 1e-6
        dL = dL / std
        # 压缩到 0~1 以增强对低亮度敏感性
        dL = torch.sigmoid(dL)

        # --- ② Q/K/V ---
        q = self.q_conv(rgb) * (1. + dL)          # 对异常像素放大 Query
        k = self.k_conv(rgb)
        v = self.v_conv(rgb)

        # reshape → [B,h,c_h,N]
        N = H*W
        q = q.view(B,h,c_h,N).transpose(-2,-1)    # [B,h,N,c_h]
        k = k.view(B,h,c_h,N)                     # [B,h,c_h,N]
        v = v.view(B,h,c_h,N).transpose(-2,-1)    # [B,h,N,c_h]

        att = torch.matmul(q, k) * self.scale     # [B,h,N,N]
        att = att.softmax(dim=-1)

        out = torch.matmul(att, v)                # [B,h,N,c_h]
        out = out.transpose(-2,-1).contiguous().view(B,C,H,W)
        out = self.out_conv(out)

        # --- ③ 残差补偿 ---
        return torch.clamp(rgb + out, 0.0, 1.0)

# ---------- 主网络 Trans_low -------------------------------------
class Trans_low(nn.Module):
    def __init__(self, num_residual_blocks=4):
        super().__init__()
        self.cnn_branch = ALGA(in_ch=3,heads=1)
        self.hvi_block = HVIEnhancer(p_hv=32, p_i=16)

    def forward(self, x):
        # HVI颜色增强
        rgb_enh = self.hvi_block(x)  # 近似目标增强图像
        delta = self.cnn_branch(rgb_enh)  # 微调/滤波
        # tanh 激活（仍保留）
        out = torch.tanh(x + delta)  # [-1, 1] 范围输出
        return out
# ------------------------------------------------------------------
class ECABlock(nn.Module):
    def __init__(self, channels, k_size=3):
        super(ECABlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # x: [B, C, H, W]
        y = self.avg_pool(x)  # [B, C, 1, 1]
        y = self.conv(y.squeeze(-1).transpose(-1, -2))  # [B, 1, C] -> [B, C, 1]
        y = self.sigmoid(y.transpose(-1, -2).unsqueeze(-1))  # [B, C, 1, 1]
        return x * y.expand_as(x)

class reduce_noise(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.ReLU()
        )
        self.eca = ECABlock(channels=32,k_size=7)
        self.out_conv = nn.Conv2d(32, 3, 3, 1, 1)

    def forward(self, x):
             # [B, 2, H, W]
        feat = self.in_conv(x)
        feat = self.eca(feat)
        out = self.out_conv(feat)
        return x - out

class Trans_high(nn.Module):
    def __init__(self, num_residual_blocks, num_high=3):
        super(Trans_high, self).__init__()

        self.num_high = num_high

        for i in range(self.num_high):
            trans_mask_block = nn.Sequential(
                reduce_noise(),
                nn.Conv2d(3, 16, 1),
                nn.LeakyReLU(),
                nn.Conv2d(16, 3, 1)
                )

            setattr(self, 'trans_mask_block_{}'.format(str(i)), trans_mask_block)
        self.low = Lap_Pyramid_Conv()

    def forward(self, x, pyr_original, fake_low):
        pyr_result = []
        #mask = self.model(x)
        mask = x
        for i in range(self.num_high):
            mask = nn.functional.interpolate(mask, size=(pyr_original[-2-i].shape[2], pyr_original[-2-i].shape[3]))#可修改地方，插值优化
            #根据当前高频图层的大小进行尺寸缩放
            result_highfreq = torch.mul(pyr_original[-2-i], mask) + pyr_original[-2-i]
            #两个相同 shape 的张量逐像素相乘，然后加上原图的高频部分，实现自适应增强：
            self.trans_mask_block = getattr(self, 'trans_mask_block_{}'.format(str(i)))
            result_highfreq = self.trans_mask_block(result_highfreq)
            setattr(self, 'result_highfreq_{}'.format(str(i)), result_highfreq)

        for i in reversed(range(self.num_high)):
            result_highfreq = getattr(self, 'result_highfreq_{}'.format(str(i)))
            pyr_result.append(result_highfreq)
        pyr_result.append(fake_low)
        return pyr_result

class LAM_Module_v2(nn.Module):
    """ Layer attention module"""
    def __init__(self, in_dim =3 ,bias=True):
        super(LAM_Module_v2, self).__init__()
        self.chanel_in = in_dim*3

        self.temperature = nn.Parameter(torch.ones(1))

        self.qkv = nn.Conv2d( self.chanel_in ,  self.chanel_in *3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(self.chanel_in*3, self.chanel_in*3, kernel_size=3, stride=1, padding=1, groups=self.chanel_in*3, bias=bias)
        self.project_out = nn.Conv2d(self.chanel_in, self.chanel_in, kernel_size=1, bias=bias)

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X N X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X N X N
        """
        m_batchsize, N, C, height, width = x.size()

        x_input = x.view(m_batchsize,N*C, height, width)
        qkv = self.qkv_dwconv(self.qkv(x_input))
        q, k, v = qkv.chunk(3, dim=1)
        q = q.view(m_batchsize, N, -1)
        k = k.view(m_batchsize, N, -1)
        v = v.view(m_batchsize, N, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out_1 = (attn @ v)
        out_1 = out_1.view(m_batchsize, -1, height, width)

        out_1 = self.project_out(out_1)
        out_1 = out_1.view(m_batchsize, N, C, height, width)

        out = out_1+x
        out = out.view(m_batchsize, -1, height, width)
        return out

class FLGenNet(nn.Module):
    def __init__(self, nrb_low=5, nrb_high=3, num_high=3):
        super(FLGenNet, self).__init__()

        self.lap_pyramid = Lap_Pyramid_Conv(num_high)
        self.trans_low = Trans_low(nrb_low).cuda()
        self.trans_high = Trans_high(nrb_high, num_high=num_high).cuda()
        self.fusion = LAM_Module_v2(in_dim=3)
        self.conv_fuss = nn.Conv2d(9, 3, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=1, bias=True)

    def forward(self, real_A_full):
        pyr_A = self.lap_pyramid.pyramid_decom(img=real_A_full)
        fake_B_low = self.trans_low(pyr_A[-1])
        real_A_up = F.interpolate(pyr_A[-1], size=pyr_A[-2].shape[2:])
        fake_B_up = F.interpolate(fake_B_low, size=pyr_A[-2].shape[2:])

        inp_fusion_123 = torch.cat([pyr_A[-2].unsqueeze(1), real_A_up.unsqueeze(1), fake_B_up.unsqueeze(1)], dim=1)
        mask = self.fusion(inp_fusion_123)
        mask = self.conv_fuss(mask)
        mask = self.conv2(mask)

        pyr_A_trans = self.trans_high(mask, pyr_A, fake_B_low)
        fake_B_full = self.lap_pyramid.pyramid_recons(pyr_A_trans)

        return fake_B_full

