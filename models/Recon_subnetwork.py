# https://github.com/openai/guided-diffusion/tree/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924
import math
from abc import abstractmethod

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb): # emb will be the combined_embed
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class PositionalEmbedding(nn.Module):
    # PositionalEmbedding
    """
    Computes Positional Embedding of the timestep
    """

    def __init__(self, dim, scale=1):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.scale = scale

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = torch.outer(x * self.scale, emb)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample(nn.Module):
    def __init__(self, in_channels, use_conv, out_channels=None):
        super().__init__()
        self.channels = in_channels
        out_channels = out_channels or in_channels
        if use_conv:
            # downsamples by 1/2
            self.downsample = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        else:
            assert in_channels == out_channels
            self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x, combined_embed=None): # Changed time_embed to combined_embed for clarity if it were used
        assert x.shape[1] == self.channels
        return self.downsample(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, use_conv, out_channels=None):
        super().__init__()
        self.channels = in_channels
        self.use_conv = use_conv
        # uses upsample then conv to avoid checkerboard artifacts
        # self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        if use_conv:
            self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x, combined_embed=None): # Changed time_embed to combined_embed for clarity if it were used
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(self, in_channels, n_heads=1, n_head_channels=-1):
        super().__init__()
        self.in_channels = in_channels
        self.norm = GroupNorm32(32, self.in_channels)
        if n_head_channels == -1:
            self.num_heads = n_heads
        else:
            assert (
                    in_channels % n_head_channels == 0
            ), f"q,k,v channels {in_channels} is not divisible by num_head_channels {n_head_channels}"
            self.num_heads = in_channels // n_head_channels

        # query, key, value for attention
        self.to_qkv = nn.Conv1d(in_channels, in_channels * 3, 1)
        self.attention = QKVAttention(self.num_heads)
        self.proj_out = zero_module(nn.Conv1d(in_channels, in_channels, 1))

    def forward(self, x, combined_embed=None): # Added combined_embed argument, though not directly used in this original AttentionBlock
        b, c, *spatial = x.shape
        x_norm = self.norm(x) # Apply norm before reshaping
        x_reshaped = x_norm.reshape(b, c, -1)
        qkv = self.to_qkv(x_reshaped)
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x.reshape(b, c, -1) + h).reshape(b, c, *spatial)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, combined_embed=None): # Added combined_embed for consistency, not used here
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
                "bct,bcs->bts", q * scale, k * scale
                )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)


class ResBlock(TimestepBlock):
    def __init__(
            self,
            in_channels,
            embed_dim, # Renamed from time_embed_dim for clarity as it's now combined
            dropout,
            out_channels=None,
            use_conv=False,
            up=False,
            down=False
            ):
        super().__init__()
        out_channels = out_channels or in_channels
        self.in_layers = nn.Sequential(
                GroupNorm32(32, in_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels, out_channels, 3, padding=1)
                )
        self.updown = up or down

        if up:
            self.h_upd = Upsample(in_channels, False) # Original uses in_channels, should be out_channels if different? No, applied before conv
            self.x_upd = Upsample(in_channels, False)
        elif down:
            self.h_upd = Downsample(in_channels, False)
            self.x_upd = Downsample(in_channels, False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.embed_layers = nn.Sequential(
                nn.SiLU(),
                nn.Linear(embed_dim, out_channels) # Uses the combined embedding dimension
                )
        self.out_layers = nn.Sequential(
                GroupNorm32(32, out_channels),
                nn.SiLU(),
                nn.Dropout(p=dropout),
                zero_module(nn.Conv2d(out_channels, out_channels, 3, padding=1))
                )
        if out_channels == in_channels:
            self.skip_connection = nn.Identity()
        elif use_conv: # This condition seems to be for skip connection, not general use_conv for ResBlock
            self.skip_connection = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        else:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, combined_embed): # Changed from time_embed to combined_embed
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x_skip = self.x_upd(x) # Apply up/down to x for skip connection
            h = in_conv(h)
        else:
            h = self.in_layers(x)
            x_skip = x # x for skip connection remains unchanged if not up/down

        emb_out = self.embed_layers(combined_embed).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        h = h + emb_out
        h = self.out_layers(h)
        return self.skip_connection(x_skip) + h


class UNetModel(nn.Module):
    # UNet model
    def __init__(
            self,
            img_size,
            base_channels,
            conv_resample=True, # This seems unused in original ResBlock/Downsample/Upsample constructors based on biggan_updown
            n_heads=1,
            n_head_channels=-1,
            channel_mults="",
            num_res_blocks=2,
            dropout=0,
            attention_resolutions="32,16,8",
            biggan_updown=True, # If True, ResBlock handles up/down, else separate Downsample/Upsample layers
            in_channels=1,
            # New parameter for current feature dimension, though fixed to 3 for embedding layer
            # current_feature_dim=3 # Not needed if embedding layer input fixed to 3
            ):
        self.dtype = torch.float32 # Default dtype
        super().__init__()

        if isinstance(img_size, int): # Ensure img_size is a tuple/list if it's used for indexing
            img_size_tuple = (img_size, img_size)
        else:
            img_size_tuple = img_size


        if channel_mults == "":
            if img_size_tuple[0] == 512:
                channel_mults = (0.5, 1, 1, 2, 2, 4, 4)
            elif img_size_tuple[0] == 256:
                channel_mults = (1, 1, 2, 2, 4, 4)
            elif img_size_tuple[0] == 128:
                channel_mults = (1, 1, 2, 3, 4)
            elif img_size_tuple[0] == 64:
                channel_mults = (1, 2, 3, 4)
            elif img_size_tuple[0] == 32: # Added for 32x32, adjust if needed
                channel_mults = (1, 2, 2, 2) # Example, adjust as needed
            else:
                raise ValueError(f"unsupported image size: {img_size_tuple[0]}")
        attention_ds = []
        for res in attention_resolutions.split(","):
            attention_ds.append(img_size_tuple[0] // int(res))

        self.image_size = img_size_tuple
        self.in_channels = in_channels
        self.model_channels = base_channels # This is the initial number of channels after first conv
        self.out_channels = in_channels # Output is typically noise of same channels as input
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions # Stored but used to derive attention_ds
        self.dropout = dropout
        self.channel_mult = channel_mults
        self.conv_resample = conv_resample # For Downsample/Upsample if not biggan_updown

        self.dtype = torch.float32 # Explicitly set
        self.num_heads = n_heads
        self.num_head_channels = n_head_channels

        time_embed_dim = base_channels * 4
        self.time_embedding = nn.Sequential(
                PositionalEmbedding(base_channels, 1), # Input to PositionalEmbedding is base_channels
                nn.Linear(base_channels, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
                )
        
        # New: Current feature embedding layer
        # Assumes raw current_features have 3 dimensions. Output matches time_embed_dim.
        self.current_feature_embedding_layer = nn.Linear(3, time_embed_dim)

        # Input block
        self.down = nn.ModuleList(
                [TimestepEmbedSequential(nn.Conv2d(self.in_channels, self.model_channels, 3, padding=1))]
                )
        
        ch = self.model_channels # Current number of channels, starting with model_channels
        channels_list = [ch] # Keep track of channels at each resolution for skip connections
        ds = 1 # Downsampling factor

        # Downsampling path
        for i, mult in enumerate(channel_mults):
            out_ch = self.model_channels * int(mult) # Target channels for this level
            for _ in range(num_res_blocks):
                layers = [ResBlock(
                        ch, # Input channels
                        embed_dim=time_embed_dim, # Combined embedding dim
                        out_channels=out_ch,
                        dropout=dropout,
                        )]
                ch = out_ch # Update current channels
                if ds in attention_ds:
                    layers.append(
                            AttentionBlock(
                                    ch,
                                    n_heads=n_heads,
                                    n_head_channels=n_head_channels,
                                    )
                            )
                self.down.append(TimestepEmbedSequential(*layers))
                channels_list.append(ch) # Store channels for skip connection

            if i != len(channel_mults) - 1: # If not the last multiplier block, add downsampling
                self.down.append(
                        TimestepEmbedSequential(
                                ResBlock( # Using ResBlock for downsampling if biggan_updown
                                        ch,
                                        embed_dim=time_embed_dim,
                                        out_channels=ch, # ResBlock for downsampling usually keeps channels same before its internal conv
                                        dropout=dropout,
                                        down=True
                                        )
                                if biggan_updown # biggan_updown uses ResBlock with down=True
                                else # else, use explicit Downsample layer
                                Downsample(ch, self.conv_resample, out_channels=ch) # Original uses out_channels=out_channels. If out_channels is different, then ResBlock above should output ch
                                )
                        )
                ds *= 2
                channels_list.append(ch) # Store channels after downsampling for skip

        # Middle block
        self.middle = TimestepEmbedSequential(
                ResBlock(
                        ch,
                        embed_dim=time_embed_dim,
                        dropout=dropout
                        ),
                AttentionBlock(
                        ch,
                        n_heads=n_heads,
                        n_head_channels=n_head_channels
                        ),
                ResBlock(
                        ch,
                        embed_dim=time_embed_dim,
                        dropout=dropout
                        )
                )
        
        # Upsampling path
        self.up = nn.ModuleList([])
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_ch = self.model_channels * int(mult)
            for j in range(num_res_blocks + 1): # +1 for the block that might include upsampling
                skip_ch = channels_list.pop()
                layers = [
                    ResBlock(
                            ch + skip_ch, # Input channels from previous layer + skip connection
                            embed_dim=time_embed_dim,
                            out_channels=out_ch,
                            dropout=dropout
                            )
                    ]
                ch = out_ch # Update current channels to the output of ResBlock

                if ds in attention_ds: # ds corresponds to the resolution before upsampling
                    layers.append(
                            AttentionBlock(
                                    ch, # Attention on the current channel depth
                                    n_heads=n_heads,
                                    n_head_channels=n_head_channels
                                    ),
                            )

                if i != 0 and j == num_res_blocks: # If not the first multiplier block (i.e., i!=0 means not the highest res) and it's the last ResBlock for this level
                    layers.append(
                            ResBlock( # Using ResBlock for upsampling if biggan_updown
                                    ch, # Input channels to upsampling ResBlock
                                    embed_dim=time_embed_dim,
                                    out_channels=ch, # Output channels (usually same before internal conv)
                                    dropout=dropout,
                                    up=True
                                    )
                            if biggan_updown
                            else # else, use explicit Upsample layer
                            Upsample(ch, self.conv_resample, out_channels=ch)
                            )
                    ds //= 2
                self.up.append(TimestepEmbedSequential(*layers))

        # Output block
        self.out = nn.Sequential(
                GroupNorm32(32, ch), # ch should be base_channels * channel_mults[0] ideally
                nn.SiLU(),
                zero_module(nn.Conv2d(ch, self.out_channels, 3, padding=1))
                )

    def forward(self, x, time, current_features): # Added current_features
        time_embed = self.time_embedding(time)
        current_embed = self.current_feature_embedding_layer(current_features)
        combined_embed = time_embed + current_embed # Combine embeddings

        skips = []
        h = x.type(self.dtype)
        for module in self.down:
            h = module(h, combined_embed) # Pass combined_embed
            skips.append(h)
        
        # The last element of `skips` is the output of the last downsampling block,
        # which is the input to the middle block.
        # The `self.down` list includes the initial conv and all downsampling stages.
        # So, `h` is already the correct input for `self.middle`.
        h = self.middle(h, combined_embed) # Pass combined_embed
        
        for module in self.up:
            h = torch.cat([h, skips.pop()], dim=1)
            h = module(h, combined_embed) # Pass combined_embed
            
        h = h.type(x.dtype) # Ensure correct dtype before final output
        return self.out(h)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def update_ema_params(target, source, decay_rate=0.9999):
    targParams = dict(target.named_parameters())
    srcParams = dict(source.named_parameters())
    for k in targParams:
        targParams[k].data.mul_(decay_rate).add_(srcParams[k].data, alpha=1 - decay_rate)


if __name__ == "__main__":
    args = {
        'img_size':          (64, 64), # Changed to tuple based on usage
        'base_channels':     32,     # From args1.json
        'dropout':           0,      # From args1.json
        'num_heads':         4,      # From args1.json
        'attention_resolutions': "8,4", # From args1.json
        'num_head_channels': -1,    # From args1.json
        'channel_mults': (1, 2, 3, 4), # For 64x64, from UNetModel default logic
        'in_channels':       3,       # From args1.json
        'Batch_Size':        64
        }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNetModel(
            img_size=args['img_size'], 
            base_channels=args['base_channels'], 
            channel_mults=args['channel_mults'], # Providing it directly for clarity
            dropout=args["dropout"], 
            n_heads=args["num_heads"], 
            n_head_channels=args["num_head_channels"],
            attention_resolutions=args["attention_resolutions"],
            in_channels=args['in_channels']
            ).to(device)

    batch_size = args['Batch_Size']
    dummy_x = torch.randn(batch_size, args['in_channels'], args['img_size'][0], args['img_size'][1]).to(device)
    dummy_t = torch.randint(0, 1000, (batch_size,), device=device).float() # Example timesteps
    dummy_current_features = torch.randn(batch_size, 3).to(device) # Example current features (N,3)
    
    print("Input x shape:", dummy_x.shape)
    print("Input time shape:", dummy_t.shape)
    print("Input current_features shape:", dummy_current_features.shape)
    
    output = model(dummy_x, dummy_t, dummy_current_features)
    print("Output shape:", output.shape)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")