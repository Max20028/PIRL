from torch import nn, optim
import torch.nn.functional as F
import torch
import numpy as np

class UNet(nn.Module):

    @staticmethod
    def _double_conv_block(in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def __init__(self, T_in, T_out, C, num_down_layers=4,alpha=0.01):
        super().__init__()
        # T_in = number of previous timesteps we are using to make the prediction
        down_channels = np.array([T_in * C] + [64 * (2 ** i)
                                               for i in range(num_down_layers)
                                               ]).astype(int)
        self.pool = nn.MaxPool2d(2)
        self.down = nn.ModuleList([
            self._double_conv_block(i, j, 3)
            for i, j in zip(down_channels[:-1], down_channels[1:])
        ])
        bottleneck_channels = down_channels[-1] * 2
        self.bottleneck = self._double_conv_block(down_channels[-1], bottleneck_channels, 3)
        skip_channels = down_channels[1:][::-1]  # channels of skip tensors
        bottleneck_channels = down_channels[-1] * 2

        up_in_channels = []
        up_out_channels = []

        prev_up_out = bottleneck_channels
        for i, skip_ch in enumerate(skip_channels):
            up_in_channels.append(prev_up_out + skip_ch)
            up_out_channels.append(skip_ch)  # intermediate up block outputs same as skip
            prev_up_out = skip_ch

        # final up block
        up_in_channels.append(prev_up_out)  # no concatenation
        up_out_channels.append(T_out * C)  # final output
        self.up = nn.ModuleList([
            self._double_conv_block(i, j, 3)
            for i, j in zip(up_in_channels, up_out_channels)
        ])
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.T_in = T_in
        self.T_out = T_out

    def _net_forward(self, x):
        skip_connections = []

        # Down path
        for down_block in self.down:
            x = down_block(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Up path
        for i, up_block in enumerate(self.up):
            if i < len(self.up) - 1:  # not final output
                skip = skip_connections[-(i + 1)]
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
                x = torch.cat([x, skip], dim=1)
            x = up_block(x)

        return x

    def forward(self, x):
        B, T_in, H, W, C = x.shape                  # (B, T_in,   H, W, C)
        x = x.permute(0, 1, 4, 2, 3)                # (B, T_in,   C, H, W)
        x = x.reshape(B, T_in * C, H, W)            # (B, T_in  * C, H, W)
        out = self._net_forward(x)                  # (B, T_out * C, H, W)
        out = out.reshape(B, self.T_out, C, H, W)   # (B, T_out,  C, H, W)
        out = out.permute(0, 1, 3, 4, 2)            # (B, T_out,  H, W, C)
        return out