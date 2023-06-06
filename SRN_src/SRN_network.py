import torch
import torch.nn as nn
import torch.nn.functional as F

from SRN_src.module import GatedConv2d, TransposeGatedConv2d


class RegistrationModule(nn.Module):
    def __init__(self):
        super(RegistrationModule, self).__init__()

        # Encoder
        self.conv_1 = GatedConv2d(in_channels=5, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_2 = GatedConv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv_3 = GatedConv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv_4 = GatedConv2d(in_channels=128, out_channels=192, kernel_size=3, stride=2, padding=1)
        self.conv_5 = GatedConv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.conv_6 = GatedConv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1)


        # Decoder
        self.conv_7 = GatedConv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.conv_8 = TransposeGatedConv2d(in_channels=192, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv_9 = TransposeGatedConv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_10 = TransposeGatedConv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_11 = GatedConv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)


    def forward(self, x):

        # Encoder
        out_1 = self.conv_1(x)
        out_2 = self.conv_2(out_1)
        out_3 = self.conv_3(out_2)
        out_4 = self.conv_4(out_3)
        out_5 = self.conv_5(out_4)
        out_6 = self.conv_6(out_5)

        # Decoder
        out_7 = self.conv_7(out_6) + out_4
        out_8 = self.conv_8(out_7) + out_3
        out_9 = self.conv_9(out_8) + out_2
        out_10 = self.conv_10(out_9) + out_1
        out = self.conv_11(out_10)


        return out

class EnhancementModule(nn.Module):
    def __init__(self):
        super(EnhancementModule, self).__init__()

        self.pad = nn.ReflectionPad2d(3)

        # Encoder
        self.conv_1 = self.conv_block(in_channels=1, out_channels=32, kernel_size=7, stride=1, padding=0)
        self.conv_2 = self.conv_block(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv_3 = self.conv_block(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv_4 = self.conv_block(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv_5 = self.conv_block(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv_6 = self.conv_block(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)

        # Necks
        self.conv_7 = self.res_conv_block(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv_8 = self.res_conv_block(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv_9 = self.res_conv_block(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv_10 = self.res_conv_block(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)

        # Decoder
        self.conv_11 = self.conv_block(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv_12 = self.conv_block(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv_13 = self.conv_block(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv_14 = self.conv_block(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_15 = self.conv_block(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_16 = self.conv_block(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)


    def conv_block(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation
            ),
            nn.ReLU(inplace=True)
        )

    def res_conv_block(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1):
        return nn.Sequential(
            self.conv_block(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation),
            self.conv_block(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation),
        )

        
    def forward(self, x):
        x = self.pad(x)
    
        # Encoder
        out_1 = self.conv_1(x)
        out_2 = self.conv_2(out_1)
        out_3 = self.conv_3(out_2)
        out_4 = self.conv_4(out_3)
        out_5 = self.conv_5(out_4)
        out_6 = self.conv_6(out_5)

        # Residual Connection Neck
        out_7 = self.conv_7(out_6) + out_6
        out_8 = self.conv_8(out_7) + out_7
        out_9 = self.conv_9(out_8) + out_8
        out_10 = self.conv_10(out_9) + out_9

        # Decoder
        out_11 = self.conv_11(out_10) + out_5
        out_12 = self.conv_12(out_11) + out_4
        out_12 = F.interpolate(out_12, scale_factor=2, mode='nearest')
        out_13 = self.conv_13(out_12) + out_3
        out_13 = F.interpolate(out_13, scale_factor=2, mode='nearest')
        out_14 = self.conv_14(out_13) + out_2
        out_14 = F.interpolate(out_14, scale_factor=2, mode='nearest')
        out_15 = self.conv_15(out_14) + out_1
        out_16 = self.conv_16(out_15)
        out = out_16

        return out