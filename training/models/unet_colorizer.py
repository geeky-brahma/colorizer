import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_ch, out_ch, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, k, 1, p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNetColorizer(nn.Module):
    def __init__(self, in_ch=1, out_ch=2, base=64):
        super().__init__()
        self.enc1 = conv_block(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = conv_block(base * 4, base * 8)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = conv_block(base * 8, base * 16)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, 2)
        self.dec4 = conv_block(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        self.dec3 = conv_block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.dec2 = conv_block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.dec1 = conv_block(base * 2, base)

        self.head = nn.Sequential(
            nn.Conv2d(base, out_ch, kernel_size=1),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        b = self.bottleneck(p4)

        # Upsample and align spatial sizes with corresponding encoder maps
        u4 = self.up4(b)
        if u4.shape[-2:] != e4.shape[-2:]:
            dh = e4.size(2) - u4.size(2)
            dw = e4.size(3) - u4.size(3)
            u4 = F.pad(u4, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        d4 = self.dec4(torch.cat([u4, e4], dim=1))

        u3 = self.up3(d4)
        if u3.shape[-2:] != e3.shape[-2:]:
            dh = e3.size(2) - u3.size(2)
            dw = e3.size(3) - u3.size(3)
            u3 = F.pad(u3, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        d3 = self.dec3(torch.cat([u3, e3], dim=1))

        u2 = self.up2(d3)
        if u2.shape[-2:] != e2.shape[-2:]:
            dh = e2.size(2) - u2.size(2)
            dw = e2.size(3) - u2.size(3)
            u2 = F.pad(u2, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        if u1.shape[-2:] != e1.shape[-2:]:
            dh = e1.size(2) - u1.size(2)
            dw = e1.size(3) - u1.size(3)
            u1 = F.pad(u1, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        out = self.head(d1)
        return out
