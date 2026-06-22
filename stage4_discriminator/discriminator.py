"""
PatchGAN Discriminator for Stage 2 (MAU-Net Corrector)
======================================================

Adversarial discriminator following Nathanael & Prasetyo (2024), section F.
Used as a light realism regularizer in Stage 2 training: it judges whether the
MAU-Net's recolored image looks like a real photo. The generator (MAU-Net) is
trained partly to fool it, which discourages unnatural recoloring artifacts.

PatchGAN (vs. a normal GAN):
  Instead of one real/fake score for the whole image, it outputs a grid of
  scores, each judging one local patch. This enforces consistent quality
  across the whole image, not just a globally plausible average. (Paper notes
  this fixes the common colorization problem of some regions staying uncolored.)

Spec (paper section F):
  - 5 convolutional layers, BatchNorm + LeakyReLU
  - total spatial downsampling of 8x  -> a 256x256 input yields a 32x32 grid
  - 256 output filters before the final score map
  - input: the recolored RGB image (3 channels) -- "method 1", per the team's
    choice (judge the recolored image alone, no conditioning on the original)

Output:
  A score map [B, 1, 32, 32] of raw logits (one logit per patch). Pair it with
  BCEWithLogitsLoss in the training loop (numerically stable; no sigmoid here).
"""

import torch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    """PatchGAN discriminator.

    Args:
        in_ch:    input channels (3 = recolored RGB, method 1).
        base:     filters in the first conv layer (doubles each stage up to 256).

    Input  : image [B, in_ch, 256, 256] in [-1, 1] (MAU-Net tanh range).
    Output : patch logits [B, 1, 32, 32]  (apply sigmoid/BCEWithLogits in loss).
    """

    def __init__(self, in_ch=3, base=64):
        super().__init__()

        def block(i, o, k, stride, norm=True):
            # k=4 for downsampling stages, k=3 for size-preserving stages.
            pad = 1
            layers = [nn.Conv2d(i, o, kernel_size=k, stride=stride, padding=pad, bias=not norm)]
            if norm:
                layers.append(nn.BatchNorm2d(o))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # 5 conv layers. Three stride-2 (4x4) layers give the 8x downsample
        # (256 -> 128 -> 64 -> 32). Then a size-preserving (3x3, pad 1) layer
        # keeps 256 filters at 32x32, and a final 3x3 layer outputs the 1-channel
        # patch score map -- staying at 32x32 to match the paper.
        layers = []
        layers += block(in_ch, base, k=4, stride=2, norm=False)  # 256 -> 128, no BN on first
        layers += block(base, base * 2, k=4, stride=2)           # 128 -> 64   (128 ch)
        layers += block(base * 2, base * 4, k=4, stride=2)       # 64  -> 32   (256 ch)
        layers += block(base * 4, base * 4, k=3, stride=1)       # 32  -> 32   (256 ch, keeps 256 filters)
        layers += [nn.Conv2d(base * 4, 1, kernel_size=3, stride=1, padding=1)]  # 32 -> 32 logits
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)   # [B, 1, ~32, ~32] logits


if __name__ == "__main__":
    disc = PatchGANDiscriminator(in_ch=3)
    n_params = sum(p.numel() for p in disc.parameters())
    print(f"PatchGAN params: {n_params:,}")

    x = torch.randn(2, 3, 256, 256)   # recolored RGB batch
    out = disc(x)
    print("input :", tuple(x.shape))
    print("output:", tuple(out.shape), "(patch logits)")

    # Check the downsampling is ~8x (256 -> ~32)
    h_out = out.shape[2]
    print(f"spatial downsample: 256 -> {h_out}  (~{256 // h_out}x)")

    # Check the layer that should hold 256 filters
    convs = [m for m in disc.model if isinstance(m, nn.Conv2d)]
    print("conv layers:", len(convs))
    print("filter progression:", [c.out_channels for c in convs])

    # Gradient flow check
    out.mean().backward()
    g = next(disc.parameters()).grad
    print("gradient flows:", g is not None and float(g.abs().sum()) > 0)

    # BCEWithLogits usability check (real=1, fake=0 labels)
    bce = nn.BCEWithLogitsLoss()
    real_target = torch.ones_like(out)
    loss = bce(out, real_target)
    print("BCEWithLogits loss computes:", float(loss))
    print("\nPatchGAN discriminator OK.")
