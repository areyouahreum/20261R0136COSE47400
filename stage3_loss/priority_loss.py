"""
Priority-Weighted Loss for Stage 2 (MAU-Net Corrector)
======================================================

Implements the project's core idea (midterm slides p.13, p.15): the recolored
image should make safety-critical regions DISTINGUISHABLE to a CVD viewer,
while PRESERVING NATURALNESS elsewhere. The priority map (U-Net output) decides
where each goal matters.

Three terms (combined as a weighted sum):

  (1) Distinguishability term  -- "critical regions stay recognizable"
      Computed on the CVD-SIMULATED recolored image. We want high-priority
      regions to remain perceptually rich after the colorblind goggle, so we
      penalize loss of perceptual feature content there using a VGG perceptual
      loss between the CVD-simulated recolored image and the CVD-simulated
      ORIGINAL... no: we compare against the ORIGINAL (non-CVD) image's VGG
      features, so the model is pushed to preserve, *through* CVD eyes, the
      feature content a normal viewer would see. Priority-weighted so traffic
      lights count more than sky.

  (2) Naturalness term (L1)  -- "don't wreck the rest of the image"
      L1 distance between recolored and original RGB, weighted by (1 - priority)
      so LOW-priority regions (sky, buildings) are strongly anchored to the
      original while high-priority regions are free to change.

  (3) Adversarial term  -- "look like a real photo" (PatchGAN, added later)
      Placeholder hook; wired in once the discriminator exists.

Final loss = lambda_distinct * (1) + lambda_natural * (2) + lambda_adv * (3)

Design notes for teammates are in the Stage 2 README.

All terms are differentiable and operate on tensors in [-1, 1] (MAU-Net tanh
range) by default; set in01=True if your images are already in [0, 1].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def _to01(x, in01):
    """Map an image tensor to [0,1] for VGG / matrix ops."""
    return x if in01 else (x + 1.0) * 0.5


class VGGPerceptual(nn.Module):
    """Perceptual feature extractor using early VGG16 layers (relu1_2, relu2_2),
    following the colorization paper's choice of shallow layers. Frozen.

    Input images must be in [0,1] and will be ImageNet-normalized internally.
    Returns a list of feature maps.
    """

    def __init__(self):
        super().__init__()
        weights = torchvision.models.VGG16_Weights.IMAGENET1K_V1
        vgg = torchvision.models.vgg16(weights=weights).features
        # relu1_2 is index 3, relu2_2 is index 8 in vgg16.features
        self.slice1 = nn.Sequential(*[vgg[i] for i in range(4)])    # -> relu1_2
        self.slice2 = nn.Sequential(*[vgg[i] for i in range(4, 9)])  # -> relu2_2
        for p in self.parameters():
            p.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x01):
        x = (x01 - self.mean) / self.std
        f1 = self.slice1(x)
        f2 = self.slice2(f1)
        return [f1, f2]


class PriorityWeightedLoss(nn.Module):
    """Combined Stage 2 loss.

    Args:
        cvd_sim:          a CVDSimulation module (from cvd_simulation.py),
                          already configured with the desired severity.
        lambda_distinct:  weight for the distinguishability (perceptual) term.
        lambda_natural:   weight for the naturalness (L1) term.
        lambda_adv:       weight for the adversarial term (0 until discriminator
                          is connected).
        lambda_contrast:  weight for the CVD-CONTRAST term (NEW). This is the
                          term that actually pushes high-priority regions to be
                          MORE distinguishable from their surroundings *as seen
                          through CVD eyes* — the perceptual term alone only
                          preserves content, it does not increase separation.
                          Set to 0.0 to recover the original two-term behavior.
        in01:             True if images are in [0,1]; False for [-1,1] (default).
        layer_weights:    per-VGG-layer weights for the perceptual term
                          (paper used 0.66 / 0.34 for relu1_2 / relu2_2).

    forward(recolored, original, priority, adv_loss=None) -> (total, parts_dict)
        recolored: [B,3,H,W] MAU-Net output
        original:  [B,3,H,W] input RGB
        priority:  [B,1,H,W] priority map in [0,1] from the frozen U-Net
        adv_loss:  optional pre-computed scalar adversarial loss for term (3)
    """

    def __init__(
        self,
        cvd_sim,
        lambda_distinct=1.0,
        lambda_natural=1.0,
        lambda_adv=0.0,
        lambda_contrast=1.0,
        in01=False,
        layer_weights=(0.66, 0.34),
        contrast_blur_kernel=15,
    ):
        super().__init__()
        self.cvd_sim = cvd_sim
        self.vgg = VGGPerceptual()
        self.lambda_distinct = lambda_distinct
        self.lambda_natural = lambda_natural
        self.lambda_adv = lambda_adv
        self.lambda_contrast = lambda_contrast
        self.in01 = in01
        self.layer_weights = layer_weights
        self.contrast_blur_kernel = contrast_blur_kernel

    def distinguishability_term(self, recolored, original, priority):
        """Priority-weighted perceptual loss between the CVD-simulated recolored
        image and the original's VGG features. High-priority regions are
        weighted more, so the model focuses on keeping critical objects
        perceptually rich even through CVD eyes."""
        # CVD-simulate the recolored image ("what the colorblind viewer sees").
        cvd_recolored = self.cvd_sim(recolored)
        # Compare its perceptual features to the ORIGINAL (normal-vision) image,
        # so we reward preserving the content a normal viewer perceives.
        rec01 = _to01(cvd_recolored, self.in01)
        ori01 = _to01(original, self.in01)

        feats_rec = self.vgg(rec01)
        feats_ori = self.vgg(ori01)

        total = 0.0
        for w, fr, fo in zip(self.layer_weights, feats_rec, feats_ori):
            # squared error per spatial location, averaged over channels
            err = (fr - fo).pow(2).mean(dim=1, keepdim=True)   # [B,1,h,w]
            # resize priority to this feature map's resolution and weight by it
            pri = F.interpolate(priority, size=err.shape[2:], mode="bilinear",
                                align_corners=False)
            total = total + w * (pri * err).mean()
        return total

    def naturalness_term(self, recolored, original, priority):
        """L1 distance weighted by (1 - priority): low-priority regions (sky,
        buildings) are anchored to the original; high-priority regions are free
        to be recolored strongly."""
        l1 = (recolored - original).abs().mean(dim=1, keepdim=True)   # [B,1,H,W]
        weight = 1.0 - priority                                        # low pri -> high weight
        return (weight * l1).mean()

    def contrast_term(self, recolored, priority):
        """CVD-contrast term (the real 'distinguishability' driver).

        Goal: through CVD eyes, a high-priority region (e.g. a traffic light)
        should stand out from its local surroundings. We measure, on the
        CVD-SIMULATED recolored image, the color distance between each pixel and
        a blurred version of itself (a cheap proxy for 'the local surrounding
        color'). We then REWARD large distance in high-priority regions, so the
        loss is the NEGATIVE of the priority-weighted local contrast.

        Because this term is maximized (negated in the loss), the model is
        actively pushed to make critical regions pop after the colorblind
        goggle — which the perceptual term alone does not do.
        """
        cvd_recolored = self.cvd_sim(recolored)
        rec01 = _to01(cvd_recolored, self.in01)

        # Local mean color via average-pool blur (separable, differentiable).
        k = self.contrast_blur_kernel
        pad = k // 2
        local_mean = F.avg_pool2d(rec01, kernel_size=k, stride=1, padding=pad,
                                  count_include_pad=False)
        # Per-pixel color distance to local surroundings (CVD space).
        local_contrast = (rec01 - local_mean).pow(2).mean(dim=1, keepdim=True)  # [B,1,H,W]

        # Reward contrast where priority is high -> loss is the negative.
        # Normalize by priority mass so it does not collapse to 0 on empty maps.
        pri_mass = priority.mean().clamp(min=1e-4)
        reward = (priority * local_contrast).mean() / pri_mass
        return -reward

    def forward(self, recolored, original, priority, adv_loss=None):
        distinct = self.distinguishability_term(recolored, original, priority)
        natural = self.naturalness_term(recolored, original, priority)
        contrast = self.contrast_term(recolored, priority)

        adv = adv_loss if adv_loss is not None else torch.zeros((), device=recolored.device)

        total = (self.lambda_distinct * distinct
                 + self.lambda_natural * natural
                 + self.lambda_contrast * contrast
                 + self.lambda_adv * adv)

        parts = {
            "total": float(total.detach()),
            "distinct": float(distinct.detach()),
            "natural": float(natural.detach()),
            "contrast": float(contrast.detach()),
            "adv": float(adv.detach()) if torch.is_tensor(adv) else float(adv),
        }
        return total, parts


if __name__ == "__main__":
    import sys
    # Patch CVD embedder-free: cvd_simulation has no downloads, but VGG does.
    # For an offline shape test we skip pretrained VGG weights.
    from cvd_simulation import CVDSimulation

    print("Building loss (VGG may need weights; offline = random init)...")
    try:
        cvd = CVDSimulation(severity=1.0, in01=False)
        loss_fn = PriorityWeightedLoss(cvd, in01=False)
        ok = True
    except Exception as e:
        print("VGG weights unavailable offline:", e)
        ok = False

    if ok:
        B, H, W = 2, 256, 256
        recolored = torch.rand(B, 3, H, W) * 2 - 1   # [-1,1]
        original = torch.rand(B, 3, H, W) * 2 - 1
        priority = torch.rand(B, 1, H, W)            # [0,1]
        recolored.requires_grad_(True)

        total, parts = loss_fn(recolored, original, priority)
        print("loss parts:", parts)
        total.backward()
        print("grad flows to recolored:",
              recolored.grad is not None and float(recolored.grad.abs().sum()) > 0)
        print("\nForward + backward OK.")
