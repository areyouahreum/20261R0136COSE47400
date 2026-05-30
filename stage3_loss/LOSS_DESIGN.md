# Stage 2 Loss Design — Priority-Weighted Loss

This document explains *why* the Stage 2 loss is built the way it is, so the
whole team can read training curves and results with a shared understanding.

## Goal (from the midterm slides)

The midterm presentation states two goals for the recolored image:

- **Distinguishability of critical regions** (slide 13) — a CVD viewer should
  be able to tell safety-critical objects (traffic lights, signs) apart.
- **Preserving overall naturalness** (slides 13, 15) — low-priority regions
  (sky, buildings) should be *left natural*, not needlessly altered.

The loss is simply these two goals written as math, with the **priority map**
(from the frozen Stage 1 U-Net) deciding *where* each goal applies.

## The three terms

Final loss = `λ_distinct · (1)` + `λ_natural · (2)` + `λ_adv · (3)`

### (1) Distinguishability term — "critical regions stay recognizable"

- The recolored image is first passed through the **CVD simulation** (the
  "colorblind goggle", `cvd_simulation.py`) to get what a deuteranomalous
  viewer would actually see.
- We then compare that CVD-simulated image's **VGG perceptual features** to the
  **original (normal-vision) image's** features. The idea: push the model so
  that, *through CVD eyes*, the perceptual content a normal viewer perceives is
  preserved.
- This term is **weighted by the priority map**, so errors on traffic lights
  count much more than errors on sky.
- Shallow VGG layers (`relu1_2`, `relu2_2`, weighted 0.66 / 0.34) are used,
  following the colorization paper (Nathanael & Prasetyo 2024).

### (2) Naturalness term — "don't wreck the rest of the image"

- Plain **L1 distance** between recolored and original RGB.
- Weighted by **`(1 - priority)`**: low-priority regions (sky, buildings,
  priority ≈ 0.1) are strongly anchored to the original, while high-priority
  regions (traffic lights, priority = 1.0) are free to change.
- Verified behavior: an identical color change is penalized ~0.45 in a
  low-priority region but ~0.0 in a high-priority region. This is exactly the
  "leave low-priority regions natural" requirement from slide 15.

### (3) Adversarial term — "look like a real photo" (added later)

- A PatchGAN discriminator term, following the colorization paper.
- Currently a placeholder (`λ_adv = 0`); wired in once the discriminator
  module exists. It acts as a light regularizer for realism, not the main
  driver — so its weight stays small.

## Why priority weighting is the key idea

The baseline (Deep Correct) applies the same correction everywhere, so traffic
lights get no more attention than a billboard. By weighting term (1) with the
priority map and term (2) with its inverse, the model is pushed to:

- **change high-priority regions strongly** (term 1 dominates there, term 2 is
  nearly off), and
- **leave low-priority regions alone** (term 2 dominates there, term 1 is
  nearly off).

That split is the project's core contribution, expressed directly in the loss.

## Default weights (starting point, to be tuned)

| Weight | Default | Meaning |
|---|---|---|
| `λ_distinct` | 1.0 | distinguishability (perceptual, CVD-simulated) |
| `λ_natural`  | 1.0 | naturalness (L1, inverse-priority weighted) |
| `λ_adv`      | 0.0 → small | adversarial realism (after discriminator) |

These are initial values. Expect to tune the ratio during training — if results
look unnatural, raise `λ_natural`; if critical regions aren't distinct enough,
raise `λ_distinct`.

## Image range convention

All terms operate on MAU-Net's `tanh` output range `[-1, 1]` by default
(`in01=False`). Conversion to `[0, 1]` for VGG / CVD matrix ops is handled
internally. If your pipeline uses `[0, 1]` images, set `in01=True`.
