# Why a PatchGAN Discriminator? — Design Note

This component is **not** in the midterm slides, so this note explains why it
was added, how it connects to the goals we already stated, and why it stays a
minor part of the system.

## Short version

The MAU-Net paper we based Stage 2 on (Nathanael & Prasetyo, 2024) uses a
PatchGAN discriminator as part of its training. We follow the same recipe. Its
job here is narrow: act as a **light realism regularizer** so the recolored
images don't become unnatural while chasing the recoloring objective.

## How it connects to the midterm goals

The midterm slides state two goals (slides 13, 15):

1. make safety-critical regions **distinguishable** to a CVD viewer, and
2. **preserve naturalness** of the overall image.

Our loss already covers both: the distinguishability term and the
inverse-priority L1 naturalness term. So why add adversarial loss?

Because L1 alone has a known failure mode: minimizing pixel-wise distance tends
to produce **dull, washed-out, "averaged" colors** (the safe bet that lowers
average error). That directly undermines goal (2): the image stays close to the
original in L1 terms but looks flat and artificial. The adversarial term pushes
the generator toward outputs that look like **real photographs**, which
reinforces the naturalness goal we already committed to — it's a stronger,
perceptually-grounded version of "keep it natural", not a new direction.

So the discriminator is best understood as **support for the naturalness goal
that's already in the slides**, borrowed from the paper, not a new objective.

## Why "Patch" GAN specifically

A normal GAN gives one real/fake score for the whole image. A PatchGAN outputs
a grid of scores, each judging a small local region. The paper chose this
because it fixes a common colorization problem: with a global score, some
regions can stay poorly colored as long as the image is plausible *on average*.
Per-patch scoring forces **consistent quality everywhere** — which matters for
us too, since we don't want some areas left looking fake.

## Why it stays minor (small weight)

This is the key point for the team: the discriminator is **not the main driver**
of our results. Our contribution is the priority-conditioned recoloring, not
the GAN. So:

- The adversarial loss weight (`lambda_adv`) is kept **small** (the paper also
  weights it far below the L1/perceptual terms — it's a regularizer, not the
  objective).
- If GAN training is unstable or eats time, we can set `lambda_adv = 0` and the
  pipeline still works — we just lose a bit of realism polish. The core
  priority-based recoloring does not depend on it.

## What it actually is (spec)

- Follows Nathanael & Prasetyo (2024), section F.
- 5 conv layers, BatchNorm + LeakyReLU, 8x downsampling, 256x256 -> 32x32 score
  map, 256 filters before the final layer.
- Input (our choice, "method 1"): the recolored RGB image alone (3 channels),
  matching the paper rather than a conditional (pix2pix-style) variant.
- Output: a 32x32 grid of patch logits; paired with `BCEWithLogitsLoss` in the
  training loop (real = 1, fake = 0).

## Summary for the team

> We didn't introduce a new objective. The discriminator is the paper's own
> realism mechanism, included to strengthen the "preserve naturalness" goal
> already in our midterm. It carries a small weight and is optional — the
> priority-based recoloring remains the heart of the project.
