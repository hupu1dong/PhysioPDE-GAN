# PhysioPDE-GAN

Reference implementation of *PDE-Driven Spatiotemporal Generative Modeling for Multilead ECG Synthesis* (AAAI-26).

## Overview

We model a 12-lead heartbeat as the numerical solution of a reaction–diffusion PDE system. Three coupled subsystems — sinoatrial (SA), atrioventricular (AV) and His–Purkinje (HP) — evolve the 12-lead voltage vector over K discrete time steps. Spatial derivatives along the lead dimension are implemented as learnable finite-difference operators with moment constraints, so inter-lead coupling is learned from data rather than assumed.

Three properties distinguish the formulation:

- **The generator is a solver.** Each PDE block performs one forward-Euler advance; stacking K blocks integrates the system over the heartbeat window.
- **Spatial coupling is trainable.** $D_x$ and $D_{xx}$ are 1D convolutions whose kernels are projected onto the constraint null space, so they remain valid first- and second-order differences throughout training.
- **Physiology enters through structure and window.** The SA→AV→HP stimulus chain enforces conduction order, and a temporal gating loss confines each component to its activation window (P wave, PR segment, QRS-T).

## Repository layout

```
src/
  models/
    pde.py          Moment-constrained difference operators; PDE evolution block
    gan.py          Generator (three coupled subsystems), discriminator, losses
    classifier.py   ResNet1d for downstream evaluation
    baselines.py    DCGAN and WaveGAN baselines
  data/
    preprocess.py         PTB-XL preprocessing
    preprocess_g12ec.py   G12EC preprocessing
    labels.py             Label mapping
    select_records.py     Record selection
  train_gan.py            Train the generator (5 folds)
  generate.py             Sample synthetic heartbeats
  realign_synthetic.py    Circular shift to align R peaks
  train_clf.py            Train the downstream classifier
  evaluate.py             Fixed-sensitivity / compare-specificity protocol
  paired_test.py          5-fold paired t-test

scripts/
  run_5fold_g12.sh        Full 5-fold pipeline on G12EC
  run_expA.sh             Attention classifier variant
  run_expBC.sh            Augmentation ratio and targeted augmentation
  run_expD.sh             Synthetic pretraining followed by finetuning
  run_ptbxl_gan.sh        Generator training on PTB-XL
  run_ptbxl_clf_after.sh  PTB-XL downstream experiments
  analyze_*.py            Result aggregation and significance testing
```

## Environment

```
Python 3.11
PyTorch 2.4 (CUDA 12.4)
numpy, scipy, scikit-learn, neurokit2, wfdb
```

```bash
pip install -r requirements.txt
```

## Data

Datasets are not redistributed here:

| Dataset | Source | Role |
|---|---|---|
| G12EC | PhysioNet Challenge 2020 | Primary cohort |
| PTB-XL | PhysioNet | Secondary cohort |

```bash
python src/data/preprocess_g12ec.py --root <G12EC path> --out data/processed/g12ec_beats.npz
python src/data/preprocess.py --root <PTB-XL path> --out data/processed/ptbxl_beats.npz
```

Records are resampled to 500 Hz, R peaks are detected on lead II, beats are cut from R−140 to R+260 samples (400 points), and each record is normalised by median and MAD. Splits are patient-wise.

## Training

```bash
# Generator, 5 folds (about 110 minutes per fold)
bash scripts/run_5fold_g12.sh

# Sample synthetic beats
python src/generate.py --npz data/processed/g12ec_beats.npz \
  --ckpt checkpoints/g12_f0/physiopde_gan_last.pt --fold 0 \
  --n 110000 --out data/synthetic/pde_g12_f0.npz
```

Default hyperparameters: 40 epochs, batch 128, Adam with lr 3e-4 for the generator and 1e-4 for the discriminator, hinge loss with R1 regularisation. The generator has 127.6 K parameters, or 143.1 K with the global spatial operator enabled.

## Evaluation

Downstream evaluation fixes the classifier sensitivity at a threshold and compares specificity on a held-out test set of unseen real samples, following the protocol in the paper.

```bash
python src/train_clf.py --fold 0 --tag real
python src/train_clf.py --fold 0 --syn data/synthetic/pde_g12_f0.npz --tag real+synthetic
python src/evaluate.py --baseline-tag real
python src/paired_test.py
```

## Generation quality

Measured on 2,000 generated beats per setting:

| Metric | Real | PhysioPDE-GAN | DCGAN |
|---|---|---|---|
| Amplitude std | 0.240 | 0.249 | 0.249 |
| Kurtosis | 7.56 | 6.74 | 5.43 |
| 10–40 Hz energy (QRS band) | 30.6% | 30.9% | 28.0% |
| 0.5–5 Hz energy | 27.0% | 20.9% | 31.4% |
| Lead-correlation distance | — | 2.55 | 0.50 |
| PCA centroid separation | — | 0.85 | 0.63 |

The generator reproduces single-lead morphology statistics (kurtosis, QRS-band energy) more closely than a DCGAN baseline, while the convolutional baseline captures inter-lead structure better; the difference follows from the design, since the PDE model couples leads through 5-point differences over 12 points.

## Implementation notes

| Item | Setting |
|---|---|
| Time steps | 400, split into 8 segments with separate PDE parameters |
| Phase features | 16 harmonics, covering the 10–40 Hz QRS band |
| Discriminator | 4-layer 1D CNN (64/128/256/256) |
| PDE residual | Crank–Nicolson midpoint form |
| Losses | Adversarial, PDE residual, temporal gating, amplitude, spectral, kurtosis, R-peak alignment |
| Spatial operator | 5-point convolution by default; 12×12 global matrix and lead attention available |

## Citation

```bibtex
@inproceedings{yehuda2026physiopde,
  title={PDE-Driven Spatiotemporal Generative Modeling for Multilead ECG Synthesis},
  author={Yehuda, Yakir and Radinsky, Kira},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```

## License

MIT.
