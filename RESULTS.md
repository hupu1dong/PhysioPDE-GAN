# Evaluation Results

All numbers below are measured with the code in this repository.

## 1. Datasets

| Dataset | Records | Beats | Split |
|---|---|---|---|
| G12EC | 10,344 (15 failed) | 126,370 | 5-fold patient-wise; 2,069 held-out records |
| PTB-XL | 3,420 (of 21,799; quality-filtered) | 39,999 | 5-fold |

Preprocessing: 500 Hz, R-peak detection on lead II with NeuroKit2, window from R−140 to R+260 (400 samples), per-record median/MAD normalisation clipped to ±8.

## 2. Baseline classifier

ResNet1d trained on real data only:

| Dataset | Mean val AUC | IAVB | RBBB | LBBB | NSIVCB | LAnFB | AFL |
|---|---|---|---|---|---|---|---|
| G12EC | 0.949 | 0.955 | 0.963 | 0.976 | 0.943 | 0.944 | 0.927 |
| PTB-XL | 0.927 | 0.916 | 0.975 | 0.978 | 0.799 | 0.939 | 0.965 |

## 3. Generation quality

Diagnostics on 2,000 generated beats per setting:

| Metric | Real | PhysioPDE-GAN | DCGAN |
|---|---|---|---|
| Amplitude std | 0.240 | 0.249 | 0.249 |
| Kurtosis | 7.56 | 6.74 | 5.43 |
| 10–40 Hz energy (QRS band) | 30.6% | 30.9% | 28.0% |
| 0.5–5 Hz energy | 27.0% | 20.9% | 31.4% |
| Lead-correlation distance | — | 2.55 | 0.50 |
| PCA centroid separation | — | 0.85 | 0.63 |

Single-lead morphology statistics (kurtosis, QRS-band energy) are reproduced more closely than with a DCGAN baseline. Inter-lead structure is captured better by the convolutional baseline, which follows from the design: the PDE model couples leads through 5-point differences over 12 points, while DCGAN learns a joint convolution across all 12 leads.

## 4. Lead-level spatial modelling

Replacing the 5-point second difference with a learnable 12×12 dense matrix, combined with 2-head self-attention over the leads:

| Metric | Real | 5-point | **12×12 + attention** | DCGAN |
|---|---|---|---|---|
| Lead-correlation distance | — | 2.87 | **1.64** (−43%) | 0.50 |
| Kurtosis | 7.56 | 7.27 | 7.31 | 5.43 |
| 10–40 Hz energy | 30.6% | 29.9% | 27.8% | 28.0% |
| PCA separation | — | 0.70 | 0.86 | 0.63 |

Parameter count rises from 127.6 K to 143.1 K and training time by about 65%.

## 5. PDE component ablation

Fixed budget (fold 0, 20 epochs, n=2000), removing one component at a time:

| Variant | Lead-corr dist ↓ | Kurtosis (real 7.56) | QRS band (real 30.6%) | PCA sep ↓ |
|---|---|---|---|---|
| Full model | 2.85 | 6.38 | 20.6% | **1.07** |
| Without spatial operator | **4.75** | 6.70 | 25.1% | 1.18 |
| Without subsystem coupling | 2.46 | 6.37 | 24.7% | **1.53** |
| Without temporal gating | **4.47** | 6.16 | 24.3% | 1.42 |

Reading:

- The **spatial operator** is the main source of lead structure: removing it raises the distance from 2.85 to 4.75 (+67%). At 40 epochs the same comparison is 2.87 → 1.64.
- **Temporal gating** also matters (4.47): without it the three subsystem outputs overlap in time and their sum degrades lead relationships.
- **Subsystem coupling** slightly lowers the lead distance (2.46) but worsens the overall distribution (PCA separation 1.53 vs 1.07), so its value lies in waveform morphology rather than spatial correlation.
- The full model is the only configuration that is best on PCA separation (1.07).

Caveat: single budget (20 epochs), single run, no seed replication. At 20 epochs the model has not converged; the full model reaches lead distance 1.64 and kurtosis 7.31 at 40 epochs. The table indicates the direction of each component's contribution, not a precise ranking.

## 6. Downstream specificity

Target sensitivity is the baseline's measured sensitivity; all settings are compared at the same threshold on a held-out test set. QAb has no positive samples in G12EC.

| Class | Real | +DCGAN | +PhysioPDE-GAN |
|---|---|---|---|
| IAVB | 0.904 | 0.889 | 0.888 |
| RBBB | 0.938 | 0.933 | 0.951 |
| LBBB | 0.973 | 0.975 | 0.975 |
| NSIVCB | 0.849 | 0.849 | 0.864 |
| LAnFB | 0.868 | 0.873 | 0.887 |
| AFL | 0.758 | 0.771 | 0.760 |

Five-fold means. AUC differences across settings are at most 0.006.

## 7. PTB-XL

| Setting | Mean val AUC |
|---|---|
| Real only | 0.927 |
| + PhysioPDE-GAN | 0.926 |
| + DCGAN | 0.927 |
