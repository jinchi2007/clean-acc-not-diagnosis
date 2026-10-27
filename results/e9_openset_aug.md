# E9: open-set rejection under noise-augmented training

Training: per-batch augmentation SNR ~ U[0, 15] dB (same as E4).
Evaluation: two held-out class configurations, softmax-max confidence.

| model | AUROC 3,6,9 | AUROC 1,4,7 | mean AUROC | TPR95 3,6,9 | TPR95 1,4,7 | mean TPR95 | 10 dB | 0 dB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wdcnn | 0.772 | 0.872 | 0.822 | 0.027 | 0.319 | 0.173 | 0.999 | 0.915 |
| ticnn | 0.944 | 0.722 | 0.833 | 0.640 | 0.037 | 0.338 | 1.000 | 0.843 |
| mcnn | 0.966 | 0.726 | 0.846 | 0.834 | 0.247 | 0.541 | 1.000 | 0.628 |
| resnet1d | 0.955 | 0.759 | 0.857 | 0.784 | 0.312 | 0.548 | 0.999 | 0.623 |
| ds_cnn | 0.727 | 0.561 | 0.644 | 0.109 | 0.236 | 0.173 | 0.995 | 0.620 |
| lenet1d | 0.726 | 0.760 | 0.743 | 0.558 | 0.369 | 0.463 | 0.938 | 0.776 |
| vit1d | 0.909 | 0.749 | 0.829 | 0.513 | 0.272 | 0.393 | 0.997 | 0.965 |
| mlp_mixer1d | 0.970 | 0.625 | 0.798 | 0.861 | 0.085 | 0.473 | 0.997 | 0.966 |

## Delta vs clean-trained (E3), paired on the same configurations

| model | clean-trained AUROC | augmented AUROC | delta | clean-trained TPR95 | augmented TPR95 | delta |
|---|---:|---:|---:|---:|---:|---:|
| wdcnn | 0.837 | 0.822 | -0.015 | 0.121 | 0.173 | +0.052 |
| ticnn | 0.908 | 0.833 | -0.075 | 0.511 | 0.338 | -0.173 |
| mcnn | 0.961 | 0.846 | -0.115 | 0.807 | 0.541 | -0.267 |
| resnet1d | 0.969 | 0.857 | -0.112 | 0.829 | 0.548 | -0.281 |
| ds_cnn | 0.972 | 0.644 | -0.328 | 0.874 | 0.173 | -0.702 |
| lenet1d | 0.785 | 0.743 | -0.041 | 0.438 | 0.463 | +0.025 |
| vit1d | 0.900 | 0.829 | -0.070 | 0.564 | 0.393 | -0.171 |
| mlp_mixer1d | 0.875 | 0.798 | -0.077 | 0.509 | 0.473 | -0.036 |

Mean delta over 8 models: AUROC -0.1042, TPR@5%FPR -0.1941.

## Paired Wilcoxon over 32 (model, holdout, seed) pairs

| metric | clean-trained | augmented | delta | improved | p | p_Holm |
|---|---:|---:|---:|---:|---:|---:|
| auroc | 0.899 | 0.797 | -0.103 | 4/32 | 5e-07 | 1e-06 |
| tpr95 | 0.570 | 0.388 | -0.182 | 11/32 | 0.0061 | 0.0061 |
