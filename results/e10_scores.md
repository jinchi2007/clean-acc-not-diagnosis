# E10: decision-layer scores for open-set rejection

Same networks as E3/E9 (8 models, two held-out class sets, two seeds).
Scores: softmax maximum (baseline), negative free energy, and negative
Mahalanobis distance to the class-conditional Gaussians in logit space.

## Mean AUROC / TPR at 5% FPR, clean-trained

| model | softmax AUROC | energy AUROC | Mahalanobis AUROC | softmax TPR95 | energy TPR95 | Mahalanobis TPR95 |
|---|---:|---:|---:|---:|---:|---:|
| wdcnn | 0.858 | 0.858 | 0.938 | 0.158 | 0.210 | 0.670 |
| ticnn | 0.904 | 0.898 | 0.969 | 0.508 | 0.503 | 0.827 |
| mcnn | 0.974 | 0.961 | 0.982 | 0.878 | 0.794 | 0.928 |
| resnet1d | 0.965 | 0.941 | 0.979 | 0.793 | 0.650 | 0.905 |
| ds_cnn | 0.968 | 0.937 | 0.968 | 0.863 | 0.694 | 0.858 |
| lenet1d | 0.763 | 0.794 | 0.946 | 0.385 | 0.523 | 0.746 |
| vit1d | 0.887 | 0.882 | 0.907 | 0.484 | 0.455 | 0.641 |
| mlp_mixer1d | 0.873 | 0.841 | 0.932 | 0.490 | 0.428 | 0.705 |

## Mean AUROC / TPR at 5% FPR, augmented (SNR ~ U[0, 15] dB)

| model | softmax AUROC | energy AUROC | Mahalanobis AUROC | softmax TPR95 | energy TPR95 | Mahalanobis TPR95 |
|---|---:|---:|---:|---:|---:|---:|
| wdcnn | 0.822 | 0.819 | 0.933 | 0.173 | 0.224 | 0.604 |
| ticnn | 0.833 | 0.808 | 0.962 | 0.337 | 0.296 | 0.774 |
| mcnn | 0.846 | 0.855 | 0.972 | 0.540 | 0.604 | 0.887 |
| resnet1d | 0.857 | 0.831 | 0.970 | 0.548 | 0.490 | 0.839 |
| ds_cnn | 0.644 | 0.628 | 0.932 | 0.172 | 0.154 | 0.775 |
| lenet1d | 0.743 | 0.713 | 0.952 | 0.463 | 0.390 | 0.779 |
| vit1d | 0.829 | 0.834 | 0.882 | 0.393 | 0.451 | 0.576 |
| mlp_mixer1d | 0.798 | 0.792 | 0.917 | 0.473 | 0.353 | 0.734 |

## clean: score choice (paired over model x holdout x seed)

| comparison | metric | softmax | other | delta | improved | p | p_Holm |
|---|---|---:|---:|---:|---:|---:|---:|
| energy_vs_softmax | auroc | 0.899 | 0.889 | -0.010 | 9/32 | 0.031 | 0.062 |
| energy_vs_softmax | tpr95 | 0.570 | 0.532 | -0.038 | 11/32 | 0.17 | 0.17 |
| mahalanobis_vs_softmax | auroc | 0.899 | 0.953 | +0.054 | 26/32 | 6.7e-05 | 0.00013 |
| mahalanobis_vs_softmax | tpr95 | 0.570 | 0.785 | +0.215 | 24/32 | 0.00011 | 0.00011 |

## aug: score choice (paired over model x holdout x seed)

| comparison | metric | softmax | other | delta | improved | p | p_Holm |
|---|---|---:|---:|---:|---:|---:|---:|
| energy_vs_softmax | auroc | 0.796 | 0.785 | -0.012 | 14/32 | 0.17 | 0.34 |
| energy_vs_softmax | tpr95 | 0.387 | 0.370 | -0.017 | 20/32 | 0.99 | 0.99 |
| mahalanobis_vs_softmax | auroc | 0.796 | 0.940 | +0.143 | 31/32 | 9.3e-10 | 1.9e-09 |
| mahalanobis_vs_softmax | tpr95 | 0.387 | 0.746 | +0.359 | 31/32 | 2.3e-09 | 2.3e-09 |

## Cost of augmentation on rejection, per score

| score | clean AUROC | aug AUROC | delta | clean TPR95 | aug TPR95 | delta |
|---|---:|---:|---:|---:|---:|---:|
| softmax | 0.899 | 0.796 | -0.103 | 0.570 | 0.387 | -0.182 |
| energy | 0.889 | 0.785 | -0.104 | 0.532 | 0.370 | -0.162 |
| mahalanobis | 0.953 | 0.940 | -0.013 | 0.785 | 0.746 | -0.039 |
