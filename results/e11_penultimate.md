# E11: logit-space vs penultimate-feature rejection scores

Same grid as E10 (8 models, two held-out class sets, two seeds, clean and
augmented training). All scores use one forward pass; the only difference
is where the class-conditional distance is measured.

## clean-trained: mean AUROC / TPR at 5% FPR

| model | softmax AUROC | logit-Maha AUROC | feat-Maha AUROC | softmax TPR95 | logit-Maha TPR95 | feat-Maha TPR95 | feat dim |
|---|---:|---:|---:|---:|---:|---:|---:|
| wdcnn | 0.858 | 0.938 | 0.915 | 0.158 | 0.670 | 0.505 | 64 |
| ticnn | 0.904 | 0.969 | 0.962 | 0.508 | 0.827 | 0.745 | 128 |
| mcnn | 0.974 | 0.982 | 0.999 | 0.878 | 0.928 | 0.994 | 144 |
| resnet1d | 0.965 | 0.979 | 0.992 | 0.793 | 0.905 | 0.966 | 128 |
| ds_cnn | 0.968 | 0.968 | 0.987 | 0.863 | 0.858 | 0.949 | 256 |
| lenet1d | 0.763 | 0.946 | 0.957 | 0.385 | 0.746 | 0.788 | 32 |
| vit1d | 0.887 | 0.907 | 0.895 | 0.484 | 0.641 | 0.658 | 64 |
| mlp_mixer1d | 0.873 | 0.932 | 0.911 | 0.490 | 0.705 | 0.697 | 128 |
| **mean** | **0.899** | **0.953** | **0.952** | **0.570** | **0.785** | **0.788** | |

## aug-trained: mean AUROC / TPR at 5% FPR

| model | softmax AUROC | logit-Maha AUROC | feat-Maha AUROC | softmax TPR95 | logit-Maha TPR95 | feat-Maha TPR95 | feat dim |
|---|---:|---:|---:|---:|---:|---:|---:|
| wdcnn | 0.822 | 0.933 | 0.909 | 0.173 | 0.604 | 0.490 | 64 |
| ticnn | 0.833 | 0.962 | 0.944 | 0.337 | 0.774 | 0.611 | 128 |
| mcnn | 0.846 | 0.972 | 0.988 | 0.540 | 0.887 | 0.935 | 144 |
| resnet1d | 0.857 | 0.970 | 0.966 | 0.548 | 0.839 | 0.797 | 128 |
| ds_cnn | 0.644 | 0.932 | 0.966 | 0.172 | 0.775 | 0.823 | 256 |
| lenet1d | 0.743 | 0.952 | 0.965 | 0.463 | 0.779 | 0.826 | 32 |
| vit1d | 0.829 | 0.882 | 0.879 | 0.393 | 0.576 | 0.595 | 64 |
| mlp_mixer1d | 0.798 | 0.917 | 0.886 | 0.473 | 0.734 | 0.605 | 128 |
| **mean** | **0.796** | **0.940** | **0.938** | **0.387** | **0.746** | **0.710** | |

## clean: paired comparisons (model x holdout x seed, Holm-corrected)

| comparison | metric | reference | other | delta | improved | p | p_Holm |
|---|---|---:|---:|---:|---:|---:|---:|
| mahalanobis_feat_vs_softmax | auroc | 0.899 | 0.952 | +0.053 | 26/32 | 2e-05 | 2e-05 |
| mahalanobis_feat_vs_softmax | tpr95 | 0.570 | 0.788 | +0.218 | 28/32 | 1.1e-06 | 2.2e-06 |
| mahalanobis_feat_vs_logit | auroc | 0.953 | 0.952 | -0.001 | 17/32 | 0.95 | 0.95 |
| mahalanobis_feat_vs_logit | tpr95 | 0.785 | 0.788 | +0.003 | 16/32 | 0.9 | 1 |

## aug: paired comparisons (model x holdout x seed, Holm-corrected)

| comparison | metric | reference | other | delta | improved | p | p_Holm |
|---|---|---:|---:|---:|---:|---:|---:|
| mahalanobis_feat_vs_softmax | auroc | 0.796 | 0.938 | +0.141 | 30/32 | 6.5e-09 | 1.3e-08 |
| mahalanobis_feat_vs_softmax | tpr95 | 0.387 | 0.710 | +0.323 | 31/32 | 6.5e-09 | 6.5e-09 |
| mahalanobis_feat_vs_logit | auroc | 0.940 | 0.938 | -0.002 | 14/32 | 0.64 | 0.64 |
| mahalanobis_feat_vs_logit | tpr95 | 0.746 | 0.710 | -0.036 | 12/32 | 0.13 | 0.26 |

## Cost of augmentation on rejection, per score

| score | clean AUROC | aug AUROC | delta | clean TPR95 | aug TPR95 | delta | p_Holm (AUROC) |
|---|---:|---:|---:|---:|---:|---:|---:|
| softmax | 0.899 | 0.796 | -0.103 | 0.570 | 0.387 | -0.182 | 1e-06 |
| mahalanobis | 0.953 | 0.940 | -0.013 | 0.785 | 0.746 | -0.039 | 0.0015 |
| mahalanobis_feat | 0.952 | 0.938 | -0.014 | 0.788 | 0.710 | -0.077 | 6e-05 |
