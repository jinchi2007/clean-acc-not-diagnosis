# Statistical summary

Runs scanned: 424

## T1 clean-trained: clean / 10 dB / 5 dB / 0 dB (mean±std over seeds)

| model | split | clean | 10 dB | 5 dB | 0 dB | n_seeds |
|---|---|---:|---:|---:|---:|---:|
| ds_cnn | crossload | 0.8999±0.0011 | — | — | — | 2 |
| ds_cnn | filedisjoint | 0.9981±0.0027 | — | — | — | 2 |
| ds_cnn | leaky | 1.0000±0.0000 | — | — | — | 2 |
| ds_cnn | safe | 0.9980±0.0065 | 0.6338±0.2036 | 0.3914±0.1808 | 0.3416±0.2207 | 30 |
| lenet1d | crossload | 0.8562±0.0018 | — | — | — | 2 |
| lenet1d | filedisjoint | 0.9543±0.0483 | — | — | — | 2 |
| lenet1d | leaky | 0.6813±0.3471 | — | — | — | 4 |
| lenet1d | safe | 0.9259±0.1825 | 0.8227±0.0883 | 0.4080±0.0827 | 0.2597±0.1562 | 21 |
| mcnn | crossload | 0.9000±0.0003 | — | — | — | 2 |
| mcnn | filedisjoint | 0.9972±0.0039 | — | — | — | 2 |
| mcnn | leaky | 0.9917±0.0101 | — | — | — | 4 |
| mcnn | safe | 0.9987±0.0047 | 0.7364±0.1804 | 0.2507±0.0694 | 0.1381±0.0462 | 24 |
| mlp_mixer1d | crossload | 0.8695±0.0076 | — | — | — | 2 |
| mlp_mixer1d | filedisjoint | 0.9528±0.0504 | — | — | — | 2 |
| mlp_mixer1d | leaky | 0.9987±0.0011 | — | — | — | 2 |
| mlp_mixer1d | safe | 0.9980±0.0015 | 0.8649±0.0822 | 0.4433±0.0124 | 0.3953±0.0798 | 19 |
| resnet1d | crossload | 0.8994±0.0003 | — | — | — | 2 |
| resnet1d | filedisjoint | 0.9972±0.0021 | — | — | — | 2 |
| resnet1d | leaky | 0.9855±0.0187 | — | — | — | 4 |
| resnet1d | safe | 0.9970±0.0089 | 0.6785±0.2198 | 0.3906±0.2397 | 0.2705±0.1720 | 32 |
| ticnn | crossload | 0.9106±0.0126 | — | — | — | 2 |
| ticnn | filedisjoint | 0.9966±0.0049 | — | — | — | 2 |
| ticnn | leaky | 1.0000±0.0000 | — | — | — | 2 |
| ticnn | safe | 0.9977±0.0072 | 0.9758±0.0358 | 0.7285±0.1157 | 0.4602±0.1797 | 21 |
| vit1d | crossload | 0.8941±0.0090 | — | — | — | 2 |
| vit1d | filedisjoint | 0.9856±0.0124 | — | — | — | 2 |
| vit1d | leaky | 0.9981±0.0011 | — | — | — | 2 |
| vit1d | safe | 0.9979±0.0019 | 0.7911±0.1298 | 0.4675±0.0055 | 0.3490±0.1587 | 28 |
| wdcnn | crossload | 0.9476±0.0027 | — | — | — | 2 |
| wdcnn | filedisjoint | 0.9545±0.0522 | — | — | — | 2 |
| wdcnn | leaky | 0.8810±0.1367 | — | — | — | 4 |
| wdcnn | safe | 0.9789±0.0631 | 0.9761±0.0316 | 0.8389±0.0598 | 0.5434±0.1664 | 29 |

## T2 paired Wilcoxon: noise augmentation vs clean training (pairs = model x seed)

| metric | n pairs | clean-trained | +aug | Δ | p (Wilcoxon) |
|---|---:|---:|---:|---:|---:|
| clean | 24 | 0.9949 | 0.9405 | -0.0543 | 2.664e-05 (adj 3.636e-05) |
| 10.0 dB | 24 | 0.6988 | 0.9672 | +0.2684 | 1.192e-07 (adj 4.768e-07) |
| 5.0 dB | 24 | 0.4517 | 0.9695 | +0.5178 | 1.818e-05 (adj 3.636e-05) |
| 0.0 dB | 24 | 0.2379 | 0.6871 | +0.4492 | 1.192e-07 (adj 4.768e-07) |

## T3 open-set rejection (unknown = held-out fault classes)

| model | holdout | known acc | AUROC | TPR@95TNR | acc@5% reject |
|---|---|---:|---:|---:|---:|
| ds_cnn | [1, 4, 7] | nan | 0.7762 | 0.5629 | 0.9389 |
| ds_cnn | [3, 6, 9] | nan | 0.8629 | 0.5309 | 0.9499 |
| lenet1d | [1, 4, 7] | nan | 0.7520 | 0.3344 | 0.9174 |
| lenet1d | [3, 6, 9] | nan | 0.7660 | 0.5324 | 0.9305 |
| mcnn | [1, 4, 7] | nan | 0.8396 | 0.4890 | 0.9499 |
| mcnn | [3, 6, 9] | nan | 0.9840 | 0.9226 | 0.9499 |
| mlp_mixer1d | [1, 4, 7] | nan | 0.6986 | 0.0676 | 0.9499 |
| mlp_mixer1d | [3, 6, 9] | nan | 0.9790 | 0.9053 | 0.9499 |
| resnet1d | [1, 4, 7] | nan | 0.8733 | 0.6132 | 0.9421 |
| resnet1d | [3, 6, 9] | nan | 0.9590 | 0.7634 | 0.9499 |
| ticnn | [1, 4, 7] | nan | 0.7737 | 0.0401 | 0.9499 |
| ticnn | [3, 6, 9] | nan | 0.9701 | 0.8209 | 0.9499 |
| vit1d | [1, 4, 7] | nan | 0.7929 | 0.2807 | 0.9499 |
| vit1d | [3, 6, 9] | nan | 0.9338 | 0.6400 | 0.9499 |
| wdcnn | [1, 4, 7] | nan | 0.8609 | 0.2355 | 0.9499 |
| wdcnn | [3, 6, 9] | nan | 0.8117 | 0.0771 | 0.9499 |

