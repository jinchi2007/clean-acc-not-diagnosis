# E0b: leakage in a few-shot regime (20/class)

| model | leaky | safe | leaky - safe |
|---|---:|---:|---:|
| wdcnn | 0.7629±0.010 | 0.7537±0.003 | +0.0092 |
| mcnn | 0.9833±0.004 | 0.9840±0.004 | -0.0006 |
| resnet1d | 0.9710±0.010 | 0.9647±0.006 | +0.0062 |
| lenet1d | 0.3809±0.016 | 0.3780±0.009 | +0.0029 |

mean leaky - safe gap: +0.0044 points (0.4 accuracy points)
