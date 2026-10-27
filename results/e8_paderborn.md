# E8 -- Paderborn generalization (2-class healthy vs outer ring)

Paderborn KAt subset (18 bearings: 6 healthy K001-K006 + 12 artificial outer-ring damage KA*), 64 kHz, 4 operating conditions, leakage-safe split (per-record 70/30 with 512-sample guard band).

train/test = 60000 / 20000 windows (30k / 10k per class stratified subsample of the full safe split), 15 epochs, seeds [0, 1], mean±std over seeds.

| config | params | clean | 10 dB | 5 dB | 0 dB |
|---|---:|---:|---:|---:|---:|
| wdcnn:plain | 33890 | 0.9619±0.002 | 0.8992±0.000 | 0.8430±0.002 | 0.7886±0.007 |
| wdcnn:aug | 33890 | 0.8961±0.005 | 0.9405±0.005 | 0.9370±0.002 | 0.8145±0.017 |
| ticnn:plain | 82882 | 0.9761±0.000 | 0.9084±0.003 | 0.8514±0.012 | 0.8075±0.019 |
| ticnn:aug | 82882 | 0.8263±0.083 | 0.9240±0.049 | 0.9443±0.018 | 0.7996±0.040 |
| resnet1d:plain | 240290 | 0.9898±0.000 | 0.8848±0.024 | 0.7686±0.005 | 0.5584±0.013 |
| resnet1d:aug | 240290 | 0.8564±0.019 | 0.9356±0.007 | 0.9513±0.008 | 0.8106±0.001 |
| ds_cnn:plain | 67586 | 0.9743±0.001 | 0.8375±0.010 | 0.6750±0.032 | 0.5165±0.006 |
| ds_cnn:aug | 67586 | 0.8221±0.026 | 0.8837±0.030 | 0.9218±0.012 | 0.7692±0.017 |

## aug - plain (delta accuracy)

| model | clean | 10 dB | 5 dB | 0 dB |
|---|---:|---:|---:|---:|
| wdcnn | -0.0658 | +0.0414 | +0.0940 | +0.0259 |
| ticnn | -0.1498 | +0.0156 | +0.0929 | -0.0079 |
| resnet1d | -0.1334 | +0.0508 | +0.1827 | +0.2522 |
| ds_cnn | -0.1522 | +0.0462 | +0.2467 | +0.2527 |

## ranking (clean vs 10 dB)

- plain: clean order ['resnet1d', 'ticnn', 'ds_cnn', 'wdcnn'] vs 10 dB order ['ticnn', 'wdcnn', 'resnet1d', 'ds_cnn'] -> inverted=True
- aug: clean order ['wdcnn', 'resnet1d', 'ticnn', 'ds_cnn'] vs 10 dB order ['wdcnn', 'resnet1d', 'ticnn', 'ds_cnn'] -> inverted=False
