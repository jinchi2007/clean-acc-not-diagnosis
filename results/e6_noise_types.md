# E6: 噪声类型扩展（clean training，测试端加噪）

协议：CWRU 10-class, split=safe, overlap=0.5, win=1024, 20 epoch, 干净训练；测试端用 add_noise(kind, SNR) 加谐波 / 脉冲 / 漂移噪声，SNR = 10 / 0 dB。数值为 seeds 均值±std（std 在只有 1 个 seed 时记 0）。

## 主表：三类非高斯噪声

| model | noise | clean | 10 dB | 0 dB |
|---|---|---:|---:|---:|
| wdcnn | harmonic | 0.997±0.001 | 0.952±0.028 | 0.412±0.055 |
| wdcnn | impulse | 0.997±0.001 | 0.980±0.015 | 0.447±0.015 |
| wdcnn | drift | 0.997±0.001 | 0.993±0.005 | 0.604±0.334 |
| resnet1d | harmonic | 1.000±0.000 | 0.632±0.016 | 0.183±0.047 |
| resnet1d | impulse | 1.000±0.000 | 0.970±0.031 | 0.444±0.020 |
| resnet1d | drift | 1.000±0.000 | 0.758±0.079 | 0.183±0.118 |
| ds_cnn | harmonic | 1.000±0.000 | 0.611±0.010 | 0.197±0.015 |
| ds_cnn | impulse | 1.000±0.000 | 0.995±0.004 | 0.929±0.039 |
| ds_cnn | drift | 1.000±0.000 | 0.545±0.021 | 0.188±0.013 |
| vit1d | harmonic | 0.996±0.002 | 0.745±0.011 | 0.258±0.023 |
| vit1d | impulse | 0.996±0.002 | 0.887±0.053 | 0.754±0.064 |
| vit1d | drift | 0.996±0.002 | 0.474±0.116 | 0.184±0.092 |

## 高斯参照（同协议，3 seeds）与对照

| model | gaussian clean | gaussian 10 dB | gaussian 0 dB | harmonic 10 dB | impulse 10 dB | drift 10 dB |
|---|---:|---:|---:|---:|---:|---:|
| wdcnn | 0.997 | 0.980±0.010 | 0.405±0.103 | 0.952 | 0.980 | 0.993 |
| resnet1d | 1.000 | 0.427±0.012 | 0.136±0.056 | 0.632 | 0.970 | 0.758 |
| ds_cnn | 1.000 | 0.428±0.016 | 0.228±0.048 | 0.611 | 0.995 | 0.545 |
| vit1d | 0.996 | 0.755±0.005 | 0.271±0.055 | 0.745 | 0.887 | 0.474 |

## 10 dB 下的模型排序：高斯 vs 三类非高斯

- gaussian: wdcnn > vit1d > ds_cnn > resnet1d
- harmonic: wdcnn > vit1d > resnet1d > ds_cnn
- impulse: ds_cnn > wdcnn > resnet1d > vit1d
- drift: wdcnn > resnet1d > ds_cnn > vit1d

- harmonic 与高斯排序不同。
- impulse 与高斯排序不同。
- drift 与高斯排序不同。

> 说明：训练是干净的（无噪声增广），所以只有测试噪声类型在变；同一 model/seed 在不同 kind 下会各自重训一次（为断点续跑而逐 run 缓存），MPS 非确定性会让 clean 列有千分位差异。
