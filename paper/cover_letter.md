# Cover letter (draft)

To the Editor-in-Chief
*Measurement Science and Technology* (IOP Publishing)

Dear Editor,

We submit for your consideration the manuscript **"Clean Accuracy Is Not a Diagnosis: A Noise-Stress
Re-evaluation of Lightweight Bearing Fault Diagnosis"** for publication as a Research Paper in
*Measurement Science and Technology*.

**What the paper does.** Lightweight 1D convolutional networks are the default choice for vibration-based
bearing diagnosis on embedded hardware, and the CWRU benchmark is their standard proving ground. Reported
accuracies on CWRU have approached 100%, which has made the benchmark increasingly uninformative. We
re-evaluate eight widely used lightweight architectures (WDCNN, TICNN, MCNN, ResNet1D, DS-CNN, LeNet1D,
ViT1D and MLP-Mixer1D) under a leakage-safe protocol in which no sampling point is shared between training
and test windows, and we stress every model with additive noise at six SNR levels.

**What we find that is new.** (i) Clean-accuracy differences on CWRU are within the seed-to-seed variation
(97-100%), so published "improvements" at this level do not identify architecture quality. (ii) When clean
accuracy is held fixed, test accuracy at 10 dB SNR spreads over 55 points (0.43-0.98), and the clean ranking
does not predict the noisy ranking. (iii) A one-line change to the training recipe - per-batch noise
augmentation - raises the mean 10 dB accuracy from 0.70 to 0.97 (paired Wilcoxon, Holm-corrected) at a cost
of 5.4 points of clean accuracy. (iv) A multi-channel wide first-layer stem recovers +13 to +52 points at
10 dB for about 1% additional parameters, and composes with augmentation to remove most of that clean cost.
(v) Rejection of unseen fault classes is a separate axis: it is anti-correlated with noise robustness
(Spearman rho = -0.64) because a noise-tolerant classifier is a smoothing operator. (vi) The ordering of
models changes again under harmonic, impulsive and drift noise, so robustness claims that name only
Gaussian noise are not reproducible. (vii) The rejection loss is mostly in the decision layer, not the
features: scoring the same trained networks with a Mahalanobis distance raises mean open-set AUROC from
0.90 to 0.95 clean-trained and from 0.80 to 0.94 augmented, i.e. the cost of the augmentation
on rejection falls from 0.10 to 0.01, so any open-set number must be reported together with its score. (viii) The gain does not depend on the coordinate system: the same distance computed on the input of the last linear layer (penultimate features) gives 0.952 clean and 0.938 augmented, so it is the distance read-out, not the logits, that carries the rejection. We also reproduce the collapse and its repair on the Paderborn
benchmark, and the leakage audit shows the effect of overlapping-window leakage is at most 0.6 points.

**Why it fits your readership.** The contribution is a measurement-methods contribution: it quantifies how
a widely used benchmark and a widely used evaluation protocol misreport the property that deployment
actually requires, and it gives the community a protocol (open leakage-safe splits, a noise stress matrix,
a rejection test and a reporting checklist) that can be applied without retraining every model in the
literature.

**Declarations.** The work is original, is not under consideration elsewhere, and has no competing
interests. All datasets used are public (CWRU, Paderborn); the split files, per-seed results and analysis
scripts will be released with the paper. A preprint is not required for our submission.

Thank you for your consideration.

Sincerely,

Han Jinchi
Southwest University, Chongqing, China
1395574811@qq.com
