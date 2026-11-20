# FraudSquad Amazon results

The baseline uses review text, word TF-IDF (unigrams and bigrams), and logistic regression. The train, validation, and test sets are split by product. Tables report means over seeds 42, 43, and 44. AP is average precision. Recall at 1% FPR comes from the test ROC curve, while F1 uses the threshold selected on validation.

## Train on one generator

Each cell reports **AP / recall at 1% FPR**.

| Train ↓ / Test → | Llama 3 8B | Qwen 72B | Qwen DSR1 |
|---|---:|---:|---:|
| Llama 3 8B | **0.995 / 0.996** | 0.933 / 0.900 | 0.975 / 0.966 |
| Qwen 72B | 0.950 / 0.941 | **0.988 / 0.993** | 0.960 / 0.947 |
| Qwen DSR1 | 0.851 / 0.794 | 0.791 / 0.731 | **0.996 / 0.999** |

Qwen DSR1 transfers least well in both directions where it is the training generator. Its AP drops to 0.791 on Qwen 72B.

## Train on two generators

| Held-out generator | AP | Recall at 1% FPR | F1 |
|---|---:|---:|---:|
| Llama 3 8B | 0.956 | 0.945 | 0.888 |
| Qwen 72B | 0.925 | 0.896 | 0.818 |
| Qwen DSR1 | 0.978 | 0.976 | 0.932 |
| Mean | 0.953 | 0.939 | 0.879 |

## Text and length controls

These means cover the three held-out generators and three seeds for each setting.

| Text setting | Word AP | Word recall at 1% FPR | Character AP | Character recall at 1% FPR |
|---|---:|---:|---:|---:|
| Raw | 0.953 | 0.939 | 0.948 | 0.923 |
| Normalized | 0.948 | 0.930 | 0.951 | 0.929 |
| Length-matched | 0.982 | 0.938 | 0.975 | 0.916 |
| Normalized and length-matched | 0.980 | 0.925 | 0.979 | 0.930 |

Length matching filters real reviews and raises the proportion of synthetic reviews. AP depends on this proportion, so the increase in AP should be read alongside recall at a fixed FPR.

[Evaluation notebook](baseline_check.ipynb) · [Full results CSV](../../results/baseline_02b.csv)
