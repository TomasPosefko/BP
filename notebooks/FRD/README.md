# FRD results

FRD contains original and computer-generated Amazon product reviews. The experiment uses a 70/15/15 split stratified by category and label. Word and character TF-IDF models use logistic regression. The artifact-only model uses surface features such as punctuation and length. F1 uses the threshold selected on validation.

## Stratified split

Each cell reports **ROC-AUC / F1**, averaged over three seeds.

| Text setting | Word TF-IDF | Character TF-IDF | Artifacts only |
|---|---:|---:|---:|
| Raw | 0.980 / 0.929 | 0.974 / 0.919 | 0.948 / 0.870 |
| Normalized whitespace and sentence boundaries | 0.980 / 0.929 | 0.971 / 0.916 | 0.935 / 0.854 |
| Normalized and trimmed to the last sentence boundary | 0.967 / 0.901 | 0.956 / 0.888 | 0.856 / 0.791 |
| Normalized, trimmed, and length-matched | 0.972 / 0.913 | 0.963 / 0.899 | 0.843 / 0.770 |

On raw reviews, the artifact-only model reaches ROC-AUC 0.948. The word model's recall at 1% false-positive rate is 0.769 on raw reviews and 0.728 on the trimmed, length-matched subset. Trimming and length matching change which reviews are evaluated, so score changes combine changes in text and test-set composition.

## Leave one category out

For each row, the word model trains on the other nine categories. The table reports ROC-AUC.

| Held-out category | Raw | Normalized, trimmed, and length-matched |
|---|---:|---:|
| Books | 0.970 | 0.965 |
| Clothing, shoes, and jewelry | 0.960 | 0.948 |
| Electronics | 0.977 | 0.968 |
| Home and kitchen | 0.982 | 0.966 |
| Kindle Store | 0.971 | 0.970 |
| Movies and TV | 0.961 | 0.956 |
| Pet supplies | 0.971 | 0.956 |
| Sports and outdoors | 0.980 | 0.968 |
| Tools and home improvement | 0.979 | 0.967 |
| Toys and games | 0.968 | 0.950 |
| **Mean across categories** | **0.972** | **0.961** |

Clothing, shoes, and jewelry gives the lowest ROC-AUC under the full set of controls. This experiment measures transfer to a different product category within FRD.

[Evaluation notebook](baseline_check.ipynb) · [Full results CSV](../../results/frd_baseline_02b.csv) · [Single-seed and category transfer CSV](../../results/frd_baseline.csv)
