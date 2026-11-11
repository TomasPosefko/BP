# Bakalársky projekt - Detekcia syntetických produktových recenzií a botmi riadenej manipulácie trhu

*Detection of Synthetic Product Reviews and Bot-Driven Market Manipulation*: bachelor's project (practical part).

Detection of LLM-generated spam reviews on Amazon data, with a focus on generalization to unseen generators and control of dataset artifacts.

## Data
This project is using only one dataset for now:

[FraudSquad](https://arxiv.org/abs/2510.01801) (Liu et al., 2025), Amazon subset:

- `amazon_real.csv`: real reviews (label 0)
- `reviews_from_llama3_8b.csv`, `reviews_from_qwen_72b.csv`, `reviews_from_qwen_dsr1.csv`: spam reviews from three LLMs, ~2,500 each (label 1)

The data is not included in this repository. Download it from the [FraudSquad repository](https://anonymous.4open.science/r/FraudSquad-5389/) and place it in:

```
data/FraudSquad-Data/Amazon/
```

## Repository structure

```
data/         dataset (not versioned)
notebooks/    Jupyter notebooks
results/      evaluation results (CSV)
```

## Notebooks

| Notebook | Description |
|---|---|
| `data_explore.ipynb` | Dataset exploration: shapes, rating distribution, text length, duplicates, missing values, product/user overlap |
| `split_baseline.ipynb` | Cleaning, metadata artifact check, product-level train/val/test split, TF-IDF + logistic regression baseline, generator × generator evaluation |
| `baseline_check.ipynb` | Multi-seed evaluation, leave-one-generator-out, validation-based threshold, ablation (text normalization, length matching, word vs. char n-grams), top features, false positive/negative analysis |

Model input is review text only. Metadata columns are excluded because they leak the label.

## Requirements

Python 3.10+, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `jupyter`.

```bash
pip install pandas numpy scikit-learn matplotlib jupyter
```