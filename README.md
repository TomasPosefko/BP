# Bakalársky projekt - Detekcia syntetických produktových recenzií a botmi riadenej manipulácie trhu

*Detection of Synthetic Product Reviews and Bot-Driven Market Manipulation*: bachelor's project.

Detection of machine-generated spam reviews on multiple datasets.

## Data
The data is not included in this repository (see `.gitignore`).

| Dataset | Source |
|---|---|
| FraudSquad (Amazon subset) | [Dataset repository](https://anonymous.4open.science/r/FraudSquad-5389/) · [Paper](https://arxiv.org/abs/2510.01801) |
| FRD (Fake Reviews Dataset) | [Dataset on OSF](https://osf.io/tyue9/) · [Authors' repository](https://github.com/joolsa/FakeReviews) |
| Flowerly (Modern Fake Reviews) | [Dataset files and variants](https://github.com/PurplexyFlower/Fake-review#datasets) |

### FraudSquad
[FraudSquad](https://arxiv.org/abs/2510.01801) (Liu et al., 2025), Amazon subset:

- `amazon_real.csv`: real reviews (label 0)
- `reviews_from_llama3_8b.csv`, `reviews_from_qwen_72b.csv`, `reviews_from_qwen_dsr1.csv`: spam reviews from three LLMs, ~2,500 each (label 1)

Download it from the [FraudSquad repository](https://anonymous.4open.science/r/FraudSquad-5389/) and place it in:

```
data/FraudSquad-Data/Amazon/
```

### FRD (Fake Reviews Dataset)
Salminen et al. (2022), *Creating and detecting fake reviews of online products*:

- `fake reviews dataset.csv`: 40,432 Amazon reviews from 10 categories, columns `category`, `rating`, `label`, `text_`
- `OR`: original reviews (label 0), `CG`: computer-generated reviews (GPT-2 / ULMFiT, label 1), balanced 50 / 50

Download it from [OSF](https://osf.io/tyue9/) and place it in:

```
data/FRD/fake reviews dataset.csv
```

### Flowerly (Modern Fake Reviews)

The baseline uses the GPT-2 and LFM2.5-1.2B variants. Get the CSV files from the [Flowerly project](https://github.com/PurplexyFlower/Fake-review#datasets) and place them in:

```
data/Flowerly/modern_reviews_gpt2.csv
data/Flowerly/modern_reviews_lfm.csv
```

## Repository structure

```
data/                         local dataset files
├── FraudSquad-Data/Amazon/
├── FRD/
└── Flowerly/
notebooks/
├── FraudSquad-Amazon/         exploration, baselines, and detailed results
├── FRD/                      exploration, baselines, and detailed results
└── Flowerly/
    └── basic_split_baseline.ipynb
results/                      evaluation results (CSV)
requirements.txt              Python dependencies
```

## Notebooks

### FraudSquad (`notebooks/FraudSquad-Amazon/`)

| Notebook | Description |
|---|---|
| `data_explore.ipynb` | Dataset exploration: shapes, rating distribution, text length, duplicates, missing values, product/user overlap |
| `split_baseline.ipynb` | Cleaning, metadata artifact check, product-level train/val/test split, TF-IDF + logistic regression baseline, generator × generator evaluation |
| `baseline_check.ipynb` | Multi-seed evaluation, leave-one-generator-out, validation-based threshold, ablation (text normalization, length matching, word vs. char n-grams), top features, false positive/negative analysis |

Model input is review text only. Metadata columns are excluded because they leak the label.

### FRD (`notebooks/FRD/`)

| Notebook | Description |
|---|---|
| `data_explore.ipynb` | Dataset exploration: class balance per category, rating distribution, text length, duplicates, missing values, formatting artifacts (truncated endings, sentences joined without a space) |
| `split_baseline.ipynb` | Cleaning, train/val/test split stratified by category × label, TF-IDF + logistic regression baseline, artifact-only model (no text content), category × category evaluation |
| `baseline_check.ipynb` | Text variants removing generation artifacts (`raw`, `norm`, `trim`, `trim_len`), artifact-only / word / char models, multi-seed evaluation, leave-one-category-out, top features, false positive/negative analysis |

Model input is review text only. FRD has no product or user IDs, so generalization is tested on unseen categories instead of unseen generators. The generated reviews contain strong formatting artifacts (truncation, joined sentences): an artifact-only model reaches ROC-AUC ~0.95, so results on the raw text overestimate detection of generated reviews.

### Flowerly

Flowerly is a working version with one notebook, [`basic_split_baseline.ipynb`](notebooks/Flowerly/basic_split_baseline.ipynb). Dataset exploration and robustness checks are still pending + more generators - DeepSeek and GLM.

## Results

All rows use word TF-IDF and logistic regression on raw review text. They report F1 for the synthetic class and ROC-AUC on held-out reviews from the same dataset and, where applicable, the same generator. Split protocols and F1 threshold rules differ by dataset.

| Dataset | Evaluation setup | F1 | ROC-AUC | Details |
|---|---|---:|---:|---|
| FraudSquad (Amazon) | Product-level split, mean across three generators and three seeds | 0.9638 | 0.9996 | [Detailed results](notebooks/FraudSquad-Amazon/README.md) |
| FRD | Category × label stratified 70/15/15 split, mean across three seeds | 0.9289 | 0.9801 | [Detailed results](notebooks/FRD/README.md) |
| Flowerly (GPT-2) | Label-stratified 80/20 split, seed 42 | 0.9326 | 0.9823 | [Baseline notebook](notebooks/Flowerly/basic_split_baseline.ipynb) |
| Flowerly (LFM2.5-1.2B) | Label-stratified 80/20 split, seed 42 | 0.9908 | 0.9996 | [Baseline notebook](notebooks/Flowerly/basic_split_baseline.ipynb) |

The linked reports cover unseen-generator tests for FraudSquad and held-out categories and artifact controls for FRD.

### Result files

| File | Notebook |
|---|---|
| `baseline_tfidf_lr.csv` | FraudSquad `split_baseline.ipynb` |
| `baseline_02b.csv` | FraudSquad `baseline_check.ipynb` |
| `frd_baseline.csv` | FRD `split_baseline.ipynb` |
| `frd_baseline_02b.csv` | FRD `baseline_check.ipynb` |
| [`flowerly_baseline.csv`](results/flowerly_baseline.csv) | [Flowerly `basic_split_baseline.ipynb`](notebooks/Flowerly/basic_split_baseline.ipynb) |

## Requirements

Python 3.10+, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `jupyter`.

```bash
pip install -r requirements.txt
```
