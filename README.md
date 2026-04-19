# Drug Consumption Analysis

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white"/>
  <img src="https://img.shields.io/badge/Status-Part%201%20Complete-4CAF50?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Dataset-UCI%20ML%20Repository-0277BD?style=for-the-badge"/>
</p>

<p align="center">
  A data science project analysing drug consumption patterns across 1,884 survey respondents, combining personality profiling with demographic data and self-reported substance use.
</p>

---

## Overview

This repository documents the complete data analysis of the **Drug Consumption** dataset from the UCI Machine Learning Repository (Fehrman et al., 2017). The analysis is structured in three sequential phases:

| Phase | Topic | Status |
|:---:|---|:---:|
| 1 | [Exploratory data analysis](#part-1---exploratory-data-analysis) | Complete |
| 2 | [Classification](#part-2---classification) | Upcoming |
| 3 | [Regression](#part-3---regression) | Upcoming |

---

## Dataset

The dataset records the drug consumption behaviour of **1,884 respondents**, collected via an online survey between 2011 and 2012. Each row contains:

- **Demographics** - Age, Gender, Education, Country, Ethnicity *(nominal / ordinal)*
- **Personality scores** - Neuroticism, Extraversion, Openness, Agreeableness, Conscientiousness, Impulsiveness, Sensation Seeking *(continuous z-scores, NEO-PI-R model)*
- **Drug consumption** - 18 substances rated on a 7-level ordinal scale from `CL0` (never used) to `CL6` (last day)

> **Note on Semer:** The survey includes a fictitious drug as a validity check. Only **0.4%** of respondents claimed to have used it, confirming general data reliability.

---

## Part 1 - Exploratory Data Analysis

### Sample Characteristics

The sample is nearly gender-balanced (943 male, 941 female) but skewed toward younger, more educated respondents, with over 55% originating from the United Kingdom.

<p align="center">
  <img src="results/gender-age-country.png" width="780" alt="Gender, Age and Country distributions"/>
</p>
<p align="center"><em>Figure 1 - Demographic distribution: gender, age group, and country of residence.</em></p>

---

### Personality Scores

All seven personality scores are approximately normally distributed and centred near zero, consistent with their z-score standardisation. The close alignment of mean and median across all features confirms approximate symmetry.

<p align="center">
  <img src="results/personality-scores.png" width="780" alt="Personality score histograms"/>
</p>
<p align="center"><em>Figure 2 - Distribution of the seven personality scores.</em></p>

| Feature | Mean | Median | Std Dev | IQR |
|---|---:|---:|---:|---:|
| Nscore (Neuroticism) | −0.0001 | 0.0426 | 0.998 | 1.308 |
| Escore (Extraversion) | 0.0001 | 0.0033 | 0.997 | 1.333 |
| Oscore (Openness) | −0.0002 | −0.0193 | 0.996 | 1.441 |
| AScore (Agreeableness) | 0.0002 | −0.0173 | 0.997 | 1.367 |
| Cscore (Conscientiousness) | −0.0004 | −0.0066 | 0.998 | 1.237 |
| Impulsive | 0.0073 | −0.2171 | 0.954 | 1.241 |
| SS (Sensation Seeking) | −0.0027 | 0.0799 | 0.963 | 1.291 |

---

### Legal vs. Illegal Substances

Legal substances show substantially higher recent use than illegal ones. Caffeine leads with **73.5%** of respondents at CL6 (last day), while Heroin has **85.1%** at CL0 (never used).

<p align="center">
  <img src="results/legal-illegal-comparison.png" width="780" alt="Drug usage frequency comparison"/>
</p>
<p align="center"><em>Figure 3 - Consumption frequency for selected substances across the legal–illegal spectrum.</em></p>

---

### Sensation Seeking Across Age Groups

Sensation Seeking declines monotonically with age - from a mean of **+0.40** in the 18–24 group to **−0.97** in the 65+ group - consistent with established findings in personality psychology.

<p align="center">
  <img src="results/ss-age.png" width="680" alt="Sensation Seeking by Age group"/>
</p>
<p align="center"><em>Figure 4 - Mean Sensation Seeking score by age group, with reference line at z = 0.</em></p>

---

### Outlier Detection

Two methods were applied: the **3-sigma rule** (Definition 5.1.1) and the **IQR rule** (Definition 5.1.3). Both identify only a small number of extreme values per feature (≤ 1.3% of observations). Given that scores originate from a validated psychometric instrument, these extremes most plausibly represent genuine individual differences rather than measurement errors - **removal is not justified**.

<p align="center">
  <img src="results/outlier-sigma.png" width="780" alt="Outlier detection boxplots"/>
</p>
<p align="center"><em>Figure 5 - Boxplot overview of all personality features; points beyond the whiskers are flagged by the IQR rule.</em></p>

---

### Key Findings Summary

| # | Finding |
|:---:|---|
| 1 | **No missing values** - mandatory survey design ensured complete data across all 1,884 rows |
| 2 | **Sample skew** - younger respondents (18–34) and UK-based participants are overrepresented |
| 3 | **Normal personality distributions** - all scores centred near zero with std ≈ 1 |
| 4 | **Clear legal/illegal divide** - Caffeine and Alcohol heavily used; hard drugs rarely so |
| 5 | **Age–SS relationship** - Sensation Seeking decreases steadily from 18–24 to 65+ |
| 6 | **Outliers retained** - few in number and plausibly genuine; robust measures preferred |
| 7 | **Valid responses** - Semer fictitious drug claimed by only 0.4% of participants |

---

## Part 2 - Classification

---

## Part 3 - Regression

---

## Repository Structure

```
drug-consumption-analysis/
│
├── explorative-data-analysis/
│   ├── EDA_Drug_Consumption.ipynb       # Jupyter Notebook - Part 1
│   ├── explorative-data-analysis.pdf    # Documentation - Part 1
│
├── results/
│   └── ...
│   
│
├── Drug_Consumption.csv                 # Dataset
├── ProjectWork_Assessment.pdf           # Assignment brief
└── README.md
```

---

## Dependencies

```python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import mode
from tabulate import tabulate
```

---

## Reference

Fehrman, E., Muhammad, A. K., Mirkes, E. M., Egan, V., & Gorban, A. N. (2017). *The Five Factor Model of personality and evaluation of drug consumption risk.* In Palumbo, Montanari & Vichi (Eds.), Data Science: Innovative Developments in Data Analysis and Clustering (pp. 231–242). Springer.

Dataset available at: [UCI ML Repository - Drug Consumption (Quantified)](https://archive.ics.uci.edu/dataset/373/drug+consumption+quantified)
