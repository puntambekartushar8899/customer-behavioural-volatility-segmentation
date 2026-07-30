# Customer Behavioural Volatility Segmentation

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Status](https://img.shields.io/badge/Status-Completed-success)

## Overview

This project investigates whether **behavioural volatility features** improve customer segmentation compared with the traditional **Recency, Frequency and Monetary (RFM)** framework.

The work was completed as part of the **MSc Data Analytics** programme at the **National College of Ireland**.

A complete machine learning pipeline was developed covering data preprocessing, feature engineering, customer segmentation, model evaluation, statistical validation and dashboard visualisation.

---

## Research Objective

Traditional customer segmentation relies on static RFM features that often fail to capture changes in purchasing behaviour over time.

This project introduces behavioural volatility features that describe customer consistency and purchasing variability, with the objective of producing more meaningful and actionable customer segments.

---

## Dataset

**Dataset:** UCI Online Retail Dataset

- Online retail transactions
- Customer purchase history
- After preprocessing: **4,338 customers**

Dataset source:

https://archive.ics.uci.edu/dataset/352/online+retail

---

## Project Workflow

```text
Raw Dataset
      │
      ▼
Data Understanding
      │
      ▼
Data Cleaning
      │
      ▼
RFM Feature Engineering
      │
      ▼
Behavioural Volatility Features
      │
      ▼
Advanced Behavioural Features
      │
      ▼
Feature Selection
      │
      ▼
Clustering Models
      │
      ▼
Evaluation
      │
      ▼
Visualisation
      │
      ▼
Model Validation
```

---

## Repository Structure

```text
customer-behavioural-volatility-segmentation/

│
├── data/
├── notebooks/
├── outputs/
├── src/
├── APP.py
├── run_pipeline.py
└── README.md
```

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- SciPy
- Matplotlib
- Streamlit
- Jupyter Notebook

---

## Machine Learning Pipeline

### Feature Engineering

- Recency
- Frequency
- Monetary
- Purchase Interval Variability
- Behavioural Volatility
- Customer Consistency
- Advanced Behavioural Features

### Clustering Algorithms

- K-Means
- Hierarchical Agglomerative Clustering
- Gaussian Mixture Models

---

## Evaluation Metrics

The clustering models were evaluated using:

- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index
- Bootstrap Stability Analysis
- PCA Visualisation

---

## Key Results

The behavioural volatility feature representation produced improved customer segmentation compared with the traditional RFM representation.

Key outcomes include:

- Improved Silhouette Score
- Better cluster stability
- More interpretable customer personas
- Enhanced behavioural insights for customer analytics

---

## Dashboard

The project includes an interactive Streamlit dashboard for exploring customer segmentation results.

To launch the dashboard:

```bash
streamlit run APP.py
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/puntambekartushar8899/customer-behavioural-volatility-segmentation.git
```

Move into the project directory

```bash
cd customer-behavioural-volatility-segmentation
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the project

```bash
python run_pipeline.py
```

Launch the dashboard

```bash
streamlit run APP.py
```

---

## Outputs

The project generates:

- Cleaned datasets
- Engineered feature datasets
- Cluster assignments
- Customer personas
- Evaluation metrics
- Visualisations
- Interactive dashboard

---

## Author

**Tushar Naresh Puntambekar**

MSc Data Analytics

National College of Ireland

GitHub: https://github.com/puntambekartushar8899



---

## License

<<<<<<< HEAD
This project is intended for educational and portfolio purposes.
=======

