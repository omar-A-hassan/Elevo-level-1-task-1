# Elevo Level 1 - Machine Learning Portfolio

This repository contains a comprehensive machine learning portfolio demonstrating proficiency across supervised learning, unsupervised learning, and advanced time series representation learning techniques.

## Phase 1: Supervised Learning Foundations

### Task 1: Linear Regression Analysis
**Student Performance Prediction**
- **Objective**: Predict student exam scores using study hours as a predictor variable
- **Methodology**: Implemented linear regression with standardized features and comprehensive evaluation
- **Key Results**: Achieved R² score of 0.232 with RMSE of 0.847 on standardized data
- **Insights**: Demonstrated moderate correlation (0.445) between study hours and exam performance

### Task 1-Bonus: End-to-End Regression Pipeline
**Advanced Student Performance Analysis**
- **Objective**: Comprehensive pipeline for predicting exam scores using multiple features and model comparison
- **Methodology**: 
  - Extensive EDA with correlation analysis and feature visualization
  - Linear vs polynomial regression comparison
  - Feature engineering and selection experiments
  - Cross-validation and hyperparameter optimization
- **Key Results**: Linear regression outperformed polynomial (R² 0.770 vs 0.768), demonstrating optimal model selection


### Task 2: Customer Segmentation
**Unsupervised Learning Pipeline**
- **Objective**: Segment mall customers based on demographics and spending behavior
- **Methodology**: 
  - K-Means clustering with elbow method and silhouette analysis
  - DBSCAN implementation for comparison
  - Comprehensive cluster profiling and business insights
- **Key Results**: Identified 5 distinct customer segments with 0.417 silhouette score
- **Business Impact**: Delivered actionable segmentation strategies for targeted marketing campaigns

## Phase 2: Advanced Classification

### Task 4: Loan Approval Prediction
**End-to-End Classification Pipeline**
- **Objective**: Build robust classification model for loan approval decisions
- **Methodology**:
  - Comprehensive data preprocessing with missing value handling
  - Class imbalance correction using SMOTE technique
  - Model comparison: Logistic Regression, Random Forest, XGBoost
  - Hyperparameter tuning with GridSearchCV
- **Key Results**: XGBoost achieved 98.24% F1-score and 99.82% ROC-AUC, representing 5.4% improvement over baseline


## Phase 3: Advanced Time Series Representation Learning

### Task 7: TS2Vec Implementation
**Time Series Representation Learning for Walmart Forecasting**
- **Objective**: Adapt TS2Vec architecture for retail forecasting using Walmart sales dataset
- **Methodology**:
  - Enhanced TS2Vec implementation with adaptive padding for short time series
  - Multi-table data preprocessing creating three analysis perspectives
  - Business-relevant forecasting horizons (1-24 weeks)
  - Multi-granularity analysis: store-level, department-level, and feature-enhanced forecasting
- **Key Contributions**:
  - Adaptive padding mechanism improving evaluation stability for datasets <500 time steps
  - Flexible prediction horizons tailored for retail domain requirements
  - Multi-table preprocessing pipeline handling complex relational datasets
- **Results**: Successfully trained models for store forecasting, multivariate feature analysis, and department-level predictions


## Acknowledgments

This implementation builds upon the original [TS2Vec repository](https://github.com/zhihanyue/ts2vec) by [@zhihanyue](https://github.com/zhihanyue).

### Original Work
- **Paper**: [TS2Vec: Towards Universal Representation of Time Series](https://arxiv.org/abs/2106.10466) (AAAI-22)
- **Authors**: Zhihan Yue, Yujing Wang, Juanyong Duan, Tianmeng Yang, Congrui Huang, Yunhai Tong, Bixiong Xu
- **Original Repository**: [@zhihanyue/ts2vec](https://github.com/zhihanyue/ts2vec)
- **License**: MIT License (maintained from original repository)

All core algorithmic contributions, theoretical foundations, and base implementation credit belongs to the original creators. This work focuses solely on dataset adaptation and preprocessing enhancements for the Walmart forecasting use case while maintaining full backward compatibility with the original implementation.

## Technical Stack

- **Languages**: Python, Jupyter Notebooks
- **ML Libraries**: scikit-learn, XGBoost, PyTorch
- **Data Analysis**: pandas, numpy, matplotlib, seaborn
- **Specialized**: SMOTE (imbalanced learning), TS2Vec (time series representation)
- **Evaluation**: Comprehensive metrics including ROC-AUC, silhouette analysis, and forecasting accuracy

## Repository Structure

```
├── Phase-1/
│   ├── Task-1/          # Basic linear regression
│   ├── Task-1-bonus/    # Advanced regression pipeline
│   └── Task-2/          # Customer segmentation
├── Phase-2/
│   └── Task-4/          # Loan approval classification
└── Phase-3/
    └── Task-7/          # TS2Vec time series learning
```

This portfolio demonstrates progression from fundamental ML concepts to cutting-edge time series representation learning, showcasing both theoretical understanding and practical implementation skills across diverse domains.