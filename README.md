# Machine Learning Assignment 2 [2025AA05019] - Dry Bean Classification

This project focuses on classifying dry beans into seven registered varieties using machine learning models. The task simulates a real-world agricultural quality control and sorting application.

## a. Problem Statement

The objective is to develop and evaluate accurate machine learning models to classify dry beans into **seven varieties**:

- Seker
- Barbunya
- Bombay
- Cali
- Dermason
- Horoz
- Sira

Classification is based on **16 morphological attributes** describing the beans' form, shape, type, and structure.

## b. Dataset Description

- **Dataset**: Dry Bean Dataset (UCI Machine Learning Repository)
- **Instances**: 13,611
- **Features**: 16 (numerical attributes extracted from bean images)
  - Area
  - Perimeter
  - MajorAxisLength
  - MinorAxisLength
  - AspectRatio
  - Eccentricity
  - ConvexArea
  - EquivDiameter
  - Extent
  - Solidity
  - roundness
  - Compactness
  - ShapeFactor1
  - ShapeFactor2
  - ShapeFactor3
  - ShapeFactor4
- *Target*: Class (7 bean varieties)
- *Source*: Images of 13,611 grains captured with a high-resolution camera and processed to extract the features  
  [UCI Machine Learning Repository - Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset)

## c. Models Used & Performance

Six machine learning models were trained and evaluated on a test split. Performance metrics are summarized below:

| ML Model Name            | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|--------------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression      | 0.9265   | 0.9949 | 0.9274    | 0.9265 | 0.9269 | 0.9112 |
| Decision Tree            | 0.8965   | 0.9459 | 0.8970    | 0.8965 | 0.8967 | 0.8750 |
| kNN                      | 0.9260   | 0.9854 | 0.9266    | 0.9260 | 0.9262 | 0.9105 |
| Naive Bayes              | 0.8978   | 0.9915 | 0.8987    | 0.8978 | 0.8978 | 0.8768 |
| Random Forest (Ensemble) | 0.9273   | 0.9944 | 0.9276    | 0.9273 | 0.9274 | 0.9121 |
| XGBoost (Ensemble)       | *0.9326* |*0.9955*| *0.9332*  |*0.9326*|*0.9329*|*0.9185*|

## d. Observations

| ML Model Name            | Observation about model performance                                                                                                                                                                                            |
|--------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Logistic Regression      | Performed well for a linear model, achieving high accuracy (92.65%) and excellent AUC (0.9949), indicating the classes are reasonably linearly separable in the high-dimensional feature space.                                |
| Decision Tree            | Showed the lowest performance (Accuracy: 89.65%). While interpretable, it likely suffered from overfitting or inability to capture complex non-linear boundaries.                                                              |
| kNN                      | Achieved competitive accuracy (92.60%) comparable to Logistic Regression. However, it requires storing the entire training dataset and can be computationally expensive at inference time.                                     |
| Naive Bayes              | Had lower accuracy (89.78%) similar to Decision Tree but very high AUC (0.9915). The assumption of feature independence might be too strong, as shape features are likely correlated.                                          |
| Random Forest (Ensemble) | Performed very well (Accuracy: 92.73%), slightly outperforming Logistic Regression. As an ensemble method, it effectively handled feature interactions and non-linearity.                                                      |
| XGBoost (Ensemble)       | *Best performing model* overall with the highest Accuracy (93.26%), F1-Score (0.9329), and MCC (0.9185). Its gradient boosting approach minimized errors and captured complex patterns most effectively.                       |

## Key Takeaways

- **XGBoost** emerged as the top performer across most metrics.
- Even simple models like **Logistic Regression** performed remarkably well, suggesting strong linear separability in the feature space.
- Tree-based ensemble methods (Random Forest & XGBoost) outperformed single-tree and distance-based approaches.


