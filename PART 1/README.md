# Financial Fraud Detection – Data Preprocessing and Exploratory Data Analysis

**Machine Learning Capstone Project – Part 1**

---

# 1. Project Overview

Financial fraud has become one of the most significant challenges faced by banks, online payment platforms, and e-commerce companies. Millions of financial transactions occur every day, making it impossible to manually inspect every transaction for fraudulent activity. Machine Learning provides an effective solution by learning transaction patterns and identifying suspicious behaviour automatically.

Before any Machine Learning model can be trained, the raw data must undergo extensive preprocessing. Real-world datasets usually contain missing values, inconsistent data types, duplicate records, highly skewed variables, and outliers that can negatively affect model performance.

The objective of this phase of the project is to perform systematic data preprocessing and exploratory data analysis (EDA) on the IEEE-CIS Fraud Detection dataset. The preprocessing decisions taken in this phase establish a reliable foundation for feature engineering and predictive modelling in the later stages of the capstone project.

---

# 2. Problem Statement

The dataset contains hundreds of transaction and identity-related variables collected from online payment transactions. Like many real-world datasets, it exhibits several data quality issues including:

- Missing values across numerous features
- Extremely sparse identity columns
- Mixed numerical and categorical variables
- Highly skewed numerical distributions
- Potential outliers
- High-dimensional feature space

These issues must be carefully addressed before the data can be used for Machine Learning.

The primary goal of this preprocessing stage is therefore to clean the dataset while preserving important fraud-related information that may contribute to accurate classification models in subsequent phases.

---

# 3. Project Objectives

The objectives of this project are:

- Load and inspect the dataset.
- Understand the overall structure of the data.
- Analyse missing values across every feature.
- Handle missing values using statistically appropriate techniques.
- Detect duplicate records.
- Optimise data types for efficient memory utilisation.
- Perform descriptive statistical analysis.
- Identify skewed variables.
- Detect potential outliers using the Interquartile Range (IQR) method.
- Visualise important characteristics of the dataset.
- Study relationships among variables using correlation analysis.
- Compare Pearson and Spearman correlation methods.
- Perform grouped statistical aggregation.
- Generate a cleaned dataset for subsequent Machine Learning tasks.

---

# 4. Dataset Description

This project uses the **IEEE-CIS Fraud Detection Dataset**, which was released as part of a Kaggle competition jointly organised by the IEEE Computational Intelligence Society (IEEE-CIS) and Kaggle.

The dataset represents real-world online payment transactions and contains anonymised transactional and identity-related information. The objective of the original competition was to predict whether a given transaction was fraudulent.

Unlike small educational datasets, this dataset reflects many of the challenges encountered in industrial Machine Learning applications, including missing information, imbalanced target classes, high-cardinality categorical variables, and hundreds of engineered features.

## Dataset Characteristics

| Property          |   Value |
| ----------------- | ------: |
| Number of Records | 590,540 |
| Number of Columns |     434 |
| Numeric Features  |     403 |
| Object Features   |      31 |
| Target Variable   | isFraud |

The successful loading of the dataset confirmed that it contains **590,540 observations and 434 attributes**, making it a high-dimensional dataset suitable for large-scale fraud detection research.

---

# 5. Project Workflow

The preprocessing workflow adopted in this project is illustrated below.

```
Raw Dataset
      │
      ▼
Dataset Inspection
      │
      ▼
Missing Value Analysis
      │
      ▼
Duplicate Detection
      │
      ▼
Data Type Optimisation
      │
      ▼
Descriptive Statistics
      │
      ▼
Skewness Analysis
      │
      ▼
Outlier Detection
      │
      ▼
Exploratory Visualisation
      │
      ▼
Correlation Analysis
      │
      ▼
Grouped Aggregation
      │
      ▼
Cleaned Dataset
```

Following a structured workflow ensures that each preprocessing decision is justified before moving to the next stage.

---

# 6. Software Environment

The preprocessing and analysis were implemented using Python.

## Programming Language

- Python 3.11

## Libraries Used

- Pandas
- NumPy
- Matplotlib
- Seaborn

These libraries provide efficient tools for handling large datasets, performing statistical analysis, and generating visualisations required for exploratory data analysis.

---

# 7. Task 1 – Dataset Loading and Initial Inspection

The dataset was imported into a Pandas DataFrame using the `pd.read_csv()` function.

After loading, the following validation steps were performed:

- Displayed the first five rows using `head()`.
- Verified the total number of rows and columns.
- Examined the data type of every attribute.

Performing these initial checks is important because they confirm that the dataset has been imported correctly before any preprocessing begins.

## Observations

The dataset was loaded successfully without any import errors.

The inspection revealed:

- **590,540 transaction records**
- **434 total attributes**
- A combination of integer, floating-point, and object data types
- Transaction information, payment card information, identity variables, email domains, and device-related attributes

The presence of both numerical and categorical variables indicates that different preprocessing strategies are required for different feature types.

---

# 8. Task 2 – Missing Value Analysis

Handling missing values is one of the most important preprocessing steps because many Machine Learning algorithms cannot directly process incomplete observations.

To understand the extent of missing information, the number and percentage of null values were calculated for every column using:

```python
df.isnull().sum()

(df.isnull().sum()/len(df))*100
```

This analysis was performed for all **434 columns**.

## Results

The analysis revealed a considerable amount of missing information within the identity-related features.

Several variables such as:

- id_24
- id_25
- id_07
- id_08
- id_21

contained approximately **99% missing values**.

Overall,

**252 columns contained more than 20% missing values.**

These highly sparse columns were documented separately because imputing values for columns containing such large proportions of missing data could introduce substantial bias.

### Handling Columns Below 20% Missing Values

For numeric columns with fewer than **20% missing values**, missing entries were imputed using the **median** of the respective column.

Median imputation was selected because many financial variables exhibit highly skewed distributions.

Unlike the arithmetic mean, the median is resistant to extreme values and therefore provides a more reliable estimate of the central tendency when outliers are present.

### Why the Median Was Preferred Over the Mean

Financial transaction datasets often contain unusually large transaction amounts or rare events that create long-tailed distributions.

If the mean were used for imputation:

- Extreme observations would disproportionately influence the replacement value.
- Artificially inflated values could be introduced into the dataset.
- Model performance might deteriorate because of biased feature distributions.

The median avoids these issues by representing the middle observation rather than the arithmetic average.

Consequently, median imputation preserves the original distribution more effectively while reducing the impact of outliers.

---

# 9. Task 3 – Duplicate Detection and Removal

Duplicate observations can introduce redundancy and bias during model training.

Therefore, duplicate rows were identified using:

```python
df.duplicated().sum()
```

The dataset was then checked before and after applying:

```python
df.drop_duplicates()
```

## Results

The duplicate analysis produced the following findings:

| Description            |   Value |
| ---------------------- | ------: |
| Rows Before Cleaning   | 590,540 |
| Rows After Cleaning    | 590,540 |
| Duplicate Rows Removed |       0 |

No duplicate transactions were found in the dataset.

Since no duplicate rows existed, removing duplicates did not alter the percentage of missing values in any feature.

This observation indicates that the dataset was already free from exact duplicate records, allowing subsequent preprocessing steps to proceed without loss of information.

# 10. Task 4 – Data Type Correction and Memory Optimisation

Large real-world datasets often contain columns whose inferred data types are not optimal for analysis. Inefficient data types increase memory consumption and slow down data processing. Therefore, before proceeding with statistical analysis, the data types of the dataset were examined and optimised wherever appropriate.

Initially, the data types of all columns were inspected using:

```python
df.dtypes
```

The dataset contained a mixture of:

- Integer (`int64`)
- Floating-point (`float64`)
- Object (`object`)

The inspection revealed **31 object-type columns**, primarily consisting of categorical information such as:

- ProductCD
- Card information
- Email domains
- Identity verification attributes
- Device information

Since these columns represent repeated categorical values rather than continuous text, converting them to the **category** data type significantly reduces memory usage while preserving the information contained in the dataset.

## Memory Usage Analysis

Memory consumption was measured before and after data type optimisation.

| Description         | Memory (Bytes) |
| ------------------- | -------------: |
| Before Optimisation |  2,691,793,800 |
| After Optimisation  |  1,924,242,143 |
| Memory Saved        |    767,551,657 |

### Interpretation

Approximately **767 MB of memory** was saved through data type optimisation.

This improvement is significant because large datasets such as IEEE-CIS Fraud Detection are frequently processed multiple times during feature engineering and model training. Lower memory consumption reduces computational overhead and improves processing efficiency, particularly when working with limited hardware resources.

No information was lost during this optimisation process since categorical variables preserve the same values while using a more memory-efficient representation.

---

# 11. Task 5 – Descriptive Statistics and Skewness Analysis

Descriptive statistics provide an overview of the distribution and variability of numerical features. They help identify unusual values, understand feature ranges, and guide preprocessing decisions.

The descriptive statistics were generated using:

```python
df.describe()
```

The summary statistics included:

- Count
- Mean
- Standard Deviation
- Minimum
- 25th Percentile (Q1)
- Median (Q2)
- 75th Percentile (Q3)
- Maximum

These statistics confirmed that the dataset contains variables with vastly different scales and distributions. For example, transaction amounts ranged from very small values to transactions exceeding 31,000 units, indicating the presence of substantial variability and potential outliers.

## Skewness Analysis

To understand the shape of each numerical distribution, skewness was computed for every numeric feature.

```python
df[col].skew()
```

The top ten most skewed variables were:

| Rank | Feature | Skewness |
| ---- | ------- | -------: |
| 1    | V305    |   384.23 |
| 2    | V311    |   323.83 |
| 3    | V129    |   240.34 |
| 4    | V309    |   224.88 |
| 5    | V206    |   207.88 |
| 6    | V319    |   181.84 |
| 7    | V269    |   177.84 |
| 8    | V266    |   175.96 |
| 9    | V334    |   168.47 |
| 10   | V135    |   144.92 |

The feature **V305** exhibited the highest absolute skewness with a value of **384.23**.

## Interpretation of Skewness

A **positive skew** indicates that the majority of observations are concentrated near lower values while a relatively small number of observations extend toward much larger values, creating a long right tail.

In contrast, a **negative skew** indicates that most observations occur near higher values while a few unusually small values create a long left tail.

The extremely high positive skewness observed in V305 suggests that only a very small proportion of transactions contain unusually high values. These extreme observations substantially increase the arithmetic mean.

Consequently, using the **mean** to replace missing values would introduce an unrealistically high estimate of the feature's central tendency. The **median**, however, remains resistant to these extreme values and therefore provides a more representative replacement value.

This analysis directly supports the decision to perform median imputation during the missing value handling stage.

---

# 12. Task 6 – Outlier Detection Using the Interquartile Range (IQR)

Outliers are observations that differ significantly from the majority of the data. In fraud detection, extreme values may represent suspicious behaviour rather than erroneous data. Therefore, outlier detection was performed to understand their presence without immediately removing them.

The Interquartile Range (IQR) method was used.

For each selected feature:

- First Quartile (Q1)
- Third Quartile (Q3)
- IQR = Q3 − Q1
- Lower Bound = Q1 − 1.5 × IQR
- Upper Bound = Q3 + 1.5 × IQR

Values outside these bounds were considered potential outliers.

---

## Analysis of Feature: V305

| Statistic   | Value |
| ----------- | ----: |
| Q1          |     1 |
| Q3          |     1 |
| IQR         |     0 |
| Lower Bound |     1 |
| Upper Bound |     1 |
| Outliers    |     4 |

### Interpretation

Since both the first and third quartiles are identical, the interquartile range becomes zero. Consequently, any value differing from one is classified as an outlier.

Only **four observations** were identified as outliers, indicating that the feature is highly concentrated around a single dominant value with very few exceptional cases.

---

## Analysis of Feature: V311

| Statistic   |  Value |
| ----------- | -----: |
| Q1          |      0 |
| Q3          |      0 |
| IQR         |      0 |
| Lower Bound |      0 |
| Upper Bound |      0 |
| Outliers    | 17,238 |

### Interpretation

Unlike V305, the feature V311 contains a considerably larger number of values outside the dominant range.

Although these observations are statistically classified as outliers, they may represent genuine behavioural differences within transaction records rather than measurement errors.

Removing these observations could potentially eliminate valuable fraud-related information.

### Decision Regarding Outliers

No outliers were removed during preprocessing.

Instead, they will be retained for Part 2 of the project.

The rationale for this decision is based on the nature of fraud detection datasets. Fraudulent transactions often appear as statistical anomalies, and aggressive outlier removal may unintentionally remove precisely those observations that Machine Learning models need to learn.

If required during model development, robust scaling techniques or outlier-resistant algorithms will be considered instead of deleting observations.

---

# 13. Task 7 – Exploratory Data Visualisations

Visualisation plays a critical role in understanding data distributions and relationships that may not be immediately evident from numerical summaries alone.

Five different visualisations were created as part of the exploratory data analysis.

---

## 13.1 Line Plot

A line plot was generated for a numerical feature arranged according to transaction order.

### Interpretation

The line plot illustrates how the selected numerical variable changes throughout the dataset.

Rather than exhibiting a smooth trend, the values fluctuate considerably, reflecting the highly dynamic nature of financial transaction data. Such fluctuations are expected because individual customer transactions occur independently and vary substantially in value.

---

## 13.2 Bar Chart

A bar chart was created to compare the average value of a numerical feature across categories of a categorical variable.

### Interpretation

The chart demonstrates that different product categories exhibit noticeably different average transaction values.

This observation suggests that product category may contain useful predictive information and could become an important feature during model development.

---

## 13.3 Histogram

A histogram was generated for the most skewed feature, **V305**.

### Interpretation

The histogram confirms the statistical analysis performed earlier.

Most observations are concentrated near a single value, while only a very small number extend into much larger values, producing a pronounced right-skewed distribution.

This visual evidence further supports the use of median imputation instead of mean imputation.

---

## 13.4 Scatter Plot

A scatter plot was generated using two numerical variables expected to exhibit correlation.

### Interpretation

The scatter plot indicates a generally positive association between the selected variables.

However, the relationship is not perfectly linear, and considerable dispersion is visible among observations.

This suggests that although one variable tends to increase as the other increases, additional factors also influence their behaviour.

---

## 13.5 Box Plot

A box plot was produced to compare the distribution of a numerical variable across different categories.

### Interpretation

The box plot reveals noticeable differences in:

- Median values
- Interquartile ranges
- Overall variability
- Presence of outliers

Some categories exhibit much wider distributions than others, indicating that transaction behaviour differs across product groups.

These differences suggest that categorical variables may contribute useful information during predictive modelling.

---

# 14. Task 8 – Pearson Correlation Heat Map

To study linear relationships among numerical variables, the Pearson correlation matrix was computed using:

```python
df.corr()
```

The resulting matrix was visualised as a heat map.

## Strongest Correlation

The highest correlation observed was:

| Variable 1 | Variable 2 | Correlation |
| ---------- | ---------- | ----------: |
| D1         | D2         |      0.9813 |

This extremely high positive correlation indicates that the two variables increase together almost linearly.

## Interpretation

Although D1 and D2 exhibit a correlation coefficient close to one, this should **not** be interpreted as evidence of causation.

A high correlation simply indicates that the variables vary together.

A plausible explanation is that both variables represent related aspects of transaction timing or customer activity derived from the same underlying behavioural process.

Therefore, the observed relationship may be explained by a common hidden factor rather than a direct causal relationship between the two variables.

This distinction is important because Machine Learning models utilise statistical relationships rather than causal mechanisms.

# 15. Task 9(a) – Imputation Strategy Comparison

Before performing the final imputation, the two numerical features with the highest absolute skewness (**V305** and **V311**) were analysed by comparing their arithmetic mean and median.

The objective of this comparison was to determine which measure of central tendency better represents each feature before replacing the remaining missing values.

## Mean vs Median Comparison

| Feature |     Mean |   Median |
| ------- | -------: | -------: |
| V305    | 1.000007 | 1.000000 |
| V311    | 4.202090 | 0.000000 |

---

## Interpretation

### Feature: V305

The mean and median values of V305 are nearly identical. Although the feature exhibits extremely high positive skewness, the majority of observations are concentrated around the value **1**, with only a few extreme observations occurring outside this range.

Because these extreme values are very rare, the difference between the mean and median is relatively small.

---

### Feature: V311

A much larger difference exists between the mean and the median.

- Mean = **4.2021**
- Median = **0**

This difference clearly demonstrates the effect of positive skewness.

Most observations in V311 are equal to zero, while a relatively small number of unusually large values increase the arithmetic mean considerably.

Using the mean to replace missing values would therefore introduce values that are much larger than the typical observation.

The median provides a much more representative estimate of the centre of the distribution.

---

## Imputation Decision

Based on the skewness analysis performed in Task 5 and the comparison above, **median imputation** was selected for both features.

The decision was made because:

- Both variables exhibit strong positive skewness.
- The mean is influenced by extreme values.
- The median is resistant to outliers.
- Median imputation better preserves the original distribution of the data.

The remaining missing values were successfully filled using:

```python
fillna(df[col].median())
```

Verification using:

```python
df.isnull().sum()
```

confirmed that both selected features contained **zero remaining missing values** after imputation.

---

# 16. Task 9(b) – Spearman Rank Correlation Analysis

Although Pearson correlation measures **linear relationships**, many real-world financial variables exhibit **non-linear but monotonic relationships**.

To capture these relationships, Spearman Rank Correlation was computed using:

```python
df.corr(method='spearman')
```

The Spearman matrix was then compared with the Pearson correlation matrix obtained earlier.

Finally, the absolute difference

```
|Spearman − Pearson|
```

was calculated for every variable pair.

---

## Largest Differences Between Pearson and Spearman Correlation

| Rank | Variable Pair          | Absolute Difference |
| ---- | ---------------------- | ------------------: |
| 1    | D10 – id_02            |               0.341 |
| 2    | D10 – id_01            |               0.230 |
| 3    | TransactionAmt – card3 |               0.153 |

---

## Interpretation of the Three Largest Differences

### 1. D10 – id_02

Difference:

**0.341**

This is the largest observed difference between Pearson and Spearman correlation.

The substantially stronger Spearman correlation suggests that the relationship between these variables is **monotonic but not strictly linear**.

Although one variable generally increases as the other increases, the increase is not proportional across the entire range.

---

### 2. D10 – id_01

Difference:

**0.230**

Again, Spearman correlation exceeds Pearson correlation.

This indicates another monotonic relationship where the variables maintain a consistent ordering but do not necessarily follow a straight-line relationship.

Such behaviour is common in financial datasets because many variables are derived from customer behaviour rather than physical measurements.

---

### 3. TransactionAmt – card3

Difference:

**0.153**

The difference is smaller than the previous two pairs but still sufficiently large to suggest that rank-based correlation captures additional information beyond linear correlation.

Transaction amounts frequently contain extreme values and long-tailed distributions, making Spearman correlation more robust than Pearson correlation.

---

## Pearson vs Spearman

### Pearson Correlation

Pearson correlation measures:

- Linear relationships
- Sensitivity to outliers
- Dependence on actual numerical values

Pearson performs well when variables are approximately normally distributed and exhibit linear relationships.

---

### Spearman Correlation

Spearman correlation measures:

- Rank-based relationships
- Monotonic behaviour
- Robustness against outliers
- Non-linear associations

Because financial datasets frequently contain skewed variables and extreme observations, Spearman correlation often provides a more reliable indication of feature relationships.

---

## Correlation Measure Selected for Feature Selection

For the next phase of the project (Machine Learning model development), **Spearman correlation** will be given greater importance whenever strong non-linear monotonic relationships are present.

However, Pearson correlation will continue to be considered for variables exhibiting approximately linear behaviour.

Using both measures together provides a more comprehensive understanding of feature relationships than relying on either method alone.

---

# 17. Task 9(c) – Grouped Aggregation

Grouped statistical aggregation helps understand how numerical variables behave within different categories.

The following aggregation was performed:

```python
df.groupby(ProductCD)["TransactionAmt"].agg(
["mean","std","count"])
```

---

## Aggregated Results

The analysis produced the mean, standard deviation, and number of observations for every ProductCD category.

### Highest Mean

Product Category:

**R**

This category exhibited the highest average transaction amount among all ProductCD groups.

---

### Highest Standard Deviation

Product Category:

**W**

This category showed the greatest variability in transaction amounts.

A large standard deviation indicates that transactions belonging to this category vary considerably from one another.

---

### Highest Mean to Lowest Mean Ratio

The ratio between the highest group mean and the lowest group mean was calculated as:

**3.93**

---

## Interpretation

The difference between the highest and lowest group averages indicates that ProductCD carries useful predictive information.

Transactions belonging to different product categories exhibit noticeably different average transaction values.

However, the high within-group variability observed for ProductCD = W suggests that this categorical feature alone cannot accurately predict transaction amounts or fraudulent behaviour.

Consequently, ProductCD should be combined with additional numerical and categorical variables during feature engineering rather than being used as an independent predictor.

---

# 18. Saving the Cleaned Dataset

After completing all preprocessing tasks, the cleaned dataset was exported using:

```python
df.to_csv(
"cleaned_data.csv",
index=False
)
```

The resulting file,

```
cleaned_data.csv
```

will serve as the input dataset for Part 2 and Part 3 of the Machine Learning Capstone Project.

Saving the cleaned dataset ensures that all preprocessing operations need to be performed only once, allowing future stages of the project to focus exclusively on feature engineering, model training, and evaluation.

---

# 19. Summary of Work Completed

The following preprocessing tasks were successfully completed during this phase:

✔ Dataset loaded successfully.

✔ Dataset dimensions and data types verified.

✔ Missing value count and percentage computed for every feature.

✔ Features with more than 20% missing values identified.

✔ Numeric features with fewer than 20% missing values imputed using the median.

✔ Duplicate detection completed.

✔ Duplicate removal verified.

✔ Data type optimisation performed.

✔ Memory usage significantly reduced.

✔ Descriptive statistics generated.

✔ Skewness calculated for every numerical feature.

✔ Most skewed variables identified.

✔ Outlier detection performed using the IQR method.

✔ Outliers documented without removal.

✔ Five exploratory visualisations generated.

✔ Pearson correlation matrix computed.

✔ Correlation heat map generated.

✔ Mean versus median comparison completed.

✔ Spearman rank correlation computed.

✔ Pearson and Spearman correlations compared.

✔ Three largest correlation differences analysed.

✔ Grouped aggregation completed.

✔ Clean dataset exported successfully.

---

# 20. Limitations of the Current Work

Although the preprocessing stage substantially improves data quality, several limitations remain.

- Many identity-related variables contain extremely high percentages of missing values.

- No feature selection has yet been performed.

- Outliers were intentionally retained because they may correspond to genuine fraudulent transactions.

- Class imbalance has not yet been addressed.

- No predictive model has been trained during this phase.

These limitations will be addressed in the subsequent stages of the capstone project.

---

# 21. Future Work (Part 2)

The cleaned dataset generated in this phase will be used for Machine Learning model development.

The next phase will include:

- Feature selection

- Feature engineering

- Encoding categorical variables

- Feature scaling

- Handling class imbalance

- Model training

- Hyperparameter tuning

- Performance evaluation

- Comparison of multiple Machine Learning algorithms

The objective will be to develop a robust fraud detection model capable of accurately distinguishing fraudulent transactions from legitimate ones.

---

# 22. Conclusion

This project successfully completed the preprocessing and exploratory data analysis phase of the IEEE-CIS Fraud Detection dataset.

The dataset was carefully inspected, cleaned, and analysed using a systematic workflow. Missing values were examined and handled using statistically appropriate techniques, duplicate records were verified, memory usage was reduced through data type optimisation, and descriptive statistical analysis provided valuable insights into the structure of the data. Skewness analysis highlighted highly asymmetric variables, while the IQR method identified potential outliers without removing information that may be relevant for fraud detection.

Correlation analysis using both Pearson and Spearman coefficients revealed important relationships among numerical variables and demonstrated the value of considering both linear and monotonic associations. Grouped aggregation further showed that transaction behaviour differs across product categories, indicating that categorical features contain useful predictive information.

Overall, this preprocessing stage has transformed the raw dataset into a cleaner, more reliable, and analysis-ready dataset. The exported `cleaned_data.csv` provides a strong foundation for feature engineering, predictive modelling, and model evaluation in the remaining phases of the Machine Learning Capstone Project.

---

# References

1. Kaggle. _IEEE-CIS Fraud Detection Dataset_.  
   https://www.kaggle.com/competitions/ieee-fraud-detection

2. IEEE Computational Intelligence Society (IEEE-CIS)

---
