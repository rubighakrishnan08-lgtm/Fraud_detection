# Part 4 – LLM Powered Model Explanation Pipeline

## Track C – Model Prediction Explanation

---

## Overview

This project demonstrates how a **Large Language Model (LLM)** can be integrated with a trained machine learning model to generate **human-readable explanations** for fraud detection predictions.
The pipeline performs the following tasks:

- Loads a trained fraud detection model.
- Prepares transaction features.
- Predicts fraud probability.
- Generates structured prompts for an LLM.
- Validates LLM responses using a predefined JSON schema.
- Detects Personally Identifiable Information (PII) before sending prompts.
- Compares responses using different temperature settings.
- Exports all results for documentation and analysis.

---

# Technologies Used

- Python 3.10+
- Google Gemini API
- scikit-learn
- pandas
- numpy
- python-dotenv
- json
- pathlib

---

# Pipeline Workflow

```text
Transaction Record
        │
        ▼
Feature Preparation
        │
        ▼
Machine Learning Model
        │
        ├── Prediction
        └── Probability
                │
                ▼
Prompt Engineering
                │
                ▼
PII Guardrail
                │
                ▼
Gemini LLM
                │
                ▼
JSON Validation
                │
                ▼
Structured Explanation
                │
                ▼
CSV Report Generation
```

---

### Objectives

- Configure Gemini API
- Create reusable LLM function
- Verify API connectivity
- Handle API exceptions

### Completed

- API configuration
- Gemini model selection
- Error handling
- Connection testing
  **Model Used**

```
gemini-2.5-flash
```

---

### Components Implemented

- System Prompt
- User Prompt Template
- JSON Output Schema
- JSON Validation
- Fallback Response
- Safe LLM Wrapper
- Prompt Formatter

### JSON Response Format

```json
{
  "prediction_label": "Fraud",
  "confidence_level": "High",
  "top_reason": "Large transaction amount",
  "second_reason": "Unusual transaction behaviour",
  "next_step": "Recommend manual verification"
}
```

---

# PII Guardrail

Before calling the LLM, every prompt is checked for Personally Identifiable Information (PII).

### Example – Blocked Input

```
Customer Name : John Doe
Customer Email : john.doe@gmail.com
```

**Result**

```
BLOCKED
```

### Example – Allowed Input

```
Prediction : Fraud
Probability : 0.9721
Transaction Amount : 1250
```

**Result**

```
ALLOWED
```

---

### Model Loading

- Trained model loaded successfully
- Model Type:

```
sklearn.pipeline.Pipeline
```

### Dataset Information

| Property      |            Value |
| ------------- | ---------------: |
| Dataset       | cleaned_data.csv |
| Rows          |           39,422 |
| Columns       |              434 |
| Features Used |              431 |

## Three sample records were processed through the prediction pipeline.

# Prediction Results

| Record   | Prediction | Probability |
| -------- | ---------- | ----------: |
| Record 1 | Not Fraud  |        0.99 |
| Record 2 | Not Fraud  |        1.00 |
| Record 3 | Not Fraud  |        0.94 |

---

# LLM Explanation Process

For every prediction, the pipeline performs:

1. Feature encoding
2. Prediction generation
3. Probability estimation
4. Prompt construction
5. PII validation
6. LLM request
7. JSON validation
8. Result export

---

# Temperature Comparison

Two temperature settings were evaluated.
| Temperature | Purpose |
| ----------: | -------------------------------------- |
| **0.0** | Deterministic, consistent JSON output |
| **0.7** | More diverse and creative explanations |
This comparison demonstrates the effect of generation randomness on explanation quality.

---

# API Quota Observation

During execution, the following API response was received:

```
429 RESOURCE_EXHAUSTED
```

This indicates that the free-tier request quota for the Gemini API was exceeded.
The machine learning prediction pipeline executed successfully. Only the LLM explanation stage was affected due to API usage limits.
The project includes exception handling to ensure that execution continues even when an external API is temporarily unavailable.

---

# Generated Output Files

The pipeline automatically generates the following reports:
| File | Description |
| ---------------------------- | ---------------------------------------- |
| `track_c_results.csv` | Prediction results with LLM explanations |
| `temperature_comparison.csv` | Temperature comparison outputs |
| `guardrail_results.csv` | PII guardrail demonstration |
| `demo_results.csv` | Final demonstration summary |

---

# Features Implemented

- Gemini API integration
- Reusable LLM function
- Prompt engineering
- JSON schema validation
- Exception handling
- Fallback responses
- PII guardrail
- Feature encoding
- Prediction generation
- Probability estimation
- Temperature comparison
- CSV report generation

---

# Limitations

- LLM explanation generation depends on an active Gemini API quota.
- When the free-tier quota is exceeded, explanation requests return a `429 RESOURCE_EXHAUSTED` response.
- The prediction pipeline remains fully functional even if the LLM service is temporarily unavailable.

---

# Conclusion

This project successfully demonstrates the integration of a machine learning fraud detection model with a Large Language Model to improve prediction interpretability.
The implementation includes structured prompt engineering, JSON validation, privacy guardrails, temperature comparison, and automated reporting, providing a practical example of explainable AI using modern LLM technologies.
