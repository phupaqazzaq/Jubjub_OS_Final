# Jubjub_OS_Final

# Thai Harassment Detection - OS-Optimized Pipeline

This Google Colab notebook demonstrates an OS-optimized pipeline for detecting Thai harassment. The pipeline includes steps for data loading, preprocessing, and saving the processed data, leveraging various operating system features for efficiency.

## Getting Started

To run this notebook, follow the steps below.

### 1. Clone the Repository

First, clone the project repository from GitHub and navigate into its directory:

```python
!git clone https://github.com/phupaqazzaq/Operating-system-Final-project.git
%cd Operating-system-Final-project
```

### 2. Run the Data Pipeline

Execute the main pipeline script. This script handles data loading, multiprocessing for preprocessing, and atomic file saving for the results. The output will provide a summary of the pipeline's performance.

```python
!python /content/Operating-system-Final-project/src/pipeline.py
```

#### Pipeline Overview:
*   **Data Loading:** Efficiently loads the `thai_toxicity_2025_train_final` dataset.
*   **Preprocessing:** Utilizes `fork()` and `multiprocessing.Pool` for parallel processing of data, classifying text as toxic or non-toxic.
*   **Saving Results:** Stores the processed data (`thai_preprocessed.csv`) using atomic write operations (`fsync()` and `rename()`) to ensure data integrity.

### 3. Review Pipeline Summary

After execution, the console output will display a detailed summary including processing times, throughput, and the specific OS components leveraged for optimization.

## Next Steps

With the data preprocessed, you can proceed with further steps such as:

*   Model training (`src/train_model.py`)
*   Model evaluation (`src/evaluate.py`)
*   Prediction using the trained model (`src/predict.py`)

Refer to the `src` directory for more scripts related to the project.
```
