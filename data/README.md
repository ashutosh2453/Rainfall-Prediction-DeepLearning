# Data Directory

This directory contains the meteorological datasets used for rainfall prediction.

## Directory Structure

```
data/
├── Rainfall.zip        # Raw rainfall dataset zip archive
├── Rainfall/           # Extracted raw rainfall data files
├── MinTemp/            # Raw minimum temperature data files
└── MaxTemp/            # Raw maximum temperature data files
```

## Datasets

1. **Rainfall**: High-resolution daily rainfall grid data.
2. **MinTemp & MaxTemp**: Daily minimum and maximum temperature grid data used as additional predictors.

## Preprocessing

The raw grid datasets in this folder are processed by running the `01_Data_Preprocessing.ipynb` notebook. The preprocessing pipeline includes:
* Extrapolating and aligning temporal and spatial coordinates.
* Scaling variables (using min-max scaling).
* Splitting the data into training, validation, and testing sets.
* Saving the processed tensors (`X_train.npy`, `y_train.npy`, etc.) to the `processed_data/` folder at the root of the project.
