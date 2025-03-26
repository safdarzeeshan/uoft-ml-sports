# Sports Classification Project

This project is the final assignment for the University of Toronto's Machine Learning course. The objective is to build a robust image classification pipeline capable of recognizing and labeling images from **100 distinct sports categories**.

## Project Structure

```
final-project/
├── sports-dectection.ipynb      # Jupyter notebook with the full classification pipeline
├── requirements.txt             # Cleaned list of dependencies
└── README.md                    # This file
```

## Setup

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

You can use a virtual environment to keep dependencies isolated:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

## How to Run

1. Clone this repository or open the folder in VSCode/Jupyter.
2. Open `sports-dectection.ipynb`.
3. Follow the cells to:
    - Load and preprocess a multi-class dataset of sports images
    - Train and evaluate a supervised classification model
    - Visualize performance with plots and metrics

## Example Results

Evaluation metrics such as accuracy, precision, recall, and confusion matrix visualizations are generated within the notebook. These help assess model performance across the 100 sports categories and highlight areas for potential improvement.

## Dataset Information

The dataset is structured into three subsets:
- `train/`
- `validation/`
- `test/`

Each folder contains subdirectories named after individual sports, holding image samples. This structure supports supervised learning with categorical labels.

## Notes

- The project focuses on building a scalable solution for multi-class classification.
- Generalization across a large number of classes is a key consideration.

## Acknowledgments

- University of Toronto - School of Continuing Studies
- Course: Machine Learning (Winter 2025)
