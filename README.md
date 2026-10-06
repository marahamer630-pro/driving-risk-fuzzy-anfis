# Driving Risk Assessment System (Fuzzy Logic & ANFIS)

A two-part soft computing project that estimates driving risk from the driving conditions and the driver's state. University project for the Neuro-Fuzzy Systems course (Arab International University).

## Part 1: Fuzzy Inference System (FIS)

`part1_fuzzy_inference_system.ipynb`

Built with **scikit-fuzzy** (`skfuzzy.control`):

| Variable | Type | Range | Fuzzy sets |
|----------|------|-------|------------|
| Speed | Input | 0 – 120 km/h | slow, medium, fast |
| Road condition | Input | 0 – 10 | poor, normal, good |
| Driver alertness | Input | 0 – 10 | low, medium, high |
| Risk level | Output | 0 – 10 | low risk, medium risk, high risk |

- Triangular membership functions for every variable
- A complete base of **27 fuzzy rules** (3 × 3 × 3 input combinations)
- Example: speed = 80, road condition = 6, driver alertness = 3 gives a risk level of **8.21** (high risk)
- Plots of all membership functions

## Part 2: ANFIS (Adaptive Neuro-Fuzzy Inference System)

`part2_anfis_training.ipynb`

1. Uses the Part 1 FIS to generate training data: a grid of **96 input combinations** (6 speeds × 4 road conditions × 4 alertness levels) with the FIS risk level as the target. If no rule fires for a combination, a neutral value of 5.0 is used.
2. Builds an ANFIS model with Gaussian membership functions (3 per input) using the `anfis` library.
3. Trains it with the hybrid learning algorithm (Jang, offline) for 20 epochs.
4. Evaluates it with **RMSE = 0.2343** and plots the original FIS output against the ANFIS prediction.

The RMSE is measured on the same 96-sample grid used for training, so it shows how closely ANFIS reproduces the FIS rather than performance on unseen data.

## Tech stack

Python, NumPy, pandas, scikit-fuzzy, ANFIS, scikit-learn (metrics), matplotlib, Jupyter Notebook

## Run it

The notebooks were written for **Google Colab**. Part 2 installs `scikit-fuzzy` and the `anfis` library in its first cells.

```bash
pip install scikit-fuzzy numpy pandas matplotlib scikit-learn jupyter
```

For Part 1 this is enough. Part 2 also needs the `anfis` package from GitHub (`pip install git+https://github.com/twmeggs/anfis.git`).
