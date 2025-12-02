# House-prices-Advanced-regression-techniques

👩‍💻 Author: Barbara Sánchez (@bsanchezv)

## 📊 Dataset split

Kaggle provides two CSV files: `train.csv` and `test.csv`. This means that the split of the dataset was already done.

| Data       | Observations | Percentage |
|------------|-------------:|-----------:|
| `train.csv` | 1460 | 50.02% |
| `test.csv` | 1459| 49.98% |

However, in this project, `train.csv` is divided into traning (80%) and validation subsets (20%). So, the final proportion is as follows:

| Data       | Observations | Percentage |
|------------|-------------:|-----------:|
| Training data | 1168 | 40.01% |
| Validation data | 292 | 10.00% |
| Testing data | 1459 | 49.98% |


Why this approach?
- Kaggle already reserves ~50% of the data for testing.
- We use 80/20 split on `train.csv` to ensure enough data for training while keeping a validation set for hyperparameter tuning.
- For better robustness, **K-Fold Cross-Validation** can be applied instead of a single split.

