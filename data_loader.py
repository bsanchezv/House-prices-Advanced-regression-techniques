import os
import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data(config):
    """Load data from CSV files and split into training and validation sets.
    Args:
        config (dict): Configuration dictionary containing file paths and parameters.
    Returns:
        X_train (pd.DataFrame): Training features.
        X_val (pd.DataFrame): Validation features.
        y_train (pd.Series): Training target.
        y_val (pd.Series): Validation target.
    """
    # Determine the project root directory
    project_root = os.path.dirname((__file__))
    train_path = os.path.join(project_root, config['data']['train_path'])

    # Load data
    train_df = pd.read_csv(train_path)

    # Split features and target
    X = train_df.drop(config['modeling']['TARGET_COLUMN'], axis=1)
    y = train_df[config['modeling']['TARGET_COLUMN']]

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=config['modeling']['VALID_SIZE'], # for validation size, actually. The test size is already defined by the test set
        random_state=config['base']['RANDOM_SEED']
    )

    return X_train, X_val, y_train, y_val