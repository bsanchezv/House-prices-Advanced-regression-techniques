import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data(config):
    # Load data
    train_df = pd.read_csv(config['data']['train_path'])
    test_df = pd.read_csv(config['data']['test_path'])

    # Split features and target
    X = train_df.drop(config['modeling']['TARGET_COLUMN'], axis=1)
    y = train_df[config['modeling']['TARGET_COLUMN']]

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=config['modeling']['VALID_SIZE'], # for validation size, actually. The test size is already defined by the test set
        random_state=config['base']['RANDOM_SEED']
    )