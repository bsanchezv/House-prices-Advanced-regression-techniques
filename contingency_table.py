import pandas as pd

def ct_x2(df, col1, col2):
    """
    Create a contingency table for two categorical variables.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    col1 (str): The name of the first categorical column.
    col2 (str): The name of the second categorical column.

    Returns:
    pd.DataFrame: A contingency table showing the frequency counts of combinations of the two categorical variables.
    """
    ct = pd.crosstab(
        df[col1].isna(),
        df[col2].isna(),
        rownames=[f'{col1} Missing'],
        colnames=[f'{col2} Missing'])
    return ct

def ct_x2_c2eq0(df, col1, col2):
    """
    Create a contingency table for a categorical variable and a numerical variable 
    where the numerical variable is compared to zero.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    col1 (str): The name of the categorical column.
    col2 (str): The name of the numerical column.

    Returns:
    pd.DataFrame: A contingency table showing the frequency counts of the categorical variable 
                  against whether the numerical variable equals zero.
    """

    ct = pd.crosstab(
        df[col1].isna(),
        df[col2] == 0,
        rownames=[f'{col1} Missing'],
        colnames=[f'{col2} == 0'])
    return ct