from __future__ import annotations  #must be first line in your library!
import pandas as pd
import numpy as np
import types
from typing import Dict, Any, Optional, Union, List, Set, Hashable, Literal, Tuple, Self, Iterable
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import sklearn
from sklearn import set_config
set_config(transform_output="pandas")
from sklearn.pipeline import Pipeline
import warnings
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score






from sklearn.base import BaseEstimator, TransformerMixin

class CustomMappingTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that maps values in a specified column according to a provided dictionary.

    This transformer follows the scikit-learn transformer interface and can be used in
    a scikit-learn pipeline. It applies value substitution to a specified column using
    a mapping dictionary, which can be useful for encoding categorical variables or
    transforming numeric values.

    Parameters
    ----------
    mapping_column : str or int
        The name (str) or position (int) of the column to which the mapping will be applied.
    mapping_dict : dict
        A dictionary defining the mapping from existing values to new values.
        Keys should be values present in the mapping_column, and values should
        be their desired replacements.

    Attributes
    ----------
    mapping_dict : dict
        The dictionary used for mapping values.
    mapping_column : str or int
        The column (by name or position) that will be transformed.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'category': ['A', 'B', 'C', 'A']})
    >>> mapper = CustomMappingTransformer('category', {'A': 1, 'B': 2, 'C': 3})
    >>> transformed_df = mapper.fit_transform(df)
    >>> transformed_df
       category
    0        1
    1        2
    2        3
    3        1
    """

    def __init__(self, mapping_column: Union[str, int], mapping_dict: Dict[Hashable, Any]) -> None:
        """
        Initialize the CustomMappingTransformer.

        Parameters
        ----------
        mapping_column : str or int
            The name (str) or position (int) of the column to apply the mapping to.
        mapping_dict : Dict[Hashable, Any]
            A dictionary defining the mapping from existing values to new values.

        Raises
        ------
        AssertionError
            If mapping_dict is not a dictionary.
        """
        assert isinstance(mapping_dict, dict), f'{self.__class__.__name__} constructor expected dictionary but got {type(mapping_dict)} instead.'
        self.mapping_dict: Dict[Hashable, Any] = mapping_dict
        self.mapping_column: Union[str, int] = mapping_column  #column to focus on

    def fit(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> Self:
        """
        Fit method - performs no actual fitting operation.

        This method is implemented to adhere to the scikit-learn transformer interface
        but doesn't perform any computation.

        Parameters
        ----------
        X : pandas.DataFrame
            The input data to fit.
        y : array-like, default=None
            Ignored. Present for compatibility with scikit-learn interface.

        Returns
        -------
        self : instance of CustomMappingTransformer
            Returns self to allow method chaining.
        """
        print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
        return self  #always the return value of fit

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the mapping to the specified column in the input DataFrame.

        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame containing the column to transform.

        Returns
        -------
        pandas.DataFrame
            A copy of the input DataFrame with mapping applied to the specified column.

        Raises
        ------
        AssertionError
            If X is not a pandas DataFrame or if mapping_column is not in X.

        Notes
        -----
        This method provides warnings if:
        1. Keys in mapping_dict are not found in the column values
        2. Values in the column don't have corresponding keys in mapping_dict
        """
        assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
        assert self.mapping_column in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.mapping_column}"'  #column legit?
        warnings.filterwarnings('ignore', message='.*downcasting.*')  #squash warning in replace method below

        #now check to see if all keys are contained in column
        column_set: Set[Any] = set(X[self.mapping_column].unique())
        keys_not_found: Set[Any] = set(self.mapping_dict.keys()) - column_set
        if keys_not_found:
            print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] does not contain these keys as values {keys_not_found}\n")

        #now check to see if some keys are absent
        keys_absent: Set[Any] = column_set - set(self.mapping_dict.keys())
        if keys_absent:
            print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] does not contain keys for these values {keys_absent}\n")

        X_: pd.DataFrame = X.copy()
        X_[self.mapping_column] = X_[self.mapping_column].replace(self.mapping_dict)
        return X_

    def fit_transform(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> pd.DataFrame:
        """
        Fit to data, then transform it.

        Combines fit() and transform() methods for convenience.

        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame containing the column to transform.
        y : array-like, default=None
            Ignored. Present for compatibility with scikit-learn interface.

        Returns
        -------
        pandas.DataFrame
            A copy of the input DataFrame with mapping applied to the specified column.
        """
        #self.fit(X,y)  #commented out to avoid warning message in fit
        result: pd.DataFrame = self.transform(X)
        return result
    

class CustomOHETransformer(BaseEstimator, TransformerMixin):
    # Applies one-hot encoding to a specified categorical column

    def __init__(self, target_column: str, drop_first: bool = False) -> None:
        # Store the target column name and whether to drop the first category
        self.target_column = target_column
        self.drop_first = drop_first

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> Self:
        # No fitting needed; included to comply with sklearn interface
        print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
        return self  # always return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Validate input type
        assert isinstance(X, pd.DataFrame), f'{self.__class__.__name__}.transform expected DataFrame but got {type(X)} instead.'
        # Check that the target column exists in the DataFrame
        assert self.target_column in X.columns, f'{self.__class__.__name__}.transform unknown column {self.target_column}'

        # Apply one-hot encoding to the target column using pandas get_dummies
        X_encoded = pd.get_dummies(X, columns=[self.target_column], drop_first=self.drop_first, dtype=int)

        return X_encoded

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        # Call transform directly since fit does nothing
        return self.transform(X)
    

class CustomDropColumnsTransformer(BaseEstimator, TransformerMixin):
    # A transformer that either drops or keeps specified columns in a DataFrame
    # Can be used in a scikit-learn pipeline

    def __init__(self, column_list: List[str], action: Literal['drop', 'keep'] = 'drop') -> None:
        # Validate inputs: action must be 'drop' or 'keep', and column_list must be a list
        assert action in ['keep', 'drop'], f'DropColumnsTransformer action {action} not in ["keep", "drop"]'
        assert isinstance(column_list, list), f'DropColumnsTransformer expected list but saw {type(column_list)}'

        # Store the list of columns to drop or keep, and the action
        self.column_list: List[str] = column_list
        self.action: Literal['drop', 'keep'] = action

    def fit(self, X, y=None):
        # No fitting needed; just return self
        return self

    def transform(self, X):
        if self.action == 'keep':
            # Check that all columns to keep exist in the DataFrame
            missing = set(self.column_list) - set(X.columns)
            assert not missing, f'CustomDropColumnsTransformer.transform unknown columns to keep: {missing}'

            # Keep only the specified columns
            return X[self.column_list].copy()
        else:  # action == 'drop'
            # Warn if any columns to drop are not in the DataFrame
            missing = set(self.column_list) - set(X.columns)
            if missing:
                warnings.warn(f'CustomDropColumnsTransformer does not contain these columns to drop: {missing}')

            # Drop the specified columns, ignore missing ones
            return X.drop(columns=self.column_list, errors='ignore').copy()
        

class CustomSigma3Transformer(BaseEstimator, TransformerMixin):
    """
    A transformer that applies 3-sigma clipping to a specified column in a pandas DataFrame.

    This transformer follows the scikit-learn transformer interface and can be used in
    a scikit-learn pipeline. It clips values in the target column to be within three standard
    deviations from the mean.

    Parameters
    ----------
    target_column : Hashable
        The name of the column to apply 3-sigma clipping on.

    Attributes
    ----------
    high_wall : Optional[float]
        The upper bound for clipping, computed as mean + 3 * standard deviation.
    low_wall : Optional[float]
        The lower bound for clipping, computed as mean - 3 * standard deviation.
    """
    def __init__(self, target_column: Hashable):
        # init the column to clip
        self.target_column = target_column
        self.high_wall: Optional[float] = None
        self.low_wall: Optional[float] = None

    def fit(self, X: pd.DataFrame, y=None):
        # 3-sigma boundaries for the column
        assert self.target_column in X.columns, f"Sigma3Transformer: unknown column {self.target_column}"
        col = X[self.target_column]
        mean = col.mean()
        std = col.std()
        self.low_wall = mean - 3 * std
        self.high_wall = mean + 3 * std
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Clip values using computed 3-sigma boundaries
        assert self.low_wall is not None and self.high_wall is not None, \
            f"{self.__class__.__name__}.fit has not been called."
        X_ = X.copy()
        X_[self.target_column] = X_[self.target_column].clip(lower=self.low_wall, upper=self.high_wall)
        return X_.reset_index(drop=True)

               
class CustomTukeyTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that applies Tukey's fences (inner or outer) to a specified column in a pandas DataFrame.

    This transformer follows the scikit-learn transformer interface and can be used in a scikit-learn pipeline.
    It clips values in the target column based on Tukey's inner or outer fences.

    Parameters
    ----------
    target_column : Hashable
        The name of the column to apply Tukey's fences on.
    fence : Literal['inner', 'outer'], default='outer'
        Determines whether to use the inner fence (1.5 * IQR) or the outer fence (3.0 * IQR).

    Attributes
    ----------
    inner_low : Optional[float]
        The lower bound for clipping using the inner fence (Q1 - 1.5 * IQR).
    outer_low : Optional[float]
        The lower bound for clipping using the outer fence (Q1 - 3.0 * IQR).
    inner_high : Optional[float]
        The upper bound for clipping using the inner fence (Q3 + 1.5 * IQR).
    outer_high : Optional[float]
        The upper bound for clipping using the outer fence (Q3 + 3.0 * IQR).

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'values': [10, 15, 14, 20, 100, 5, 7]})
    >>> tukey_transformer = CustomTukeyTransformer(target_column='values', fence='inner')
    >>> transformed_df = tukey_transformer.fit_transform(df)
    >>> transformed_df
    """
    def __init__(self, target_column: Hashable, fence: Literal['inner', 'outer'] = 'outer'):
        # init with target column and fence type
        assert fence in ['inner', 'outer'], f"Invalid fence value: {fence}"
        self.target_column = target_column
        self.fence = fence
        self.low_wall: Optional[float] = None
        self.high_wall: Optional[float] = None

    def fit(self, X: pd.DataFrame, y=None):
        # find IQR-based boundaries using Tukey's rule
        assert self.target_column in X.columns, f"TukeyTransformer: unknown column {self.target_column}"
        col = X[self.target_column]
        q1 = col.quantile(0.25)
        q3 = col.quantile(0.75)
        iqr = q3 - q1
        multiplier = 1.5 if self.fence == 'inner' else 3.0
        self.low_wall = q1 - multiplier * iqr
        self.high_wall = q3 + multiplier * iqr
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # clip column values to within Tukey boundaries
        assert self.low_wall is not None and self.high_wall is not None, \
            f"{self.__class__.__name__}.transform fit has not been called."
        X_ = X.copy()
        X_[self.target_column] = X_[self.target_column].clip(lower=self.low_wall, upper=self.high_wall)
        return X_
    
class CustomRobustTransformer(BaseEstimator, TransformerMixin):
  """Applies robust scaling to a specified column in a pandas DataFrame.
    This transformer calculates the interquartile range (IQR) and median
    during the `fit` method and then uses these values to scale the
    target column in the `transform` method.

    Parameters
    ----------
    column : str
        The name of the column to be scaled.

    Attributes
    ----------
    target_column : str
        The name of the column to be scaled.
    iqr : float
        The interquartile range of the target column.
    med : float
        The median of the target column.
  """
  def __init__(self, target_column: str):
        self.target_column = target_column
        self.iqr: float = None
        self.med: float = None

  def fit(self, X: pd.DataFrame, y=None):
      # Check that X is a DataFrame and column exists
      assert isinstance(X, pd.DataFrame), f'{self.__class__.__name__}.fit expected DataFrame but got {type(X)} instead.'
      assert self.target_column in X.columns, f'{self.__class__.__name__}.fit unrecognizable column {self.target_column}.'

      col = X[self.target_column].dropna()
      q1 = col.quantile(0.25)
      q3 = col.quantile(0.75)
      self.iqr = q3 - q1
      self.med = col.median()
      return self

  def transform(self, X: pd.DataFrame) -> pd.DataFrame:
      # Check that fit was called
      if self.iqr is None or self.med is None:
          raise AssertionError(f"{self.__class__.__name__}: NotFittedError: This instance is not fitted yet. Call 'fit' first.")
      
      # Make a copy
      X_ = X.copy()
      
      # Only scale if IQR is not zero
      if self.iqr != 0:
          X_[self.target_column] = (X_[self.target_column] - self.med) / self.iqr
      
      return X_


class CustomKNNTransformer(BaseEstimator, TransformerMixin):
  """Imputes missing values using KNN.

  This transformer wraps the KNNImputer from scikit-learn and hard-codes
  add_indicator to be False. It also ensures that the input and output
  are pandas DataFrames.

  Parameters
  ----------
  n_neighbors : int, default=5
      Number of neighboring samples to use for imputation.
  weights : {'uniform', 'distance'}, default='uniform'
      Weight function used in prediction. Possible values:
      "uniform" : uniform weights. All points in each neighborhood
      are weighted equally.
      "distance" : weight points by the inverse of their distance.
      in this case, closer neighbors of a query point will have a
      greater influence than neighbors which are further away.
  """
  #your code below
  def __init__(self, n_neighbors=5, weights="uniform"):
        self.n_neighbors = n_neighbors
        self.weights = weights
        # Make the imputer, force add_indicator to False (required for our pipeline)
        self.knn_imputer = KNNImputer(n_neighbors=self.n_neighbors,
                                      weights=self.weights,
                                      add_indicator=False)
        self._fitted = False        # Just to track if fit was actually called
        self._fit_columns = None    # Save the column names from fit() to check later

  def fit(self, X: pd.DataFrame, y=None):
      # Warn if you're asking for more neighbors than there are rows — won't work right
      if self.n_neighbors > len(X):
          warnings.warn(f"n_neighbors={self.n_neighbors} is more than number of rows ({len(X)}).", UserWarning)

      self.knn_imputer.fit(X)
      self._fitted = True
      self._fit_columns = list(X.columns)  # Store column names to check against later
      return self

  def transform(self, X: pd.DataFrame) -> pd.DataFrame:
    if not self._fitted:
        raise AssertionError(f"{self.__class__.__name__}: NotFittedError: You need to call fit() before transform().")

    # Check for column mismatch before calling .transform()
    if list(X.columns) != self._fit_columns:
        warnings.warn("Warning: Columns or column order are different from when you fit the data.", UserWarning)

    try:
        result = self.knn_imputer.transform(X)
        return pd.DataFrame(result, columns=X.columns, index=X.index)
    except ValueError as e:
        raise ValueError(str(e))

class CustomTargetTransformer(BaseEstimator, TransformerMixin):
    """
    A target encoder that applies smoothing and returns np.nan for unseen categories.

    Parameters:
    -----------
    col: name of column to encode.
    smoothing : float, default=10.0
        Smoothing factor. Higher values give more weight to the global mean.
    """

    def __init__(self, col: str, smoothing: float =10.0):
        self.col = col
        self.smoothing = smoothing
        self.global_mean_ = None
        self.encoding_dict_ = None

    def fit(self, X, y):
        """
        Fit the target encoder using training data.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data features.
        y : array-like of shape (n_samples,)
            Target values.
        """
        assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.fit expected Dataframe but got {type(X)} instead.'
        assert self.col in X, f'{self.__class__.__name__}.fit column not in X: {self.col}. Actual columns: {X.columns}'
        assert isinstance(y, Iterable), f'{self.__class__.__name__}.fit expected Iterable but got {type(y)} instead.'
        assert len(X) == len(y), f'{self.__class__.__name__}.fit X and y must be same length but got {len(X)} and {len(y)} instead.'

        #Create new df with just col and target - enables use of pandas methods below
        X_ = X[[self.col]]
        target = self.col+'_target_'
        X_[target] = y

        # Calculate global mean
        self.global_mean_ = X_[target].mean()

        # Get counts and means
        counts = X_[self.col].value_counts().to_dict()    #dictionary of unique values in the column col and their counts
        means = X_[target].groupby(X_[self.col]).mean().to_dict() #dictionary of unique values in the column col and their means

        # Calculate smoothed means
        smoothed_means = {}
        for category in counts.keys():
            n = counts[category]
            category_mean = means[category]
            # Apply smoothing formula: (n * cat_mean + m * global_mean) / (n + m)
            smoothed_mean = (n * category_mean + self.smoothing * self.global_mean_) / (n + self.smoothing)
            smoothed_means[category] = smoothed_mean

        self.encoding_dict_ = smoothed_means

        return self

    def transform(self, X):
        """
        Transform the data using the fitted target encoder.
        Unseen categories will be encoded as np.nan.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data to transform.
        """

        assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
        assert self.encoding_dict_, f'{self.__class__.__name__}.transform not fitted'

        X_ = X.copy()

        # Map categories to smoothed means, naturally producing np.nan for unseen categories, i.e.,
        # when map tries to look up a value in the dictionary and doesn't find the key, it automatically returns np.nan. That is what we want.
        X_[self.col] = X_[self.col].map(self.encoding_dict_)

        return X_

    def fit_transform(self, X, y):
        """
        Fit the target encoder and transform the input data.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data features.
        y : array-like of shape (n_samples,)
            Target values.
        """
        return self.fit(X, y).transform(X)
    


def find_random_state(
    features_df: pd.DataFrame,
    labels: Iterable,
    transformer: TransformerMixin,
    n: int = 200
                  ) -> Tuple[int, List[float]]:
    """
    Finds an optimal random state for train-test splitting based on F1-score stability.

    This function iterates through `n` different random states when splitting the data,
    applies a transformation pipeline, and trains a K-Nearest Neighbors classifier.
    It calculates the ratio of test F1-score to train F1-score and selects the random
    state where this ratio is closest to the mean.

    Parameters
    ----------
    features_df : pd.DataFrame
        The feature dataset.
    labels : Union[pd.Series, List]
        The corresponding labels for classification (can be a pandas Series or a Python list).
    transformer : TransformerMixin
        A scikit-learn compatible transformer for preprocessing.
    n : int, default=200
        The number of random states to evaluate.

    Returns
    -------
    rs_value : int
        The optimal random state where the F1-score ratio is closest to the mean.
    Var : List[float]
        A list containing the F1-score ratios for each evaluated random state.

    Notes
    -----
    - If the train F1-score is below 0.1, that iteration is skipped.
    - A higher F1-score ratio (closer to 1) indicates better train-test consistency.
    """

    model = KNeighborsClassifier(n_neighbors=5)
    Var: List[float] = []  # Collect test_f1/train_f1 ratios

    for i in range(n):
        train_X, test_X, train_y, test_y = train_test_split(
            features_df, labels, test_size=0.2, shuffle=True,
            random_state=i, stratify=labels  # Works with both lists and pd.Series
        )

        # Apply transformation pipeline
        transform_train_X = transformer.fit_transform(train_X, train_y)
        transform_test_X = transformer.transform(test_X)

        # Train model and make predictions
        model.fit(transform_train_X, train_y)
        train_pred = model.predict(transform_train_X)
        test_pred = model.predict(transform_test_X)

        train_f1 = f1_score(train_y, train_pred)

        if train_f1 < 0.1:
            continue  # Skip if train_f1 is too low

        test_f1 = f1_score(test_y, test_pred)
        f1_ratio = test_f1 / train_f1  # Ratio of test to train F1-score

        Var.append(f1_ratio)

    mean_f1_ratio: float = np.mean(Var)
    rs_value: int = np.abs(np.array(Var) - mean_f1_ratio).argmin()  # Index of value closest to mean

    return rs_value, Var    


def dataset_setup(original_table: pd.DataFrame, label_column_name: str, the_transformer, rs: int, ts: float = 0.2):
    """
    Splits a dataset and applies a transformer pipeline.

    Parameters:
    - original_table: Full dataframe including features and label
    - label_column_name: Name of the column to treat as label
    - the_transformer: Transformer to apply
    - rs: Random state value
    - ts: Test size (default 0.2)

    Returns:
    - X_train_numpy, X_test_numpy, y_train_numpy, y_test_numpy
    """
    from sklearn.model_selection import train_test_split
    labels = original_table[label_column_name].to_list()
    features = original_table.drop(columns=[label_column_name])
    
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=ts, shuffle=True, random_state=rs, stratify=labels
    )
    
    X_train_transformed = the_transformer.fit_transform(X_train, y_train)
    X_test_transformed = the_transformer.transform(X_test)

    return X_train_transformed.to_numpy(), X_test_transformed.to_numpy(), np.array(y_train), np.array(y_test)

# random state variables
titanic_variance_based_split = 107
customer_variance_based_split = 113





titanic_transformer = Pipeline(steps=[
    ('map_gender', CustomMappingTransformer('Gender', {'Male': 0, 'Female': 1})),
    ('map_class', CustomMappingTransformer('Class', {'Crew': 0, 'C3': 1, 'C2': 2, 'C1': 3})),
    ('target_joined', CustomTargetTransformer(col='Joined', smoothing=10)),
    ('tukey_age', CustomTukeyTransformer(target_column='Age', fence='outer')),
    ('tukey_fare', CustomTukeyTransformer(target_column='Fare', fence='outer')),
    ('scale_age', CustomRobustTransformer(target_column='Age')),
    ('scale_fare', CustomRobustTransformer(target_column='Fare')),
    ('impute', CustomKNNTransformer(n_neighbors=5)),
    ], verbose=True)


customer_transformer = Pipeline(steps=[
    ('map_os', CustomMappingTransformer('OS', {'Android': 0, 'iOS': 1})),
    ('target_isp', CustomTargetTransformer(col='ISP')),
    ('map_level', CustomMappingTransformer('Experience Level', {'low': 0, 'medium': 1, 'high':2})),
    ('map_gender', CustomMappingTransformer('Gender', {'Male': 0, 'Female': 1})),
    ('tukey_age', CustomTukeyTransformer('Age', 'inner')),  #from chapter 4
    ('tukey_time spent', CustomTukeyTransformer('Time Spent', 'inner')),  #from chapter 4
    ('scale_age', CustomRobustTransformer(target_column='Age')), #from 5
    ('scale_time spent', CustomRobustTransformer(target_column='Time Spent')), #from 5
    ('impute', CustomKNNTransformer(n_neighbors=5)),
    ], verbose=True)



def titanic_setup(titanic_table: pd.DataFrame, transformer=titanic_transformer, rs=titanic_variance_based_split, ts: float = 0.2):
    return dataset_setup(titanic_table, 'Survived', transformer, rs, ts)

def customer_setup(customer_table, transformer=customer_transformer, rs=customer_variance_based_split, ts=.2):
    return dataset_setup(customer_table, 'Rating', transformer, rs, ts)



"""
def threshold_results(thresh_list, actuals, predicted):
    result_df = pd.DataFrame(columns=['threshold', 'precision', 'recall', 'f1', 'accuracy', 'auc'])
    
    # Loop through each threshold and calculate the metrics
    for t in thresh_list:
        # Apply threshold to predicted values
        yhat = [1 if v >= t else 0 for v in predicted]
        
        # Calculate precision, recall, f1, accuracy, and auc
        precision = precision_score(actuals, yhat, zero_division=0)
        recall = recall_score(actuals, yhat, zero_division=0)
        f1 = f1_score(actuals, yhat)
        accuracy = accuracy_score(actuals, yhat)
        auc = roc_auc_score(actuals, predicted)
        
        # Add the results to the DataFrame
        result_df.loc[len(result_df)] = {
            'threshold': t,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'auc': auc
        }
    
    # Round values to 2 decimal places to match output style
    result_df = result_df.round(2)
    
    # Return the DataFrame with the max values highlighted in pink
    return result_df, result_df.style.highlight_max(color='pink', axis=0)
"""

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score

def threshold_results(thresh_list, actuals, predicted):
    result_df = pd.DataFrame(columns=['threshold', 'precision', 'recall', 'f1', 'accuracy', 'auc'])

    auc = roc_auc_score(actuals, predicted)  # compute once

    for t in thresh_list:
        yhat = [1 if v >= t else 0 for v in predicted]

        precision = precision_score(actuals, yhat, zero_division=0)
        recall = recall_score(actuals, yhat, zero_division=0)
        f1 = f1_score(actuals, yhat, zero_division=0)
        accuracy = accuracy_score(actuals, yhat)

        result_df.loc[len(result_df)] = {
            'threshold': t,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'auc': auc
        }

    result_df = result_df.round(2)
    return result_df, result_df.style.highlight_max(color='pink', axis=0)
