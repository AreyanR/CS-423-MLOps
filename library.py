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




#first define the pipeline
titanic_transformer = Pipeline(steps=[
    ('map_gender', CustomMappingTransformer('Gender', {'Male': 0, 'Female': 1})),
    ('map_class', CustomMappingTransformer('Class', {'Crew': 0, 'C3': 1, 'C2': 2, 'C1': 3})),
    ('ohe_joined', CustomOHETransformer(target_column='Joined')),
    ('tukey_age', CustomTukeyTransformer(target_column='Age', fence='outer')),
    ('tukey_fare', CustomTukeyTransformer(target_column='Fare', fence='outer')),
    ('scale_age', CustomRobustTransformer('Age')), 
    ('scale_fare', CustomRobustTransformer('Fare')), 
], verbose=True)



customer_transformer = Pipeline(steps=[
    ('drop_columns', CustomDropColumnsTransformer(column_list=['ID'], action='drop')),
    ('gender_mapping', CustomMappingTransformer('Gender', {'Male': 0, 'Female': 1})),
    ('experience_mapping', CustomMappingTransformer('Experience Level', {'low': 0, 'medium': 1, 'high': 2})),
    ('os_ohe', CustomOHETransformer('OS')),
    ('isp_ohe', CustomOHETransformer('ISP')),
    ('knn_impute', CustomKNNTransformer()),
    ('age_tukey', CustomTukeyTransformer('Age', fence='inner')),
    ('time_spent_tukey', CustomTukeyTransformer('Time Spent', fence='inner')),
    ('age_scale', CustomRobustTransformer('Age')),
    ('time_spent_scale', CustomRobustTransformer('Time Spent')),
], verbose=True)