

################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################

import torch
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split

from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor, XGBRFRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

# --- Ensemble of Ensemble
from sklearn.ensemble import VotingRegressor
from itertools import combinations

import sys
# sys.path.append(r'D:/작업방/업무 - 자동차 ★★★/Workspace_Python/DS_Module')



import copy
import functools
import numpy as np
import pandas as pd
import torch
# import tensorflow as tf











################################################################################################
# Customizing DS_NoneEncoder
class DS_NoneEncoder:
    """
    - DataFrame / ndarray 모두를 그대로 통과시키는 'no-op encoder'
    - 2D DataFrame: column name 복원
    - 2D ndarray: column index 복원
    - ND ndarray: 그냥 pass, feature_names 개념 없음 → get_feature_names_out() = None
    """
    def __init__(self):
        self.feature_names_ = None
        self.input_type_ = None
        self.original_shape_ = None
        self._fitted = False

    def fit(self, X):
        if isinstance(X, pd.DataFrame):
            self.input_type_ = "dataframe"
            self.feature_names_ = list(X.columns)
            self.original_shape_ = X.shape

        elif isinstance(X, np.ndarray):
            self.input_type_ = "ndarray"
            self.original_shape_ = X.shape

            if X.ndim == 2:
                self.feature_names_ = list(range(X.shape[1]))
            else:
                self.feature_names_ = None  # 다차원에서는 feature name 없음

        else:
            raise TypeError("Input must be pandas.DataFrame or numpy.ndarray")

        self._fitted = True
        return self

    def transform(self, X):
        if not self._fitted:
            raise RuntimeError("DS_NoneEncoder is not fitted yet.")
        return np.array(X)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        if not self._fitted:
            raise RuntimeError("DS_NoneEncoder is not fitted yet.")

        X_arr = np.array(X)

        # --- 원래 DataFrame이었다면 DataFrame으로 복원 ---
        if self.input_type_ == "dataframe":
            if X_arr.ndim != 2:
                raise ValueError(
                    f"Original data was a 2D DataFrame but got ndim={X_arr.ndim}"
                )

            index = X.index if isinstance(X, pd.DataFrame) else list(range(X_arr.shape[0]))

            columns = (
                self.feature_names_
                if self.feature_names_ is not None and len(self.feature_names_) == X_arr.shape[1]
                else list(range(X_arr.shape[1]))
            )
            return pd.DataFrame(X_arr, index=index, columns=columns)

        # --- 원래 ndarray였다면 shape 복원 ---
        if self.input_type_ == "ndarray":
            if (
                self.original_shape_ is not None
                and X_arr.size == np.prod(self.original_shape_)
            ):
                try:
                    X_arr = X_arr.reshape(self.original_shape_)
                except Exception:
                    pass
            return X_arr

        raise RuntimeError("Unknown input_type_ in DS_NoneEncoder.")

    def get_feature_names_out(self):
        """
        - 2D DataFrame → column name 반환
        - 2D ndarray → column index 반환
        - ND ndarray → 에러 대신 None 반환  (사용자가 체크해서 무시 가능)
        """
        if not self._fitted:
            raise RuntimeError("DS_NoneEncoder is not fitted yet.")

        # DataFrame or 2D ndarray
        if self.feature_names_ is not None:
            return np.array(self.feature_names_, dtype=object)

        # ND array → feature name 개념 없음 → None 반환
        return None

    def __repr__(self):
        return f"DS_NoneEncoder(fitted={self._fitted}, input_type={self.input_type_}, shape={self.original_shape_})"


# class DS_NoneEncoder():
#     def __init__(self):
#         self.feature_names = None

#     def fit(self, X):
#         if isinstance(X, pd.DataFrame):
#             self.feature_names = list(X.columns)
#         elif isinstance(X, np.ndarray):
#             self.feature_names = list(range(X.shape[1]))
#         else:
#             raise TypeError("Input must be pandas.DataFrame or numpy.ndarray")
    
#     def transform(self, X):
#         return np.array(X)
    
#     def fit_transform(self, X):
#         self.fit(X)
#         return self.transform(X)
    
#     def inverse_transform(self, X):
#         if isinstance(X, pd.DataFrame):
#             index = list(X.index)
#         elif isinstance(X, np.ndarray):
#             index = list(range(X.shape[0]))
#         else:
#             raise TypeError("Input must be pandas.DataFrame or numpy.ndarray")
        
#         return pd.DataFrame(X, index=index, columns=self.feature_names)
    
#     def get_feature_names_out(self):
#         return np.array(self.feature_names, dtype=object)
    
#     def __repr__(self):
#         return "DS_NoneEncoder()"





# Customizing LabelEncoder
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class DS_LabelEncoder:
    """
    - 원래 의도: 2D tabular data (DataFrame / ndarray)에 대해 컬럼별 LabelEncoder를 적용
    - 확장:
        * DataFrame: 기존과 동일하게 2D 컬럼별 인코딩
        * ndarray(1D/2D/3D/... ND):
            - 마지막 축을 feature 축으로 보고, 나머지 축은 모두 sample 축으로 flatten
            - shape (..., n_features) -> (-1, n_features) 로 펴서 각 feature별 LabelEncoder 적용
            - transform 후 다시 원래 shape로 reshape
            - inverse_transform도 동일한 방식으로 복원
    """

    def __init__(self, nan_value=-1, unseen_as_nan=False):
        self.encoder = {}              # col_name -> LabelEncoder
        self.feature_names = None      # 마지막 축의 feature 이름
        self.nan_replacements = {}     # col_name -> nan 대체값
        self.original_dtypes = {}      # col_name -> dtype
        self.nan_value = nan_value
        self.unseen_as_nan = unseen_as_nan

        self.input_type_ = None        # "dataframe" or "ndarray"
        self.original_shape_ = None    # ndarray일 때 원본 shape 저장
        self._fitted = False

    # -----------------------
    # 내부 유틸
    # -----------------------
    def _to_2d_dataframe_for_fit(self, X):
        """
        fit 시 입력을 2D DataFrame으로 통일해서 처리하는 헬퍼.
        - DataFrame이면 그대로 (shape: (n_samples, n_features))
        - ndarray면 마지막 축을 feature 축으로 보고 flatten:
            shape: (..., C) -> (-1, C)
        """
        if isinstance(X, pd.DataFrame):
            self.input_type_ = "dataframe"
            self.original_shape_ = X.shape  # (n_samples, n_features)
            self.feature_names = list(X.columns)
            return X.copy()

        elif isinstance(X, np.ndarray):
            self.input_type_ = "ndarray"
            self.original_shape_ = X.shape

            if X.ndim == 1:
                # (N,) -> (N, 1)
                X_2d = X.reshape(-1, 1)
            else:
                # (..., C)에서 마지막 축을 feature 축으로 간주
                last_dim = X.shape[-1]
                X_2d = X.reshape(-1, last_dim)

            # 컬럼 이름: x_cat_1, x_cat_2, ...
            self.feature_names = [f"x_cat_{i+1}" for i in range(X_2d.shape[1])]
            return pd.DataFrame(X_2d, columns=self.feature_names)

        else:
            raise TypeError("Input must be pandas.DataFrame or numpy.ndarray")

    def _to_2d_dataframe_for_transform(self, X):
        """
        transform / inverse_transform 공용 헬퍼
        - DataFrame이면 그대로 사용
        - ndarray면 fit과 동일하게 마지막 축을 feature 축으로 보고 flatten
        """
        if isinstance(X, pd.DataFrame):
            return X.copy(), X.shape, "dataframe"

        elif isinstance(X, np.ndarray):
            orig_shape = X.shape

            if X.ndim == 1:
                X_2d = X.reshape(-1, 1)
            else:
                last_dim = X.shape[-1]
                X_2d = X.reshape(-1, last_dim)

            # fit 때의 feature_names 기준으로 DataFrame 생성
            if self.feature_names is None:
                cols = [f"x_cat_{i+1}" for i in range(X_2d.shape[1])]
            else:
                cols = self.feature_names

            df = pd.DataFrame(X_2d, columns=cols)
            return df, orig_shape, "ndarray"

        else:
            raise TypeError("Input must be pandas.DataFrame or numpy.ndarray")

    # -----------------------
    # fit / transform / inverse
    # -----------------------
    def fit(self, X):
        data = self._to_2d_dataframe_for_fit(X)

        for col in self.feature_names:
            col_data = data[col]
            self.original_dtypes[col] = col_data.dtype

            # object지만 전부 숫자면 숫자로 캐스팅
            if col_data.dtype == object:
                try:
                    col_data = pd.to_numeric(col_data)
                except ValueError:
                    pass

            le = LabelEncoder()

            # dtype별 NaN 처리 전략
            if np.issubdtype(col_data.dtype, np.floating):
                replacement = self.nan_value
                self.nan_replacements[col] = replacement
                col_data = col_data.fillna(replacement).astype(np.int64)

            elif np.issubdtype(col_data.dtype, np.integer):
                replacement = self.nan_value
                self.nan_replacements[col] = replacement
                col_data = col_data.fillna(replacement)

            elif col_data.dtype == object:
                replacement = "__missing__"
                self.nan_replacements[col] = replacement
                col_data = col_data.fillna(replacement)

            else:
                raise ValueError(f"Unsupported dtype for column {col}: {col_data.dtype}")

            le.fit(col_data)
            self.encoder[col] = le

        self._fitted = True
        return self

    def transform(self, X):
        if not self._fitted:
            raise RuntimeError("DS_LabelEncoder is not fitted yet.")

        data, orig_shape, input_kind = self._to_2d_dataframe_for_transform(X)
        transformed = pd.DataFrame(index=data.index)

        for col in self.feature_names:
            col_data = data[col]

            # object인데 숫자만 있으면 numeric으로
            if col_data.dtype == object:
                try:
                    col_data = pd.to_numeric(col_data)
                except ValueError:
                    pass

            replacement = self.nan_replacements[col]
            col_data = col_data.fillna(replacement)

            le = self.encoder[col]
            known_classes = set(le.classes_)

            if self.unseen_as_nan:
                # unseen → NaN 대체값으로 치환 후 transform
                col_data = col_data.apply(
                    lambda x: x if x in known_classes else replacement
                )
                transformed[col] = le.transform(col_data)
            else:
                # unseen → 새로운 category로 추가
                unseen_values = set(col_data) - known_classes
                if unseen_values:
                    le.classes_ = np.append(le.classes_, list(unseen_values))
                transformed[col] = le.transform(col_data)

        arr = np.array(transformed)

        # 원래 ndarray였으면 shape 복원
        if input_kind == "ndarray":
            # original_shape_: (..., C), arr.shape: (N_flat, C)
            # 원소 수는 그대로고 last_dim도 동일하다고 가정
            if len(orig_shape) == 1:
                # (N,)->(N,1)로 펴서 인코딩했으니 다시 (N,)로
                return arr.reshape(orig_shape)
            else:
                return arr.reshape(orig_shape)
        else:
            # DataFrame 입력이면 그냥 2D ndarray 반환
            return arr

    def inverse_transform(self, X):
        if not self._fitted:
            raise RuntimeError("DS_LabelEncoder is not fitted yet.")

        data_2d, orig_shape, input_kind = self._to_2d_dataframe_for_transform(X)
        inversed = pd.DataFrame(index=data_2d.index)

        for col in self.feature_names:
            le = self.encoder[col]
            decoded = le.inverse_transform(data_2d[col])
            replacement = self.nan_replacements[col]

            decoded = np.where(decoded == replacement, np.nan, decoded)
            inversed[col] = decoded

        if input_kind == "ndarray":
            # ndarray로 다시 변환 + 원래 shape로 reshape
            arr = inversed.to_numpy()

            if len(orig_shape) == 1:
                # (N,1) -> (N,)
                return arr.reshape(orig_shape)
            else:
                return arr.reshape(orig_shape)
        else:
            # DataFrame으로 fit/transform 했으면 DataFrame 반환
            # 컬럼명도 feature_names 기준으로 맞춰줌
            inversed.columns = self.feature_names
            return inversed

    # -----------------------
    # 그 외 유틸
    # -----------------------
    def get_feature_names_out(self):
        """
        - 마지막 축 기준 feature 이름 반환
        - DataFrame / 2D ndarray / ND ndarray 모두에서 동일하게 동작
        """
        if not self._fitted:
            raise RuntimeError("DS_LabelEncoder is not fitted yet.")

        return np.array(self.feature_names, dtype=object) if self.feature_names is not None else None

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def __repr__(self):
        repr_str = "DS_LabelEncoder("
        if len(self.encoder) > 0:
            repr_str += str(list(self.encoder.keys()))
        repr_str += ")"
        return repr_str

# class DS_LabelEncoder:
#     def __init__(self, nan_value=-1, unseen_as_nan=False):
#         self.encoder = {}
#         self.feature_names = None
#         self.nan_replacements = {}
#         self.original_dtypes = {}
#         self.nan_value = nan_value
#         self.unseen_as_nan = unseen_as_nan
    
#     def fit(self, X):
#         if isinstance(X, pd.DataFrame):
#             self.feature_names = list(X.columns)
#             data = X.copy()
#         elif isinstance(X, np.ndarray):
#             self.feature_names = list([f"x_cat_{i+1}" for i in range(X.shape[1])])
#             data = pd.DataFrame(X, columns=self.feature_names)
#         else:
#             raise TypeError("Input must be pandas.DataFrame or numpy.ndarray")
        
#         for col in self.feature_names:
#             col_data = data[col]
#             self.original_dtypes[col] = col_data.dtype
            
#             # object dtype이지만 내부 값이 전부 숫자면 숫자로 처리
#             if col_data.dtype == object:
#                 try:
#                     col_data = pd.to_numeric(col_data)
#                 except ValueError:
#                     pass
            
#             le = LabelEncoder()
            
#             if np.issubdtype(col_data.dtype, np.floating):
#                 replacement = self.nan_value
#                 self.nan_replacements[col] = replacement
#                 col_data = col_data.fillna(replacement).astype(np.int64)
#             elif np.issubdtype(col_data.dtype, np.integer):
#                 replacement = self.nan_value
#                 self.nan_replacements[col] = replacement
#                 col_data = col_data.fillna(replacement)
#             elif col_data.dtype == object:
#                 replacement = '__missing__'
#                 self.nan_replacements[col] = replacement
#                 col_data = col_data.fillna(replacement)
#             else:
#                 raise ValueError(f"Unsupported dtype for column {col}: {col_data.dtype}")
            
#             le.fit(col_data)
#             self.encoder[col] = le
        
#         return self
    
#     def transform(self, X):
#         if isinstance(X, pd.DataFrame):
#             data = X.copy()
#         elif isinstance(X, np.ndarray):
#             data = pd.DataFrame(X, columns=self.feature_names)
#         else:
#             raise TypeError("Input must be pandas.DataFrame or numpy.ndarray")
        
#         transformed = pd.DataFrame(index=data.index)
        
#         for col in self.feature_names:
#             col_data = data[col]
            
#             if col_data.dtype == object:
#                 try:
#                     col_data = pd.to_numeric(col_data)
#                 except ValueError:
#                     pass
            
#             replacement = self.nan_replacements[col]
#             col_data = col_data.fillna(replacement)
            
#             le = self.encoder[col]
#             known_classes = set(le.classes_)
            
#             if self.unseen_as_nan:
#                 # unseen 값을 NaN 대체값으로 변환
#                 col_data = col_data.apply(lambda x: x if x in known_classes else replacement)
#                 transformed[col] = le.transform(col_data)
#             else:
#                 # unseen 값을 새로운 category로 추가
#                 unseen_values = set(col_data) - known_classes
#                 if unseen_values:
#                     le.classes_ = np.append(le.classes_, list(unseen_values))
#                 transformed[col] = le.transform(col_data)
        
#         return np.array(transformed)
    
#     def inverse_transform(self, X):
#         if isinstance(X, pd.DataFrame):
#             data = X.copy()
#         elif isinstance(X, np.ndarray):
#             data = pd.DataFrame(X)
#         else:
#             raise TypeError("Input must be pandas.DataFrame or numpy.ndarray")
        
#         inversed = pd.DataFrame(index=data.index)
        
#         for col in self.feature_names:
#             le = self.encoder[col]
#             decoded = le.inverse_transform(data[col])
#             replacement = self.nan_replacements[col]
            
#             decoded = np.where(decoded == replacement, np.nan, decoded)
#             inversed[col] = decoded
        
#         return inversed

#     def get_feature_names_out(self):
#         return np.array(self.feature_names, dtype=object)
    
#     def fit_transform(self, X):
#         return self.fit(X).transform(X)

#     def __repr__(self):
#         repr_str = "DS_LabelEncoder("
#         if len(self.encoder) > 0:
#             # encoders_str = '\n'.join([f"  {k}: {v}" for k, v in self.encoder.items()])
#             # return repr_str + '\n{\n' + encoders_str + '\n}'
#             repr_str += str(list(self.encoder.keys()))
        
#         repr_str += ")"
#         return repr_str


# Customizing StandardScaler
class DS_StandardScaler:
    """
    - np.nanmean / np.nanstd 기반의 Standard Scaling
    - DataFrame / ndarray 모두 지원
    - axis에 따라 다차원에서도 유연하게 동작
        * axis=None      : 전체에 대해 스칼라 mean / std
        * axis=k (int)   : 해당 축에 대해 mean / std (keepdims=True로 브로드캐스팅 가능)
    - fit 시 입력 타입 / shape / feature_names 기록해서
      inverse_transform에서 최대한 원래 형태로 복원
    """

    def __init__(self, axis=None, eps=1e-8, with_mean=True, with_std=True):
        """
        axis   : np.nanmean / np.nanstd에 전달할 axis
                 - None 이면 전체에 대해 스칼라 mean / std
                 - int 또는 tuple 로 전달 가능 (np.nanmean 규칙 그대로)
        eps    : std가 0인 경우 분모를 eps로 보정
        with_mean : 평균 빼기 여부
        with_std  : 표준편차 나누기 여부
        """
        self.axis = axis
        self.eps = eps
        self.with_mean = with_mean
        self.with_std = with_std

        self.mean_ = None
        self.std_ = None

        self.input_type_ = None      # "dataframe" or "ndarray"
        self.original_shape_ = None  # ndarray일 때 원래 shape
        self.feature_names_ = None   # DataFrame 컬럼명 (또는 2D ndarray feature index)
        self._fitted = False

    # ---------------------------
    # fit / transform / inverse
    # ---------------------------
    def fit(self, X):
        """
        X : pd.DataFrame 또는 np.ndarray (차원 제한 없음)
        """
        if isinstance(X, pd.DataFrame):
            self.input_type_ = "dataframe"
            self.original_shape_ = X.shape
            self.feature_names_ = list(X.columns)
            arr = X.to_numpy(dtype=float)  # numeric이라고 가정
        elif isinstance(X, np.ndarray):
            self.input_type_ = "ndarray"
            self.original_shape_ = X.shape
            arr = np.asarray(X, dtype=float)

            # 2D tabular인 경우만 feature_names를 만들어 둠
            if arr.ndim == 2:
                self.feature_names_ = [f"x_{i}" for i in range(arr.shape[1])]
            else:
                self.feature_names_ = None
        else:
            raise TypeError("Input must be pandas.DataFrame or numpy.ndarray")

        if self.axis is None:
            mean = np.nanmean(arr)
            std = np.nanstd(arr)
        else:
            mean = np.nanmean(arr, axis=self.axis, keepdims=True)
            std = np.nanstd(arr, axis=self.axis, keepdims=True)

        # std == 0 방지
        std = np.where(std < self.eps, 1.0, std)

        self.mean_ = mean
        self.std_ = std
        self._fitted = True
        return self

    def transform(self, X):
        """
        X : DataFrame 또는 ndarray
        반환 : ndarray (입력 타입은 유지하지 않음, 수치 연산용으로 사용)
        """
        if not self._fitted:
            raise RuntimeError("DS_StandardScaler is not fitted yet. Call 'fit' first.")

        if isinstance(X, pd.DataFrame):
            arr = X.to_numpy(dtype=float)
        elif isinstance(X, np.ndarray):
            arr = np.asarray(X, dtype=float)
        else:
            raise TypeError("Input must be pandas.DataFrame or numpy.ndarray")

        out = arr
        if self.with_mean:
            out = out - self.mean_
        if self.with_std:
            out = out / self.std_

        return out

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        """
        - transform(X) 결과나, 같은 shape의 숫자 배열을 받아서
        - scaling 이전 값으로 복원
        - fit 당시 DataFrame 이었으면 DataFrame으로,
          ndarray 였으면 원래 shape로 reshape해서 ndarray로 반환
        """
        if not self._fitted:
            raise RuntimeError("DS_StandardScaler is not fitted yet. Call 'fit' first.")

        # 우선 ndarray로 통일
        if isinstance(X, pd.DataFrame):
            arr = X.to_numpy(dtype=float)
            index = X.index
        else:
            arr = np.asarray(X, dtype=float)
            index = None

        out = arr
        if self.with_std:
            out = out * self.std_
        if self.with_mean:
            out = out + self.mean_

        # 원래 타입/shape 복원
        if self.input_type_ == "dataframe":
            # DataFrame이었으니 (n_samples, n_features)일 것으로 가정
            # feature_names가 있으면 그대로 사용
            cols = self.feature_names_ if self.feature_names_ is not None else None
            if index is None:
                # inverse_transform에 ndarray 들어온 경우: 새 index 생성
                index = range(out.shape[0])
            return pd.DataFrame(out, index=index, columns=cols)

        elif self.input_type_ == "ndarray":
            # 원래 ndarray였으면 original_shape_로 reshape 시도
            if (
                self.original_shape_ is not None
                and out.size == int(np.prod(self.original_shape_))
            ):
                try:
                    out = out.reshape(self.original_shape_)
                except Exception:
                    # reshape 실패하면 현재 모양 그대로
                    pass
            return out

        else:
            # 이 경우는 발생하면 안 됨
            return out

    # ---------------------------
    # feature name / repr
    # ---------------------------
    def get_feature_names_out(self):
        """
        - DataFrame / 2D ndarray로 fit 했을 때만 feature_names_ 반환
        - 그 외(ND)에는 None
        """
        if not self._fitted:
            raise RuntimeError("DS_StandardScaler is not fitted yet.")

        if self.feature_names_ is None:
            return None
        return np.array(self.feature_names_, dtype=object)

    def __repr__(self):
        return (
            f"DS_StandardScaler("
            f"axis={self.axis}, "
            f"with_mean={self.with_mean}, "
            f"with_std={self.with_std}, "
            f"fitted={self._fitted}, "
            f"shape={self.original_shape_}"
            f")"
        )


# class DS_StandardScaler():
#     def __init__(self, axis=None):
#         self.mean = None
#         self.std = None
#         self.axis = axis
        
#     def fit(self, x):
#         if self.axis is None:
#             self.mean = np.nanmean(x)
#             self.std = np.nanstd(x)
#         else:
#             self.mean = np.nanmean(x, axis=self.axis, keepdims=True)
#             self.std = np.nanstd(x, axis=self.axis, keepdims=True)
        
#     def transform(self, x):
#         if (self.mean is not None) and (self.std is not None):
#             return (x - self.mean)/self.std
    
#     def fit_transform(self, x):
#         self.fit(x)
#         return self.transform(x)
    
#     def inverse_transform(self, x):
#         if (self.mean is not None) and (self.std is not None):
#             return x * self.std + self.mean
    
#     def __repr__(self):
#         return "DS_StandardScaler()"

################################################################################################



# Customizing DataPreprocessing
# Customizing DataPreprocessing
class DataPreprocessing:
    """
    다차원(Tabular + Time series 등) 데이터를 대상으로

      1) (옵션) split 이전 전처리(pre_split_fn)
      2) split (train/valid/test)
      3) (옵션) split 이후 전처리(pre_encoding_fn)
      4) encoder 기반 변환
      5) TensorDataset / DataLoader 생성

    을 일관된 파이프라인으로 처리하는 유틸리티.
    """

    def __init__(
        self,
        *data_args,
        split_size=(0.7, 0.1, 0.2),
        encoder=None,
        batch_size=1,
        index=None,
        random_state=None,
        shuffle=True,
        stratify=None,
        **kwargs,
        ):
        """
        Args:
            *data_args : (X, y, ...) 처럼 샘플 차원을 공유하는 데이터들
                         각 data는 (N, ...) 형상이어야 함.
            split_size : (train, valid, test) 또는 (train, test) 비율
            encoder    : 각 data_arg에 대응하는 encoder 리스트
                         (None -> NoneEncoder로 대체)
            batch_size : DataLoader 기본 배치 크기
            index      : {'train': idx_array, 'valid': ..., 'test': ...}
                         미리 정의된 인덱스 dict (선택)
            random_state : 시드
            shuffle      : split 시 셔플 여부
            stratify     : 계층 분할용 라벨 (1D array-like)
            **kwargs     : DataLoader 옵션 (num_workers 등)
        """
        self.data_args = data_args

        # 샘플 수 일치 검사
        if len(data_args) > 0:
            lengths = [len(d) for d in data_args]
            assert len(set(lengths)) == 1, "Arguments must have same length in the first dimension"
            self.features_names = []

        # 입력 데이터를 numpy 배열로 통일
        self.np_data = tuple(self._to_numpy(data) for data in data_args)
        self.full_index = np.arange(len(self.np_data[0])) if len(self.np_data) > 0 else None

        # index (mutable default 방지)
        if index is None:
            self.index = {}
        else:
            self.index_names = ['train', 'valid', 'test']
            self.index = {idx_name: np.array(index[idx_name]) for idx_name in self.index_names if idx_name in index.keys()}

        self.split_data = {}
        self.transformed_data = {}
        self.tensor_dataset = {}
        self.tensor_dataloader = {}

        # encoder 설정
        if encoder is None:
            self.encoder = [DS_NoneEncoder() for _ in range(len(self.np_data))]
        else:
            self.encoder = [(DS_NoneEncoder() if enc is None else enc) for enc in encoder]

        # split 비율 설정
        self._set_split_size(split_size)

        # random / stratify / dataloader 옵션
        self.random_state = random_state
        self.generator = torch.Generator()
        if self.random_state is not None:
            self.generator.manual_seed(self.random_state)

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.stratify = self._to_numpy(stratify) if stratify is not None else None
        self.kwargs = kwargs

        # dataset & dataloader 핸들
        self.dataset = None
        self.dataloader = None

    # ------------------------------------------------------------------
    # 기본 유틸
    # ------------------------------------------------------------------
    def _to_numpy(self, data):
        """
        DataFrame, Series, torch.Tensor 등 다양한 입력을 numpy로 통일.
        """
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return data.values
        return np.array(data)

    def _set_split_size(self, split_size):
        """
        split 비율 정규화 및
        train_test_split_size / train_valid_split_size 계산.
        """
        self.split_size = [s / np.sum(split_size) for s in split_size]

        if len(self.split_size) == 2:
            # train / test
            self.train_test_split_size = self.split_size
            self.train_valid_split_size = None
        elif len(self.split_size) == 3:
            # train / valid / test
            self.train_test_split_size = [
                self.split_size[0] + self.split_size[1],  # (train+valid)
                self.split_size[2],                       # test
            ]
            self.train_valid_split_size = [
                s / self.train_test_split_size[0] for s in self.split_size[:2]
            ]
        else:
            raise ValueError("split_size must have length 2 or 3.")

    # ------------------------------------------------------------------
    # 전처리 훅
    # ------------------------------------------------------------------
    def pre_split(self, pre_split_fn):
        """
        split 이전 전체 데이터(np_data)에 대한 전처리.

        pre_split_fn:
            - callable: fn(*np_data) -> tuple(np_data_like)
            - list/tuple[callable]: 순차적으로 적용
        """
        if pre_split_fn is None:
            return

        if callable(pre_split_fn):
            new_data = pre_split_fn(*self.np_data)
        else:
            new_data = self.np_data
            for fn in pre_split_fn:
                new_data = fn(*new_data)

        self.np_data = tuple(new_data)
        # 데이터 길이가 바뀌었을 수 있으므로 full_index 재생성
        self.full_index = np.arange(len(self.np_data[0]))

    def pre_encoding(self, pre_encoding_fn):
        """
        split 이후 split_data에 대한 encoding process이전의 전처리.

        pre_encoding_fn:
            - callable: 모든 split(train/valid/test)에 동일 적용
            - dict: {'train': fn_train, 'valid': fn_valid, 'test': fn_test}
                    각 split별로 다른 함수 적용 가능

        각 fn은 (*data_tuple) -> tuple(data_tuple_like) 형태여야 함.
        """
        if pre_encoding_fn is None or len(self.split_data) == 0:
            return

        for split_name, data_tuple in self.split_data.items():
            if callable(pre_encoding_fn):
                fn = pre_encoding_fn
            elif isinstance(pre_encoding_fn, dict):
                fn = pre_encoding_fn.get(split_name, None)
            else:
                raise ValueError("pre_encoding_fn must be callable or dict of callables.")

            if fn is None:
                continue

            new_tuple = fn(*data_tuple)
            self.split_data[split_name] = tuple(new_tuple)
            
    def pre_tensor_dataset(self, pre_tensor_dataset_fn):
        """
        split 이후 split_data에 대한 encoding process이전의 전처리.

        pre_tensor_dataset_fn:
            - callable: encoding 이후 모든 split(train/valid/test)에 동일 적용
            - dict: {'train': fn_train, 'valid': fn_valid, 'test': fn_test}
                    각 split별로 다른 함수 적용 가능

        각 fn은 (*data_tuple) -> tuple(data_tuple_like) 형태여야 함.
        """
        if pre_tensor_dataset_fn is None or len(self.transformed_data) == 0:
            return

        for split_name, data_tuple in self.transformed_data.items():
            if callable(pre_tensor_dataset_fn):
                fn = pre_tensor_dataset_fn
            elif isinstance(pre_tensor_dataset_fn, dict):
                fn = pre_tensor_dataset_fn.get(split_name, None)
            else:
                raise ValueError("pre_tensor_dataset_fn must be callable or dict of callables.")

            if fn is None:
                continue

            new_tuple = fn(*data_tuple)
            self.transformed_data[split_name] = tuple(new_tuple)

    # ------------------------------------------------------------------
    # Split
    # ------------------------------------------------------------------
    def split(
        self,
        index=None,
        split_size=None,
        random_state=None,
        shuffle=None,
        stratify=None,
        verbose=0,
        ):
        """
        데이터를 train/valid/test 로 분할.

        index:
            - None: 내부 로직(train_test_split) 사용
            - dict: {'train': idx_array, 'valid': idx_array, 'test': idx_array}
                    그대로 사용

        split_size, random_state, shuffle, stratify 모두 호출 시 덮어쓰기 가능.
        """
        # 파라미터 업데이트
        if random_state is not None:
            self.random_state = random_state
            self.generator = torch.Generator()
            self.generator.manual_seed(self.random_state)

        if shuffle is not None:
            self.shuffle = shuffle

        self.stratify = self.stratify if stratify is None else self._to_numpy(stratify)
        if split_size is not None:
            self._set_split_size(split_size)

        # index 우선순위: 인자 > self.index
        if index is None:
            index = self.index
        else:
            index = {idx_name: np.array(index[idx_name]) for idx_name in self.index_names if idx_name in index.keys()}

        # --------------------------------------------------
        # 1) index가 이미 주어진 경우
        # --------------------------------------------------
        if len(index) > 0:
            self.index = index
            self.split_data = {key: [] for key in self.index.keys()}

            # split_size를 index 길이 기준으로 재계산
            lengths = np.array([len(v) for v in self.index.values()], dtype=float)
            self.split_size = (lengths / lengths.sum()).tolist()

        # --------------------------------------------------
        # 2) index가 없는 경우: train_test_split 사용
        # --------------------------------------------------
        else:
            if self.full_index is None:
                raise ValueError("No data to split.")

            full_index = self.full_index
            stratify_tt = self.stratify if self.stratify is not None else None

            # train / test
            self.train_idx, self.test_idx = train_test_split(
                full_index,
                test_size=self.train_test_split_size[-1],
                random_state=self.random_state,
                shuffle=self.shuffle,
                stratify=stratify_tt,
            )

            # valid 포함 여부에 따라 분기
            if self.train_valid_split_size is not None:
                train_valid_stratify = (
                    self.stratify[self.train_idx] if self.stratify is not None else None
                )
                self.train_idx, self.valid_idx = train_test_split(
                    self.train_idx,
                    test_size=self.train_valid_split_size[-1],
                    random_state=self.random_state,
                    shuffle=self.shuffle,
                    stratify=train_valid_stratify,
                )
                self.index = {
                    "train": self.train_idx,
                    "valid": self.valid_idx,
                    "test": self.test_idx,
                }
                self.split_data = {"train": [], "valid": [], "test": []}
            else:
                self.index = {"train": self.train_idx, "test": self.test_idx}
                self.split_data = {"train": [], "test": []}

        # --------------------------------------------------
        # 실제 데이터 split
        # --------------------------------------------------
        for split_name, idx in self.index.items():
            self.split_data[split_name] = tuple(
                data[idx] for data in self.np_data
            )

        # full_index 재설정 (모든 split 인덱스 합집합)
        self.full_index = np.concatenate(list(self.index.values()), axis=0)
        self.full_index.sort()

        # verbose
        if verbose > 0:
            print({k: len(v) for k, v in self.index.items()})

        return self.split_data

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def encoding(self, encoder=None):
        """
        split_data에 encoder 적용.
        train: fit_transform
        나머지: transform
        """
        if encoder is not None:
            self.encoder = [(DS_NoneEncoder() if e is None else e) for e in encoder]

        if len(self.split_data) == 0:
            raise RuntimeError("'split' must be performed first.")
        if len(self.encoder) != len(self.np_data):
            raise RuntimeError("'encoder' must have the same number as data_args.")

        self.transformed_data = {}

        for split_name, data_tuple in self.split_data.items():
            transformed_data_list = []
            for ei, enc in enumerate(self.encoder):
                X = data_tuple[ei]

                # train은 fit_transform, 나머지는 transform
                if split_name == "train":
                    if hasattr(enc, "fit_transform"):
                        transformed = enc.fit_transform(X)
                    else:
                        transformed = enc.transform(X)
                else:
                    transformed = enc.transform(X)

                # OneHotEncoder sparse 대비
                if "OneHotEncoder" in enc.__class__.__name__:
                    transformed = transformed.toarray().astype(np.int64)

                transformed_data_list.append(transformed)

            self.transformed_data[split_name] = tuple(transformed_data_list)

        return self.transformed_data

    # ------------------------------------------------------------------
    # TensorDataset / DataLoader
    # ------------------------------------------------------------------
    def make_tensor_dataset(self):
        """
        transformed_data를 TensorDataset으로 변환.
        """
        if len(self.transformed_data) == 0:
            raise RuntimeError("'encoding' must be performed first.")

        self.tensor_dataset = {}

        for split_name, data_tuple in self.transformed_data.items():
            tensors = []
            for data in data_tuple:
                dtype_str = str(data.dtype)
                if "float" in dtype_str:
                    tensors.append(torch.tensor(data, dtype=torch.float32))
                elif "int" in dtype_str:
                    tensors.append(torch.tensor(data, dtype=torch.long))
                elif "bool" in dtype_str:
                    tensors.append(torch.tensor(data, dtype=torch.bool))
                else:
                    raise TypeError("Only float / int / bool numpy arrays can be converted to torch tensor.")
            self.tensor_dataset[split_name] = TensorDataset(*tensors)

        return self.tensor_dataset

    def make_tensor_dataloader(self, batch_size=None, shuffle=None, random_state=None, **kwargs):
        """
        TensorDataset 기반으로 DataLoader 생성.
        """
        # 파라미터 업데이트
        self.batch_size = self.batch_size if batch_size is None else batch_size
        self.shuffle = self.shuffle if shuffle is None else shuffle
        if random_state is not None:
            self.random_state = random_state
            self.generator = torch.Generator()
            self.generator.manual_seed(self.random_state)

        # kwargs 병합 (기존 kwargs + 새 kwargs)
        loader_kwargs = dict(self.kwargs)
        loader_kwargs.update(kwargs)

        if len(self.tensor_dataset) == 0:
            raise RuntimeError("'make_tensor_dataset' must be performed first.")

        self.tensor_dataloader = {}
        for name, tensor_dataset in self.tensor_dataset.items():
            self.tensor_dataloader[name] = DataLoader(
                tensor_dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,  # 원래 로직 유지 (원하면 호출 시 shuffle=False로)
                generator=self.generator,
                **loader_kwargs,
            )

        return self.tensor_dataloader

    # ------------------------------------------------------------------
    # full pipeline
    # ------------------------------------------------------------------
    def fit_transform(
        self,
        split_size=None,
        encoder=None,
        random_state=None,
        shuffle=None,
        stratify=None,
        verbose=0,
        pre_split_fn=None,
        pre_encoding_fn=None,
        ):
        """
        pre_split → split → post_split → encoding 전체 수행.
        """
        # 1) pre-split 전처리
        if pre_split_fn is not None:
            self.pre_split(pre_split_fn)

        # 2) split
        self.split(
            split_size=split_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify,
            verbose=verbose,
        )

        # 3) post-split 전처리
        if pre_encoding_fn is not None:
            self.pre_encoding(pre_encoding_fn)

        # 4) encoding
        return self.encoding(encoder=encoder)

    def fit_tensor_dataloader(
        self,
        split_size=None,
        encoder=None,
        batch_size=None,
        random_state=None,
        shuffle=None,
        stratify=None,
        verbose=0,
        pre_split_fn=None,
        pre_encoding_fn=None,
        pre_tensor_dataset_fn=None,
        **kwargs,
        ):
        """
        pre_split → split → post_split → encoding → TensorDataset → DataLoader
        전체 파이프라인 한 번에 수행.
        """
        self.fit_transform(
            split_size=split_size,
            encoder=encoder,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify,
            verbose=verbose,
            pre_split_fn=pre_split_fn,
            pre_encoding_fn=pre_encoding_fn,
        )
        
        # tensor화 직전에 전처리
        if pre_tensor_dataset_fn is not None:
            self.pre_tensor_dataset(pre_tensor_dataset_fn)
            
        self.make_tensor_dataset()
        return self.make_tensor_dataloader(
            batch_size=batch_size,
            shuffle=shuffle,
            random_state=random_state,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # 기타 유틸
    # ------------------------------------------------------------------
    def _repr_dictformat(self, dict_in, intend=4, parentheses_sep="\n"):
        """객체 상태 요약 Helper."""
        if len(dict_in) == 0:
            return "{}"
        repr_output = "{" + f"{parentheses_sep}"
        for i, (name, shapes) in enumerate(dict_in.items()):
            comma = "," if i < len(dict_in) - 1 else ""
            if i == 0:
                if parentheses_sep == "\n":
                    repr_output += f"{' ' * intend}'{name}': {shapes}{comma}"
                else:
                    repr_output += f"'{name}': {shapes}{comma}"
            else:
                repr_output += f"\n{' ' * intend}'{name}': {shapes}{comma}"
        if parentheses_sep == "\n":
            repr_output += f"{parentheses_sep}{' ' * intend}" + "}"
        else:
            repr_output += f"{parentheses_sep}" + "}"
        return repr_output

    def get_feature_names_out(self):
        """
        encoder들이 가진 feature 이름 반환.
        (get_feature_names_out이 없는 encoder는 None 반환)
        """
        names = []
        for enc in self.encoder:
            if hasattr(enc, "get_feature_names_out"):
                names.append(enc.get_feature_names_out())
            else:
                names.append(None)
        return tuple(names)

    def __repr__(self):
        """객체 상태 요약."""
        repr_str = "<np_data : " + ", ".join([str(arr.shape) for arr in self.np_data]) + ">"

        if len(self.split_data) > 0:
            split_ratio_str = {name: float(size) for name, size in zip(self.split_data.keys(), self.split_size)}
            split_size_str = {
                name: tuple([data.shape for data in data_tuple])
                for name, data_tuple in self.split_data.items()
            }
            repr_str += (
                f"\n  └ self.split_size: {split_ratio_str}"
                + f"\n    split_data: {self._repr_dictformat(split_size_str, intend=20, parentheses_sep='')}"
            )

        if len(self.transformed_data) > 0:
            encoder_str = str(tuple(self.encoder))
            transformed_str = {
                name: tuple([data.shape for data in data_tuple])
                for name, data_tuple in self.transformed_data.items()
            }
            repr_str += (
                f"\n  └ self.encoder: {encoder_str}"
                + f"\n    transformed_data: {self._repr_dictformat(transformed_str, intend=20, parentheses_sep='')}"
            )

        if len(self.tensor_dataset) > 0:
            ds_str = {name: str(ds) for name, ds in self.tensor_dataset.items()}
            repr_str += f"\n  └ self.tensor_dataset: {self._repr_dictformat(ds_str, intend=20, parentheses_sep='')}"

        if len(self.tensor_dataloader) > 0:
            repr_str += (
                f"\n  └ (dataloader options) batch_size: {self.batch_size}, shuffle: {self.shuffle}, random_state: {self.random_state}"
                + f"\n    self.tensor_dataloader: {self._repr_dictformat(self.tensor_dataloader, intend=20, parentheses_sep='')}"
            )
        return repr_str


# df_y = pd.DataFrame(np.random.rand(20,1), columns=['y'])
# df_X_con = pd.DataFrame(np.random.rand(20,5), columns=[f"x{i+1}" for i in range(5)])
# X_cat_gen = np.concatenate([np.random.randint(0,4, size=(20,2)), np.random.randint(0,2, size=(20,3))], axis=1)
# df_X_cat = pd.DataFrame(X_cat_gen, columns=[f"x{i+6}" for i in range(5)])

# np_y = (np.random.rand(100,1) > 0.3).astype(np.int64)
# np_X = np.random.rand(100,50,4)
# np_bool = (np.random.rand(100,50) > 0.5).astype(bool)



# data_tuple = (df_y, df_X_con, df_X_cat)
# data_tuple = (df_y.to_numpy(), df_X_con.to_numpy(), df_X_cat.to_numpy())
# data_tuple = (np_y, np_X, dnp_bool)


# pc = DataPreprocessing(*data_tuple, split_size=(0.8, 0.2), stratify=df_X_cat['x10'])
# pc = DataPreprocessing(*data_tuple, stratify=df_X_cat['x10'], encoder=[StandardScaler(), StandardScaler(), DS_LabelEncoder()])
# pc = DataPreprocessing(*data_tuple, stratify=df_X_cat['x10'], encoder=[StandardScaler(), StandardScaler(), OneHotEncoder(sparse_output=True)])
# pc = DataPreprocessing(*data_tuple, stratify=df_X_cat['x10'], encoder=[StandardScaler(), None, DS_LabelEncoder()])
# pc = DataPreprocessing(*data_tuple, stratify=df_X_cat['x10'])

# pc = DataPreprocessing(*data_tuple
#                        ,split_size=(5,1,2)
#                        ,stratify=df_X_cat.iloc[:,[-1]]
#                        ,encoder=[StandardScaler(), StandardScaler(), DS_LabelEncoder()]
#                        ,batch_size=128
#                        ,shuffle=True
#                        ,random_state=1
#                        )

# a1 = pc.split()
# a2 = pc.encoding()
# a3 = pc.make_tensor_dataset()
# a4 = pc.make_tensor_dataloader()
# a2 = pc.fit_transform()
# a4 = pc.fit_tensor_dataloader()

# pc.np_data
# pc.index
# pc.get_feature_names_out()
# pc.split_size
# pc.transformed_data
# pc.tensor_dataset
# pc.tensor_dataloader




# --------------------------------------------------------------------------------------



class DS_KFold():
    def __init__(self, n_splits=5, shuffle=False, random_state=None, valid_ratio=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.valid_ratio = valid_ratio
        self.skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        self.kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def _combine_labels(self, stratify):
        """
        다중 출력 stratify 데이터를 하나의 레이블로 변환
        """
        stratify = np.array(stratify)
        if stratify.ndim == 1:
            return stratify
        return np.array(["_".join(map(str, row)) for row in stratify])

    def split(self, X, stratify=None, groups=None):
        """
        X: 특징 데이터
        stratify: 레이블 데이터 (None이면 stratify 없이 random split)
        """
        if stratify is None:
            splitter = self.kf
            stratify_combined = None
        else:
            stratify_combined = self._combine_labels(stratify)
            splitter = self.skf

        for train_idx, test_idx in splitter.split(X, stratify_combined, groups):
            if self.valid_ratio is not None and self.valid_ratio > 0:
                # validation split 시 stratify 가능 여부 확인
                if stratify is None:
                    stratify_arg = None
                else:
                    unique, counts = np.unique(stratify_combined[train_idx], return_counts=True)
                    stratify_arg = stratify_combined[train_idx] if np.min(counts) >= 2 else None

                train_sub_idx, valid_idx = train_test_split(
                    train_idx,
                    test_size=self.valid_ratio,
                    stratify=stratify_arg,
                    shuffle=self.shuffle,
                    random_state=self.random_state
                )
                yield train_sub_idx, valid_idx, test_idx
            else:
                yield train_idx, test_idx

    def get_n_splits(self, X=None, stratify=None, groups=None):
        return self.n_splits




################################################################################################




























# ---------------------------------------------------------------------------------------------------------------------

# Data 형태 및 정보를 바꿔주는 Class
class DataHandler:
    """
    【required (Library)】 numpy, pandas, collections.namedtuple
    【required (Function)】vector_info

    < input >
    . data : Scalar, Vector, (1-D, 2-D) list, matrix, DataFrame
    . columns : user defined name(columns)  
                    * {i} auto name sequence
    . dtypes : setting dtypes manually
    . ndim : setting ndim manually 
    . reset_dtype : whether reset_dtype (if True, dtypes are automatically reassigned)
    . object_threshold : numeric column automatically transform to object (only operate when 'reset_dtype' is True)

    < output >
    . vector : ('vector_info' : ('data' : {'data': , 'index': , 'name': ]), kind, ndim, dtypes, nuniques)
    . matrix : ('matrix_info' : ('data' : {'data': , 'index': , 'columns': ]), kind', ndim, dtypes, nunique)
    """
    def __init__(self):
        pass
    
    # input data 의 정보(kind, ndim, shape, possible_vector)를 알려주는 함수
    def data_info(self, data, dict_type=None, save_data=True):
        data_info_object = namedtuple('data_info', ['frame', 'kind', 'ndim', 'shape', 'possible_vector'])

        if 'list' in str(type(data)):
            if type(data[0]) == dict:
                kind = 'dict_records'
                data_result = pd.DataFrame(data)
                input_ndim = 2
            else:
                kind = 'list'
                if type(data[0]) == list:
                    input_ndim = 2
                    data_result = pd.DataFrame(data)
                else:
                    input_ndim = 1
                    data_result = pd.Series(data)
        elif 'dict' in str(type(data)):
            dict_first_value = list(data.values())[0]
            if type(dict_first_value) == list:
                kind = 'dict_split' if list(data.keys()) == ['index', 'columns', 'data'] else 'dict_list'
                input_ndim = 2
                data_result = pd.DataFrame(data=data['data'], index=data['index'], columns=data['columns']) if kind == 'dict_split' else pd.DataFrame(data)
            elif 'series' in str(type(dict_first_value)):
                kind = 'dict_series'
                input_ndim = 2
                data_result = pd.DataFrame(data)
            elif type(dict_first_value) == dict:
                if dict_type is None:
                    kind = 'dict_index' if type(list(data.keys())[0]) == int else 'dict'
                    data = pd.DataFrame(data).T.infer_objects() if kind == 'dict_index' else pd.DataFrame(data)
                elif dict_type is not None:
                    kind = 'dict_' + dict_type
                input_ndim = 2
                data_result = pd.DataFrame(data)
            elif sum([d in str(type(dict_first_value)) for d in ['int','float','str','bool']]) > 0:
                kind = 'dict'
                input_ndim = 1
                data_result = pd.Series(data)

        elif 'numpy' in str(type(data)):
            kind = 'numpy'
            input_ndim = data.ndim
            data_result = pd.Series(data) if input_ndim == 1 else pd.DataFrame(data)
        elif 'pandas' in str(type(data)):
            kind = 'pandas'
            input_ndim = data.ndim
            data_result = pd.Series(data) if input_ndim == 1 else pd.DataFrame(data)

        if input_ndim == 1:
            possible_vector = True
        elif data_result.shape[1] == 1:
            possible_vector = True
        else:
            possible_vector = False

        if save_data:
            frame_data = data_result.to_frame() if input_ndim == 1 else data_result
        else:
            frame_data = data_result.shape
            
        return data_info_object(frame_data, kind, input_ndim, data_result.shape, possible_vector)

    def possible_vector_verify(self, x):
        data_info_object = self.data_info(x)
        if data_info_object.possible_vector == False:
            raise ValueError("x required only '(n,) Scalar' or '(n,1) Matrix'. ")

    # '(n,) Scalar' or '(n,1) Matrix' 의 name을 추출 및 자동 부여해주는 함수
    def vector_info_split(self, x, index=None, name=None, dtype=None, reset_dtype=False, object_threshold=3, save_data=True):
        """
        【required (Library)】 numpy
        """
        data_info_object = self.data_info(x)
        if data_info_object.possible_vector == False:
            raise ValueError("x required only '(n,) Scalar' or '(n,1) Matrix'. ")

        # try:
        vector_instance = namedtuple('vector_info', ['data', 'kind', 'ndim', 'dtypes', 'nuniques'])

        # kind ***
        kind = data_info_object.kind

        # data ***
        
        data_series = data_info_object.frame.iloc[:,0]
            
        # nunique ***
        unique_vector = data_series.drop_duplicates()
        data = np.array(data_series).ravel()
        
        # ndim ***
        ndim = 1


        # index ***
        if index is not None:
            if data_info_object.shape[0] != len(index):
                raise ValueError("The length of index is different.")
            else:
                index_result = index
        else:
            if type(x) == list:
                index_result = np.arange(len(x))
            else:
                try:
                    index_result = np.array(x.index)
                except:
                    index_result = np.arange(data_info_object.shape[0])
        
        # dtype ***
        if dtype is None:       # auto dtype
            try:
                dtype_result = x.dtype
            except:
                dtype_result = np.array(x).dtype
            if reset_dtype and sum([t in str(dtype_result) for t in ['int', 'float']]) > 0:
                if len(unique_vector) <= object_threshold:
                    dtype_result = np.array(list(map(str, np.unique(np.array(x))))).dtype
        elif type(dtype) == dict:
            dtype_result = list(dtype.values())[0]
            dtype_result = dtype_result if 'dtype(' in str(type(dtype_result)).lower() else pd.Series(False, dtype=dtype_result).dtype
        elif type(dtype) == list:
            dtype_result = list(dtype)[0]
            dtype_result = dtype_result if 'dtype(' in str(type(dtype_result)).lower() else pd.Series(False, dtype=dtype_result).dtype

        else:
            dtype_result = dtype if 'dtype' in str(type(dtype)) else pd.Series(True, dtype=dtype).dtype

        # name ***
        if name is not None:
            name_result = name[0] if type(name) == list else name
        else:
            if data_info_object.kind == 'pandas' or ('dict' in data_info_object.kind and data_info_object.ndim == 2):
                name_result = list(data_info_object.frame.columns)[0]
            else:
              name_result = 'x'
        
        # save_data
        if not save_data:
            data = data.shape
        else:
            data = np.array(data)

        # result ***
        vector_dict = {}
        vector_dict['data'] = data
        vector_dict['index'] = index_result
        vector_dict['name'] = name_result
        

        return vector_instance(vector_dict, kind, ndim, dtype_result, len(unique_vector))

    # Data Split : Vector, Matrix Data 를 data, index, name(columns), dtype(s) 로 나눠주는 함수
    def data_info_split(self, data, index=None, columns=None, dtypes=None, reset_dtypes=False, object_threshold=3, ndim=None, dict_type=None, save_data=True):

        data_info_object = self.data_info(data)

        #####
        if ndim == 1 or (ndim is None and data_info_object.ndim == 1):  # Scalar or Series
            return self.vector_info_split(x=data, index=index, name=columns, dtype=dtypes, 
                                reset_dtype=reset_dtypes, object_threshold=object_threshold)

        elif ndim == 2 or (ndim is None and data_info_object.ndim == 2):  # 2-dim DataFrame

            matrix_instance = namedtuple('matrix_info', ['data', 'kind', 'ndim', 'dtypes', 'nuniques'])
            # frame_instance = namedtuple('value', ['data', 'index', 'columns'])
            
            # kind ***
            kind = data_info_object.kind
            
            # data ***
            matrix_X = data_info_object.frame

            # ndim ***
            result_ndim = 2

            # index ***
            if index is not None:
                if data_info_object.shape[0] != len(index):
                    raise ValueError("The length of index is different.")
                else:
                    index_result = index
            else:
                try:
                    index_result = np.array(matrix_X.index)
                except:
                    index_result = np.arange(matrix_X.shape[0])
            
            # columns ***
            if columns is not None:
                if '{i}' in columns:
                    columns_result = [eval(f"f'{columns}'") for i in range(1, matrix_X.shape[1]+1)]
                else:
                    if 'str' in str(type(columns)) and data_info_object.possible_vector == True:
                        columns_result = [columns]
                    else:
                        if matrix_X.shape[1] != len(columns) or type(columns) != list:
                            raise ValueError("'columns' error : the number of input_data's columns is equal to length of 'columns' list.")
                        else:
                            columns_result = columns
            else:
                if data_info_object.kind == 'pandas' or ('dict' in data_info_object.kind and data_info_object.ndim == 2):
                    columns_result = list(data_info_object.frame.columns)
                elif data_info_object.possible_vector == True:
                    columns_result = ['x']
                else:
                    columns_result = ('x'+pd.Series(np.arange(1,matrix_X.shape[1]+1)).astype(str)).tolist()
                    # [f'x{c}' for c in range(1, np.array(matrix_X).shape[1]+1)]

            # nuniques ***
            nuniques = pd.DataFrame(matrix_X).apply(lambda x:len(x.value_counts().index) ,axis=0)
            nuniques.index = columns_result

            # dtypes ***
            if dtypes is not None:
                if type(dtypes) == dict:
                    dtypes_result = dtypes.copy()
                    try:
                        dtypes_origin = matrix_X.dtypes
                    except:
                        dtypes_origin = pd.DataFrame(np.array(matrix_X)).infer_objects().dtypes
                    dtypes_origin.index = columns_result
                    dtypes_origin_dict = dtypes_origin.to_dict()

                    dtypes_dict = {c: d if 'dtype' in str(type(d)) else pd.Series(True, dtype=d).dtype for c, d in dtypes_result.items()}
                    dtypes_origin_dict.update(dtypes_dict)
                    dtypes_result = dtypes_origin_dict.copy()
                elif type(dtypes) == list:
                    dtypes_result = dtypes.copy()
                    dtypes_list = [d if 'dtype' in str(type(d)) else pd.Series(True, dtype=d).dtype for d in dtypes_result]
                    dtypes_result = dict(zip(columns_result, dtypes_list))
                else:
                    dtypes_result = dtypes
                    dtypes_result = {c: pd.Series(True, dtype=dtypes_result).dtype for c in columns_result}
            else:
                try:
                    dtype_series = pd.DataFrame(data).dtypes
                    dtype_series.index = columns_result
                    dtypes_result = dtype_series.to_dict()
                except:
                    dtypes_origin = pd.DataFrame(np.array(matrix_X)).infer_objects().dtypes
                    dtypes_origin.index = columns_result
                    dtypes_result = dtypes_origin.to_dict().copy()
                
                if reset_dtypes:
                    numeric_columns_dict = dict(filter(lambda x: sum([t in str(x) for t in ['int', 'float']]) > 0, dtypes_result.items()))
                    dtypes_result.update({c: np.array(list(map(str, np.array(matrix_X)[:,list(dtypes_result.keys()).index(c)]))).dtype for c in numeric_columns_dict.keys() if c in nuniques[nuniques<=object_threshold].index})
            
            # save_data ***
            if not save_data:
                matrix_X = matrix_X.shape
            else:
                matrix_X = np.array(matrix_X)

            # result ***
            matrix_dict = {}
            matrix_dict['data'] = matrix_X
            matrix_dict['index'] = index_result
            matrix_dict['columns'] = columns_result
        
            return matrix_instance(matrix_dict, kind, result_ndim, dtypes_result, nuniques.to_dict())

    # Split Object를 data로 바꿔주는 함수
    def info_to_data(self, instance):
        """
        【required (Library)】 numpy, pandas, collections.namedtuple, copy
        """
        copy_instance = copy.deepcopy(instance)
        kind = copy_instance.kind
        # # kind ***
        # if kind is None:
        #     kind = copy_instance.kind
        
        # # # name (columns)
        # # if copy_instance.ndim == 1:
        # #     columns = [copy_instance.data['name']]
        # # elif copy_instance.ndim == 2:
        # #     columns = copy_instance.data['columns']

        # # ndim ***
        # if ndim == 2:
        #     if copy_instance.ndim == 1:
        #         copy_instance = self.data_info_split(copy_instance.data['data'], columns=copy_instance.data['name'], 
        #                                 ndim=2, dtypes=copy_instance.dtypes, reset_dtypes=reset_dtypes, object_threshold=object_threshold)
        #         copy_instance = copy_instance._replace(kind=kind)

        # elif ndim == 1:
        #     if copy_instance.ndim == 2:
        #         if copy_instance.data['data'].shape[1] > 1:
        #             raise ValueError("'vector' or 'Series' only allows 1D-Array")
        #         else:
        #             copy_instance = self.data_info_split(copy_instance.data['data'], columns=copy_instance.data['columns'][0], 
        #                 ndim=1, dtypes=copy_instance.dtypes, reset_dtypes=reset_dtypes, object_threshold=object_threshold)
        #             copy_instance = copy_instance._replace(kind=kind)

        # transform ***
        if kind == 'numpy':
            if copy_instance.ndim == 1:
                return np.array(pd.Series(**copy_instance.data).astype(copy_instance.dtypes))
            if copy_instance.ndim == 2:
                return np.array(pd.DataFrame(**copy_instance.data).astype(copy_instance.dtypes))
        elif kind == 'list':
            if copy_instance.ndim == 1:
                    return np.array(pd.Series(**copy_instance.data).astype(copy_instance.dtypes)).tolist()
            if copy_instance.ndim == 2:
                return np.array(pd.DataFrame(**copy_instance.data).astype(copy_instance.dtypes)).tolist()
        elif kind == 'pandas':
            if copy_instance.ndim == 1:
                return pd.Series(**copy_instance.data).astype(copy_instance.dtypes)
            elif copy_instance.ndim == 2:
                return pd.DataFrame(**copy_instance.data).astype(copy_instance.dtypes)
        elif 'dict' in kind:
            if copy_instance.ndim == 1:
                return pd.Series(**copy_instance.data).astype(copy_instance.dtypes).to_dict()
            if copy_instance.ndim == 2:
                pd_frame = pd.DataFrame(**copy_instance.data).astype(copy_instance.dtypes)
                return pd_frame.to_dict() if kind == 'dict' else pd_frame.to_dict(kind.split('_')[1])

    # Data를 특정 Format에 맞게 바꿔주는 함수
    def transform(self, data, apply_data=None, apply_instance=None, return_type='data',
            apply_kind=True, apply_ndim=True, apply_index=False, apply_columns=False, apply_dtypes=True,
            reset_dtypes=False, object_threshold=3):
        """
        < input >
          . data
          . apply_data
          . apply_instance
          . apply_options : ['name', 'dtypes', 'shape', 'kind'] are allowed
        """
        # input_data_info = self.data_info(data)
        # matrix_data = input_data_info.frame

        if apply_instance is not None:
            apply_instance = copy.deepcopy(apply_instance)
        elif apply_data is not None:
            apply_instance = self.data_info_split(apply_data, save_data=False)
        else:
            apply_instance = self.data_info_split(data, save_data=False)
        to_instance_dict = {}

        if apply_ndim:
            if type(apply_ndim) != bool:
                to_instance_dict['ndim'] = apply_ndim
            else:
                to_instance_dict['ndim'] = 1 if apply_instance.ndim == 1 else 2
        if apply_index is not False:
            if type(apply_index) != bool:
                to_instance_dict['index'] = apply_index
            else:
                to_instance_dict['index'] = apply_instance.data['index']
        if apply_columns is not False:
            if type(apply_columns) != bool:
                to_instance_dict['columns'] = apply_columns
            else:
                to_instance_dict['columns'] = apply_instance.data['name'] if apply_instance.ndim == 1 else apply_instance.data['columns']
        if apply_dtypes is not False and reset_dtypes is False:
            if type(apply_dtypes) != bool:
                to_instance_dict['dtypes'] = apply_dtypes
            else:
                to_instance_dict['dtypes'] = apply_instance.dtypes if apply_instance.ndim == 1 else list(apply_instance.dtypes.values())
        # elif apply_dtypes is True and  reset_dtypes is False:

        result_instance = self.data_info_split(data, **to_instance_dict, reset_dtypes=reset_dtypes, object_threshold=object_threshold)
        
        if apply_kind:
            if type(apply_kind) != bool:
                result_instance = result_instance._replace(kind=apply_kind)
            else:
                result_instance = result_instance._replace(kind=apply_instance.kind)

        if return_type == 'data':
            return self.info_to_data(result_instance)
        elif return_type == 'instance':
            return result_instance
        elif return_type == 'all':
            return {'data': self.info_to_data(result_instance), 'instance': result_instance}

# ---------------------------------------------------------------------------------------------------------------------





# Tabular(or 2D-matrix) Data에서 여러 dtype에 대해 유연하게 LabelEncoding을 적용하는 Class
class TabularLabelEncoder:
    def __init__(self, nan_value=-1, unseen_as_nan=False):
        self.encoders = {}
        self.feature_names = None
        self.nan_replacements = {}
        self.original_dtypes = {}
        self.nan_value = nan_value
        self.unseen_as_nan = unseen_as_nan
    
    def fit(self, X):
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            data = X.copy()
        elif isinstance(X, np.ndarray):
            self.feature_names = list(range(X.shape[1]))
            data = pd.DataFrame(X)
        else:
            raise TypeError("Input must be pandas.DataFrame or numpy.ndarray")
        
        for col in self.feature_names:
            col_data = data[col]
            self.original_dtypes[col] = col_data.dtype
            
            # object dtype이지만 내부 값이 전부 숫자면 숫자로 처리
            if col_data.dtype == object:
                try:
                    col_data = pd.to_numeric(col_data)
                except ValueError:
                    pass
            
            le = LabelEncoder()
            
            if np.issubdtype(col_data.dtype, np.floating):
                replacement = self.nan_value
                self.nan_replacements[col] = replacement
                col_data = col_data.fillna(replacement).astype(np.int64)
            elif np.issubdtype(col_data.dtype, np.integer):
                replacement = self.nan_value
                self.nan_replacements[col] = replacement
                col_data = col_data.fillna(replacement)
            elif col_data.dtype == object:
                replacement = '__missing__'
                self.nan_replacements[col] = replacement
                col_data = col_data.fillna(replacement)
            else:
                raise ValueError(f"Unsupported dtype for column {col}: {col_data.dtype}")
            
            le.fit(col_data)
            self.encoders[col] = le
        
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            data = X.copy()
        elif isinstance(X, np.ndarray):
            data = pd.DataFrame(X)
        else:
            raise TypeError("Input must be pandas.DataFrame or numpy.ndarray")
        
        transformed = pd.DataFrame(index=data.index)
        
        for col in self.feature_names:
            col_data = data[col]
            
            if col_data.dtype == object:
                try:
                    col_data = pd.to_numeric(col_data)
                except ValueError:
                    pass
            
            replacement = self.nan_replacements[col]
            col_data = col_data.fillna(replacement)
            
            le = self.encoders[col]
            known_classes = set(le.classes_)
            
            if self.unseen_as_nan:
                # unseen 값을 NaN 대체값으로 변환
                col_data = col_data.apply(lambda x: x if x in known_classes else replacement)
                transformed[col] = le.transform(col_data)
            else:
                # unseen 값을 새로운 category로 추가
                unseen_values = set(col_data) - known_classes
                if unseen_values:
                    le.classes_ = np.append(le.classes_, list(unseen_values))
                transformed[col] = le.transform(col_data)
        
        return transformed
    
    def inverse_transform(self, X):
        if isinstance(X, pd.DataFrame):
            data = X.copy()
        elif isinstance(X, np.ndarray):
            data = pd.DataFrame(X)
        else:
            raise TypeError("Input must be pandas.DataFrame or numpy.ndarray")
        
        inversed = pd.DataFrame(index=data.index)
        
        for col in self.feature_names:
            le = self.encoders[col]
            decoded = le.inverse_transform(data[col])
            replacement = self.nan_replacements[col]
            
            decoded = np.where(decoded == replacement, np.nan, decoded)
            inversed[col] = decoded
        
        return inversed
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def __repr__(self):
        repr_str = "<customize.TabularLabelEncoder>"
        if len(self.encoders) > 0:
            encoders_str = '\n'.join([f"  {k}: {v}" for k, v in self.encoders.items()])
            return repr_str + '\n{\n' + encoders_str + '\n}'
        else:
            return repr_str



# ('matrix_info' : ('frame_info' : ['data', 'index', 'columns']), kind, dtypes, nuniques)
# Vector 형태 (1차원 vector 또는 (-1,1) Shpaed Matrix)의 숫자형 데이터에 Scaler를 적용하는 Class
class ScalerVector:
    """
    【required (Library)】 numpy, pandas, sklearn.preprocessing.*, copy.deepcopy
    【required (Function)】DataHandler, class_object_execution

    < Input >
     . scaler : Scaler Object or String
                * required: 'str type Name_of_Class', 'str type Name_of_Instance', 'Class object', 'instance object'
     . x : '1dim vector' or '(-1, 1)shaped matrix'

    < Method >
     . fit
     . transform
     . fit_transform
     . inverse_transform
    """
    def __init__(self, scaler='StandardScaler', **kwargs):
        self.name = 'Undefined'
        self.scaler = class_object_execution(scaler, **kwargs)
        self.DataHandler = DataHandler()

    def fit(self, x):
        fitted_info = self.DataHandler.data_info(x, save_data=False)
        self.fitted_ndim = fitted_info.ndim
        self.fitted_object = self.DataHandler.vector_info_split(x)
        self.scaler.fit(self.fitted_object.data['data'].reshape(-1,1))
        self.name = self.fitted_object.data['name']
        self.transformed_names = [self.name]

    def transform(self, x, fitted_format=False, apply_name=False, ndim=None, kind=None):
        transformed_info = self.DataHandler.data_info(x, save_data=False)
        transformed_object = self.DataHandler.vector_info_split(x)
        transformed_data = self.scaler.transform(transformed_object.data['data'].reshape(-1,1))

        if fitted_format:
            apply_name = transformed_object.data['name'] if apply_name is False else apply_name
            return self.DataHandler.transform(transformed_data, apply_instance=self.fitted_object, apply_columns=apply_name,
                apply_ndim=self.fitted_ndim, apply_index=transformed_object.data['index'], apply_dtypes='float')
        else:
            apply_kind = True if kind is None else kind
            apply_ndim = transformed_info.ndim if ndim is None else ndim
            return self.DataHandler.transform(transformed_data, apply_instance=transformed_object, apply_columns=True,
                apply_kind=apply_kind, apply_ndim=apply_ndim, apply_dtypes='float')

    def fit_transform(self, x, ndim=None, kind=None):
        self.fit(x)
        fitted_format = True if ndim is None and kind is None else False
        return self.transform(x, fitted_format=fitted_format, apply_name=True, ndim=ndim, kind=kind)

    def inverse_transform(self, x, fitted_format=True, apply_name=False, ndim=None, kind=None, dtypes=None):
        inversed_info = self.DataHandler.data_info(x, save_data=False)
        inversed_object = self.DataHandler.vector_info_split(x)
        inversed_data = self.scaler.inverse_transform(inversed_object.data['data'].reshape(-1,1))

        if fitted_format:
            apply_name = inversed_object.data['name'] if apply_name is False else apply_name
            # apply_kind = self.fitted_object.kind if kind is None else kind
            apply_ndim = self.fitted_ndim
            apply_dtypes = dtypes if dtypes is not None else True
            return self.DataHandler.transform(inversed_data, apply_instance=self.fitted_object, apply_columns=apply_name,
                apply_ndim=apply_ndim,
                apply_dtypes=apply_dtypes, apply_index=inversed_object.data['index'])
        else:
            apply_name = inversed_object.data['name'] if apply_name is False else apply_name
            apply_kind = True if kind is None else kind
            apply_ndim = inversed_info.ndim if ndim is None else ndim
            apply_dtypes = dtypes if dtypes is not None else self.fitted_object.dtypes
            return self.DataHandler.transform(inversed_data, apply_instance=inversed_object, apply_columns=apply_name,
                apply_kind=apply_kind, apply_ndim=apply_ndim, 
                apply_dtypes=apply_dtypes, apply_index=True)

    def __repr__(self):
        return f"(ScalerInstance) {self.name}: {self.scaler}"


# Vector 형태 (1차원 vector 또는 (-1,1) Shpaed Matrix)의 문자형 데이터에 Encoder를 적용하는 Class
class EncoderVector:
    """
    【required (Library)】 numpy, pandas, sklearn.preprocessing.*, copy.deepcopy
    【required (Function)】DataHandler, class_object_execution

    < Input >
     . encoder : Scaler Object or String
                * required: 'str type Name_of_Class', 'str type Name_of_Instance', 'Class object', 'instance object'
     . x : '1dim vector' or '(-1, 1)shaped matrix'

    < Method >
     . fit
     . transform
     . fit_transform
     . inverse_transform
     . get_params
     . get_feature_names_out
    """
    def __init__(self, encoder='OneHotEncoder', **kwargs):
        self.name='undefined'

        if 'OneHotEncoder' in str(encoder):
            if 'drop' not in kwargs.keys():
                kwargs.update({'drop':'first'})
            if 'sparse' not in kwargs.keys():
                kwargs.update({'sparse':False})
        # self.kwargs = kwargs
        self.DataHandler = DataHandler()
        self.encoder = class_object_execution(encoder, **kwargs)

    def fit(self, x):
        encoder_str = str(self.encoder)
        fitted_info = self.DataHandler.data_info(x, save_data=False)
        self.fitted_ndim = fitted_info.ndim
        self.fitted_object = self.DataHandler.vector_info_split(x)
        fitted_series = pd.Series(**self.fitted_object.data)
        # fitted_series = pd.Series(**self.fitted_object.data).apply(lambda x: str(x))
        
        if 'OneHotEncoder' in encoder_str:
            self.encoder.fit(fitted_series.to_frame())
            try:
                self.transformed_names = list(map(lambda x: str(self.fitted_object.data['name']) + str(x)[2:], self.encoder.get_feature_names_out()))
            except:
                self.transformed_names = list(map(lambda x: str(self.fitted_object.data['name']) + str(x)[2:], self.encoder.get_feature_names()))
        elif 'LabelEncoder' in encoder_str:
            self.encoder.fit(fitted_series)
            self.transformed_names = [self.fitted_object.data['name']]
        elif 'OrdinalEncoder' in encoder_str:
            self.encoder.fit(fitted_series.to_frame())
            self.transformed_names = [self.fitted_object.data['name']]

        self.name = self.fitted_object.data['name']

    def transform(self, x, fitted_format=False, apply_name=False, ndim=None, kind=None):
        encoder_str = str(self.encoder)
        transformed_info = self.DataHandler.data_info(x, save_data=False)
        transformed_object = self.DataHandler.vector_info_split(x)
        transformed_series = pd.Series(**transformed_object.data)
        # transformed_series = pd.Series(**transformed_object.data).apply(lambda x: str(x))
        if 'OneHotEncoder' in encoder_str or 'OrdinalEncoder' in encoder_str:
            transformed_data = self.encoder.transform(transformed_series.to_frame())
        elif 'LabelEncoder' in encoder_str:
            transformed_data = self.encoder.transform(transformed_series)
        
        # name
        if (fitted_format is True) or (apply_name is True):
            apply_name = self.transformed_names
        elif apply_name is False:
            if 'OneHotEncoder' in encoder_str:
                try:
                    apply_name = list(map(lambda x: str(transformed_object.data['name']) + str(x)[2:], self.encoder.get_feature_names_out()))
                except:
                    apply_name = list(map(lambda x: str(transformed_object.data['name']) + str(x)[2:], self.encoder.get_feature_names()))
            else:
                apply_name = [transformed_object.data['name']]
        else:
            if 'OneHotEncoder' in encoder_str:
                try:
                    apply_name = list(map(lambda x: str(apply_name) + str(x)[2:], self.encoder.get_feature_names_out()))
                except:
                    apply_name = list(map(lambda x: str(apply_name) + str(x)[2:], self.encoder.get_feature_names()))
            else:
                apply_name = apply_name

        # transform
        if 'OneHotEncoder' not in encoder_str:
            apply_name = apply_name[0]
            if fitted_format:
                return self.DataHandler.transform(transformed_data, apply_instance=self.fitted_object, apply_columns=apply_name,
                            apply_ndim=self.fitted_ndim, apply_index=transformed_object.data['index'], apply_dtypes='int')
            else:
                apply_kind = True if kind is None else kind
                apply_ndim = transformed_info.ndim if ndim is None else ndim
                return self.DataHandler.transform(transformed_data, apply_instance=transformed_object, apply_columns=True,
                    apply_kind=apply_kind, apply_ndim=apply_ndim, apply_dtypes='int')
        else:       # OneHotEncoder
            parmas = self.encoder.get_params()
            if parmas['sparse']:
                transformed_data = transformed_data.toarray()
            
            if fitted_format:
                # apply_kind = self.fitted_object.kind if kind is None else kind
                apply_ndim = self.fitted_ndim if transformed_data.shape[1] == 1 else 2
                return self.DataHandler.transform(transformed_data, apply_kind=self.fitted_object.kind, apply_columns=apply_name,
                    apply_ndim=apply_ndim, apply_index=transformed_object.data['index'], apply_dtypes='int')
            else:
                apply_kind = transformed_object.kind if kind is None else kind
                apply_ndim = (transformed_info.ndim if ndim is None else ndim) if transformed_data.shape[1] == 1 else 2

                return self.DataHandler.transform(transformed_data, apply_kind=apply_kind, apply_columns=apply_name,
                    apply_ndim=apply_ndim, apply_index=transformed_object.data['index'], apply_dtypes='int')

    def fit_transform(self, x, ndim=None, kind=None):
        self.fit(x)
        fitted_format = True if ndim is None and kind is None else False
        return self.transform(x, fitted_format=fitted_format, apply_name=True, ndim=ndim, kind=kind)

    def inverse_transform(self, x, fitted_format=True, apply_name=False, ndim=None, kind=None, dtypes=None):
        encoder_str = str(self.encoder)
        inversed_info = self.DataHandler.data_info(x, save_data=False)
        inversed_object = self.DataHandler.data_info_split(x, ndim=2)
        inversed_data = self.encoder.inverse_transform(inversed_object.data['data'])

        # name
        if (fitted_format is True) or (apply_name is True):
            apply_name = self.fitted_object.data['name']
        elif apply_name is False:
            if 'OneHotEncoder' in encoder_str:
                apply_name = list(map(lambda x: x[:x.rfind('_')], inversed_object.data['columns']))[0]
            else:
                apply_name = inversed_object.data['columns']
        else:
            apply_name = apply_name if type(apply_name) == list else [apply_name]

        if fitted_format:
            apply_dtypes = dtypes if dtypes is not None else True
            return self.DataHandler.transform(inversed_data, apply_instance=self.fitted_object, apply_columns=apply_name,
                apply_ndim=self.fitted_ndim, apply_index=inversed_object.data['index'], apply_dtypes=apply_dtypes)
        else:
            apply_kind = True if kind is None else kind
            apply_ndim = inversed_info.ndim if ndim is None else ndim
            apply_dtypes = dtypes if dtypes is not None else False
            
            return self.DataHandler.transform(inversed_data, apply_instance=inversed_object, apply_columns=apply_name,
                apply_kind=apply_kind, apply_ndim=apply_ndim, 
                apply_dtypes=apply_dtypes, apply_index=True)

    def get_params(self):
        return self.encoder.get_params()

    def get_feature_names_out(self):
        return np.array(self.transformed_names)

    def __repr__(self):
        return f"(EncoderInstance) {self.name}: {self.encoder}"




### ★★★ ###
# Matrix/Frame/DataFrame 데이터에 Scaler 또는 Encoder를 적용하는 Class
class ScalerEncoder:
    """
    【required (Library)】 numpy, pandas, sklearn.preprocessing.*, copy.deepcopy, functools.reduce
    【required (Function)】DataHandler, class_object_execution, ScalerVector, EncoderVector, dtypes_split

    < Input >
     . encoder : dictionay type {'columns' : Scaler/Encoder Object or String, ...}
                      (default) {'#numeric' : 'StandardScaler', '#object':'OneHotEncoder', '#time', 'StandardScaler'}
                * required Scaler/Encoder: 'str type Name_of_Class', 'str type Name_of_Instance', 'Class object', 'instance object'
     . X : 1dim, 2dim vector or matrix

    < Method >
     . fit
     . transform
     . fit_transform
     . inverse_transform
    """
    def __init__(self, encoder=None, **kwargs):
        self.apply_encoder = {'#numeric':'StandardScaler', '#object':'OneHotEncoder', '#time': 'StandardScaler'}
        if encoder is not None:
            self.apply_encoder.update(encoder)

        self.DataHandler = DataHandler()
        self.kwargs = kwargs
        self.encoder = {}
        self.match_columns = {}

    def fit(self, X):
        self.encoder = {}

        fitted_info = self.DataHandler.data_info(X, save_data=False)
        self.fitted_ndim = fitted_info.ndim
        self.fitted_object = self.DataHandler.data_info_split(X, ndim=2)

        fitted_DataFrame = pd.DataFrame(**self.fitted_object.data).astype(self.fitted_object.dtypes)
        
        self.columns_dtypes = pd.DataFrame(dtypes_split(fitted_DataFrame, return_type='columns_all')).T

        for c in fitted_DataFrame:
            if c in self.apply_encoder.keys():
                if 'scaler' in str(self.apply_encoder[c]).lower():
                    se = ScalerVector(scaler=self.apply_encoder[c])
                elif 'encoder' in str(self.apply_encoder[c]).lower():
                    se = EncoderVector(encoder=self.apply_encoder[c])
            else:
                apply_se = self.apply_encoder['#' + self.columns_dtypes.loc[c, 'dtype_group']]
                if 'scaler' in str(apply_se).lower():
                    se = ScalerVector(scaler=apply_se)
                elif 'encoder' in str(apply_se).lower():
                    se = EncoderVector(encoder=apply_se)
            se.fit(fitted_DataFrame[c])
            self.encoder[c] = copy.deepcopy(se)
            self.match_columns[c] = se.transformed_names

    def transform(self, X, fitted_format=False, columns=None, ndim=None, kind=None):
        transformed_info = self.DataHandler.data_info(X, save_data=False)
        transformed_object = self.DataHandler.data_info_split(X, ndim=2)
        
        X_DataFrame = pd.DataFrame(**transformed_object.data).astype(transformed_object.dtypes)
        if transformed_object.kind != 'pandas':
            X_DataFrame.columns = self.encoder.keys() if columns is None else columns
        
        # transform ***
        transformed_DataFrame = pd.DataFrame()
        for c in X_DataFrame:
            transformed_columnvector = pd.DataFrame(self.encoder[c].transform(X_DataFrame[c], fitted_format=True))
            transformed_DataFrame = pd.concat([transformed_DataFrame, transformed_columnvector], axis=1)
        
        # return ***
        if fitted_format:
            apply_ndim = self.fitted_ndim if transformed_DataFrame.shape[1] == 1 else 2
            return self.DataHandler.transform(transformed_DataFrame, apply_kind=self.fitted_object.kind, apply_ndim=apply_ndim)
        else:
            apply_kind = transformed_object.kind if kind is None else kind
            apply_ndim = (transformed_info.ndim if ndim is None else ndim) if transformed_DataFrame.shape[1] == 1 else 2
            return self.DataHandler.transform(transformed_DataFrame, apply_kind=apply_kind, apply_ndim=apply_ndim)
        
    def fit_transform(self, X, ndim=None, kind=None):
        self.fit(X)
        fitted_format = True if ndim is None and kind is None else False
        return self.transform(X, fitted_format=fitted_format, ndim=ndim, kind=kind)

    def inverse_transform(self, X, fitted_format=False, columns=None, ndim=None, kind=None, dtypes=None):
        inversed_info = self.DataHandler.data_info(X, save_data=False)
        inversed_object = self.DataHandler.data_info_split(X, ndim=2)
        
        X_DataFrame = pd.DataFrame(**inversed_object.data).astype(inversed_object.dtypes)
        if inversed_object.kind != 'pandas':
            X_DataFrame.columns = reduce(lambda x,y : x + y, self.match_columns.values()) if columns is None else columns

        Xcolumns = copy.deepcopy(X_DataFrame.columns)
        match_columns = copy.deepcopy(self.match_columns)

        # inverse_transform ***
        inversed_DataFrame = pd.DataFrame()
        while bool(len(Xcolumns)):
            inversed_target = pd.DataFrame()

            c = Xcolumns[0]
            Xcolumns = Xcolumns.drop(c)
            # print(c, Xcolumns)
            if type(dtypes) == dict:
                if c in dtypes:
                    apply_dtypes = dtypes[c]
                else:
                    apply_dtypes = None
            else:       # bool, str, dtype
                apply_dtypes = dtypes

            for fc, tc in match_columns.items():
                if c in tc:
                    inversed_target = X_DataFrame[tc]

                    del match_columns[fc]
                    tc.remove(c)
                    Xcolumns = Xcolumns.drop(tc)
                    
                    inversed_data = self.encoder[fc].inverse_transform(inversed_target, fitted_format=fitted_format, ndim=2, dtypes=apply_dtypes)
                    inversed_data.columns = [self.encoder[fc].name]

                    inversed_DataFrame = pd.concat([inversed_DataFrame, inversed_data], axis=1)
                    break
            
            if c == X_DataFrame.columns[-1]:
                break
            
        # return ***
        if fitted_format:
            apply_ndim = self.fitted_ndim if inversed_DataFrame.shape[1] == 1 else 2
            # apply_dtypes = dict(filter(lambda x: x[0] in X_DataFrame.columns, self.fitted_object.dtypes.items()))
            apply_dtypes = dict(filter(lambda x: x[0] in inversed_DataFrame.columns, self.fitted_object.dtypes.items()))
            # print(inversed_DataFrame, self.fitted_object.kind, apply_ndim, apply_dtypes)
            return self.DataHandler.transform(inversed_DataFrame, apply_kind=self.fitted_object.kind, 
                    apply_ndim=apply_ndim, apply_dtypes=apply_dtypes)
        else:
            apply_kind = inversed_object.kind if kind is None else kind
            apply_ndim = (inversed_info.ndim if ndim is None else ndim) if inversed_DataFrame.shape[1] == 1 else 2
            return self.DataHandler.transform(inversed_DataFrame, apply_kind=apply_kind, apply_ndim=apply_ndim, apply_dtypes=dtypes)

    def __repr__(self):
        return f"(ScalerEncoder) {self.encoder}"
















# class DataSet():
#     """
#     【required (Library)】copy, functools, numpy(np), pandas(pd), torch, tensorflow(tf)
    
#     < Attribute >
#       self.inputdata_info
#       self.dataloader_info
#       self.dataloader
      
#     < Method >
#       . self.Dataset
#       . self.Split
#       . self.Encoding (self.Decoding)
#       . self.Batch
#       . self.Reset_dataloader
      
#     < Funtion >
#       . self.make_data_info
#       . self.dataloader_to_info
#       . self.info_to_dataloader
#       . self.data_transform_from_numpy
#       . self.split_size
#       . self.data_slicing
#       . self.data_split
#       . self.data_transform
#       . self.make_batch
      
    
#     """
#     def __init__(self, X=None, y=None,  X_columns=None, y_columns=None, kwargs_columns={},
#                 type = ['pandas', 'numpy', 'tensorflow', 'torch'], set_type={},
#                 X_encoder=None, y_encoder=None, encoder={},
#                 shuffle=False, random_state=None, **kwargs):
        
#         # all input value save
#         local_values = locals().copy()
        
#         dataloader_info = {}
#         # dataloader = {}
#         self.data_columns_couple = {None: None, 'X':X_columns, 'y': y_columns}
        
#         for arg_name, arg_value in local_values.items():
#             if arg_name not in ['self', 'random_state', 'kwargs']:
#                 exec(f"self.{arg_name} = arg_value")
                
#                 if arg_name in ['X', 'y']:
#                     if arg_value is not None and arg_value.shape[0] > 0:
#                         dataloader_info[arg_name] = {}
#                         dataloader_info[arg_name]['data'] = self.make_data_info(arg_value, prefix=arg_name, columns=self.data_columns_couple[arg_name], return_data=True) if arg_value is not None else None
                    
#         for kwarg_name, kwarg_value in kwargs.items():
#             if kwarg_value is not None and kwarg_value.shape[0] > 0:
#                 kwargs_column = kwargs_columns[kwarg_name] if kwarg_name in kwargs_columns.keys() else None
#                 dataloader_info[kwarg_name] = {}
#                 dataloader_info[kwarg_name]['data'] = self.make_data_info(kwarg_value, prefix=kwarg_name, columns=kwargs_column, return_data=True) if kwarg_value is not None else None
        
#         # self.local_values = local_values
#         self.set_type = set_type
#         # self.inputdata_info = dataloader_info
        
#         initialize_object = self.info_to_dataloader(dataloader_info, set_type=set_type)
#         self.dataloader = initialize_object['dataloader']
#         self.dataloader_info = initialize_object['data_info']
#         self.inputdata_info = copy.deepcopy(self.dataloader_info)
                
#         self.random_state = random_state
#         self.define_random_generate(random_state)
#         self.indices = {}
#         self.length = {}
        
#         self.dataloader_process = ''

#     # [funcation] ---------------------------
#     def define_random_generate(self, random_state=None):
#         self.random_generate = np.random.RandomState(self.random_state) if random_state is None else np.random.RandomState(random_state)
#         # self.random_generate = np.random.default_rng(random_state)

#     def make_data_info(self, data, prefix=None, columns=None, iloc_index=None, loc_index=None, dtype=None, return_data=False):
#         prefix = 'v' if prefix is None else prefix
#         type_of_data = str(type(data))
        
#         array_data = np.array(data)
#         shape = array_data.shape
#         n_dim = len(shape)
        
#         # type
#         result_of_type = None
#         if 'pandas' in type_of_data:
#             result_of_type = 'pandas'
#         elif 'numpy' in type_of_data:
#             result_of_type = 'numpy'
#         elif 'tensorflow' in type_of_data:
#             result_of_type = 'tensorflow'
#             # result_of_type =  'tensorflow - variable' if 'variable' in type_of_data.lower() else 'tensorflow - constant'      
#         elif 'torch' in type_of_data:
#             result_of_type = 'torch'
#         else:
#             result_of_type = 'else'
        
#         # dtype
#         if dtype is None:
#             if  result_of_type == 'pandas':
#                 dtype = data.dtype if n_dim == 1 else data.dtypes.to_dict()
#             else:
#                 dtype = data.dtype
            
            
#         # columns
#         if columns is not None:
#             set_columns = columns
#         elif result_of_type == 'pandas':
#             set_columns = str(data.name) if n_dim == 1 else np.array(data.columns)
#         else:
#             if n_dim == 1:
#                 set_columns = prefix
#             elif n_dim > 1:
#                 col_base = np.array(range(shape[-1])).astype('str')
#                 set_columns = np.tile(col_base, np.array(shape[:-2]).prod()).reshape(*list(shape[:-2]), shape[-1]).astype('str')
#                 set_columns = np.char.add(prefix, set_columns)
                
#         # index
#         if iloc_index is None and loc_index is None:
#             if result_of_type == 'pandas':
#                 loc_index = np.array(data.index)
#                 iloc_index = np.array(range(shape[0]))
#             else:
#                 loc_index = np.array(range(shape[0]))
#                 iloc_index = np.array(range(shape[0]))
#         else:
#             if result_of_type == 'pandas':
#                 loc_index = np.array(data.index) if loc_index is None else np.array(loc_index)
#                 iloc_index = (np.array(range(shape[0])) if loc_index is None else loc_index) if iloc_index is None else np.array(iloc_index)
#             else:
#                 loc_index = (np.array(range(shape[0])) if iloc_index is None else iloc_index) if loc_index is None else np.array(loc_index)
#                 iloc_index = (np.array(range(shape[0])) if loc_index is None else loc_index) if iloc_index is None else np.array(iloc_index)
        
#         if return_data:
#             return {'type': result_of_type, 'ndim': n_dim, 'dtype': dtype, 'columns': set_columns, 'loc_index' :loc_index, 'iloc_index':iloc_index, 'data': array_data}
#         else:
#             return {'type': result_of_type, 'ndim': n_dim, 'dtype': dtype, 'columns': set_columns, 'loc_index' :loc_index, 'iloc_index':iloc_index}

#     def dataloader_to_info(self, dataloader, data_info=None, update_loader=False):
#         return_data_info = {}
#         for name, dataset in dataloader.items():
#             return_data_info[name] = {}
#             for dataset_name, data in dataset.items():
#                 if data_info is None:
#                     return_data_info[name][dataset_name] = self.make_data_info(data=data)
#                 else:
#                     info = data_info[name][dataset_name]
#                     return_data_info[name][dataset_name] = self.make_data_info(data=data, columns=info['columns'], iloc_index=info['iloc_index'], loc_index=info['loc_index'], return_data=True)
#         if update_loader is True:
#             return_dataloader = self.info_to_dataloader(return_data_info, update_info=False)
#             return {'dataloader': return_dataloader['dataloader'], 'data_info':return_data_info}
#         else:
#             return {'dataloader': dataloader, 'data_info':return_data_info}

#     def info_to_dataloader(self, data_info, set_type={}, update_info=False):
#         return_dataloader = {}
#         self.set_type.update(set_type)
        
#         for name, dataset in data_info.items():
#             return_dataloader[name] = {}
#             for dataset_name, info in dataset.items():
#                 if self.set_type is None:
#                     apply_type = info['type']
#                 elif type(self.set_type) == str:
#                     apply_type = self.set_type
#                 elif type(self.set_type) == dict:
#                     apply_type = self.set_type[name] if name in self.set_type.keys() else info['type']
#                 return_dataloader[name][dataset_name] = self.data_transform_from_numpy(numpy_data=info['data'], set_type=apply_type,
#                                                         index=info['loc_index'], columns=info['columns'], dtype=info['dtype'])
#         if len(self.set_type) > 0 or update_info is True:
#             return_data_info = self.dataloader_to_info(return_dataloader, data_info=data_info, update_loader=False)
#             return {'dataloader': return_dataloader, 'data_info':return_data_info['data_info']}
#         else:
#             return {'dataloader':return_dataloader, 'data_info':data_info}

#     def data_transform_from_numpy(self, numpy_data, set_type, index=None, columns=None, dtype=None):
#         shape = numpy_data.shape
#         ndim = len(shape)
        
#         # dtype
#         if dtype is not None:
#             if type(dtype) == dict:     # pandas
#                 if set_type == 'pandas':
#                     apply_dtype = dtype
#                 else:
#                     apply_dtype = np.dtype('float32')
#             elif 'torch' in str(dtype):     # torch
#                 apply_dtype = np.dtype(str(dtype).split('.')[-1])
#             elif '<dtype:' in str(dtype):   # tensorflow
#                 apply_dtype = np.dtype(dtype.as_numpy_dtype)
#             else:
#                 apply_dtype = dtype
#         else:
#             apply_dtype = dtype
                
#         # set_type transform
#         if set_type == 'numpy':
#             return numpy_data
#         elif set_type == 'pandas':
#             if ndim == 1:
#                 return pd.Series(numpy_data, index=index, name=columns, dtype=apply_dtype)
#             elif ndim == 2:
#                 if apply_dtype is None:
#                     return pd.DataFrame(numpy_data, index=index, columns=columns)
#                 else:
#                     if type(apply_dtype) != dict:
#                         if columns is None:
#                             apply_dtype = {c: apply_dtype for c in range(numpy_data.shape[1])}
#                         else:
#                             apply_dtype = {c: apply_dtype for c in columns}
#                     return pd.DataFrame(numpy_data, index=index, columns=columns).astype(apply_dtype)
#         elif set_type == 'tensorflow':
#             return tf.constant(numpy_data, dtype=apply_dtype)
#         elif set_type == 'torch':
#             return torch.FloatTensor(numpy_data)
#             # return torch.tensor(numpy_data, dtype=eval(f'torch.{str(apply_dtype)}'))

#     def split_size(self, valid_size, test_size):
#         self.valid_size = valid_size
#         self.test_size = test_size
        
#         self.train_size = 1 - test_size
#         self.train_train_size = self.train_size - valid_size
#         self.train_valid_size = valid_size

#     def generate_index(self, data, data_info=None, valid_size=None, test_size=None, shuffle=True,
#                       random_state=None):
#         try:
#             data_length = len(data)
#         except:
#             data_length = data.shape[0]
#         if data_info is None:
#             data_info = self.make_data_info(data=data)
            
#         index = data_info['iloc_index']
#         indices = {}

#         # shuffle
#         if shuffle is True:
#             random_generate = self.random_generate if random_state is None else np.random.RandomState(random_state)
#             apply_index = random_generate.permutation(index)
#         else:
#             apply_index = index
        
#         # split_size
#         self.split_size(valid_size = 0.0 if valid_size is None else valid_size,
#                         test_size = 0.3 if test_size is None else test_size)
        
#         # train_valid_test split
#         train_len = int(data_length * (1-self.test_size))
#         train_train_len = int(train_len * (1-self.train_valid_size))
#         train_valid_len = train_len - train_train_len
#         test_len = data_length - train_len
        
#         for k, v in zip(['data','train', 'train_train', 'train_valid', 'test'], [train_len, train_train_len, train_valid_len, test_len]):
#             if v > 0:
#                 self.length[k] = v
        
#         # save
#         indices['all_index'] = index
#         indices['apply_index'] = apply_index
#         indices['train_index'] = apply_index[:train_len]
#         indices['train_train_index'] = indices['train_index'][:train_train_len]
#         if train_valid_len > 0:
#             indices['train_valid_index'] = indices['train_index'][train_train_len:]
#         indices['test_index'] = apply_index[train_len:]
        
#         return indices

#     def data_slicing(self, data=None, apply_index=None, data_info=None, index_type='iloc'):
#         # data_info
#         if data_info is None:
#             data_info = self.make_data_info(data=data, return_data=True)

#         numpy_data = data_info['data']
        
#         if index_type == 'iloc':
#             index_series = pd.Series(data_info['loc_index'], index=range(len(numpy_data)) )
#             iloc_index = np.array(index_series[apply_index].index) 
#             loc_index = np.array(index_series[apply_index].values)
#         elif index_type == 'loc':
#             index_series = pd.Series(range(len(numpy_data)), index=data_info['loc_index'])
#             iloc_index = np.array(index_series[apply_index].values) 
#             loc_index = np.array(index_series[apply_index].index)
        
#         # slicing
#         sliced_data = np.take(numpy_data, iloc_index, axis=0)
#         # sliced_index = np.take(data_index, apply_index, axis=0)
#         return self.data_transform_from_numpy(numpy_data=sliced_data, index=loc_index, 
#                                             set_type=data_info['type'], columns=data_info['columns'], dtype=data_info['dtype'])       

#     def data_split(self, data, data_info=None, indices=None, valid_size=0, test_size=0, shuffle=True,
#                    index_type='iloc', random_state=None, verbose=0):
#         # data_info
#         if data_info is None:
#             data_info = self.make_data_info(data=data, return_data=True)
            
#         if indices is None:
#             if len(self.indices) == 0:
#                 indices = self.generate_index(data=data, valid_size=valid_size, test_size=test_size, 
#                                               shuffle=shuffle, random_state=random_state)
#             else:
#                 indices = self.indices
        
#         split_data = {}
#         for key, index in indices.items():
#             split_data[key] = self.data_slicing(data=data, apply_index=index, data_info=data_info, index_type=index_type)
#             if verbose > 0:
#                 print(f"{key}: {split_data[key].shape}")
            
#         return {'split_data': split_data, 'indices':indices}

#     def data_transform(self, data, data_info=None, encoder=None, type='encoding'):
#         if encoder is None:
#             return data
#         else:
#             try:
#                 if 'enco' in type:
#                     return encoder.transform(data)
#                 elif 'deco' in type:
#                     return encoder.inverse_transform(data)
#             except:
#                 # data_info
#                 if data_info is None:
#                     data_info = self.make_data_info(data=data, return_data=True)
#                 if 'enco' in type:
#                     transformed_np_data = encoder.transform(data_info['data'])
#                 elif 'deco' in type:
#                     transformed_np_data = encoder.inverse_transform(data_info['data'])
#                 transformed_data = self.data_transform_from_numpy(numpy_data=transformed_np_data, 
#                                                set_type=data_info['type'], columns=data_info['columns'], dtype=None)
#                 return transformed_data

#     def make_batch(self, data, batch_size=None):
#         try:
#             data_length = len(data)
#         except:
#             data_length = data.shape[0]
        
#         batch_size = data_length if batch_size is None or batch_size <= 0 or batch_size > data_length else batch_size
#         batch_index = 0
#         batch = []
#         while True:
#             if batch_index + batch_size >= data_length:
#                 batch.append(data[batch_index:])
#                 batch_index = data_length
#                 # batch = np.array(batch)
#                 break
#             else:
#                 batch.append(data[batch_index:batch_index + batch_size])
#                 batch_index = batch_index + batch_size
#         return batch

#     # [method] ---------------------------
#     def Reset_dataloader(self):
#         self.dataloader_info = copy.deepcopy(self.inputdata_info)
#         self.dataloader = self.info_to_dataloader(self.dataloader_info)['dataloader']
#         self.dataloader_process = ''

#     def Dataset(self, kwargs_columns={}, set_type={}, **kwargs):
#         for kwarg_name, kwarg_value in kwargs.items():
#             if kwarg_value is not None and kwarg_value.shape[0] > 0:
#                 kwargs_column = kwargs_columns[kwarg_name] if kwarg_name in kwargs_columns.keys() else None
#                 self.dataloader_info[kwarg_name] = {}
#                 self.dataloader_info[kwarg_name]['data'] = self.make_data_info(kwarg_value, prefix=kwarg_name, columns=kwargs_column, return_data=True) if kwarg_value is not None else None
#                 self.inputdata_info[kwarg_name] = {}
#                 self.inputdata_info[kwarg_name] = self.dataloader_info[kwarg_name]['data']
#         # set dataset / data_info
#         update_object = self.info_to_dataloader(dataloader_info, set_type=set_type)
#         self.dataloader = self.info_to_dataloader(self.dataloader_info, set_type=set_type)['dataloader']
#         self.dataloader_info = update_object['data_info']
#         self.inputdata_info = copy.deepcopy(self.dataloader_info)
#         return self

#     def Encoding(self, X_encoder=None, y_encoder=None, encoder={}):
#         # encoder dictionary
#         self.encoder.update(encoder)
#         X_encoder = self.X_encoder if X_encoder is None else X_encoder
#         y_encoder = self.y_encoder if y_encoder is None else y_encoder
#         if X_encoder is not None:
#             encoder['X'] = X_encoder
#         if y_encoder is not None:
#             encoder['y'] = y_encoder
        
#         # encoding
#         encoder_keys = encoder.keys()
#         for name, data_dict in self.dataloader.items():
#             for dataset_name, data in data_dict.items():
#                 if name in encoder_keys and encoder[name] is not None:
#                     data_info = self.dataloader_info[name][dataset_name]
#                     encodered_data = self.data_transform(data=data, data_info=data_info, encoder=encoder[name], type='encoding')
#                     self.dataloader[name][dataset_name] = encodered_data
#                     self.dataloader_info[name][dataset_name] = self.make_data_info(data=encodered_data, columns=data_info['columns'], 
#                                                                                     iloc_index= data_info['iloc_index'],
#                                                                                     return_data=True)
#         # __repr__
#         self.dataloader_process = self.dataloader_process + ('\n . ' if self.dataloader_process == '' else ' > ') + 'Encoding'

#     def Decoding(self, X_encoder=None, y_encoder=None, encoder={}):
#         # encoder dictionary
#         self.encoder.update(encoder)
#         X_encoder = self.X_encoder if X_encoder is None else X_encoder
#         y_encoder = self.y_encoder if y_encoder is None else y_encoder
#         if X_encoder is not None:
#             encoder['X'] = X_encoder
#         if y_encoder is not None:
#             encoder['y'] = y_encoder
        
#         # decoding
#         encoder_keys = encoder.keys()
#         for name, data_dict in self.dataloader.items():
#             for dataset_name, data in data_dict.items():
#                 if name in encoder_keys and encoder[name] is not None:
#                     data_info = self.dataloader_info[name][dataset_name]
#                     encodered_data = self.data_transform(data=data, data_info=data_info, encoder=encoder[name], type='decoding')
#                     self.dataloader[name][dataset_name] = encodered_data
#                     self.dataloader_info[name][dataset_name] = self.make_data_info(data=encodered_data, columns=data_info['columns'], 
#                                                                                     iloc_index= data_info['iloc_index'],
#                                                                                     dtype=self.inputdata_info[name]['data']['dtype'],
#                                                                                     return_data=True)
#         # __repr__
#         self.dataloader_process = self.dataloader_process + ('\n . ' if self.dataloader_process == '' else ' > ') + 'Decoding'

#     def Split(self, valid_size=0, test_size=0.3, shuffle=True, random_state=None):
#         random_state = self.random_state if random_state is None else random_state
        
#         # train_valid_test_split
#         for name, data_dict in self.dataloader.items():
#             data_info = self.dataloader_info[name]['data']
#             splited_dict = self.data_split(data=data_dict['data'], data_info=data_info,
#                             valid_size=valid_size, test_size=test_size, shuffle=shuffle, 
#                             index_type='iloc', random_state=self.random_state)
#             del splited_dict['split_data']['all_index']
#             del splited_dict['split_data']['apply_index']
#             del splited_dict['indices']['all_index']
#             del splited_dict['indices']['apply_index']
            
#             for dataset_name, index in splited_dict['indices'].items():
#                 data = splited_dict['split_data'][dataset_name]
#                 set_name = dataset_name.replace('index','set')
#                 self.dataloader[name][set_name] = data
#                 self.dataloader_info[name][set_name] = self.make_data_info(data=data,
#                                                                    columns=data_info['columns'],
#                                                                    iloc_index=index,
#                                                                    loc_index=np.take(data_info['loc_index'], index),
#                                                                    dtype=data_info['dtype'],
#                                                                    return_data=True)
#         # __repr__
#         self.dataloader_process = self.dataloader_process + ('\n . ' if self.dataloader_process == '' else ' > ') + 'Split'

#     def Batch(self, batch_size=None, shuffle=True, random_state=None):
#         if shuffle is True:
#             random_generate = self.random_generate if random_state is None else np.random.RandomState(random_state)
        
#         # transform to batch
#         for name, data_dict in self.dataloader.items():
#             for dataset_name, data in data_dict.items():
#                 batch_data = []
#                 data_info = self.dataloader_info[name][dataset_name]
#                 dataset_index = data_info['loc_index']
#                 if shuffle is True:
#                     dataset_index = random_generate.permutation(dataset_index)
#                 batch_indices = self.make_batch(dataset_index, batch_size=batch_size)
                
#                 for batch_index in batch_indices:
#                     batch_sliced_data = self.data_slicing(data, data_info=data_info, apply_index=batch_index, index_type='loc')
#                     batch_data.append( batch_sliced_data )
                
#                 # update dataloader
#                 self.dataloader[name][dataset_name] = batch_data
                
#                 # update dataloader_info
#                 self.dataloader_info[name][dataset_name]['batch_iloc_index'] = batch_indices
#         # __repr__
#         self.dataloader_process = self.dataloader_process + ('\n . ' if self.dataloader_process == '' else ' > ') + f'Batch({batch_size})'

#     def __repr__(self):
#         data_set_names = ', '.join(list(self.dataloader.keys()))
#         return f'<class: DataLoader Object ({data_set_names})>{self.dataloader_process}'



# a0 = np.arange(60).reshape(-1,4)
# a1 = np.arange(120).reshape(-1,4)
# a1 = np.arange(60).reshape(-1,4)
# a2 = np.array(range(20)).ravel()
# a3 = a1.reshape(6,-1,5)

# a5 = pd.DataFrame(a1, columns=['A','CC','D','K']).sample(len(a1))
# a7 = pd.DataFrame(a0, columns=['A','CC','D','K'])
# a6 = pd.Series(a2).sample(len(a2))

# a10 = torch.FloatTensor(a1)
# a11 = tf.constant(a1, dtype='float')
# a12 = tf.Variable(a1, dtype='float')
# a13 = tf.Variable(a2, dtype='float')



# ds = DataSet(X=a1, y=a2, z=a7, w=a7.sample(12), 
#              kk=a7.sample(15), kwa=a7.sample(15), kwb=a7.sample(15), 
#              set_type={'kk':'torch', 'kwa':'numpy', 'kwb':'tensorflow'})
# ds.inputdata_info
# ds.dataloader_info['X']['train_valid_set']
# ds.dataloader

# ds.Split(valid_size=0.1)

# se = ScalerEncoder()
# se.fit(a1)
# se2 = ScalerEncoder()
# se2.fit(a7)

# ds.Encoding(encoder={'X':se, 'z':se2})
# ds.Decoding(encoder={'X':se, 'z':se2})
# ds.Batch(batch_size=7)

# ds.Reset_dataloader()

################################################################################################







################################################################################################################

# # HyperParameter Tunning
# from bayes_opt import BayesianOptimization, UtilityFunction

# # (git) bayes_opt : https://github.com/fmfn/BayesianOptimization 
# # (git_advance) bayes_opt : https://github.com/fmfn/BayesianOptimization/blob/master/examples/advanced-tour.ipynb
# # (install) conda install -c conda-forge bayesian-optimization


# # from bayes_opt import BayesianOptimization
# # from bayes_opt import UtilityFunction
# class BayesOpt:
#     """
#      【required (Library)】 bayes_opt.BayesianOptimization, bayes_opt.UtilityFunction
#      【required (Custom Module)】 EarlyStopping
     
#       . __init__(self, f, pbounds, random_state=None, verbose=2)
#          f : function
#          pbounds : {'x':(-150, 150), 'y':(-50, 100), 'z':(1000, 1200)}
#          random_state : 1, 2, 3...
#          verbose : 1, 2, 3... 
#     """
#     def __init__(self, f, pbounds, random_state=None, verbose=2):
#         self.verbose = verbose
#         self.f = f
#         self.pbounds = pbounds
#         self.random_state = random_state
#         self.random_generate = np.random.RandomState(self.random_state)
        
#         self.bayes_opt = BayesianOptimization(f=f, pbounds=pbounds, random_state=random_state, verbose=verbose)
#         self._space = self.bayes_opt._space
        
#         self.res = []
#         self.max = {'target':-np.inf, 'params':{}}
#         self.repr_max = {}
        
#         self.last_state = ''
    
#     def decimal(self, x, rev=0):
#         return 2 if x == 0 else int(-1*(np.floor(np.log10(abs(x)))-3-rev))
    
#     def auto_decimal(self, x, rev=0):
#         if np.isnan(x):
#             return np.nan
#         else:
#             decimals = self.decimal(x, rev=rev)
#             if decimals < 0:
#                 return x
#             else:
#                 return round(x, decimals)

#     def print_result(self):
#         epoch = len(self.bayes_opt._space.target)
#         last_target = self.auto_decimal(self.bayes_opt._space.target[-1])
#         last_params = {k: self.auto_decimal(v) for k, v in zip(self.bayes_opt._space.keys, self.bayes_opt._space.params[-1])}
#         last_state = '**Maximum' if epoch == np.argmax(self.bayes_opt._space.target) + 1 else self.last_state
        
#         if self.verbose > 0:
#             if self.verbose > 1 or last_state == '**Maximum':
#                 print(f"{epoch} epoch) target: {last_target}, params: {str(last_params)[:255]} {last_state}")
#         self.last_state = ''
    
#     def maximize(self, init_points=5, n_iter=25, acq='ucb', kappa=2.576, xi=0.0, patience=None, **gp_params):
#         if patience is not None:
#             bayes_utils = UtilityFunction(kind=acq, kappa=kappa, xi=xi)
#             n = 1
            
#             # init_points bayesian
#             for i in range(init_points):
#                 self.bayes_opt.probe(self.bayes_opt._space.random_sample(), lazy=False)
#                 self.print_result()
#                 n += 1
            
#             # EarlyStop
#             early_stop_instance = EarlyStopping(patience=patience, optimize='maximize')
#             early_stop_instance.early_stop(score=self.bayes_opt.max['target'], save=self.bayes_opt.max['params'])
            
#             last_state = 'break' if patience == 0 else None
#             while last_state != 'break' or n < n_iter:
#                 # Bayesian Step
#                 next_points = self.bayes_opt.suggest(bayes_utils)
#                 next_target = self.f(**next_points)
#                 self.bayes_opt.register(params=next_points, target=next_target)
            
#                 if n >= n_iter:
#                     last_state = early_stop_instance.early_stop(score=next_target, save=next_points)
#                     self.last_state = '' if last_state == 'None' else last_state

#                 self.print_result()
#                 n += 1
            
#         else:
#             self.bayes_opt.maximize(init_points=init_points, n_iter=n_iter, acq=acq, kappa=kappa, xi=xi, **gp_params)
        
#         # result            
#         target_auto_format = self.auto_decimal(self.bayes_opt.max['target'])
#         parmas_auto_format = {k: self.auto_decimal(v) for k, v in self.bayes_opt.max['params'].items()}
#         self.repr_max = {'target':target_auto_format, 'params': parmas_auto_format}

#         self.res = self.bayes_opt.res
#         self.max = self.bayes_opt.max

#     def __repr__(self):
#         if len(self.repr_max) > 0:
#             return f"(bayes_opt) BayesianOptimization: {self.repr_max}"
#         else:
#             return f"(bayes_opt) BayesianOptimization: undefined"
        



################################################################################################################
# print(get_python_lib())
# class EstimatorSearch:
#     """
#      【required (Library)】 bayes_opt.BayesianOptimization, bayes_opt.UtilityFunction
#      【required (Class)】 BayesOpt
#      【required (Function)】auto_formating

#     """
#     def __init__(self, estimator, train_X=None, train_y=None, valid_X=None, valid_y=None, 
#         params={}, params_dtypes={},
#         optim='bayes', optimizer_params={}, optimize_params={},
#         scoring=None, scoring_type='metrics', scoring_params={}, negative_scoring=False,
#         verbose=0
#         ):
#         self.estimator = estimator

#         self.train_X = train_X
#         self.train_y = train_y

#         self.valid_X = train_X if valid_X is None else valid_X
#         self.valid_y = train_y if valid_y is None else valid_y

#         self.params = params
#         self.params_dtypes = params_dtypes
#         if verbose>0 and len(params) > 0:
#             print(f"fixed parmas: {params}")

#         self.scoring = scoring
#         self.scoring_type = scoring_type
#         self.scoring_params = scoring_params
#         self.negative_scoring = negative_scoring

#         self.optim_method = optim
#         self.optim = None
#         self.optimizer_params = optimizer_params
#         self.optimize_params = optimize_params

#         self.verbose = verbose

#         self.best_estimator = None

#     def transform_dtype(self, dtype, x):
#         if 'class' in str(dtype).lower():
#             if 'int' in str(dtype).lower():
#                 return int(round(x,0))
#             else:
#                 return dtype(x)
#         elif type(dtype) == str:
#             if 'int' in dtype.lower():
#                 return int(round(x,0))
#             else:
#                 return eval(f"{dtype.lower()}({x})") 

#     def gap_between_pred_true(self, true_y, pred_y):
#         return np.sum((true_y - pred_y)**2) / len(true_y)

#     def __call__(self, **params):
#         if len(self.params_dtypes) > 0:
#             apply_params = {k: (self.transform_dtype(self.params_dtypes[k], v) if k in self.params_dtypes.keys() else v) for k, v in params.items()}
#         else:
#             apply_params = params
#         apply_params.update(self.params)
        
#         # print(apply_params)
#         model = self.estimator(**apply_params)
#         model.fit(self.train_X, self.train_y)
#         pred_y = model.predict(self.valid_X)
#         score_result = self.score(y_true=self.valid_y, y_pred=pred_y, X=self.valid_X, y=self.valid_y, estimator=model, **self.scoring_params)
        
#         if self.negative_scoring:
#             return -score_result
#         else:
#             return score_result
    
#     def optimizer(self, optim='bayes', verbose=None, **optimizer_params):
#         """
#          . bayesian_optimization : pbounds, random_state=None, verbose=2
#         """
#         verbose = self.verbose if verbose is None else verbose
#         if len(optimizer_params) == 0:
#             optimizer_params.update(self.optimizer_params)
        
#         def optim_params_setting(params_dict, name, init):
#             params_dict[name] = init if name not in optimizer_params.keys() else params_dict[name]
#             return params_dict

#         optim_method = self.optim_method if optim is None else optim
#         if 'bayes' in optim_method:
#             self.optim = BayesOpt(f=self.__call__, verbose=verbose, **optimizer_params)

#         return self

#     def optimize(self, **optimize_params):
#         """
#          . bayesian_optimization : init_points=5, n_iter=25, acq='ucb', kappa=2.576, xi=0.0, **gp_params
#         """
#         if len(optimize_params) == 0:
#             optimize_params.update(self.optimize_params)

#         if 'bayes' in self.optim_method:
#             self.optim.maximize(**optimize_params)
#             self.res = self.optim.res
#             self.opt = self.optim.max

#             if len(self.params_dtypes) > 0:
#                 self.res = [{'target': e['target'], 'params': {k: (self.transform_dtype(self.params_dtypes[k], v) if k in self.params_dtypes.keys() else v) for k, v in e['params'].items()}} for e in self.res]
#                 self.opt = {'target': self.opt['target'], 'params': {k: (self.transform_dtype(self.params_dtypes[k], v) if k in self.params_dtypes.keys() else v) for k, v in self.opt['params'].items()}}

#         if self.best_estimator is not None:
#             self.best_params = self.params.copy()
#             self.best_params.update(self.opt['params'])
#             self.best_estimator = self.estimator(**self.best_params)

#             train_X_overall = pd.concat([self.train_X, self.valid_X], axis=0)
#             train_y_overall = pd.concat([self.train_y, self.valid_y], axis=0)
#             self.best_estimator.fit(train_X_overall, train_y_overall)
#             print(f"(best_estimator is updated) result: {self.opt['target']}, best_params: {self.best_params}")

#     def fit(self, train_X=None, train_y=None, valid_X=None, valid_y=None, scoring=None, scoring_type=None, negative_scoring=None, optim=None,
#         verbose=1, optimizer_params={}, optimize_params={}, return_result=True):
#         self.train_X = self.train_X if train_X is None else train_X
#         self.train_y = self.train_y if train_y is None else train_y
#         self.valid_X = (self.train_X if self.valid_X is None else self.valid_X) if valid_X is None else valid_X
#         self.valid_y = (self.train_y if self.valid_y is None else self.valid_y) if valid_y is None else valid_y
#         self.scoring = self.scoring if scoring is None else scoring
#         self.scoring_type = self.scoring_type if scoring_type is None else scoring_type
#         self.negative_scoring = self.negative_scoring if negative_scoring is None else negative_scoring
#         optimizer_params = self.optimizer_params if len(optimizer_params) == 0 else optimizer_params
#         optimize_params = self.optimize_params if len(optimizer_params) == 0 else optimize_params


#         if optim is None:
#             if self.optim is None:
#                 optim_method = 'bayes' if self.optim_method is None else self.optim_method
#                 self.optimizer(optim=optim_method, **optimizer_params).optimize(**optimize_params)
#         else:
#             optim_method = optim
#             self.optimizer(optim=optim_method, **optimizer_params).optimize(**optimize_params)
        
#         self.best_params = self.params.copy()
#         self.best_params.update(self.opt['params'])

#         self.best_estimator = self.estimator(**self.best_params)

#         train_X_overall = pd.concat([self.train_X, self.valid_X], axis=0)
#         train_y_overall = pd.concat([self.train_y, self.valid_y], axis=0)
#         self.best_estimator.fit(train_X_overall, train_y_overall)

#         if verbose:
#             print(f"(Opimize) result: {self.opt['target']}, best_params: {self.best_params}")
#         if return_result:
#             return self.best_estimator

#     def score(self, y_true=None, y_pred=None, X=None, y=None, estimator=None, **scoring_params):
#         if 'metric' in self.scoring_type.lower() :
#             if self.scoring is None:
#                 self.scoring = self.gap_between_pred_true
#             result = self.scoring(y_true, y_pred)

#         elif 'cross_val' in self.scoring_type.lower():
#             result = np.mean(cross_val_score(estimator=estimator, X=X, y=y, scoring=self.scoring, **scoring_params))
#         return result




################################################################################################################
# ['Mo', 'Ba', 'Cr', 'Sr', 'Pb', 'B', 'Mg', 'Ca', 'K']
class FeatureInfluence():
    """
    【required (Library)】 numpy, pandas
    【required (Class)】DataHandler, Mode
    【required (Function)】class_object_execution, auto_formating, dtypes_split

    < Input >

    < Output >
    
    """
    def __init__(self, train_X=None, estimator=None, n_points=5, encoder=None, encoderX=None, encoderY=None, conditions={}, y_name='pred_y', confidential_interval=None):
        self.estimator=estimator
        self.train_X = train_X
        self.n_points = n_points
        self.conditions = conditions

        self.encoder = encoder
        self.encoderX = encoderX
        self.encoderY = encoderY
        self.y_name = y_name

        self.DataHandler = DataHandler()
        self.grid_X = None
        
        self.confidential_interval = confidential_interval
        
    # define train, grid data handler instance
    def define_train_grid_data(self, train_X=None, grid_X=None, conditions={}, n_points=None):
        # train_data
        if train_X is not False:
            if train_X is None:
                if self.train_X is None:
                    raise Exception('train_X must be required to predict')
                else:
                    train_X = self.train_X

            self.train_X_info = self.DataHandler.data_info(train_X, save_data=False)
            if self.train_X_info.kind == 'pandas':
                self.train_X_info_split = self.DataHandler.data_info_split(train_X)
            else:
                self.train_X_info_split = self.DataHandler.data_info_split(train_X, columns=list(range(self.train_X.shape[1])))
        
        # grid_data
        if grid_X is not False:
            if grid_X is None:
                if self.grid_X is None:
                    grid_X = self.make_grid(train_X=train_X, conditions=conditions, n_points=n_points, return_result=True, save_result=False)
                else:
                    grid_X = self.grid_X

            self.grid_X_info = self.DataHandler.data_info(grid_X, save_data=False)
            if self.grid_X_info.kind == 'pandas':
                self.grid_X_info_split = self.DataHandler.data_info_split(grid_X)
            else:
                self.grid_X_info_split = self.DataHandler.data_info_split(grid_X, columns=list(range(self.grid_X.shape[1])))

    # checking unpredictable
    def predictable_check(self, target_data=None, criteria_data=None):
        target_frame = self.DataHandler.transform(target_data, apply_kind='pandas')
        criteria_frame = self.DataHandler.transform(criteria_data, apply_kind='pandas')
        
        numeric_columns = dtypes_split(target_frame, return_type='columns_list')['numeric']
        target_numeric = target_frame[numeric_columns]
        criteria_numeric = criteria_frame[numeric_columns]

        criteria_min = criteria_numeric.min()
        criteria_max = criteria_numeric.max()

        if target_numeric.shape == (1,0):
            return pd.DataFrame(columns=['unpred', 'unpred_columns', 'lower_columns', 'upper_columns'])
        else:
            unpredictable = pd.Series((target_numeric < criteria_min).any(axis=1) | (target_numeric > criteria_max).any(axis=1))
            unpredictable_cols_lower = pd.DataFrame(target_numeric < criteria_min).apply(lambda x: [i for i, xc in zip(x.index, x) if xc==True] ,axis=1)
            unpredictable_cols_upper = pd.DataFrame(target_numeric > criteria_max).apply(lambda x: [i for i, xc in zip(x.index, x) if xc==True] ,axis=1)
            unpredictable_cols = unpredictable_cols_lower + unpredictable_cols_upper

            unpredictable_result = pd.concat([unpredictable, unpredictable_cols, unpredictable_cols_lower, unpredictable_cols_upper], axis=1)
            unpredictable_result.columns = ['unpred', 'unpred_columns', 'lower_columns', 'upper_columns']
            return unpredictable_result

    # generate grid_table *
    def make_grid(self, train_X=None, conditions={}, n_points=None, save_result=True, return_result=False):
        re_num = re.compile('\d')

        # define train instance
        self.define_train_grid_data(train_X=train_X, grid_X=False)
        
        if n_points is None:
            n_points = self.n_points

        X_analysis = pd.DataFrame(**self.train_X_info_split.data).astype(self.train_X_info_split.dtypes)

        X_mean = X_analysis.apply(lambda x: x.mean() if 'int' in str(x.dtype) or 'float' in str(x.dtype) else x.value_counts().index[0] ,axis=0)
        X_std = X_analysis.apply(lambda x: x.std() if 'int' in str(x.dtype) or 'float' in str(x.dtype) else np.nan, axis=0)
        X_dict = X_mean.to_dict()
        dtypes = self.train_X_info_split.dtypes

        for xc in conditions:
            el = conditions[xc]
            if 'int' in str(dtypes[xc]) or 'float' in str(dtypes[xc]):
                if type(el) == str:
                    if 'monte' in el.lower():
                        if '(' in el:
                            monte_str, X_mean_el = el.split('(')
                            X_mean_el = int(X_mean_el.replace(')',''))
                        else:
                            monte_str = el
                            X_mean_el = X_mean[xc]
                        monte_n_list = re_num.findall(monte_str)
                        monte_n_points = int(''.join(monte_n_list)) if monte_n_list else n_points
                        el_list = np.random.randn(monte_n_points) * X_std[xc] + X_mean_el
                    elif '~' in el:
                        x_min, x_max = map(str, X_analysis[xc].agg(['min','max']))
                        el_split = el.split('~')
                        if len(el_split) == 2:
                            split_list = list(map(lambda x: str(x).strip().replace('min', x_min).replace('max',x_max), el_split))
                            if split_list[0] == '':
                                split_list[0] = x_min
                            if split_list[1] == '':
                                split_list[1] = x_max
                            el_list = np.linspace(*map(float, split_list), n_points)
                        elif len(el_split) == 1:
                            el_list = float(el_split[0].strip().replace('min', x_min).replace('max',x_max))
                elif 'int' in str(type(el)) or 'float' in str(type(el)):
                    el_list = conditions[xc]
                elif 'numpy' in str(type(el)):
                    el_list = conditions[xc]
            else:       # object dtype
                el_unique_vc = X_analysis[xc].value_counts()/len(X_analysis[xc])
                
                if type(el) == str:
                    if el.strip() in ['all', '~', 'min~max', 'min ~ max']:
                        el_list = list(el_unique_vc.index)
                    elif 'monte' in el.lower(): 
                        monte_str = el
                        monte_n_list = re_num.findall(monte_str)
                        monte_n_points = int(''.join(monte_n_list)) if monte_n_list else n_points
                        if monte_n_points > len(el_unique_vc):
                            monte_n_points = len(el_unique_vc)
                        el_list = list(np.random.choice(el_unique_vc.index, size=monte_n_points, replace=False, p=el_unique_vc))
                    else:
                        el_list = el
                elif type(el) == list:
                    el_list = el.copy()
                else:
                    el_list = el
            X_dict[xc] = el_list
            # break
        # X_dict 
        
        X_dict_array =  {k: v for k, v in X_dict.items() if type(v) == np.ndarray or type(v) == list}
        if len(X_dict_array) > 0:
            grid_X_frame_temp = pd.DataFrame(np.array(np.meshgrid(*X_dict_array.values())).reshape(len(X_dict_array),-1).T, columns=X_dict_array.keys())
                       
            for k, v in X_dict.items():
                if k not in X_dict_array.keys():
                    grid_X_frame_temp[k] = v
            grid_X_frame = grid_X_frame_temp[list(X_dict.keys())]
        else:
            grid_X_frame = pd.Series(X_dict).to_frame().T
        grid_X = self.DataHandler.transform(grid_X_frame, apply_kind=self.train_X_info_split.kind)
        
        for xc in grid_X:
            apply_dtype = 'float' if 'int' in str(dtypes[xc]) or 'float' in str(dtypes[xc]) else 'object'
            if apply_dtype == 'object':
                obj_type = type(X_analysis[xc].iloc[0])
                if 'str' in str(obj_type):
                    grid_X[xc] = grid_X[xc].apply(lambda x: str(x))
                if 'int' in str(obj_type):
                    grid_X[xc] = grid_X[xc].apply(lambda x: int(x))
                if 'float' in str(obj_type):
                    grid_X[xc] = grid_X[xc].apply(lambda x: float(x))
                if 'bool' in str(obj_type):
                    grid_X[xc] = grid_X[xc].apply(lambda x: bool(x))
                grid_X[xc] = grid_X[xc].astype(self.train_X_info_split.dtypes[xc])
            else:
                grid_X[xc] = grid_X[xc].astype(apply_dtype)

        if save_result:
            self.n_points = n_points
            self.conditions = conditions
            self.grid_X = grid_X

        if return_result:
            return grid_X
        else:
            return self

    # apply model to grid_table
    def grid_apply_model(self, grid_X=None, x=None, y=None, model='LinearRegression', train_X=None, conditions={}, n_points=None,
        inplace=True, return_result=False):
        
        # define train, grid instance
        self.define_train_grid_data(train_X=train_X, grid_X=grid_X, conditions=conditions, n_points=n_points)

        # grid, train frame
        grid_X_frame = pd.DataFrame(**self.grid_X_info_split.data).astype(self.grid_X_info_split.dtypes)
        train_X_frame = pd.DataFrame(**self.train_X_info_split.data).astype(self.train_X_info_split.dtypes)

        # x, y → list
        if x is not None:
            if type(x) == list:
                x_list = x.copy()
            else:
                x_list = [x]
        
        if y is not None:
            if type(y) == list:
                y_list = y.copy()
            else:
                y_list = [y]

        # grid_apply by model
        for yc in y_list:
            model_instance = class_object_execution(model)
            model_instance.fit(train_X_frame[x_list], train_X_frame[yc])
            grid_X_frame[yc] = model_instance.predict(grid_X_frame[x_list])

        result = grid_X_frame[y_list + x_list]

        # inplace
        if inplace is True:
            self.grid_X = self.DataHandler.transform(grid_X_frame, apply_kind=self.grid_X_info.kind)
        
        # return
        if return_result is True:
            return self.DataHandler.transform(result, apply_kind=self.grid_X_info.kind)
        else:
            return self

    # predict_from_grid(train_X)
    def predict_from_train_X(self, train_X=None, grid_X=None, conditions={}, n_points=None,
                    estimator=None, encoder=None, encoderX=None, encoderY=None):
        if grid_X is None:
            if self.grid_X is None:
                grid_X = self.make_grid(train_X=train_X, conditions=conditions, n_points=n_points, return_result=True, save_result=False)
            else:
                grid_X = self.grid_X
        
        # if estimator is None:
        #     estimator = self.estimator
        # if encoder is None:
        #     encoder = self.encoder
        # if encoderX is None:
        #     encoderX = self.encoderX
        # if encoderY is None:
        #     encoderY = self.encoderY
        grid_instance = self.DataHandler.data_info_split(grid_X)
        grid_frame = pd.DataFrame(**grid_instance.data).astype(grid_instance.dtypes)
        
        if encoder is not None:
            grid_frame_apply = encoder.transform(grid_frame)
        elif encoderX is not None:
            grid_frame_apply = encoderX.transform(grid_frame)
        else:
            grid_frame_apply = grid_frame.copy()

        pred_y_temp = estimator.predict(grid_frame_apply)
        pred_y_temp = self.DataHandler.transform(pred_y_temp, apply_kind='pandas', apply_index=grid_frame.index, apply_columns=[self.y_name])

        if encoderY is None and encoder is not None:
            encoderY = copy.deepcopy(encoder)
            for xc in self.train_X_info_split.data['columns']:
                del encoderY.encoder[xc]
            encoderY = list(encoderY.encoder.values())[0]

        if encoderY is not None:
            pred_y = self.DataHandler.transform(encoderY.inverse_transform(pred_y_temp), apply_kind=grid_instance.kind)
        else:
            pred_y = self.DataHandler.transform(pred_y_temp, apply_kind=grid_instance.kind)
        return {'grid_frame': grid_frame, 'pred_y':pred_y}

    # predict ***
    def predict(self, grid_X=None, train_X=None, conditions={}, n_points=None,
        estimator=None, encoder=None, encoderX=None, encoderY=None, y_name=None, 
        unpredictable=True,
        return_all=False):

        if train_X is None:
            train_X = self.train_X
        
        # grid_X auto_filling        
        if grid_X.shape[1] < train_X.shape[1] :
            train_mean = self.make_grid(train_X, save_result=False, return_result=True)
            grid_X_temp = pd.DataFrame()
            for i in train_mean:
                if i in grid_X.columns:
                    grid_X_temp[i] = grid_X[i]
                else:
                    grid_X_temp[i] = [train_mean[i].values[0]] * grid_X.shape[0]
            grid_X = grid_X_temp.copy()
        
        # define train instance
        self.define_train_grid_data(train_X=train_X, grid_X=grid_X, conditions=conditions, n_points=n_points)

        if estimator is None:
            if  self.estimator is None:
                raise Exception("estimator must be required.")
            else:
                estimator = self.estimator
        if encoder is None:
            encoder = self.encoder
        if encoderX is None:
            encoderX = self.encoderX
        if encoderY is None:
            encoderY = self.encoderY
        if y_name is None:
            if self.y_name is None:
                y_name = 'pred_y' 
            else:
                y_name = self.y_name

        result = self.predict_from_train_X(train_X=train_X, grid_X=grid_X, conditions=conditions, n_points=n_points,
            estimator=estimator, encoder=encoder, encoderX=encoderX, encoderY=encoderY)
        grid_frame = result['grid_frame']
        # return result
        pred_y = pd.Series(result['pred_y'], name=y_name, index=result['pred_y'].index)
        # pred_y = pd.Series(result[y_name], name=y_name, index=result[y_name].index)
    
        predictable_frame = self.predictable_check(target_data=grid_frame, criteria_data=train_X)
        unpredict = pd.Series()
        if len(predictable_frame) > 0:
            unpredict = pred_y[predictable_frame['unpred']]

        if unpredictable is True and len(unpredict) > 0:
            print(f"{list(unpredict.index)} index data is unpredictable date (from estimator).")
        if return_all is False:
            return self.DataHandler.transform(pred_y, apply_ndim=1, apply_kind=self.train_X_info_split.kind)
        elif return_all is True:
            if predictable_frame:
                return pd.concat([grid_frame, pred_y, predictable_frame], axis=1)
            else:
                return pd.concat([grid_frame, pred_y], axis=1)

    # plot one element
    def plot_element(self, x=None, x2=None, train_X=None, grid_X=None, conditions={}, estimator=None, 
        encoder=None, encoderX=None, encoderY=None, y_name=None,
        n_points=None, xlim=None, x2lim=None, ylim=None,
        figsize=None, title=None, contour=True, display_unpredictable=True, text_points = 7,
        decimals = None,
        return_plot=True):

        if train_X is None:
            train_X = self.train_X

        # define train instance
        self.define_train_grid_data(train_X=train_X, grid_X=False)


        # estimator, encoderX, encoderY, y_name
        if estimator is None:
            if  self.estimator is None:
                raise Exception("estimator must be required.")
        else:
            self.estimator = estimator
        if encoder is not None:
            self.encoder = encoder
        if encoderX is not None:
            self.encoderX = encoderX
        if encoderY is not None:
            self.encoderY = encoderY
        if y_name is None:
            if self.y_name is None:
                y_name = 'pred_y' 
            else:
                y_name = self.y_name
        
        if len(conditions) == 0:
            conditions = self.conditions
        
        if x2 is None:
            apply_condition = {x:'min~max'}
            apply_condition.update(dict(filter(lambda el: el[0]==x, conditions.items())))
        else:
            apply_condition = {x:'min~max', x2:'min~max'}
            apply_condition.update(dict(filter(lambda el: el[0]==x, conditions.items())))
            apply_condition.update(dict(filter(lambda el: el[0]==x2, conditions.items())))

        # data_set
        pred_result = self.predict_from_train_X(train_X=train_X, conditions=apply_condition, n_points=n_points,
                    estimator=self.estimator, encoder=self.encoder, encoderX=self.encoderX, encoderY=self.encoderY)

        grid_frame = pred_result['grid_frame']
        pred_y = pd.Series(pred_result['pred_y'], name=y_name)
        predictable_frame = self.predictable_check(target_data=grid_frame, criteria_data=train_X)

        plot_table_1D = pd.concat([grid_frame, pred_y, predictable_frame], axis=1)
        plot_table_lower_1D = plot_table_1D[plot_table_1D['lower_columns'].apply(lambda k: False if len(k) == 0 else True)]
        plot_table_upper_1D = plot_table_1D[plot_table_1D['upper_columns'].apply(lambda k: False if len(k) == 0 else True)]

        outlier_colors = {True:'orange', False:'steelblue'}

        if return_plot:
            f = plt.figure()
        
        # plot
        if x2 is None:       # 1D
            if title is not None:
                plt.title(title)
            else:
                plt.title(f"{y_name} by {x} (y_range: {auto_formating(pred_y.max() - pred_y.min())})")
            
            # plot_scatter
            if display_unpredictable:
                for gi, gv in plot_table_1D.groupby('unpred'):
                    plt.scatter(gv[x], gv[y_name], edgecolor='white', alpha=0.3, color=outlier_colors[gi])
            else:
                plt.scatter(plot_table_1D[x], plot_table_1D[y_name], edgecolor='white', alpha=0.3)
            
            # plot_line
            self.plot_data = plot_table_1D[[x, y_name]]
            plt.plot(plot_table_1D.groupby(x)[y_name].mean(), alpha=0.7)
            
            text_points_list = [int(i) for i in np.linspace(0, len(plot_table_1D)-1, text_points)]
            for pei, (pti, ptd) in enumerate(plot_table_1D.iterrows()):
                if pei in text_points_list:
                    plt.text(ptd[x], ptd[y_name], auto_formating(ptd[y_name]))
            
            if self.confidential_interval is not None:
                plt.fill_between(plot_table_1D[x], plot_table_1D[y_name]+self.confidential_interval, plot_table_1D[y_name]-self.confidential_interval,
                                 alpha=0.1, facecolor='green')
            
            if ylim is not None:
                plt.ylim(ylim[0], ylim[1])
            if display_unpredictable:
                if len(plot_table_lower_1D) > 0:
                    plt.plot(plot_table_lower_1D.groupby(x)[y_name].mean(), color='orange')
                if len(plot_table_upper_1D) > 0:
                    plt.plot(plot_table_upper_1D.groupby(x)[y_name].mean(), color='orange')
            plt.xlabel(x)
            plt.ylabel(y_name)
            
        else:
            plot_table_2D = plot_table_1D.groupby([x2,x])[y_name].mean().unstack(x).sort_index(ascending=False)
            self.trainX = train_X
            self.apply_condition = apply_condition
            self.plot_data = plot_table_2D
            
            if title is not None:
                plt.title(title)
            else:
                plt.title(f"{y_name} by {x}~{x2} (y_range: {auto_formating(plot_table_2D.max().max() - plot_table_2D.min().min())})")
            
            vmin = None if ylim is None else ylim[0]
            vmax = None if ylim is None else ylim[1]
            CTMap = plt.contourf(plot_table_2D.columns, plot_table_2D.index, plot_table_2D, cmap='jet', vmin=vmin, vmax=vmax)
            # plt.contour(plot_table_2D.columns, plot_table_2D.index, plot_table_2D, cmap='jet', vmin=vmin, vmax=vmax)
            
            if decimals is None:
                decimals = 0 if -np.log10(plot_table_2D.mean().mean())+1 < 0 else int(-np.log10(plot_table_2D.mean().mean())+1)
            plt.clabel(CTMap, inline=True, colors ='grey', fmt=f'%.{decimals}f', fontsize=15)
            CTbar = plt.colorbar(extend = 'both')
            CTbar.set_label(y_name)
            plt.xlabel(x)
            plt.ylabel(x2)
            
            if x2lim is not None:
                plt.ylim(x2lim[0], x2lim[1])
        
        if xlim is not None:
            plt.xlim(xlim[0], xlim[1])
            
        if return_plot:
            plt.close()
            self.plot = f
            return self.plot

        
        # return pd.concat([plot_table_1D, plot_table_lower_1D, plot_table_upper_1D])
        
        # pass

    # influence_summary ***
    def influence_summary(self, train_X=None, conditions={}, n_points=None, grid_X=None,
        estimator=None, encoder=None, encoderX=None, encoderY=None, y_name=None, 
        feature_importances=True, summary_plot=True, summary_table=True, sort=False):
        '''
         . *train_X : training Dataset
         . *estimator : model
         . (if need) encoder : apply both X and y variables.
         . (if need) encoderX : apply only X variables.
         . (if need) encoderY : apply only y variable.
        '''
        if train_X is None:
            if self.train_X is not None:
                train_X = self.train_X
        # define train instance
        self.define_train_grid_data(train_X=train_X, grid_X=False)

        # define train, grid instance
        conditions_dict = {c:'min~max' for c in self.train_X_info_split.data['columns']}
        if conditions:
            conditions_dict.update(conditions)

        # estimator, encoderX, encoderY, y_name
        if estimator is None:
            if  self.estimator is None:
                raise Exception("estimator must be required.")
            else:
                estimator = self.estimator
        if encoder is None:
            encoder = self.encoder
        if encoderX is None:
            encoderX = self.encoderX
        if encoderY is None:
            encoderY = self.encoderY
        if y_name is None:
            if self.y_name is None:
                y_name = 'pred_y' 
            else:
                y_name = self.y_name

        if n_points is None:
            n_points = self.n_points

        # influence analysis
        self.influence_dict_all = {}
        self.influence_dict = {}
        feature_influence = {}

        dtypes = self.train_X_info_split.dtypes

        y_min_total = np.inf
        y_max_total = -np.inf
        for ic in conditions_dict:
            grid_X = self.make_grid(train_X=train_X, conditions={ic: conditions_dict[ic]}, n_points=n_points, return_result=True, save_result=False)
            pred_result = self.predict_from_train_X(train_X=train_X, grid_X=grid_X, conditions={ic: conditions_dict[ic]}, n_points=n_points,
                    estimator=estimator, encoder=encoder, encoderX=encoderX, encoderY=encoderY)
            
            grid_temp_frame = pred_result['grid_frame']
            grid_temp_frame[y_name] = pred_result['pred_y']
            self.influence_dict_all[ic] = self.DataHandler.transform(grid_temp_frame, apply_kind=self.train_X_info.kind)
            self.influence_dict[ic] = self.DataHandler.transform(grid_temp_frame[[ic]+[y_name]], apply_kind=self.train_X_info.kind)

            # summary_plot
            y_argmin_idx = grid_temp_frame[y_name].argmin()
            y_argmax_idx = grid_temp_frame[y_name].argmax()
            x_argmin = grid_temp_frame[ic].iloc[y_argmin_idx]
            x_argmax = grid_temp_frame[ic].iloc[y_argmax_idx]
            x_min = grid_temp_frame[ic].min()
            x_max = grid_temp_frame[ic].max()
            y_min = grid_temp_frame[y_name].min()
            y_max = grid_temp_frame[y_name].max()
            
            if 'int' in str(train_X[ic].dtype).lower() or 'float' in str(train_X[ic].dtype).lower():
                feature_influence[ic] = {'delta_y': y_max - y_min, 'delta_x_ymax': x_argmax - x_argmin,
                        'delta_x': x_max - x_min, 'max_slope': (y_max - y_min) / (x_argmax - x_argmin)}
            else:
                feature_influence[ic] = {'delta_y': y_max - y_min, 'delta_x_ymax': f"{x_argmin} ~ {x_argmax}",
                        'delta_x': np.nan, 'max_slope': np.nan}
            feature_influence[ic] = {k: auto_formating(v, return_type='str') for k, v in feature_influence[ic].items()}
            feature_influence[ic]['y_range'] = auto_formating(y_min, return_type='str') + ' ~ ' + auto_formating(y_max, return_type='str')
            feature_influence[ic]['x_ymax_range'] = auto_formating(x_argmin, return_type='str') + ' ~ ' + auto_formating(x_argmax, return_type='str')
            feature_influence[ic]['x_range'] = auto_formating(x_min, return_type='str') + ' ~ ' + auto_formating(x_max, return_type='str')

            y_min_total = min(y_min_total, y_min)
            y_max_total = max(y_max_total, y_max)
             
        # feature_plot
        self.feature_plot = {}
        for ic in conditions_dict:
            f = plt.figure()
            self.plot_element(x=ic, train_X=train_X, grid_X=self.influence_dict_all[ic], conditions={ic:'min~max'},
                estimator=estimator, encoder=encoder, encoderX=encoderX, encoderY=encoderY, y_name=y_name, return_plot=False)
            plt.ylim(y_min_total*0.95, y_max_total*1.05)
            plt.close()
            self.feature_plot[ic] = f
        
        # feature_influence (table)
        self.summary_table = pd.DataFrame(feature_influence).T
        self.summary_table['plot'] = pd.Series(self.feature_plot)
        
        if feature_importances:
            print(f'==== < Feature Importances Plot > ====')
            print(f' → self.feature_importances_plot')

        self.feature_importances_plot = plt.figure(figsize=(5, self.train_X.shape[1]*0.13+2) )
        plt.barh(self.summary_table.index[::-1], self.summary_table['delta_y'].apply(lambda x: x.replace(',','')).astype('float')[::-1])
        
        if feature_importances:
            plt.show()
        else:
            plt.close()
        

        # summary_plot
        if summary_plot:
            print(f'==== < Feature Influence Summary Plot > ====')
            print(f' → self.summary_plot')
        
        ncols = len(conditions_dict)
        fig_ncols = 4 if ncols > 4 else ncols
        fig_nrows = ((ncols // 4)+1) if ncols > 4 else 0.4

        fig = plt.figure(figsize=(fig_ncols * 4, fig_nrows * 4))
        fig.subplots_adjust(hspace=0.5)   # 위아래, 상하좌우 간격

        for idx, ic in enumerate(conditions_dict, 1):
            plt.subplot(int(fig_nrows)+1, fig_ncols, idx)
            self.plot_element(x=ic, train_X=train_X, grid_X=self.influence_dict_all[ic], conditions={ic:'min~max'},
                estimator=estimator, encoder=encoder, encoderX=encoderX, encoderY=encoderY, y_name=y_name, return_plot=False)
            plt.ylim(y_min_total*0.95, y_max_total*1.05)
        if summary_plot:
            plt.show()
        else:
            plt.close()
        self.summary_plot = fig

        if sort:
            self.summary_table = self.summary_table.sort_values('delta_y', ascending=False)

        print(f'==== < Feature Influence Summary Table > ====')
        print(f' → self.summary_table')
        if summary_table:
            print_DataFrame(self.summary_table)



# fi = FeatureInfluence(estimator=XGR, n_points=50, encoder=se_encoder, train_X=train_set[X_cols], y_name='Fare')
# # fi.make_grid(conditions={'Pclass':'~','Embarked':'monte2'}, return_result=True)
# fi.predict(test_set[X_cols], return_all=True)

# fi.plot_element(x='Survived')







# ################################################################################################################
# class BestEstimatorSearch:
#     """
#      【required (Library)】numpy, pandas, bayes_opt.BayesianOptimization, bayes_opt.UtilityFunction, 
#                            copy.deepcopy, itertools.combinations
#      【required (Class)】 BayesOpt, EstimatorSearch, ModelEvaluate
#      【required (Function)】 auto_formating, print_DataFrame


#      < Result Guidance by method>
#       (init) estimators, metrics, scoring_option, train_X, train_y, valid_X, valid_y, test_X, test_y

#       (fit) train_X, train_y, valid_X, valid_y, test_X, test_y, verbose
#         → self.estimators

#       (emsemble) weights, sorting, verbose
#         → self.ensemble_result
#         → self.best_ensemble
#         → self.best_ensemble_estimator

#       (ensemble_summary) best_estimator, n_points, encoder, encoderX, encoderY, verbose
#         → self.summary_table
#         → self.summary_plot
#     """
    
#     def __init__(self, estimators={'LS':('linear', Lasso, {'random_state':0}, {'optimizer_params':{'pbounds': {'alpha':(0.0001,100)}, 'random_state':0}} )}, 
#         metrics={'r2_adj': 'r2_adj', 'rmse': 'rmse'}, 
#         scoring_option={'scoring':'neg_mean_squared_error', 'scoring_type':'cross_val_score'},
#         train_X=None, train_y=None, valid_X=None, valid_y=None, test_X=None, test_y=None):
#         """
#         estimators = {'estimator_name':('linear', 'estimator', 'parmas', 'optimizer_params') ...}
#         metrics = ['metric1', 'metric2']
#         """
#         self.estimators_params = estimators
#         self.metrics = metrics
#         self.scoring_option = scoring_option

#         self.train_X = train_X
#         self.train_y = train_y
#         self.valid_X = valid_X
#         self.valid_y = valid_y
#         self.test_X = test_X
#         self.test_y = test_y

#         self.estimators = None
#         self.ensemble_result = None
#         self.feature_influence = None

#     def fit(self, train_X=None, train_y=None, valid_X=None, valid_y=None, test_X=None, test_y=None, verbose=1):
#         self.train_X = self.train_X if train_X is None else train_X
#         self.train_y = self.train_y if train_y is None else train_y
#         self.valid_X = (self.train_X if self.valid_X is None else self.valid_X) if valid_X is None else valid_X
#         self.valid_y = (self.train_y if self.valid_y is None else self.valid_y) if valid_y is None else valid_y
#         self.test_X = (self.train_X if self.test_X is None else self.test_X) if test_X is None else test_X
#         self.test_y = (self.train_y if self.test_y is None else self.test_y) if test_y is None else test_y

#         models = {}
#         for en, ev in self.estimators_params.items():
#             models[en] = {}
#             if verbose > 0:
#                 print(f"【 {en} model fitting 】", end=' ')

#             if 'verbose' not in ev[3].keys():
#                 ev[3].update({'verbose':0})
            
#             # optimize model
#             ms_otim = EstimatorSearch(estimator=ev[1], params=ev[2], **ev[3], **self.scoring_option)
#             ms_otim.fit(train_X=self.train_X, train_y=self.train_y, valid_X=self.valid_X, valid_y=self.valid_y, verbose=0, return_result=False)
            
#             pred_y = ms_otim.best_estimator.predict(self.test_X)

#             # metric model
#             ms_me = ModelEvaluate(self.test_X, self.test_y, model=ms_otim.best_estimator, verbose=0)
#             ms_metric = {}
#             for mk, mv in self.metrics.items():
#                 try:
#                     ms_metric[mk] = eval(f'ms_me.{mv}')
#                 except:
#                     ms_metric[mk] = mv(self.test_y, pred_y)
#             ms_metric_str = ', '.join([f"{k}: {auto_formating(v)}" for k,v in ms_metric.items()])

#             # plotting model
#             ms_plot = plt.figure(figsize=(5, self.train_X.shape[1]*0.13+2))
#             plt.title(f"{en}\n{ms_metric_str}")
#             if 'linear' in ev[0]:
#                 pd.Series(ms_otim.best_estimator.coef_, index=self.train_X.columns).sort_values().plot.barh()
#             elif ('ensemble' in ev[0]) and ('tree' in ev[0]):
#                 pd.Series(ms_otim.best_estimator.feature_importances_, index=self.train_X.columns).sort_values().plot.barh()
#             plt.close()

#             models[en]['estimator'] = copy.deepcopy(ms_otim.best_estimator)
#             models[en]['evaluate'] = dict(ms_me.metrics._asdict())
#             models[en]['metric'] = ms_metric
#             models[en]['plot'] = ms_plot
#             if verbose > 0:
#                 print(f"   (estimator) {ms_otim.best_estimator}\n   (metric) {ms_metric_str} ***")
        
#         self.estimators = models
#         if verbose > 0:
#             print('='*100)
#             print('done. → (result) self.estimators')
        
#         return self

#     def ensemble(self, train_X=None, train_y=None, valid_X=None, valid_y=None, test_X=None, test_y=None,
#         estimators=None, weights=None, sorting='auto', verbose=1):

#         self.train_X = self.train_X if train_X is None else train_X
#         self.train_y = self.train_y if train_y is None else train_y
#         self.valid_X = (self.train_X if self.valid_X is None else self.valid_X) if valid_X is None else valid_X
#         self.valid_y = (self.train_y if self.valid_y is None else self.valid_y) if valid_y is None else valid_y
#         self.test_X = (self.train_X if self.test_X is None else self.test_X) if test_X is None else test_X
#         self.test_y = (self.train_y if self.test_y is None else self.test_y) if test_y is None else test_y

#         if self.estimators is None:
#             self.fit(verbose=0)

#         models = {en: ev['estimator'] for en, ev in self.estimators.items()} if estimators is None else estimators
#         weights = {en: 1/ev['evaluate']['rmse'] for en, ev in self.estimators.items()} if weights is None else weights

#         # models combinations
#         estimators = {}
#         estimators_weights = {}       

#         count = len(models) + 1
#         for n in range(2, len(models)+1):
#             comb_mdl_name = list(combinations(models.keys(), n))
#             comb_mdl_models = list(combinations(models.values(), n))
#             comb_mdl_weights = list(combinations(weights.values(), n))

#             for name, model, weight in zip(comb_mdl_name, comb_mdl_models, comb_mdl_weights):
#                 estimators[f'M{count:03}'] = [(mn, mm) for mn, mm in zip(name, model)]
#                 estimators_weights[f'M{count:03}'] = list(weight)
#                 count += 1
#         estimators_comb_series = pd.Series({k: [e[0] for e in v] for k, v in estimators.items()}, name='estimators_comb')

#         # basic estimators
#         basic_estimator_idxs = [f"M{n:03}" for n in np.arange(1, len(models)+1)]
#         basic_summary_dict = {}
#         for e, (en, mdl) in zip(basic_estimator_idxs, models.items()):
#             if verbose > 0:
#                 print(f"< {e} : {en} >")

#             basic_me = ModelEvaluate(self.test_X, self.test_y, model=mdl, verbose=verbose)
#             basic_metric = dict(basic_me.metrics._asdict())

#             basic_summary_dict[e] = {'estimators_comb':en, **basic_metric, 'ensemble_estimators':mdl}
#         basic_score_frame = pd.DataFrame(basic_summary_dict).T
        
#         # votting
#         votting_models = {}
#         votting_socres = {}
#         for i, (en, e) in enumerate(zip(estimators_comb_series, estimators)):
#             if verbose > 0:
#                 print(f"< {e} : {', '.join(en)} >")
            
#             VR = VotingRegressor(estimators=estimators[e], weights=estimators_weights[e])
#             VR.fit(self.train_X, self.train_y)
#             pred_y = VR.predict(self.train_X)
            
#             VR_me = ModelEvaluate(self.test_X, self.test_y, model=VR, verbose=verbose)

#             # votting_metric = {}
#             # for mk, mv in self.metrics.items():
#             #     try:
#             #         votting_metric[mk] = eval(f'VR_me.{mv}')
#             #     except:
#             #         votting_metric[mk] = mv(self.test_y, pred_y)
#             votting_metric = dict(VR_me.metrics._asdict())

#             votting_models[e] = copy.deepcopy(VR)
#             votting_socres[e] = votting_metric

#         estimators_series = pd.Series(votting_models, name='ensemble_estimators')

#         votting_scores_frame = pd.DataFrame(votting_socres)
#         votting_scores_frame = pd.concat([estimators_comb_series.to_frame().T, votting_scores_frame, estimators_series.to_frame().T], axis=0)
#         votting_scores_frame = votting_scores_frame.T

#         # concat result (basics ↔ combinations)
#         votting_scores_frame = pd.concat([basic_score_frame, votting_scores_frame], axis=0)

#         # sorting
#         if sorting == 'auto' or sorting is None:
#             ensemble_sort = votting_scores_frame.copy()
#             ensemble_sort.insert(ensemble_sort.shape[1]-1, 'ensemble_score', ensemble_sort['rmse'] * (1-ensemble_sort['r2_score']) * (1-ensemble_sort['mape']), True)
#             # ensemble_sort['ensemble_score'] = ensemble_sort['rmse'] * (1-ensemble_sort['r2_score'])
#             ensemble_sort.sort_values('ensemble_score', axis=0, inplace=True)
            
#             if sorting == 'auto':
#                 votting_scores_frame = ensemble_sort.copy()
#         else:
#             ensemble_sort = votting_scores_frame.copy()
#             ensemble_sort.sort_values(sorting, axis=0, inplace=True)
#             votting_scores_frame = ensemble_sort.copy()

#         self.ensemble_result = votting_scores_frame
#         self.best_ensemble = ensemble_sort.iloc[0, :]
#         self.best_ensemble_estimator = self.best_ensemble['ensemble_estimators']

#         if verbose > 0:
#             print()
#             print(f"done.")
#             print(f"→ (ensemble_result) self.ensemble_result")
#             print(f"  (best_ensemble) self.best_ensemble")
#             print(f"  (best_ensemble_estimator) self.best_ensemble_estimator")
#             print_DataFrame(ensemble_sort)
        
#         return self

#     def ensemble_summary(self, train_X=None, train_y=None, valid_X=None, valid_y=None, test_X=None, test_y=None,
#             best_estimator=None, n_points=50,
#             encoder=None, encoderX=None, encoderY=None, verbose=1):
        
#         self.train_X = self.train_X if train_X is None else train_X
#         self.train_y = self.train_y if train_y is None else train_y
#         self.valid_X = (self.train_X if self.valid_X is None else self.valid_X) if valid_X is None else valid_X
#         self.valid_y = (self.train_y if self.valid_y is None else self.valid_y) if valid_y is None else valid_y
#         self.test_X = (self.train_X if self.test_X is None else self.test_X) if test_X is None else test_X
#         self.test_y = (self.train_y if self.test_y is None else self.test_y) if test_y is None else test_y

#         if self.estimators is None:
#             self.fit(verbose=0)
#         if self.ensemble_result is None:
#             self.ensemble(verbose=0)

#         if encoderX is not None:
#             train_X = encoderX.inverse_transform(self.train_X)
#         elif encoder is not None:
#             train_X = encoderX.inverse_transform(self.train_X)
#         else:
#             train_X = self.train_X
        
#         if best_estimator is None:
#             best_estimator = self.best_ensemble_estimator

#         feature_influence = FeatureInfluence(train_X=train_X, estimator=best_estimator, 
#                     encoder=encoder, encoderX=encoderX, encoderY=encoderY,
#                     n_points=n_points)

#         summary_table = True if verbose > 0 else False
#         summary_plot = True if verbose > 0 else False
#         feature_influence.influence_summary(summary_table=summary_table, summary_plot=summary_plot)

#         self.feature_influence = feature_influence
#         self.summary_table = feature_influence.summary_table
#         self.summary_plot = feature_influence.summary_plot

#         if verbose > 0:
#             print()
#             print(f"done.")
#             print(f"  (summary_table) self.summary_table")
#             print(f"  (summary_plots) self.summary_plot")


################################################################################################################




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
try:
    font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
    rc('font', family=font_name)
except:
    pass
import seaborn as sns
# import cvxpy
import scipy.stats as stats


import copy
from collections import namedtuple

# from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import *

# import sys
# sys.path.append(r'D:\작업방\업무 - 자동차 ★★★\Worksapce_Python\DS_Module')
# sys.path.append('d:\\Python\\★★Python_POSTECH_AI\\DS_Module')    # 모듈 경로 추가

# from DS_OLS import *

# absolute_path = 'D:/Python/★★Python_POSTECH_AI/Postech_AI 4) Aritificial_Intelligent/교재_실습_자료/'
# absolute_path = 'D:/Python/★★Python_POSTECH_AI/Dataset_AI/DataMining/'




# ModelEvaluate
class ModelEvaluate():
    '''
    < input >
    X : DataFrame or 2-Dim matrix
    y : Series
    model : sklearnModel
    '''
    def __init__(self, X, y, model, model_type=None, verbose=1):
        model_name = str(type(model)).lower()
        X_frame = pd.DataFrame(X)
        y_true = np.array(y).ravel()
        y_pred = np.array(model.predict(X)).ravel()
        n_data, dof = X.shape

        if model_type is None:
            if ('regress' in model_name) or ('lasso' in model_name) or ('ridge' in model_name) or ('elasticnet' in model_name):
                model_type = 'regressor'
            if 'classi' in model_name:
                model_type = 'classifier'

        if model_type == 'regressor':
            #### sum_square ****
            sum_square_instance = namedtuple('sum_square', ['sst', 'ssr', 'sse'])
            self.sst = sum((y_true-y_true.mean())**2)
            self.ssr = sum((y_true.mean() - y_pred)**2)
            self.sse = sum((y_true - y_pred)**2)            

            sum_square_list = [self.sst, self.ssr, self.sse]
            self.sum_square = sum_square_instance(*[self.auto_decimal(m) for m in sum_square_list]) 

            #### metrics ****
            metrics_instance = namedtuple('metrics', ['r2_score', 'r2_adj', 'mse', 'rmse', 'mae', 'mape'])

            self.r2_score = 1 - self.sse/self.sst
            # self.r2_adj = 1 - ((n_data-1) * self.ssr/self.sst) / (n_data - dof - 1)
            self.r2_adj = 1 - ((n_data-1) * self.sse/self.sst) / (n_data - dof - 1)
            

            self.mse = self.sse/(n_data-2)
            self.rmse = np.sqrt(self.mse)
            self.mae = sum(np.abs(y_true - y_pred)) / n_data
            mape_series = pd.Series(1 - np.abs( (y_true - y_pred) / y_true ))
            mape_series = mape_series[~((mape_series == np.inf) | (mape_series == -np.inf))]
            self.mape = mape_series.mean()
            
            metrics_list = [self.r2_score, self.r2_adj, self.mse, self.rmse, self.mae, self.mape]
            self.metrics = metrics_instance(*[self.auto_decimal(m) for m in metrics_list]) 

            if verbose > 0:
                print(' .', self.sum_square) 
                print(' .', self.metrics) 

            #### hypothesis ****
            try:
                if sum([i.lower() in model_name for i in ['LinearRegression', 'Ridge', 'Lasso', 'ElasticNet']]):
                    hypothesis_instance = namedtuple('hypothesis', ['tvalues', 'pvalues'])

                    params = np.array([model.intercept_] + list(model.coef_))
                    newX = pd.DataFrame({"Constant": np.ones(n_data)}, index=X_frame.index).join(X_frame)
                    std_b = np.sqrt(self.mse*(np.linalg.inv(np.dot(newX.T,newX)).diagonal()))
                    t = params/ std_b               
                    p = 2 * (1 - stats.t.cdf(np.abs(t), n_data - dof))

                    try:
                        X_names = ['const'] + list(X.columns)
                    except:
                        X_names = ['const'] + ['x' + str(i+1) for i in np.arange(dof)]
                    try:
                        y_name = y.name
                    except:
                        y_name = 'Y'


                    self.tvalues = {k: self.auto_decimal(v) for k, v in zip(X_names, t)}
                    self.pvalues = {k: self.auto_decimal(v) for k, v in zip(X_names, p)}

                    self.hypothesis = hypothesis_instance(self.tvalues, self.pvalues)

                    #### linear ****
                    linear_instance = namedtuple('linear', ['coef', 'formula'])
                    self.coef = {k: self.auto_decimal(v) for k, v in zip(X_names, params)}
                    self.formula = y_name + ' = ' + ''.join([ f'{str(v)}·{k}' if i == 0 else (' + ' if v > 0 else ' - ') + (str(abs(v)) if k=='const' else f'{abs(v)}·{str(k)}') for i, (k, v) in enumerate(self.coef.items())])

                    self.linear = linear_instance(self.coef, self.formula)
                    if verbose > 0:
                        print(' .', self.hypothesis) 
                        print(' .', self.linear)
            except:
                pass

        elif model_type == 'classifier':
            pass

    def decimal(self, x, rev=0):
        return 2 if x == 0 else int(-1*(np.floor(np.log10(abs(x)))-3-rev))

    def auto_decimal(self, x, rev=0):
        if pd.isna(x):
            return np.nan
        else:
            return round(x, self.decimal(x, rev=rev))



# # class FeatureInfluence
# class FeatureInfluence():
#     def __init__(self, estimator, scaler, n_points=50, X_range=None, X_mean=None, y_name=None):
#         self.estimator = estimator
#         self.scaler = scaler
#         self.n_points = n_points
#         self.X_range = X_range
#         self.X_mean = X_mean
#         self.y_name = y_name
    
#     def fit(self, X, y, on=None, plot=False):
#         estimator = self.estimator
#         scaler = self.scaler
#         n_points = self.n_points
#         X_range = self.X_range
#         X_mean = self.X_mean
#         y_name = self.y_name

#         if on is None:
#             features = X.columns.to_list()
#         else:
#             features = on.copy()

#         result = {}
#         plot_temper = {}
#         for x in features:
#             # make mean_X DataFrame
#             if X_mean is None:
#                 data_mean = pd.concat([X.mean().to_frame().T]*n_points, ignore_index=True)
#             else:
#                 data_mean = pd.concat([X_mean.to_frame().T]*n_points, ignore_index=True)
#                 data_mean = pd.DataFrame(scaler.transform(data_mean), columns=data_mean.columns)

#             if X_range is None:
#                 x_min, x_max = X[x].agg(['max','min'])
#                 x_min_max = np.linspace(x_max, x_min, num=n_points)
#             else:
#                 x_min_max = np.linspace(X_range[1], X_range[0], num=n_points)
#             data_mean[x] = x_min_max

#             x_column_idx = np.argwhere(data_mean.columns == x)[0][0]

#             data_mean.T.to_clipboard()
#             # evaluate
#             x_vector = scaler.inverse_transform(data_mean)[:, x_column_idx]
#             predict_vector = estimator.predict(data_mean)
#             predict_summary = pd.Series(predict_vector).agg(['mean', 'std', 'min', 'max'])
#             predict_decimal = fun_Decimalpoint(predict_summary['mean'])
#             predict_range = predict_summary['max'] - predict_summary['min']

#             predict_result = predict_summary.to_dict().copy()
#             predict_result['range'] = predict_range

#             if y_name is None:
#                 try:
#                     y_name = y.name
#                 except:
#                     y_name = 'Target'
            
#             fig = plt.figure()
#             plt.title(f"Influence of {y_name} by {x}\nstd: {round(predict_summary['std'], predict_decimal+1)},  range: {round(predict_range, predict_decimal+1)}")
#             plt.plot(x_vector, predict_vector, 'o-', markerfacecolor=(70/255,130/255,180/255,0.3), markersize=5)
#             predict_result['plot'] = fig
#             plt.close()
#             result[x] = predict_result

#             plot_temper[x] = {}
#             plot_temper[x]['predict_vector'] = predict_vector
#             plot_temper[x]['x_vector'] = x_vector

#         feature_influence = pd.DataFrame(result).T
#         self.feature_influence = feature_influence[['range', 'std', 'min', 'max']].applymap(lambda x: np.round(x, fun_Decimalpoint(x)))
#         self.feature_influence['plot'] = feature_influence['plot']
#         self.feature_influence.sort_values('range', ascending=False, inplace=True)
        
#         self.predict_frame = pd.DataFrame(plot_temper).T
#         # self.feature_influence = pd.concat([self.feature_influence, pd.DataFrame(plot_temper).T], axis=1 )

#         if plot is not False:
#             print(f'==== < Feature Influence Plot > ====')
#             print(f' → self.influence_plot')
#             ncols = len(self.feature_influence.index)
#             fig_ncols = 4 if ncols > 4 else ncols
#             fig_nrows = ((ncols // 4)+1) if ncols > 4 else 0.4

#             fig = plt.figure(figsize=(fig_ncols * 4, fig_nrows * 4))
#             fig.subplots_adjust(hspace=0.5)   # 위아래, 상하좌우 간격

#             for idx, x in enumerate(self.feature_influence.index, 1):
#                 plt.subplot(int(fig_nrows)+1, fig_ncols, idx)
#                 plt.title(f"Influence of {y_name} by {x}\nstd: {self.feature_influence.loc[x, 'std']},  range: {self.feature_influence.loc[x, 'range']}")
#                 plt.plot(plot_temper[x]['x_vector'], plot_temper[x]['predict_vector'], 'o-', markerfacecolor=(70/255,130/255,180/255,0.3), markersize=5)
#             plt.show()
#             self.influence_plot = fig

#         print(f'==== < Summary Feature Influence > ====')
#         print(f' → self.feature_influence')
#         print_DataFrame(self.feature_influence)
#         # return self.feature_influence

#     def compare(self, normal_data, abnormal_data, plot=False):
#         normal_predict = self.estimator.predict(self.scaler.transform(normal_data.to_frame().T))[0]
#         abnormal_predict = self.estimator.predict(self.scaler.transform(abnormal_data.to_frame().T))[0]
#         def calc_predict_from_predict_table(data, x):
#             idx_vector = np.argwhere(self.predict_frame['x_vector'][x] < float(data[x]))
#             if len(idx_vector):
#                 idx = np.max(idx_vector)
#             else:
#                 idx = 0

#             if len(idx_vector):
#                 try:
#                     y_grad = (self.predict_frame['predict_vector'][x][idx+1] - self.predict_frame['predict_vector'][x][idx])
#                     x_grad = (self.predict_frame['x_vector'][x][idx+1] - self.predict_frame['x_vector'][x][idx])
#                     predict = y_grad/x_grad * (self.predict_frame['x_vector'][x][idx+1] - float(data[x])) + self.predict_frame['predict_vector'][x][idx]
#                 except:
#                     predict = self.predict_frame['predict_vector'][x][-1]
#             else:
#                 predict = self.predict_frame['predict_vector'][x][0]
                
#             return predict

#         result = {}
#         for x in self.predict_frame.index:
#             normal_x_predict = calc_predict_from_predict_table(data=normal_data, x=x)
#             abnormal_x_predict = calc_predict_from_predict_table(data=abnormal_data, x=x)
#             result[x] = abnormal_x_predict - normal_x_predict

#         self.differences = round((abnormal_predict - normal_predict), fun_Decimalpoint(abnormal_predict - normal_predict))
#         self.normal_predict = round(normal_predict, fun_Decimalpoint(normal_predict))
#         self.abnormal_predict = round(abnormal_predict, fun_Decimalpoint(abnormal_predict))

#         self.explain_feature = pd.Series(result).sort_values(ascending=False).to_frame()
#         self.explain_feature.columns = ['explain_amount']
#         self.explain_feature = self.explain_feature.applymap(lambda x: round(float(x), fun_Decimalpoint(x)-1))
        
#         if plot is not False:
#             print(f'==== < Total Difference Normal to Abnormal Plot > ====')
#             print(f' → self.explain_plot')
#             self.explain_plot = plt.figure()
#             plt.title('Explain Difference Amount of Features\nbetween Normal and Abnormal Data')
#             self.explain_feature['explain_amount'].sort_values(ascending=True).plot.barh()
#             plt.show()
#             print()
#         print(f'==== < Total Difference Normal to Abnormal Data > ====')
#         print(f'Predicted Differences between Normal and Abnormal : {self.differences} (Abnormal: {self.abnormal_predict}, Normal: {self.normal_predict})')
#         print(f' → self.explain_feature')
#         print_DataFrame(self.explain_feature)
#         # return self.explain_feature













































"""
# test_df = pd.read_clipboard()
test_dict = {'y': {0: 10, 1: 13, 2: 20, 3: 7, 4: 15},
        'x1': {0: 2, 1: 4, 2: 5, 3: 2, 4: 4},
        'x2': {0: 'a', 1: 'a', 2: 'b', 3: 'b', 4: 'b'},
        'x3': {0: 10, 1: 8, 2: 5, 3: 12, 4: 7},
        'x4': {0: 'g1', 1: 'g2', 2: 'g1', 3: 'g2', 4: 'g3'}}

test_dict2 = {'y': [10, 17, 8, 15],
        'x1': ['a', 'b', 'b', 'a'],
        'x2': [2, 6, 7, 10],
        'x3': [0, 3, 5, 2],
        'x4': ['g2', 'g1', 'g2', 'g1']}
test_df = pd.DataFrame(test_dict)
test_df2 = pd.DataFrame(test_dict2)


test_df
pd.get_dummies(test_df, drop_first=True)

test_1 = test_df[['x2','x4']]
test_2 = test_df2[['x1','x4']]


OE1 = DS_OneHotEncoder()
df1 = OE1.fit_transform(test_df)

LE1 = DS_LabelEncoder()
df2 = LE1.fit_transform(test_df)


y1 = df1[['y']]
x1 = df1.iloc[:,1:]

y2 = df2[['y']]
x2 = df2.iloc[:,1:]

# Model
LR = LinearRegression()
LR.fit(x1, y1)

RF = RandomForestRegressor()
RF.fit(x2, y2)

GB = GradientBoostingRegressor()
GB.fit(x2, y2)



models = [('LR', LR), ('RF', RF), ('GB', GB)]
ES = VotingRegressor(models)
ES.fit(x2, y2)


# class Feature_Influence:
#     def __init__(self, model):
#         pass

#     def fit(self, X):
#         pass
    
#     def plot(self, X):
#         pass

# ***


# ------------------------------------------------------------------------
from itertools import product

test_num = test_df[['x1','x3']]

std_scale = StandardScaler()
std_scale.fit(test_num)
norm_test_num = std_scale.transform(test_num)

std_scale.inverse_transform(norm_test_num)


# if normalize:
X_mdl = x2
model = ES

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler

'class' in str(type(MinMaxScaler()))
type(StandardScaler())
type('StandardScaler')

# class DF_Scaler:
# # StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler
# def __init__(self, scaler):
#     if type(scaler) == str:
#         pass
#     elif "class 'sklearn.preprocessing._data." in str(type(scaler)):
#         pass











# case Product
def product_case(dictionary):
    if dictionary:
        from itertools import product

        product_case = []
        for k in fix_params:
            if type(fix_params[k]) == list:
                product_case.append(fix_params[k])
            else:
                product_case.append([fix_params[k]])
        return pd.DataFrame(eval('list(product(' + str(product_case)[1:-1] + '))'), columns=dictionary.keys())
    else:
        return False

class Quantile:
    def __init__(self, q):
        self.q = q
    
    def __call__(self, X):
        return X.quantile(self.q)




# fix_params = {'x4_g2': 0, 'x4_g3':[0, 1]}
fix_params = {}

n_points = 30

method = 'MinMax'
method = 'Quantile'
q_low = 0.05
q_upper = 0.95



q_l = Quantile(q_low)
q_u = Quantile(q_upper)
q_low_str = ('0' + str(int(q_low*100)) if q_low < 0.1 else str(int(q_low*100)) ) + '%'
q_upper_str = ('0' + str(int(q_upper*100)) if q_upper < 0.1 else str(int(q_upper*100)) ) + '%'

X_agg = X_mdl.agg(['mean','std','min','max', 'median', q_l, q_u])
X_agg.index = ['mean','std','min','max', '50%', q_low_str, q_upper_str]



if method == 'MinMax':
    mid = 'mean'
    limit_l = 'min'
    limit_u = 'max'
elif method == 'Quantile':
    mid = '50%'
    limit_l = q_low_str
    limit_u = q_upper_str


influence_init = pd.concat([X_agg.loc[[mid],:]]*n_points, ignore_index=True)
influence_linspace = pd.DataFrame(np.linspace(X_agg.loc[limit_l,:], X_agg.loc[limit_u,:], n_points), columns=X_mdl.columns)
grid_params = product_case(fix_params)
# grid_params

fix_params
grid_params


X_mdl
for xi, xc in enumerate(X_mdl):
    print(xc)




# influence_matrix ------------------------------------------------------
influence_matrix = {}

for xi, xc in enumerate(X_mdl):
    # print(xc)
    influence_df = influence_init.copy()
    influence_df[xc] = influence_linspace[xc]
    influence_df['group'] = mid

    if type(grid_params) != bool:    
        influence_grid_temper = influence_df.copy()

        if xc in grid_params.columns:
            params_df = grid_params.drop(xc, axis=1)
            params_df.drop_duplicates(inplace=True)
        else:
            params_df = grid_params.copy()

        for g in params_df.index:
            # print(xc, params_df.loc[g,:].to_dict())
            grid_temper = influence_grid_temper.copy()
            param_case = params_df.loc[g,:]
            grid_temper.loc[:,params_df.columns] = param_case.to_numpy()
            grid_temper['group'] = str(param_case.to_dict())

            influence_df = pd.concat([influence_df, grid_temper],axis=0)
        
    influence_matrix[xc] = influence_df
influence_matrix





# influence_predict ------------------------------------------------------
influence_predict = {}
for ic in influence_matrix:
    influence_predict[ic] = influence_matrix[ic][[ic]]
    influence_predict[ic]['predict'] = model.predict(influence_matrix[ic].iloc[:,:-1])
    influence_predict[ic]['group'] = influence_matrix[ic]['group']
    
influence_predict

# sns.relplot(data=influence_predict['x1'], x='x1', y='predict', kind='line', hue='group')


x_value = 'x1'
for gi, gv in influence_matrix[x_value].groupby('group'):
    plt_df = gv.drop('group', axis=1)

    if gi == 'mean':

        plt.plot(plt_df[x_value], model.predict(plt_df), '--', label=gi, c='red', linewidth=2, alpha=0.5)        # model Model
    else:
        plt.plot(plt_df[x_value], model.predict(plt_df), '-', label=gi, alpha=0.5)        # model Model

plt.legend()
plt.ylim([model.predict(X_mdl).min(), model.predict(X_mdl).max()])
plt.show()


pd.Series(model.feature_importances_, index=X_mdl.columns)





# LR.coef_
# plt.plot(influence_linspace['x1'], LR.predict(influence_matrix['x1']), 'o-')      # LR Model
# plt.plot(influence_linspace['x2'], LR.predict(influence_matrix['x2']), 'o-')      # LR Model
# plt.plot(influence_linspace['x3_a'], LR.predict(influence_matrix['x3_a']), 'o-')      # LR Model
# plt.plot(influence_linspace['x3_b'], LR.predict(influence_matrix['x3_b']), 'o-')      # LR Model

plt.plot(influence_linspace['x1'], RF.predict(influence_matrix['x1']), 'o-')        # RF Model
plt.plot(influence_linspace['x2'], RF.predict(influence_matrix['x2']), 'o-')        # RF Model
# plt.plot(influence_linspace['x3_a'], RF.predict(influence_matrix['x3_a']), 'o-')        # RF Model
# plt.plot(influence_linspace['x3_b'], RF.predict(influence_matrix['x3_b']), 'o-')        # RF Model

# pd.DataFrame( np.hstack([test_mean_df['x1'].to_numpy().reshape(-1,1), LR.predict(test_mean_df).reshape(-1,1)]), columns=['x1', 'x1_predict'])






# X_np = X_mdl.to_numpy()
# X_np

# def rbf_basis(X_mdl, d=5, sigma='auto'):
#     X_agg = X_mdl.agg(['mean','std','min','max'])
#     if sigma == 'auto':
#         sigma =  X_agg.loc['std',:]
#     return np.exp(-((X_mdl - X_agg.loc['mean',:])**2 / (2 * sigma**2)))

# pd.DataFrame(np.linspace(X_agg.loc['min',:], X_agg.loc['max',:], 10),columns=X_mdl.columns)

"""



















