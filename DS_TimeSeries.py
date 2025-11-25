import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp

try:
    import numba
    import cvxpy 
except:
    pass

from collections import namedtuple
import datetime
import dateutil
import re

import statsmodels.api as sm

# https://ellun.tistory.com/320
# 날짜
# import pandas as pd 
# print(pd.to_datetime('2020-07-21')) 
# 2020-07-21 00:00:00 print(pd.to_datetime('07-21-2020')) 
# # 2020-07-21 00:00:00 print(pd.to_datetime('2020/07/21')) 
# # 2020-07-21 00:00:00 print(pd.to_datetime('07/21/2020')) 
# # 2020-07-21 00:00:00 
# 
# #2-2. 날짜 텍스트 → 날짜 변환(스칼라값)- 
# 
# numpy import numpy as np 
# print(np.datetime64('2020-07-21')) 
# # 2020-07-21 print(np.datetime64('07-21-2020')) # ValueError: Month out of range in datetime string "07-21-2020" 
# print(np.datetime64('2020/07/21')) # ValueError: Error parsing datetime string "2020/07/21" at position 4 
# print(np.datetime64('07/21/2020')) # ValueError: Error parsing datetime string "07/21/2020" at position 2 
# 
# #2-3. 날짜 텍스트 → 날짜 변환(스칼라값) - python 
# import datetime print(datetime.date.fromisoformat('2020-07-21')) 
# # 2020-07-21 print(datetime.datetime.fromisoformat('2020-07-21')) 
# # 2020-07-21 00:00:00 print(datetime.time.fromisoformat('07:01:45')) # 07:01:45



# time_period_transform : 원본 날짜들 → 주기별로 그룹핑 → 각각의 구간에서 원래 값의 최소값을 반환
def time_period_transform(date_time, freq, format=None):
    """
    날짜 시리즈를 특정 주기(freq) 단위로 변환한 뒤,
    각 주기별로 원본 데이터의 최소값을 반환하는 함수.

    Parameters
    ----------
    date_time : pandas.Series 
        날짜 또는 문자열로 구성된 시리즈.
        문자열인 경우 `format`에 따라 datetime으로 변환됨.
    
    freq : str
        그룹핑할 주기.
        pandas Period의 freq 규칙과 동일.
        예: 'Y' (연도), 'Q' (분기), 'M' (월), 'W' (주), 'D' (일) 등
    
    format : str, optional
        data가 문자열일 경우, datetime 파싱을 위한 포맷 문자열.
        예: '%Y-%m-%d', '%Y%m%d', '%Y/%m/%d', ...
        기본값 None이면 pandas가 자동으로 해석.

    Returns
    -------
    pandas.DataFrame
        두 개의 컬럼을 가진 DataFrame을 반환:
        - 첫 번째 컬럼: 각 주기(freq) 그룹에 대한 원본 값의 최소값
        - 두 번째 컬럼: 해당 최소값이 속한 Period (freq 단위)

    Notes
    -----
    - 날짜를 pandas Period로 변환하여 동일한 주기에 속한 값끼리 그룹화함.
    - 각 period별로 원래 date_time 값 중 최소값을 선택하여 반환함.
    - 예를 들어 freq='M'이면, 각 달(month)별로 최소 날짜가 반환됨.

    Examples
    --------
    >>> s = pd.Series(['2024-03-01', '2024-03-15', '2024-04-02'])
    >>> time_period_transform(s, freq='M')
         0       freq
    0  2024-03  2024-03
    1  2024-04  2024-04
    """
    # 1) datetime 변환
    if 'datetime' not in str(date_time.dtype):
        data_time = pd.to_datetime(date_time, format=format)
    else:
        data_time = date_time.copy()

    # 2) 날짜 → Period
    data_period = data_time.apply(lambda x: pd.period_range(x, x, freq=freq)[0])
    data_period.name = 'freq'

    # 3) 원본 데이터 + period 결합
    df = pd.concat([data_period, data_time], axis=1)
    
    value_name = 'value'
    if 'Series' in str(type(date_time)) and date_time.name is not None:
        value_name = date_time.name 
    df.columns = ['freq', value_name]   # 명시적으로 이름 지정

    # 4) 각 period별 최소값의 index를 가져옴 (원본 index 유지)
    idx = df.groupby('freq')[value_name].idxmin()

    # 5) 해당 index의 행만 DataFrame으로 반환
    result = df.loc[idx]
    # result = df.loc[idx].sort_index()
    return result

# rng = pd.date_range(start='2024-01-01', end='2024-06-30', periods=30)   # '24.1/1~'24.6/30 까지 30 구간으로 나눔
# sample = pd.Series(rng, name='time')
# time_period_transform(sample, freq='M')     # 월별로 첫 data를 추출








# datetime_split : datetime 객체(또는 datetime 시리즈)에서 원하는 날짜·시간 성분을 뽑아내고, freq(빈도 단위)에 따라 필요한 성분만 자동 필터링해서 반환하려는 목적
def datetime_split(date_time, date_format=['year','month','weekday','day','hour','second'], freq='S'):
    """
    datetime 객체에서 여러 날짜·시간 구성요소(year, month, weekday, day, hour, minute, second)를
    추출한 뒤, freq(시간 단위) 규칙에 따라 필요한 구성요소만 선택하여 반환하는 함수.

    Parameters
    ----------
    date_time : pandas.Timestamp or pandas.Series (element-wise 적용 가능)
        날짜·시간 정보를 포함하는 Timestamp 객체 또는 datetime-like 객체.
    
    date_format : list of str, optional
        추출할 datetime 속성 이름 목록.
        기본값은 ['year','month','weekday','day','hour','second'].
        사용 가능한 값:
        - 'year', 'month', 'weekday', 'day', 'hour', 'minute', 'second'

    freq : str, {'y','m','w','d','H','M','S'}
        반환할 datetime 구성요소의 수준(granularity)을 결정하는 단위.
        - 'y' : 연도 단위 → ['year']
        - 'm' : 월 단위 → ['year', 'month']
        - 'w' : 주 단위 → ['year', 'month','weekday']
        - 'd' : 일 단위 → ['year', 'month','weekday','day']
        - 'H' : 시 단위 → ['year', 'month','day','weekday','hour']
        - 'M' : 분 단위 → ['year', 'month','day','weekday','hour','minute']
        - 'S' : 초 단위 → ['year', 'month','day','weekday','hour','minute','second']

    Returns
    -------
    dict
        freq 규칙에 의해 선택된 datetime 속성만 포함하는 dictionary.

    Notes
    -----
    - 시계열 feature engineering 목적의 datetime 분해 기능.
    - freq를 높일수록 더 많은 구성요소가 반환됨.
    - pandas.Timestamp 및 pandas datetime-like 객체에 적용 가능.
    """
    freq_map = {
    'y':['year'],'m':['year','month'], 'w':['year','month', 'weekday'], 'd':['year','month','weekday','day'],
    'H':['year', 'month','day','weekday','hour'], 
    'M':['year', 'month','day','weekday','hour','minute'], 
    'S':['year', 'month','day','weekday','hour','minute', 'second'], 
    }
        
    transform_result = {}
    for format in date_format:
        if format  == 'weekday':
            transform = date_time.weekday()
        else:
            transform = eval(f"date_time.{format}")
        transform_result[format] = transform
    
    return dict(filter(lambda e : e[0] in freq_map[freq], transform_result.items()))

# x = pd.Timestamp("2024-02-05 13:28:10")
# datetime_split(x, freq='S')
# datetime_split(x, date_format=['year','day'], freq='S')

# dates = pd.date_range("2024-01-01 12:00:00", periods=5, freq="3H")
# s = pd.Series(dates)
# s.apply(lambda x: datetime_split(x, freq='M'))



# sequential_transform : 시계열 슬라이딩 윈도우(sliding window) 기능을 일반화한 함수
#                       여러 개의 배열/시계열을 동시에, 동일한 슬라이딩 윈도우 범위로 잘라서 “윈도우 단위의 sequence 묶음”을 만들어주는 함수
def sequential_transform(*args, window=1, start=0, stride=1):
    """
    여러 시계열(배열)을 동일한 슬라이딩 윈도우(sequential window)로 분리하여
    window-shaped sequence 데이터를 생성하는 함수.

    Parameters
    ----------
    *args : array-like or pandas.Series
        슬라이딩 윈도우로 분리할 시계열(하나 이상).
        여러 개가 들어올 경우 동일한 인덱스로 window가 적용됨.
    
    window : int, default=1
        슬라이딩 윈도우 길이 (각 sequence 길이).
    
    start : int, default=0
        (현재 코드에서는 사용되지 않지만) 윈도우 시작 offset을 지정하는 용도.
    
    stride : int, default=1
        윈도우 이동 간격 (stride 길이).  
        예: stride=1 → 겹치게 이동, stride=2 → 두 칸씩 이동.

    Returns
    -------
    idx : numpy.ndarray, shape (num_windows, window)
        슬라이딩 윈도우에 해당하는 원본 데이터의 인덱스 조합.

    result : numpy.ndarray or list of numpy.ndarray
        - 입력이 1개인 경우: 단일 3D 배열 (num_windows, window, feature_dim)
        - 입력이 여러 개인 경우: 각 입력마다 동일한 윈도우를 적용한 배열 리스트.
    
    Notes
    -----
    - 1차원 입력은 자동으로 (N,1) 형태로 reshape되어 feature 차원이 유지됨.
    - seq2seq, 시계열 예측, sliding-window feature engineering에 활용 가능.
    - 모든 *args 는 길이가 같아야 한다.

    Examples
    --------
    >>> x = [10, 20, 30, 40, 50]
    >>> idx, seq = sequential_transform(x, window=3, stride=1)
    >>> seq.shape
    (3, 3, 1)
    """
    idx = np.arange(0, len(args[0])-window + 1, step=stride).reshape(-1,1) + np.arange(0,window)
    
    result = []
    for x in args:
        x = np.array(x) if 'pandas' in str(type(x)) else x
        if x.ndim == 1:
            x = x.reshape(-1,1)
        result.append(x[idx,:])
    return idx, result[0] if len(args) == 1 else result

# x = np.array([10, 20, 30, 40, 50])
# idx, seq = sequential_transform(x, window=3, stride=1)



# sequential_filter_index_from_src_index : 이미 만들어진 source 윈도우 인덱스(src_idx)를 기준으로, 
#                                         타깃 구간(target window) 인덱스를 계산하고, 범위를 벗어나는 경우를 필터링하는 함수
def sequential_filter_index_from_src_index(src_idx, trg_window=1):
    """
    source window 인덱스(src_idx)를 기준으로, 지정한 타깃 offset 구간(trg_window)을
    모두 포함할 수 있는 유효한 window만 필터링하여,
    source window 인덱스와 target window 인덱스를 함께 반환하는 함수.

    이 함수는 보통 다음과 같은 시나리오에서 사용된다.
    - src_idx : 과거 구간을 나타내는 입력 시퀀스(window) 인덱스 (예: X window)
    - trg_window : 각 source window의 마지막 시점으로부터 얼마만큼
                   앞/뒤 구간을 target window (예: y window)로 사용할지 정의

    Parameters
    ----------
    src_idx : array-like, shape (num_windows, window_size)
        sequential_transform 등으로 생성된 source window 인덱스 배열.
        각 행은 하나의 window에 해당하고, 열은 해당 window를 구성하는 시점 인덱스를 의미.

    trg_window : int or iterable, default=1
        target window를 정의하는 offset.
        - int인 경우:
            마지막 인덱스 + trg_window 를 단일 target 시점으로 사용.
            예: trg_window=1 → one-step-ahead 예측.
        - iterable (길이 2) 인 경우:
            (low, high) 로 간주하고 [low, ..., high] 범위를 모두 포함하는
            target window를 사용. (두 값 모두 inclusive)
            예: trg_window=(1,3) → 마지막 시점 기준 +1, +2, +3 시점을 타깃 구간으로 사용.

    Returns
    -------
    filtered_idx : numpy.ndarray, shape (N, 1)
        유효한 source window에 대해, 각 window의 기준 인덱스
        (보통 source window의 마지막 시점 인덱스)를 담은 배열.

    filtered_src_idx : numpy.ndarray, shape (N, window_size)
        trg_window를 만족하는 유효한 source window 인덱스만 남긴 배열.

    filtered_trg_idx : numpy.ndarray, shape (N, T)
        각 source window에 대응하는 target window 인덱스.
        여기서 T는 trg_window에 의해 결정되는 target 길이
        (단일 int인 경우 T=1, (low, high)인 경우 T = high - low + 1).

    Notes
    -----
    - 전체 인덱스 범위는 [0, src_idx.max()] 로 가정하고,
      target window의 모든 인덱스가 이 범위를 벗어나지 않는 경우만 남긴다.
    - 보통 시계열 예측에서 (입력 window, 출력 window) 쌍을 만들 때
      source–target 인덱스를 맞추는 용도로 사용된다.
    """
    src_idx = np.asarray(src_idx)

    # trg_window 해석: int → 단일 offset, iterable → [low, high] inclusive
    if hasattr(trg_window, "__iter__") and not isinstance(trg_window, (int, np.integer)):
        win_low = int(trg_window[0])
        win_high = int(trg_window[1])   # inclusive
    else:
        win_low = int(trg_window)
        win_high = int(trg_window)

    offsets = np.arange(win_low, win_high + 1)   # [low, ..., high]
    src_last_idx = src_idx[:, -1]

    min_idx = 0
    max_idx = src_idx.max()

    # target window의 시작/끝이 전체 인덱스 범위를 벗어나지 않는지 체크
    mask = (src_last_idx + offsets[0] >= min_idx) & (src_last_idx + offsets[-1] <= max_idx)

    # 기준 인덱스(보통 source window의 마지막 시점)
    filtered_idx = src_last_idx[mask].reshape(-1, 1)

    # target window 인덱스들
    filtered_trg_idx = filtered_idx + offsets

    # 유효한 source window 인덱스
    filtered_src_idx = src_idx[mask]

    return filtered_idx, filtered_src_idx, filtered_trg_idx


# date_data = np.arange(10)   # [0,1,2,3,4,5,6,7,8,9]

# # 1) 길이 3인 source window 만들기
# src_idx, src_windows = sequential_transform(date_data, window=3, stride=1)

# # 2) 마지막 시점 + 1 을 target으로 쓰기 (one-step-ahead)
# from_src_idx, filtered_src_idx, filtered_trg_idx = sequential_filter_index_from_src_index(
#     src_idx, trg_window=1
# )

# filtered_src_idx
# filtered_trg_idx
# for s_idx, t_idx in zip(filtered_src_idx, filtered_trg_idx):
#     print(f"X idx = {s_idx}, X = {date_data[s_idx]}  -->  y idx = {t_idx}, y = {date_data[t_idx]}")





# ------------------------------------------------------------------------------
# sereis_plot : 시계열(또는 여러 개의 시계열 컬럼)을 한 번에 예쁘게 그려주는 helper 함수
def series_plot(date_time, index=None, columns=None,
                yscale=None, figsize=(15, 3), return_plot=True):
    """
    1개 또는 여러 개의 시계열(Series / DataFrame / numpy 배열)을
    위아래(subplot)로 나누어 한 번에 선 그래프로 그려주는 함수.

    Parameters
    ----------
    date_time : pandas.Series, pandas.DataFrame, or array-like
        - 시계열 데이터.
        - Series인 경우: 단일 시계열로 처리되며, name 속성이 있으면 legend label로 사용.
        - DataFrame인 경우: 각 컬럼을 별도의 subplot에 그린다.
        - numpy 배열 등 array-like인 경우: shape (N,) 또는 (N, K)를 허용.
    
    index : array-like, optional
        x축에 사용할 인덱스(시간축). 기본값은:
        - pandas 객체인 경우: 해당 객체의 index
        - 그 외: 0, 1, 2, ... 의 정수 인덱스

    columns : list of str, optional
        각 시리즈에 대한 label 목록. 기본값은:
        - Series: [series.name] 또는 ['series']
        - DataFrame: list(date_time.columns)
        - numpy 배열: None (label 없이 그림)

    yscale : {'linear', 'log', ...}, optional
        y축 스케일. Matplotlib의 set_yscale 옵션과 동일.
        예: 'linear', 'log', 'symlog', 'logit' 등.

    figsize : tuple, default=(15, 3)
        전체 figure의 가로, 세로 크기 (단위: inch).
        세로 크기는 시리즈 개수에 비례해서 자동으로 곱해 사용된다.

    return_plot : bool, default=True
        - True : Matplotlib Figure 객체를 반환하고, 내부에서 plt.close(fig)를 호출.
                 (노트북 등에서 수동으로 표시하거나 저장할 때 유용)
        - False: 즉시 plt.show()로 그림을 화면에 표시하고 None을 반환.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        - return_plot=True 인 경우: 생성된 Figure 객체
        - return_plot=False 인 경우: None

    Notes
    -----
    - 여러 컬럼을 가진 시계열을 빠르게 시각적으로 비교할 때 쓸 수 있는 간단한 wrapper.
    - subplot은 각 시리즈별로 하나씩 생성되며, x축은 공유(sharex=True)한다.
    - pandas 객체를 넣을 때 별도의 index/columns를 지정하지 않으면 자동으로 가져온다.
    """
    # 1) pandas 객체 처리 (Series / DataFrame)
    if isinstance(date_time, (pd.Series, pd.DataFrame)):
        if isinstance(date_time, pd.Series):
            value = date_time.to_numpy().reshape(-1, 1)
            if index is None:
                index = date_time.index
            if columns is None:
                columns = [date_time.name if date_time.name is not None else "series"]
        else:  # DataFrame
            value = date_time.to_numpy()
            if index is None:
                index = date_time.index
            if columns is None:
                columns = list(date_time.columns)
    else:
        # pandas가 아니면 그냥 numpy로 변환
        value = np.asarray(date_time)
        if value.ndim == 1:
            value = value.reshape(-1, 1)
        # index, columns은 사용자가 직접 넣지 않으면 None으로 둠

    n_series = value.shape[1]

    # 2) figure / axes 생성
    fig, axes = plt.subplots(n_series, 1,
                            figsize=(figsize[0], figsize[1] * n_series),
                            sharex=True)
    

    # axes를 배열 형태로 맞춰주기
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    # x축 값
    if index is None:
        x = np.arange(value.shape[0])
    else:
        x = index

    # 3) 각 시리즈 플롯
    for i in range(n_series):
        ax = axes[i]
        label = None if columns is None else columns[i]
        ax.plot(x, value[:, i], label=label, alpha=0.5)
        if label is not None:
            ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
        if yscale is not None:
            ax.set_yscale(yscale)

    plt.tight_layout()

    if return_plot:
        # 노트북에서 자동 표시를 막고 fig만 리턴할 경우 close
        plt.close(fig)
        return fig
    else:
        # 즉시 화면에 보여주고 아무것도 리턴하지 않음
        plt.show()
        
# (Example1) 30일짜리 랜덤 데이터
# idx = pd.date_range("2024-01-01", periods=30, freq="D")
# s = pd.Series(np.random.randn(30).cumsum(), index=idx, name="random_walk")

# series_plot(s, figsize=(12, 3), return_plot=True)


# # (Example2) 3개의 시계열 컬럼을 가진 DataFrame
# df = pd.DataFrame({
#     "temp": 20 + np.random.randn(30).cumsum(),
#     "humidity": 50 + np.random.randn(30).cumsum(),
#     "pressure": 1000 + np.random.randn(30).cumsum()
# }, index= pd.date_range("2024-01-01", periods=30, freq="D"))

# series_plot(df, figsize=(12, 3), return_plot=True)
# ------------------------------------------------------------------------------







# # example_code
# import sys
# sys.path.append(r'C:\Users\Admin\Desktop\DataScience\★★ DS_Library')

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# from DS_DeepLearning import shape
# from DS_TimeSeries import series_plot, sequential_transform, sequential_filter_index_from_src_index


# # (example) ---------------------------------------------------------------------
# # https://engineer-mole.tistory.com/239
# # np.put()
# # np.place()
# # np.putmask()

# df1 = pd.DataFrame()
# df1['date'] = pd.date_range('2021-01-01', '2021-01-31')
# # df1['date'] = pd.date_range('2021-01-01', '2021-01-10')
# df1['value'] = range(len(df1['date']))
# df1['rand'] = np.random.rand(len(df1['date']))
# df1['target'] = np.arange(len(df1['date']))+100
# # d1['date'] = d1['date'].astype('str').apply(lambda x: x.replace('-',''))
# # ------------------------------------------------------------------------------

# y_col = 'target'
# # X_cols = ['value','rand']
# X_cols = ['target']

# df_anal = df1.set_index('date')

# # series_plot
# series_plot(df_anal, return_plot=False)    # Graph


# # sequential_transform: Split stacked data ***
# idx1, X_stack = sequential_transform( df_anal[X_cols], window=10)
           
# print(idx1.shape, X_stack.shape)   ## (22, 10), (22, 10, 2)


# # sequential_filter_index_from_src_index
# # idx2, src_idx, trg_idx = sequential_filter_index_from_src_index(idx1)
# idx2, src_idx, trg_idx = sequential_filter_index_from_src_index(idx1, trg_window=(-3,3))
# print(idx2.shape, src_idx.shape, trg_idx.shape)

# # data filter
# X = np.array(df_anal[X_cols])[src_idx]
# y = np.array(df_anal[y_col])[trg_idx]


# date_index_np = df_anal.index.to_numpy()   # 또는 .values
# time_index        = date_index_np[idx2.ravel()]   # (N,)
# time_index_Xmatrix = date_index_np[src_idx]       # (N, window)
# time_index_ymatrix = date_index_np[trg_idx]       # (N, target_horizon=5

# # time_index = df_anal.index[idx2.ravel()]
# # time_index_Xmatrix = df_anal.index[src_idx]
# # time_index_ymatrix = df_anal.index[trg_idx]


# print(X.shape, y.shape, time_index.shape, time_index_Xmatrix.shape, time_index_ymatrix.shape)
# # ((20, 10, 2), (20, 5), (20, 1), (20, 10), (20, 5))
# # pd.DataFrame(X[:,:,0]).to_clipboard(index=False,header=False)
# # pd.DataFrame(y).to_clipboard(index=False,header=False)
# # pd.DataFrame(time_index_ymatrix).to_clipboard(index=False,header=False)


# # Graph ***
# text_dict = {'0_start':(0,0), '0_end':(0,-1), '-1_start':(-1,0), '-1_end':(-1,-1)}

# plt.figure(figsize=(15,8))
# plt.subplot(3,1,1)
# plt.plot(df_anal.index, df_anal[y_col], color='mediumseagreen', marker='o')
# plt.plot(time_index_ymatrix[:,0], y[:,0], color='red',alpha=0.5)
# plt.plot(time_index_ymatrix[:,-1], y[:,-1], color='red',alpha=0.5)
# for name, point in text_dict.items():
#     plt.text(time_index_ymatrix[point[0],point[1]], y[point[0], point[1]], f"↓ {name}")
# for e,c in enumerate(df_anal[X_cols].columns):
#     plt.subplot(3,1,e+2)
#     plt.plot(df_anal.index, df_anal[c], color='steelblue', marker='o')
#     plt.plot(time_index, X[:,-1,e], color='gold')
# plt.show()


# # Train_Test_Split
# from sklearn.model_selection import train_test_split
# test_size = 0.2
# X_train, X_test, y_train, y_test, train_index, test_index = train_test_split(X, y, time_index, test_size=test_size, shuffle=False)

# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape, train_index.shape, test_index.shape)
# print(train_index[[0,-1]], test_index[[0, -1]])







###########################################################################################################







# 마스킹된 시계열 데이터에 대해 롤링 평균을 적용하여 결측값을 보간하는 함수
def smoothing(x, mask, window=3, min_periods=1, center=True, agg='mean', **kwargs):
    """
    Apply smoothing to masked values in a Pandas Series or DataFrame using a rolling window.

    This function replaces values that do not satisfy the given mask with NaN,
    computes a rolling aggregation (default: mean) over the masked data, and then
    interpolates the masked positions with the aggregated values while keeping
    unmasked positions unchanged.

    Parameters
    ----------
    x : pandas.Series or pandas.DataFrame
        Input data to be smoothed. 
        (처리할 원본 시계열 데이터)
    mask : pandas.Series or pandas.DataFrame of bool
        Boolean mask indicating positions to keep (`True`) and positions to replace (`False`).
    window : int, default 3
        Size of the moving window for rolling aggregation.
    min_periods : int, default 1
        Minimum number of observations in the window required to have a value.
    center : bool, default True
        Set the labels at the center of the window.
    agg : str, default 'mean'
        Aggregation function to apply on the rolling window (e.g., 'mean', 'sum', 'max').
    **kwargs
        Additional keyword arguments passed to the rolling aggregation.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        Smoothed data with masked positions replaced by rolling aggregated values.

    Notes
    -----
    - The function uses `DataFrame.where` to apply the mask and replace non-matching
      positions with NaN.
    - Rolling aggregation is applied only to masked positions.
    - Unmasked positions retain their original values.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> data = pd.Series([1, 2, 10, 4, 5])
    >>> mask = data < 9
    >>> smoothing(data, mask, window=3)
    0    1.0
    1    2.0
    2    2.333333
    3    4.0
    4    5.0
    dtype: float64
    """
    mask_x = x.where(mask, other=np.nan).rolling(window=window, min_periods=min_periods, center=center, **kwargs).agg(agg)
    interpolate_x = x.where(mask, other=np.nan).fillna(0) + mask_x.where(~mask, other=np.nan).fillna(0)
    return interpolate_x




# 서로 다른 길이의 1D 시계열을 최대 길이에 맞춰 padding.
def pad_series_list_1d(series_list, pad_value=np.nan):
    """
    서로 다른 길이의 시계열을 최대 길이에 맞춰 zero padding.
    Args:
        series_list (list[np.ndarray]): 각 시계열 (길이 다름)
        pad_value (float): 패딩값 (기본 0)
    Returns:
        np.ndarray: shape = (N, max_len)
    """
    max_len = max(len(s) for s in series_list)
    padded = np.full((len(series_list), max_len), pad_value, dtype=float)
    for i, s in enumerate(series_list):
        padded[i, :len(s)] = s
    return padded

# 서로 다른 길이의 2D 시계열을 최대 길이에 맞춰 padding.
def pad_series_list_2d(series_list, pad_value=np.nan):
    """
    서로 다른 길이의 2D 시계열을 최대 길이에 맞춰 padding.
    Args:
        series_list (list[np.ndarray]): 각 시계열 (shape: features × time_length)
        pad_value (float): 패딩값
    Returns:
        np.ndarray: shape = (N, features, max_len)
    """
    # features 수는 동일하다고 가정
    features = series_list[0].shape[0]
    max_len = max(s.shape[1] for s in series_list)
    
    padded = np.full((len(series_list), features, max_len), pad_value, dtype=float)
    for i, s in enumerate(series_list):
        padded[i, :, :s.shape[1]] = s
    return padded







###########################################################################################################
###########################################################################################################
###########################################################################################################
# str → date → number → date → str
class DatetimeHandler():
    """
    【required (Library)】datetime, dateutil, numpy, pandas, collections.namedtuple
    """
    def __init__(self, date=None, base_time='1899-12-30 00:00:00', date_compile='\d*', sep_compile='\W*'):
        self.re_date = re.compile(date_compile)
        self.re_str = re.compile('[a-zA-Z]')
        self.re_sep = re.compile(sep_compile)

        self.date = self.str_to_date(date) if date is not None else None

        self.split_date = self.str_to_date(base_time, return_type='all')
        self.base_date = self.split_date['date']

        self.datetime_format = {'Date': '%Y%m%d', 'date': '%y%m%d',
                'DateHour': '%Y%m%d %H', 'dateHour': '%y%m%d %H',
                'Datetime':'%Y%m%d %H:%M:%S', 'datetime':'%y%m%d %H:%M:%S'}

    # datetime format union
    def union_to_datetime(self, date):
        if 'datetime.datetime' in str(type(date)) :
            return date
        elif 'pandas' in str(type(date)) or 'numpy' in str(type(date)):
            return pd.Series([date])[0].to_pydatetime()

    # (excel) date → numberize
    def to_number(self, datediff):
        return datediff.days + datediff.seconds/(24*60*60)

    # string → date  
    def str_to_date(self, date_str, return_type='date', thousand_criteria=30, strp_format=None):
        if str(date_str)[-2:] == '.0':
            date_str = str(date_str)[:-2]
        else:
            date_str = str(date_str)

        if strp_format is not None:
            return datetime.datetime.strptime(date_str, strp_format)

        date_instance = namedtuple('date_info', ['date', 'sep', 'strf_format'])

        date_list = list(filter(lambda x: x != '', self.re_date.findall(date_str)))
        str_list = list(filter(lambda x: x != '', self.re_str.findall(date_str)))
        sep_list = list(filter(lambda x: x != '', self.re_sep.findall(date_str)))

        str_list = list(filter(lambda x: x not in sep_list, str_list))
        if str_list:
            raise ValueError('DateTime cannot contains string.')
        
        if len(date_list) == 1 and len(sep_list) == 0:
            date_string = date_list[0]
            if len(date_string) < 4:
                raise ValueError('DateTime requies Year/Month at least.')
            elif len(date_string) == 8:
                date_list = [date_string[:4], date_string[4:6], date_string[6:]]
                sep_list = [''] * 2
            elif len(date_string) == 6:
                date_list = [date_string[:2], date_string[2:4], date_string[4:]]
                sep_list = [''] * 2
            elif len(date_string) == 4:
                date_list = [date_string[:2], date_string[2:], '01']
                sep_list = [''] * 2
            

        str_format = '%Y' if len(date_list[0]) == 4 else '%y'
        format_all = [str_format] + ['%m', '%d', '%H', '%M', '%S']
        format_list = format_all[:len(date_list)]

        for s, f in zip(sep_list, format_list[1:]):
            str_format += (s + f)
        
        breakN = 0
        while len(date_list) < 6:
            len_date_list = len(date_list)
            # format_list += [format_all[len_date_list]]
            if len_date_list < 3:
                date_list += ['01']
            else:
                date_list += ['00']
            
            breakN += 1
            if breakN >= 30:
                break
                
        if len(date_list[0]) == 2:
            if int(date_list[0]) >= int(str(datetime.datetime.now().year + thousand_criteria)[2:]):
                date_list[0] = '19' + date_list[0]
            else:
                date_list[0] = '20' + date_list[0]
        
        date_result = datetime.datetime(*map(int, date_list))
        
        breakN = 0
        while len(sep_list) < 5:
            len_sep_list = len(sep_list)
            if len_sep_list < 2:
                sep_list += sep_list
            elif len_sep_list < 3:
                sep_list += [' ']
            else:
                sep_list += [':']
            
            breakN += 1
            if breakN >= 30:
                break
        
        
        date_object = date_instance(date_list, sep_list, str_format)
        self.object_dict = {'date':  date_result, 'object': date_object}
        if return_type == 'date':
            return date_result
        elif return_type == 'object':
            return date_object
        elif return_type == 'all':
            return self.object_dict

    # date → string
    def date_to_str(self, o, strf_format=None):
        if 'date_info' in str(type(o)):
            if strf_format is None:
                return ''.join(([d+s for d, s in zip(o.date[:-1], o.sep)] + [o.date[-1]]))
            else:
                return o['date'].strftime(strf_format)
        elif type(o) == dict and 'date' in o.keys() and 'object' in o.keys():
            if sum(['date_info' in str(type(v)) for v in o.values()]) > 0:
                if strf_format is None:
                    strf_format = o['object'].strf_format
            else:
                if strf_format is None:
                    strf_format = '%Y-%m-%d %H:%M:%S'    
            return o['date'].strftime(strf_format)
        elif 'time' in str(type(o)):
            if strf_format is None:
                strf_format = '%Y-%m-%d %H:%M:%S'
            return o.strftime(strf_format)

    # number → date
    def number_to_date(self, date_number, strp_format=None):
        if 'series' in str(type(date_number)).lower():
            return date_number.apply(lambda x: self.base_date + datetime.timedelta(days=x))
        else:
            date_string = str(date_number)
            if strp_format is not None:
                return datetime.datetime.strptime(date_string, strp_format)
            elif '.' not in date_string and len(date_string) == 8:
                return datetime.datetime.strptime(date_string, '%Y%m%d')
            elif '.' not in date_string and len(date_string) == 6:
                return datetime.datetime.strptime(date_string, '%y%m%d')
            elif '.' not in date_string and len(date_string) == 4:
                return datetime.datetime.strptime(date_string, '%y%m')
            else:
                return self.base_date + datetime.timedelta(days=date_number)

    # date → number
    def date_to_number(self, date=None):
        if date is None:
            date = self.date
        else:
            self.date = self.str_to_date(date) if type(date) == str else date

        str_dtype = str(type(self.date))
        if 'series' in str_dtype.lower():
            if 'date' not in str(self.date.dtype):
                raise ValueError("Only require date type data")
            else:
                return (self.date - self.base_date).apply(lambda x: self.to_number(x))
        elif 'time' not in str_dtype and 'date' not in str_dtype:
            raise ValueError("Only require date type data")
        else:
            return self.to_number(self.date - self.base_date)

    # transform (string ↔ date ↔ number)
    def transform(self, date, return_type=None, apply_format=None, thousand_criteria=30, delta=None, strp_format=None):
        # input foramt ***
        str_type = str(type(date))
        if 'str' in str_type:
            input_type = 'str'
            if strp_format is None:
                date_object = self.str_to_date(date, return_type='all', thousand_criteria=thousand_criteria)
                self.date = date_object['date']
            else:
                date_object = None
                self.date = self.str_to_date(date, return_type='date', thousand_criteria=thousand_criteria, strp_format=strp_format)
        elif 'date' in str_type or 'time' in str_type:
            input_type = 'date'
            self.date = self.union_to_datetime(date)
        elif 'int' in str_type or 'float' in str_type:
            input_type = 'number'
            self.date = self.number_to_date(date, strp_format=strp_format)
        elif 'dict' in str_type:
            input_type = 'object'
            self.date =  date['date']

        # timedelta
        if delta is not None:
            if 'delta' in str(type(delta)):
                self.date += delta
            elif type(delta) == dict:
                self.date += dateutil.relativedelta.relativedelta(**delta)

        
        # string type return foramt ***
        return_format = apply_format

        if apply_format is None:
            if input_type == 'str' and date_object is not None:
                apply_format = date_object['object'].strf_format
            else:
                apply_format = '%Y-%m-%d %H:%M:%S'
        elif sum([apply_format[:-1] in k for k in self.datetime_format.keys()]) > 0:
            if apply_format not in self.datetime_format.keys():
                strf_sep = apply_format[-1]
                apply_format = self.datetime_format[apply_format[:-1]]
                apply_format = apply_format[:2] + strf_sep + apply_format[2:4] + strf_sep + apply_format[4:]
            else:
                apply_format = self.datetime_format[apply_format]
        elif '%' not in apply_format:
            apply_format = apply_format.replace('Y', '%Y').replace('y', '%y').replace('m', '%m').replace('d', '%d').replace('H', '%H').replace('M','%M').replace('S','%S')

        # apply_format ***
        if return_type is None:
            return_type = input_type
        
        if 'date' in return_type:
            if return_format is not None:
                self.date = datetime.datetime.strptime(self.date.strftime(apply_format), apply_format)
            return self.date
        elif 'num' in return_type:
            if return_format is not None:
                self.date = datetime.datetime.strptime(self.date.strftime(apply_format), apply_format)
            return self.date_to_number(self.date)
        else:
            date_string = self.date.strftime(apply_format)
            
            if return_type == 'str':
                return date_string
            else:
                return self.str_to_date(date_string, return_type=return_type, thousand_criteria=thousand_criteria)

    # generate random date vector
    def randdate(self, start, end, size=1, freq=None, return_type=None, strf_format=None):
        start_date = self.str_to_date(start) if 'date' not in str(type(start)) else start
        end_date = self.str_to_date(end) if 'date' not in str(type(start)) else end

        if type(start) == str:
            start_object = self.str_to_date(start, return_type='all')
            strf_format = start_object['object'].strf_format if strf_format is None else strf_format
            if return_type is None:
                return_type = 'str'
        elif type(end) == str:
            end_object = self.str_to_date(end, return_type='all')
            strf_format = end_object['object'].strf_format if strf_format is None else strf_format
            if return_type is None:
                return_type = 'str'

        if strf_format is None:
            strf_format = '%Y-%m-%d %H:%M:%S'

        date_diff = end_date - start_date

        if freq is None:
            if  date_diff.days // (365*5) > 0:
                freq = 'Y'
            elif  date_diff.days // (30*5) > 0:
                freq = 'M'
            elif  date_diff.days // (5) > 0:
                freq = 'D'
            elif  date_diff.seconds // (60*60*5) > 0:
                freq = 'H'
            elif  date_diff.seconds // (60*5) > 0:
                freq = 'min'
            else:
                freq = 'S'
        
        date_range = pd.date_range(start_date, end_date, freq=freq)
        result = np.random.choice(date_range, size)

        if return_type == 'date' or return_type == 'datetime' or return_type is None:
            return np.array(list(map(lambda x: datetime.datetime.utcfromtimestamp(int(x) * (1e-9)), result)))
        elif return_type == 'str':
            return np.array(list(map(lambda x: datetime.datetime.utcfromtimestamp(int(x) * (1e-9)).strftime(strf_format), result)))   
        elif return_type == 'numpy':
            return result

    # valid datetime vector
    def is_datetime(self, x, valid_number_range=[{'years':-20}, {'years':+10}], possible_formats=[], return_transform=False):
        isdatetime = None
        if 'datetime' in str(np.array(x).dtype):
            isdatetime = True
            return isdatetime
        if pd.Series(x).isna().all():
            return False
        
        # dataset ***
        date_now = datetime.datetime.now()

        series_x = pd.Series(x)
        series_x_drop_duplicates = series_x.drop_duplicates()
        try:
            minmax_x = series_x.dropna().sort_values().iloc[[0,-1]]
        except:
            minmax_x = series_x.dropna().apply(lambda x: str(x)).sort_values()
        transformed_x = series_x

        if return_transform:
            apply_sereis = series_x
        else:
            apply_sereis = minmax_x
        
        # validation possible transform ***
        re_date = series_x_drop_duplicates.dropna().astype('str').apply(lambda x: len([v for v in self.re_date.findall(x) if v != '']))
        re_sep = series_x_drop_duplicates.dropna().astype('str').apply(lambda x: len([v for v in self.re_sep.findall(x) if v != '']))

        try:
            series_x.astype('float')
            possible_float = True
        except:
            possible_float = False
        contain_special = True if re_sep.sum() > 0 else False
        
        # print(possible_float, contain_special)
        # print(re_date.max(), re_date.min(), re_sep.max(), re_sep.min())
        if re_date.max() != re_date.min() or re_sep.max() != re_sep.min():      # dropna dtype confirm
            isdatetime = False
        else:
            try:
                lower_valid_date = date_now + dateutil.relativedelta.relativedelta(**valid_number_range[0])
                upper_valid_date = date_now + dateutil.relativedelta.relativedelta(**valid_number_range[1])
                lower_valid_number = self.date_to_number(lower_valid_date) 
                upper_valid_number = self.date_to_number(upper_valid_date)

                if possible_float is True:
                    # 44407 (excel numeric)
                    if float(series_x.min()) >= lower_valid_number and float(series_x.min()) <= upper_valid_number \
                        and float(series_x.max()) >= lower_valid_number and float(series_x.max()) <= upper_valid_number:
                        transformed_x = apply_sereis.apply(lambda x: np.nan if pd.isna(x) else self.number_to_date(float(x)) )
                        isdatetime = True

                    # 210807 (numeric)
                    elif self.str_to_date(series_x.min()) >= lower_valid_date and self.str_to_date(series_x.min()) <= upper_valid_date \
                        and self.str_to_date(series_x.max()) >= lower_valid_date and self.str_to_date(series_x.max()) <= upper_valid_date:
                        # print('b', self.str_to_date(series_x.min()), self.str_to_date(series_x.max()), lower_valid_date, upper_valid_date)
                        transformed_x = apply_sereis.apply(lambda x: np.nan if pd.isna(x) else self.str_to_date(str(x), return_type='date'))
                        isdatetime = True
                    else:
                        isdatetime = False
                    
                elif possible_float is False and contain_special is True:      # 2021/08/01 (string)
                    transformed_x = apply_sereis.apply(lambda x: np.nan if pd.isna(x) else self.str_to_date(str(x), return_type='date'))
                    # print('c', transformed_x.min(), transformed_x.max(), lower_valid_date, upper_valid_date)
                    if transformed_x.min() >= lower_valid_date and transformed_x.min() <= upper_valid_date \
                        and transformed_x.max() >= lower_valid_date and transformed_x.max() <= upper_valid_date:
                        isdatetime = True
                    else:
                        isdatetime = False
                else:
                    isdatetime = False
            except:
                if possible_formats:
                    if type(possible_formats) == list:
                        for f in possible_formats:
                            try:
                                transformed_x = apply_sereis.apply(lambda x: np.nan if pd.isna(x) else datetime.datetime.strptime(x, f))
                                isdatetime = True
                            except:
                                isdatetime = False
                    elif type(possible_formats) == str:
                        try:
                            transformed_x = apply_sereis.apply(lambda x: np.nan if pd.isna(x) else datetime.datetime.strptime(x, possible_formats))
                            isdatetime = True
                        except:
                            isdatetime = False
                else:
                    isdatetime=False
        
        if return_transform:
            if type(x) == list:
                return (isdatetime, transformed_x.tolist())
            elif 'series' in str(type(x)).lower():
                return (isdatetime, transformed_x)
            elif 'ndarray' in str(type(x)).lower():
                return (isdatetime, np.array(transformed_x))
            else:
                return (isdatetime, transformed_x.iloc[0])
        else:
            return isdatetime
    
    # infer datetime (from series, ndarray, list)
    def infer_datetime(self, x, valid_number_range=[{'years':-20}, {'years':+10}], possible_formats=[]):
        if pd.Series(x).isna().all():
            return x

        valid_datetime = self.is_datetime(x, valid_number_range=valid_number_range,
                    possible_formats=possible_formats, return_transform=False)

        if valid_datetime:
            return self.is_datetime(x, valid_number_range=valid_number_range,
                    possible_formats=possible_formats, return_transform=True)
        else:
            return x









# self_operation : 이전 값에 자기 자신을 누적해서 연산하는 1D 누적 연산 함수 (주로 누적합)
# @numba.njit
def self_operation(x, operator='+', init='first'):
    """
    1차원 배열 x에 대해, 앞에서부터 자기 자신과 누적 연산을 수행하는 함수.

    result[0] = x[0] (init='first') 또는 지정한 init 값
    result[i] = result[i-1] (operator) x[i]

    Parameters
    ----------
    x : array-like
        1차원 입력 데이터.

    operator : {'+','-','*','/'}, default '+'
        누적에 사용할 이항 연산자.

    init : {'first', scalar}, default 'first'
        - 'first' : result[0] = x[0] 에서 시작.
        - 숫자    : result[0] = init 값에서 시작.

    Returns
    -------
    result : numpy.ndarray
        x와 같은 길이의 누적 연산 결과 배열.
    """
    x_np = np.asarray(x)
    if x_np.ndim != 1:
        x_np = x_np.ravel()

    result = np.empty_like(x_np)
    if init == 'first':
        result[0] = x_np[0]
    else:
        result[0] = init

    op = operator

    if op == '+':
        for i in range(1, len(x_np)):
            result[i] = result[i-1] + x_np[i]
    elif op == '-':
        for i in range(1, len(x_np)):
            result[i] = result[i-1] - x_np[i]
    elif op == '*':
        for i in range(1, len(x_np)):
            result[i] = result[i-1] * x_np[i]
    elif op == '/':
        for i in range(1, len(x_np)):
            result[i] = result[i-1] / x_np[i]
    else:
        raise ValueError(f"Unsupported operator: {operator}")

    return result

# arr = np.array([1, 0, 0, 1, 0, 1, 1])
# print("원본:", arr)
# print("누적합:", self_operation(arr, operator='+'))


# Trend Analysis Class : 시계열을 L2(HP filter) / L1 trend filter로 분해하고, trend의 기울기/전환점/구간 그룹을 분석해서 요약해주는 클래스
class TrendAnalysis():
    """
    시계열 데이터를 L2(HP filter) 또는 L1 trend filter로 분해하고,
    추세(trend)의 기울기 및 전환점 정보를 분석하는 클래스.
    
    【 Required Library 】
    import statsmodels.api as sm  (sm.tsa.filters.hpfilter)
    
    【 self.hp_filter 】
     . lamb=1600 : The Hodrick-Prescott smoothing parameter. 
        (suggesting) month: 129600, quarter: 1600, year: 6.25 
    
    주요 기능
    --------
    1) 필터링
       - hp_filter (L2): Hodrick-Prescott filter
         sm.tsa.filters.hpfilter(x, lamb)를 래핑.
       - L1_filter (L1): 2차 차분(두 번째 미분)에 대한 L1 penalty를 주는
         trend filtering 문제를 cvxpy로 푸는 형태.

    2) trend_slope
       - rolling window를 이용해 trend의 시작/끝 평균 차이를 이용한
         국소 기울기(local slope)를 계산.

    3) trend_info
       - trend_slope의 부호 변화를 기준으로 각 시점을
         'up', 'down', 'max', 'min', 'keep' 등으로 라벨링.

    4) trend_group (L1 전용)
       - trend_slope 변화량이 일정 허용 오차(ptolfloat)를 넘는 지점을
         trend change point로 보고, 구간별 그룹 번호를 부여.
       - 각 그룹별 평균 slope 및 특성을 요약 (plus_max, minus_max 등).

    5) fit
       - 위의 과정들을 한 번에 실행하고, summary DataFrame과
         그룹별 요약(group_summary) 등을 저장.

    Parameters
    ----------
    x : pandas.Series or array-like, optional
        분석할 시계열 데이터. 보통 pandas.Series를 가정.

    filter : {'hp_filter', 'L2', 'L1'}, default 'hp_filter'
        사용할 필터 종류.
        - 'hp_filter', 'L2' : Hodrick-Prescott filter
        - 'L1'             : L1 trend filter (cvxpy 기반)

    rolling : int, default 2
        trend_slope 계산 시 rolling window 크기.

    **kwargs :
        필터 및 내부 메서드에 전달할 추가 파라미터.
        대표적으로:
        - lamb : HP filter / L1 filter의 lambda 파라미터.

    Attributes
    ----------
    x : pandas.Series
        입력 시계열 (내부에 저장된 최신 데이터).

    cycle : pandas.Series
        필터링 결과의 cycle(순환) 성분.

    trend : pandas.Series
        필터링 결과의 trend(추세) 성분.

    trend_slope_ : pandas.Series
        rolling window 기반으로 계산된 trend의 국소 기울기.

    trend_info_ : pandas.Series
        각 시점별 추세 정보 ('up', 'down', 'max', 'min', 'keep' 등).

    trend_change_ : pandas.Series (L1 전용)
        trend_slope 변화가 기준을 넘는 구간의 change point 마스크 ('point' / NaN).

    trend_group_ : pandas.Series (L1 전용)
        change point를 기준으로 구간 그룹 번호.

    summary : pandas.DataFrame
        원 시계열, cycle, trend, trend_slope, trend_info,
        (필요 시 trend_change, trend_group)를 한 번에 모은 요약 테이블.

    group_summary : pandas.DataFrame (L1 전용)
        trend_group별 평균 slope와 그에 대한 특성(minus_max, plus_max 등).

    Notes
    -----
    - hp_filter 사용 시 statsmodels.api(sm)이 필요하다.
    - L1_filter 사용 시 cvxpy, scipy.sparse, pandas, numpy 등이 필요하다.
    - L1 필터는 최적화 문제를 푸는 것이므로 데이터 길이가 길면 계산 비용이 커질 수 있다.
    """
    def __init__(self, x=None, filter='hp_filter', rolling=2, **kwargs):
        self.x = x
        self.rolling = rolling
        
        self.cycle = None
        self.trend = None
        self.trend_slope_ = None
        self.trend_info_ = None
        
        self.params = {}
        self.params.update(kwargs)
        self.filter_result = None
        
        self.filter = filter
        
        if x is not None:
            if filter in ['L1', 'L2', 'hp_filter']:
                self.filter_result = self.analysis(x=self.x, filter=self.filter, **self.params)
    
    # 【 filters 】
    def analysis(self, x=None, filter=None, **kwargs):
        self.params.update(kwargs)

        x = self.x if x is None else x
        filter = self.filter if filter is None else filter
        
        # filters
        if filter == 'hp_filter' or filter == 'L2':
            lamb = self.params['lamb'] if 'lamb' in self.params.keys() else 1600
            self.cycle, self.trend = self.hp_filter(x, lamb)
            self.filter_result = (self.cycle, self.trend)
        elif filter == 'L1':
            lamb = self.params['lamb'] if 'lamb' in self.params.keys() else 1600
            self.cycle, self.trend = self.L1_filter(x, lamb)
            self.filter_result = (self.cycle, self.trend)
        return self.filter_result
    
    def hp_filter(self, x=None, lamb=None, save_params=False):
        x = self.x if x is None else x
        lamb = (self.params['lamb'] if 'lamb' in self.params.keys() else 1600) if lamb is None else lamb
        
        if save_params:
            self.x = x
            self.params['lamb'] = lamb
        
        if x is not None:
            return sm.tsa.filters.hpfilter(x, lamb)

    def L1_filter(self, x=None, lamb=1600):
        # x info
        if 'pandas' in str(type(x)):
            if x.ndim > 1 and x.shape[1] > 1:
                raise ValueError('shape of x is allowed (n,) or (n,1).')
            input_type = 'pandas'
            index = x.index
            name = x.name if x.ndim == 1 else x.columns[0]
        else:
            input_type = 'numpy'

        # preparation
        x_np = np.array(x).ravel()
        n = x_np.size

        ones_row = np.ones((1, n))
        Dx = sp.sparse.spdiags(np.vstack((ones_row, -2*ones_row, ones_row)), range(3), n-2, n)
        
        # solver = cvxpy.CVXOPT   # L2
        solver = cvxpy.ECOS     # L1
        reg_norm = 1
        
        # solve
        x_fit = cvxpy.Variable(shape=n) 
        objective = cvxpy.Minimize(0.5 * cvxpy.sum_squares(x_np - x_fit) 
                        + lamb * cvxpy.norm(Dx @ x_fit, reg_norm))
        problem = cvxpy.Problem(objective)
        problem.solve(solver=solver, verbose=False)
        
        # summary
        x_fit_np = np.array(x_fit.value)
        x_fit_cycle_np = x_np - x_fit_np
        if input_type == 'pandas':
            x_fit_series = pd.Series(x_fit_np, index=index, name=name).astype(float)
            x_fit_cycle_series = pd.Series(x_fit_cycle_np, index=index, name=name).astype(float)
            return (x_fit_cycle_series, x_fit_series)
        else:
            return (x_fit_cycle_np, x_fit_np)
    
    # 【 trend_slope 】
    def calc_trend_slope(self, trend):
        return trend.tail(1).mean() - trend.head(1).mean()   
                    
    def trend_slope(self, x=None, trend=None, filter=None, rolling=None, **kwargs):
        self.params.update(kwargs)
        input_x = self.x if x is None else x
        input_trend = self.trend if trend is None else trend
        input_filter = self.filter if filter is None else filter
        input_rolling = self.rolling if rolling is None else rolling
        
        if input_x is None:
            if input_trend is None:
                raise('x, or trend is nessasary for working.')
        else:
            if input_trend is None:
                self.analysis(x=input_x, filter=input_filter, **self.params)
                input_trend = self.trend

        self.trend_slope_ = input_trend.rolling(input_rolling).agg(self.calc_trend_slope)
        return self.trend_slope_

    # 【 trend_info 】
    def calc_trend_info(self, trend_slope):
        trend_slope_shift = trend_slope.iloc[1:]
        info_list = []
        now_sign = trend_slope_shift.iloc[0] > 0
        for e, i in enumerate(trend_slope_shift):
            if i != 0:
                i_sign = i > 0
                if now_sign != i_sign:
                    if now_sign:
                        info_list.append('max')
                    else:
                        info_list.append('min')
                    now_sign = i_sign
                else:
                    info_list.append('up' if i_sign else 'down')
            else:
                info_list.append('keep')
        info_list.append('')

        return pd.Series(info_list, index=trend_slope.index, name=f'{trend_slope.name}_trend_info')
    
    def trend_info(self, x=None, trend=None, trend_slope=None, filter=None, rolling=None, **kwargs):
        self.params.update(kwargs)
        input_trend_slope = self.trend_slope_ if trend_slope is None else trend_slope
        if input_trend_slope is None:
            input_trend_slope = self.trend_slope(x, trend, filter, rolling, **self.params)
            
        self.trend_info_ = self.calc_trend_info(input_trend_slope)
        return self.trend_info_ 
    
    def trend_group(self, x=None, ptolfloat=0.05, trend=None, trend_slope=None, filter=None, rolling=None, **kwargs):
        """
         . ptolfloat(default: 0.01) : percentage tolerance unit
        """
        filter = self.filter if filter is None else filter
        if filter != 'L1':
            raise("this function is only able to be applied at 'L1' filter.")
        
        self.params.update(kwargs)
        input_trend_slope = self.trend_slope_ if trend_slope is None else trend_slope
        if input_trend_slope is None:
            input_trend_slope = self.trend_slope(x, trend, filter, rolling, **self.params)
            
        slope_diff = abs(input_trend_slope - input_trend_slope.shift().fillna(method='bfill'))

        if ptolfloat == 0:
            slope_change = ~( slope_diff.apply(lambda x: np.allclose(x, 0, atol=1e-05)) )
        else:
            slope_change = ~( slope_diff < abs(input_trend_slope) * ptolfloat )

        self.trend_change_ = slope_change.apply(lambda x: 'point' if x is True else np.nan)
        self.trend_group_ = pd.Series(self_operation(slope_change.astype(int)), index=slope_change.index, name='group')
        return (self.trend_change_, self.trend_group_)
    
    # 【 Summary 】
    def fit(self, x=None, trend=None, trend_slope=None, filter=None, rolling=None, ptolfloat=0.05, **kwargs):
        self.params.update(kwargs)
        self.x = self.x if x is None else x
        filter = self.filter if filter is None else filter
        self.trend_info(self.x, trend, trend_slope, filter, rolling, **self.params)
        
        if filter == 'L1':
            self.trend_group(self.x, ptolfloat, trend, trend_slope, filter, rolling, **self.params)
            self.summary = pd.concat([self.x, self.cycle, self.trend, self.trend_slope_, self.trend_info_, self.trend_change_, self.trend_group_], axis=1)
            self.summary.columns = [self.x.name, 'cycle', 'trend', 'trend_slope', 'trend_info', 'trend_change', 'trend_group']
            
            trend_group_summary_series = self.summary.groupby(['trend_group'])['trend_slope'].mean()
            trend_group_minus_max_slope_idx = trend_group_summary_series.argmin()+1
            trend_group_plus_max_slope_idx = trend_group_summary_series.argmax()+1
            trend_group_abs_min_slope_idx = trend_group_summary_series.apply(lambda x: abs(x)).argmin()+1
            trend_group_abs_max_slope_idx = trend_group_summary_series.apply(lambda x: abs(x)).argmax()+1

            trend_group_describe = {'minus_max': (trend_group_minus_max_slope_idx, trend_group_summary_series.loc[trend_group_minus_max_slope_idx]),
                                    'plus_max': (trend_group_plus_max_slope_idx, trend_group_summary_series.loc[trend_group_plus_max_slope_idx]),
                                    'abs_min': (trend_group_abs_min_slope_idx, trend_group_summary_series.loc[trend_group_abs_min_slope_idx]),
                                    'abs_max': (trend_group_abs_max_slope_idx, trend_group_summary_series.loc[trend_group_abs_max_slope_idx]),
                                    }
            trend_group_summary = trend_group_summary_series.to_frame()
            trend_group_summary['describe'] = np.nan

            for k, v in trend_group_describe.items():
                if pd.isna(trend_group_summary.loc[v[0], 'describe']):
                    trend_group_summary.loc[v[0], 'describe'] = k
                else:
                    trend_group_summary.loc[v[0], 'describe'] = trend_group_summary.loc[v[0], 'describe'] + ', ' + k
            
            self.group_describe = trend_group_describe
            self.group_summary = trend_group_summary
            return self.summary
        
        else:
            self.summary = pd.concat([self.x, self.cycle, self.trend, self.trend_slope_, self.trend_info_], axis=1)
            self.summary.columns = [self.x.name, 'cycle', 'trend', 'trend_slope', 'trend_info']
        return self.summary



# # 1) 예제 시계열 데이터 생성
# np.random.seed(0)
# n = 200
# idx = pd.date_range("2020-01-01", periods=n, freq="D")

# # 추세 + 순환 + 노이즈
# trend_true = np.linspace(0, 10, n)
# cycle_true = 2 * np.sin(np.linspace(0, 4*np.pi, n))
# noise = np.random.normal(scale=0.5, size=n)

# y = trend_true + cycle_true + noise
# s = pd.Series(y, index=idx, name="y")

# series_plot(s)

# # 2) TrendAnalysis 객체 생성 (HP filter 사용)
# ta = TrendAnalysis(x=s, filter='hp_filter', lamb=1600, rolling=5)

# # 3) trend_slope, trend_info, summary 계산
# summary = ta.fit()   # hp_filter이므로 group은 없음

# print("=== summary head ===")
# print(summary.head())

# # 4) 결과 시각화 (원 시계열 vs trend)
# fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
# axes[0].plot(summary.index, summary["y"], label="y", alpha=0.7)
# axes[0].legend()

# axes[1].plot(summary.index, summary["trend"], label="trend (HP)", color="orange")
# axes[1].legend()

# axes[2].plot(summary.index, summary["trend_slope"], label="trend_slope", color="green")
# axes[2].legend()

# plt.tight_layout()
# plt.show()




# # 1) 동일한 시계열에 대해 L1 filter 사용
# ta_L1 = TrendAnalysis(x=s, filter='L1', lamb=50, rolling=5)

# summary_L1 = ta_L1.fit(ptolfloat=0.05)   # ptolfloat: slope 변화 허용비율

# print("=== L1 summary head ===")
# print(summary_L1.head())

# print("\n=== group_summary (L1) ===")
# print(ta_L1.group_summary)

# # 2) L1 trend + group 시각화
# fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

# axes[0].plot(summary_L1.index, summary_L1["y"], label="y", alpha=0.7)
# axes[0].legend()

# axes[1].plot(summary_L1.index, summary_L1["trend"], label="trend (L1)", color="orange")
# axes[1].legend()

# # trend_group을 color bar 형식으로 살짝 시각화
# axes[2].plot(summary_L1.index, summary_L1["trend_slope"], label="trend_slope", color="green")
# for t, g in zip(summary_L1.index, summary_L1["trend_group"]):
#     axes[2].axvline(t, color="gray", alpha=0.05 * g)  # 그룹 번호에 따라 살짝 진하게

# axes[2].legend()
# plt.tight_layout()
# plt.show()