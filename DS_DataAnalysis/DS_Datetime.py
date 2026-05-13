import numba
import numpy as np
import pandas as pd
import scipy as sp


import cvxpy 
import cvxopt 

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



def time_period_transform(data, freq, format=None):
    # temp = pd.to_datetime(data, format=format)
    # temp_frame = temp.to_frame()
    # temp_frame.index = temp
    # temp_frame.index.name = 'freq'
    
    # temp_result = temp_frame.resample(freq, label='left').min().dropna().reset_index()
    # if ('y' in freq.lower()) or ('m' in freq.lower()) or ('q' in freq.lower()):
    #     temp_result['freq'] = temp_result['freq'] + pd.DateOffset(days=1)
    # result = temp_result[temp_result.columns[::-1]]
    # result.iloc[:,0] = result.iloc[:,0].astype(data.dtype)
    # result['freq'] = result['freq'].astype(str)
    # return result

    if 'datetime' not in str(data.dtype):
        data_time = pd.to_datetime(data, format=format)
    else:
        data_time = data.copy()
    data_time_temp = data_time.apply(lambda x: pd.period_range(x, x, freq=freq)[0])
    data_time_temp.name = 'freq'
    
    data_time_temp_group = pd.concat([data_time_temp, data], axis=1)
    return data_time_temp_group.groupby('freq')[data_time_temp_group.columns[1]].min().reset_index().iloc[:,[1,0]]

# tpt = time_period_transform(d2['소둔_작업완료일시'], freq='d')
# tpt2 = tpt.set_index('freq').resample('5d').min().dropna().reset_index()
# tpt3= tpt2[tpt2.columns[::-1]]

# f = plt.figure(figsize=(20,3))
# plt.title('(고YS) 980DP 시계열실적 (실적 - 예측)')
# plt.scatter(d2['소둔_작업완료일시'], d2['diff_YP'], alpha=0.2, color='steelblue', s=5)
# plt.plot(d2['소둔_작업완료일시'], d2['diff_YP'],alpha=0.3, color='steelblue')
# plt.plot(d2['소둔_작업완료일시'], trend, alpha=1, color='orange')
# plt.xticks(*np.array(tpt3.T), rotation=45)
# plt.axhline(0, color='black', alpha=0.7)
# plt.show()
# img_to_clipboard(f, dpi=150)





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










# @numba.njit
def self_operation(x, operator='+', init='first'):
    x_np = np.array(x)
    result = np.zeros_like(x_np)
    result[0] = x_np[0] if init == 'first' else init
    for i in range(1, len(x_np)):
        result[i] = eval(f'result[i-1] {operator} x_np[i]')
    return result


# Trend Analysis Class
class TrendAnalysis():
    """
    【 Required Library 】
    import statsmodels.api as sm  (sm.tsa.filters.hpfilter)
    
    【 self.hp_filter 】
     . lamb=1600 : The Hodrick-Prescott smoothing parameter. 
        (suggesting) month: 129600, quarter: 1600, year: 6.25 
    
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

