import os
from functools import reduce
import datetime

# @numba.njit
# def self_operation(x, operator='+', init='first'):
#     x_np = np.array(x)
#     result = np.zeros_like(x_np)
#     result[0] = x_np[0] if init == 'first' else init
#     for i in range(1, len(x_np)):
#         result[i] = eval(f'result[i-1] {operator} x_np[i]')
#     return result


# from IPython.display import clear_output
# pip install nbconvert
# conda install nbconvert
# pip install ipynb-py-convert
# jupyter nbconvert --to markdown JUPYTER_NOTEBOOK.ipynb

# path = 'D:/작업방/업무 - 자동차 ★★★/Workspace_Python/기타'
# path1 = 'D:/작업방/업무 - 자동차 ★★★/Workspace_Python/기타/test03.py'
# path2 = 'D:/작업방/업무 - 자동차 ★★★/Workspace_Python/기타/test01.ipynb'
# path3 = 'D:/작업방/업무 - 자동차 ★★★/Workspace_Python/기타/test02.ipynb'
# path4 = [path2, path3]

# path.split('.')
# path1.split('.')
# os.system(f'jupyter nbconvert --to script "{path2}" "{path1}"')
# os.system(f'jupyter nbconvert --to notebook "{path1}" "{path3}"')
# os.system(f'jupyter nbconvert --config "{path1}"')

# os.system(f'ipynb-py-convert "{path2}" "{path1}"')
# os.system(f'ipynb-py-convert "{path1}" "{path3}"')



# ic = IpynbConverter()
# # ic.listdir(path)
# ic.convert(path, input='ipynb', output='py')
# ic.convert(path, input='py', output='html')
# ic.convert(path, input='ipynb', output='py')
# ic.convert(path1, output='ipynb')


# 【 Data function 】  ################################################################################


# ★ argmax with duplicated max number (max 값이 여러개일 때, max값 중 랜덤하게 sample해주는 함수)
def rand_argmax(a, axis=None, return_max=False, random_state=None):
    rng = np.random.RandomState(random_state)
    if axis is None:
        mask = (a == a.max())
        idx = rng.choice(np.flatnonzero(mask))
        if return_max:
            return idx, a.flatten()[idx]
        else:
            return idx
    else:
        mask = (a == a.max(axis=axis, keepdims=True))
        idx = np.apply_along_axis(lambda x: rng.choice(np.flatnonzero(x)), axis=axis, arr=mask)
        expanded_idx = np.expand_dims(idx, axis=axis)
        if return_max:
            return idx, np.take_along_axis(a, expanded_idx, axis=axis)
        else:
            return idx



# Dictionary를 보기좋게 Printing 해주는 함수
def print_dict(d, indent=0):
    for key, value in d.items():
        if isinstance(value, dict):
            print('\t' * indent + '【'+ str(key) + '】')
            print_dict(value, indent+1)
        else:
            print('\t' * indent + '【'+ str(key) + '】', end=' : ')
            print(str(value))


# 정의된 변수명을 return하는 함수
def get_variable_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var][0]


# DecimalPoint : 어떤 값에 대하여 자동으로 소수점 자리수를 부여
def fun_Decimalpoint(value):
    if value == 0:
        return 3
    try:
        point_log10 = np.floor(np.log10(abs(value)))
        point = int((point_log10 - 3)* -1) if point_log10 >= 0 else int((point_log10 - 2)* -1)
    except:
        point = 0
    return point

# 'num ~ num' format data → [num, num]
def criteria_split(criteria, error_handle=np.nan):
    try:
        if criteria == '' or pd.isna(criteria):
            return [np.nan, np.nan]
        else:
            criteria_list = list(map(lambda x: x.strip(), criteria.split('~')))
            criteria_list[0] = -np.inf if criteria_list[0] == '' else float(criteria_list[0])
            criteria_list[1] = np.inf if criteria_list[1] == '' else float(criteria_list[1])
            return criteria_list
    except:
        if error_handle == 'error':
            raise ValueError("An unacceptable value has been entered.\n .Allow format : num ~ num ")
        else:
            return [error_handle] * 2
    
# 'num ~ num' format series → min: [num...], max: [num...] seriess
def lsl_usl_split(criteria_data):
    if criteria_data.ndim == 1:
        splited_series = criteria_data.apply(lambda x: pd.Series(criteria_split(x))).apply(lambda x: x.drop_duplicates().apply(lambda x: x if -np.inf < x < np.inf else np.nan ).sort_values().dropna().to_list(), axis=0)
        splited_series.index = ['lsl','usl']
        splited_series.name = criteria_data.name
        return splited_series.to_dict()
    elif criteria_data.ndim == 2:
        result_dict = {}
        for c in criteria_data:
            splited_series = criteria_data[c].apply(lambda x: pd.Series(criteria_split(x))).apply(lambda x: x.drop_duplicates().apply(lambda x: x if -np.inf < x < np.inf else np.nan ).sort_values().dropna().to_list(), axis=0)
            splited_series.index = ['lsl','usl']
            splited_series.name = c
            result_dict[c] =  splited_series.to_dict()
        return result_dict



# 특정 값이나 vector에 자동으로 소수점 부여
# function auto_formating
def auto_formating(x, criteria='max', return_type=None, decimal=None, decimal_revision=0, thousand_format=True):
    special_case=False
    if type(x) == str:
        return x
    
    x_type_str = str(type(x))
    if 'int' in x_type_str or 'float' in x_type_str:
        if np.isnan(x) or x is np.nan:
            return np.nan
        x_array = np.array([x])
    else:
        x_array = np.array(x)
    
    x_Series_dropna = pd.Series(x_array[np.isnan(x_array) == False])        

    # 소수점 자릿수 Auto Setting
    if decimal is None:
        if criteria == 'median':
            quantile = Quantile(q=0.5)
            criteria_num = quantile(x_Series_dropna)
        elif 'q' in criteria:
            quantile = Quantile(q=float(criteria.replace('q',''))/100)
            criteria_num = quantile(x_Series_dropna)
        elif criteria == 'mode':
            criteria_num = x_Series_dropna.mode()[0]
        else:
            criteria_num = eval('x_Series_dropna.' + criteria + '()')

        decimal = fun_Decimalpoint(criteria_num) + decimal_revision
    
    # 소수점 자릿수에 따른 dtype 변환
    if decimal < 1:
        if x_Series_dropna.min() == -np.inf or x_Series_dropna.max() == np.inf:
            result_Series = pd.Series(x_array).round().astype('float')
            special_case = 'inf'
        else:
            result_Series = pd.Series(x_array).round().astype('Int64')
    else:
        result_Series = pd.Series(x_array).apply(lambda x: round(x, decimal))
    
    # Output dtype 변환
    if return_type == 'str':
        result_Series = result_Series.apply(lambda x: '' if (type(x) == pd._libs.missing.NAType or np.isnan(x) or x is np.nan) else str(x) )
        if thousand_format:
            if decimal < 1:
                result_Series = result_Series.apply(lambda x: '' if x == '' else (str(x) if abs(float(x)) == np.inf else format(int(x), ',')) )
                # result_Series = result_Series.apply(lambda x: '' if x == '' else format(int(x), ','))
            else:    
                result_Series = result_Series.apply(lambda x: '' if x == '' else format(float(x), ','))
    elif return_type == 'float':
        if decimal < 1:
            result_Series = result_Series.round().astype('Int64')
        else:
            result_Series = result_Series.astype(float)
    elif return_type == 'int':
        result_Series = result_Series.round().astype('Int64')
       
    # Output Type에 따른 Return
    if 'str' in x_type_str or 'float' in x_type_str or 'int' in x_type_str:
        if return_type is None:
            if 'str' in x_type_str:
                return str(result_Series[0])
            elif 'float' in x_type_str:
                if decimal < 1:
                    if special_case == 'inf':
                        return float(result_Series[0])
                    else:
                        return int(result_Series[0])
                else:
                    return float(result_Series[0])
            elif 'int' in x_type_str:
                return int(result_Series[0])
        else:
            return eval(return_type + '(result_Series[0])')
    elif 'array' in x_type_str:
        return np.array(result_Series)
    elif 'Series' in x_type_str:
        result_Series.index = x.index
        return result_Series



# from functools import reduce
class IpynbConverter:
    """
    【Requried Library】import os, import nbconvert, import ipynb-py-convert, from functools import reduce
    """
    def __init__(self, path=None):
        self.listdir(path)
        self.convert_dict = {'markdown': 'markdown', 'md': 'markdown', 'html': 'html', 'pdf':'pdf'}
        self.splitext_dict = {'py':'py', 'python': 'py',
                              'notebook':'ipynb', 'ipynb':'ipynb', 'ipython': 'ipynb',
                              'markdown': 'md', 'md': 'md', 'html': 'html', 'pdf':'pdf'}

    # filtering py, ipynb file     
    def _filter_sep_files(self, path):
        path = path.replace('\\', '/')
        
        if os.path.splitext(path)[1] == '': # folder
            folder_path = path + '/'
            path_files = os.listdir(path)
        else:
            folder_path = ''
            path_files = [path]
        path_dict = {}
        path_dict['py'] = [folder_path + f for f in path_files if os.path.splitext(f)[1] == '.py' in f]
        path_dict['ipynb'] = [folder_path + f for f in path_files if os.path.splitext(f)[1] == '.ipynb' in f]
        return path_dict
    
    def _convert_list_type_path(self, path, splitext):
        if path is None:
            path = self.path_dict[splitext]
        elif type(path) == list:
            path = [p.replace('\\', '/') for p in path]
        elif type(path) == dict:
            path = path[splitext]
        else:
            path = self.listdir(path, verbose=0)[splitext]
        return path
    
    # list py, ipynb file
    def listdir(self, path=None, verbose=1):
        if path is not None:
            self.path_dict = self._filter_sep_files(path)
            if verbose > 0:
                print("self.path_dict")
            return self.path_dict
    
    # debug_recording
    def _debug_record(self, syscode, input_path, debug=False):
        if debug is False:
            try:
                os.system(syscode)
                self.convert_success.append(input_path)
            except:
                self.convert_failure.append(input_path)
        else:
            os.system(syscode)
    
    # ipynb-py-convert
    def _ipynb_py_convert_command(self, input_path, output_path, debug=False):
        syscode = f'ipynb-py-convert "{input_path}" "{output_path}"'
        self._debug_record(syscode=syscode, input_path=input_path, debug=debug)
    
    # nbcovert
    def _nbcovert_command(self, input_path, output, execute=True, debug=False):
        execute_code = '--execute ' if execute is True else ''
        syscode = f'jupyter nbconvert {execute_code}--to {self.convert_dict[output]} "{input_path}"'
        self._debug_record(syscode=syscode, input_path=input_path, debug=debug)
       
    # Convert ipynb to py
    def convert(self, path=None, input=None, output='py', execute=True, verbose=1, debug=False):
        if (input is None) and (type(path) == str):
            input = os.path.splitext(path)[1][1:]
        elif (input is None) and (type(path) == list):
            splitext_list = [os.path.splitext(p)[1][1:] for p in path]
            splitext = reduce(lambda x,y: x if x == y else 0, splitext_list)
            if splitext != 0:
                input = splitext
        path_list = self._convert_list_type_path(path=path, splitext=input)
        len_input_splitext = len(input)
        
        # return path
        self.convert_success = []
        self.convert_failure = []
        e = 1
        
        output_splitext = self.splitext_dict[output]
        
        # return path_list
        for p in path_list:
            if verbose > 0:
                print(f'({round((e)/len(path_list)*100,1)}%) "{p.split("/")[-1]}" Converting...', end='\r')

            if (input == 'ipynb' and output_splitext == 'py') or (input=='py' and output_splitext == 'ipynb'):
                # ipynb → py / py → ipynb
                self._ipynb_py_convert_command(input_path=p, 
                                               output_path=f'{p[:-len_input_splitext]}{output_splitext}', 
                                               debug=debug)
                # if output_splitext == 'ipynb' and execute is True:
                #     os.system(f"jupyter nbconvert --to notebook --execute {p[:-len_input_splitext]}{output_splitext}")
                    
            elif input == 'ipynb':
                # ipynb → html, md, pdf
                self._nbcovert_command(input_path=p, output=output_splitext, execute=execute, debug=debug)
                
            elif input == 'py':
                # py → ipynb → html, md, pdf
                time_now = '_' + str(datetime.datetime.now()).replace('-','').replace(' ','').replace(':','').replace('.','')
                temp_name = f'{p[:-len_input_splitext-1]}{time_now}.ipynb'
                
                self._ipynb_py_convert_command(input_path=p, output_path=temp_name, debug=debug)
                self._nbcovert_command(input_path=temp_name, output=output_splitext, execute=execute, debug=debug)
                os.remove(temp_name)
                os.rename(f"{temp_name[:-6]}.{output_splitext}", f"{temp_name[:-(len(b)+6)]}.{output_splitext}")
                
            e += 1
    
        if verbose > 0 and len(self.convert_failure) >0:
            print('convert_failure file list')
            print(self.convert_failure)








