import numpy as np
import pandas as pd
import scipy as sp

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import font_manager, rc    # 한글폰트사용
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

import seaborn as sns

from DS_OLS import *
from DS_DataFrame import *
from DS_MachineLearning import *


# def ttest_each
# 여러개의 Group별로 평균, 편차, ttest 결과를 Return 하는 함수
# import scipy as sp
# from collections import namedtuple
from itertools import combinations
def ttest_each(data, x, group, equal_var=False, decimal_point=4, return_result='all', return_type='vector'):
    """
    < input >
     . data (DataFrame): DataFrame
     . x (str): column name
     . group (str, list): grouping columns
     . equal_var (bool): whether variance is equal between group when processing ttest
     . decimal_point (int, None): pvalue decimal 
     . return_result (str): 'all', 'count', 'mean', 'std', 'ttest', 'plot'
     . return_type (str): 'matrix', 'vector'

    < output >
     . table by group (table)
    """
    result = namedtuple('ttest_each', ['count', 'mean', 'std', 'ttest'])

    if type(group) == list and len(group) > 1:
        group_unique = data[group].sort_values(by=group).drop_duplicates()
        # group_index_names = group_unique.apply(lambda x: ', '.join([f"{idx}: {v}" for idx, v in zip(x.index, x)]),axis=1).tolist()
        group_index = pd.MultiIndex.from_frame(group_unique)
        groups = group.copy()
    else:
        if type(group) == list:
            group_unique = data[group[0]].drop_duplicates()
            groups = group.copy()
        elif type(group) == str:
            group_unique = data[group].drop_duplicates()
            groups = [group].copy()
        # group_index_names = group_unique.copy()
        group_index = group_unique.to_list().copy()
    # print(group_index)
    # print(groups)

    group_table = pd.DataFrame(np.zeros(shape=(len(group_index), len(group_index))), index=group_index, columns=group_index)
    group_table[group_table== 0] = np.nan
    
    table_count = group_table.copy()
    table_mean = group_table.copy()
    table_std = group_table.copy()
    table_ttest = group_table.copy()
    table_plot = group_table.copy()

    groups_dict = {}
    for gi, gv in data.groupby(groups):
        groups_dict[gi] = np.array(gv[x])

    vector_table_list = []
    for g in combinations(group_index, 2):
        data_group = [groups_dict[g[1]], groups_dict[g[0]]]
        data_group_count = [int(len(x)) for x in data_group]
        data_group_mean = [auto_formating(np.mean(x)) for x in data_group]
        data_group_std = [auto_formating(np.std(x)) for x in data_group]

        group_count = f" {data_group_count[1]} - {data_group_count[0]}"
        group_mean = f" {data_group_mean[1]} - {data_group_mean[0]}"
        group_std = f" {data_group_std[1]} - {data_group_std[0]}"
        group_ttest = sp.stats.ttest_ind(data_group[1], data_group[0], equal_var=equal_var).pvalue
        if decimal_point is not None:
            group_ttest = round(group_ttest, decimal_point)
            
        table_count.loc[g[0], g[1]] = group_count
        table_mean.loc[g[0], g[1]] = group_mean
        table_std.loc[g[0], g[1]] = group_std
        table_ttest.loc[g[0], g[1]] = group_ttest
        
        # if return_result == 'plot':
        data_group1 = pd.Series(data_group[1], name=x).to_frame()
        data_group1[group] = g[1]
        data_group2 = pd.Series(data_group[0], name=x).to_frame()
        data_group2[group] = g[0]
        data_concat = pd.concat([data_group1, data_group2], axis=0)
        group_plot = distbox(data=data_concat, on=x, group=group)
        table_plot.loc[g[0], g[1]] = group_plot
        
        vector_table_list.append([g[0], g[1], group_count, group_mean, group_std, group_ttest, group_plot])
    vector_table = pd.DataFrame(vector_table_list, columns=['group1', 'group2', 'count','mean', 'std', 'ttest', 'plot']).set_index(['group1', 'group2'])
    
    if return_result == 'all':
        return result(table_count, table_mean, table_std, table_ttest) if return_type == 'matrix' else vector_table
    elif return_result == 'count':
        return table_count if return_type == 'matrix' else vector_table['count']
    elif return_result == 'mean':
        return table_mean if return_type == 'matrix' else vector_table['mean']
    elif return_result == 'std':
        return table_std if return_type == 'matrix' else vector_table['std']
    elif return_result == 'ttest':  
        return table_ttest if return_type == 'matrix' else vector_table['ttest']
    elif return_result == 'plot':  
        return table_plot if return_type == 'matrix' else vector_table['plot']



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
        return result_Series

# function calc cpk
def cpk(mean, std, lsl=None, usl=None, lean=False):
    if np.isnan(std) or std == 0:
        return np.nan
    if (lsl is None or np.isnan(lsl))and (usl is None or np.isnan(usl)):
        return np.nan
    lsl = -np.inf if (lsl is None or np.isnan(lsl)) else lsl
    usl = np.inf if (usl is None or np.isnan(usl)) else usl

    cpk = min(usl-mean, mean-lsl) / (3 * std)
    if lean:
       sign = 1 if usl-mean < mean-lsl else -1
       cpk = 0.01 if cpk < 0 else cpk
       cpk *= sign
    return cpk

# cpk_line in histogram
def cpk_line(x, bins=50, density=False):
    x_describe = x.describe()
    x_lim = x_describe[['min', 'max']]
    x_min = min(x_describe['min'], x_describe['mean'] - 3 * x_describe['std'])
    x_max = max(x_describe['max'], x_describe['mean'] + 3 * x_describe['std'])
    x_100Divide = np.linspace(x_min, x_max, 101)   # x 정의
    y_100Norm = (1 / (np.sqrt(2 * np.pi)*x_describe['std'])) * np.exp(-1* (x_100Divide - x_describe['mean'])** 2 / (2* (x_describe['std']**2)) )
    if not density:
        y_rev = len(x)/(bins) * (x_describe['max'] -x_describe['min'])
        y_100Norm *= y_rev
    return pd.DataFrame([x_100Divide,y_100Norm], index=[x.name, 'cpk']).T


# -----------------------------------------------------------------------------------------------------
# function jitter (make jitter list)
def jitter(x, ratio=0.6, method='uniform', sigma=5, transform=True):
    type_string = str(type(x))
    if 'Series' in type_string:
        dtype = 'Series'
        series_x = x.copy()
    else:
        if 'list' in type_string:
            dtype = 'list'
        elif 'ndarray' in type_string:
            dtype = 'ndarray'
        series_x = pd.Series(x)
    x1 = series_x.drop_duplicates().sort_values()
    x2 = series_x.drop_duplicates().sort_values().shift()
    jitter_range = (x1-x2).min()*ratio

    # apply distribution
    if method == 'uniform':
        jitter = pd.Series(np.random.rand(len(x))*jitter_range - jitter_range/2)
    elif method == 'gaussian' or method == 'normal':
        jitter = pd.Series(np.random.randn(len(x))*(jitter_range/sigma))
    if dtype == 'Series':
        jitter.index = x.index
    
    if transform:
        jitter += series_x
    
    # transform return type
    if dtype == 'Series':
        result = jitter
    if dtype == 'list':
        result = jitter.tolist()
    elif dtype == 'ndarray':
        result = jitter.values
    
    return result


# -----------------------------------------------------------------------------------------------------
# Data Transform : plot → group multi plot 
def transform_group_array(x, group, data=None):
    data_type_str = str(type(data))
    dtype_error_msg = 'Only below arguments are available \n . data: DataFrame, x: colname in data, group: colname in data \n  or\n . x: list/ndarray/Series, group: list/ndarray/Series'
    list_types = ['str', 'list', 'ndarray', 'Series', 'DataFrame']
    x_type_list = [lt in str(type(x))  for lt in list_types]
    group_type_list = [lt in str(type(group))  for lt in list_types]
    x_type_str = list_types[np.argwhere(x_type_list)[0][0]]
    group_type_str = list_types[np.argwhere(group_type_list)[0][0]]
    
    # dtype error check
    if data is None:
        if not (sum(x_type_list) * sum(group_type_list)):
            raise(dtype_error_msg)
        elif len(x) != len(group):
            raise('Unequal array lenght between x and group.')
    elif 'DataFrame' in data_type_str:
        if not (sum(x_type_list) * sum(group_type_list)):
            raise(dtype_error_msg)
        elif 'int' not in str(data[x].dtype) and data[x].dtype != float:
            raise('x column type must be numeric.')
    else:
        raise(dtype_error_msg)

    if 'DataFrame' not in data_type_str:
        if x_type_str == 'Series':
            if group_type_str != 'DataFrame':
                group = pd.Series(group)
            group.index = x.index
        elif group_type_str == 'Series':
            x = pd.Series(x)
            x.index = group.index
        else:
            x = pd.Series(x)
            group = pd.Series(group)

        data = pd.concat([group, x], axis=1)
        if group_type_str == 'DataFrame':
            data.columns = list(group.columns) + ['x']
            group = list(group.columns)
        else:
            data.columns = ['group', 'x']
            group = 'group'
        x = 'x'
    
    if type(group) == list: # group to list
        group_list = group.copy()
    else:
        group_list = [group].copy()
    group_obj = {}
    group_obj['index'] = []
    group_obj['value'] = []
    group_obj['count'] = []
    group_obj['mean'] = []
    group_obj['std'] = []
    for i, g in data[group_list + [x]].groupby(group_list):
        # group_array = np.array(g[x])
        group_obj['index'].append(i)

        if x_type_str == 'list':
            group_obj['value'].append(g[x].tolist())
        elif x_type_str == 'ndarray':
            group_obj['value'].append(g[x].values)
        else:
            group_obj['value'].append(g[x])

        group_obj['count'].append(len(g[x].dropna()))
        group_obj['mean'].append(g[x].mean())
        group_obj['std'].append(g[x].std())
    group_obj['pvalue'] = anova(*group_obj['value'], equal_var=False).pvalue

    return group_obj


# -----------------------------------------------------------------------------------------------------
# Dist_Box Plot Graph Function
def distbox(data, on, group=None, figsize=[5,5], title='auto', bins=None,
            mean_line=None, axvline=None, lsl=None, usl=None, xscale='linear',
            xlim=None, ylim=None,
            equal_var=False, return_plot='close'):
    # group = change_target
    # on = 'YP'
    # title = 'abc'
    normal_data = data.copy()
    # box_colors = ['steelblue','orange']
    box_colors = sns.color_palette()

    figs, axes = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=figsize)
    

    # distplot
    if title is not None and title != 'auto':
        figs.suptitle(title, fontsize=13)
    elif title == 'auto':
        title_name = on + '_Plot'
        if group is not None:
            title_name += ' (group: ' + group + ')'
        figs.suptitle(title_name, fontsize=13)

    if group is not None:
        # group_mean
        group_mean = normal_data.groupby(group)[on].mean()
        len_group_mean = len(group_mean)
        group_mean.sort_index(ascending=True, inplace=True)

        # distplot
        data_group = []
        for i, (gi, gv) in enumerate(normal_data.groupby(group)):
            data_group.append(gv[on].dropna())
            try:
                sns.distplot(gv[on], label=gi, ax=axes[0], bins=bins)
                if mean_line is not None:
                    axes[0].axvline(x=group_mean[gi], c=box_colors[i], alpha=0.5)
            except:
                pass
        axes[0].legend()
        axes[0].set_xscale(xscale)
        
        print(group)
        # boxplot
        boxes = sns.boxplot(x=on, y=group, data=normal_data, 
                orient='h', color='white', linewidth=1, ax=axes[1],
                order=sorted(normal_data[group].unique()) )
        axes[1].set_xscale(xscale)

        # mean_point
        axes[1].scatter(x=group_mean, y=list(range(0,len_group_mean)), 
                        color=box_colors[:len_group_mean], edgecolors='white', s=70)
        
        if len(data_group) == 2:
            pavlues = sp.stats.ttest_ind(*data_group, equal_var=equal_var).pvalue
        else:
            pavlues = sp.stats.f_oneway(*data_group).pvalue
        label_name = 'Anova Pvalue: ' + format(pavlues, '.3f')

        summary_dict = normal_data.groupby(group)[on].agg(['count','mean','std']).applymap(lambda x: auto_formating(x)).to_dict('index')

        if lsl is not None or usl is not None:
            cpk_list = ['-' if v['count'] < 5 else str(round(cpk(v['mean'], v['std'], lsl, usl),2)) for k, v in summary_dict.items()]
            iter_object = zip(summary_dict.items(), cpk_list)  
            label_summary = '\n'.join(['* ' + str(k) + ': ' + str(v).replace('{','').replace('}','').replace("'",'').replace(':','') + ' (cpk: '+cpk_value+')' for (k,v), cpk_value in iter_object ])
        else:
            iter_object = summary_dict.items()
            label_summary = '\n'.join(['* ' + str(k) + ': ' + str(v).replace('{','').replace('}','').replace("'",'').replace(':','') for k,v in iter_object ])
        
        label_name = label_name + '\n' + label_summary
        plt.xlabel(label_name, fontsize=11)
    else:
        # group_mean
        group_mean, group_std = normal_data[on].agg(['mean','std'])

        # distplot
        sns.distplot(normal_data[on], ax=axes[0], bins=bins)
        if mean_line:
            axes[0].axvline(x=group_mean, c=box_colors[0], alpha=0.5)
        # boxplot
        axes[0].set_xscale(xscale)
        boxes = sns.boxplot(data=normal_data, x=on, orient='h', color='white', linewidth=1, ax=axes[1])
        
        # mean_points
        plt.scatter(x=group_mean, y=[0], color=box_colors[0], edgecolors='white', s=70)
        axes[1].set_xscale(xscale)

        summary_dict = normal_data[on].agg(['count','mean', 'std']).apply(lambda x: auto_formating(x)).to_dict()
        label_summary = '* All: ' + ', '.join([ k + ' ' + str(v) for k,v in summary_dict.items() ])
        label_name = '\n' + label_summary

        if lsl is not None or usl is not None:
            if len(normal_data) < 5:
                cpk_value = '-'
            cpk_value = str(round(cpk(group_mean, group_std, lsl=lsl, usl=usl),2))
            label_name = label_name + ' (cpk: ' + cpk_value +')'
        plt.xlabel(label_name, fontsize=11)

    # Box-plot option
    for bi, box in enumerate(boxes.artists):
        box.set_edgecolor(box_colors[bi])
        for bj in range(6*bi,6*(bi+1)):    # iterate over whiskers and median lines
            boxes.lines[bj].set_color(box_colors[bi])
    plt.grid(alpha=0.1)
    figs.subplots_adjust(hspace=0.5)
    # figs.subplots_adjust(bottom=0.2)
    
    if xlim is not None:
        axes[0].set_xlim(xlim)
        axes[1].set_xlim(xlim)
        
    if ylim is not None:
        axes[0].set_ylim(ylim)

    # axvline
    if axvline is not None and type(axvline) == list:
        for vl in axvline:
            axes[0].axvline(vl, color='orange', ls='--', alpha=0.3)
            axes[1].axvline(vl, color='orange', ls='--', alpha=0.3)

    if lsl is not None:
        axes[0].axvline(lsl, color='red', ls='--', alpha=0.3)
        axes[1].axvline(lsl, color='red', ls='--', alpha=0.3)
    if usl is not None:
        axes[0].axvline(usl, color='red', ls='--', alpha=0.3)
        axes[1].axvline(usl, color='red', ls='--', alpha=0.3)

    if return_plot == 'close':
        plt.close()
    elif return_plot == 'show':
        plt.show()
    elif return_plot is None or return_plot == False:
        pass
    return figs

    



# Histogram Compare Graph Function
def hist_compare(data1, data2, figsize=None, title=None, bins=30, label=None, hist_alpha=0.5, histtype='stepfilled',
    legend=True, color=['skyblue','orange'], cpk_color=None, lsl=None, usl=None,
    cpk_alpha=0.7, legend_loc='upper right', return_plot=True, axvline=None, axvline_color='red', **hist_kwargs):

    hist_data1 = pd.Series(data1).dropna().astype('float')
    hist_data2 = pd.Series(data2).dropna().astype('float')

    len_data1 = len(hist_data1)
    len_data2 = len(hist_data2)


    mean_data1 = auto_formating(hist_data1.mean()) if len_data1 > 1 else hist_data1.iloc[0]
    mean_data2 = auto_formating(hist_data2.mean()) if len_data2 > 1 else hist_data2.iloc[0]
    std_data1 = auto_formating(hist_data1.std()) if len_data1 > 1 else np.nan
    std_data2 = auto_formating(hist_data2.std()) if len_data2 > 1 else np.nan
    # pvalue = sp.stats.ttest_ind(hist_data1, hist_data2, equal_var=False)[1]
    if len_data1 == 1 and len_data2 == 1:
        pavlue = np.nan
    elif len_data1 == 1:
        pvalue = round(sp.stats.ttest_1samp(hist_data1.iloc[0], hist_data2).pvalue, 3)   # x1 Column의 평균이 4와 같은가?
    elif len_data2 == 1:
        pvalue = round(sp.stats.ttest_1samp(hist_data1, hist_data2.iloc[0]).pvalue, 3) 
    else:
        pvalue = round(sp.stats.ttest_ind_from_stats(*hist_data1.agg(['mean','std','count']),
                *hist_data2.agg(['mean','std','count']), equal_var=False).pvalue, 3)

    try:
        name_data1 = data1.name
    except:
        name_data1 = 'Group1'
    try:
        name_data2 = data2.name
    except:
        name_data2 = 'Group2'

    if label:
        name_data1 = label[0]
        name_data2 = label[1]

    cpk_color = color if cpk_color is None else cpk_color

    if return_plot:
        fig = plt.figure(figsize=figsize)
    if title:
        plt.title(title)
    elif name_data1 == name_data2:
        plt.title(f'{name_data1} Histogram')
    else:
        plt.title(f'{name_data2} - {name_data2} Histogram')
    
    if label:
        label_content = label
    else:
        label_content = [name_data1, name_data2]
    plt.hist([hist_data1, hist_data2], histtype=histtype, bins=bins, edgecolor='darkgray', 
            color=color, alpha=hist_alpha, label=label_content, **hist_kwargs)
    if 'density' in hist_kwargs.keys() and hist_kwargs['density'] is True:
        pass
    else:
        plt.plot(*np.array(cpk_line(hist_data1)).T, color=cpk_color[0])
        plt.plot(*np.array(cpk_line(hist_data2)).T, color=cpk_color[1])
    plt.axvline(mean_data1, color=cpk_color[0], ls='dashed', alpha=cpk_alpha)
    plt.axvline(mean_data2, color=cpk_color[1], ls='dashed', alpha=cpk_alpha)
    
    if lsl is not None or usl is not None:
        for l in [lsl, usl]:
            if l is not None:
                plt.axvline(l, color='red', alpha=0.7)
        cpk_value1 = round(cpk(mean_data1, std_data1, lsl=lsl, usl=usl),3)
        cpk_value2 = round(cpk(mean_data2, std_data2, lsl=lsl, usl=usl),3)
        xlabel = f'{name_data1}: (n) {len(hist_data1)}   (mean) {mean_data1}   (std) {std_data1}   (cpk) {cpk_value1}\n{name_data2}: (n) {len(hist_data2)}   (mean) {mean_data2}   (std) {std_data2}   (cpk) {cpk_value2}\np-value: {round(pvalue,3)}'
    else:
        xlabel = f'{name_data1}: (n) {len(hist_data1)}   (mean) {mean_data1}   (std) {std_data1}\n{name_data2}: (n) {len(hist_data2)}   (mean) {mean_data2}   (std) {std_data2}\np-value: {round(pvalue,3)}'
    plt.xlabel(xlabel, fontsize=12)

    if legend is True:
        plt.legend(loc=legend_loc)
    
    
    if type(axvline) == list:
        if len(axvline) > 0 and axvline[0] is not None:
            for axvl in axvline:
                plt.axvline(axvl, color=axvline_color, alpha=0.5, ls='--')
    elif axvline is not None:
        plt.axvline(axvline, color=axvline_color, alpha=0.5, ls='--')

    if return_plot:
        plt.close()
        return fig
    elif return_plot == 'show':
        plt.show()
    else:
        pass





# Group histogram
def group_histogram(data, x, group, categories=None, ascending=True, norm={}, histtype='barstacked', edgecolor='grey', color=mpl.cm.get_cmap('Pastel1'), **hist_kwargs):
    data_hist = data[data[[x, group]].isna().sum(1).apply(lambda x: not bool(x))]
    if categories is None:
        group_level = data_hist[group].drop_duplicates().sort_values(ascending=ascending).values
    else:
        group_level = categories
    data_hist['Hist_Group'] = pd.Categorical(data_hist[group], categories=group_level, ordered=True)

    group_dict = {}
    for gi, gv in data_hist.groupby(['Hist_Group']):
        group_dict[gi] = gv

    print(group_level)
    fig = plt.figure()
    plt.title(f'Historgram {x} group by {group}')
    plt.hist([v[x] for k, v in group_dict.items()], histtype='barstacked', color=color, edgecolor='grey', label=group_level, **hist_kwargs)
    plt.legend()
    plt.close()
    return fig



# Multi histogram
# x_multi = [np.random.randn(n) for n in [10, 50, 20]]
# plt.hist(x_multi, stacked=True, edgecolor='grey', linewidth=1.2, label=['a','b','c'])
# plt.legend()
def fun_Hist(data, x, figsize=[6,4], bins=10, density=False,
            color=None, xtick=0, alpha=1,
            norm=False, 
            group=False, group_type='dodge', 
            spec=False, 
            spec_display='all',
            line_x=False,
            line_x_display='all',
            title=False,
            legend=False, 
            xlim=False,
            ylim=False,
            grid=True):
    '''
    data(DataFrame): Histogram Root Data
    x(str, list): Target Variable
    bins(int): The Number of bar in histogram
    hist_alpha(float 0~1): Histogram bar alpha

    '''
    result_obj = {}
    if type(x) == list: # x to list
        x_list = x.copy()
    else:
        x_list = [x].copy()

    # spec
    for cx in x_list:
        normal_data = data[data[cx].isna()==False].copy()

        # plot figure
        hist_fig = plt.figure(figsize=figsize)

        if not group:
            plt.hist(x=normal_data[cx], bins=bins, density=density, color=color, edgecolor='grey', alpha=alpha, label=cx)
            if legend:
                plt.legend()
        else:
            group_obj = fun_Group_Array(data=normal_data, x=cx, group=group)

            if group_type == 'identity':
                for j, v in enumerate(group_obj['value']):
                    plt.hist(x=v, bins=bins, density=density, edgecolor='grey', label=group_obj['index'][j], color=color, alpha=alpha)
            elif group_type == 'stack':
                plt.hist(x=group_obj['value'], bins=bins, density=density, edgecolor='grey', stacked=True, label=group_obj['index'], color=color, alpha=alpha)
            else:
                plt.hist(x=group_obj['value'], bins=bins, density=density, edgecolor='grey', stacked=False, label=group_obj['index'], color=color, alpha=alpha)

            # if type(group) == list: # group to list
            #     group_list = group.copy()
            # else:
            #     group_list = [group].copy()
            # group_value = []
            # group_idx = []
            # for i, g in normal_data[group_list + [cx]].groupby(group_list):
            #     group_array = np.array(g[cx])
            #     group_idx.append(i)
            #     group_value.append(group_array)

            # if group_type == 'identity':
            #     for j, v in enumerate(group_value):
            #         plt.hist(v, edgecolor='grey', label=group_idx[j], color=color, alpha=alpha)
            # elif group_type == 'stack':
            #     plt.hist(group_value, edgecolor='grey', stacked=True, label=group_idx, color=color, alpha=alpha)
            # else:
            #     plt.hist(group_value, edgecolor='grey', stacked=False, label=group_idx, color=color, alpha=alpha)
            plt.legend()
        
        # norm
        if norm:
            x_Summary = normal_data[cx].describe().T
            norm_left = np.min([x_Summary['min'], x_Summary['mean'] - 3*x_Summary['std']])
            norm_right = np.max([x_Summary['max'], x_Summary['mean'] + 3*x_Summary['std']])
            x_100Divide = np.linspace(norm_left, norm_right, 101)   # x 정의
            y_100Norm = (1 / (np.sqrt(2 * np.pi)*x_Summary['std'])) * np.exp(-1* (x_100Divide - x_Summary['mean'])** 2 / (2* (x_Summary['std']**2)) )
            # y = (1 / np.sqrt(2 * np.pi)) * np.exp(- x ** 2 / 2 )
            # y = stats.norm(0, 1).pdf(x) 
            if not density:
                y_rev = len(normal_data[cx])/(bins*1.2) * (x_Summary['max'] -x_Summary['min'])
                y_100Norm *= y_rev
            plt.plot(x_100Divide, y_100Norm, c='tomato', linewidth=1 )

        # Spec-line
        spec_n = 0
        while(spec_n == 0):
            if spec:
                if type(spec) == dict:
                    try:
                        if type(spec[cx]) == list:
                            spec_list = spec[cx].copy()
                        else:
                            spec_list = [spec[cx]].copy()
                    except:
                        break       # while Loop escape
                else:
                    if type(spec) == list:
                        spec_list = spec.copy()
                    else:
                        spec_list = [spec].copy()

                for sl in spec_list:
                    if type(sl) == int or type(sl) == float:
                        s = sl
                        plt.axvline(x=s, c='r', alpha=0.7, linestyle='--')
                    elif sl in normal_data.columns and normal_data[sl].dtype!=str :
                        if spec_display=='auto':
                            if '하한' in sl or 'min' in sl.lower():
                                s = normal_data[sl].max()
                                plt.axvline(x=s, c='r', alpha=0.7, linestyle='--')
                            elif '상한' in sl or 'max' in sl.lower():
                                s = normal_data[sl].min()
                                plt.axvline(x=s, c='r', alpha=0.7, linestyle='--')
                            else:
                                for s in list(normal_data[sl].dropna().drop_duplicates()):
                                    plt.axvline(x=s, c='r', alpha=0.7, linestyle='--')
                        elif spec_display=='all':
                            for s in list(normal_data[sl].dropna().drop_duplicates()):
                                plt.axvline(x=s, c='r', alpha=0.7, linestyle='--')
                    else:
                        print('spec must be numeric or numeric_columns.')
                        pass
            spec_n = 1


        # Sub-line x-Axis
        line_x_n = 0
        while(line_x_n == 0):
            if line_x:
                if type(line_x) == dict:
                        try:
                            if type(line_x[cx]) == list:
                                line_x_list = line_x[cx].copy()
                            else:
                                line_x_list = [line_x[cx]].copy()
                        except:
                            break       # while Loop escape
                else:
                    if type(line_x) == list:
                        line_x_list = line_x.copy()
                    else:
                        line_x_list = [line_x].copy()

                for slx in line_x_list:
                    if type(slx) == int or type(slx) == float:
                        lx = slx
                        plt.axvline(x=lx, c='tomato', alpha=0.5, linestyle='--') 
                    elif slx in normal_data.columns and normal_data[slx].dtype!=str :
                            if line_x_display=='auto':
                                if '하한' in slx or 'min' in slx.lower():
                                    lx = normal_data[slx].max()
                                    plt.axvline(x=lx, c='tomato', alpha=0.5, linestyle='--')
                                elif '상한' in slx or 'max' in slx.lower():
                                    lx = normal_data[slx].min()
                                    plt.axvline(x=lx, c='tomato', alpha=0.5, linestyle='--')
                                else:
                                    for lx in list(normal_data[slx].dropna().drop_duplicates()):
                                        plt.axvline(x=lx, c='tomato', alpha=0.5, linestyle='--')
                            elif line_x_display=='all':
                                for lx in list(normal_data[slx].dropna().drop_duplicates()):
                                    plt.axvline(x=lx, c='tomato', alpha=0.5, linestyle='--')
                    else:
                        print('spec must be numeric or numeric_columns.')
                        pass
            line_x_n = 1

                 

        #title
        if title:
            plt.title(title)
        # Label
        plt.xlabel(cx)

        # Axis
        plt.xticks(rotation=xtick)

        # x_Limit
        if xlim and type(xlim) == list:
            if type(xlim[0]) == float or type(xlim[0]) == int:
                if type(xlim[1]) == float or type(xlim[1]) == int:
                    plt.xlim(left=xlim[0], right=xlim[1])
                else:
                    plt.xlim(left=xlim[0])
            else:
                if type(xlim[1]) == float or type(xlim[1]) == int:
                    plt.xlim(right=xlim[1])
                else:
                    pass
        
        # y_Limit
        if ylim and type(ylim) == list:
            if type(ylim[0]) == float or type(ylim[0]) == int:
                if type(ylim[1]) == float or type(ylim[1]) == int:
                    plt.ylim(bottom=ylim[0], top=ylim[1])
                else:
                    plt.ylim(bottom=ylim[0])
            else:
                if type(ylim[1]) == float or type(ylim[1]) == int:
                    plt.ylim(top=ylim[1])
                else:
                    pass

        # Grid
        if grid:
            plt.grid(alpha=0.2)
        # plt.show()

        result_obj[cx] = hist_fig   # Result save to object

    return result_obj




# violin_box_plot
def violin_box_plot(x=None, y=None, data=None, group=None, figsize=None,
    title=None, color=None, label=None, alpha=0.13, return_plot=True):
    if data is None:
        if len(x) != len(y):
            raise 'Different length error between x and y'
        else:
            violin_box_data = pd.concat([pd.Series(x).astype('str'), pd.Series(y)], axis=1)
            if x is None:
                try:
                    x = group.name
                except:
                    x = 'x'
            elif group is None:
                try:
                    x = x.name
                except:
                    x = 'x'
            try:
                y = y.name
            except:
                y = 'y'   
    else:
        if x is None:
            x = group
        elif group is None:
            x = x
        violin_box_data = data[[x,y]]
        
    violin_box_data.columns = [x, y]


    def decimal(x, rev=0):
        return 2 if x == 0 else int(-1*(np.floor(np.log10(abs(x)))-3-rev))
    
    def auto_decimal(x, rev=0):
        if np.isnan(x):
            return np.nan
        else:
            return round(x, decimal(x, rev=rev))

    def describe_string(x):
        mean = auto_decimal(x.mean())
        std = auto_decimal(x.std())
        return f'mean {mean},  std {std}'

    box_data_dict_ = {gi: np.array(gv[y]) for gi, gv in violin_box_data.groupby(x,dropna=True)}
    box_data_dict = {gi: np.array([np.nan, np.nan]) if len(gv) == 0 else gv for gi, gv in box_data_dict_.items()}
    
    box_describe_dict = {gi: describe_string(gv[y]) for gi, gv in violin_box_data.groupby(x)}

    if return_plot:
        fig = plt.figure(figsize=figsize)
    if title:
        plt.title(title)
    else:
        plt.title(f'{y} by {x} Violin Box Plot', fontsize=14)

    top_violin = plt.violinplot(box_data_dict.values(), showextrema=True, widths=0.7)

    if type(color) is list:
        assert len(color) == len(top_violin['bodies']), 'lengths are different'
        for tv, c in zip(top_violin['bodies'], color):
            tv.set_alpha(alpha)
            tv.set_facecolor(c)
    else:
        for tv in top_violin['bodies']:
            tv.set_alpha(alpha)
            if color is not None:
                tv.set_facecolor(color)
    top_violin['cbars'].set_color('none')
    top_violin['cmaxes'].set_color('none')
    top_violin['cmins'].set_color('none')
    
    box_props = {}
    box_props['meanprops'] = {'marker':'o', 'markerfacecolor':'red', 'markeredgecolor':'none'}
    if color is not None and type(color) is not list:
        box_props['boxprops'] = {'color': color}
        box_props['capprops'] = {'color': color}
        box_props['whiskerprops'] = {'color': color}
        box_props['meanprops']['markerfacecolor'] = color
        box_props['medianprops'] = {'color': color}
        
    top_box = plt.boxplot(box_data_dict.values(), labels=box_data_dict.keys(),
        showmeans=True, widths=0.2, **box_props)
    
    if violin_box_data[x].nunique() == 1:
        xlabel = f'{x}\n\n' + '\n'.join([f'({di})  {dv}' for di, dv in box_describe_dict.items()])
    elif violin_box_data[x].nunique() > 1:
        pvalues = sp.stats.f_oneway(*box_data_dict.values()).pvalue
        xlabel = f'{x}\n\n' + '\n'.join([f'({di})  {dv}' for di, dv in box_describe_dict.items()]) + f'\npvalues: {str(round(pvalues,3))}'
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(f'{y}')
    
    if return_plot == 'show':
        plt.show()
    elif return_plot is True:
        plt.close()
        return fig






# 데이터의 X, Y값에 따른 Scatter Plot, Histogram, Regression Line을 동시에 Display해주는 함수 ------------------------------------------------------
def fun_Group_OLS_Plot(df, y, x, group=[],
        figsize = [5,3],
        PointPlot=True, fitLine=True, histY=True, histX=True,
        specY=[], specX=[], spec_display='auto',
        lineX=[], lineY=[],
        xlim=False, ylim=False,
        alpha=0.7
        ):
    '''
    # 데이터의 X, Y값에 따른 Scatter Plot, Histogram, Regression Line을 동시에 Display해주는 함수

    < Input >
    df (DataFrame) : Input Raw Data
    y (Str) : Y variable
    x (Str) : X variable
    group (list) : Grouping variable List

    PointPlot, fitLine, histY, histX (Boolean) : Plot Display
    SpecY, SpecX (list) : Spec Line Display (it can be variable name)
    Spec_display ('auto', 'all') : if Spec Line has variable name, it display all? or auto?
    lineX, lineY (list) : subline Display
    xlim, ylim (list) : x, y axis Display Limit
    alpha(float number) : graph alpha

    < Output >
    Object['OLS'] (DataFrame) : OLS Regression Result Summary
    Object['plot'] (DataFrame) :  Plot, X_Histogram, Y_Histogram
        Object['plot']['scatter'] : scatter plot
        Object['plot']['histY'] : Y variable Histogram
        Object['plot']['histX'] : X variable Histogram
    '''
    df_plot = df.copy()
    if not group:
        df_plot['total'] = 'total'
        group = ['total']
    
    group = group if type(group) == list else [group]
    df_group = df_plot.groupby(group)    # Group

    if fitLine:
        groupOLS_Base = df_group.count().iloc[:,0].to_frame()   # Group Count
        groupOLS_Base.columns = [['Total'],['count']]
        groupOLS_df = fun_Concat_Group_OLS(base=groupOLS_Base, groupby=df_group, y=y, x=x, const=True)  # OLS Function
    
    if xlim:
        xlim_revision = (df_plot[x].max().item() - df_plot[x].min().item())*0.05
        xlim = [df_plot[x].min().item() - xlim_revision, df_plot[x].max().item() + xlim_revision]

    if ylim:
        ylim_revision = (df_plot[y].max().item() - df_plot[y].min().item())*0.05
        ylim = [df_plot[y].min().item() - ylim_revision, df_plot[y].max().item() + ylim_revision]


    result_Obj={}
    df_part_Object={}
    result_Plot = pd.DataFrame()

    for gi, gv in df_group:
        result_Plot_part = {}
        print(gi)
        print(len(gv))
        df_part_Object[gi] = gv

        spec_listX =[]
        spec_listY =[]
        line_x_list = []
        line_y_list = []
        if specY:
            if type(specY)==dict:
                try:
                    spec_listY = specY[gi]
                except:
                    spec_listY = []
            else:
                spec_listY = specY

        if specX:
            if type(specX)==dict:
                try:
                    spec_listX = specX[gi]
                except:
                    spec_listX = []
        else:
            spec_listX = specX

        if PointPlot:     # 1D OLS
            groupPlot = plt.figure(figsize=figsize)
            plt.scatter(x=x, y=y, data=gv, alpha=alpha, edgecolors='black', linewidth=0.5)

            if fitLine:
                if groupOLS_df.loc[gi].swaplevel(i=0,j=1)['nTrain'].values != '':
                    try:
                        df_part = groupOLS_df.loc[gi]
                        df_part = df_part.reset_index().drop('level_0', axis=1)
                        df_part = df_part.set_index('level_1').T
                        pred_y = df_part_Object[gi][x]*df_part['coef_' + x].values + df_part['coef_const'].values
                        plt.plot(gv[x], pred_y, 'r')
                        result_Obj['OLS'] = groupOLS_df
                        # pass
                    except:
                        pass

            if specY:
                for s in spec_listY:
                    plt.axhline(y=s, c='r', alpha =0.7, linestyle='--')
            if specX:
                for s in spec_listX:
                    plt.axvline(x=s, c='r', alpha =0.7, linestyle='--')
            if ylim:
                plt.ylim(bottom=ylim[0], top=ylim[1])
            if xlim:
                plt.xlim(left=xlim[0], right=xlim[1])

            plt.title(gi)
            plt.ylabel(y)
            plt.xlabel(x)
            plt.grid(alpha=0.2)
            plt.show()
            result_Plot_part['scatter'] = groupPlot

        # line Y
        if type(lineY) == dict:
            try:
                line_y_list = lineY[gi]
            except:
                line_y_list = []
        else:
            line_y_list = lineY

        # line X
        if type(lineX) == dict:
            try:
                line_x_list = lineX[gi]
            except:
                line_x_list = []
        else:
            line_x_list = lineX

        if histY:
            result_Plot_part['histY'] = fun_Hist(data=gv, x=y, figsize=figsize, title=gi, xtick=45, norm=True,
                                    color='mediumseagreen', alpha=0.7,
                                    spec=spec_listY, spec_display=spec_display, 
                                    line_x=line_y_list,
                                    xlim=ylim)[y]

        if histX:
            result_Plot_part['histX']  = fun_Hist(data=gv, x=x, figsize=figsize, title=gi, xtick=45, norm=True,
                                    color='skyblue', alpha=0.7,
                                    spec=spec_listX, spec_display=spec_display, 
                                    line_x=line_x_list,
                                    xlim=xlim)[x]
        result_Plot = pd.concat([result_Plot, pd.DataFrame([result_Plot_part], index=[gi]).T], axis=1)

    result_Obj['plot'] = result_Plot
    return result_Obj





from sklearn.linear_model import LinearRegression

def model_plot(X, y, model, xcols=None, model_evaluate=None, fitted_data=50,
            title=None, figsize=None, x_name=None, y_name=None, c=None, vmin=None, vmax=None,
            return_plot=True):
    if return_plot:
        fig = plt.figure(figsize=figsize)
    
    if model_evaluate is not None:
        me = model_evaluate
    else:
        me = ModelEvaluate(X, y, model, verbose=0)
    
    if title is None:
        title = str(model)
    
    if sum([i.lower() in str(model).lower() for i in ['LinearRegression', 'Ridge', 'Lasso', 'ElasticNet']]):
        plt.title(f'{title}\n{me.linear.formula}\n(r2: {me.metrics.r2_score}, rmse: {me.metrics.rmse}, mape: {me.metrics.mape}) \n pvalues: {me.hypothesis.pvalues}')
    else:
        plt.title(f'{title}\n(r2: {me.metrics.r2_score}, rmse: {me.metrics.rmse}, mape: {me.metrics.mape})')

    if not xcols:
        plot_dim = X.shape[1]
    else:
        plot_dim = len(xcols)

    if plot_dim == 1:
        x_array = np.array(X).ravel()
        y_array = np.array(y).ravel()
        
        if type(fitted_data) == int:
            x_min = x_array.min()
            x_max = x_array.max()
            x_interval = np.linspace(x_min, x_max, fitted_data)
        elif type(fitted_data) == list or type(fitted_data) == np.ndarray:
            x_interval = np.array(fitted_data)
        elif type(fitted_data) == dict:
            x_interval = np.array(list(fitted_data.values())[0])
        y_interval = model.predict(x_interval.reshape(-1,1))

        if c is None:
            plt.scatter(x_array, y_array, edgecolor='white', alpha=0.7)
        else:
            if 'int' in str(c.dtype) or 'float' in str(c.dtype):
                plt.scatter(x_array, y_array, edgecolor='white', alpha=0.7, c=c, vmin=vmin, vmax=vmax)
                plt.colorbar()
            # else:
            #     for gi, gv in 
        plt.plot(x_interval, y_interval, color='red', alpha=0.5)

        if x_name is None:
            try:
                x_name = X.columns[0]
            except:
                pass
        if y_name is None:
            try:
                x_name = y.name
            except:
                pass
        if y_name:
            plt.ylabel(y_name)
    elif plot_dim > 1:
        pass

    if x_name:
        plt.xlabel(x_name)

    if return_plot:
        return fig
        
    


# ------------------------------------
class FittedLine():
    '''
    < Input > 
     . x : Series or 1D-array
     . y : Series or 1D-array
    \n
    < Output >
     . model : instance of linear model
     . summary : summary
     . coef : coefficient parameters
     . formula : formula of linear-regression 
     . pvales : pvalues of each coefficient 
     . fitted_data : list type data to draw a fitted-plot
     . predict : predict function
    '''
    def __init__(self, x, y):
        self.train_x = x
        self.train_y = y

        x_add = sm.add_constant(x)
        x_min, x_max = pd.Series(x).agg(['min', 'max'])

        self.model = sm.OLS(y, x_add).fit()
        
        self.summary = self.model.summary()       # summary
        self.coef = pd.Series(self.model.params)[::-1].apply(lambda x: self.auto_decimal(x)).to_dict()     # coef
        try:
            self.y_name = y.name
        except:
            self.y_name = 'Y'
        try:
            self.x_name = x.name
        except:
            self.x_name = 'X'
        self.r2_score = round(self.model.rsquared,3)
        self.r2_score_adj = round(self.model.rsquared_adj,3)
        self.formula = self.y_name + ' = ' + ''.join([ f'{str(v)}·{k}' if i == 0 else (' + ' if v > 0 else ' - ') + (str(abs(v)) if k=='const' else f'{str(v)}·{abs(k)}') for i, (k, v) in enumerate(self.coef.items())])
        self.pvalues = pd.Series(self.model.pvalues)[::-1].apply(lambda x: round(x,4)).to_dict()       # p-values

        x_linspace = sm.add_constant(pd.Series(np.linspace(x_min, x_max, 50), name=list(self.pvalues.keys())[0]))
        self.fitted_data = [list(x_linspace.iloc[:,1]), list(self.model.predict(x_linspace))]
        # return self
    
    def decimal(self, x, rev=0):
        return 2 if x == 0 else int(-1*(np.floor(np.log10(abs(x)))-3-rev))

    def auto_decimal(self,x, rev=0):
        if np.isnan(x):
            return np.nan
        else:
            return round(x, self.decimal(x, rev=rev))

    def predict(self, x, decimal_revision=True):
        x_add = sm.add_constant(x)
        if decimal_revision:
            return self.model.predict(x_add).apply(lambda x: auto_decimal(x))
        else:
            return self.model.predict(x_add)
    
    def plot(self, figsize=None, return_plot=True):
        self.fig = plt.figure(figsize=figsize)
        plt.title(f'formula: {self.formula}  (r2: {self.r2_score}) \n pvalues: {self.pvalues}')
        plt.scatter(self.train_x, self.train_y, edgecolor='white', alpha=0.7)
        plt.plot(*self.fitted_data, color='red', alpha=0.5)
        plt.ylabel(self.y_name)
        plt.xlabel(self.x_name)
        
        if return_plot:
            plt.close()
            return self.fig
        else:
            plt.show()
        


# ------------------------------------
class FittedModel():
    '''
    < Input > 
     . x : Series or 1D-array
     . y : Series or 1D-array
    \n
    < Output >
     . model : instance of linear model
     . summary : summary
     . coef : coefficient parameters
     . formula : formula of linear-regression 
     . pvales : pvalues of each coefficient 
     . fitted_data : list type data to draw a fitted-plot
     . predict : predict function
    '''
    def __init__(self, X, y, model='LinearRegression', model_type=None, fitted_point=50, print_summary=False):
        
        self.X_shape = X.shape
        self.X_dim = 1 if (np.array(X).ndim == 1 or (np.array(X).ndim == 2 and self.X_shape[1] == 1) ) else 2

        y_frame = self.series_to_frame(y)
        X_frame = self.series_to_frame(X)
        Xy_frame = pd.concat([y_frame, X_frame], axis=1).dropna()
        
        # DataSet
        self.X = Xy_frame[[Xy_frame.columns[1]]]
        self.y = Xy_frame[Xy_frame.columns[0]]
        
        # ColumnName
        try:
            self.y_name = self.y.name
        except:
            self.y_name = 'Y'
        if self.X_dim == 1:
            try:
                self.x_name = self.X.columns[0]
            except:
                self.x_name = 'X'
        elif self.X_dim == 2:
            self.x_name = list(self.X.columns)
        
        if type(model) == str:
            exec(f'self.model = {model}()')
        else:
            self.model = model
        
        # Model
        self.model_name = str(model)
        self.model.fit(self.X, self.y)
        self.model_evaluate = ModelEvaluate(self.X, self.y, self.model, model_type=model_type, verbose=0)
        self.sum_square = self.model_evaluate.sum_square
        self.metrics = self.model_evaluate.metrics
        try:
            self.hypothesis = self.model_evaluate.hypothesis
            self.linear = self.model_evaluate.linear
        except:
            pass
        
        # FittedData
        self.fitted_point = fitted_point

        if self.X_dim == 1:
            x_min = np.array(self.X).min()
            x_max = np.array(self.X).max()
            x_interval = np.linspace(x_min, x_max, fitted_point)

        y_interval = self.model.predict(x_interval.reshape(-1,1))
        
        if self.X_dim == 1:
            self.fitted_data = np.concatenate([x_interval.reshape(-1,1), y_interval.reshape(-1,1)], axis=1).T

        if print_summary:
            print(' .', self.sum_square) 
            print(' .', self.metrics) 
            try:
                print(' .', self.hypothesis) 
                print(' .', self.linear) 
            except:
                pass      

    def series_to_frame(self, X):
        X_shape = X.shape
        X_dim = 1 if (np.array(X).ndim == 1 or (np.array(X).ndim == 2 and X_shape[1] == 1) ) else 2

        # DataSet
        if np.array(X).ndim == 1:
            X = pd.Series(X).to_frame()
        else:
            X = pd.DataFrame(X)
            if type(X) == np.ndarray:
                X.columns = ['x'+str(i+1) for i in np.arange(X_shape)]
        return X

    def decimal(self, x, rev=0):
        return 2 if x == 0 else int(-1*(np.floor(np.log10(abs(x)))-3-rev))

    def auto_decimal(self,x, rev=0):
        if np.isnan(x):
            return np.nan
        else:
            return round(x, self.decimal(x, rev=rev))

    def predict(self, X, decimal_revision=True):
        X = self.series_to_frame(X)

        if decimal_revision:
            return self.model.predict(X).apply(lambda x: auto_decimal(x))
        else:
            return self.model.predict(X)
    
    def plot(self, X=None, y=None, figsize=None, model_evaluate=None, fitted_data=None,
            title=None, c=None, vmin=None, vmax=None,
            return_plot=True):
        if X is None:
            X = self.X
        else:
            X = self.series_to_frame(X)
        if y is None:
            y = self.y
        if model_evaluate is None:
            model_evaluate = self.model_evaluate
        if fitted_data is None:
            fitted_data = self.fitted_point
        if title is None:
            title = self.model_name

        if return_plot:
            self.fig = plt.figure(figsize=figsize)

        model_plot(X=X, y=y, model=self.model, model_evaluate=model_evaluate, fitted_data=fitted_data,
                title=title, x_name=self.x_name, y_name=self.y_name, c=c, vmin=vmin, vmax=vmax, 
                return_plot=False)

        if return_plot:
            plt.close()
            return self.fig
        else:
            pass
            # plt.show()







