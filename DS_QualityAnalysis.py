import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
try:
    font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
    rc('font', family=font_name)
excpet:
    pass
import seaborn as sns

import re
import tqdm
import datetime
from IPython.core.display import display, HTML
import copy

from bs4 import BeautifulSoup

from DS_DataFrame import *
from DS_Plot import distbox


# today_date
dir(datetime.datetime)
today = datetime.datetime.today()
today_str = str(today.year)[2:] + '. ' + str(today.month) + '/' + str(today.day)
today_date = str(today.year)[2:] + format(str(today.month), '0>2s') + format(str(today.day), '0>2s')




# And So On ###############################################################################################################################
# 품질관제 모니터링 format
def quality_monitoring_format(data):
    colsdd = {'품명':'주문품명코드', '규격약호':'제품규격약호', '출강목표':'출강목표번호', 
        '주문두께_이상':'제품주문두께', '주문두께_미만':'제품주문두께', '주문폭_이상':'제품주문폭', '주문폭_미만':'제품주문폭',
        '고객사코드':'고객사코드', 'MainKey_번호':'품질설계MainKeyNo', '사내보증번호':'제품사내보증번호',
        '고객요구번호':'고객품질요구기준번호', '공장공정':'공장공정코드', '국가코드':'국가코드'}
    df = data.copy()
    dfT = df[df.iloc[3,:].dropna().index].T
    dfT.index = [colsdd[k] for k in dfT.index]
    dfT2 = dfT.reset_index()
    dfT2.columns = ['표준항목명(DD)', '표준항목 ID', 'Data Type', '기호', '값']
    return dfT2


# Steel Compoent Formula ###############################################################################################################################

# Calculate Component Score
def calc_soluted_Nb_proba(C, SolAl, Nb, N, **kwargs):
    return Nb - 7.75*C - 6.65*(N/10000 - SolAl/1.93)

def calc_solute_Nb_temp(C, Nb, N, **kwargs):
    if 'Series' in str(type(C))  and 'Series' in str(type(Nb)) and 'Series' in str(type(N)):
        return -10400/((Nb * (N/10000)**0.65 * C ** 0.24).apply(lambda x: np.nan if np.isclose(x, 0) else np.log10(x)) -4.09) -273.5
    elif 'float' in str(type(C))  and 'float' in str(type(Nb)) and 'float' in str(type(N)) :
        if np.isclose(Nb * N * C, 0):
            return np.nan
        else:
            return -10400/(np.log10((Nb * (N/10000)**0.65 * C ** 0.24)) -4.09) -273.5
    else:
        return np.nan

def calc_theroical_rolling_TS(C, Si, Mn, SolAl, Cu, Nb, Cr, Mo, V, slab_thick=250, product_thick=0.6, **kwargs):
    return ( (24.5 if product_thick >= 8 else 25) + 70*C + 13*Si + 8.7*Mn + 5*Cr + 13*Mo + 7*Cu 
        + 0.086*(slab_thick/product_thick) - 0.066*product_thick 
        + 22*V + 165*Nb 
        + (SolAl.apply(lambda x: 1 if x >=0.015 else 0) if 'Series' in str(type(SolAl)) else 1 if SolAl >= 0.015 else 0)
        - 0.2 + 3*np.log10(product_thick/10)**2 ) * 9.80665


# 용접 Crack관련
# **Fp=2.5*(0.5-(C+0.04Mn-0.14Si+0.1Ni-0.04Cr-0.1Mo-0.24Ti+0.7N))
# **CP= C + 0.04Mn + 0.1Ni + 0.7N - 0.14Si-0.04Cr-0.1Mo-0.24Ti

def calc_RST(C, Si, SolAl, Nb, Ti, V, **kwargs):
    return 887 + 464*C + (6445*Nb - 644*np.sqrt(Nb)) + (732*V - 230*np.sqrt(V)) + 890*Ti + 363*SolAl - 357*Si

def calc_Ar3(C, Mn, Cu, Ni, Cr, Mo, product_thick=0.6, **kwargs):
    return 910 - 310*C - 80*Mn - 20*Cu - 15*Cr - 55*Ni - 80*Mo - 0.35*(product_thick-8)
    # (NEW) AR3 = 910-310XC-80XMn-20Cu-15Cr-55Ni-80Mo+0.35(t-8)

def calc_Ar1(Si, Mn, Ni, Cr, **kwargs):
    return 723 - 10.7*Mn - 16.9*Ni - 29.1*Si + 16.9*Cr

def calc_Ac3(C, Si, Ni, Mo, V, **kwargs):
    return 910 - 203*np.sqrt(C) - 15.2*Ni + 44.7*Si + 104*V + 31.5*Mo
    # (NEW) AC3 = 912-203X√C-30XMn-15.2XNi-11XCr+44.7Si+31.5XMo-20XCu+13.1W+104V+120As+400Ti+400Al+700P

def calc_Ac1(Si, Mn, Ni, Cr, **kwargs):
    return 723 - 10.7*Mn - 16.9*Ni - 29.1*Si + 16.9*Cr

def calc_Bs(C, Mn, Ni, Cr, Mo, **kwargs):
    return 830 - 270*C - 90*Mn -37*Ni - 70*Cr -83*Mo

def calc_Ms(C, Mn, Ni, Cr, Mo, **kwargs):
    return 539 - 423*C - 30.4*Mn - 17.7*Ni - 12.1*Cr - 7.5*Mo

def calc_CEQ(C=None, Si=None, Mn=None, Cu=None, Nb=None, B=None, Ni=None, Cr=None, Mo=None, Ti=None, V=None, N=None,
    code='G', **kwargs):
    if code == 'A': return C + Mn/6
    elif code == 'B': return C + Mn/10
    elif code == 'C': return C + Mn/6 + Si/24
    elif code == 'D': return C + Mn/6 + Si/24 + Cr/5 + V/14
    elif code == 'E': return C + Mn/6 + Si/24 + Cr/5 + V/14 + Ni/40
    elif code == 'F': return C + Mn/6 + Si/24 + Cr/5 + V/14 + Ni/40 + Mo/4
    elif code == 'G': return C + Mn/6 + (Ni + Cu)/15 + (Cr + Mo + V)/5
    elif code == 'H': return C + Mn/6 + Cr/10 + Mo/50 + V/10 + Ni/20 + Cu/40
    elif code == 'J': return C + Mn/6 + Si/24 + Ni/40 + Mo/4 + V/14
    elif code == 'K': return C + Mn/3
    elif code == 'L': return C + Mn/4
    elif code == 'M': return C + Mn/8
    elif code == 'N': return C + (Mn + Si)/4
    elif code == 'P': return C + Mn/6 + Si/24 + Ni/40 + Cr/5 + Mo/4 + V/14 + Cu/13
    elif code == 'Q': return C + Mn/6 + (Cr + Mo + V)/5 + (Ni + Cr)/15
    elif code == 'R': return C + (Mn + Si + Cr + Mo)/6 + (Ni + Cu)/16
    elif code == 'S': return C + Mn/5
    elif code == 'T': return C + Mn/6 + (Cu + Ni)/15 + (Cr + Mo + V)/5 + Si/24
    # elif code == 'U': return C + Mn/4 + Cr/10 ? V/10 + Ni/20 + Cu/20 ? Mo/50
    # elif code == 'V': return C + ※F * {Mn/6 + Si/24 + Cu/15 + Ni/20 + (Cr+Mo+V+Nb)/5 + 5B}
    elif code == 'W': return C + Mn/20 + Si/30 + Ni/60 + Cr/20 + Mo/15 + V/10 + Cu/20 + 5*B
    elif code == 'X': return C + Mn/6 +(Cr+Ti+Mo+Nb+V)/5 + (Ni+Cu)/15 + 15*B
    elif code == 'Y': return C + Si/6 + Mn/4.5 + Cu/15 + Ni/15 + Cr/4 + Mo/2.5 + 1.8*V
    # elif code == 'Z': return Al/N



# class SteelGrdInfo : 강재성분정보 정리해주는 Class
class SteelGrdInfo():
    def __init__(self, key='index', sg_column_names=['출강목표', '출강목표N'],
        colnames = {'rst':'실적', 'aim': '목표', 'min': '하한', 'max': '상한'},
        component=['C', 'Si', 'Mn', 'P', 'S', 'SolAl', 'TotAl', 'Cu', 'Nb',
                    'B', 'Ni', 'Cr', 'Mo', 'Ti', 'V', 'Sn', 'Ca', 'Sb', 'N', 'As'],
        temp=['RST', 'Ar3', 'Ar1', 'Ac3', 'Ac1', 'Bs', 'Ms'],
        other=['theroical_rolling_TS', 'solute_Nb_temp', 'soluted_Nb_proba', 'CEQ'],
        ceq_code='G'):

        self.key = key
        self.sg_column_names = sg_column_names
        self.colnames = colnames
        self.component = component
        self.temp = temp
        self.other = other
        self.ceq_code = ceq_code
        self.result_dict = None
    
    def analysis(self, data, steel_grd=None, key=None, sg_column_names=None, 
                colnames=None, component=None, temp=None, other=None, ceq_code=None, display_summary=True):
        if key is None:
            key = self.key
        if sg_column_names is None:
            sg_column_names = self.sg_column_names
        if colnames is None:
            colnames = self.colnames
        if component is None:
            component = self.component
        component_upper = list(map(lambda x: x.upper(), component))
        if temp is None:
            temp = self.temp
        if other is None:
            other = self.other
        if ceq_code is None:
            ceq_code = self.ceq_code

        if 'Series' in str(type(data)):
            data_copy = data.to_frame().T
        else:
            data_copy = data.loc[data[sg_column_names].drop_duplicates().index,:]
            # data_copy = data_copy.reset_index(drop=True)

        if steel_grd is not None:
            if type(steel_grd) != list:
                steel_grd = [steel_grd]
            data_copy = data_copy[ (pd.Series(data_copy.index, index=data_copy.index).isin(steel_grd)) |
                        (data_copy[sg_column_names].isin(steel_grd).sum(1).apply(lambda x: False if x==0 else True))]

        result_dict = {}

        p1 = re.compile('[a-zA-Z0-9\u3131-\u3163\uac00-\ud7a3]+')   # (Regex) 영어소문자, 대문자, 숫자, 한글

        for data_idx, data_row in tqdm.tqdm_notebook(data_copy.iterrows()):
            data_row_DF = data_row.to_frame().T

            if key == 'index':
                dict_idx = data_idx
            else:
                dict_idx = data_row[key]
            result_dict[dict_idx] = {}

            for ck, cv in colnames.items():
                p2 = re.compile(cv)     #  (Regex)
                try:
                    # component
                    # column_name = list(map(lambda x: x+'_'+ck, component))
                    column_match1 = np.array( list(map(lambda x: type(p2.search(x)) == re.Match, data_row_DF.columns)) )
                    column_match2 = np.array( list(map(lambda x: ''.join(p1.findall(x)).replace(cv,'').upper() in component_upper, data_row_DF.columns)) )
                    column_name = list(data_row_DF.columns[column_match1 & column_match2])

                    temp_comp = data_row_DF[column_name].applymap(lambda x: str(round(x, fun_Decimalpoint(x)))).T.reset_index(drop=True)
                    temp_comp.columns = [ck]
                    temp_comp.index = component
                    # comp_df = pd.concat([comp_df,temp_comp], axis=1)
                    
                    result_dict[dict_idx][ck] = {}
                    for sg in sg_column_names:
                        try:
                            result_dict[dict_idx][ck][sg] = data_row_DF[sg].iloc[0]
                        except:
                            pass
                    result_dict[dict_idx][ck].update(temp_comp.to_dict()[ck])

                    # temperature
                    for t in temp:
                        result_dict[dict_idx][ck][t] = int(eval('calc_' + t + '(**temp_comp[ck].astype(float))'))

                    # other:
                    for o in other:
                        if o == 'CEQ':
                            code = ceq_code if len(ceq_code) == 1 else data_row[ceq_code]
                            calc_other = eval("calc_" + o + "(code='" + code + "', **temp_comp[ck].astype(float))")
                        else:
                            calc_other = eval('calc_' + o + '(**temp_comp[ck].astype(float))')

                        if o == 'soluted_Nb_proba' or o == 'CEQ':
                            append_other = str( round(calc_other, fun_Decimalpoint(calc_other)) )
                        else:
                            append_other = str(int(calc_other))
                        result_dict[dict_idx][ck][o] = append_other
                except:
                    pass
        
        self.steel_idx = data_copy[sg_column_names]

        self.result_dict = result_dict
        result_dict_1st_key = list(result_dict.keys())
        result_dict_2nd_key = list(result_dict[result_dict_1st_key[0]].keys())
        result_dict_3rd_key = list(result_dict[result_dict_1st_key[0]][result_dict_2nd_key[0]])

        self.result_dict_keys = {}
        self.result_dict_keys['1st'] = result_dict_1st_key
        self.result_dict_keys['2nd'] = result_dict_2nd_key
        self.result_dict_keys['3rd'] = result_dict_3rd_key
        
        self.result_frame_keys = pd.Series(self.result_dict_keys).to_frame()
        self.result_frame_keys.columns = ['keys']

        if display_summary:
            self.show_summary()

    def show_summary(self):
        print(' (Key_Info) self.result_dict_keys  (Key_Frame) self.result_frame_keys\n (Data) self.result_dict')
        print_DataFrame(self.result_frame_keys)

    def steel_info(self, steel_grd=None, mode='vertical', contains=['other', 'temp', 'component'], data=None):
        if self.result_dict is None:
            self.analysis(data=data, steel_grd=steel_grd, display_summary=False)

        if type(steel_grd) != list:
            steel_grd = [steel_grd]
        search_idx_df = self.steel_idx[ (pd.Series(self.steel_idx.index, index=self.steel_idx.index).isin(steel_grd)) |
                                    (self.steel_idx[self.sg_column_names].isin(steel_grd).sum(1).apply(lambda x: False if x==0 else True))]

        if self.key == 'index':
            search_idx = search_idx_df.index[0]
        else:
            search_idx = search_idx_df[self.key].iloc[0]

        steel_info_data = pd.DataFrame(self.result_dict[search_idx])

        colexist_rst = True if 'rst' in steel_info_data.columns else False
        colexist_aim = True if 'aim' in steel_info_data.columns else False
        colexist_min = True if 'min' in steel_info_data.columns else False
        colexist_max = True if 'max' in steel_info_data.columns else False

        if colexist_min and colexist_max:
            steel_info_data['range'] = steel_info_data.apply(lambda x: x['min'] if x.name in self.sg_column_names else 
                                            (('' if float(x['min']) == 0 or np.isnan(float(x['min'])) else str(x['min']) )
                                            + ('' if float(x['max']) == 0 or np.isnan(float(x['max'])) else ' ~ '  + str(x['max']) )
                                            ), axis=1)

        p1 = re.compile('[a-zA-Z0-9\u3131-\u3163\uac00-\ud7a3]+')   # (Regex) 영어소문자, 대문자, 숫자, 한글

        # steel_info_steelgrd = steel_info_data.loc[self.sg_column_names,:].iloc[:,0].to_frame().T.reset_index(drop=True)
        steelgrd = ': '.join(p1.findall(str(steel_info_data.loc[self.sg_column_names,:].iloc[:,0].to_dict()).replace(',','ㅇ'))).replace(': ㅇ: ', '\n')
        steel_info_steelgrd = pd.Series([np.nan] * (steel_info_data.shape[1])).to_frame().T
        steel_info_steelgrd.columns = steel_info_data.columns
        steel_info_steelgrd.index = [steelgrd]

        steel_info_component = steel_info_data.loc[self.component,:]
        steel_info_temp = steel_info_data.loc[self.temp,:]
        steel_info_other = steel_info_data.loc[self.other,:]

        if colexist_min and colexist_max:
            steel_info_steelgrd.drop(['min','max'], axis=1, inplace=True)
            steel_info_component.drop(['min','max'], axis=1, inplace=True)
            steel_info_temp.drop(['min','max'], axis=1, inplace=True)
            steel_info_other.drop(['min','max'], axis=1, inplace=True)

        self.steel_summary = pd.DataFrame()


        if mode == 'vertical' or mode == 'v':
            self.steel_summary = steel_info_steelgrd
            for c in contains:
                self.steel_summary = eval('self.steel_summary.append(add_row(steel_info_' + c + '))')
            
            self.steel_info_other = steel_info_other
            self.steel_info_temp = steel_info_temp
            self.steel_info_component = steel_info_component

        elif mode == 'horizontal' or mode == 'h':
            height = 4 + steel_info_other.shape[1] + steel_info_component.shape[1]
            width = max(len(steel_info_other) + len(steel_info_temp), len(steel_info_component))+1
            
            steel_summary_matrix = np.zeros((height, width))
            steel_summary_matrix = steel_summary_matrix.astype(str)
            steel_summary_matrix[steel_summary_matrix == '0.0'] = ''

            steel_summary_matrix[0,0] = list(steel_info_steelgrd.index)[0]

            self.steel_info_other = steel_info_other.T
            self.steel_info_temp = steel_info_temp.T
            self.steel_info_component = steel_info_component.T
            self.steel_info_othertemp = steel_info_other.append(steel_info_temp).T
            
            sot_h, sot_w = self.steel_info_othertemp.shape
            print(sot_h, sot_w)
            steel_summary_matrix[2:2+sot_h, 0] = np.array(self.steel_info_othertemp.index)
            steel_summary_matrix[1, 1:1+sot_w] = np.array(self.steel_info_othertemp.columns)
            steel_summary_matrix[2:2+sot_h, 1:1+sot_w] = self.steel_info_othertemp

            sc_h, sc_w = self.steel_info_component.shape
            print(sc_h, sc_w)
            steel_summary_matrix[4+sot_h:4+sot_h+sc_h, 0] = np.array(self.steel_info_component.index)
            steel_summary_matrix[3+sot_h, 1:1+sc_w] = np.array(self.steel_info_component.columns)
            steel_summary_matrix[4+sot_h:4+sot_h+sc_h, 1:1+sc_w] = self.steel_info_component

            self.steel_summary = pd.DataFrame(steel_summary_matrix)

        print(' → (Summary): self.steel_summary')
        if mode == 'vertical' or mode == 'v':
            print('   (Other_Infomation): self.steel_info_other')
            print('   (Temperature): self.steel_info_temp')
        elif mode =='horizontal' or mode == 'h':
            print('   (Other/Temperature_Infomation): self.steel_info_othertemp')
            print('   (Other_Infomation): self.steel_info_other, (Temperature): self.steel_info_temp')
        print('   (Component): self.steel_info_component')

        return self.steel_summary

    def __call__(self, steel_grd=None, mode='vertical', contains=['other', 'temp', 'component'], data=None):
        return self.steel_info(steel_grd=steel_grd, mode=mode, contains=contains, data=data)
        

# Special Test ###############################################################################################################################

# from bs4 import BeautifulSoup
# 특별시험 HTML Script에서 Coil번호를 추출해주는 함수
def extract_coils(html=None, html_from_clipboard=False, first_filters=['H', 'C']):
    if html is None and html_from_clipboard is True:
        html = pyperclip.paste()
    soup = BeautifulSoup(html)
    soup_result1 = soup.find_all('div', {'class', 'grid-input-render__field'})      # coil_no
    soup_result2 = soup.find_all('div', {'class', 'ag-custom-select-list'})         # LOC: T/B
    soup_result3 = soup.select("div[col-id='matSpcTeTePicStkDt']")                  # Date
    # soup_result3 = soup.find_all('div', {'class', 'ag-cell ag-cell-not-inline-editing ag-cell-with-height ag-cell-value ag-cell-range-right'})    # Date
    
    series_coil_no = pd.Series(list(filter(lambda y: len(y) > 0 and y[0][0] in first_filters, map(lambda x: x.contents, soup_result1)) ), name='coil_no').apply(lambda x: x[0])
    n_list = []
    for e in soup_result1:
        try:
            n_list.append( int(e.getText()) )
        except:
            pass
    series_n = pd.Series(n_list, name='n')                                      # coil_no
    series_loc = pd.Series([e.getText() for e in soup_result2], name='loc')     # LOC: T/B
    series_date = pd.Series([e.getText() for e in soup_result3 if '-' in e.getText() ], name='date')   # Date
    # series_date = pd.Series([e.getText() for e in soup_result3], name='date')   # Date
    
    result = pd.concat([series_coil_no, series_n, series_loc, series_date], axis=1, ignore_index=True)
    result.columns = ['coil_no','n', 'loc', 'date']
    # coils.columns = ['CoilNo']
    # coils = coils.sort_index()
    result.to_clipboard()
    return result


# Special Test ###############################################################################################################################

# 특별시험 인장실적 정리
def special_test(data, mode='tensile', reverse=True, dir_cd='-', loc={'001':'WS','002':'1W','003':'2W','004':'3W','005':'DS'}):
    '''
    dir_cd = 'C04'
    loc = {'001':'WS','002':'1W','003':'2W' ...}
    '''
    special_name = {'재질인장시험실적YP': 'YP', '인장시험Upper_YP실적치': 'YP_U', '인장시험Low_YP실적치': 'YP_L',
                '인장시험YP02실적치': 'YP_02', '인장시험YP05실적치': 'YP_05', '인장시험YP1실적치': 'YP_1',
                '재질인장시험실적TS': 'TS', '재질인장시험실적EL': 'EL', '인장시험RA실적치': 'TS_RA',
                '인장시험YR실적치': 'YR', '인장시험YP_EL실적치': 'YP_EL', '인장시험영율실적치': 'YM',
                '인장시험N가공경화지수': 'TN', '인장시험U_EL실적치': 'U_EL',
                '재질구멍확장성시험실적평균구멍확장률': 'HER_평균', '재질구멍확장성시험실적구멍확장률1': 'HER_1', '재질구멍확장성시험실적구멍확장률2': 'HER_2',
                '재질구멍확장성시험실적구멍확장률3': 'HER_3', '재질구멍확장성시험실적구멍확장률4': 'HER_4', '재질구멍확장성시험실적구멍확장률5': 'HER_5'}

    # df_special = pd.read_clipboard()
    df_special = data.copy()
    df_special_1 = df_special.rename(columns=special_name)

    
    if mode.lower() == 'tensile':
        df_special_1.insert(loc=0, column='시편_SEQ', value=df_special_1['시편번호'].apply(lambda x: str(x[-3:])))
        if loc is not None:
            if type(loc) == list:
                df_special_1['위치'] = df_special_1['시편_SEQ'].apply(lambda x: loc[x-1])
                df_special_1['위치'] = pd.Categorical(df_special_1['위치'], categories=loc, ordered=True)
            if type(loc) == dict:
                df_special_1['위치'] = df_special_1['시편_SEQ'].apply(lambda x: loc[str(x)])
                df_special_1['위치'] = pd.Categorical(df_special_1['위치'], categories=list(set(loc.values())), ordered=True)
                           
            if  type(dir_cd) == str:
                df_special_1['인장_방향호수'] = dir_cd
            elif type(dir_cd) == dict:
                df_special_1['인장_방향호수'] = df_special_1['시편_SEQ'].apply(lambda x: dir_cd[str(x)])
                df_special_1['인장_방향호수'] = pd.Categorical(df_special_1['인장_방향호수'], categories=list(set(dir_cd.values())), ordered=True)

    df_special_1.insert(loc=0, column='시험위치L', value=df_special_1['시편번호'].apply(lambda x: x[-4:-3]))
    
    if df_special_1['시험위치L'].apply(lambda x: x in ['T','M','B']).product() == 0:
        df_special_1['시험위치L'] = df_special_1['채취\n위치']
    # df_special_1.insert(loc=0, column='시험위치L', value=df_special_1['시편번호'].apply(lambda x: 'T' if 'T' in x else 'B'))
    
    if reverse:
        df_special_1['시험위치L'] = df_special_1['시험위치L'].apply(lambda x: 'B' if x == 'T' else ('T' if x == 'B' else x))
    df_special_1.insert(loc=0, column='재료번호', value=df_special_1['시편번호'].apply(lambda x: x[:-3]))
    
    df_special_2 = df_special_1.drop(['시편번호','채취\n위치','시험\n항목'],axis=1)
    # df_special_2 = df_special_1.drop(['시편번호','부위','MODE'],axis=1)

    if mode.lower() == 'tensile':
        df_special_3 = df_special_2.set_index(['재료번호','시험위치L','인장_방향호수', '위치'])
    elif mode.lower() == 'her':
        df_special_3 = df_special_2.set_index(['재료번호','시험위치L'])
        return df_special_3
    
    df_special_4 = df_special_3.dropna(axis=1, how='all')
    df_special_4.sort_index()
    
    # return df_special_4
    df_result = pd.DataFrame()
    for v in df_special_4.columns:
        df_unstack = df_special_4[v].unstack(['위치', '인장_방향호수'])
        df_unstack.columns  = pd.MultiIndex.from_tuples([tuple([v] + list(c)) for c in df_unstack.columns], names=['Content', 'DIR_CD', 'LOC'])
        df_result = pd.concat([df_result, df_unstack], axis=1)
        # df_result.columns.names = ['Content', 'DIR_CD', 'LOC']
        
    # print_DataFrame(df_result)
    # df_result.to_clipboard()
    df_result = df_result.sort_values(['재료번호','시험위치L'], axis=0, ascending=[True, False])

    return df_result



# Reject Manager ###############################################################################################################################

# Reject Data Summary
# function reject_summary
def reject_summary(df):
    df_reject_result = pd.DataFrame()
    df_reject01 = df[(~df['재질시험_대표구분'].isna()) & (df['이상재'].str.contains('Q'))]
    df_reject01.reset_index(inplace=True)

    df_reject_result['등록일'] = np.array([today_str]).repeat(len(df_reject01))
    df_reject_result['강종_구분'] = df_reject01['강종_소구분']
    df_reject_result['품종'] = df_reject01['품종명']
    df_reject_result['출강목표'] = df_reject01['출강목표']  + ' /\n ' + df_reject01['출강목표N']
    df_reject_result['위치'] = df_reject01['시험위치L']
    df_reject_result['불량유형'] = df_reject01['이상재'].apply(lambda x: x.split()[1].replace('(','').replace(')', ''))
    df_reject_result['재질값/Cpk'] = ''
    df_reject_result['재시험'] = ''
    df_reject_result['MainKey_No'] = df_reject01['MainKey_번호']

    df_reject_result['열연제조표준'] = df_reject01['열연제조표준'] + ' /\n ' + df_reject01['열연제조표준N'] 
    df_reject_result['냉연제조표준'] = df_reject01['냉연제조표준'] + ' /\n ' + df_reject01['냉연제조표준N'] 
    df_reject_result['사내보증번호'] = df_reject01['사내보증번호'] + ' /\n ' + df_reject01['사내보증번호N']
    df_reject_result['고객요구번호'] = df_reject01['고객요구번호'].apply(lambda x: '' if pd.isna(x) else x) + ' / ' + df_reject01['고객사명']
    df_reject_result['Size'] = df_reject01.apply(lambda x: str(x['주문두께']) + 't × '+ str(format(x['주문폭'], ',')) + 'w', axis=1)
    df_reject_result['소둔공장'] = df_reject01['소둔공장']
    df_reject_result['Mode변경재'] = df_reject01['Mode변경_구분']
    # df_reject_result['Mode변경재'] = df_reject01['Mode변경재선정']
    df_reject_result['현상/원인개선'] = ''
    df_reject_result['Coil_NO'] = df_reject01['냉연코일번호']
    df_reject_result.set_index('Coil_NO', inplace=True)
    return df_reject_result
    



# Reject product Dataset Split
# class RejectDataset
class RejectDataset():
    def __init__(self, data_analysis, data_history, dropna_analysis=None, dropna_history=['재질시험_대표구분']):
        if dropna_history:
            self.data_analysis = data_analysis[(~data_analysis[dropna_history].isna()).sum(1).astype(bool)]
        else:
            self.data_analysis = data_analysis
        self.data_history = data_history
        self.dropna_analysis = dropna_analysis
        if dropna_analysis is not None and type(dropna_analysis) == list:
            self.data_history = data_history.loc[data_history[dropna_analysis].dropna().index,:]
            # self.data_history = data_history.loc[data_history[(data_history[dropna_analysis] == 0).sum(1) > 0].index,:]

    def make_dataset(self, 
        filters=['출강목표Group', '출강목표', '열연제조표준', '냉연제조표준N', '소둔공장'],
        group=[], dropna_analysis=None, name=['냉연코일번호', '시험위치L'], filter_condition='auto'):
        # insp_loc=['시험위치L'],
        # , '제품사내보증번호', '고객품질요구기준번호'

        data_dict = {}
        data_shape_dict = {}
        datagroup_shape_dict = {}

        if dropna_analysis is None:
            dropna_analysis = self.dropna_analysis
        if dropna_analysis is not None and type(dropna_analysis) == list:
            data_history = self.data_history.loc[self.data_history[dropna_analysis].dropna().index,:]
            # self.data_history = data_history.loc[data_history[(data_history[dropna_analysis] == 0).sum(1) > 0].index,:]

        for i, analysis_row in self.data_analysis.iterrows():
            # i : enumerate_index
            # analysis_row : Series

            row_name = '_'.join(analysis_row[name].values)
            data_dict[row_name] = {}
            data_dict[row_name]['all'] = {}
            data_dict[row_name]['group'] = {}
            data_shape_dict[row_name] = {}
            datagroup_shape_dict[row_name] = {}
            # data_history_temp = self.data_history[self.data_history[insp_loc[0]] == analysis_row[insp_loc[0]]]
            data_history_temp = self.data_history.copy()

            for f in filters:
                if f == '출강목표Group':
                    f_name = '출강목표N'
                    f_col = analysis_row['출강목표N'][:-1]
                    if f_col is np.nan:
                        data_history_temp = data_history_temp[data_history_temp[f_name].isna()]
                    else:
                        data_history_temp = data_history_temp[data_history_temp[f_name].str.contains(f_col).apply(lambda x: False if np.isnan(x) else x)]

                elif 'object' in str(data_history_temp[f].dtypes):
                    f_name = f
                    f_col = analysis_row[f]
                    if f_col is np.nan:
                        data_history_temp = data_history_temp[data_history_temp[f_name].isna()]
                    else:
                        data_history_temp = data_history_temp[data_history_temp[f_name].str.contains(f_col).apply(lambda x: False if np.isnan(x) else x)]
                
                elif 'float' in str(data_history_temp[f].dtypes) or 'int' in str(data_history_temp[f].dtypes):
                    f_name = f
                    
                    def range_to_list(target):
                        return [ (-np.inf if i==0 else np.inf) if str(e) == '' else e for i, e in enumerate(target.split('~')) ]

                    if filter_condition == 'auto' or f_name in filter_condition.keys():
                        if filter_condition == 'auto' or filter_condition[f_name] == '==' or filter_condition[f_name] == 'equal':
                            data_history_temp = data_history_temp[data_history_temp[f_name] == analysis_row[f_name]]
                        else:
                            if '%' in str(filter_condition[f_name]):
                                if '~' in filter_condition[f_name]:
                                    range_str = filter_condition[f_name].replace('%','')
                                else:
                                    range_str = '-' + filter_condition[f_name].replace('%','') +'~' + filter_condition[f_name].replace('%','')
                                filter_range = list(analysis_row[f_name] * (np.array(range_to_list(range_str)).astype('float')/100+1))
                            elif '~' in str(filter_condition[f_name]):
                                if '-' in filter_condition[f_name] or '+' in filter_condition[f_name]:
                                    filter_range = list(analysis_row[f_name] + np.array(range_to_list(filter_condition[f_name])).astype(float))
                                else:
                                    filter_range = list(np.array(range_to_list(filter_condition[f_name])).astype(float))
                            else:
                                try:
                                    filter_revision = float(filter_condition[f_name])
                                    condition_type = 'float'
                                except:
                                    raise("Value Error: Numeric-Variable Condition must be 'number' or 'range' or 'persent' format")
                                if condition_type == 'float':
                                    filter_range = list(analysis_row[f_name] + np.array([-1,1]) * filter_revision)
                            data_history_temp = data_history_temp[(data_history_temp[f_name] >= filter_range[0]) & (data_history_temp[f_name] < filter_range[1])]

                data_dict[row_name]['all'][f] = data_history_temp
                data_shape_dict[row_name][f] = data_history_temp.shape[0]

                if group:
                    df_group_temp = data_history_temp.copy()
                    for g in group:
                        df_group_temp = df_group_temp[df_group_temp[g] == analysis_row[g]]
                        
                    data_dict[row_name]['group'][f] = df_group_temp
                    datagroup_shape_dict[row_name][f] = df_group_temp.shape[0]
        
        self.filters = filters
        self.group = group

        self.data = data_dict
        self.data_structure = pd.DataFrame(data_shape_dict).T[filters].T
        self.data_structure['dataset'] ='all'
        if group:
            datagroup_structure = pd.DataFrame(datagroup_shape_dict).T[filters].T
            datagroup_structure['dataset'] ='group'
            self.datagroup_structure = datagroup_structure
            self.data_structure = pd.concat([self.data_structure, datagroup_structure], axis=0)
        
        self.data_structure.reset_index(inplace=True)
        self.data_structure.columns = ['filters'] + list(self.data_structure.columns[1:])
        self.data_structure.set_index(['dataset','filters'], inplace=True)

        print(f'==== < Filtered Data Summary > ====')
        print(f' → (data) self.data')
        print(f' → (structure) self.data_structure')
        print_DataFrame(self.data_structure)
        print()
        # return self.data_structure

    def auto_select(self, threshold=100, dataset='auto'):
        if dataset == 'auto' or dataset == 'group':
            if self.group:
                auto_select_data = self.datagroup_structure.copy().drop('dataset', axis=1)
                key = 'group'
            else:
                data_structure_temp = self.data_structure.reset_index().set_index('filters')
                data_structure_temp = data_structure_temp[data_structure_temp['dataset'] == 'all']
                auto_select_data = data_structure_temp.drop('dataset',axis=1)
                key = 'all'
        elif dataset == 'all':
            data_structure_temp = self.data_structure.reset_index().set_index('filters')
            data_structure_temp = data_structure_temp[data_structure_temp['dataset'] == 'all']
            auto_select_data = data_structure_temp.drop('dataset',axis=1)
            key = 'all'

        auto_data = {}
        auto_structure = {}
        auto_filter_idx = auto_select_data.apply(lambda x: np.min(np.where(x < threshold))-1 if len(np.where(x < threshold)[0]) else len(self.filters)-1, axis=0).apply(lambda x: 0 if x<0 else x)
        auto_filter = auto_select_data.apply(lambda x: x.index[auto_filter_idx[x.name]],axis=0)

        for i, f in auto_filter.items():
            auto_data[i] = self.data[i][key][f]
            auto_structure[i] = self.filters[:auto_filter_idx[i]+1]
        
        self.auto_data = auto_data
        self.auto_structure = pd.Series(auto_structure).to_frame()
        self.auto_structure = pd.concat([pd.Series({k: v.shape[0] for k,v in auto_data.items()}).to_frame(), self.auto_structure], axis=1)
        self.auto_structure.columns = ['counts', 'filter_conditions']
        self.auto_structure['final_condition'] = self.auto_structure['filter_conditions'].apply(lambda x: x[-1])
        self.auto_structure = self.auto_structure[['counts', 'final_condition', 'filter_conditions']]
        
        print(f'==== < Auto Selected Data Summary: {key}, (count > {threshold}) > ====')
        print(f' → (data) self.auto_data')
        print(f' → (structure) self.auto_structure')
        print_DataFrame(self.auto_structure)
        print()
        # return auto_data






# Operation Analysis
# class OperationAnalysis
class OperationAnalysis():
    def __init__(self, history=False, reject_data=False, reject_info=[]):
        self.history = history
        self.reject_data = reject_data
        self.reject_info = reject_info

    def make_anal_tb(self, df_describe):
        result = df_describe.loc[['mean', 'std']].T.applymap(lambda x: str(round(x, fun_Decimalpoint(x))) if x>0 else 0 if ~np.isnan(x) else np.nan)
        result['sigma_range'] = df_describe.loc[['lf_sigma', 'uf_sigma']].T.apply(lambda x: 
                str(round(x['lf_sigma'], fun_Decimalpoint(x['lf_sigma']))) + ' ~ ' +
                str(round(x['uf_sigma'], fun_Decimalpoint(x['uf_sigma']))) if ~np.isnan(x['lf_sigma']) and ~np.isnan(x['uf_sigma']) else np.nan, axis=1)
        # result = df_describe[['mean', 'std']].applymap(lambda x: str(round(x, fun_Decimalpoint(x))) if x>0 else 0 if ~np.isnan(x) else np.nan)
        # result['sigma_range'] = df_describe[['lf_sigma', 'uf_sigma']].apply(lambda x: 
        #         str(round(x['lf_sigma'], fun_Decimalpoint(x['lf_sigma']))) + ' ~ ' +
        #         str(round(x['uf_sigma'], fun_Decimalpoint(x['uf_sigma']))) if ~np.isnan(x['lf_sigma']) and ~np.isnan(x['uf_sigma']) else np.nan
        #         , axis=1)
        return result

    def fit(self, history=False, reject_data=False, y=False, x=False, group=False, reject_info=[]):
        if type(history)==bool and history==False:
            history=self.history
        else:
            self.history=history
        
        if type(reject_data)==bool and reject_data==False:
            reject_data=self.reject_data
        else:
            self.reject_data=reject_data

        if reject_info:
            self.reject_info = reject_info
        else:
            reject_info = self.reject_info
            

        # Summary Instance
        # des = Describe(mode='series')
        y_exist, x_exist = False, False
        if type(y) == list:
            y_exist = True
            # history_Ysummary = self.make_anal_tb(history[y].agg(des)) #.apply(lambda x: pd.Series(x)))
            history_Ysummary = self.make_anal_tb(history[y].describe()) #.apply(lambda x: pd.Series(x)))
            

        if type(x) == list:
            x_exist = True
            xr = [e[0] for e in x]
            xi = [e[1] for e in x]
            xd = [e[2] for e in x]
            # history_Xsummary = self.make_anal_tb(history[xr].agg(des)) #.apply(lambda x: pd.Series(x)))
            history_Xsummary = self.make_anal_tb(history[xr].describe()) #.apply(lambda x: pd.Series(x)))

        if y_exist == False and x_exist== False:
            raise('Value_Error: need either y or x')

        # group
        if type(group) != bool and group != False:
            if type(group) != list:
                group = [group]
            if type(reject_data) == bool and reject_data == False:
                pass
            else:
                target_group = reject_data[group[0]]
                
            data_Ysummary_group = {}
            data_Xsummary_group = {}
            for gi, gv in history.groupby(group):
                if y_exist:
                    # data_Ysummary_group[gi] = self.make_anal_tb(gv[y].agg(des)) #.apply(lambda x: pd.Series(x)))
                    data_Ysummary_group[gi] = self.make_anal_tb(gv[y].describe()) #.apply(lambda x: pd.Series(x)))
                if x_exist:
                    # data_Xsummary_group[gi] = self.make_anal_tb(gv[xr].agg(des)) #.apply(lambda x: pd.Series(x)))
                    data_Xsummary_group[gi] = self.make_anal_tb(gv[xr].describe()) #.apply(lambda x: pd.Series(x)))
            
            if type(reject_data) == bool and reject_data == False:
                data = history
                if y_exist:
                    data_Ysummary = history_Ysummary
                if x_exist:
                    data_Xsummary = history_Xsummary
            else:
                data = history[history[group[0]] == target_group]
                if y_exist:
                    data_Ysummary = data_Ysummary_group[target_group]
                if x_exist:
                    data_Xsummary = data_Xsummary_group[target_group]
            self.group = f'{group[0]}: {target_group}'
            if y_exist:
                self.data_Ysummary_group = data_Ysummary_group
            if x_exist:
                self.data_Xsummary_group = data_Xsummary_group
        else:
            data = history
            if y_exist:
                data_Ysummary = history_Ysummary
            if x_exist:
                data_Xsummary = history_Xsummary

        # Save Variable
        if reject_info:
            self.info = reject_data[reject_info].to_frame().T

        self.data = data
        if y_exist:
            self.history_Ysummary = history_Ysummary
            self.data_Ysummary = data_Ysummary
        if x_exist:
            self.history_Xsummary = history_Xsummary
            self.data_Xsummary = data_Xsummary

        # Summary_Table + Reject Information
        summary = {}
        if type(reject_data) == bool and reject_data == False:
            if y_exist:
                summary['y'] = self.data_Ysummary
            if x_exist:
                summary['x'] = self.data_Xsummary
            self.summary = summary

        else:
            
            if y_exist:
                self.data_Ysummary['reject'] = reject_data[y]
                self.data_Ysummary['reject_sigma'] = ((self.data_Ysummary['reject'].astype(float) - self.data_Ysummary['mean'].astype(float)) / self.data_Ysummary['std'].astype(float)).apply(lambda x: round(x, 1))


                for yc in y:
                    if ~(yc + '_하한' in data.columns) or ~(yc + '_상한' in data.columns):
                        data[yc+'_하한'] = data[yc+'_보증범위'].apply(lambda x: np.nan if pd.isna(x) else float(np.nan if x.split('~')[0] == ' ' else x.split('~')[0].strip()))
                        data[yc+'_상한'] = data[yc+'_보증범위'].apply(lambda x: np.nan if pd.isna(x) else float(np.nan if x.split('~')[1] == ' ' else x.split('~')[1].strip()))

                    if ~(yc + '_하한' in reject_data.index) or ~(yc + '_상한' in reject_data.index):
                        if pd.isna(reject_data[yc + '_보증범위']):
                            criteria_range_list = [np.nan, np.nan]
                        else:
                            criteria_range_list = list(map(lambda x: float(np.nan if x==' ' else x.strip()), reject_data[yc + '_보증범위'].split('~')))
                        reject_data[yc + '_하한'] = criteria_range_list[0]
                        reject_data[yc + '_상한'] = criteria_range_list[1]

                lower_columns = list(map(lambda x: str(x)+'_하한', y))
                upper_columns = list(map(lambda x: str(x)+'_상한', y))
                upper_lower_columns = list(set(lower_columns + upper_columns) & set(list(data.columns)))
                upper_lower_columns

                self.data_Ysummary['lsl'] = pd.Series({yc: reject_data[yc +'_하한'] if yc +'_하한' in upper_lower_columns else np.nan for yi, yc in enumerate(y)}).apply(lambda x: str(round(x, fun_Decimalpoint(x))) if ~np.isnan(x) else np.nan)
                self.data_Ysummary['usl'] = pd.Series({yc: reject_data[yc +'_상한'] if yc +'_상한' in upper_lower_columns else np.nan for yi, yc in enumerate(y)}).apply(lambda x: str(round(x, fun_Decimalpoint(x))) if ~np.isnan(x) else np.nan)
                self.data_Ysummary['cpk'] = self.data_Ysummary.apply(lambda x: 
                    round(cpk(mean=float(x['mean']), std=float(x['std']),
                        lsl=float(x['lsl']), usl=float(x['usl']), lean=True),2), axis=1)

                self.data_Ysummary['abnormal'] = self.data_Ysummary.apply(lambda x: '' if (abs(x['cpk']) >= 1 and abs(x['reject_sigma']) < 2) else
                                        ('cpk ~0.3' if abs(x['cpk']) < 0.3 else 'cpk 0.3~0.5' if abs(x['cpk']) < 0.5 else 
                                        'cpk 0.5~0.7' if abs(x['cpk']) < 0.7 else 'cpk 0.7~1.0' if abs(x['cpk']) < 1 else 
                                        '') + (', ' if abs(x['cpk']) < 1  and abs(x['reject_sigma']) >= 2 else
                                        '') + ('3 σ'  if abs(x['reject_sigma']) >= 3 else
                                        '2 σ' if abs(x['reject_sigma']) >= 2 else ''), axis=1)
                summary['y'] = self.data_Ysummary

            if x_exist:
                self.data_Xsummary['reject'] = reject_data[xr]
                self.data_Xsummary['reject_sigma'] = ((self.data_Xsummary['reject'].astype(float) - self.data_Xsummary['mean'].astype(float)) / self.data_Xsummary['std'].astype(float)).apply(lambda x: round(x, 1))
                self.data_Xsummary['지시'] = pd.Series({xrc: np.nan if xi[xri] == '' else reject_data[xi[xri]] for xri, xrc in enumerate(xr)})
                self.data_Xsummary['설계'] = pd.Series({xrc: np.nan if xd[xri] == '' else reject_data[xd[xri]] for xri, xrc in enumerate(xr)})

                self.data_Xsummary['abnormal'] = self.data_Xsummary['reject_sigma'].apply(lambda x: '3 σ' if abs(x) >= 3 else '2 σ' if abs(x) >= 2 else '')
                summary['x'] = self.data_Xsummary
        
        self.__repr__()

    def __repr__(self):
        try:
            if self.reject_info:
                print(f' #### Reject Product Information ##############################################')
                print(f' > self.info')
                display(HTML(self.info._repr_html_()))
                print()
                print()
        except:
            pass
        try:
            print(f' #### {self.group},  Count: {len(self.data):,} #########################################')               # Printing Group
        except:
            pass
        
        try:
            self.data_Ysummary
            print()
            print(f' > self.data_Ysummary')
            display(HTML(self.data_Ysummary._repr_html_()))
        except:
            pass

        try:
            self.data_Xsummary
            print()
            print(f' > self.data_Xsummary')
            display(HTML(self.data_Xsummary._repr_html_()))
        except:
            pass

        
        return ''

    def __str__(self):
        self.__repr__()










# 조업품질지시기준 정리 기준별 지시실적
class QualityCriteria:
    '''
    < input >
     . data: data
     . mode: class Mode

    < result >
     . self.data_criteria : 보증기준별 실적
     . self.data_aim : 조업품질지시기준별 실적
     . self.data_design : 설계기준별(냉연제조표준) 실적
     . self.data_aim_design : 설계기준별(냉연제조표준) 실적

    '''
    def __init__(self, data, mode, criteria=None, aim=None, design=None, comp_num=None, comp_obj=None, comp_minmax=None, date=None,
        add_criteria=[],  add_aim=[], add_design=[], add_comp_num=[], add_comp_obj=[], add_comp_minmax=[],
        comp_obj_format='{i}: {n} ({round(p*100,1)}%)'):
        if criteria:
            self.criteria_list = ['출강목표N', '품종명', '소둔_공장공정', '인장_호수', '인장_방향'] + criteria
        else:
            self.criteria_list = ['출강목표N', '품종명', '소둔_공장공정', '인장_호수', '인장_방향', 'YP_보증범위', 'TS_보증범위', 'EL_보증범위', 'YR_보증범위', 'HER_보증','HER_평균_보증범위', 'BMB_보증','BMB_방향','BMB_시험굴곡각도', 'BMB_시험굴곡간격구분'] + add_criteria

        if aim:
            self.aim_list = ['출강목표N', '품종명', '소둔_공장공정'] + aim
        else:
            self.aim_list = ['출강목표N', '품종명', '소둔_공장공정', '소둔_HS목표온도', '소둔_SS목표온도', '소둔_RCS목표온도', '소둔_OAS목표온도', '소둔_SPM_EL목표'] + add_aim
        
        if design:
            self.design_lsit = ['출강목표N', '품종명', '소둔_공장공정', '냉연제조표준N'] + design
        else:
            self.design_list = ['출강목표N', '품종명', '소둔_공장공정', '냉연제조표준N', '설계_HS목표온도', '설계_SS목표온도', '설계_RCS목표온도', '설계_OAS목표온도', '설계_SPM_EL목표'] + add_design
        
        if comp_num:
            self.comp_num_list = comp_num
        else:
            self.comp_num_list = ['YP', 'TS', 'EL', 'HER_평균'] + add_comp_num
        
        if comp_obj:
            self.comp_obj_list = comp_obj
        else:
            self.comp_obj_list = ['BMB'] + add_comp_obj
        
        if comp_minmax:
            self.comp_minmax_list = comp_minmax
        else:
            self.comp_minmax_list = ['주문두께'] + add_comp_minmax
       

        ma = copy.deepcopy(mode)
        ma2 = copy.deepcopy(mode)
        ma2.return_type = ',  '
        # if comp_obj_format is None:
            # ma2.return_format = '{i}: {n} ({round(p*100,1)}%)'
        # else:
        ma2.return_format = comp_obj_format

        if date:
            self.date = date
        else:
            self.date = ['소둔_작업계상일자']


        data_anlyasis = data.copy()
        keys = list(dict.fromkeys(self.criteria_list + self.aim_list + self.design_list))
        data_anlyasis[keys] = data_anlyasis[keys].fillna('.')

        # data_criteria
        dc_group = data_anlyasis.groupby(self.criteria_list)
        dc1 = pd.DataFrame(dc_group.size(), columns=['count'])
        dc2 = dc_group['고객사코드', '냉연제조표준N', 'MainKey_번호'].agg(ma)
        
        # data_aim
        da_group = data_anlyasis.groupby(self.aim_list)
        da1 = pd.DataFrame(da_group.size(), columns=['count'])
        da2 = pd.DataFrame(da_group[self.date].agg('max'))
        da3 = da_group['냉연제조표준N', '고객사코드'].agg(ma)
        
        # data_design
        dd_group = data_anlyasis.groupby(self.design_list)
        dd1 = pd.DataFrame(dd_group.size(), columns=['count'])
        dd2 = pd.DataFrame(dd_group[self.date].agg('max'))
        dd3 = dd_group['냉연제조표준N', '고객사코드'].agg(ma)

        # data_aim_design
        dad_group = data_anlyasis.groupby( list(dict.fromkeys(self.aim_list + self.design_list)) )
        dad1 = pd.DataFrame(dad_group.size(), columns=['count'])
        dad2 = pd.DataFrame( dad_group[self.date].agg('max'))
        dad3 = dad_group['냉연제조표준N', '고객사코드'].agg(ma)
        
        
        if self.comp_num_list:
            dc6 = dc_group[self.comp_num_list].agg('mean').applymap(lambda x: self.auto_decimal(x))
            da6 = da_group[self.comp_num_list].agg('mean').applymap(lambda x: self.auto_decimal(x))
            dd6 = dd_group[self.comp_num_list].agg('mean').applymap(lambda x: self.auto_decimal(x))
            dad6 = dad_group[self.comp_num_list].agg('mean').applymap(lambda x: self.auto_decimal(x))
        else:
            dc6 = pd.DataFrame()
            da6 = pd.DataFrame()
            dd6 = pd.DataFrame()
            dad6 = pd.DataFrame()

        if self.comp_minmax_list:
            dc7 = dc_group[self.comp_minmax_list].agg(self.min_max)
            da7 = da_group[self.comp_minmax_list].agg(self.min_max)
            dd7 = dd_group[self.comp_minmax_list].agg(self.min_max)
            dad7 = dad_group[self.comp_minmax_list].agg(self.min_max)
        else:
            dc7 = pd.DataFrame()
            da7 = pd.DataFrame()
            dd7 = pd.DataFrame()
            dad7 = pd.DataFrame()

        if self.comp_obj_list:
            dc8 = dc_group[self.comp_obj_list].agg(ma2)
            da8 = da_group[self.comp_obj_list].agg(ma2)
            dd8 = dd_group[self.comp_obj_list].agg(ma2)
            dad8 = dad_group[self.comp_obj_list].agg(ma2)
        else:
            dc8 = pd.DataFrame()
            da8 = pd.DataFrame()
            dd8 = pd.DataFrame()
            dad8 = pd.DataFrame()

        self.data_criteria = pd.concat([dc1, dc2, dc6, dc7, dc8], axis=1)
        self.data_aim = pd.concat([da1, da2, da3, da6, da7, da8], axis=1)
        self.data_design = pd.concat([dd1, dd2, dd3, dd6, dd7, dd8], axis=1)
        self.data_aim_design = pd.concat([dad1, dad2, dad3, dad6, dad7, dad8], axis=1)

        print()
        print('----------------------------------------------------------------')
        print(f' * data_criteria: {self.data_criteria.shape}, data_aim: {self.data_aim.shape}, data_design: {self.data_design.shape} data_aim_design: {self.data_aim_design.shape}')

    def decimal(self, x, rev=0):
        return 2 if x == 0 else int(-1*(np.floor(np.log10(abs(x)))-3-rev))

    def auto_decimal(self,x, rev=0):
        if np.isnan(x):
            return np.nan
        else:
            decimals = self.decimal(x, rev=rev)
            if decimals < 0:
                return x
            else:
                return round(x, decimals)

    def min_max(self, x):
        return str(self.auto_decimal(min(x))) + ' ~ ' + str(self.auto_decimal(max(x)))




# INQ Review #########################################################################################
def cummax_summary(data, x, group, title=None, annotation=True, rotation=0, return_plot=True):
    result = {}
    data_agg = data.groupby(group)[x].agg('max')
    data_group = data_agg.agg('cummax').to_frame()
    data_group[x + '_Min'] = data_group[x].shift()
    data_group.reset_index(inplace=True)

    data_group_melt = pd.melt(data_group, id_vars=[group], value_vars=[x,x + '_Min'])
    data_group_melt.sort_values([group,'value'], ascending=[True,True], inplace=True)
    data_group_melt.dropna(inplace=True)

    if return_plot:
        fig = plt.figure()
    if title is not None:
        plt.title(title)
    plt.scatter(data[group], data[x], edgecolor='white', alpha=0.3)
    plt.plot(data_group_melt[group], data_group_melt['value'], color='navy')
    plt.xlabel(group)
    plt.ylabel(x)
    if annotation:
        annotation_data = data_group_melt.groupby('value')[group].min().reset_index()
        for r in annotation_data.iterrows():
            plt.text(r[1][group], r[1]['value'], f"{r[1][group]}t×{int(r[1]['value'])}w",rotation=rotation)
    if return_plot:
        plt.close()

    result['data_agg'] = data.groupby(group)[x].agg(['count','min','max'])
    result['data_group'] = data_group
    result['data_group_melt'] = data_group_melt
    if return_plot:
        result['plot'] = fig
        return result



# -----------------------------------------------------------------------
# 열연목표두께 Plot
def hr_reduction_plot(data=None, figsize=(8,5), title=None, 
                    from_clipboard=False, return_plot=True):
    if from_clipboard:
        data = pd.read_clipboard()
    data_use = data[['두께이상', '두께미만', '폭이상', '폭미만', '열연목표두께']]
    data_use['두께이상'].drop_duplicates().to_clipboard()
    pd.Series(list(set(data_use['폭이상'].tolist()  + data_use['폭미만'].tolist()))).sort_values().to_clipboard()

    data_use['두께'] =  data_use[['두께이상', '두께미만']].mean(1)
    data_use['폭'] =  data_use[['폭이상', '폭미만']].mean(1)

    data_use[['두께','폭']]
    data_use[['두께','폭','열연목표두께']].set_index(['두께','폭']).unstack('두께')

    data_use['두께이상_압하율'] = 1- data_use['두께이상']/data_use['열연목표두께']
    data_use['두께미만_압하율'] = 1- data_use['두께미만']/data_use['열연목표두께']

    max_press_ratio = max(data_use['두께이상_압하율'].max(), data_use['두께미만_압하율'].max())
    min_press_ratio = min(data_use['두께이상_압하율'].min(), data_use['두께미만_압하율'].min())

    # 열연목표두께 Plot
    if return_plot is not False:
        f = plt.figure(figsize=figsize)
    if title is not None:
        plt.title(title)

    plt.plot(np.nan,np.nan)
    for ri, rd in data_use.iloc[:-1,:].iterrows():
        rmax = 1-rd['두께이상']/rd['열연목표두께']
        rmin = 1-rd['두께미만']/rd['열연목표두께']
        rmean = (rmax+rmin)/2 
        color_grade = 1 - (rmean - min_press_ratio) / (max_press_ratio - min_press_ratio)

        thick = (rd['두께이상'] + rd['두께미만'])/2
        width = (rd['폭이상'] + rd['폭미만'])/2

        plt.fill_between( [rd['두께이상'], rd['두께미만']],  rd['폭미만'], rd['폭이상']
                ,facecolor=(color_grade, color_grade, color_grade)
                    ,edgecolor='gray', linewidth=1
                    ,alpha=0.5)

        # ---------------------------------------------------------------------------------------------
        # from matplotlib.patches import Rectangle
        # import matplotlib.patches as patches
        # plt.gca().add_patch(
        #     
        #         patches.Rectangle((rd['두께이상'], rd['폭이상']), rd['두께미만'] - rd['두께이상'], rd['폭미만'] - rd['폭이상']
        #         , fill=True
        #     #     color='black',
        #     #     ,facecolor='orange'
        #         ,facecolor=(color_grade, color_grade, color_grade)
        #         ,edgecolor='gray', linewidth=1
        #         ,alpha=0.5)
        #         )
        # plt.text(thick, width, round(rmean,2))
        # ---------------------------------------------------------------------------------------------
        plt.text(thick-0.05, width, f"{round(rmean,2)}\n({rd['열연목표두께']}t)")
        # plt.text(rd['두께이상'], width, f"{round(rmean,2)}\n{round(rmin,2)}~{round(rmax,2)}")

    plt.xlabel('주문두께')
    plt.ylabel('주문폭')
    
    if return_plot is not False:
        plt.show()
        if return_plot == 'clipboard':
            print('* plot is move to clipboard.')
            img_to_clipboard(f)





# Micro Data #########################################################################################
# Micro Plot
def micro_plot(df, line=None, ylim=None, label=None, fill=False, return_plot=True):
    zones = df['ZONE'].drop_duplicates()
    if return_plot:
        fig = plt.figure(figsize=(13,len(zones)*2.5))
    for i, c in enumerate(zones):
        df_t = df[df['ZONE'] == c]
        plt.subplot(len(zones),1,i+1)
        plt.ylabel(c, fontsize=15)
        plt.plot(df_t['LEN_POS'], df_t['VALUE'], label=label)
        if type(line) == dict:
            if c in line:
                if type(line[c]) == list:
                    for cc in line[c]:
                        plt.axhline(cc, color='mediumseagreen', ls='--', alpha=0.5)
                        plt.text(df_t['LEN_POS'].max(), cc, cc, color='red', fontsize=13)
                    if len(line[c]) == 2:
                        if fill is not False:
                            plt.fill_between(df_t['LEN_POS']
                                             , np.linspace(line[c][0], line[c][1], len(df_t['LEN_POS'])) - fill
                                             , np.linspace(line[c][0], line[c][1], len(df_t['LEN_POS'])) + fill
                                             , color='mediumseagreen', alpha=0.15)        
                else:
                    plt.axhline(line[c], color='mediumseagreen', ls='--', alpha=0.5)
                    if fill is not False:
                        plt.fill_between(df_t['LEN_POS'], df_t['LEN_POS']*0+line[c]-fill, df_t['LEN_POS']*0+line[c]+fill, color='mediumseagreen', alpha=0.15)
                    plt.text(df_t['LEN_POS'].max(), line[c], line[c], color='red', fontsize=13)
        plt.text(df_t['LEN_POS'].min(), df_t['VALUE'].mean(), df_t['MTL_NO'].head(1).values[0], fontsize=13, color='blue')
        if ylim is not None:
            if type(ylim) == list:
                plt.ylim(ylim[0], ylim[1])
            if type(ylim) == dict:
                if c in ylim.keys():
                    plt.ylim(ylim[c][0], ylim[c][1])
    if return_plot:
        plt.close()
        return fig
    
    



class CT_Handler:
    """
    【 Required Library 】: numpy, pandas, matplotlib.pyplot
    
    【 methods 】
      . __init__(data..):
      . remove_abnormal_CT : remove outlier
      . fit : split position (head, mid, tail)
      . plot : draw CT chart
        
    【 attribute 】
      . data : overall CT data
      . data_head : CT_head data
      . data_mid : CT_mid data
      . data_tail : CT_tail data
      . ct : CT (mid)
      . ct_std : CT std (mid)
    """
    def __init__(self, data):
        self.data = data
        self.length = None
        self.ct_mid = None
        self.decide_from_filter = False
        
    def remove_abnormal_CT(self, data=None, normal_region=200, use_function=False, **kwargs):
        data = self.data if data is None else data
        data_25, data_50, data_75 = data['VALUE'].describe()[['25%','50%','75%']]
        data_mean, data_std = data[(data['VALUE'] >= data_25) & (data['VALUE'] <= data_75)]['VALUE'].agg(['mean','std'])
        data_low_3sigma = data_mean - 3* data_std
        data_high_3sigma = data_mean + 3* data_std
        
        data_len_min, data_len_max = data['LEN_POS'].agg(['min','max'])
        data_new = data[data.apply(lambda x: ((x['VALUE'] >= data_low_3sigma) ) 
                                if (x['LEN_POS'] < data_len_min + normal_region) or (x['LEN_POS'] > data_len_max - normal_region) 
                                else True, axis=1)]
        data_new['LEN_POS'] = data_new['LEN_POS'] - data_new['LEN_POS'].min()
        self.length = data_new['LEN_POS'].max()
        
        if use_function is False:
            self.data = data_new
            return self
        elif use_function is True:
            return data_new

    def fit(self, data=None, pos_criteria=200, pattern_factor='trend', use_as_function=False, 
                  filter=None, lamb=1000, decide_from_filter=True, **kwargs):
        '''
        【 required Class 】TrendAnalysis
          filter : None (optional)
           . 'hp_filter' : (lambda=1600)
        '''
        data = self.data if data is None else data
        self.filter = filter
        
        if self.length is None:
            self.remove_abnormal_CT(data=data, **kwargs)
        
        # appply filter
        if (filter is not None) or (decide_from_filter is True):
            data_new = data.copy()
            filter = 'hp_filter' if filter is None else filter
            TA_Object = TrendAnalysis(filter=filter, lamb=lamb, **kwargs)
            filtered_data = TA_Object.fit(data['VALUE'])
            filtered_data_dropna = filtered_data[~filtered_data['VALUE'].isna()].drop('VALUE',axis=1)
            data_new = pd.concat([data_new, filtered_data_dropna], axis=1)
        else:
            data_new = data.copy()
        
        pos_dict = {}
        if type(pos_criteria) == float or type(pos_criteria) == int:
            pos_dict = {k: pos_criteria for k in ['head','tail']}
        elif type(pos_criteria) == list:
            pos_dict = {k: pos_criteria[ei] for ei, k in enumerate(['head','tail'])}
        
        # add columns (CALORIE, POS_GRUOP)
        # data_new['CALORIE'] = ((data_new['LEN_POS'] - data_new['LEN_POS'].shift(1))*data_new['VALUE'])
        data_new['POS_GROUP'] = data_new.apply(lambda x: 'head' if x['LEN_POS'] <= pos_dict['head'] 
                else ('tail' if x['LEN_POS'] >= data_new['LEN_POS'].max() - pos_dict['tail'] else 'mid'),axis=1)
        
        
        # split and analysis 
        if use_as_function is False:
            self.data = data_new
            self.data_head = data_new[data_new['POS_GROUP'] == 'head']
            self.data_mid = data_new[data_new['POS_GROUP'] == 'mid']
            self.data_tail = data_new[data_new['POS_GROUP'] == 'tail']
                                   
            apply_content = 'trend' if filter is not None else 'VALUE'
            
            self.ct = self.data['VALUE'].mean()
            self.ct_head, self.ct_mid, self.ct_tail = self.data.groupby('POS_GROUP')['VALUE'].mean()
            self.ct_mid_std = self.data_mid[apply_content].std()
            
            self.head_max = self.data_head.iloc[self.data_head[apply_content].argmax()]
            self.tail_max = self.data_tail.iloc[self.data_tail[apply_content].argmax()]
            
            if decide_from_filter is True:
                head_pattern_candidate = self.data_head[(self.data_head['LEN_POS'] > 10) & (self.data_head[apply_content] <= self.ct_mid + 3*self.ct_mid_std) & (self.data_head['trend_info'] == 'min') ]
                if len(head_pattern_candidate) == 0:
                    if (self.data_head['trend_info'] == 'min').sum() == 0:
                        head_pattern_candidate = self.data_head[(self.data_head['LEN_POS'] > 10)]
                        head_pattern_candidate = head_pattern_candidate.iloc[[head_pattern_candidate['trend_slope'].apply(abs).argmin()]]
                    else:
                        head_pattern_candidate = self.data_head[(self.data_head['LEN_POS'] > 10) & (self.data_head['trend_info'] == 'min') ]
                self.head_pattern = head_pattern_candidate.iloc[0]
                tail_pattern_candidate = self.data_tail[(self.data_tail['LEN_POS'] < self.length-10)  & (self.data_tail[apply_content] <= self.ct_mid + 3*self.ct_mid_std) & (self.data_tail['trend_info'] == 'min')]
                
                if len(tail_pattern_candidate) == 0:
                    if (self.data_tail['trend_info'] == 'min').sum() == 0:
                        tail_pattern_candidate = self.data_tail[(self.data_tail['LEN_POS'] < self.length-10)]
                        tail_pattern_candidate = tail_pattern_candidate.iloc[[tail_pattern_candidate['trend_slope'].apply(abs).argmin()]]
                    else:
                        tail_pattern_candidate = self.data_tail[(self.data_tail['LEN_POS'] < self.length-10) & (self.data_tail['trend_info'] == 'min')]
                self.tail_pattern = tail_pattern_candidate.iloc[-1]
                self.decide_from_filter = decide_from_filter
            else:
                if not ( 'float' in str(type(pattern_factor)) or 'int' in str(type(pattern_factor)) ):
                    pattern_factor = 3
                
                head_pattern_candidate = self.data_head[(self.data_head['LEN_POS'] > 10) & (self.data_head[apply_content] <= self.ct_mid + pattern_factor*self.ct_mid_std)]
                if len(head_pattern_candidate) == 0:
                    head_pattern_candidate = self.data_head[(self.data_head['LEN_POS'] > 10) & (self.data_head[apply_content] <= self.data_mid.head(1)['VALUE'].iloc[0] + pattern_factor*self.ct_mid_std)]
                self.head_pattern = head_pattern_candidate.iloc[0]
                
                tail_pattern_candidate = self.data_tail[(self.data_tail['LEN_POS'] < self.length-10) & (self.data_tail[apply_content] >= self.ct_mid + pattern_factor*self.ct_mid_std)]
                if len(tail_pattern_candidate) == 0:
                    tail_pattern_candidate = self.data_tail[(self.data_tail['LEN_POS'] < self.length-10) & (self.data_tail[apply_content] >= self.data_mid.tail(1)['VALUE'].iloc[0] + pattern_factor*self.ct_mid_std)]
                self.tail_pattern = tail_pattern_candidate.iloc[-1]
            
            self.summary_dict = ct_dict = {
                'Coil_Length':self.length,
                
                'CT_Head_Mean':self.ct_head,
                'CT_Mid_Mean':self.ct_mid,
                'CT_Tail_Mean':self.ct_tail,
                'CT_Mid_Std':self.ct_mid_std,

                'CT_Head_MAX':self.head_max['trend'],
                'CT_Head_MAX_POS':self.head_max['LEN_POS'],
                'CT_Head_PATTERN_POS':self.head_pattern['LEN_POS'],

                'CT_Tail_MAX':self.tail_max['trend'],
                'CT_Tail_MAX_POS':self.length - self.tail_max['LEN_POS'],
                'CT_Tail_PATTERN_POS': self.length - self.tail_pattern['LEN_POS']
                }
            
            return self
        elif use_as_function is True:
            return data_new

    def plot(self, data=None, figsize=(10,3), ylim=None, color=None, return_plot=True):
        data = self.data if data is None else data
        
        if return_plot is True:
            f= plt.figure(figsize=figsize)
        plt.plot(data['LEN_POS'], data['VALUE'], color=color)   
        
        if ylim is not None:
            plt.ylim(ylim)

        if return_plot is True:
            plt.close()
            return f
    
    def summary_plot(self, figsize=(10,3), ylim=None, color=None, 
                  trend_plot=True, trend_color=None, return_plot=True, **kwargs):
        if self.ct_mid is None:
            if self.length is None:
                self.remove_abnormal_CT(**kwargs)
            self.fit(**kwargs)
            
        if return_plot is True:
            f = plt.figure(figsize=(10,3))
        
        plt.plot(self.data['LEN_POS'], self.data['VALUE'], color=color)

        display_point = 'VALUE'
        if self.filter is not None:
            if trend_plot is True:
                plt.plot(self.data['LEN_POS'], self.data['trend'], color=trend_color, alpha=0.7)
            if self.decide_from_filter is True:
                display_point = 'trend'
        
        # head_max
        plt.scatter(self.head_max['LEN_POS'], self.head_max[display_point], color='purple')
        plt.text(self.head_max['LEN_POS'], self.head_max[display_point], 
                f"head_max: {self.head_max['LEN_POS']:0.0f}m / {self.head_max[display_point]:0.0f}℃")

        # tail_max
        plt.scatter(self.tail_max['LEN_POS'], self.tail_max[display_point], color='purple')
        plt.text(self.tail_max['LEN_POS'], self.tail_max[display_point], 
                f"tail_max: {self.tail_max['LEN_POS']:0.0f}m / {self.tail_max[display_point]:0.0f}℃")

        # head_pattern
        plt.scatter(self.head_pattern['LEN_POS'], self.head_pattern[display_point], color='red')
        plt.text(self.head_pattern['LEN_POS'], self.head_pattern[display_point], 
                f"head_pattern: {self.head_pattern['LEN_POS']:0.0f}m / {self.head_pattern[display_point]:0.0f}℃")

        # tail_pattern
        plt.scatter(self.tail_pattern['LEN_POS'], self.tail_pattern[display_point], color='red')
        plt.text(self.tail_pattern['LEN_POS'], self.tail_pattern[display_point], 
                f"tail_pattern: {self.tail_pattern['LEN_POS']:0.0f}m / {self.tail_pattern[display_point]:0.0f}℃")

        # COIL_NAME
        plt.text(self.length/5, (self.data['VALUE'].max() + self.ct)/2, f"{self.data['MTL_NO'].iloc[0]}", color='darkblue')
        
        # CT
        plt.text(self.length/2, self.ct, f"CT: {self.ct:0.0f}℃", color='blue', alpha=0.7)
        plt.axhline(self.ct, color='steelblue', alpha=0.2, ls=':')
        
        # CT mid mean/std
        plt.text(self.length/2, self.ct_mid, f"CT_Mid: {self.ct_mid:0.0f}℃ (std: {self.ct_mid_std:0.1f})")
        plt.axhline(self.ct_mid, color='mediumseagreen', alpha=0.5, ls='--')
        plt.axhline(self.ct_mid + 3*self.ct_mid_std, color='mediumseagreen', alpha=0.3)
        plt.axhline(self.ct_mid - 3*self.ct_mid_std, color='mediumseagreen', alpha=0.3)

        # CT head/mid/tail
        plt.axvline(self.data_mid.head(1)['LEN_POS'].iloc[0], color='orange',alpha=0.5)
        plt.axvline(self.data_mid.tail(1)['LEN_POS'].iloc[0], color='orange',alpha=0.5)

        if ylim is not None:
            plt.ylim(*ylim)

        if return_plot is True:
            plt.close()
            self.summary_plot_ = f
            return f





def shape_meter(data=None, display_plot=False):
    if data is None:
        data = pd.read_clipboard(sep='\t')

    df_load = data[[i for i, c in (data==0).all(axis=0).items() if c is False]]
    df = df_load.sort_values(by = ['COIL_NO','COIL_POSITION'], ascending = True) # 코일,길이 기준으로 오름차순 정렬
    coil = df.drop_duplicates(subset = ['COIL_NO'])['COIL_NO'] # Coil 기준 중복 제거 
    
    plots = []
    for x in coil :
        df1 = df[df['COIL_NO']==x]
        df2 = df1.drop(['COIL_NO','No'], axis = 1).set_index('COIL_POSITION')
        fig = plt.figure(figsize=(50,10))
        #sns.heatmap(df2.T , cmap='jet', vmin=-50, vmax=50)
        plt.contourf(df2.index, df2.columns, df2.T, cmap='jet' , vmin=-50, vmax=50)
        plt.text(df2.index.max()-500, 2, x, fontsize=70, weight='bold')
        # plt.axis('off')
        interval = np.arange(0, df2.index.max()+1,100)
        plt.colorbar()
        plt.xticks(interval, fontsize=30)
        if display_plot:
            plt.show()
        else:
            plt.close()
        
        plots.append(fig)
    return plots
        