########################################################################################################################
"""
LOAD PACKAGES
"""
import pickle
import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd, DateOffset, Week
import openpyxl
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FormatStrFormatter
########################################################################################################################
"""
LOAD MODULES
"""
from Code.Project_parameters import path
from Code.Project_functions import winsorize
########################################################################################################################
"""
OPENDATA
"""
# VOIR PK MESSAGE D'AVERTISSEMENT CONCERNANT L'EXTENSION DU FICHIER EXCEL - PEUT ETRE PAS IMP
path_to_file = path.get('Inputs') + '/DATA_Project_1.xlsx'
data_source_df = pd.read_excel(path_to_file, header=1, sheet_name='sheet1', engine='openpyxl')
# FORMAT VARIABLE DATE, VOIR SI NECESSAIRE
data_source_df['Clean_DATE'] = pd.to_datetime(data_source_df['DATE'], format='%Y-%m-%d')
########################################################################################################################


########################################################################################################################
"""
CREATION DATAFRAME POUR CALCUL DAILY SIMPLE / LOG RETURN
"""
daily_indices_df = data_source_df.copy()
daily_indices_df['Clean_DATE'] = daily_indices_df.Clean_DATE.dt.to_period('D')
daily_indices_df = daily_indices_df.set_index('Clean_DATE')
daily_indices_df = daily_indices_df.drop(columns=daily_indices_df.select_dtypes(exclude='number').columns)
########################################################################################################################
"""
CALCUL DAILY SIMPLE / LOG RETURN
"""
# CREA DTF AVEC DECALAGE DE L'INDEX D'UN UNDICE
lagged_day_indices_df = daily_indices_df.shift(1)
# CALCULER LES RENDEMENTS QUOTIDIENS SIMPLES / LOG A L'AIDE DES VALEURS DECALEES
indices_day_ret_df = (daily_indices_df - lagged_day_indices_df) / lagged_day_indices_df
indices_log_day_ret_df = pd.DataFrame(np.log(daily_indices_df).values - np.log(lagged_day_indices_df).values,
                                      index=daily_indices_df.index, columns=daily_indices_df.columns)
"""
# VOIR SI NECESSAIRE DE SUPPRIMER PREMIERE LIGNE OU FIGURE NAN
indices_ret_df = indices_ret_df.dropna()
"""
'''
# POSSIBLE DE CALC RENDMNT QUOTI COMME CA AUSSI:
lagged_index = (indices_df.index + DateOffset(days=-1)).to_period('D')
indices_df.index = indices_df.index.to_period('D')
indices_ret_df = (indices_df.values - indices_df.reindex(lagged_index).values) / indices_df.reindex(lagged_index).values
indices_ret_df = pd.DataFrame(indices_ret_df, index=indices_df.index, columns=indices_df.columns)
'''
########################################################################################################################


########################################################################################################################
"""
CREATION DATAFRAME POUR CALCUL WEEKLY SIMPLE / LOG RETURN
"""
weekly_indices_df = data_source_df.copy()
weekly_indices_df['Clean_DATE'] = weekly_indices_df.Clean_DATE.dt.to_period('W-FRI')
weekly_indices_df = weekly_indices_df.groupby(['Clean_DATE']).last().reset_index()
weekly_indices_df = weekly_indices_df.set_index('Clean_DATE')
weekly_indices_df = weekly_indices_df.drop(columns=weekly_indices_df.select_dtypes(exclude='number').columns)
'''
# POSSIBLE DE FAIRE COMME ÇA MAIS JSP PK LES SEMAINES DEBUTENT LE SAMEDI
Weekly_indices_df = Daily_indices_df.copy()
Weekly_indices_df = Weekly_indices_df.resample('W-FRI').last()
'''
########################################################################################################################
"""
CALCUL WEEKLY SIMPLE / LOG RETURN
"""
# CREA DTF AVEC DECALAGE DE L'INDEX D'UN UNDICE
lagged_week_indices_df = weekly_indices_df.shift(1)
# CALCULER LES RENDEMENTS hebdo A L'AIDE DES VALEURS DECALEES
indices_week_ret_df = (weekly_indices_df - lagged_week_indices_df) / lagged_week_indices_df
indices_log_week_ret_df = pd.DataFrame(np.log(weekly_indices_df).values - np.log(lagged_week_indices_df).values,
                                       index=weekly_indices_df.index, columns=weekly_indices_df.columns)
########################################################################################################################

########################################################################################################################
"""
SUMMARY STATISTICS AND CORRELATIONS
"""
sumstat_day_ret_df = indices_day_ret_df.describe()
# print(sumstat_day_ret_df.to_latex())
sumstat_log_day_ret_df = indices_log_day_ret_df.describe()
# print(sumstat_log_day_ret_df.to_latex())
sumstat_week_ret_df = indices_week_ret_df.describe()
# print(sumstat_week_ret_df.to_latex())
sumstat_log_week_ret_df = indices_log_week_ret_df.describe()
# print(sumstat_log_week_ret_df.to_latex())
########################################################################################################################
