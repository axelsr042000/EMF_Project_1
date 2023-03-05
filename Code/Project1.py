#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 13:04:08 2023

@author: tim
"""
#conda update anaconda
#conda install spyder=5.4.1

# If you re-run programm, delete everything before, otherwise .appen function will do not change values into dictionnaries, but sum up them.

import pandas as pd
import numpy as np
from scipy.stats import skew
from scipy.stats import kurtosis
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import chi2


# For CSV FILE
#data = pd.read_csv("DATA_Project_1.csv", header = 1)
# If we used CSV, need to convert date columns to datetime format
#data['DATE'] = pd.to_datetime(data['DATE'])
# THIS NEED TO BE PASS INTO COMMENTS IF USE EXCEL READ AND NOT CSV
#data = data.iloc[:-3,] # Take out last 3 rows with NaN (1 is the saturday 31.12.2022, others are just not in the range we are looking at)  


########################################################################################################################
# load modules
from Code.Project_parameters import path
from Code.Project_functions import winsorize
########################################################################################################################
# open data
path_to_file = path.get('Inputs') + '/DATA_Project_1.xlsx'
data = pd.read_excel(path_to_file, header=1, sheet_name='sheet1', engine='openpyxl')
########################################################################################################################
# For EXCEL FILE
data = pd.read_excel("DATA_Project_1.xlsx", header = 1, sheet_name='sheet1')



sd_returns = {'S&PCOMP(RI)': [], 'MLGTRSA(RI)': [], 'MLCORPM(RI)': [], 'WILURET(RI)':[], 'RJEFCRT(TR)': [], 'JPUSEEN': []}
sw_returns = {'S&PCOMP(RI)': [], 'MLGTRSA(RI)': [], 'MLCORPM(RI)': [], 'WILURET(RI)':[], 'RJEFCRT(TR)': [], 'JPUSEEN': []}
ld_returns = {'S&PCOMP(RI)': [], 'MLGTRSA(RI)': [], 'MLCORPM(RI)': [], 'WILURET(RI)':[], 'RJEFCRT(TR)': [], 'JPUSEEN': []}
lw_returns = {'S&PCOMP(RI)': [], 'MLGTRSA(RI)': [], 'MLCORPM(RI)': [], 'WILURET(RI)':[], 'RJEFCRT(TR)': [], 'JPUSEEN': []}


# In[1.1]:
# SIMPLE RETURNS

# Compute simple returns
def simple_returns(r0, r1):
    if r0 > 0:
        returns = ((r1-r0)/r0)*100
    else:
        returns = np.nan
        
    return returns
    

# Stock the computed simple daily returns you want into sd_returns
def stock_simple_dailyRtn(name):
    for i in range(0,len(data)-1): 
        x0 = data[name][i]
        x1 = data[name][i+1]
        sd_returns[name].append(simple_returns(x0, x1)) # Becareful, if you don't clear and re-run, data will be added into the table, so size will change
    return sd_returns[name]

# Use the function to stock the differents class assets returns into sd_returns
stock_simple_dailyRtn('S&PCOMP(RI)')
stock_simple_dailyRtn('MLGTRSA(RI)')
stock_simple_dailyRtn('MLCORPM(RI)')
stock_simple_dailyRtn('WILURET(RI)')
stock_simple_dailyRtn('RJEFCRT(TR)')
stock_simple_dailyRtn('JPUSEEN')

# Transform sd_returns into a dataframe named simple_daily_returns 
simple_daily_returns = pd.DataFrame(sd_returns)
# Add a row of NaN for 1st row, since at day 0 we do not have any returns
new_row = pd.Series(np.nan, index=simple_daily_returns.columns)  # create a new row of NaN values with the same column names as the DataFrame
simple_daily_returns = pd.concat([pd.DataFrame([new_row]), simple_daily_returns]).reset_index(drop=True)  # add the new row to the DataFrame at the first row index and reset the row index
# Add the date into the table
simple_daily_returns = pd.concat([data['DATE'], simple_daily_returns], axis=1)


# Stock the computed simple weekly returns you want into sw_returns
def stock_simple_weeklyRtn(name):
    for i in range(0,len(data)-1, 5):
        x0 = data[name][i]
        x1 = data[name][i+5]
        sw_returns[name].append(simple_returns(x0,x1)) # Becareful, if you don't clear and re-run, data will be added into the table, so size will change
    #return sw_returns[name]

# Use the function to stock the differents class assets returns into sw_returns
stock_simple_weeklyRtn('S&PCOMP(RI)')
stock_simple_weeklyRtn('MLGTRSA(RI)')
stock_simple_weeklyRtn('MLCORPM(RI)')
stock_simple_weeklyRtn('WILURET(RI)')
stock_simple_weeklyRtn('RJEFCRT(TR)')
stock_simple_weeklyRtn('JPUSEEN')

# Transform sw_returns into a dataframe named simple_weekly_returns 
simple_weekly_returns = pd.DataFrame(sw_returns)

# Add a row of NaN for 1st row, since at week 0 we do not have any returns
new_row = pd.Series(np.nan, index=simple_weekly_returns.columns)  # create a new row of NaN values with the same column names as the DataFrame
simple_weekly_returns = pd.concat([pd.DataFrame([new_row]), simple_weekly_returns]).reset_index(drop=True)  # add the new row to the DataFrame at the first row index and reset the row index

# Add weekly date into table of weekly.
# Create a weekly date table, and add into it each date
weekly_date_table = {'DATE': []}
for i in range(0,len(data), 5):
    d = data['DATE'][i]
    weekly_date_table['DATE'].append(d)
# Transform weekly_date_table into a dataframe 
weekly_date_table = pd.DataFrame(weekly_date_table)


# Add the date into the table
simple_weekly_returns = pd.concat([weekly_date_table, simple_weekly_returns], axis=1)



# In[1.2]:
# LOG RETURNS
    
# Compute continuous returns (or log returns)
def log_returns(r0,r1):
    if (r0 > 0) and (r1/r0 > 0):
        returns = np.log((r1/r0))*100
    else :
        returns = np.nan
        
    return returns


# Stock the computed daily log returns you want into ld_returns
def stock_log_dailyRtn(name):
    for i in range(0,len(data)-1): # -3 if want to not take 3 last values are NaN. Here -1 because we will assign same size than data table. Also easier to assign date to each return.
        x0 = data[name][i]
        x1 = data[name][i+1]
        ld_returns[name].append(log_returns(x0,x1)) # Becareful, if you don't clear and re-run, data will be added into the table, so size will change
    return ld_returns[name]

# Use the function to stock the differents class assets returns into ld_returns
stock_log_dailyRtn('S&PCOMP(RI)')
stock_log_dailyRtn('MLGTRSA(RI)')
stock_log_dailyRtn('MLCORPM(RI)')
stock_log_dailyRtn('WILURET(RI)')
stock_log_dailyRtn('RJEFCRT(TR)')
stock_log_dailyRtn('JPUSEEN')

# Transform ld_returns into a dataframe named log_daily_returns 
log_daily_returns = pd.DataFrame(ld_returns)
# Add a row of NaN for 1st column, since at day 0 we do not have any returns
new_row = pd.Series(np.nan, index=log_daily_returns.columns)  # create a new row of NaN values with the same column names as the DataFrame
log_daily_returns = pd.concat([pd.DataFrame([new_row]), log_daily_returns]).reset_index(drop=True)  # add the new row to the DataFrame at the first row index and reset the row index
# Add the date into the table
log_daily_returns = pd.concat([data['DATE'], log_daily_returns], axis=1)


# Stock the computed weekly log returns you want into lw_returns
def stock_log_weeklyRtn(name):
    for i in range(0,len(data)-1, 5): # -8 because 3 last values are NaN and -5 because we do not want to go outside of the bound since we go 5 by 5
        x0 = data[name][i]
        x1 = data[name][i+5]
        lw_returns[name].append(log_returns(x0,x1)) # Becareful, if you don't clear and re-run, data will be added into the table, so size will change
    return lw_returns[name]

# Use the function to stock the differents class assets returns into lw_returns
stock_log_weeklyRtn('S&PCOMP(RI)')
stock_log_weeklyRtn('MLGTRSA(RI)')
stock_log_weeklyRtn('MLCORPM(RI)')
stock_log_weeklyRtn('WILURET(RI)')
stock_log_weeklyRtn('RJEFCRT(TR)')
stock_log_weeklyRtn('JPUSEEN')

# Transform lw_returns into a dataframe named log_weekly_returns 
log_weekly_returns = pd.DataFrame(lw_returns)
# Add a row of NaN for 1st row, since at week 0 we do not have any returns
new_row = pd.Series(np.nan, index=log_weekly_returns.columns)  # create a new row of NaN values with the same column names as the DataFrame
log_weekly_returns = pd.concat([pd.DataFrame([new_row]), log_weekly_returns]).reset_index(drop=True)  # add the new row to the DataFrame at the first row index and reset the row index
# Add the date into the table
log_weekly_returns = pd.concat([weekly_date_table, log_weekly_returns], axis=1)

    
# In[1.3]:
# Mean
µsDR = simple_daily_returns.mean()
µsWR = simple_weekly_returns.mean()
µlDR = log_daily_returns.mean()
µlWR = log_weekly_returns.mean()

# Variance
VarsDR = np.var(simple_daily_returns)
VarsWR = np.var(simple_weekly_returns)
VarlDR = np.var(log_daily_returns)
VarlWR = np.var(log_weekly_returns)

# Skewness
SksDR = simple_daily_returns.skew()
SksWR = simple_weekly_returns.skew() 
SklDR = log_daily_returns.skew() 
SklWR = log_weekly_returns.skew() 

# Kurtosis
KsDR = simple_daily_returns.kurtosis() 
KsWR = simple_weekly_returns.kurtosis() 
KlDR = log_daily_returns.kurtosis() 
KlWR = log_weekly_returns.kurtosis() 

# Min
MinsDR = np.min(simple_daily_returns)
MinsWR = np.min(simple_weekly_returns)
MinlDR = np.min(log_daily_returns)
MinlWR = np.min(log_weekly_returns)

# Max
MaxsDR = np.max(simple_daily_returns)
MaxsWR = np.max(simple_weekly_returns)
MaxlDR = np.max(log_daily_returns)
MaxlWR = np.max(log_weekly_returns)

#Create a table of name s.t. Mean etc
N_table = pd.DataFrame(['Mean','Variance', 'Skewness', 'Kurtosis', 'Min', 'Max'])

#Stock parameters into tables
sDR_parameters = pd.DataFrame(pd.concat([N_table, pd.DataFrame([µsDR,VarsDR,SksDR,KsDR,MinsDR,MaxsDR])], axis=1))
sDR_parameters = sDR_parameters.drop(sDR_parameters.columns[-1], axis=1) #Take out date column (don't know why it is here)

sWR_parameters = pd.DataFrame(pd.concat([N_table, pd.DataFrame([µsWR,VarsWR,SksWR,KsWR,MinsWR,MaxsWR])], axis=1))
sWR_parameters = sWR_parameters.drop(sWR_parameters.columns[-1], axis=1) #Take out date column (don't know why it is here)

lDR_parameters = pd.DataFrame(pd.concat([N_table, pd.DataFrame([µlDR,VarlDR,SklDR,KlDR,MinlDR,MaxlDR])], axis=1))
lDR_parameters = lDR_parameters.drop(lDR_parameters.columns[-1], axis=1) #Take out date column (don't know why it is here)

lWR_parameters = pd.DataFrame(pd.concat([N_table, pd.DataFrame([µlWR,VarlWR,SklWR,KlWR,MinlWR,MaxlWR])], axis=1))
lWR_parameters = lWR_parameters.drop(lWR_parameters.columns[-1], axis=1) #Take out date column (don't know why it is here)

# In[1.a-b]:   
'''
  # Function that plot 2 returns tables into the same graph (to see the differences between the 2)  
def plot_returns(name, table1, table2, frequency, table1_label, table2_label):
    # plot the returns from the two tables on the same graph
    plt.plot(table1['DATE'], table1[name], label= table1_label, color = 'black')
    plt.plot(table2['DATE'], table2[name], label=table2_label, alpha = 1, color = 'red')

    # format the x-axis ticks to show dates every 2 years
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.YearLocator(base=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # add axis labels and a legend
    plt.xlabel('Time')
    plt.ylabel(frequency)
    plt.legend()
    plt.title(name)

    # display the graph
    plt.show()

# Plot simple daily returns vs log daily returns for each asset class
plot_returns('S&PCOMP(RI)',simple_daily_returns, log_daily_returns, 'Daily returns (%)', 'Simple Returns', 'Log Returns')
plot_returns('MLGTRSA(RI)',simple_daily_returns, log_daily_returns, 'Daily returns (%)', 'Simple Returns', 'Log Returns')
plot_returns('MLCORPM(RI)',simple_daily_returns, log_daily_returns, 'Daily returns (%)', 'Simple Returns', 'Log Returns')
plot_returns('WILURET(RI)',simple_daily_returns, log_daily_returns, 'Daily returns (%)', 'Simple Returns', 'Log Returns')
plot_returns('RJEFCRT(TR)',simple_daily_returns, log_daily_returns, 'Daily returns (%)', 'Simple Returns', 'Log Returns')
plot_returns('JPUSEEN',simple_daily_returns, log_daily_returns, 'Daily returns (%)', 'Simple Returns', 'Log Returns')

# Plot simple weekly returns vs log weekly returns for each asset class
plot_returns('S&PCOMP(RI)',simple_weekly_returns, log_weekly_returns, 'Weekly returns (%)', 'Simple Returns', 'Log Returns')
plot_returns('MLGTRSA(RI)',simple_weekly_returns, log_weekly_returns, 'Weekly returns (%)', 'Simple Returns', 'Log Returns')
plot_returns('MLCORPM(RI)',simple_weekly_returns, log_weekly_returns, 'Weekly returns (%)', 'Simple Returns', 'Log Returns')
plot_returns('WILURET(RI)',simple_weekly_returns, log_weekly_returns, 'Weekly returns (%)', 'Simple Returns', 'Log Returns')
plot_returns('RJEFCRT(TR)',simple_weekly_returns, log_weekly_returns, 'Weekly returns (%)', 'Simple Returns', 'Log Returns')
plot_returns('JPUSEEN',simple_weekly_returns, log_weekly_returns, 'Weekly returns (%)', 'Simple Returns', 'Log Returns')


# Plot log daily returns vs log weekly returns for each asset class
plot_returns('S&PCOMP(RI)',log_daily_returns, log_weekly_returns, 'Returns (%)', 'Daily Log Returns', 'Weekly Log Returns')
plot_returns('MLGTRSA(RI)',log_daily_returns, log_weekly_returns, 'Returns (%)', 'Daily Log Returns', 'Weekly Log Returns')
plot_returns('MLCORPM(RI)',log_daily_returns, log_weekly_returns, 'Returns (%)', 'Daily Log Returns', 'Weekly Log Returns')
plot_returns('WILURET(RI)',log_daily_returns, log_weekly_returns, 'Returns (%)', 'Daily Log Returns', 'Weekly Log Returns')
plot_returns('RJEFCRT(TR)',log_daily_returns, log_weekly_returns, 'Returns (%)', 'Daily Log Returns', 'Weekly Log Returns')
plot_returns('JPUSEEN',log_daily_returns, log_weekly_returns, 'Returns (%)', 'Daily Log Returns', 'Weekly Log Returns')
'''

# Function that plot the difference between 2 returns table
def plot_diff_returns(name, table1, table2, frequency, table_label):
    dif_return = pd.concat([table1['DATE'], table1.iloc[:, 1:]-table2.iloc[:, 1:]], axis=1)
    
    # plot the returns from the two tables on the same graph
    plt.plot(dif_return['DATE'], dif_return[name], label= table_label, color = 'black')

    # format the x-axis ticks to show dates every 2 years
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.YearLocator(base=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # add axis labels and a legend
    plt.xlabel('Time')
    plt.ylabel(frequency)
    plt.legend()
    plt.title(name)

    # display the graph
    plt.show()

# Use the function to see the difference for each class of assets
# DAILY
plot_diff_returns('S&PCOMP(RI)', simple_daily_returns, log_daily_returns, '∆ Daily returns (%)', 'Simple Returns - Log Returns')
plot_diff_returns('MLGTRSA(RI)', simple_daily_returns, log_daily_returns, '∆ Daily returns (%)', 'Simple Returns - Log Returns')
plot_diff_returns('MLCORPM(RI)', simple_daily_returns, log_daily_returns, '∆ Daily returns (%)', 'Simple Returns - Log Returns')
plot_diff_returns('WILURET(RI)', simple_daily_returns, log_daily_returns, '∆ Daily returns (%)', 'Simple Returns - Log Returns')
plot_diff_returns('RJEFCRT(TR)', simple_daily_returns, log_daily_returns, '∆ Daily returns (%)', 'Simple Returns - Log Returns')
plot_diff_returns('JPUSEEN', simple_daily_returns, log_daily_returns, '∆ Daily returns (%)', 'Simple Returns - Log Returns')
# WEEKLY
plot_diff_returns('S&PCOMP(RI)', simple_weekly_returns, log_weekly_returns, '∆ Weekly returns (%)', 'Simple Returns - Log Returns')
plot_diff_returns('MLGTRSA(RI)', simple_weekly_returns, log_weekly_returns, '∆ Weekly returns (%)', 'Simple Returns - Log Returns')
plot_diff_returns('MLCORPM(RI)', simple_weekly_returns, log_weekly_returns, '∆ Weekly returns (%)', 'Simple Returns - Log Returns')
plot_diff_returns('WILURET(RI)', simple_weekly_returns, log_weekly_returns, '∆ Weekly returns (%)', 'Simple Returns - Log Returns')
plot_diff_returns('RJEFCRT(TR)', simple_weekly_returns, log_weekly_returns, '∆ Weekly returns (%)', 'Simple Returns - Log Returns')
plot_diff_returns('JPUSEEN', simple_weekly_returns, log_weekly_returns, '∆ Weekly returns (%)', 'Simple Returns - Log Returns')



# In[2-a]:

# Get the 5 largest and smallest daily returns of column 'S&PCOMP(RI)'
largest_SP500D = pd.DataFrame(log_daily_returns['S&PCOMP(RI)'].nlargest(5))
smallest_SP500D = pd.DataFrame(log_daily_returns['S&PCOMP(RI)'].nsmallest(5))
# Get and stock their index into a list
highest_indexD = pd.DataFrame(pd.DataFrame(largest_SP500D.index.tolist()))
smallest_indexD = pd.DataFrame(smallest_SP500D.index.tolist())

# Get the 5 largest and smallest weekly returns of column 'S&PCOMP(RI)'
largest_SP500W = pd.DataFrame(log_weekly_returns['S&PCOMP(RI)'].nlargest(5))
smallest_SP500W = pd.DataFrame(log_weekly_returns['S&PCOMP(RI)'].nsmallest(5))
# Get and stock their index into a list
highest_indexW = pd.DataFrame(largest_SP500W.index.tolist())
smallest_indexW = pd.DataFrame(smallest_SP500W.index.tolist())

# Plot graphs of returns with highligted 5 extremes returns
def highlight_returns(table1, frequency, table1_label, highestVal, lowestVal, highest_index, lowest_index):
    # plot the returns from the two tables on the same graph
    plt.plot(table1['DATE'], table1['S&PCOMP(RI)'], label= table1_label, color = 'black')
    
    # Plot the highest and lowest values as separate points
    plt.scatter(table1.loc[highest_index[0], 'DATE'], highestVal['S&PCOMP(RI)'], marker='o', color='green', label='Highest Values')
    plt.scatter(table1.loc[lowest_index[0], 'DATE'], lowestVal['S&PCOMP(RI)'], marker='o', color='red', label='Lowest Values')

    # format the x-axis ticks to show dates every 2 years
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.YearLocator(base=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # add axis labels and a legend
    plt.xlabel('Time')
    plt.ylabel(frequency)
    plt.legend()
    plt.title('S&PCOMP(RI)')

    # display the graph
    plt.show()
    
# Use the function to plot graphs for log daily and logweekly returns
highlight_returns(log_daily_returns, 'Daily returns (%)', 'Log Returns', largest_SP500D, smallest_SP500D, highest_indexD, smallest_indexD)
highlight_returns(log_weekly_returns, 'Weekly returns (%)', 'Log Returns', largest_SP500W, smallest_SP500W, highest_indexW, smallest_indexW)

# In[2-b]: 

    



# In[2-c]: 
#JB test (using wikipedia formula: JB = n/6 * (S^2 + (1/4) * (K-3)^2))


JB_test = {'S&PCOMP(RI)': [], 'MLGTRSA(RI)': [], 'MLCORPM(RI)': [], 'WILURET(RI)':[], 'RJEFCRT(TR)': [], 'JPUSEEN': []}
DW = pd.DataFrame(['Log Daily Returns', 'Log Weekly Returns'])

# Get JB test value and put it into a table
def JB_log(returns_table, parameters_table, asset_class, alpha):
    JB_log = (len(returns_table)-1)/6 * (parameters_table[asset_class][2]**2 + (1/4) * (parameters_table[asset_class][3] - 3)**2) #length -1 because 1st row is Nan
    #k_value = chi2.ppf(1-alpha, len(returns_table)-1)
    JB_test[asset_class].append(JB_log) # Becareful, if you don't clear and re-run, data will be added into the table, so size will change
    
# LOG DAILY RETURNS
JB_log(log_daily_returns, lDR_parameters, 'S&PCOMP(RI)')
JB_log(log_daily_returns, lDR_parameters, 'MLGTRSA(RI)')
JB_log(log_daily_returns, lDR_parameters, 'MLCORPM(RI)')
JB_log(log_daily_returns, lDR_parameters, 'WILURET(RI)')
JB_log(log_daily_returns, lDR_parameters, 'RJEFCRT(TR)')
JB_log(log_daily_returns, lDR_parameters, 'JPUSEEN')

# LOG WEEKLY RETURNS
JB_log(log_weekly_returns, lWR_parameters, 'S&PCOMP(RI)')
JB_log(log_weekly_returns, lWR_parameters, 'MLGTRSA(RI)')
JB_log(log_weekly_returns, lWR_parameters, 'MLCORPM(RI)')
JB_log(log_weekly_returns, lWR_parameters, 'WILURET(RI)')
JB_log(log_weekly_returns, lWR_parameters, 'RJEFCRT(TR)')
JB_log(log_weekly_returns, lWR_parameters, 'JPUSEEN')

# Show which JB test values are for daily and which ones are for weekly returns
JB_test = pd.concat([DW, pd.DataFrame(JB_test)], axis = 1)

# Test normality assumption using JB test

alpha = 0.05
k_value = chi2.ppf(1-alpha, len(log_daily_returns)-1)
    

if (JB_test['S&PCOMP(RI)'][0]> k_value):
    print('We reject the assumption of normality for S&PCOMP(RI)')
else :
    print('We do not reject the assumption of normality for S&PCOMP(RI)')
