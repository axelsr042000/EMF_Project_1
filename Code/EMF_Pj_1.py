########################################################################################################################
"""
LOAD PACKAGES
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scipy.stats as stats
from scipy.stats import chi2
import math
########################################################################################################################
"""
LOAD MODULES
"""
from Code.Project_parameters import path
# from Code.Project_functions import winsorize
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
1 - A : CREATION DATAFRAME POUR CALCUL DAILY SIMPLE / LOG RETURN
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
indices_day_ret_df = ((daily_indices_df - lagged_day_indices_df) / lagged_day_indices_df) * 100
indices_log_day_ret_df = pd.DataFrame((np.log(daily_indices_df).values - np.log(lagged_day_indices_df).values)*100,
                                      index=daily_indices_df.index, columns=daily_indices_df.columns)

########################################################################################################################


########################################################################################################################
"""
1 - B : CREATION DATAFRAME POUR CALCUL WEEKLY SIMPLE / LOG RETURN
"""
weekly_indices_df = data_source_df.copy()
weekly_indices_df['Clean_DATE'] = weekly_indices_df.Clean_DATE.dt.to_period('W-FRI')
weekly_indices_df = weekly_indices_df.groupby(['Clean_DATE']).last().reset_index()
weekly_indices_df = weekly_indices_df.set_index('Clean_DATE')
weekly_indices_df = weekly_indices_df.drop(columns=weekly_indices_df.select_dtypes(exclude='number').columns)
########################################################################################################################
"""
CALCUL WEEKLY SIMPLE / LOG RETURN
"""
# CREA DTF AVEC DECALAGE DE L'INDEX D'UN UNDICE
lagged_week_indices_df = weekly_indices_df.shift(1)
# CALCULER LES RENDEMENTS hebdo A L'AIDE DES VALEURS DECALEES
indices_week_ret_df = ((weekly_indices_df - lagged_week_indices_df) / lagged_week_indices_df)*100
indices_log_week_ret_df = pd.DataFrame((np.log(weekly_indices_df).values - np.log(lagged_week_indices_df).values)*100,
                                       index=weekly_indices_df.index, columns=weekly_indices_df.columns)
########################################################################################################################

########################################################################################################################
"""
1 - C : SUMMARY STATISTICS AND CORRELATIONS
"""
# Mean
usDR = indices_day_ret_df.mean()
usWR = indices_week_ret_df.mean()
ulDR = indices_log_day_ret_df.mean()
ulWR = indices_log_week_ret_df.mean()

# Variance
VarsDR = np.var(indices_day_ret_df)
VarsWR = np.var(indices_week_ret_df)
VarlDR = np.var(indices_log_day_ret_df)
VarlWR = np.var(indices_log_week_ret_df)

# Skewness
SksDR = indices_day_ret_df.skew()
SksWR = indices_week_ret_df.skew()
SklDR = indices_log_day_ret_df.skew()
SklWR = indices_log_week_ret_df.skew()

# Kurtosis
KsDR = indices_day_ret_df.kurtosis()
KsWR = indices_week_ret_df.kurtosis()
KlDR = indices_log_day_ret_df.kurtosis()
KlWR = indices_log_week_ret_df.kurtosis()

# Min
MinsDR = np.min(indices_day_ret_df)
MinsWR = np.min(indices_week_ret_df)
MinlDR = np.min(indices_log_day_ret_df)
MinlWR = np.min(indices_log_week_ret_df)

# Max
MaxsDR = np.max(indices_day_ret_df)
MaxsWR = np.max(indices_week_ret_df)
MaxlDR = np.max(indices_log_day_ret_df)
MaxlWR = np.max(indices_log_week_ret_df)

# Create a table of name s.t. Mean etc
N_table = pd.DataFrame(['Mean', 'Variance', 'Skewness', 'Kurtosis', 'Min', 'Max'])

# Stock parameters into tables
sDR_parameters = pd.DataFrame(pd.concat([N_table, pd.DataFrame([usDR, VarsDR, SksDR, KsDR, MinsDR, MaxsDR])], axis=1))
sWR_parameters = pd.DataFrame(pd.concat([N_table, pd.DataFrame([usWR, VarsWR, SksWR, KsWR, MinsWR, MaxsWR])], axis=1))
lDR_parameters = pd.DataFrame(pd.concat([N_table, pd.DataFrame([ulDR, VarlDR, SklDR, KlDR, MinlDR, MaxlDR])], axis=1))
lWR_parameters = pd.DataFrame(pd.concat([N_table, pd.DataFrame([ulWR, VarlWR, SklWR, KlWR, MinlWR, MaxlWR])], axis=1))
########################################################################################################################


########################################################################################################################
"""
2 - A : PLOT DIFF OF RETURN
"""
indices_day_ret_df.index = indices_day_ret_df.index.to_timestamp()
indices_log_day_ret_df.index = indices_log_day_ret_df.index.to_timestamp()
indices_week_ret_df.index = indices_week_ret_df.index.to_timestamp()
indices_log_week_ret_df.index = indices_log_week_ret_df.index.to_timestamp()


def setup_plot(name, frequency):
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
    # plt.show()


# import matplotlib.backends.backend_pdf


def plot_diff_returns(name, table1, table2, frequency, table_label):
    dif_return = table1-table2

    # plot the returns from the two tables on the same graph
    plt.plot(dif_return.index, dif_return[name], label=table_label, color='black')

    top5 = dif_return[name].nlargest(5)
    bottom5 = dif_return[name].nsmallest(5)

    for date, value in top5.items():
        plt.scatter(date, value, marker='o', color='green')
    for date, value in bottom5.items():
        plt.scatter(date, value, marker='o', color='red')

    plt.scatter([], [], marker='o', color='green', label='Highest Values')
    plt.scatter([], [], marker='o', color='red', label='Lowest Values')
    plt.legend()

    setup_plot(name, frequency)

    # plt.savefig(name + ".pdf", bbox_inches="tight")

    # pdf = matplotlib.backends.backend_pdf.PdfPages(name + ".pdf")
    # pdf.savefig(plt.gcf())
    # plt.close()


# Use the function to see the difference for each class of assets
for x in daily_indices_df.columns:
    # DAILY
    plot_diff_returns(x, indices_day_ret_df, indices_log_day_ret_df, '∆ Daily returns (percentage points)',
                      'Simple Returns - Log Returns')
    # WEEKLY
    plot_diff_returns(x, indices_week_ret_df, indices_log_week_ret_df, '∆ Weekly returns (percentage points)',
                      'Simple Returns - Log Returns')
########################################################################################################################

########################################################################################################################
"""
PLOT GRAPHS OF RETURNS WITH HIGHLIGTED 5 EXTREMES RETURNS
"""
# Plot graphs of returns with highligted 5 extremes returns


def highlight_returns(name, table1, frequency, table1_label):
    plt.plot(table1.index, table1[name], label=table1_label, color='black')

    top5 = table1[name].nlargest(5)
    bottom5 = table1[name].nsmallest(5)

    for date, value in top5.items():
        plt.scatter(date, value, marker='o', color='green')

    for date, value in bottom5.items():
        plt.scatter(date, value, marker='o', color='red')

    plt.scatter([], [], marker='o', color='green', label='Highest Values')
    plt.scatter([], [], marker='o', color='red', label='Lowest Values')
    plt.legend()

    setup_plot(name, frequency)


for r in daily_indices_df.columns:
    # Use the function to plot graphs for log daily and logweekly returns
    highlight_returns(r, indices_log_day_ret_df, 'Daily returns (%)', 'Log Returns')
    highlight_returns(r, indices_log_week_ret_df, 'Weekly returns (%)', 'Log Returns')
    highlight_returns(r, indices_day_ret_df, 'Daily returns (%)', 'Simple Returns')
    highlight_returns(r, indices_week_ret_df, 'Weekly returns (%)', 'Simple Returns')

########################################################################################################################


########################################################################################################################
"""
2 - B : HYPOTHESIS OF NORMALITY
"""
# Get the 5 largest and smallest daily and weekly returns of column 'S&PCOMP(RI)'
largest_SP500D = pd.DataFrame(indices_log_day_ret_df['S&PCOMP(RI)'].nlargest(5))
smallest_SP500D = pd.DataFrame(indices_log_day_ret_df['S&PCOMP(RI)'].nsmallest(5))
largest_SP500W = pd.DataFrame(indices_log_week_ret_df['S&PCOMP(RI)'].nlargest(5))
smallest_SP500W = pd.DataFrame(indices_log_week_ret_df['S&PCOMP(RI)'].nsmallest(5))

# Get the 5 largest and smallest daily and weekly returns of column MLGTRSA(RI)'
largest_MLGD = pd.DataFrame(indices_log_day_ret_df['MLGTRSA(RI)'].nlargest(5))
smallest_MLGD = pd.DataFrame(indices_log_day_ret_df['MLGTRSA(RI)'].nsmallest(5))
largest_MLGW = pd.DataFrame(indices_log_week_ret_df['MLGTRSA(RI)'].nlargest(5))
smallest_MLGW = pd.DataFrame(indices_log_week_ret_df['MLGTRSA(RI)'].nsmallest(5))

# Get the 5 largest and smallest daily and weekly returns of column 'MLCORPM(RI)'
largest_MLCD = pd.DataFrame(indices_log_day_ret_df['MLCORPM(RI)'].nlargest(5))
smallest_MLCD = pd.DataFrame(indices_log_day_ret_df['MLCORPM(RI)'].nsmallest(5))
largest_MLCW = pd.DataFrame(indices_log_week_ret_df['MLCORPM(RI)'].nlargest(5))
smallest_MLCW = pd.DataFrame(indices_log_week_ret_df['MLCORPM(RI)'].nsmallest(5))

# Get the 5 largest and smallest daily and weekly returns of column 'WILURET(RI)'
largest_WILD = pd.DataFrame(indices_log_day_ret_df['WILURET(RI)'].nlargest(5))
smallest_WILD = pd.DataFrame(indices_log_day_ret_df['WILURET(RI)'].nsmallest(5))
largest_WILW = pd.DataFrame(indices_log_week_ret_df['WILURET(RI)'].nlargest(5))
smallest_WILW = pd.DataFrame(indices_log_week_ret_df['WILURET(RI)'].nsmallest(5))

# Get the 5 largest and smallest daily and weekly returns of column 'RJEFCRT(TR)'
largest_RJED = pd.DataFrame(indices_log_day_ret_df['RJEFCRT(TR)'].nlargest(5))
smallest_RJED = pd.DataFrame(indices_log_day_ret_df['RJEFCRT(TR)'].nsmallest(5))
largest_RJEW = pd.DataFrame(indices_log_week_ret_df['RJEFCRT(TR)'].nlargest(5))
smallest_RJEW = pd.DataFrame(indices_log_week_ret_df['RJEFCRT(TR)'].nsmallest(5))

# Get the 5 largest and smallest daily and weekly returns of column 'JPUSEEN'
largest_JPUD = pd.DataFrame(indices_log_day_ret_df['JPUSEEN'].nlargest(5))
smallest_JPUD = pd.DataFrame(indices_log_day_ret_df['JPUSEEN'].nsmallest(5))
largest_JPUW = pd.DataFrame(indices_log_week_ret_df['JPUSEEN'].nlargest(5))
smallest_JPUW = pd.DataFrame(indices_log_week_ret_df['JPUSEEN'].nsmallest(5))


def ttest(g, mu, s, n):
    t_stat = (g - mu) / (s / math.sqrt(n))
    df = n - 1
    return t_stat, df


def pvalue(t_stat, df):
    p_value = stats.t.sf(abs(t_stat), df) * 2
    return p_value


# LDR_parameters SP500 test of normality of the boom and crashes
# daily
for i in range(0, 5):
    print(pvalue(
        ttest(smallest_SP500D.iloc[i, 0], lDR_parameters.iloc[0, 1], math.sqrt(lDR_parameters.iloc[1, 1]), 6000)[0],
        5999))

# TEST SI FORMULE OK
# variable = (-12.760459618107234 - 0.02329432878111254) / (math.sqrt(1.5132061081450896) / math.sqrt(6000))
# resultat = stats.t.sf(abs(variable), 5999) * 2
# print("{:.10e}".format(resultat))

for i in range(0, 5):
    print(pvalue(
        ttest(largest_SP500D.iloc[i, 0], lDR_parameters.iloc[0, 1], math.sqrt(lDR_parameters.iloc[1, 1]), 6000)[0],
        5999))

# weekly
for i in range(0, 5):
    print(pvalue(
        ttest(smallest_SP500W.iloc[i, 0], lWR_parameters.iloc[0, 1], math.sqrt(lWR_parameters.iloc[1, 1]), 1200)[0],
        1199))

for i in range(0, 5):
    print(pvalue(
        ttest(largest_SP500W.iloc[i, 0], lWR_parameters.iloc[0, 1], math.sqrt(lWR_parameters.iloc[1, 1]), 1200)[0],
        1199))

# LDR_parameters MLG test of normality of the boom and crashes
# daily
for i in range(0, 5):
    print(pvalue(
        ttest(smallest_MLGD.iloc[i, 0], lDR_parameters.iloc[0, 2], math.sqrt(lDR_parameters.iloc[1, 2]), 6000)[0],
        5999))

for i in range(0, 5):
    print(
        pvalue(ttest(largest_MLGD.iloc[i, 0], lDR_parameters.iloc[0, 2], math.sqrt(lDR_parameters.iloc[1, 2]), 6000)[0],
               5999))

# weekly
for i in range(0, 5):
    print(pvalue(
        ttest(smallest_MLGW.iloc[i, 0], lWR_parameters.iloc[0, 2], math.sqrt(lWR_parameters.iloc[1, 2]), 1200)[0],
        1199))

for i in range(0, 5):
    print(
        pvalue(ttest(largest_MLGW.iloc[i, 0], lWR_parameters.iloc[0, 2], math.sqrt(lWR_parameters.iloc[1, 2]), 1200)[0],
               1199))

# LDR_parameters MLC test of normality of the boom and crashes
# daily
for i in range(0, 5):
    print(pvalue(
        ttest(smallest_MLCD.iloc[i, 0], lDR_parameters.iloc[0, 3], math.sqrt(lDR_parameters.iloc[1, 3]), 6000)[0],
        5999))

for i in range(0, 5):
    print(
        pvalue(ttest(largest_MLCD.iloc[i, 0], lDR_parameters.iloc[0, 3], math.sqrt(lDR_parameters.iloc[1, 3]), 6000)[0],
               5999))

# weekly
for i in range(0, 5):
    print(pvalue(
        ttest(smallest_MLCW.iloc[i, 0], lWR_parameters.iloc[0, 3], math.sqrt(lWR_parameters.iloc[1, 3]), 1200)[0],
        1199))

for i in range(0, 5):
    print(
        pvalue(ttest(largest_MLCW.iloc[i, 0], lWR_parameters.iloc[0, 3], math.sqrt(lWR_parameters.iloc[1, 3]), 1200)[0],
               1199))

# LDR_parameters WIL test of normality of the boom and crashes
# daily
for i in range(0, 5):
    print(pvalue(
        ttest(smallest_WILD.iloc[i, 0], lDR_parameters.iloc[0, 4], math.sqrt(lDR_parameters.iloc[1, 4]), 6000)[0],
        5999))

for i in range(0, 5):
    print(
        pvalue(ttest(largest_WILD.iloc[i, 0], lDR_parameters.iloc[0, 4], math.sqrt(lDR_parameters.iloc[1, 4]), 6000)[0],
               5999))

# weekly
for i in range(0, 5):
    print(pvalue(
        ttest(smallest_WILW.iloc[i, 0], lWR_parameters.iloc[0, 4], math.sqrt(lWR_parameters.iloc[1, 4]), 1200)[0],
        1199))

for i in range(0, 5):
    print(
        pvalue(ttest(largest_WILW.iloc[i, 0], lWR_parameters.iloc[0, 4], math.sqrt(lWR_parameters.iloc[1, 4]), 1200)[0],
               1199))

# LDR_parameters RJE test of normality of the boom and crashes
# daily
for i in range(0, 5):
    print(pvalue(
        ttest(smallest_RJED.iloc[i, 0], lDR_parameters.iloc[0, 5], math.sqrt(lDR_parameters.iloc[1, 5]), 6000)[0],
        5999))

for i in range(0, 5):
    print(
        pvalue(ttest(largest_RJED.iloc[i, 0], lDR_parameters.iloc[0, 5], math.sqrt(lDR_parameters.iloc[1, 5]), 6000)[0],
               5999))

# weekly
for i in range(0, 5):
    print(pvalue(
        ttest(smallest_RJEW.iloc[i, 0], lWR_parameters.iloc[0, 5], math.sqrt(lWR_parameters.iloc[1, 5]), 1200)[0],
        1199))

for i in range(0, 5):
    print(
        pvalue(ttest(largest_RJEW.iloc[i, 0], lWR_parameters.iloc[0, 5], math.sqrt(lWR_parameters.iloc[1, 5]), 1200)[0],
               1199))

# LDR_parameters JPU test of normality of the boom and crashes
# daily
for i in range(0, 5):
    print(pvalue(
        ttest(smallest_JPUD.iloc[i, 0], lDR_parameters.iloc[0, 6], math.sqrt(lDR_parameters.iloc[1, 6]), 6000)[0],
        5999))

for i in range(0, 5):
    print(
        pvalue(ttest(largest_JPUD.iloc[i, 0], lDR_parameters.iloc[0, 6], math.sqrt(lDR_parameters.iloc[1, 6]), 6000)[0],
               5999))

# weekly
for i in range(0, 5):
    print(pvalue(
        ttest(smallest_JPUW.iloc[i, 0], lWR_parameters.iloc[0, 6], math.sqrt(lWR_parameters.iloc[1, 6]), 1200)[0],
        1199))

for i in range(0, 5):
    print(
        pvalue(ttest(largest_JPUW.iloc[i, 0], lWR_parameters.iloc[0, 6], math.sqrt(lWR_parameters.iloc[1, 6]), 1200)[0],
               1199))
########################################################################################################################


########################################################################################################################
"""
2 - C : SUMMARY STATISTICS AND CORRELATIONS
"""
# JB test (using wikipedia formula: JB = n/6 * (S^2 + (1/4) * (K-3)^2))

JB_test = {'S&PCOMP(RI)': [], 'MLGTRSA(RI)': [], 'MLCORPM(RI)': [], 'WILURET(RI)': [], 'RJEFCRT(TR)': [], 'JPUSEEN': []}
DW = pd.DataFrame(['Log Daily Returns', 'Log Weekly Returns'])


# Get JB test value and put it into a table
def JB_log(returns_table, parameters_table, asset_class, alpha, label):
    JB_log = (len(returns_table)-1) / 6 * (parameters_table[asset_class][2] ** 2 + (1 / 4) * (
                parameters_table[asset_class][3] - 3) ** 2)  # length -1 because 1st row is Nan
    k_value = chi2.ppf(1 - alpha, 2)
    JB_test[asset_class].append(
        JB_log)  # Becareful, if you don't clear and re-run, data will be added into the table, so size will change
    if JB_test[asset_class][0] > k_value:
        print('We reject the assumption of normality of', label, 'for', asset_class)
    else:
        print('We do not reject the assumption of normality of', label, 'for', asset_class)


# LOG DAILY RETURNS
JB_log(indices_log_day_ret_df, lDR_parameters, 'S&PCOMP(RI)', 0.05, 'Log Daily Returns')
JB_log(indices_log_day_ret_df, lDR_parameters, 'MLGTRSA(RI)', 0.05, 'Log Daily Returns')
JB_log(indices_log_day_ret_df, lDR_parameters, 'MLCORPM(RI)', 0.05, 'Log Daily Returns')
JB_log(indices_log_day_ret_df, lDR_parameters, 'WILURET(RI)', 0.05, 'Log Daily Returns')
JB_log(indices_log_day_ret_df, lDR_parameters, 'RJEFCRT(TR)', 0.05, 'Log Daily Returns')
JB_log(indices_log_day_ret_df, lDR_parameters, 'JPUSEEN', 0.05, 'Log Daily Returns')

# LOG WEEKLY RETURNS
print('\n')
JB_log(indices_log_week_ret_df, lWR_parameters, 'S&PCOMP(RI)', 0.05, 'Log Weekly Returns')
JB_log(indices_log_week_ret_df, lWR_parameters, 'MLGTRSA(RI)', 0.05, 'Log Weekly Returns')
JB_log(indices_log_week_ret_df, lWR_parameters, 'MLCORPM(RI)', 0.05, 'Log Weekly Returns')
JB_log(indices_log_week_ret_df, lWR_parameters, 'WILURET(RI)', 0.05, 'Log Weekly Returns')
JB_log(indices_log_week_ret_df, lWR_parameters, 'RJEFCRT(TR)', 0.05, 'Log Weekly Returns')
JB_log(indices_log_week_ret_df, lWR_parameters, 'JPUSEEN', 0.05, 'Log Weekly Returns')

# Show which JB test values are for daily and which ones are for weekly returns
JB_test = pd.concat([DW, pd.DataFrame(JB_test)], axis=1)
########################################################################################################################


########################################################################################################################
"""
3 - A/B : SUMMARY STATISTICS AND CORRELATIONS
"""
portfolio_day_ret = indices_day_ret_df.mean(axis=1)
porfolio_week_returns = indices_week_ret_df.mean(axis=1)

# Mean
usADR = portfolio_day_ret.mean()
usAWR = porfolio_week_returns.mean()

# Variance
VarsADR = np.var(portfolio_day_ret)
VarsAWR = np.var(porfolio_week_returns)

# Skewness
SksADR = portfolio_day_ret.skew()
SksAWR = porfolio_week_returns.skew()

# Kurtosis
KsADR = portfolio_day_ret.kurtosis()
KsAWR = porfolio_week_returns.kurtosis()

# Min
MinsADR = np.min(portfolio_day_ret)
MinsAWR = np.min(porfolio_week_returns)

# Max
MaxsADR = np.max(portfolio_day_ret)
MaxsAWR = np.max(porfolio_week_returns)

# Stock parameters into tables
Portfolio_D_parameters = pd.DataFrame(
    pd.concat([N_table, pd.DataFrame([usADR, VarsADR, SksADR, KsADR, MinsADR, MaxsADR])], axis=1))

# Stock parameters into tables
Portfolio_W_parameters = pd.DataFrame(
    pd.concat([N_table, pd.DataFrame([usAWR, VarsAWR, SksAWR, KsAWR, MinsAWR, MaxsAWR])], axis=1))

# Test
# JB test (using wikipedia formula: JB = n/6 * (S^2 + (1/4) * (K-3)^2))

Portfolio_JB_test = {'Portfolio equally weighted': []}
Portfolio_DW = pd.DataFrame(['Simple Daily Returns', 'Simple Weekly Returns'])


# Get JB test value and put it into a table
def Portfolio_JB(returns_table, parameters_table, alpha, label):
    P_JB = (len(returns_table)-1) / 6 * (parameters_table.iloc[2, 1] ** 2 + (1 / 4) * (
                parameters_table.iloc[3, 1] - 3) ** 2)  # length -1 because 1st row is Nan
    k_value = chi2.ppf(1 - alpha, 2)
    Portfolio_JB_test['Portfolio equally weighted'].append(
        P_JB)  # Becareful, if you don't clear and re-run, data will be added into the table, so size will change
    if Portfolio_JB_test['Portfolio equally weighted'][0] > k_value:
        print('We reject the assumption of normality of', label, 'for the equally weighted portfolio.')
    else:
        print('We do not reject the assumption of normality of', label, 'for the equally weighted portfolio.')


# SIMPLE DAILY RETURNS
Portfolio_JB(portfolio_day_ret, Portfolio_D_parameters, 0.05, 'Simple Daily Returns')

# LOG WEEKLY RETURNS
Portfolio_JB(porfolio_week_returns, Portfolio_W_parameters, 0.05, 'Simple Weekly Returns')

# Show which JB test values are for daily and which ones are for weekly returns of the portfolio
Portfolio_JB_test = pd.concat([Portfolio_DW, pd.DataFrame(Portfolio_JB_test)], axis=1)
########################################################################################################################


########################################################################################################################
"""
EXTRACT TABLES

sDR_parameters = sDR_parameters.set_index(0)
sWR_parameters = sWR_parameters.set_index(0)
lDR_parameters = lDR_parameters.set_index(0)
lWR_parameters = lWR_parameters.set_index(0)
JB_test = JB_test.set_index(0)
Portfolio_W_parameters.set_index(0)
Portfolio_D_parameters.set_index(0)
Portfolio_JB_test.set_index(0)


with pd.ExcelWriter('Outputs/tabl_extratct.xlsx') as writer:
    sDR_parameters.to_excel(writer, sheet_name='Feuil 1')
    sWR_parameters.to_excel(writer, sheet_name='Feuil 2')
    lDR_parameters.to_excel(writer, sheet_name='Feuil 3')
    lWR_parameters.to_excel(writer, sheet_name='Feuil 4')
    JB_test.to_excel(writer, sheet_name='Feuil 5')
    Portfolio_W_parameters.to_excel(writer, sheet_name='Feuil 6')
    Portfolio_D_parameters.to_excel(writer, sheet_name='Feuil 7')
    Portfolio_JB_test.to_excel(writer, sheet_name='Feuil 8')
"""
