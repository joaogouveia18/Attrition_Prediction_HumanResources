#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import the required libraries
import pandas as pd
import numpy as np
import scipy as sp
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mtick
import matplotlib.colors as colors
import seaborn as sns
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar
#import datetime as dt

# Set the style for the Seaborn's plots
sns.set_style('whitegrid',{
    'xtick.bottom': True,
    'xtick.color': '.1',
    'xtick.direction': 'out',
    'xtick.top': False,
    'xtick.major.size': 1,
    'xtick.minor.size': 0.5,
    'ytick.left': True,
    'ytick.color': '.1',
    'ytick.direction': 'out',
    'ytick.right': False,
    'ytick.major.size': 1,
    'ytick.minor.size': 0.5,    
    'ytick.color': '.1',
    'grid.linestyle': '--',
    'axes.edgecolor': '.1',
    'grid.color': '0.8',
    'axes.spines.top': False,
    'axes.spines.right': False
 })


# In[2]:


# Define indexes and colors for plots

# Gender
genderlevels_index = ['Male', 'Female'] #list(df.Gender.unique()) was used to get the levels
genderlevels_colors = ['#4376a2', '#a80055']

# Marital Status
mstatuslevels_index = ['Single', 'Divorced', 'Married'] #list(df.MaritalStatus.unique()) was used to get the levels
mstatuslevels_colors = ['#ffebcd', '#a9c3d8', '#4294e5']

# Relation between the EducationField & Education
EducationField = ['Human Resources', 'Marketing', 'Medical', 'Life Sciences', 'Technical Degree', 'Other'] #df.EducationField.unique() was used to get the levels

# Department
departmentlevels_index = ['Sales', 'Research & Development', 'Human Resources'] #list(df.Department.unique()) was used to get the levels
departmentlevels_colors = ['#d11141', '#00aedb', '#00b159']

# Relation between the JobLevel & JobRole
JobRole = ['Laboratory Technician', 'Research Scientist',           'Human Resources', 'Sales Representative', 'Sales Executive',           'Healthcare Representative', 'Manufacturing Director',           'Research Director', 'Manager'] #df.JobRole.unique() was used to get the levels

# WorkLifeBalance
worklifebalancelevels_index = [1, 2 ,3 ,4] #list(df.WorkLifeBalance.unique()) was used to get the levels
worklifebalancelevels_colors = ['#264653', '#2a9d8f', '#e9c46a', '#f4a461']

# BusinessTravel
businesstravellevels_index = ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'] #list(df.BusinessTravel.unique()) was used to get the levels
businesstravellevels_colors = ['#211a28', '#705860', '#f9c233']

# OverTime
overtimelevels_index = [0, 1] #list(df.OverTime.unique()) was used to get the levels
overtimelevels_colors = ['#211a28', '#f9c233']

# Colors for Ratings
palette = sns.color_palette("RdYlGn", 6)
palette_relscale = {}
palette_relscale_str = {}
for pos in range(1,6):
    col = colors.rgb2hex((palette[pos - 1][0], palette[pos - 1][1], palette[pos - 1][2]))
    palette_relscale.update({pos: col})
    palette_relscale_str.update({'#' + str(pos): col})
    
# Colors for StockOptions
palette = sns.color_palette("PiYG", 4)
palette_stockkoptn = {}
palette_stockkoptn_str = {}
for pos in range(0,4):
    col = colors.rgb2hex((palette[pos][0], palette[pos][1], palette[pos][2]))
    palette_stockkoptn.update({pos: col})
    palette_stockkoptn_str.update({'#' + str(pos): col})   
    
# Colors for Training 
palette = sns.color_palette("PiYG", 7)
palette_training = {}
for pos in range(0,7):
    col = colors.rgb2hex((palette[pos][0], palette[pos][1], palette[pos][2]))
    palette_training.update({pos: col})


# In[3]:


# Define some helpful variables for the plots
# Boxplot
boxprops = dict(alpha = 0.85, linewidth = 1.0)
whiskerprops = dict(color = 'black', linewidth = 1.0, linestyle = '--')
capprops = dict(color = 'red', linewidth = 2.0)
medianprops = dict(color = 'red', linewidth = 1.0)
flierprops = dict(color = 'black', markeredgecolor = 'black', markerfacecolor = 'white',                              linewidth = 0.5, markersize = 4.0, marker = 'd')
whis = 3.0

# Define some helful functions for the plots
def f_find_nearest_Lmax(var_in, threshold_in):
    """This function determines the closest observation to a certain input threshold for the whisker higher limit """
    
    var = var_in.sort_values(ascending = False)
    if(threshold_in >= var.iloc[0]):
        return var.iloc[0]
    else:
        idx = (var - threshold_in).le(0).idxmax()
        return var_in[idx]

def f_find_nearest_Lmin(var_in, threshold_in):
    """This function determines the closest observation to a certain input threshold for the whisker lower limit """
    
    var = var_in.sort_values(ascending = True)
    if(threshold_in <= var.iloc[0]):
        return var.iloc[0]
    else:
        idx = (threshold_in - var).le(0).idxmax()
        return var_in[idx]

def f_stats_boxplot(var_in):
    """This function determines the statistics that are typically represented on a box plot."""

    median = var_in.median()
    IQR = sp.stats.iqr(var_in)
    Q1 = np.percentile(var_in, 25)
    Q3 = np.percentile(var_in, 75)
    lim1 = f_find_nearest_Lmin(var_in, Q1 - whis * IQR)
    lim2 = f_find_nearest_Lmax(var_in, Q3 + whis * IQR)
    skew = sp.stats.skew(var_in) 
    return (IQR, lim1, Q1, median, Q3, lim2, skew)

def f_pielabels(pct, allvalues):
    """This function determines the labels to be used on a pie chart (%, value)."""
    
    absolute = int(pct / 100 * np.sum(allvalues))
    return '{:.1f}%\n({:.0f})'.format(pct, absolute)


# In[4]:


def f_DepVar_analysis(df_in):
    figDepVar, axDepVar = plt.subplots(figsize = (3.5, 3.5), dpi = 80, facecolor = 'w', edgecolor = 'k',                                 constrained_layout = False)

    DepVar = pd.Series([df_in[df_in['Attrition'] == 0]['EmployeeNumber'].count(),                       df_in[df_in['Attrition'] == 1]['EmployeeNumber'].count()], name = 'Attrition')

    DepVar_colorsdict = {0:'#0F9149', 1:'#911A0F'}
    DepVar_colors = [DepVar_colorsdict[idx] for idx in DepVar.index]

    wedges, texts, autotexts = axDepVar.pie(DepVar, radius = .8, startangle = 90,                                        autopct = lambda pct: f_pielabels(pct, DepVar),                                        pctdistance = 1.4, textprops = dict(color = 'black'),                                        colors = DepVar_colors, wedgeprops = {'alpha': 1})
    axDepVar.legend(wedges, DepVar.index.tolist(), title = 'Attrition', loc = 'lower center',                bbox_to_anchor = (0, -.4, 1, 1), prop = dict(size = 12))

    return figDepVar


# In[5]:


def f_Demographics_analysis(df_in):
    # Define indexes
    # Education
    min_Education = df_in['Education'].min()
    max_Education = df_in['Education'].max()
    Education = range(min_Education, max_Education + 1)
    
    # Create the plot
    df_in_0 = df_in[df_in['Attrition'] == 0]
    df_in_1 = df_in[df_in['Attrition'] == 1]
    
    figDemographics = plt.figure(figsize = (7.5, 23), dpi = 80, facecolor = 'w', edgecolor = 'k',                                 constrained_layout = False)
    gsDemographics = gridspec.GridSpec(nrows = 5, ncols = 2, hspace = .3, wspace = .08,                        height_ratios = [1, .8, .8, 1, 1], width_ratios = [1, 1], figure = figDemographics)

    
    
    #[0,0]--------: Boxplot + Histogram for Age (Attrition = 0)
    sgsDemographicsAge_x0 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec = gsDemographics[0, 0],                                       hspace = .05, wspace = .2, height_ratios = (.15, .85))
    # boxplot
    axDemographicsAge_x0_0 = plt.Subplot(figDemographics, sgsDemographicsAge_x0[0, 0])
    figDemographics.add_subplot(axDemographicsAge_x0_0, sharex = True)
    axDemographicsAge_x0_0.tick_params(left = False, bottom = False, labelleft = False)
    axDemographicsAge_x0_0.spines['left'].set_visible(False)
    axDemographicsAge_x0_0.spines['bottom'].set_visible(False)

    sns.boxplot(df_in_0['Age'], color = 'white', boxprops = boxprops, whiskerprops = whiskerprops, 
                capprops = capprops, medianprops = medianprops, flierprops = flierprops, whis = whis,\
                ax = axDemographicsAge_x0_0).set_title('Age (Attrition = 0)', size = 14)
    
    axDemographicsAge_x0_0.tick_params(labelbottom = False) 
    axDemographicsAge_x0_0.set_xlabel('')

    stats = f_stats_boxplot(df_in_0['Age'])
    axDemographicsAge_x0_0.text(stats[1], -.35, '{:.1f}'.format(stats[1]), color = 'red', ha = 'center')
    axDemographicsAge_x0_0.text(stats[2] - 0.15 * stats[0], .6, '{:.1f}'.format(stats[2]), color = 'black',                              ha = 'center')
    axDemographicsAge_x0_0.text(stats[3], .1, '{:.1f}'.format(stats[3]), color = 'black', ha = 'center')
    axDemographicsAge_x0_0.text(stats[4] + 0.15 * stats[0], .6, '{:.1f}'.format(stats[4]), color = 'black',                              ha = 'center')
    axDemographicsAge_x0_0.text(stats[5], -.35, '{:.1f}'.format(stats[5]), color = 'red', ha = 'center')

    # histogram
    axDemographicsAge_x0_1 = plt.Subplot(figDemographics, sgsDemographicsAge_x0[1, 0])
    figDemographics.add_subplot(axDemographicsAge_x0_1)

    sns.distplot(df_in_0['Age'], kde = False, rug = False, bins = 15,                     color = '#03366f', hist_kws = {'alpha': .85}, ax = axDemographicsAge_x0_1)

    axDemographicsAge_x0_1.set_xlabel('Age')
    axDemographicsAge_x0_1.get_xaxis().set_minor_locator(mtick.AutoMinorLocator()) 
    ymin, ymax = axDemographicsAge_x0_1.get_ylim()
    axDemographicsAge_x0_1.text(20, -0.12*(ymax - ymin), 'skewness: ' + '{:.2f}'.format(stats[6]),                               color = 'red', ha = 'center', fontsize = 10)
    axDemographicsAge_x0_1.set_ylabel('Employee Count')
    axDemographicsAge_x0_1.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())
    
    
    
    #[0,1]--------: Boxplot + Histogram for Age (Attrition = 1)
    sgsDemographicsAge_x1 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec = gsDemographics[0, 1],                                       hspace = .05, wspace = .2, height_ratios = (.15, .85))
    # boxplot
    axDemographicsAge_x1_0 = plt.Subplot(figDemographics, sgsDemographicsAge_x1[0, 0])
    figDemographics.add_subplot(axDemographicsAge_x1_0, sharex = True)
    axDemographicsAge_x1_0.tick_params(left = False, bottom = False, labelleft = False)
    axDemographicsAge_x1_0.spines['left'].set_visible(False)
    axDemographicsAge_x1_0.spines['bottom'].set_visible(False)

    sns.boxplot(df_in_1['Age'], color = 'white', boxprops = boxprops, whiskerprops = whiskerprops,
                capprops = capprops, medianprops = medianprops, flierprops = flierprops, whis = whis,\
                ax = axDemographicsAge_x1_0).set_title('Age (Attrition = 1)', size = 14)
    
    axDemographicsAge_x1_0.tick_params(labelbottom = False) 
    axDemographicsAge_x1_0.set_xlabel('')

    stats = f_stats_boxplot(df_in_1['Age'])
    axDemographicsAge_x1_0.text(stats[1], -.35, '{:.1f}'.format(stats[1]), color = 'red', ha = 'center')
    axDemographicsAge_x1_0.text(stats[2] - 0.15 * stats[0], .6, '{:.1f}'.format(stats[2]), color = 'black',                              ha = 'center')
    axDemographicsAge_x1_0.text(stats[3], .1, '{:.1f}'.format(stats[3]), color = 'black', ha = 'center')
    axDemographicsAge_x1_0.text(stats[4] + 0.15 * stats[0], .6, '{:.1f}'.format(stats[4]), color = 'black',                              ha = 'center')
    axDemographicsAge_x1_0.text(stats[5], -.35, '{:.1f}'.format(stats[5]), color = 'red', ha = 'center')

    # histogram
    axDemographicsAge_x1_1 = plt.Subplot(figDemographics, sgsDemographicsAge_x1[1, 0])
    figDemographics.add_subplot(axDemographicsAge_x1_1)
    axDemographicsAge_x1_1.spines['left'].set_visible(False)
    axDemographicsAge_x1_1.spines['right'].set_visible(True)

    sns.distplot(df_in_1['Age'], kde = False, rug = False, bins = 15,                     color = '#03366f', hist_kws = {'alpha': .85}, ax = axDemographicsAge_x1_1)

    axDemographicsAge_x1_1.set_xlabel('Age')
    axDemographicsAge_x1_1.get_xaxis().set_minor_locator(mtick.AutoMinorLocator()) 
    ymin, ymax = axDemographicsAge_x1_1.get_ylim()
    axDemographicsAge_x1_1.text(20, -0.12*(ymax - ymin), 'skewness: ' + '{:.2f}'.format(stats[6]),                                color = 'red', ha = 'center', fontsize = 10)
    axDemographicsAge_x1_1.set_ylabel('Employee Count')
    axDemographicsAge_x1_1.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())
    
    axDemographicsAge_x1_1.get_yaxis().set_label_position("right")
    axDemographicsAge_x1_1.get_yaxis().tick_right()


    
    #[1,0]--------: Barplot for Gender (Attrition = 0)
    axDemographicsGender_x0 = plt.Subplot(figDemographics, gsDemographics[1, 0])
    figDemographics.add_subplot(axDemographicsGender_x0)

    gender_values = df_in_0.groupby(['Gender'])['EmployeeNumber'].count()
    genderlevels = pd.Series(gender_values, index = genderlevels_index, name = 'GenderLevels')
    palette = sns.color_palette(genderlevels_colors)

    sns.barplot(x = genderlevels.index, y = genderlevels, palette = palette, ax = axDemographicsGender_x0)    .set_title('Gender (Attrition = 0)', size = 14)
    axDemographicsGender_x0.set_ylabel('Employee Count')
    axDemographicsGender_x0.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())
    axDemographicsGender_x0.set_xlabel('')
    pos = 0
    total = genderlevels.sum()
    for index, value in genderlevels.iteritems():
        axDemographicsGender_x0.text(pos, value, '{:.1f}%'.format(value / total * 100), color = 'black', ha = 'center')
        pos = pos + 1
    
    
    
    #[1,1]--------: Barplot for Gender (Attrition = 1)
    axDemographicsGender_x1 = plt.Subplot(figDemographics, gsDemographics[1, 1])
    figDemographics.add_subplot(axDemographicsGender_x1)
    axDemographicsGender_x1.spines['left'].set_visible(False)
    axDemographicsGender_x1.spines['right'].set_visible(True)

    gender_values = df_in_1.groupby(['Gender'])['EmployeeNumber'].count()
    genderlevels = pd.Series(gender_values, index = genderlevels_index, name = 'GenderLevels')
    palette = sns.color_palette(genderlevels_colors)
    
    sns.barplot(x = genderlevels.index, y = genderlevels, palette = palette, ax = axDemographicsGender_x1)    .set_title('Gender (Attrition = 1)', size = 14)
    axDemographicsGender_x1.set_ylabel('Employee Count')
    axDemographicsGender_x1.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())
    axDemographicsGender_x1.set_xlabel('')
    pos = 0
    total = genderlevels.sum()
    for index, value in genderlevels.iteritems():
        axDemographicsGender_x1.text(pos, value, '{:.1f}%'.format(value / total * 100), color = 'black', ha = 'center')
        pos = pos + 1
    axDemographicsGender_x1.get_yaxis().set_label_position("right")
    axDemographicsGender_x1.get_yaxis().tick_right()

    
    
    #[2,0]--------: Barplot for MaritalStatus (Attrition = 0)
    axDemographicsMaritalStatus_x0 = plt.Subplot(figDemographics, gsDemographics[2, 0])
    figDemographics.add_subplot(axDemographicsMaritalStatus_x0)

    mstatus_values = df_in_0.groupby(['MaritalStatus'])['EmployeeNumber'].count()
    mstatuslevels = pd.Series(mstatus_values, index = mstatuslevels_index, name = 'MaritalStatusLevels')
    palette = sns.color_palette(mstatuslevels_colors)

    sns.barplot(x = mstatuslevels.index, y = mstatuslevels, palette = palette, ax = axDemographicsMaritalStatus_x0)    .set_title('MaritalStatus (Attrition = 0)', size = 14)
    axDemographicsMaritalStatus_x0.set_ylabel('Employee Count')
    axDemographicsMaritalStatus_x0.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())
    axDemographicsMaritalStatus_x0.set_xlabel('')
    pos = 0
    total = mstatuslevels.sum()
    for index, value in mstatuslevels.iteritems():
        axDemographicsMaritalStatus_x0.text(pos, value, '{:.1f}%'.format(value / total * 100),                                             color = 'black', ha = 'center')
        pos = pos + 1
    
    
    
    #[2,1]--------: Barplot for MaritalStatus (Attrition = 1)
    axDemographicsMaritalStatus_x1 = plt.Subplot(figDemographics, gsDemographics[2, 1])
    figDemographics.add_subplot(axDemographicsMaritalStatus_x1)
    axDemographicsMaritalStatus_x1.spines['left'].set_visible(False)
    axDemographicsMaritalStatus_x1.spines['right'].set_visible(True)

    mstatus_values = df_in_1.groupby(['MaritalStatus'])['EmployeeNumber'].count()
    mstatuslevels = pd.Series(mstatus_values, index = mstatuslevels_index, name = 'MaritalStatusLevels')
    palette = sns.color_palette(mstatuslevels_colors)

    sns.barplot(x = mstatuslevels.index, y = mstatuslevels, palette = palette, ax = axDemographicsMaritalStatus_x1)    .set_title('MaritalStatus (Attrition = 1)', size = 14)
    axDemographicsMaritalStatus_x1.set_ylabel('Employee Count')
    axDemographicsMaritalStatus_x1.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())
    axDemographicsMaritalStatus_x1.set_xlabel('')
    pos = 0
    total = mstatuslevels.sum()
    for index, value in mstatuslevels.iteritems():
        axDemographicsMaritalStatus_x1.text(pos, value, '{:.1f}%'.format(value / total * 100),                                             color = 'black', ha = 'center')
        pos = pos + 1
    axDemographicsMaritalStatus_x1.get_yaxis().set_label_position("right")
    axDemographicsMaritalStatus_x1.get_yaxis().tick_right()

    
    
    #[3,0]--------: Boxplot + Histogram for DistanceFromHome (Attrition = 0)
    sgsDemographicsDistance_x0 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec = gsDemographics[3, 0],                                       hspace = .05, wspace = .2, height_ratios = (.15, .85))
    # boxplot
    axDemographicsDistance_x0_0 = plt.Subplot(figDemographics, sgsDemographicsDistance_x0[0, 0])
    figDemographics.add_subplot(axDemographicsDistance_x0_0, sharex = True)
    axDemographicsDistance_x0_0.tick_params(left = False, bottom = False, labelleft = False)
    axDemographicsDistance_x0_0.spines['left'].set_visible(False)
    axDemographicsDistance_x0_0.spines['bottom'].set_visible(False)

    sns.boxplot(df_in_0['DistanceFromHome'], color = 'white', boxprops = boxprops, whiskerprops = whiskerprops,                capprops = capprops, medianprops = medianprops, flierprops = flierprops, whis = whis,                ax = axDemographicsDistance_x0_0).set_title('Dist.FromHome (Attrition = 1)', size = 14)
    
    axDemographicsDistance_x0_0.tick_params(labelbottom = False) 
    axDemographicsDistance_x0_0.set_xlabel('')

    stats = f_stats_boxplot(df_in_0['DistanceFromHome'])
    axDemographicsDistance_x0_0.text(stats[1], -.35, '{:.1f}'.format(stats[1]), color = 'red', ha = 'center')
    axDemographicsDistance_x0_0.text(stats[2] - 0.05 * stats[0], .6, '{:.1f}'.format(stats[2]), color = 'black',                              ha = 'center')
    axDemographicsDistance_x0_0.text(stats[3], .1, '{:.1f}'.format(stats[3]), color = 'black', ha = 'center')
    axDemographicsDistance_x0_0.text(stats[4] + 0.05 * stats[0], .6, '{:.1f}'.format(stats[4]), color = 'black',                              ha = 'center')
    axDemographicsDistance_x0_0.text(stats[5], -.35, '{:.1f}'.format(stats[5]), color = 'red', ha = 'center')

    # histogram
    axDemographicsDistance_x0_1 = plt.Subplot(figDemographics, sgsDemographicsDistance_x0[1, 0])
    figDemographics.add_subplot(axDemographicsDistance_x0_1)

    sns.distplot(df_in_0['DistanceFromHome'], kde = False, rug = False, bins = 15,                     color = '#4ec5b0', hist_kws = {'alpha': .85}, ax = axDemographicsDistance_x0_1)

    axDemographicsDistance_x0_1.set_xlabel('DistanceFromHome')
    axDemographicsDistance_x0_1.get_xaxis().set_minor_locator(mtick.AutoMinorLocator()) 
    ymin, ymax = axDemographicsDistance_x0_1.get_ylim()
    axDemographicsDistance_x0_1.text(2, -0.12*(ymax - ymin), 'skewness: ' + '{:.2f}'.format(stats[6]),                               color = 'red', ha = 'center', fontsize = 10)
    axDemographicsDistance_x0_1.set_ylabel('Employee Count')
    axDemographicsDistance_x0_1.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())

    
    
    #[3,1]--------: Boxplot + Histogram for DistanceFromHome (Attrition = 1)
    sgsDemographicsDistance_x1 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec = gsDemographics[3, 1],                                       hspace = .05, wspace = .2, height_ratios = (.15, .85))
    # boxplot
    axDemographicsDistance_x1_0 = plt.Subplot(figDemographics, sgsDemographicsDistance_x1[0, 0])
    figDemographics.add_subplot(axDemographicsDistance_x1_0, sharex = True)
    axDemographicsDistance_x1_0.tick_params(left = False, bottom = False, labelleft = False)
    axDemographicsDistance_x1_0.spines['left'].set_visible(False)
    axDemographicsDistance_x1_0.spines['bottom'].set_visible(False)

    sns.boxplot(df_in_1['DistanceFromHome'], color = 'white', boxprops = boxprops, whiskerprops = whiskerprops,                capprops = capprops, medianprops = medianprops, flierprops = flierprops, whis = whis,                ax = axDemographicsDistance_x1_0).set_title('Dist.FromHome (Attrition = 1)', size = 14)
    
    axDemographicsDistance_x1_0.tick_params(labelbottom = False) 
    axDemographicsDistance_x1_0.set_xlabel('')

    stats = f_stats_boxplot(df_in_1['DistanceFromHome'])
    axDemographicsDistance_x1_0.text(stats[1], -.35, '{:.1f}'.format(stats[1]), color = 'red', ha = 'center')
    axDemographicsDistance_x1_0.text(stats[2] - 0.05 * stats[0], .6, '{:.1f}'.format(stats[2]), color = 'black',                              ha = 'center')
    axDemographicsDistance_x1_0.text(stats[3], .1, '{:.1f}'.format(stats[3]), color = 'black', ha = 'center')
    axDemographicsDistance_x1_0.text(stats[4] + 0.05 * stats[0], .6, '{:.1f}'.format(stats[4]), color = 'black',                              ha = 'center')
    axDemographicsDistance_x1_0.text(stats[5], -.35, '{:.1f}'.format(stats[5]), color = 'red', ha = 'center')

    # histogram
    axDemographicsDistance_x1_1 = plt.Subplot(figDemographics, sgsDemographicsDistance_x1[1, 0])
    figDemographics.add_subplot(axDemographicsDistance_x1_1)
    axDemographicsDistance_x1_1.spines['left'].set_visible(False)
    axDemographicsDistance_x1_1.spines['right'].set_visible(True)

    sns.distplot(df_in_1['DistanceFromHome'], kde = False, rug = False, bins = 15,                     color = '#4ec5b0', hist_kws = {'alpha': .85}, ax = axDemographicsDistance_x1_1)

    axDemographicsDistance_x1_1.set_xlabel('DistanceFromHome')
    axDemographicsDistance_x1_1.get_xaxis().set_minor_locator(mtick.AutoMinorLocator()) 
    ymin, ymax = axDemographicsDistance_x1_1.get_ylim()
    axDemographicsDistance_x1_1.text(2, -0.12*(ymax - ymin), 'skewness: ' + '{:.2f}'.format(stats[6]),                               color = 'red', ha = 'center', fontsize = 10)
    axDemographicsDistance_x1_1.set_ylabel('Employee Count')
    axDemographicsDistance_x1_1.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())    
    
    axDemographicsDistance_x1_1.get_yaxis().set_label_position("right")
    axDemographicsDistance_x1_1.get_yaxis().tick_right()
        


    #[4,0]--------: Heatmap for the relation between the EducationField & Education (Attrition = 0)
    axDemographicsEducation_x0 = plt.Subplot(figDemographics, gsDemographics[4, 0])
    figDemographics.add_subplot(axDemographicsEducation_x0)
    axDemographicsEducation_x0.spines['right'].set_visible(False)

    df_Education_EducationField0 = pd.DataFrame(np.zeros(shape = (len(EducationField), len(Education))))
    row = 0
    for Ed in EducationField:
        col = 0
        for e in Education:
            df_Education_EducationField0.iloc[row, col] = df_in_0.loc[(df_in_0['EducationField'] == Ed) &                                                            (df_in_0['Education'] == e)]['EmployeeNumber'].count()
            col = col + 1
        row = row + 1
        
    sns.heatmap(df_Education_EducationField0, cbar = False, annot = True, fmt = '.0f', cmap = 'YlGnBu',                vmin = 0, vmax = df_Education_EducationField0.values.max(),                linewidths = .5, ax = axDemographicsEducation_x0)

    axDemographicsEducation_x0.set_aspect('equal')    
    axDemographicsEducation_x0.get_yaxis().set_label_position('left')
    axDemographicsEducation_x0.get_yaxis().tick_left()
    axDemographicsEducation_x0.invert_yaxis()
    axDemographicsEducation_x0.set_ylabel('EducationField')
    axDemographicsEducation_x0.set_yticklabels(EducationField, **{'rotation': 0}) 
    axDemographicsEducation_x0.set_xlabel('Education')
    axDemographicsEducation_x0.set_xticklabels(Education, **{'rotation': 0})  
    
    
    axdividerDemographicsEducation_x0 = make_axes_locatable(axDemographicsEducation_x0)
    axdividerDemographicsEducation_x0.set_anchor((1,.5))
    caxDemographicsEducation_x0 = axdividerDemographicsEducation_x0.append_axes('bottom', size = '5%', pad = '20%')
    colorbar(axDemographicsEducation_x0.get_children()[0], cax = caxDemographicsEducation_x0,
             orientation = 'horizontal', **{'ticks': (0, df_Education_EducationField0.values.max())})
    caxDemographicsEducation_x0.xaxis.set_ticks_position('bottom')
    caxDemographicsEducation_x0.set_xlabel('Emp. Count')
    caxDemographicsEducation_x0.get_xaxis().set_label_position('bottom')
    
    axDemographicsEducation_x0.set_title('EducationField vs Education (Attrition = 0)', size = 14,                               **{'horizontalalignment': 'right'})  

    
    
    #[4,1]--------: Heatmap for the relation between the EducationField & Education (Attrition = 1)
    axDemographicsEducation_x1 = plt.Subplot(figDemographics, gsDemographics[4, 1])
    figDemographics.add_subplot(axDemographicsEducation_x1)
    axDemographicsEducation_x1.spines['left'].set_visible(False)

    df_Education_EducationField1 = pd.DataFrame(np.zeros(shape = (len(EducationField), len(Education))))
    row = 0
    for Ed in EducationField:
        col = 0
        for e in Education:
            df_Education_EducationField1.iloc[row, col] = df_in_1.loc[(df_in_1['EducationField'] == Ed) &                                                            (df_in_1['Education'] == e)]['EmployeeNumber'].count()
            col = col + 1
        row = row + 1
        
    sns.heatmap(df_Education_EducationField1, cbar = False, annot = True, fmt = '.0f', cmap = 'YlGnBu',                vmin = 0, vmax = df_Education_EducationField1.values.max(),                linewidths = .5, ax = axDemographicsEducation_x1)

    axDemographicsEducation_x1.set_aspect('equal')    
    axDemographicsEducation_x1.get_yaxis().set_label_position('right')
    axDemographicsEducation_x1.get_yaxis().tick_right()
    axDemographicsEducation_x1.invert_yaxis()
    axDemographicsEducation_x1.set_ylabel('EducationField')
    axDemographicsEducation_x1.set_yticklabels(EducationField, **{'rotation': 0}) 
    axDemographicsEducation_x1.set_xlabel('Education')
    axDemographicsEducation_x1.set_xticklabels(Education, **{'rotation': 0})  
    
    
    axdividerDemographicsEducation_x1 = make_axes_locatable(axDemographicsEducation_x1)
    axdividerDemographicsEducation_x1.set_anchor((0,.5))
    caxDemographicsEducation_x1 = axdividerDemographicsEducation_x1.append_axes('bottom', size = '5%', pad = '20%')
    colorbar(axDemographicsEducation_x1.get_children()[0], cax = caxDemographicsEducation_x1,
             orientation = 'horizontal', **{'ticks': (0, df_Education_EducationField1.values.max())})
    caxDemographicsEducation_x1.xaxis.set_ticks_position('bottom')
    caxDemographicsEducation_x1.set_xlabel('Emp. Count')
    caxDemographicsEducation_x1.get_xaxis().set_label_position('bottom')
    
    axDemographicsEducation_x1.set_title('EducationField vs Education (Attrition = 1)', size = 14,                               **{'horizontalalignment': 'left'}) 
    
    return figDemographics


# In[6]:


def f_JobRoles_analysis(df_in):
    # Define indexes
    # Relation between the JobLevel & JobRole
    min_JobLevel = df_in['JobLevel'].min()
    max_JobLevel = df_in['JobLevel'].max()
    JobLevel = range(min_JobLevel, max_JobLevel + 1)

    # Create the plot
    df_in_0 = df_in[df_in['Attrition'] == 0]
    df_in_1 = df_in[df_in['Attrition'] == 1]
    
    
    figJobRoles = plt.figure(figsize = (10, 50), dpi = 80, facecolor = 'w', edgecolor = 'k',                                 constrained_layout = False)
    gsJobRoles = gridspec.GridSpec(nrows = 9, ncols = 2, hspace = .25, wspace = .08,                        height_ratios = [.85, 1.25, 1.25, 1.25, .85, .85, 1.25, .85, 1.25], width_ratios = [1, 1],                                    figure = figJobRoles)

    
    
    #[0,0]--------: Barplot for Department (Attrition = 0)
    axJobRolesDepartment_x0 = plt.Subplot(figJobRoles, gsJobRoles[0, 0])
    figJobRoles.add_subplot(axJobRolesDepartment_x0)

    department_values = df_in_0.groupby(['Department'])['EmployeeNumber'].count()
    departmentlevels = pd.Series(department_values, index = departmentlevels_index, name = 'DepartmentLevels')
    palette = sns.color_palette(departmentlevels_colors)

    sns.barplot(x = departmentlevels.index, y = departmentlevels, palette = palette, ax = axJobRolesDepartment_x0)    .set_title('Department (Attrition = 0)', size = 14)
    axJobRolesDepartment_x0.set_ylabel('Employee Count')
    axJobRolesDepartment_x0.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())
    axJobRolesDepartment_x0.set_xlabel('')
    axJobRolesDepartment_x0.set_xticklabels(departmentlevels_index, **{'rotation': 10}, ha = 'right')
    
    pos = 0
    total = departmentlevels.sum()
    for index, value in departmentlevels.iteritems():
        axJobRolesDepartment_x0.text(pos, value, '{:.1f}%'.format(value / total * 100),                                      color = 'black', ha = 'center')
        pos = pos + 1
    
    
    
    #[0,1]--------: Barplot for Department (Attrition = 0)
    axJobRolesDepartment_x1 = plt.Subplot(figJobRoles, gsJobRoles[0, 1])
    figJobRoles.add_subplot(axJobRolesDepartment_x1)
    axJobRolesDepartment_x1.spines['left'].set_visible(False)
    axJobRolesDepartment_x1.spines['right'].set_visible(True)

    department_values = df_in_1.groupby(['Department'])['EmployeeNumber'].count()
    departmentlevels = pd.Series(department_values, index = departmentlevels_index, name = 'DepartmentLevels')
    palette = sns.color_palette(departmentlevels_colors)

    sns.barplot(x = departmentlevels.index, y = departmentlevels, palette = palette, ax = axJobRolesDepartment_x1)    .set_title('Department (Attrition = 1)', size = 14)
    axJobRolesDepartment_x1.set_ylabel('Employee Count')
    axJobRolesDepartment_x1.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())
    axJobRolesDepartment_x1.set_xlabel('')
    axJobRolesDepartment_x1.set_xticklabels(departmentlevels_index, **{'rotation': 10}, ha = 'right')
    
    pos = 0
    total = departmentlevels.sum()
    for index, value in departmentlevels.iteritems():
        axJobRolesDepartment_x1.text(pos, value, '{:.1f}%'.format(value / total * 100),
                                      color = 'black', ha = 'center')
        pos = pos + 1
    axJobRolesDepartment_x1.get_yaxis().set_label_position("right")
    axJobRolesDepartment_x1.get_yaxis().tick_right()  
    
    
    
    #[1,0]--------: Heatmap for the relation between the JobRole & JobLevel (Attrition = 0)
    axJobRolesLevels_x0 = plt.Subplot(figJobRoles, gsJobRoles[1, 0])
    axJobRolesLevels_x0.set_anchor((1,.5))
    figJobRoles.add_subplot(axJobRolesLevels_x0)

    df_JobRole_JobLevel0 = pd.DataFrame(np.zeros(shape = (len(JobRole), len(JobLevel))))
    row = 0
    for JR in JobRole:
        col = 0
        for j in JobLevel:
            df_JobRole_JobLevel0.iloc[row, col] = (df_in_0.loc[(df_in_0['JobRole'] == JR) &                                                            (df_in_0['JobLevel'] == j)]['EmployeeNumber'].count())
            col = col + 1
        row = row + 1

    sns.heatmap(df_JobRole_JobLevel0, cbar = False, annot = True, fmt = '.0f', cmap = 'YlGnBu',                vmin = 0, vmax = df_JobRole_JobLevel0.values.max(),                linewidths = .5, ax = axJobRolesLevels_x0)
    
    axJobRolesLevels_x0.set_aspect('equal')    
    axJobRolesLevels_x0.get_yaxis().set_label_position('left')
    axJobRolesLevels_x0.get_yaxis().tick_left()
    axJobRolesLevels_x0.invert_yaxis()
    axJobRolesLevels_x0.set_ylabel('JobRole')
    axJobRolesLevels_x0.set_yticklabels(JobRole, **{'rotation': 0})  
    axJobRolesLevels_x0.set_xlabel('JobLevel')
    axJobRolesLevels_x0.set_xticklabels(JobLevel, **{'rotation': 0})  
    
    axdividerJobRolesLevels_x0 = make_axes_locatable(axJobRolesLevels_x0)
    axdividerJobRolesLevels_x0.set_anchor((1,.5))
    caxJobRolesLevels_x0 = axdividerJobRolesLevels_x0.append_axes('bottom', size = '5%', pad = '20%')
    colorbar(axJobRolesLevels_x0.get_children()[0], cax = caxJobRolesLevels_x0, orientation = 'horizontal',             **{'ticks': (0, df_JobRole_JobLevel0.values.max())})
    caxJobRolesLevels_x0.xaxis.set_ticks_position('bottom')
    caxJobRolesLevels_x0.set_xlabel('Employee Count')
    caxJobRolesLevels_x0.get_xaxis().set_label_position('bottom')
    
    axJobRolesLevels_x0.set_title('JobRole vs JobLevel (Attrition = 0)', size = 14,                               **{'horizontalalignment': 'right'})
    

    
    #[1,1]--------: Heatmap for the relation between the JobRole & JobLevel (Attrition = 1)
    axJobRolesLevels_x1 = plt.Subplot(figJobRoles, gsJobRoles[1, 1])
    axJobRolesLevels_x1.set_anchor((0,.5))
    figJobRoles.add_subplot(axJobRolesLevels_x1)

    df_JobRole_JobLevel1 = pd.DataFrame(np.zeros(shape = (len(JobRole), len(JobLevel))))
    row = 0
    for JR in JobRole:
        col = 0
        for j in JobLevel:
            df_JobRole_JobLevel1.iloc[row, col] = (df_in_1.loc[(df_in_1['JobRole'] == JR) &                                                            (df_in_1['JobLevel'] == j)]['EmployeeNumber'].count())
            col = col + 1
        row = row + 1

    sns.heatmap(df_JobRole_JobLevel1, cbar = False, annot = True, fmt = '.0f', cmap = 'YlGnBu',                vmin = 0, vmax = df_JobRole_JobLevel1.values.max(),                linewidths = .5, ax = axJobRolesLevels_x1)
    
    axJobRolesLevels_x1.set_aspect('equal')    
    axJobRolesLevels_x1.get_yaxis().set_label_position('right')
    axJobRolesLevels_x1.get_yaxis().tick_right()
    axJobRolesLevels_x1.invert_yaxis()
    axJobRolesLevels_x1.set_ylabel('JobRole')
    axJobRolesLevels_x1.set_yticklabels(JobRole, **{'rotation': 0})  
    axJobRolesLevels_x1.set_xlabel('JobLevel')
    axJobRolesLevels_x1.set_xticklabels(JobLevel, **{'rotation': 0})  
    

    axdividerJobRolesLevels_x1 = make_axes_locatable(axJobRolesLevels_x1)
    axdividerJobRolesLevels_x1.set_anchor((0,.5))
    caxJobRolesLevels_x1 = axdividerJobRolesLevels_x1.append_axes('bottom', size = '5%', pad = '20%')
    colorbar(axJobRolesLevels_x1.get_children()[0], cax = caxJobRolesLevels_x1, orientation = 'horizontal',             **{'ticks': (0, df_JobRole_JobLevel1.values.max())})
    caxJobRolesLevels_x1.xaxis.set_ticks_position('bottom')
    caxJobRolesLevels_x1.set_xlabel('Employee Count')
    caxJobRolesLevels_x1.get_xaxis().set_label_position('bottom')
    
    axJobRolesLevels_x1.set_title('JobRole vs JobLevel (Attrition = 1)', size = 14,                                   **{'horizontalalignment': 'left'})

    
    
    #[2,0]--------: Relation between JobRole & Age (Attrition = 0)
    axJobRolesAge_x0 = plt.Subplot(figJobRoles, gsJobRoles[2, 0])
    figJobRoles.add_subplot(axJobRolesAge_x0)

    sns.boxplot(x = 'Age', y = 'JobRole', data = df_in_0, order = JobRole[::-1],                color = '#03366f',                boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                medianprops = medianprops, flierprops = flierprops, whis = whis,                ax = axJobRolesAge_x0).set_title('JobRole vs Age (Attrition = 0)', size = 14)

    pos = range(len(JobRole))
    for tick, label in zip(pos, axJobRolesAge_x0.get_yticklabels()):
        
        df_aux_0 = df_in_0[df_in_0['JobRole'] == JobRole[::-1][tick]]['Age']
        stats = f_stats_boxplot(df_aux_0)
        axJobRolesAge_x0.text(stats[1] - 1.2, pos[tick], '{:.0f}'.format(stats[1]),                                       color = 'red', ha = 'center')
        axJobRolesAge_x0.text(stats[2] - 1.2, pos[tick] - .2, '{:.0f}'.format(stats[2]),                                       color = 'black', ha = 'center')
        axJobRolesAge_x0.text(stats[3], pos[tick], '{:.0f}'.format(stats[3]),                                       color = 'white', ha = 'center')
        axJobRolesAge_x0.text(stats[4] + 1.2, pos[tick] - .2, '{:.0f}'.format(stats[4]),                                       color = 'black', ha = 'center')
        axJobRolesAge_x0.text(stats[5] + 1.2, pos[tick], '{:.0f}'.format(stats[5]),                                       color = 'red', ha = 'center')    
    
    axJobRolesAge_x0.set_xlabel('Age')
    axJobRolesAge_x0.set_ylabel('JobRole')
    axJobRolesAge_x0.set_xlim((15, 65))
    axJobRolesAge_x0.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
    
    
    
    #[2,1]--------: Relation between JobRole & Age (Attrition = 1)
    axJobRolesAge_x1 = plt.Subplot(figJobRoles, gsJobRoles[2, 1])
    figJobRoles.add_subplot(axJobRolesAge_x1)
    axJobRolesAge_x1.spines['left'].set_visible(False)
    axJobRolesAge_x1.spines['right'].set_visible(True)

    sns.boxplot(x = 'Age', y = 'JobRole', data = df_in_1, order = JobRole[::-1],                color = '#03366f',                boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                medianprops = medianprops, flierprops = flierprops, whis = whis,                ax = axJobRolesAge_x1).set_title('JobRole vs Age (Attrition = 1)', size = 14)

    pos = range(len(JobRole))
    for tick, label in zip(pos, axJobRolesAge_x1.get_yticklabels()):
        
        df_aux_1 = df_in_1[df_in_1['JobRole'] == JobRole[::-1][tick]]['Age']
        stats = f_stats_boxplot(df_aux_1)
        axJobRolesAge_x1.text(stats[1] - 1.2, pos[tick], '{:.0f}'.format(stats[1]),                                       color = 'red', ha = 'center')
        axJobRolesAge_x1.text(stats[2] - 1.2, pos[tick]- .2, '{:.0f}'.format(stats[2]),                                       color = 'black', ha = 'center')
        axJobRolesAge_x1.text(stats[3], pos[tick], '{:.0f}'.format(stats[3]),                                       color = 'white', ha = 'center')
        axJobRolesAge_x1.text(stats[4] + 1.2, pos[tick]- .2, '{:.0f}'.format(stats[4]),                                       color = 'black', ha = 'center')
        axJobRolesAge_x1.text(stats[5] + 1.2, pos[tick], '{:.0f}'.format(stats[5]),                                       color = 'red', ha = 'center')    
    
    axJobRolesAge_x1.set_xlabel('Age')
    axJobRolesAge_x1.set_ylabel('JobRole')
    axJobRolesAge_x1.get_yaxis().set_label_position("right")
    axJobRolesAge_x1.get_yaxis().tick_right()
    axJobRolesAge_x1.set_xlim((15, 65))
    axJobRolesAge_x1.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())

    
    
    #[3,0]--------: Heatmap for the relation between the JobRole & Gender (Attrition = 0)
    axJobRolesGender_x0 = plt.Subplot(figJobRoles, gsJobRoles[3, 0])
    figJobRoles.add_subplot(axJobRolesGender_x0)

    df_JobRole_Gender0 = pd.DataFrame(np.zeros(shape = (len(JobRole), len(genderlevels_index))))
    row = 0
    for JR in JobRole:
        col = 0
        for g in genderlevels_index:
            df_JobRole_Gender0.iloc[row, col] = (df_in_0.loc[(df_in_0['JobRole'] == JR) &                                                            (df_in_0['Gender'] == g)]['EmployeeNumber'].count()) /             (df_in_0.loc[(df_in_0['JobRole'] == JR)]['EmployeeNumber'].count()) * 100
                                    
            col = col + 1
        row = row + 1

    sns.heatmap(df_JobRole_Gender0,  cbar = False, annot = True, fmt = '.0f', cmap = 'YlGnBu',                vmin = 0, vmax = 100,                linewidths = .5, ax = axJobRolesGender_x0)
    
    axJobRolesGender_x0.set_aspect('equal')    
    axJobRolesGender_x0.get_yaxis().set_label_position('left')
    axJobRolesGender_x0.get_yaxis().tick_left()
    axJobRolesGender_x0.invert_yaxis()
    axJobRolesGender_x0.set_ylabel('JobRole')
    axJobRolesGender_x0.set_yticklabels(JobRole, **{'rotation': 0})  
    axJobRolesGender_x0.set_xlabel('Gender')
    axJobRolesGender_x0.set_xticklabels(genderlevels_index, **{'rotation': 20})  
    
    
    axdividerJobRolesGender_x0 = make_axes_locatable(axJobRolesGender_x0)
    axdividerJobRolesGender_x0.set_anchor((1,.5))
    caxJobRolesGender_x0 = axdividerJobRolesGender_x0.append_axes('bottom', size = '5%', pad = '20%')
    colorbar(axJobRolesGender_x0.get_children()[0], cax = caxJobRolesGender_x0, orientation = 'horizontal',             **{'ticks': (0, 100)})
    caxJobRolesGender_x0.xaxis.set_ticks_position('bottom')
    caxJobRolesGender_x0.set_xlabel('Emp.Count by Role[%]')
    caxJobRolesGender_x0.get_xaxis().set_label_position('bottom')
    
    axJobRolesGender_x0.set_title('JobRole vs Gender (Attrition = 0)', size = 14,                               **{'horizontalalignment': 'right'})
    
    
    #[3,1]--------: Heatmap for the relation between the JobRole & Gender (Attrition = 0)
    axJobRolesGender_x1 = plt.Subplot(figJobRoles, gsJobRoles[3, 1])
    figJobRoles.add_subplot(axJobRolesGender_x1)

    df_JobRole_Gender1 = pd.DataFrame(np.zeros(shape = (len(JobRole), len(genderlevels_index))))
    row = 0
    for JR in JobRole:
        col = 0
        for g in genderlevels_index:
            df_JobRole_Gender1.iloc[row, col] = (df_in_1.loc[(df_in_1['JobRole'] == JR) &                                                            (df_in_1['Gender'] == g)]['EmployeeNumber'].count()) /             (df_in_1.loc[(df_in_1['JobRole'] == JR)]['EmployeeNumber'].count()) * 100
            
            col = col + 1
        row = row + 1

    sns.heatmap(df_JobRole_Gender1, cbar = False, annot = True, fmt = '.0f', cmap = 'YlGnBu',                vmin = 0, vmax = 100,                linewidths = .5, ax = axJobRolesGender_x1)
    
    axJobRolesGender_x1.set_aspect('equal')    
    axJobRolesGender_x1.get_yaxis().set_label_position('right')
    axJobRolesGender_x1.get_yaxis().tick_right()
    axJobRolesGender_x1.invert_yaxis()
    axJobRolesGender_x1.set_ylabel('JobRole')
    axJobRolesGender_x1.set_yticklabels(JobRole, **{'rotation': 0})  
    axJobRolesGender_x1.set_xlabel('Gender')
    axJobRolesGender_x1.set_xticklabels(genderlevels_index, **{'rotation': 20})  
    

    axdividerJobRolesGender_x1 = make_axes_locatable(axJobRolesGender_x1)
    axdividerJobRolesGender_x1.set_anchor((0,.5))
    caxJobRolesGender_x1 = axdividerJobRolesGender_x1.append_axes('bottom', size = '5%', pad = '20%')
    colorbar(axJobRolesGender_x1.get_children()[0], cax = caxJobRolesGender_x1, orientation = 'horizontal',             **{'ticks': (0, 100)})
    caxJobRolesGender_x1.xaxis.set_ticks_position('bottom')
    caxJobRolesGender_x1.set_xlabel('=')
    caxJobRolesGender_x1.get_xaxis().set_label_position('bottom')
    
    axJobRolesGender_x1.set_title('JobRole vs Gender (Attrition = 1)', size = 14,                                           **{'horizontalalignment': 'left'})      

    
    
    #[4,0]--------: Barplot for WorkLifeBalance (Attrition = 0)
    axJobRolesWorkLifeBalance_x0 = plt.Subplot(figJobRoles, gsJobRoles[4, 0])
    figJobRoles.add_subplot(axJobRolesWorkLifeBalance_x0)

    worklifebalance_values = df_in_0.groupby(['WorkLifeBalance'])['EmployeeNumber'].count()
    worklifebalancelevels = pd.Series(worklifebalance_values, index = worklifebalancelevels_index,                                     name = 'WorkLifeLevels')
    palette = sns.color_palette(worklifebalancelevels_colors)

    sns.barplot(x = worklifebalancelevels.index, y = worklifebalancelevels, palette = palette,                ax = axJobRolesWorkLifeBalance_x0)    .set_title('WorkLifeBalance (Attrition = 0)', size = 14)
    axJobRolesWorkLifeBalance_x0.set_ylabel('Employee Count')
    axJobRolesWorkLifeBalance_x0.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())
    axJobRolesWorkLifeBalance_x0.set_xlabel('WorkLifeBalance')
    axJobRolesWorkLifeBalance_x0.set_xticklabels(worklifebalancelevels_index, **{'rotation': 0})
    
    pos = 0
    total = worklifebalancelevels.sum()
    for index, value in worklifebalancelevels.iteritems():
        if(value > 0):
            axJobRolesWorkLifeBalance_x0.text(pos, value, '{:.1f}%'.format(value / total * 100),                                          color = 'black', ha = 'center')
        pos = pos + 1  
    
    
    
    #[4,1]--------: Barplot for WorkLifeBalance (Attrition = 1)
    axJobRolesWorkLifeBalance_x1 = plt.Subplot(figJobRoles, gsJobRoles[4, 1])
    figJobRoles.add_subplot(axJobRolesWorkLifeBalance_x1)
    axJobRolesWorkLifeBalance_x1.spines['left'].set_visible(False)
    axJobRolesWorkLifeBalance_x1.spines['right'].set_visible(True)

    worklifebalance_values = df_in_1.groupby(['WorkLifeBalance'])['EmployeeNumber'].count()
    worklifebalancelevels = pd.Series(worklifebalance_values, index = worklifebalancelevels_index,                                     name = 'WorkLifeLevels')
    palette = sns.color_palette(worklifebalancelevels_colors)

    sns.barplot(x = worklifebalancelevels.index, y = worklifebalancelevels, palette = palette,                ax = axJobRolesWorkLifeBalance_x1)    .set_title('WorkLifeBalance (Attrition = 1)', size = 14)
    axJobRolesWorkLifeBalance_x1.set_ylabel('Employee Count')
    axJobRolesWorkLifeBalance_x1.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())
    axJobRolesWorkLifeBalance_x1.set_xlabel('WorkLifeBalance')
    axJobRolesWorkLifeBalance_x1.set_xticklabels(worklifebalancelevels_index, **{'rotation': 0})
    
    pos = 0
    total = worklifebalancelevels.sum()
    for index, value in worklifebalancelevels.iteritems():
        if(value > 0):
            axJobRolesWorkLifeBalance_x1.text(pos, value, '{:.1f}%'.format(value / total * 100),                                               color = 'black', ha = 'center')
        pos = pos + 1
    axJobRolesWorkLifeBalance_x1.get_yaxis().set_label_position("right")
    axJobRolesWorkLifeBalance_x1.get_yaxis().tick_right()

    
    
    #[5,0]--------: Barplot for BusinessTravel (Attrition = 0)
    axJobRolesBusinessTravel_x0 = plt.Subplot(figJobRoles, gsJobRoles[5, 0])
    figJobRoles.add_subplot(axJobRolesBusinessTravel_x0)

    businesstravel_values = df_in_0.groupby(['BusinessTravel'])['EmployeeNumber'].count()
    businesstravellevels = pd.Series(businesstravel_values, index = businesstravellevels_index,                                     name = 'BusinessTravelLevels')
    palette = sns.color_palette(businesstravellevels_colors)

    sns.barplot(x = businesstravellevels.index, y = businesstravellevels, palette = palette,                ax = axJobRolesBusinessTravel_x0)    .set_title('BusinessTravel (Attrition = 0)', size = 14)
    axJobRolesBusinessTravel_x0.set_ylabel('Employee Count')
    axJobRolesBusinessTravel_x0.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())
    axJobRolesBusinessTravel_x0.set_xlabel('')
    axJobRolesBusinessTravel_x0.set_xticklabels(businesstravellevels_index, **{'rotation': 0})
    
    pos = 0
    total = businesstravellevels.sum()
    for index, value in businesstravellevels.iteritems():
        axJobRolesBusinessTravel_x0.text(pos, value, '{:.1f}%'.format(value / total * 100),                                      color = 'black', ha = 'center')
        pos = pos + 1    

        
        
    #[5,1]--------: Barplot for BusinessTravel (Attrition = 1)
    axJobRolesBusinessTravel_x1 = plt.Subplot(figJobRoles, gsJobRoles[5, 1])
    figJobRoles.add_subplot(axJobRolesBusinessTravel_x1)
    axJobRolesBusinessTravel_x1.spines['left'].set_visible(False)
    axJobRolesBusinessTravel_x1.spines['right'].set_visible(True)

    businesstravel_values = df_in_1.groupby(['BusinessTravel'])['EmployeeNumber'].count()
    businesstravellevels = pd.Series(businesstravel_values, index = businesstravellevels_index,                                     name = 'BusinessTravelLevels')
    palette = sns.color_palette(businesstravellevels_colors)

    sns.barplot(x = businesstravellevels.index, y = businesstravellevels, palette = palette,                ax = axJobRolesBusinessTravel_x1)    .set_title('BusinessTravel (Attrition = 1)', size = 14)
    axJobRolesBusinessTravel_x1.set_ylabel('Employee Count')
    axJobRolesBusinessTravel_x1.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())
    axJobRolesBusinessTravel_x1.set_xlabel('')
    axJobRolesBusinessTravel_x1.set_xticklabels(businesstravellevels_index, **{'rotation': 0})
    
    pos = 0
    total = businesstravellevels.sum()
    for index, value in businesstravellevels.iteritems():
        axJobRolesBusinessTravel_x1.text(pos, value, '{:.1f}%'.format(value / total * 100),                                          color = 'black', ha = 'center')
        pos = pos + 1  
    axJobRolesBusinessTravel_x1.get_yaxis().set_label_position("right")
    axJobRolesBusinessTravel_x1.get_yaxis().tick_right()    
    

    
    #[6,0]--------: Heatmap for the relation between the JobRole & BusinessTravel (Attrition = 0)
    axJobRolesBusinessTravel_x0 = plt.Subplot(figJobRoles, gsJobRoles[6, 0])
    figJobRoles.add_subplot(axJobRolesBusinessTravel_x0)

    df_JobRole_BusinessTravel0 = pd.DataFrame(np.zeros(shape = (len(JobRole), len(businesstravellevels_index))))
    row = 0
    for JR in JobRole:
        col = 0
        for bt in businesstravellevels_index:
            df_JobRole_BusinessTravel0.iloc[row, col] = (df_in_0.loc[(df_in_0['JobRole'] == JR) &                                                           (df_in_0['BusinessTravel'] == bt)]['EmployeeNumber'].count()) /             (df_in_0.loc[(df_in_0['JobRole'] == JR)]['EmployeeNumber'].count()) * 100
            
            col = col + 1
        row = row + 1

    sns.heatmap(df_JobRole_BusinessTravel0,  cbar = False, annot = True, fmt = '.0f', cmap = 'YlGnBu',                vmin = 0, vmax = 100,                linewidths = .5, ax = axJobRolesBusinessTravel_x0)
    
    axJobRolesBusinessTravel_x0.set_aspect('equal')    
    axJobRolesBusinessTravel_x0.get_yaxis().set_label_position('left')
    axJobRolesBusinessTravel_x0.get_yaxis().tick_left()
    axJobRolesBusinessTravel_x0.invert_yaxis()
    axJobRolesBusinessTravel_x0.set_ylabel('JobRole')
    axJobRolesBusinessTravel_x0.set_yticklabels(JobRole, **{'rotation': 0})  
    axJobRolesBusinessTravel_x0.set_xlabel('BusinessTravel')
    axJobRolesBusinessTravel_x0.set_xticklabels(businesstravellevels_index, **{'rotation': 20})  
    
    
    axdividerJobRolesBusinessTravel_x0 = make_axes_locatable(axJobRolesBusinessTravel_x0)
    axdividerJobRolesBusinessTravel_x0.set_anchor((1,.5))
    caxJobRolesBusinessTravel_x0 = axdividerJobRolesBusinessTravel_x0.append_axes('bottom', size = '5%', pad = '20%')
    colorbar(axJobRolesBusinessTravel_x0.get_children()[0], cax = caxJobRolesBusinessTravel_x0,             orientation = 'horizontal', **{'ticks': (0, 100)})
    caxJobRolesBusinessTravel_x0.xaxis.set_ticks_position('bottom')
    caxJobRolesBusinessTravel_x0.set_xlabel('Emp.Count by Role[%]')
    caxJobRolesBusinessTravel_x0.get_xaxis().set_label_position('bottom')
    
    axJobRolesBusinessTravel_x0.set_title('JobRole vs BusinessTravel (Attrition = 0)', size = 14,                               **{'horizontalalignment': 'right'})
    
    
    
    #[6,1]--------: Heatmap for the relation between the JobRole & BusinessTravel (Attrition = 0)
    axJobRolesBusinessTravel_x1 = plt.Subplot(figJobRoles, gsJobRoles[6, 1])
    figJobRoles.add_subplot(axJobRolesBusinessTravel_x1)

    df_JobRole_BusinessTravel1 = pd.DataFrame(np.zeros(shape = (len(JobRole), len(businesstravellevels_index))))
    row = 0
    for JR in JobRole:
        col = 0
        for bt in businesstravellevels_index:
            df_JobRole_BusinessTravel1.iloc[row, col] = (df_in_1.loc[(df_in_1['JobRole'] == JR) &                                                            (df_in_1['BusinessTravel'] == bt)]['EmployeeNumber'].count()) /             (df_in_1.loc[(df_in_1['JobRole'] == JR)]['EmployeeNumber'].count()) * 100
            
            col = col + 1
        row = row + 1

    sns.heatmap(df_JobRole_BusinessTravel1, cbar = False, annot = True, fmt = '.0f', cmap = 'YlGnBu',                vmin = 0, vmax = 100,                linewidths = .5, ax = axJobRolesBusinessTravel_x1)
    
    axJobRolesBusinessTravel_x1.set_aspect('equal')    
    axJobRolesBusinessTravel_x1.get_yaxis().set_label_position('right')
    axJobRolesBusinessTravel_x1.get_yaxis().tick_right()
    axJobRolesBusinessTravel_x1.invert_yaxis()
    axJobRolesBusinessTravel_x1.set_ylabel('JobRole')
    axJobRolesBusinessTravel_x1.set_yticklabels(JobRole, **{'rotation': 0})  
    axJobRolesBusinessTravel_x1.set_xlabel('BusinessTravel')
    axJobRolesBusinessTravel_x1.set_xticklabels(businesstravellevels_index, **{'rotation': 20})  
    

    axdividerJobRolesBusinessTravel_x1 = make_axes_locatable(axJobRolesBusinessTravel_x1)
    axdividerJobRolesBusinessTravel_x1.set_anchor((0,.5))
    caxJobRolesBusinessTravel_x1 = axdividerJobRolesBusinessTravel_x1.append_axes('bottom', size = '5%', pad = '20%')
    colorbar(axJobRolesBusinessTravel_x1.get_children()[0], cax = caxJobRolesBusinessTravel_x1,
             orientation = 'horizontal', **{'ticks': (0, 100)})
    caxJobRolesBusinessTravel_x1.xaxis.set_ticks_position('bottom')
    caxJobRolesBusinessTravel_x1.set_xlabel('=')
    caxJobRolesBusinessTravel_x1.get_xaxis().set_label_position('bottom')
    
    axJobRolesBusinessTravel_x1.set_title('JobRole vs BusinessTravel (Attrition = 1)', size = 14,                                           **{'horizontalalignment': 'left'})    

    
    
    #[7,0]--------: Barplot for OverTime (Attrition = 0)
    axJobRolesOverTime_x0 = plt.Subplot(figJobRoles, gsJobRoles[7, 0])
    figJobRoles.add_subplot(axJobRolesOverTime_x0)

    overtime_values = df_in_0.groupby(['OverTime'])['EmployeeNumber'].count()
    overtimelevels = pd.Series(overtime_values, index = overtimelevels_index,                                     name = 'OverTimeLevels')
    palette = sns.color_palette(overtimelevels_colors)

    sns.barplot(x = overtimelevels.index, y = overtimelevels, palette = palette,                ax = axJobRolesOverTime_x0)    .set_title('OverTime (Attrition = 0)', size = 14)
    axJobRolesOverTime_x0.set_ylabel('Employee Count')
    axJobRolesOverTime_x0.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())
    axJobRolesOverTime_x0.set_xlabel('')
    axJobRolesOverTime_x0.set_xticklabels(overtimelevels_index, **{'rotation': 0})
    
    pos = 0
    total = overtimelevels.sum()
    for index, value in overtimelevels.iteritems():
        axJobRolesOverTime_x0.text(pos, value, '{:.1f}%'.format(value / total * 100),                                      color = 'black', ha = 'center')
        pos = pos + 1  
    
    
    
    #[7,1]--------: Barplot for OverTime (Attrition = 1)
    axJobRolesOverTime_x1 = plt.Subplot(figJobRoles, gsJobRoles[7, 1])
    figJobRoles.add_subplot(axJobRolesOverTime_x1)
    axJobRolesOverTime_x1.spines['left'].set_visible(False)
    axJobRolesOverTime_x1.spines['right'].set_visible(True)

    overtime_values = df_in_1.groupby(['OverTime'])['EmployeeNumber'].count()
    overtimelevels = pd.Series(overtime_values, index = overtimelevels_index,                                     name = 'OverTimeLevels')
    palette = sns.color_palette(overtimelevels_colors)

    sns.barplot(x = overtimelevels.index, y = overtimelevels, palette = palette,                ax = axJobRolesOverTime_x1)    .set_title('OverTime (Attrition = 1)', size = 14)
    axJobRolesOverTime_x1.set_ylabel('Employee Count')
    axJobRolesOverTime_x1.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())
    axJobRolesOverTime_x1.set_xlabel('')
    axJobRolesOverTime_x1.set_xticklabels(overtimelevels_index, **{'rotation': 0})
    
    pos = 0
    total = overtimelevels.sum()
    for index, value in overtimelevels.iteritems():
        axJobRolesOverTime_x1.text(pos, value, '{:.1f}%'.format(value / total * 100),                                      color = 'black', ha = 'center')
        pos = pos + 1            
    axJobRolesOverTime_x1.get_yaxis().set_label_position("right")
    axJobRolesOverTime_x1.get_yaxis().tick_right() 
    
    
    
    #[8,0]--------: Heatmap for the relation between the JobRole & OverTime (Attrition = 0)
    axJobRolesOverTime_x0 = plt.Subplot(figJobRoles, gsJobRoles[8, 0])
    figJobRoles.add_subplot(axJobRolesOverTime_x0)

    df_JobRole_OverTime0 = pd.DataFrame(np.zeros(shape = (len(JobRole), len(overtimelevels_index))))
    row = 0
    for JR in JobRole:
        col = 0
        for ot in overtimelevels_index:
            df_JobRole_OverTime0.iloc[row, col] = (df_in_0.loc[(df_in_0['JobRole'] == JR) &                                                            (df_in_0['OverTime'] == ot)]['EmployeeNumber'].count()) /             (df_in_0.loc[(df_in_0['JobRole'] == JR)]['EmployeeNumber'].count()) * 100
            col = col + 1
        row = row + 1

    sns.heatmap(df_JobRole_OverTime0, cbar = False, annot = True, fmt = '.0f', cmap = 'YlGnBu',                vmin = 0, vmax = 100,                linewidths = .5, ax = axJobRolesOverTime_x0)
    
    axJobRolesOverTime_x0.set_aspect('equal')    
    axJobRolesOverTime_x0.get_yaxis().set_label_position('left')
    axJobRolesOverTime_x0.get_yaxis().tick_left()
    axJobRolesOverTime_x0.invert_yaxis()
    axJobRolesOverTime_x0.set_ylabel('JobRole')
    axJobRolesOverTime_x0.set_yticklabels(JobRole, **{'rotation': 0}) 
    axJobRolesOverTime_x0.set_xlabel('OverTime')
    axJobRolesOverTime_x0.set_xticklabels(overtimelevels_index, **{'rotation': 0})  
    
    
    axdividerJobRolesOverTime_x0 = make_axes_locatable(axJobRolesOverTime_x0)
    axdividerJobRolesOverTime_x0.set_anchor((1,.5))
    caxJobRolesOverTime_x0 = axdividerJobRolesOverTime_x0.append_axes('bottom', size = '5%', pad = '20%')
    colorbar(axJobRolesOverTime_x0.get_children()[0], cax = caxJobRolesOverTime_x0, orientation = 'horizontal',                         **{'ticks': (0, 100)})
    caxJobRolesOverTime_x0.xaxis.set_ticks_position('bottom')
    caxJobRolesOverTime_x0.set_xlabel('Emp.Count by Role[%]')
    caxJobRolesOverTime_x0.get_xaxis().set_label_position('bottom')
    
    axJobRolesOverTime_x0.set_title('JobRole vs OverTime (Attrition = 0)', size = 14,                               **{'horizontalalignment': 'right'})  
        
        
        
    #[8,1]--------: Heatmap for the relation between the JobRole & OverTime (Attrition = 0)
    axJobRolesOverTime_x1 = plt.Subplot(figJobRoles, gsJobRoles[8, 1])
    figJobRoles.add_subplot(axJobRolesOverTime_x1)

    df_JobRole_OverTime1 = pd.DataFrame(np.zeros(shape = (len(JobRole), len(overtimelevels_index))))
    row = 0
    for JR in JobRole:
        col = 0
        for ot in overtimelevels_index:
            df_JobRole_OverTime1.iloc[row, col] = (df_in_1.loc[(df_in_1['JobRole'] == JR) &                                                            (df_in_1['OverTime'] == ot)]['EmployeeNumber'].count()) /             (df_in_1.loc[(df_in_1['JobRole'] == JR)]['EmployeeNumber'].count()) * 100
            
            col = col + 1
        row = row + 1

    sns.heatmap(df_JobRole_OverTime1, cbar = False, annot = True, fmt = '.0f', cmap = 'YlGnBu',                vmin = 0, vmax = 100,                linewidths = .5, ax = axJobRolesOverTime_x1)
    
    axJobRolesOverTime_x1.set_aspect('equal')    
    axJobRolesOverTime_x1.get_yaxis().set_label_position('right')
    axJobRolesOverTime_x1.get_yaxis().tick_right()
    axJobRolesOverTime_x1.invert_yaxis()
    axJobRolesOverTime_x1.set_ylabel('JobRole')
    axJobRolesOverTime_x1.set_yticklabels(JobRole, **{'rotation': 0})  
    axJobRolesOverTime_x1.set_xlabel('OverTime')
    axJobRolesOverTime_x1.set_xticklabels(overtimelevels_index, **{'rotation': 0})  
    

    axdividerJobRolesOverTime_x1 = make_axes_locatable(axJobRolesOverTime_x1)
    axdividerJobRolesOverTime_x1.set_anchor((0,.5))
    caxJobRolesOverTime_x1 = axdividerJobRolesOverTime_x1.append_axes('bottom', size = '5%', pad = '20%')
    colorbar(axJobRolesOverTime_x1.get_children()[0], cax = caxJobRolesOverTime_x1, orientation = 'horizontal',             **{'ticks': (0, 100)})
    caxJobRolesOverTime_x1.xaxis.set_ticks_position('bottom')
    caxJobRolesOverTime_x1.set_xlabel('=')
    caxJobRolesOverTime_x1.get_xaxis().set_label_position('bottom')
    
    axJobRolesOverTime_x1.set_title('JobRole vs OverTime (Attrition = 1)', size = 14,                                   **{'horizontalalignment': 'left'})  

    return figJobRoles


# In[7]:


def f_JobHistory_analysis(df_in):
    df_in_0 = df_in[df_in['Attrition'] == 0]
    df_in_1 = df_in[df_in['Attrition'] == 1]
    
    
    figJobHistory = plt.figure(figsize = (10, 35), dpi = 80, facecolor = 'w', edgecolor = 'k',                                 constrained_layout = False)
    gsJobHistory = gridspec.GridSpec(nrows = 5, ncols = 2, hspace = .25, wspace = .08,                        height_ratios = [1, 1, 1, 1, 1], width_ratios = [1, 1],                                    figure = figJobHistory)
    
    

    #[0,0]--------: Relation between JobRole & YearsAtCompany (Attrition = 0)
    axJobHistoryYearsAtCompany_x0 = plt.Subplot(figJobHistory, gsJobHistory[0, 0])
    figJobHistory.add_subplot(axJobHistoryYearsAtCompany_x0)

    sns.boxplot(x = 'YearsAtCompany', y = 'JobRole', data = df_in_0, order = JobRole[::-1],                color = '#D1E3FF',                boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                medianprops = medianprops, flierprops = flierprops, whis = whis,                ax = axJobHistoryYearsAtCompany_x0).set_title('JobRole vs YearsAtCompany (Attrition = 0)',                                                                 size = 14)

    pos = range(len(JobRole))
    for tick, label in zip(pos, axJobHistoryYearsAtCompany_x0.get_yticklabels()):
        
        df_aux_0 = df_in_0[df_in_0['JobRole'] == JobRole[::-1][tick]]['YearsAtCompany']
        stats = f_stats_boxplot(df_aux_0)
        axJobHistoryYearsAtCompany_x0.text(stats[1] - .9, pos[tick], '{:.0f}'.format(stats[1]),                                       color = 'red', ha = 'center')
        axJobHistoryYearsAtCompany_x0.text(stats[2] - .9, pos[tick] + .45, '{:.0f}'.format(stats[2]),                                       color = 'black', ha = 'center')
        axJobHistoryYearsAtCompany_x0.text(stats[3] + .9, pos[tick], '{:.0f}'.format(stats[3]),                                       color = 'black', ha = 'center')
        axJobHistoryYearsAtCompany_x0.text(stats[4] + .9, pos[tick] + .45, '{:.0f}'.format(stats[4]),                                       color = 'black', ha = 'center')
        axJobHistoryYearsAtCompany_x0.text(stats[5] + .9, pos[tick], '{:.0f}'.format(stats[5]),                                       color = 'red', ha = 'center')    
    
    axJobHistoryYearsAtCompany_x0.set_xlabel('YearsAtCompany')
    axJobHistoryYearsAtCompany_x0.set_ylabel('JobRole')
    axJobHistoryYearsAtCompany_x0.set_xlim((-3, 42))
    axJobHistoryYearsAtCompany_x0.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
    
    
    
    #[0,1]--------: Relation between JobRole & TotalWorkingYears (Attrition = 1)
    axJobHistoryYearsAtCompany_x1 = plt.Subplot(figJobHistory, gsJobHistory[0, 1])
    figJobHistory.add_subplot(axJobHistoryYearsAtCompany_x1)
    axJobHistoryYearsAtCompany_x1.spines['left'].set_visible(False)
    axJobHistoryYearsAtCompany_x1.spines['right'].set_visible(True)

    sns.boxplot(x = 'YearsAtCompany', y = 'JobRole', data = df_in_1, order = JobRole[::-1],                color = '#D1E3FF',                boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                medianprops = medianprops, flierprops = flierprops, whis = whis,                ax = axJobHistoryYearsAtCompany_x1).set_title('JobRole vs YearsAtCompany (Attrition = 1)', size = 14)

    pos = range(len(JobRole))
    for tick, label in zip(pos, axJobHistoryYearsAtCompany_x1.get_yticklabels()):
        
        df_aux_1 = df_in_1[df_in_1['JobRole'] == JobRole[::-1][tick]]['YearsAtCompany']
        stats = f_stats_boxplot(df_aux_1)
        axJobHistoryYearsAtCompany_x1.text(stats[1] - .9, pos[tick], '{:.0f}'.format(stats[1]),                                       color = 'red', ha = 'center')
        axJobHistoryYearsAtCompany_x1.text(stats[2] - .9, pos[tick] + .45, '{:.0f}'.format(stats[2]),                                       color = 'black', ha = 'center')
        axJobHistoryYearsAtCompany_x1.text(stats[3] + .9, pos[tick], '{:.0f}'.format(stats[3]),                                       color = 'black', ha = 'center')
        axJobHistoryYearsAtCompany_x1.text(stats[4] + .9, pos[tick] + .45, '{:.0f}'.format(stats[4]),                                       color = 'black', ha = 'center')
        axJobHistoryYearsAtCompany_x1.text(stats[5] + .9, pos[tick], '{:.0f}'.format(stats[5]),                                       color = 'red', ha = 'center')    
    
    axJobHistoryYearsAtCompany_x1.set_xlabel('YearsAtCompany')
    axJobHistoryYearsAtCompany_x1.set_ylabel('JobRole')
    axJobHistoryYearsAtCompany_x1.get_yaxis().set_label_position("right")
    axJobHistoryYearsAtCompany_x1.get_yaxis().tick_right()
    axJobHistoryYearsAtCompany_x1.set_xlim((-3, 42))
    axJobHistoryYearsAtCompany_x1.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())

    
    
    #[1,0]--------: Relation between JobRole & YearsWithCurrManager (Attrition = 0)
    axJobHistoryYearsWithCurrManager_x0 = plt.Subplot(figJobHistory, gsJobHistory[1, 0])
    figJobHistory.add_subplot(axJobHistoryYearsWithCurrManager_x0)

    sns.boxplot(x = 'YearsWithCurrManager', y = 'JobRole', data = df_in_0, order = JobRole[::-1],                color = '#D1E3FF',                boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                medianprops = medianprops, flierprops = flierprops, whis = whis,                ax = axJobHistoryYearsWithCurrManager_x0).set_title('JobRole vs Y.WithCurrManager (Attrition = 0)',                                                                 size = 14)

    pos = range(len(JobRole))
    for tick, label in zip(pos, axJobHistoryYearsWithCurrManager_x0.get_yticklabels()):
        
        df_aux_0 = df_in_0[df_in_0['JobRole'] == JobRole[::-1][tick]]['YearsWithCurrManager']
        stats = f_stats_boxplot(df_aux_0)
        axJobHistoryYearsWithCurrManager_x0.text(stats[1] - .5, pos[tick], '{:.0f}'.format(stats[1]),                                       color = 'red', ha = 'center')
        axJobHistoryYearsWithCurrManager_x0.text(stats[2] - .5, pos[tick] + .45, '{:.0f}'.format(stats[2]),                                       color = 'black', ha = 'center')
        axJobHistoryYearsWithCurrManager_x0.text(stats[3] + .3, pos[tick], '{:.0f}'.format(stats[3]),                                       color = 'black', ha = 'center')
        axJobHistoryYearsWithCurrManager_x0.text(stats[4] + .5, pos[tick] + .45, '{:.0f}'.format(stats[4]),                                       color = 'black', ha = 'center')
        axJobHistoryYearsWithCurrManager_x0.text(stats[5] + .5, pos[tick], '{:.0f}'.format(stats[5]),                                       color = 'red', ha = 'center')    
    
    axJobHistoryYearsWithCurrManager_x0.set_xlabel('YearsWithCurrManager')
    axJobHistoryYearsWithCurrManager_x0.set_ylabel('JobRole')
    axJobHistoryYearsWithCurrManager_x0.set_xlim((-1, 20))
    axJobHistoryYearsWithCurrManager_x0.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
    
    
    
    #[1,1]--------: Relation between JobRole & YearsWithCurrManager (Attrition = 1)
    axJobHistoryYearsWithCurrManager_x1 = plt.Subplot(figJobHistory, gsJobHistory[1, 1])
    figJobHistory.add_subplot(axJobHistoryYearsWithCurrManager_x1)
    axJobHistoryYearsWithCurrManager_x1.spines['left'].set_visible(False)
    axJobHistoryYearsWithCurrManager_x1.spines['right'].set_visible(True)

    sns.boxplot(x = 'YearsWithCurrManager', y = 'JobRole', data = df_in_1, order = JobRole[::-1],                color = '#D1E3FF',                boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                medianprops = medianprops, flierprops = flierprops, whis = whis,                ax = axJobHistoryYearsWithCurrManager_x1).set_title('JobRole vs Y.WithCurrManager (Attrition = 1)',                                                                    size = 14)

    pos = range(len(JobRole))
    for tick, label in zip(pos, axJobHistoryYearsWithCurrManager_x1.get_yticklabels()):
        
        df_aux_1 = df_in_1[df_in_1['JobRole'] == JobRole[::-1][tick]]['YearsWithCurrManager']
        stats = f_stats_boxplot(df_aux_1)
        axJobHistoryYearsWithCurrManager_x1.text(stats[1] - .5, pos[tick], '{:.0f}'.format(stats[1]),                                       color = 'red', ha = 'center')
        axJobHistoryYearsWithCurrManager_x1.text(stats[2] - .5, pos[tick] + .45, '{:.0f}'.format(stats[2]),                                       color = 'black', ha = 'center')
        axJobHistoryYearsWithCurrManager_x1.text(stats[3] + .3, pos[tick], '{:.0f}'.format(stats[3]),                                       color = 'black', ha = 'center')
        axJobHistoryYearsWithCurrManager_x1.text(stats[4] + .5, pos[tick] + .45, '{:.0f}'.format(stats[4]),                                       color = 'black', ha = 'center')
        axJobHistoryYearsWithCurrManager_x1.text(stats[5] + .5, pos[tick], '{:.0f}'.format(stats[5]),                                       color = 'red', ha = 'center')    
    
    axJobHistoryYearsWithCurrManager_x1.set_xlabel('YearsWithCurrManager')
    axJobHistoryYearsWithCurrManager_x1.set_ylabel('JobRole')
    axJobHistoryYearsWithCurrManager_x1.get_yaxis().set_label_position("right")
    axJobHistoryYearsWithCurrManager_x1.get_yaxis().tick_right()
    axJobHistoryYearsWithCurrManager_x1.set_xlim((-1, 20))
    axJobHistoryYearsWithCurrManager_x1.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())

    
    
    #[2,0]--------: Relation between JobRole & YearsInCurrentRole (Attrition = 0)
    axJobHistoryYearsInCurrentRole_x0 = plt.Subplot(figJobHistory, gsJobHistory[2, 0])
    figJobHistory.add_subplot(axJobHistoryYearsInCurrentRole_x0)

    sns.boxplot(x = 'YearsInCurrentRole', y = 'JobRole', data = df_in_0, order = JobRole[::-1],                color = '#D1E3FF',                boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                medianprops = medianprops, flierprops = flierprops, whis = whis,                ax = axJobHistoryYearsInCurrentRole_x0).set_title('JobRole vs Y.InCurrentRole (Attrition = 0)',                                                                 size = 14)

    pos = range(len(JobRole))
    for tick, label in zip(pos, axJobHistoryYearsInCurrentRole_x0.get_yticklabels()):
        
        df_aux_0 = df_in_0[df_in_0['JobRole'] == JobRole[::-1][tick]]['YearsInCurrentRole']
        stats = f_stats_boxplot(df_aux_0)
        axJobHistoryYearsInCurrentRole_x0.text(stats[1] - .5, pos[tick], '{:.0f}'.format(stats[1]),                                       color = 'red', ha = 'center')
        axJobHistoryYearsInCurrentRole_x0.text(stats[2] - .5, pos[tick] + .45, '{:.0f}'.format(stats[2]),                                       color = 'black', ha = 'center')
        axJobHistoryYearsInCurrentRole_x0.text(stats[3] + .3, pos[tick], '{:.0f}'.format(stats[3]),                                       color = 'black', ha = 'center')
        axJobHistoryYearsInCurrentRole_x0.text(stats[4] + .5, pos[tick] + .45, '{:.0f}'.format(stats[4]),                                       color = 'black', ha = 'center')
        axJobHistoryYearsInCurrentRole_x0.text(stats[5] + .5, pos[tick], '{:.0f}'.format(stats[5]),                                       color = 'red', ha = 'center')    
    
    axJobHistoryYearsInCurrentRole_x0.set_xlabel('YearsInCurrentRole')
    axJobHistoryYearsInCurrentRole_x0.set_ylabel('JobRole')
    axJobHistoryYearsInCurrentRole_x0.set_xlim((-1, 20))
    axJobHistoryYearsInCurrentRole_x0.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
    
    
    
    #[2,1]--------: Relation between JobRole & YearsInCurrentRole (Attrition = 1)
    axJobHistoryYearsInCurrentRole_x1 = plt.Subplot(figJobHistory, gsJobHistory[2, 1])
    figJobHistory.add_subplot(axJobHistoryYearsInCurrentRole_x1)
    axJobHistoryYearsInCurrentRole_x1.spines['left'].set_visible(False)
    axJobHistoryYearsInCurrentRole_x1.spines['right'].set_visible(True)

    sns.boxplot(x = 'YearsInCurrentRole', y = 'JobRole', data = df_in_1, order = JobRole[::-1],                color = '#D1E3FF',                boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                medianprops = medianprops, flierprops = flierprops, whis = whis,                ax = axJobHistoryYearsInCurrentRole_x1).set_title('JobRole vs Y.InCurrentRole (Attrition = 1)',                                                                    size = 14)

    pos = range(len(JobRole))
    for tick, label in zip(pos, axJobHistoryYearsInCurrentRole_x1.get_yticklabels()):
        
        df_aux_1 = df_in_1[df_in_1['JobRole'] == JobRole[::-1][tick]]['YearsInCurrentRole']
        stats = f_stats_boxplot(df_aux_1)
        axJobHistoryYearsInCurrentRole_x1.text(stats[1] - .5, pos[tick], '{:.0f}'.format(stats[1]),                                       color = 'red', ha = 'center')
        axJobHistoryYearsInCurrentRole_x1.text(stats[2] - .5, pos[tick] + .45, '{:.0f}'.format(stats[2]),                                       color = 'black', ha = 'center')
        axJobHistoryYearsInCurrentRole_x1.text(stats[3] + .3, pos[tick], '{:.0f}'.format(stats[3]),                                       color = 'black', ha = 'center')
        axJobHistoryYearsInCurrentRole_x1.text(stats[4] + .5, pos[tick] + .45, '{:.0f}'.format(stats[4]),                                       color = 'black', ha = 'center')
        axJobHistoryYearsInCurrentRole_x1.text(stats[5] + .5, pos[tick], '{:.0f}'.format(stats[5]),                                       color = 'red', ha = 'center')    
    
    axJobHistoryYearsInCurrentRole_x1.set_xlabel('YearsInCurrentRole')
    axJobHistoryYearsInCurrentRole_x1.set_ylabel('JobRole')
    axJobHistoryYearsInCurrentRole_x1.get_yaxis().set_label_position("right")
    axJobHistoryYearsInCurrentRole_x1.get_yaxis().tick_right()
    axJobHistoryYearsInCurrentRole_x1.set_xlim((-1, 20))
    axJobHistoryYearsInCurrentRole_x1.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())

    
    
    #[3,0]--------: Relation between JobRole & NumCompaniesWorked (Attrition = 0)
    axJobHistoryNumCompaniesWorked_x0 = plt.Subplot(figJobHistory, gsJobHistory[3, 0])
    figJobHistory.add_subplot(axJobHistoryNumCompaniesWorked_x0)

    sns.boxplot(x = 'NumCompaniesWorked', y = 'JobRole', data = df_in_0, order = JobRole[::-1],                color = '#D1E3FF',                boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                medianprops = medianprops, flierprops = flierprops, whis = whis,                ax = axJobHistoryNumCompaniesWorked_x0).set_title('JobRole vs N.Comp.Worked (Attrition = 0)',                                                                 size = 14)

    pos = range(len(JobRole))
    for tick, label in zip(pos, axJobHistoryNumCompaniesWorked_x0.get_yticklabels()):
        
        df_aux_0 = df_in_0[df_in_0['JobRole'] == JobRole[::-1][tick]]['NumCompaniesWorked']
        stats = f_stats_boxplot(df_aux_0)
        axJobHistoryNumCompaniesWorked_x0.text(stats[1] - .2, pos[tick], '{:.0f}'.format(stats[1]),                                       color = 'red', ha = 'center')
        axJobHistoryNumCompaniesWorked_x0.text(stats[2] - .2, pos[tick] + .45, '{:.0f}'.format(stats[2]),                                       color = 'black', ha = 'center')
        axJobHistoryNumCompaniesWorked_x0.text(stats[3] + .15, pos[tick], '{:.0f}'.format(stats[3]),                                       color = 'black', ha = 'center')
        axJobHistoryNumCompaniesWorked_x0.text(stats[4] + .2, pos[tick] + .45, '{:.0f}'.format(stats[4]),                                       color = 'black', ha = 'center')
        axJobHistoryNumCompaniesWorked_x0.text(stats[5] + .2, pos[tick], '{:.0f}'.format(stats[5]),                                       color = 'red', ha = 'center')    
    
    axJobHistoryNumCompaniesWorked_x0.set_xlabel('NumCompaniesWorked')
    axJobHistoryNumCompaniesWorked_x0.set_ylabel('JobRole')
    axJobHistoryNumCompaniesWorked_x0.set_xlim((-1, 11))
    axJobHistoryNumCompaniesWorked_x0.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
    
    
    
    #[3,1]--------: Relation between JobRole & NumCompaniesWorked (Attrition = 1)
    axJobHistoryNumCompaniesWorked_x1 = plt.Subplot(figJobHistory, gsJobHistory[3, 1])
    figJobHistory.add_subplot(axJobHistoryNumCompaniesWorked_x1)
    axJobHistoryNumCompaniesWorked_x1.spines['left'].set_visible(False)
    axJobHistoryNumCompaniesWorked_x1.spines['right'].set_visible(True)

    sns.boxplot(x = 'NumCompaniesWorked', y = 'JobRole', data = df_in_1, order = JobRole[::-1],                color = '#D1E3FF',                boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                medianprops = medianprops, flierprops = flierprops, whis = whis,                ax = axJobHistoryNumCompaniesWorked_x1).set_title('JobRole vs N.Comp.Worked (Attrition = 1)',                                                                    size = 14)

    pos = range(len(JobRole))
    for tick, label in zip(pos, axJobHistoryNumCompaniesWorked_x1.get_yticklabels()):
        
        df_aux_1 = df_in_1[df_in_1['JobRole'] == JobRole[::-1][tick]]['NumCompaniesWorked']
        stats = f_stats_boxplot(df_aux_1)
        axJobHistoryNumCompaniesWorked_x1.text(stats[1] - .2, pos[tick], '{:.0f}'.format(stats[1]),                                       color = 'red', ha = 'center')
        axJobHistoryNumCompaniesWorked_x1.text(stats[2] - .2, pos[tick] + .45, '{:.0f}'.format(stats[2]),                                       color = 'black', ha = 'center')
        axJobHistoryNumCompaniesWorked_x1.text(stats[3] + .15, pos[tick], '{:.0f}'.format(stats[3]),                                       color = 'black', ha = 'center')
        axJobHistoryNumCompaniesWorked_x1.text(stats[4] + .2, pos[tick] + .45, '{:.0f}'.format(stats[4]),                                       color = 'black', ha = 'center')
        axJobHistoryNumCompaniesWorked_x1.text(stats[5] + .2, pos[tick], '{:.0f}'.format(stats[5]),                                       color = 'red', ha = 'center')    
    
    axJobHistoryNumCompaniesWorked_x1.set_xlabel('NumCompaniesWorked')
    axJobHistoryNumCompaniesWorked_x1.set_ylabel('JobRole')
    axJobHistoryNumCompaniesWorked_x1.get_yaxis().set_label_position("right")
    axJobHistoryNumCompaniesWorked_x1.get_yaxis().tick_right()
    axJobHistoryNumCompaniesWorked_x1.set_xlim((-1, 11))
    axJobHistoryNumCompaniesWorked_x1.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
    

    
    #[4,0]--------: Relation between YAtCompany_TotalWorkY_dv & NumCompaniesWorked (Attrition = 0)
    axYatCompTotal_NumComp_x0 = plt.Subplot(figJobHistory, gsJobHistory[4, 0])
    figJobHistory.add_subplot(axYatCompTotal_NumComp_x0)

    sns.scatterplot(x = 'NumCompaniesWorked', y = 'YAtCompany_TotalWorkY_dv',                    data = df_in_0,                    edgecolor = 'black', linewidth = .3, color = '#303D69', marker = 'o', s = 20, alpha = .5,                    ax = axYatCompTotal_NumComp_x0).set_title('(Attrition = 0)', size = 14)

    axYatCompTotal_NumComp_x0.set_xlabel('NumCompaniesWorked')
    axYatCompTotal_NumComp_x0.set_xlim((-1, 11))
    axYatCompTotal_NumComp_x0.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())

    axYatCompTotal_NumComp_x0.set_ylabel('YAtCompany_TotalWorkY_dv')
    axYatCompTotal_NumComp_x0.set_ylim((-.07, 1.07))
    axYatCompTotal_NumComp_x0.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())

    # Plot line Y=x
    x_lim = axYatCompTotal_NumComp_x0.get_xlim()[1]
    y_lim = axYatCompTotal_NumComp_x0.get_ylim()[1]
    axYatCompTotal_NumComp_x0.plot((0, min(x_lim, y_lim)), (0, min(x_lim, y_lim)), color = 'red',                                   alpha = 0.75, zorder = 0)
    axYatCompTotal_NumComp_x0.text(0, 0, 'Y=x', ha = 'center', **dict(size = 10, color = 'red'))   
    
    

    #[4,1]--------: Relation between YAtCompany_TotalWorkY_dv & NumCompaniesWorked (Attrition = 1)
    axYatCompTotal_NumComp_x1 = plt.Subplot(figJobHistory, gsJobHistory[4, 1])
    figJobHistory.add_subplot(axYatCompTotal_NumComp_x1)
    axYatCompTotal_NumComp_x1.spines['left'].set_visible(False)
    axYatCompTotal_NumComp_x1.spines['right'].set_visible(True)

    sns.scatterplot(x = 'NumCompaniesWorked', y = 'YAtCompany_TotalWorkY_dv',                    data = df_in_1,                    edgecolor = 'black', linewidth = .3, color = '#303D69', marker = 'o', s = 20, alpha = .5,                    ax = axYatCompTotal_NumComp_x1).set_title('(Attrition = 1)', size = 14)

    axYatCompTotal_NumComp_x1.set_xlabel('NumCompaniesWorked')
    axYatCompTotal_NumComp_x1.set_xlim((-1, 11))
    axYatCompTotal_NumComp_x1.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())


    axYatCompTotal_NumComp_x1.set_ylabel('YAtCompany_TotalWorkY_dv')
    axYatCompTotal_NumComp_x1.set_ylim((-.07, 1.07))
    axYatCompTotal_NumComp_x1.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())
    axYatCompTotal_NumComp_x1.get_yaxis().set_label_position("right")
    axYatCompTotal_NumComp_x1.get_yaxis().tick_right() 
    
    # Plot line Y=x
    x_lim = axYatCompTotal_NumComp_x1.get_xlim()[1]
    y_lim = axYatCompTotal_NumComp_x1.get_ylim()[1]
    axYatCompTotal_NumComp_x1.plot((0, min(x_lim, y_lim)), (0, min(x_lim, y_lim)), color = 'red',                                   alpha = 0.75, zorder = 0)
    axYatCompTotal_NumComp_x1.text(0, 0, 'Y=x', ha = 'center', **dict(size = 10, color = 'red'))  
    
    return figJobHistory


# In[8]:


def f_PerfSatisfTran_analysis(df_in):
    # Define indexes
    # PerformanceRating
    min_PerformanceRating = df_in['PerformanceRating'].min()
    max_PerformanceRating = df_in['PerformanceRating'].max()
    PerformanceRating = range(min_PerformanceRating, max_PerformanceRating + 1)

    # JobInvolvement
    min_JobInvolvement = df_in['JobInvolvement'].min()
    max_JobInvolvement = df_in['JobInvolvement'].max()
    JobInvolvement = range(min_JobInvolvement, max_JobInvolvement + 1)

    # JobSatisfaction
    min_JobSatisfaction = df_in['JobSatisfaction'].min()
    max_JobSatisfaction = df_in['JobSatisfaction'].max()
    JobSatisfaction = range(min_JobSatisfaction, max_JobSatisfaction + 1)

    # EnvironmentSatisfaction
    min_EnvironmentSatisfaction = df_in['EnvironmentSatisfaction'].min()
    max_EnvironmentSatisfaction = df_in['EnvironmentSatisfaction'].max()
    EnvironmentSatisfaction = range(min_EnvironmentSatisfaction, max_EnvironmentSatisfaction + 1)

    # RelationshipSatisfaction
    min_RelationshipSatisfaction = df_in['RelationshipSatisfaction'].min()
    max_RelationshipSatisfaction = df_in['RelationshipSatisfaction'].max()
    RelationshipSatisfaction = range(min_RelationshipSatisfaction, max_RelationshipSatisfaction + 1)

    # TrainingTimesLastYear
    min_TrainingTimesLastYear = df_in['TrainingTimesLastYear'].min()
    max_TrainingTimesLastYear = df_in['TrainingTimesLastYear'].max()
    TrainingTimesLastYear = range(min_TrainingTimesLastYear, max_TrainingTimesLastYear + 1)    
    
    
    # Create the plot
    df_in_0 = df_in[df_in['Attrition'] == 0]
    df_in_1 = df_in[df_in['Attrition'] == 1]
    
    
    figPerfSatisfTran = plt.figure(figsize = (10, 70), dpi = 80, facecolor = 'w', edgecolor = 'k',                                 constrained_layout = False)
    gsPerfSatisfTran = gridspec.GridSpec(nrows = 12, ncols = 2, hspace = .3, wspace = .08,                         height_ratios = [1, .7, 1, .7, 1, .7, 1, .7, 1, .7, 1, .7], width_ratios = [1, 1],                          figure = figPerfSatisfTran)
    

    #[0,0]--------: Heatmap for the relation between the JobRole & PerformanceRating (Attrition = 0)
    axJobRolePerf_x0 = plt.Subplot(figPerfSatisfTran, gsPerfSatisfTran[0, 0])
    axJobRolePerf_x0.set_anchor((1,.5))
    figPerfSatisfTran.add_subplot(axJobRolePerf_x0)

    df_JobRolePerfl0 = pd.DataFrame(np.zeros(shape = (len(JobRole), len(PerformanceRating))))
    row = 0
    for JR in JobRole:
        col = 0
        for p in PerformanceRating:
            df_JobRolePerfl0.iloc[row, col] = (df_in_0.loc[(df_in_0['JobRole'] == JR) &                                                            (df_in_0['PerformanceRating'] == p)]                                                              ['EmployeeNumber'].count()) /             (df_in_0.loc[(df_in_0['JobRole'] == JR)]['EmployeeNumber'].count()) * 100 
                         
            col = col + 1
        row = row + 1

    sns.heatmap(df_JobRolePerfl0, cbar = False, annot = True, fmt = '.0f', cmap = 'YlGnBu',                vmin = 0, vmax = 100,                linewidths = .5, ax = axJobRolePerf_x0)
    
    axJobRolePerf_x0.set_aspect('equal')    
    axJobRolePerf_x0.get_yaxis().set_label_position('left')
    axJobRolePerf_x0.get_yaxis().tick_left()
    axJobRolePerf_x0.invert_yaxis()
    axJobRolePerf_x0.set_ylabel('JobRole')
    axJobRolePerf_x0.set_yticklabels(JobRole, **{'rotation': 0})  
    axJobRolePerf_x0.set_xlabel('Perf.Rating')
    axJobRolePerf_x0.set_xticklabels(PerformanceRating, **{'rotation': 0})  
    
    axdividerJobRolePerf_x0 = make_axes_locatable(axJobRolePerf_x0)
    axdividerJobRolePerf_x0.set_anchor((1,.5))
    caxJobRolePerf_x0 = axdividerJobRolePerf_x0.append_axes('bottom', size = '5%', pad = '20%')
    colorbar(axJobRolePerf_x0.get_children()[0], cax = caxJobRolePerf_x0, orientation = 'horizontal',             **{'ticks': (0, 100)})
    caxJobRolePerf_x0.xaxis.set_ticks_position('bottom')
    caxJobRolePerf_x0.set_xlabel('Emp.Count by Role[%]')
    caxJobRolePerf_x0.get_xaxis().set_label_position('bottom')
    
    axJobRolePerf_x0.set_title('JobRole vs PerformanceRating (Attrition = 0)', size = 14,                               **{'horizontalalignment': 'right'})

    
    
    #[0,1]--------: Heatmap for the relation between the JobRole & PerformanceRating (Attrition = 1)
    axJobRolePerf_x1 = plt.Subplot(figPerfSatisfTran, gsPerfSatisfTran[0, 1])
    axJobRolePerf_x1.set_anchor((0,.5))
    figPerfSatisfTran.add_subplot(axJobRolePerf_x1)

    df_JobRolePerfl1 = pd.DataFrame(np.zeros(shape = (len(JobRole), len(PerformanceRating))))
    row = 0
    for JR in JobRole:
        col = 0
        for p in PerformanceRating:
            df_JobRolePerfl1.iloc[row, col] = df_in_1.loc[(df_in_1['JobRole'] == JR) &                                                            (df_in_1['PerformanceRating'] == p)]                                                                         ['EmployeeNumber'].count() /             (df_in_1.loc[(df_in_1['JobRole'] == JR)]['EmployeeNumber'].count()) * 100
            
            col = col + 1
        row = row + 1

    sns.heatmap(df_JobRolePerfl1, cbar = False, annot = True, fmt = '.0f', cmap = 'YlGnBu',                vmin = 0, vmax = 100,                linewidths = .5, ax = axJobRolePerf_x1)
    
    axJobRolePerf_x1.set_aspect('equal')    
    axJobRolePerf_x1.get_yaxis().set_label_position('right')
    axJobRolePerf_x1.get_yaxis().tick_right()
    axJobRolePerf_x1.invert_yaxis()
    axJobRolePerf_x1.set_ylabel('JobRole')
    axJobRolePerf_x1.set_yticklabels(JobRole, **{'rotation': 0})  
    axJobRolePerf_x1.set_xlabel('Perf.Rating')
    axJobRolePerf_x1.set_xticklabels(PerformanceRating, **{'rotation': 0})  
    

    axdividerJobRolePerf_x1 = make_axes_locatable(axJobRolePerf_x1)
    axdividerJobRolePerf_x1.set_anchor((0,.5))
    caxJobRolePerf_x1 = axdividerJobRolePerf_x1.append_axes('bottom', size = '5%', pad = '20%')
    colorbar(axJobRolePerf_x1.get_children()[0], cax = caxJobRolePerf_x1, orientation = 'horizontal',             **{'ticks': (0, 100)})
    caxJobRolePerf_x1.xaxis.set_ticks_position('bottom')
    caxJobRolePerf_x1.set_xlabel('=')
    caxJobRolePerf_x1.get_xaxis().set_label_position('bottom')
    
    axJobRolePerf_x1.set_title('JobRole vs PerformanceRating (Attrition = 1)', size = 14,                                   **{'horizontalalignment': 'left'})

    
        
    #[1,0]--------: Relation between YearsAtCompany & PerformanceRating (Attrition = 0)
    axYearsAtCompanyInvolvement_x0 = plt.Subplot(figPerfSatisfTran, gsPerfSatisfTran[1, 0])
    axYearsAtCompanyInvolvement_x0.set_anchor((0,.5))
    figPerfSatisfTran.add_subplot(axYearsAtCompanyInvolvement_x0)

    sns.boxplot(x = 'PerformanceRating', y = 'YearsAtCompany', data = df_in_0, palette = palette_relscale,                boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                medianprops = medianprops, flierprops = flierprops, whis = whis,                ax = axYearsAtCompanyInvolvement_x0).set_title('Y.AtComp. vs Perf.Rating (Attrition = 0)', size = 14)

    pos = range(len(JobInvolvement))
    for tick, label in zip(pos, axYearsAtCompanyInvolvement_x0.get_xticklabels()):
        df_aux = df_in_0.loc[(df_in_0['PerformanceRating'] == int(label.get_text()))]

        stats = f_stats_boxplot(df_aux['YearsAtCompany'])
        axYearsAtCompanyInvolvement_x0.text(pos[tick] - .15, stats[1] - 2.5, '{:.0f}'.format(stats[1]),                                       color = 'red', ha = 'center')
        axYearsAtCompanyInvolvement_x0.text(pos[tick] + .2, stats[2] - 2.5, '{:.0f}'.format(stats[2]),                                       color = 'black', ha = 'center')
        axYearsAtCompanyInvolvement_x0.text(pos[tick], stats[3], '{:.0f}'.format(stats[3]),                                       color = 'black', ha = 'center')
        axYearsAtCompanyInvolvement_x0.text(pos[tick] + .2, stats[4] + 1, '{:.0f}'.format(stats[4]),                                       color = 'black', ha = 'center')
        axYearsAtCompanyInvolvement_x0.text(pos[tick] - .15, stats[5] + 1, '{:.0f}'.format(stats[5]),                                       color = 'red', ha = 'center')    

    axYearsAtCompanyInvolvement_x0.set_xlabel('PerformanceRating')
    axYearsAtCompanyInvolvement_x0.set_ylabel('YearsAtCompany')
    axYearsAtCompanyInvolvement_x0.set_ylim((-3,42))
    axYearsAtCompanyInvolvement_x0.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())

    
    
    #[1,1]--------: Relation between YearsAtCompany & JobInvolvement (Attrition = 1)
    axYearsAtCompanyInvolvement_x1 = plt.Subplot(figPerfSatisfTran, gsPerfSatisfTran[1, 1])
    axYearsAtCompanyInvolvement_x1.set_anchor((0,.5))
    figPerfSatisfTran.add_subplot(axYearsAtCompanyInvolvement_x1)
    axYearsAtCompanyInvolvement_x1.spines['left'].set_visible(False)
    axYearsAtCompanyInvolvement_x1.spines['right'].set_visible(True)

    sns.boxplot(x = 'PerformanceRating', y = 'YearsAtCompany', data = df_in_1, palette = palette_relscale,                 boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                medianprops = medianprops, flierprops = flierprops, whis = whis,                ax = axYearsAtCompanyInvolvement_x1).set_title('Y.AtComp. vs Perf.Rating (Attrition = 1)', size = 14)

    pos = range(len(JobInvolvement))
    for tick, label in zip(pos, axYearsAtCompanyInvolvement_x1.get_xticklabels()):
        df_aux = df_in_1.loc[(df_in_1['PerformanceRating'] == int(label.get_text()))]

        stats = f_stats_boxplot(df_aux['YearsAtCompany'])
        axYearsAtCompanyInvolvement_x1.text(pos[tick] - .15, stats[1] - 2.5, '{:.0f}'.format(stats[1]),                                       color = 'red', ha = 'center')
        axYearsAtCompanyInvolvement_x1.text(pos[tick] + .2, stats[2] - 2.5, '{:.0f}'.format(stats[2]),                                       color = 'black', ha = 'center')
        axYearsAtCompanyInvolvement_x1.text(pos[tick], stats[3], '{:.0f}'.format(stats[3]),                                       color = 'black', ha = 'center')
        axYearsAtCompanyInvolvement_x1.text(pos[tick] + .2, stats[4] + 1, '{:.0f}'.format(stats[4]),                                       color = 'black', ha = 'center')
        axYearsAtCompanyInvolvement_x1.text(pos[tick] - .15, stats[5] + 1, '{:.0f}'.format(stats[5]),                                       color = 'red', ha = 'center')    

    axYearsAtCompanyInvolvement_x1.set_xlabel('PerformanceRating')
    axYearsAtCompanyInvolvement_x1.set_ylabel('YearsAtCompany')
    axYearsAtCompanyInvolvement_x1.set_ylim((-3,42))
    axYearsAtCompanyInvolvement_x1.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())    
    axYearsAtCompanyInvolvement_x1.get_yaxis().set_label_position("right")
    axYearsAtCompanyInvolvement_x1.get_yaxis().tick_right()  

    
    #[2,0]--------: Heatmap for the relation between the JobRole & JobInvolvement (Attrition = 0)
    axJobRoleInvolvement_x0 = plt.Subplot(figPerfSatisfTran, gsPerfSatisfTran[2, 0])
    axJobRoleInvolvement_x0.set_anchor((1,.5))
    figPerfSatisfTran.add_subplot(axJobRoleInvolvement_x0)

    df_JobRoleInvolvementl0 = pd.DataFrame(np.zeros(shape = (len(JobRole), len(JobInvolvement))))
    row = 0
    for JR in JobRole:
        col = 0
        for JI in JobInvolvement:
            df_JobRoleInvolvementl0.iloc[row, col] = (df_in_0.loc[(df_in_0['JobRole'] == JR) &                                                            (df_in_0['JobInvolvement'] == JI)]                                                              ['EmployeeNumber'].count()) /             (df_in_0.loc[(df_in_0['JobRole'] == JR)]['EmployeeNumber'].count()) * 100
            
            col = col + 1
        row = row + 1
    
    sns.heatmap(df_JobRoleInvolvementl0, cbar = False, annot = True, fmt = '.0f', cmap = 'YlGnBu',                vmin = 0, vmax = 100,                linewidths = .5, ax = axJobRoleInvolvement_x0)
    
    axJobRoleInvolvement_x0.set_aspect('equal')    
    axJobRoleInvolvement_x0.get_yaxis().set_label_position('left')
    axJobRoleInvolvement_x0.get_yaxis().tick_left()
    axJobRoleInvolvement_x0.invert_yaxis()
    axJobRoleInvolvement_x0.set_ylabel('JobRole')
    axJobRoleInvolvement_x0.set_yticklabels(JobRole, **{'rotation': 0})  
    axJobRoleInvolvement_x0.set_xlabel('JobInvolvement')
    axJobRoleInvolvement_x0.set_xticklabels(JobInvolvement, **{'rotation': 0})  
    
    axdividerJobRoleInvolvement_x0 = make_axes_locatable(axJobRoleInvolvement_x0)
    axdividerJobRoleInvolvement_x0.set_anchor((1,.5))
    caxJobRoleInvolvement_x0 = axdividerJobRoleInvolvement_x0.append_axes('bottom', size = '5%', pad = '20%')
    colorbar(axJobRoleInvolvement_x0.get_children()[0], cax = caxJobRoleInvolvement_x0, orientation = 'horizontal',             **{'ticks': (0, 100)})
    caxJobRoleInvolvement_x0.xaxis.set_ticks_position('bottom')
    caxJobRoleInvolvement_x0.set_xlabel('Emp.Count by Role[%]')
    caxJobRoleInvolvement_x0.get_xaxis().set_label_position('bottom')
    
    axJobRoleInvolvement_x0.set_title('JobRole vs JobInvolvement (Attrition = 0)', size = 14,                               **{'horizontalalignment': 'right'})
    
    
    
    #[2,1]--------: Heatmap for the relation between the JobRole & JobInvolvement (Attrition = 1)
    axJobRoleInvolvement_x1 = plt.Subplot(figPerfSatisfTran, gsPerfSatisfTran[2, 1])
    axJobRoleInvolvement_x1.set_anchor((0,.5))
    figPerfSatisfTran.add_subplot(axJobRoleInvolvement_x1)

    df_JobRoleInvolvementl1 = pd.DataFrame(np.zeros(shape = (len(JobRole), len(JobInvolvement))))
    row = 0
    for JR in JobRole:
        col = 0
        for JI in JobInvolvement:
            df_JobRoleInvolvementl1.iloc[row, col] = (df_in_1.loc[(df_in_1['JobRole'] == JR) &                                                            (df_in_1['JobInvolvement'] == JI)]                                                                         ['EmployeeNumber'].count()) /             (df_in_1.loc[(df_in_1['JobRole'] == JR)]['EmployeeNumber'].count()) * 100
            
            col = col + 1
        row = row + 1

    sns.heatmap(df_JobRoleInvolvementl1, cbar = False, annot = True, fmt = '.0f', cmap = 'YlGnBu',                vmin = 0, vmax = 100,                linewidths = .5, ax = axJobRoleInvolvement_x1)
    
    axJobRoleInvolvement_x1.set_aspect('equal')    
    axJobRoleInvolvement_x1.get_yaxis().set_label_position('right')
    axJobRoleInvolvement_x1.get_yaxis().tick_right()
    axJobRoleInvolvement_x1.invert_yaxis()
    axJobRoleInvolvement_x1.set_ylabel('JobRole')
    axJobRoleInvolvement_x1.set_yticklabels(JobRole, **{'rotation': 0})  
    axJobRoleInvolvement_x1.set_xlabel('JobInvolvement')
    axJobRoleInvolvement_x1.set_xticklabels(JobInvolvement, **{'rotation': 0})  
    

    axdividerJobRoleInvolvement_x1 = make_axes_locatable(axJobRoleInvolvement_x1)
    axdividerJobRoleInvolvement_x1.set_anchor((0,.5))
    caxJobRoleInvolvement_x1 = axdividerJobRoleInvolvement_x1.append_axes('bottom', size = '5%', pad = '20%')
    colorbar(axJobRoleInvolvement_x1.get_children()[0], cax = caxJobRoleInvolvement_x1, orientation = 'horizontal',             **{'ticks': (0, 100)})
    caxJobRoleInvolvement_x1.xaxis.set_ticks_position('bottom')
    caxJobRoleInvolvement_x1.set_xlabel('Emp.Count by Role[%]')
    caxJobRoleInvolvement_x1.get_xaxis().set_label_position('bottom')
    
    axJobRoleInvolvement_x1.set_title('JobRole vs JobInvolvement (Attrition = 1)', size = 14,                                   **{'horizontalalignment': 'left'})
    

    
    #[3,0]--------: Relation between YearsAtCompany & JobInvolvement (Attrition = 0)
    axYearsAtCompanyInvolvement_x0 = plt.Subplot(figPerfSatisfTran, gsPerfSatisfTran[3, 0])
    axYearsAtCompanyInvolvement_x0.set_anchor((0,.5))
    figPerfSatisfTran.add_subplot(axYearsAtCompanyInvolvement_x0)

    sns.boxplot(x = 'JobInvolvement', y = 'YearsAtCompany', data = df_in_0, palette = palette_relscale,                boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                medianprops = medianprops, flierprops = flierprops, whis = whis,                ax = axYearsAtCompanyInvolvement_x0).set_title('Y.AtComp. vs JobInvolvement (Attrition = 0)', size = 14)

    pos = range(len(JobInvolvement))
    for tick, label in zip(pos, axYearsAtCompanyInvolvement_x0.get_xticklabels()):
        df_aux = df_in_0.loc[(df_in_0['JobInvolvement'] == int(label.get_text()))]

        stats = f_stats_boxplot(df_aux['YearsAtCompany'])
        axYearsAtCompanyInvolvement_x0.text(pos[tick] - .15, stats[1] - 2.5, '{:.0f}'.format(stats[1]),                                       color = 'red', ha = 'center')
        axYearsAtCompanyInvolvement_x0.text(pos[tick] + .2, stats[2] - 2.5, '{:.0f}'.format(stats[2]),                                       color = 'black', ha = 'center')
        axYearsAtCompanyInvolvement_x0.text(pos[tick], stats[3], '{:.0f}'.format(stats[3]),                                       color = 'black', ha = 'center')
        axYearsAtCompanyInvolvement_x0.text(pos[tick] + .2, stats[4] + 1, '{:.0f}'.format(stats[4]),                                       color = 'black', ha = 'center')
        axYearsAtCompanyInvolvement_x0.text(pos[tick] - .15, stats[5] + 1, '{:.0f}'.format(stats[5]),                                       color = 'red', ha = 'center')    

    axYearsAtCompanyInvolvement_x0.set_xlabel('JobInvolvement')
    axYearsAtCompanyInvolvement_x0.set_ylabel('YearsAtCompany')
    axYearsAtCompanyInvolvement_x0.set_ylim((-3,42))
    axYearsAtCompanyInvolvement_x0.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())
    
    
    
    #[3,1]--------: Relation between YearsAtCompany & JobInvolvement (Attrition = 1)
    axYearsAtCompanyInvolvement_x1 = plt.Subplot(figPerfSatisfTran, gsPerfSatisfTran[3, 1])
    axYearsAtCompanyInvolvement_x1.set_anchor((0,.5))
    figPerfSatisfTran.add_subplot(axYearsAtCompanyInvolvement_x1)
    axYearsAtCompanyInvolvement_x1.spines['left'].set_visible(False)
    axYearsAtCompanyInvolvement_x1.spines['right'].set_visible(True)

    sns.boxplot(x = 'JobInvolvement', y = 'YearsAtCompany', data = df_in_1, palette = palette_relscale,                boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                medianprops = medianprops, flierprops = flierprops, whis = whis,                ax = axYearsAtCompanyInvolvement_x1).set_title('Y.AtComp. vs JobInvolvement (Attrition = 1)', size = 14)

    pos = range(len(JobInvolvement))
    for tick, label in zip(pos, axYearsAtCompanyInvolvement_x1.get_xticklabels()):
        df_aux = df_in_1.loc[(df_in_1['JobInvolvement'] == int(label.get_text()))]

        stats = f_stats_boxplot(df_aux['YearsAtCompany'])
        axYearsAtCompanyInvolvement_x1.text(pos[tick] - .15, stats[1] - 2.5, '{:.0f}'.format(stats[1]),                                       color = 'red', ha = 'center')
        axYearsAtCompanyInvolvement_x1.text(pos[tick] + .2, stats[2] - 2.5, '{:.0f}'.format(stats[2]),                                       color = 'black', ha = 'center')
        axYearsAtCompanyInvolvement_x1.text(pos[tick], stats[3], '{:.0f}'.format(stats[3]),                                       color = 'black', ha = 'center')
        axYearsAtCompanyInvolvement_x1.text(pos[tick] + .2, stats[4] + 1, '{:.0f}'.format(stats[4]),                                       color = 'black', ha = 'center')
        axYearsAtCompanyInvolvement_x1.text(pos[tick] - .15, stats[5] + 1, '{:.0f}'.format(stats[5]),                                       color = 'red', ha = 'center')    

    axYearsAtCompanyInvolvement_x1.set_xlabel('JobInvolvement')
    axYearsAtCompanyInvolvement_x1.set_ylabel('YearsAtCompany')
    axYearsAtCompanyInvolvement_x1.set_ylim((-3,42))
    axYearsAtCompanyInvolvement_x1.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())    
    axYearsAtCompanyInvolvement_x1.get_yaxis().set_label_position("right")
    axYearsAtCompanyInvolvement_x1.get_yaxis().tick_right()  

    
    
    #[4,0]--------: Heatmap for the relation between the JobRole & JobSatisfaction (Attrition = 0)
    axJobRoleSatisfaction_x0 = plt.Subplot(figPerfSatisfTran, gsPerfSatisfTran[4, 0])
    axJobRoleSatisfaction_x0.set_anchor((1,.5))
    figPerfSatisfTran.add_subplot(axJobRoleSatisfaction_x0)

    df_JobRoleSatisfactionl0 = pd.DataFrame(np.zeros(shape = (len(JobRole), len(JobSatisfaction))))
    row = 0
    for JR in JobRole:
        col = 0
        for JS in JobSatisfaction:
            df_JobRoleSatisfactionl0.iloc[row, col] = (df_in_0.loc[(df_in_0['JobRole'] == JR) &                                                            (df_in_0['JobSatisfaction'] == JS)]                                                                         ['EmployeeNumber'].count()) /             (df_in_0.loc[(df_in_0['JobRole'] == JR)]['EmployeeNumber'].count())*100
            
            col = col + 1
        row = row + 1

    sns.heatmap(df_JobRoleSatisfactionl0, cbar = False, annot = True, fmt = '.0f', cmap = 'YlGnBu',                vmin = 0, vmax = 100,                linewidths = .5, ax = axJobRoleSatisfaction_x0)
    
    axJobRoleSatisfaction_x0.set_aspect('equal')    
    axJobRoleSatisfaction_x0.get_yaxis().set_label_position('left')
    axJobRoleSatisfaction_x0.get_yaxis().tick_left()
    axJobRoleSatisfaction_x0.invert_yaxis()
    axJobRoleSatisfaction_x0.set_ylabel('JobRole')
    axJobRoleSatisfaction_x0.set_yticklabels(JobRole, **{'rotation': 0})  
    axJobRoleSatisfaction_x0.set_xlabel('JobSatisfaction')
    axJobRoleSatisfaction_x0.set_xticklabels(JobSatisfaction, **{'rotation': 0})  
    
    axdividerJobRoleSatisfaction_x0 = make_axes_locatable(axJobRoleSatisfaction_x0)
    axdividerJobRoleSatisfaction_x0.set_anchor((1,.5))
    caxJobRoleSatisfaction_x0 = axdividerJobRoleSatisfaction_x0.append_axes('bottom', size = '5%', pad = '20%')
    colorbar(axJobRoleSatisfaction_x0.get_children()[0], cax = caxJobRoleSatisfaction_x0, orientation = 'horizontal',             **{'ticks': (0, 100)})
    caxJobRoleSatisfaction_x0.xaxis.set_ticks_position('bottom')
    caxJobRoleSatisfaction_x0.set_xlabel('Emp.Count by Role[%]')
    caxJobRoleSatisfaction_x0.get_xaxis().set_label_position('bottom')
    
    axJobRoleSatisfaction_x0.set_title('JobRole vs JobSatisfaction (Attrition = 0)', size = 14,                               **{'horizontalalignment': 'right'})
    
    
    
    #[4,1]--------: Heatmap for the relation between the JobRole & JobSatisfaction (Attrition = 1)
    axJobRoleSatisfaction_x1 = plt.Subplot(figPerfSatisfTran, gsPerfSatisfTran[4, 1])
    axJobRoleSatisfaction_x1.set_anchor((0,.5))
    figPerfSatisfTran.add_subplot(axJobRoleSatisfaction_x1)
    axJobRoleSatisfaction_x1.spines['left'].set_visible(False)
    axJobRoleSatisfaction_x1.spines['right'].set_visible(True)

    df_JobRoleSatisfactionl1 = pd.DataFrame(np.zeros(shape = (len(JobRole), len(JobSatisfaction))))
    row = 0
    for JR in JobRole:
        col = 0
        for JS in JobSatisfaction:
            df_JobRoleSatisfactionl1.iloc[row, col] = (df_in_1.loc[(df_in_1['JobRole'] == JR) &                                                        (df_in_1['JobSatisfaction'] == JS)]                                                       ['EmployeeNumber'].count()) /             (df_in_1.loc[(df_in_1['JobRole'] == JR)]['EmployeeNumber'].count())*100

            col = col + 1
        row = row + 1

    sns.heatmap(df_JobRoleSatisfactionl1, cbar = False, annot = True, fmt = '.0f', cmap = 'YlGnBu',                vmin = 0, vmax = 100,                linewidths = .5, ax = axJobRoleSatisfaction_x1)
    
    axJobRoleSatisfaction_x1.set_aspect('equal')    
    axJobRoleSatisfaction_x1.get_yaxis().set_label_position('right')
    axJobRoleSatisfaction_x1.get_yaxis().tick_right()
    axJobRoleSatisfaction_x1.invert_yaxis()
    axJobRoleSatisfaction_x1.set_ylabel('JobRole')
    axJobRoleSatisfaction_x1.set_yticklabels(JobRole, **{'rotation': 0})  
    axJobRoleSatisfaction_x1.set_xlabel('JobSatisfaction')
    axJobRoleSatisfaction_x1.set_xticklabels(JobSatisfaction, **{'rotation': 0})  
    

    axdividerJobRoleSatisfaction_x1 = make_axes_locatable(axJobRoleSatisfaction_x1)
    axdividerJobRoleSatisfaction_x1.set_anchor((0,.5))
    caxJobRoleSatisfaction_x1 = axdividerJobRoleSatisfaction_x1.append_axes('bottom', size = '5%', pad = '20%')
    colorbar(axJobRoleSatisfaction_x1.get_children()[0], cax = caxJobRoleSatisfaction_x1, orientation = 'horizontal',             **{'ticks': (0, 100)})
    caxJobRoleSatisfaction_x1.xaxis.set_ticks_position('bottom')
    caxJobRoleSatisfaction_x1.set_xlabel('Emp.Count by Role[%]')
    caxJobRoleSatisfaction_x1.get_xaxis().set_label_position('bottom')
    
    axJobRoleSatisfaction_x1.set_title('JobRole vs JobSatisfaction (Attrition = 1)', size = 14,                                   **{'horizontalalignment': 'left'})

    
    
    #[5,0]--------: Relation between YearsAtCompany & JobSatisfaction (Attrition = 0)
    axYearsAtCompanySatisfaction_x0 = plt.Subplot(figPerfSatisfTran, gsPerfSatisfTran[5, 0])
    axYearsAtCompanySatisfaction_x0.set_anchor((0,.5))
    figPerfSatisfTran.add_subplot(axYearsAtCompanySatisfaction_x0)

    sns.boxplot(x = 'JobSatisfaction', y = 'YearsAtCompany', data = df_in_0, palette = palette_relscale,                boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                medianprops = medianprops, flierprops = flierprops, whis = whis,                ax = axYearsAtCompanySatisfaction_x0).set_title('Y.AtComp. vs JobSatisfaction (Attrition = 0)', size = 14)

    pos = range(len(JobSatisfaction))
    for tick, label in zip(pos, axYearsAtCompanySatisfaction_x0.get_xticklabels()):
        df_aux = df_in_0.loc[(df_in_0['JobSatisfaction'] == int(label.get_text()))]

        stats = f_stats_boxplot(df_aux['YearsAtCompany'])
        axYearsAtCompanySatisfaction_x0.text(pos[tick] - .15, stats[1] - 2.5, '{:.0f}'.format(stats[1]),                                       color = 'red', ha = 'center')
        axYearsAtCompanySatisfaction_x0.text(pos[tick] + .2, stats[2] - 2.5, '{:.0f}'.format(stats[2]),                                       color = 'black', ha = 'center')
        axYearsAtCompanySatisfaction_x0.text(pos[tick], stats[3], '{:.0f}'.format(stats[3]),                                       color = 'black', ha = 'center')
        axYearsAtCompanySatisfaction_x0.text(pos[tick] + .2, stats[4] + 1, '{:.0f}'.format(stats[4]),                                       color = 'black', ha = 'center')
        axYearsAtCompanySatisfaction_x0.text(pos[tick] - .15, stats[5] + 1, '{:.0f}'.format(stats[5]),                                       color = 'red', ha = 'center')    

    axYearsAtCompanySatisfaction_x0.set_xlabel('JobSatisfaction')
    axYearsAtCompanySatisfaction_x0.set_ylabel('YearsAtCompany')
    axYearsAtCompanySatisfaction_x0.set_ylim((-3,42))
    axYearsAtCompanySatisfaction_x0.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())
    
    
    
    #[5,1]--------: Relation between YearsAtCompany & JobSatisfaction (Attrition = 1)
    axYearsAtCompanySatisfaction_x1 = plt.Subplot(figPerfSatisfTran, gsPerfSatisfTran[5, 1])
    axYearsAtCompanySatisfaction_x1.set_anchor((0,.5))
    figPerfSatisfTran.add_subplot(axYearsAtCompanySatisfaction_x1)
    axYearsAtCompanySatisfaction_x1.spines['left'].set_visible(False)
    axYearsAtCompanySatisfaction_x1.spines['right'].set_visible(True)

    sns.boxplot(x = 'JobSatisfaction', y = 'YearsAtCompany', data = df_in_1, palette = palette_relscale,                boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                medianprops = medianprops, flierprops = flierprops, whis = whis,                ax = axYearsAtCompanySatisfaction_x1).set_title('Y.AtComp. vs JobSatisfaction (Attrition = 1)', size = 14)

    pos = range(len(JobSatisfaction))
    for tick, label in zip(pos, axYearsAtCompanySatisfaction_x1.get_xticklabels()):
        df_aux = df_in_1.loc[(df_in_1['JobSatisfaction'] == int(label.get_text()))]

        stats = f_stats_boxplot(df_aux['YearsAtCompany'])
        axYearsAtCompanySatisfaction_x1.text(pos[tick] - .15, stats[1] - 2.5, '{:.0f}'.format(stats[1]),                                       color = 'red', ha = 'center')
        axYearsAtCompanySatisfaction_x1.text(pos[tick] + .2, stats[2] - 2.5, '{:.0f}'.format(stats[2]),                                       color = 'black', ha = 'center')
        axYearsAtCompanySatisfaction_x1.text(pos[tick], stats[3], '{:.0f}'.format(stats[3]),                                       color = 'black', ha = 'center')
        axYearsAtCompanySatisfaction_x1.text(pos[tick] + .2, stats[4] + 1, '{:.0f}'.format(stats[4]),                                       color = 'black', ha = 'center')
        axYearsAtCompanySatisfaction_x1.text(pos[tick] - .15, stats[5] + 1, '{:.0f}'.format(stats[5]),                                       color = 'red', ha = 'center')    

    axYearsAtCompanySatisfaction_x1.set_xlabel('JobSatisfaction')
    axYearsAtCompanySatisfaction_x1.set_ylabel('YearsAtCompany')
    axYearsAtCompanySatisfaction_x1.set_ylim((-3,42))
    axYearsAtCompanySatisfaction_x1.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())    
    axYearsAtCompanySatisfaction_x1.get_yaxis().set_label_position("right")
    axYearsAtCompanySatisfaction_x1.get_yaxis().tick_right() 

    
    
    #[6,0]--------: Heatmap for the relation between the JobRole & EnvironmentSatisfaction (Attrition = 0)
    axJobRoleEnvSatisfaction_x0 = plt.Subplot(figPerfSatisfTran, gsPerfSatisfTran[6, 0])
    axJobRoleEnvSatisfaction_x0.set_anchor((1,.5))
    figPerfSatisfTran.add_subplot(axJobRoleEnvSatisfaction_x0)

    df_JobRoleEnvSatisfactionl0 = pd.DataFrame(np.zeros(shape = (len(JobRole), len(EnvironmentSatisfaction))))
    row = 0
    for JR in JobRole:
        col = 0
        for ES in EnvironmentSatisfaction:
            df_JobRoleEnvSatisfactionl0.iloc[row, col] = (df_in_0.loc[(df_in_0['JobRole'] == JR) &                                                            (df_in_0['EnvironmentSatisfaction'] == ES)]                                                                         ['EmployeeNumber'].count()) /             (df_in_0.loc[(df_in_0['JobRole'] == JR)]['EmployeeNumber'].count())*100
            
            col = col + 1
        row = row + 1

    sns.heatmap(df_JobRoleEnvSatisfactionl0, cbar = False, annot = True, fmt = '.0f', cmap = 'YlGnBu',                vmin = 0, vmax = 100,                linewidths = .5, ax = axJobRoleEnvSatisfaction_x0)
    
    axJobRoleEnvSatisfaction_x0.set_aspect('equal')    
    axJobRoleEnvSatisfaction_x0.get_yaxis().set_label_position('left')
    axJobRoleEnvSatisfaction_x0.get_yaxis().tick_left()
    axJobRoleEnvSatisfaction_x0.invert_yaxis()
    axJobRoleEnvSatisfaction_x0.set_ylabel('JobRole')
    axJobRoleEnvSatisfaction_x0.set_yticklabels(JobRole, **{'rotation': 0})  
    axJobRoleEnvSatisfaction_x0.set_xlabel('EnvironmentSatisfaction')
    axJobRoleEnvSatisfaction_x0.set_xticklabels(EnvironmentSatisfaction, **{'rotation': 0})  
    
    axdividerJobRoleEnvSatisfaction_x0 = make_axes_locatable(axJobRoleEnvSatisfaction_x0)
    axdividerJobRoleEnvSatisfaction_x0.set_anchor((1,.5))
    caxJobRoleEnvSatisfaction_x0 = axdividerJobRoleEnvSatisfaction_x0.append_axes('bottom', size = '5%', pad = '20%')
    colorbar(axJobRoleEnvSatisfaction_x0.get_children()[0], cax = caxJobRoleEnvSatisfaction_x0,             orientation = 'horizontal', **{'ticks': (0, 100)})
    caxJobRoleEnvSatisfaction_x0.xaxis.set_ticks_position('bottom')
    caxJobRoleEnvSatisfaction_x0.set_xlabel('Emp.Count by Role[%]')
    caxJobRoleEnvSatisfaction_x0.get_xaxis().set_label_position('bottom')
    
    axJobRoleEnvSatisfaction_x0.set_title('JobRole vs EnvironmentSatisfaction (Attrition = 0)', size = 14,                               **{'horizontalalignment': 'right'})
    
    
    
    #[6,1]--------: Heatmap for the relation between the JobRole & EnvironmentSatisfaction (Attrition = 1)
    axJobRoleEnvSatisfaction_x1 = plt.Subplot(figPerfSatisfTran, gsPerfSatisfTran[6, 1])
    axJobRoleEnvSatisfaction_x1.set_anchor((0,.5))
    figPerfSatisfTran.add_subplot(axJobRoleEnvSatisfaction_x1)

    df_JobRoleEnvSatisfactionl1 = pd.DataFrame(np.zeros(shape = (len(JobRole), len(EnvironmentSatisfaction))))
    row = 0
    for JR in JobRole:
        col = 0
        for ES in EnvironmentSatisfaction:
            df_JobRoleEnvSatisfactionl1.iloc[row, col] = (df_in_1.loc[(df_in_1['JobRole'] == JR) &                                                        (df_in_1['EnvironmentSatisfaction'] == ES)]                                                       ['EmployeeNumber'].count()) /             (df_in_1.loc[(df_in_1['JobRole'] == JR)]['EmployeeNumber'].count())*100

            col = col + 1
        row = row + 1

    sns.heatmap(df_JobRoleEnvSatisfactionl1, cbar = False, annot = True, fmt = '.0f', cmap = 'YlGnBu',                vmin = 0, vmax = 100,                linewidths = .5, ax = axJobRoleEnvSatisfaction_x1)
    
    axJobRoleEnvSatisfaction_x1.set_aspect('equal')    
    axJobRoleEnvSatisfaction_x1.get_yaxis().set_label_position('right')
    axJobRoleEnvSatisfaction_x1.get_yaxis().tick_right()
    axJobRoleEnvSatisfaction_x1.invert_yaxis()
    axJobRoleEnvSatisfaction_x1.set_ylabel('JobRole')
    axJobRoleEnvSatisfaction_x1.set_yticklabels(JobRole, **{'rotation': 0})  
    axJobRoleEnvSatisfaction_x1.set_xlabel('EnvironmentSatisfaction')
    axJobRoleEnvSatisfaction_x1.set_xticklabels(EnvironmentSatisfaction, **{'rotation': 0})  
    

    axdividerJobRoleEnvSatisfaction_x1 = make_axes_locatable(axJobRoleEnvSatisfaction_x1)
    axdividerJobRoleEnvSatisfaction_x1.set_anchor((0,.5))
    caxJobRoleEnvSatisfaction_x1 = axdividerJobRoleEnvSatisfaction_x1.append_axes('bottom', size = '5%', pad = '20%')
    colorbar(axJobRoleEnvSatisfaction_x1.get_children()[0], cax = caxJobRoleEnvSatisfaction_x1,             orientation = 'horizontal', **{'ticks': (0, 100)})
    caxJobRoleEnvSatisfaction_x1.xaxis.set_ticks_position('bottom')
    caxJobRoleEnvSatisfaction_x1.set_xlabel('Emp.Count by Role[%]')
    caxJobRoleEnvSatisfaction_x1.get_xaxis().set_label_position('bottom')
    
    axJobRoleEnvSatisfaction_x1.set_title('JobRole vs EnvironmentSatisfaction (Attrition = 1)', size = 14,                                   **{'horizontalalignment': 'left'})

    
    
    #[7,0]--------: Relation between YearsAtCompany & EnvironmentSatisfaction (Attrition = 0)
    axYearsAtCompanyEnvSatisfaction_x0 = plt.Subplot(figPerfSatisfTran, gsPerfSatisfTran[7, 0])
    axYearsAtCompanyEnvSatisfaction_x0.set_anchor((0,.5))
    figPerfSatisfTran.add_subplot(axYearsAtCompanyEnvSatisfaction_x0)

    sns.boxplot(x = 'EnvironmentSatisfaction', y = 'YearsAtCompany', data = df_in_0, palette = palette_relscale,                boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                medianprops = medianprops, flierprops = flierprops, whis = whis,                ax = axYearsAtCompanyEnvSatisfaction_x0).set_title('Y.AtComp. vs Env.Satisfaction (Attrition = 0)', size = 14)

    pos = range(len(EnvironmentSatisfaction))
    for tick, label in zip(pos, axYearsAtCompanyEnvSatisfaction_x0.get_xticklabels()):
        df_aux = df_in_0.loc[(df_in_0['EnvironmentSatisfaction'] == int(label.get_text()))]

        stats = f_stats_boxplot(df_aux['YearsAtCompany'])
        axYearsAtCompanyEnvSatisfaction_x0.text(pos[tick] - .15, stats[1] - 2.5, '{:.0f}'.format(stats[1]),                                       color = 'red', ha = 'center')
        axYearsAtCompanyEnvSatisfaction_x0.text(pos[tick] + .2, stats[2] - 2.5, '{:.0f}'.format(stats[2]),                                       color = 'black', ha = 'center')
        axYearsAtCompanyEnvSatisfaction_x0.text(pos[tick], stats[3], '{:.0f}'.format(stats[3]),                                       color = 'black', ha = 'center')
        axYearsAtCompanyEnvSatisfaction_x0.text(pos[tick] + .2, stats[4] + 1, '{:.0f}'.format(stats[4]),                                       color = 'black', ha = 'center')
        axYearsAtCompanyEnvSatisfaction_x0.text(pos[tick] - .15, stats[5] + 1, '{:.0f}'.format(stats[5]),                                       color = 'red', ha = 'center')    

    axYearsAtCompanyEnvSatisfaction_x0.set_xlabel('EnvironmentSatisfaction')
    axYearsAtCompanyEnvSatisfaction_x0.set_ylabel('YearsAtCompany')
    axYearsAtCompanyEnvSatisfaction_x0.set_ylim((-3,42))
    axYearsAtCompanyEnvSatisfaction_x0.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())
    
    
    
    #[7,1]--------: Relation between YearsAtCompany & EnvironmentSatisfaction (Attrition = 1)
    axYearsAtCompanyEnvSatisfaction_x1 = plt.Subplot(figPerfSatisfTran, gsPerfSatisfTran[7, 1])
    axYearsAtCompanyEnvSatisfaction_x1.set_anchor((0,.5))
    figPerfSatisfTran.add_subplot(axYearsAtCompanyEnvSatisfaction_x1)
    axYearsAtCompanyEnvSatisfaction_x1.spines['left'].set_visible(False)
    axYearsAtCompanyEnvSatisfaction_x1.spines['right'].set_visible(True)

    sns.boxplot(x = 'EnvironmentSatisfaction', y = 'YearsAtCompany', data = df_in_1, palette = palette_relscale,                boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                medianprops = medianprops, flierprops = flierprops, whis = whis,                ax = axYearsAtCompanyEnvSatisfaction_x1).set_title('Y.AtComp. vs Env.Satisfaction (Attrition = 1)', size = 14)

    pos = range(len(EnvironmentSatisfaction))
    for tick, label in zip(pos, axYearsAtCompanyEnvSatisfaction_x1.get_xticklabels()):
        df_aux = df_in_1.loc[(df_in_1['EnvironmentSatisfaction'] == int(label.get_text()))]

        stats = f_stats_boxplot(df_aux['YearsAtCompany'])
        axYearsAtCompanyEnvSatisfaction_x1.text(pos[tick] - .15, stats[1] - 2.5, '{:.0f}'.format(stats[1]),                                       color = 'red', ha = 'center')
        axYearsAtCompanyEnvSatisfaction_x1.text(pos[tick] + .2, stats[2] - 2.5, '{:.0f}'.format(stats[2]),                                       color = 'black', ha = 'center')
        axYearsAtCompanyEnvSatisfaction_x1.text(pos[tick], stats[3], '{:.0f}'.format(stats[3]),                                       color = 'black', ha = 'center')
        axYearsAtCompanyEnvSatisfaction_x1.text(pos[tick] + .2, stats[4] + 1, '{:.0f}'.format(stats[4]),                                       color = 'black', ha = 'center')
        axYearsAtCompanyEnvSatisfaction_x1.text(pos[tick] - .15, stats[5] + 1, '{:.0f}'.format(stats[5]),                                       color = 'red', ha = 'center')    

    axYearsAtCompanyEnvSatisfaction_x1.set_xlabel('EnvironmentSatisfaction')
    axYearsAtCompanyEnvSatisfaction_x1.set_ylabel('YearsAtCompany')
    axYearsAtCompanyEnvSatisfaction_x1.set_ylim((-3,42))
    axYearsAtCompanyEnvSatisfaction_x1.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())    
    axYearsAtCompanyEnvSatisfaction_x1.get_yaxis().set_label_position("right")
    axYearsAtCompanyEnvSatisfaction_x1.get_yaxis().tick_right() 

    
    
    #[8,0]--------: Heatmap for the relation between the JobRole & RelationshipSatisfaction (Attrition = 0)
    axJobRoleRelSatisfaction_x0 = plt.Subplot(figPerfSatisfTran, gsPerfSatisfTran[8, 0])
    axJobRoleRelSatisfaction_x0.set_anchor((1,.5))
    figPerfSatisfTran.add_subplot(axJobRoleRelSatisfaction_x0)

    df_JobRoleRelSatisfactionl0 = pd.DataFrame(np.zeros(shape = (len(JobRole), len(RelationshipSatisfaction))))
    row = 0
    for JR in JobRole:
        col = 0
        for RS in RelationshipSatisfaction:
            df_JobRoleRelSatisfactionl0.iloc[row, col] = (df_in_0.loc[(df_in_0['JobRole'] == JR) &                                                            (df_in_0['RelationshipSatisfaction'] == RS)]                                                                         ['EmployeeNumber'].count()) /             (df_in_0.loc[(df_in_0['JobRole'] == JR)]['EmployeeNumber'].count())*100
            
            col = col + 1
        row = row + 1

    sns.heatmap(df_JobRoleRelSatisfactionl0, cbar = False, annot = True, fmt = '.0f', cmap = 'YlGnBu',                vmin = 0, vmax = 100,                linewidths = .5, ax = axJobRoleRelSatisfaction_x0)
    
    axJobRoleRelSatisfaction_x0.set_aspect('equal')    
    axJobRoleRelSatisfaction_x0.get_yaxis().set_label_position('left')
    axJobRoleRelSatisfaction_x0.get_yaxis().tick_left()
    axJobRoleRelSatisfaction_x0.invert_yaxis()
    axJobRoleRelSatisfaction_x0.set_ylabel('JobRole')
    axJobRoleRelSatisfaction_x0.set_yticklabels(JobRole, **{'rotation': 0})  
    axJobRoleRelSatisfaction_x0.set_xlabel('RelationshipSatisfaction')
    axJobRoleRelSatisfaction_x0.set_xticklabels(RelationshipSatisfaction, **{'rotation': 0})  
    
    axdividerJobRoleRelSatisfaction_x0 = make_axes_locatable(axJobRoleRelSatisfaction_x0)
    axdividerJobRoleRelSatisfaction_x0.set_anchor((1,.5))
    caxJobRoleRelSatisfaction_x0 = axdividerJobRoleRelSatisfaction_x0.append_axes('bottom', size = '5%', pad = '20%')
    colorbar(axJobRoleRelSatisfaction_x0.get_children()[0], cax = caxJobRoleRelSatisfaction_x0,             orientation = 'horizontal', **{'ticks': (0, 100)})
    caxJobRoleRelSatisfaction_x0.xaxis.set_ticks_position('bottom')
    caxJobRoleRelSatisfaction_x0.set_xlabel('Emp.Count by Role[%]')
    caxJobRoleRelSatisfaction_x0.get_xaxis().set_label_position('bottom')
    
    axJobRoleRelSatisfaction_x0.set_title('JobRole vs RelationshipSatisfaction (Attrition = 0)', size = 14,                               **{'horizontalalignment': 'right'})
    
    
    
    #[8,1]--------: Heatmap for the relation between the JobRole & RelationshipSatisfaction (Attrition = 1)
    axJobRoleRelSatisfaction_x1 = plt.Subplot(figPerfSatisfTran, gsPerfSatisfTran[8, 1])
    axJobRoleRelSatisfaction_x1.set_anchor((0,.5))
    figPerfSatisfTran.add_subplot(axJobRoleRelSatisfaction_x1)

    df_JobRoleRelSatisfactionl1 = pd.DataFrame(np.zeros(shape = (len(JobRole), len(RelationshipSatisfaction))))
    row = 0
    for JR in JobRole:
        col = 0
        for RS in RelationshipSatisfaction:
            df_JobRoleRelSatisfactionl1.iloc[row, col] = (df_in_1.loc[(df_in_1['JobRole'] == JR) &                                                        (df_in_1['RelationshipSatisfaction'] == RS)]                                                       ['EmployeeNumber'].count()) /             (df_in_1.loc[(df_in_1['JobRole'] == JR)]['EmployeeNumber'].count())*100

            col = col + 1
        row = row + 1

    sns.heatmap(df_JobRoleRelSatisfactionl1, cbar = False, annot = True, fmt = '.0f', cmap = 'YlGnBu',                vmin = 0, vmax = 100,                linewidths = .5, ax = axJobRoleRelSatisfaction_x1)
    
    axJobRoleRelSatisfaction_x1.set_aspect('equal')    
    axJobRoleRelSatisfaction_x1.get_yaxis().set_label_position('right')
    axJobRoleRelSatisfaction_x1.get_yaxis().tick_right()
    axJobRoleRelSatisfaction_x1.invert_yaxis()
    axJobRoleRelSatisfaction_x1.set_ylabel('JobRole')
    axJobRoleRelSatisfaction_x1.set_yticklabels(JobRole, **{'rotation': 0})  
    axJobRoleRelSatisfaction_x1.set_xlabel('RelationshipSatisfaction')
    axJobRoleRelSatisfaction_x1.set_xticklabels(RelationshipSatisfaction, **{'rotation': 0})  
    

    axdividerJobRoleRelSatisfaction_x1 = make_axes_locatable(axJobRoleRelSatisfaction_x1)
    axdividerJobRoleRelSatisfaction_x1.set_anchor((0,.5))
    caxJobRoleRelSatisfaction_x1 = axdividerJobRoleRelSatisfaction_x1.append_axes('bottom', size = '5%', pad = '20%')
    colorbar(axJobRoleRelSatisfaction_x1.get_children()[0], cax = caxJobRoleRelSatisfaction_x1,             orientation = 'horizontal', **{'ticks': (0, 100)})
    caxJobRoleRelSatisfaction_x1.xaxis.set_ticks_position('bottom')
    caxJobRoleRelSatisfaction_x1.set_xlabel('Emp.Count by Role[%]')
    caxJobRoleRelSatisfaction_x1.get_xaxis().set_label_position('bottom')
    
    axJobRoleRelSatisfaction_x1.set_title('JobRole vs RelationshipSatisfaction (Attrition = 1)', size = 14,                                   **{'horizontalalignment': 'left'})

    
    
    #[9,0]--------: Relation between YearsAtCompany & RelationshipSatisfaction (Attrition = 0)
    axYearsAtCompanyRelSatisfaction_x0 = plt.Subplot(figPerfSatisfTran, gsPerfSatisfTran[9, 0])
    axYearsAtCompanyRelSatisfaction_x0.set_anchor((0,.5))
    figPerfSatisfTran.add_subplot(axYearsAtCompanyRelSatisfaction_x0)

    sns.boxplot(x = 'RelationshipSatisfaction', y = 'YearsAtCompany', data = df_in_0, palette = palette_relscale,                boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                medianprops = medianprops, flierprops = flierprops, whis = whis,                ax = axYearsAtCompanyRelSatisfaction_x0).set_title('Y.AtComp. vs Rel.Satisfaction (Attrition = 0)', size = 14)

    pos = range(len(RelationshipSatisfaction))
    for tick, label in zip(pos, axYearsAtCompanyRelSatisfaction_x0.get_xticklabels()):
        df_aux = df_in_0.loc[(df_in_0['RelationshipSatisfaction'] == int(label.get_text()))]

        stats = f_stats_boxplot(df_aux['YearsAtCompany'])
        axYearsAtCompanyRelSatisfaction_x0.text(pos[tick] - .15, stats[1] - 2.5, '{:.0f}'.format(stats[1]),                                       color = 'red', ha = 'center')
        axYearsAtCompanyRelSatisfaction_x0.text(pos[tick] + .2, stats[2] - 2.5, '{:.0f}'.format(stats[2]),                                     color = 'black', ha = 'center')
        axYearsAtCompanyRelSatisfaction_x0.text(pos[tick], stats[3], '{:.0f}'.format(stats[3]),                                       color = 'black', ha = 'center')
        axYearsAtCompanyRelSatisfaction_x0.text(pos[tick] + .2, stats[4] + 1, '{:.0f}'.format(stats[4]),                                       color = 'black', ha = 'center')
        axYearsAtCompanyRelSatisfaction_x0.text(pos[tick] - .15, stats[5] + 1, '{:.0f}'.format(stats[5]),                                       color = 'red', ha = 'center')    

    axYearsAtCompanyRelSatisfaction_x0.set_xlabel('RelationshipSatisfaction')
    axYearsAtCompanyRelSatisfaction_x0.set_ylabel('YearsAtCompany')
    axYearsAtCompanyRelSatisfaction_x0.set_ylim((-3,42))
    axYearsAtCompanyRelSatisfaction_x0.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())
    
    
    
    #[9,1]--------: Relation between YearsAtCompany & RelationshipSatisfaction (Attrition = 1)
    axYearsAtCompanyRelSatisfaction_x1 = plt.Subplot(figPerfSatisfTran, gsPerfSatisfTran[9, 1])
    axYearsAtCompanyRelSatisfaction_x1.set_anchor((0,.5))
    figPerfSatisfTran.add_subplot(axYearsAtCompanyRelSatisfaction_x1)
    axYearsAtCompanyRelSatisfaction_x1.spines['left'].set_visible(False)
    axYearsAtCompanyRelSatisfaction_x1.spines['right'].set_visible(True)

    sns.boxplot(x = 'RelationshipSatisfaction', y = 'YearsAtCompany', data = df_in_1, palette = palette_relscale,                boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                medianprops = medianprops, flierprops = flierprops, whis = whis,                ax = axYearsAtCompanyRelSatisfaction_x1).set_title('Y.AtComp. vs Rel.Satisfaction (Attrition = 1)', size = 14)

    pos = range(len(RelationshipSatisfaction))
    for tick, label in zip(pos, axYearsAtCompanyRelSatisfaction_x1.get_xticklabels()):
        df_aux = df_in_1.loc[(df_in_1['RelationshipSatisfaction'] == int(label.get_text()))]

        stats = f_stats_boxplot(df_aux['YearsAtCompany'])
        axYearsAtCompanyRelSatisfaction_x1.text(pos[tick] - .15, stats[1] - 2.5, '{:.0f}'.format(stats[1]),                                       color = 'red', ha = 'center')
        axYearsAtCompanyRelSatisfaction_x1.text(pos[tick] + .2, stats[2] - 2.5, '{:.0f}'.format(stats[2]),                                       color = 'black', ha = 'center')
        axYearsAtCompanyRelSatisfaction_x1.text(pos[tick], stats[3], '{:.0f}'.format(stats[3]),                                       color = 'black', ha = 'center')
        axYearsAtCompanyRelSatisfaction_x1.text(pos[tick] + .2, stats[4] + 1, '{:.0f}'.format(stats[4]),                                     color = 'black', ha = 'center')
        axYearsAtCompanyRelSatisfaction_x1.text(pos[tick] - .15, stats[5] + 1, '{:.0f}'.format(stats[5]),                                       color = 'red', ha = 'center')    

    axYearsAtCompanyRelSatisfaction_x1.set_xlabel('RelationshipSatisfaction')
    axYearsAtCompanyRelSatisfaction_x1.set_ylabel('YearsAtCompany')
    axYearsAtCompanyRelSatisfaction_x1.set_ylim((-3,42))
    axYearsAtCompanyRelSatisfaction_x1.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())    
    axYearsAtCompanyRelSatisfaction_x1.get_yaxis().set_label_position("right")
    axYearsAtCompanyRelSatisfaction_x1.get_yaxis().tick_right() 


    
    #[10,0]--------: Heatmap for the relation between the JobRole & TrainingTimesLastYear (Attrition = 0)
    axJobRoleTTLY_x0 = plt.Subplot(figPerfSatisfTran, gsPerfSatisfTran[10, 0])
    axJobRoleTTLY_x0.set_anchor((1,.5))
    figPerfSatisfTran.add_subplot(axJobRoleTTLY_x0)

    df_JobRoleTTLYl0 = pd.DataFrame(np.zeros(shape = (len(JobRole), len(TrainingTimesLastYear))))
    row = 0
    for JR in JobRole:
        col = 0
        for TTLY in TrainingTimesLastYear:
            df_JobRoleTTLYl0.iloc[row, col] = (df_in_0.loc[(df_in_0['JobRole'] == JR) &                                                            (df_in_0['TrainingTimesLastYear'] == TTLY)]                                                                         ['EmployeeNumber'].count()) /             (df_in_0.loc[(df_in_0['JobRole'] == JR)]['EmployeeNumber'].count())*100
            
            col = col + 1
        row = row + 1

    sns.heatmap(df_JobRoleTTLYl0, cbar = False, annot = True, fmt = '.0f', cmap = 'YlGnBu',                vmin = 0, vmax = 100,                linewidths = .5, ax = axJobRoleTTLY_x0)
    
    axJobRoleTTLY_x0.set_aspect('equal')    
    axJobRoleTTLY_x0.get_yaxis().set_label_position('left')
    axJobRoleTTLY_x0.get_yaxis().tick_left()
    axJobRoleTTLY_x0.invert_yaxis()
    axJobRoleTTLY_x0.set_ylabel('JobRole')
    axJobRoleTTLY_x0.set_yticklabels(JobRole, **{'rotation': 0})  
    axJobRoleTTLY_x0.set_xlabel('TrainingTimesLastYear')
    axJobRoleTTLY_x0.set_xticklabels(TrainingTimesLastYear, **{'rotation': 0})  
    
    axdividerJobRoleTTLY_x0 = make_axes_locatable(axJobRoleTTLY_x0)
    axdividerJobRoleTTLY_x0.set_anchor((1,.5))
    caxJobRoleTTLY_x0 = axdividerJobRoleTTLY_x0.append_axes('bottom', size = '5%', pad = '20%')
    colorbar(axJobRoleTTLY_x0.get_children()[0], cax = caxJobRoleTTLY_x0,             orientation = 'horizontal', **{'ticks': (0, 100)})
    caxJobRoleTTLY_x0.xaxis.set_ticks_position('bottom')
    caxJobRoleTTLY_x0.set_xlabel('Emp.Count by Role[%]')
    caxJobRoleTTLY_x0.get_xaxis().set_label_position('bottom')
    
    axJobRoleTTLY_x0.set_title('JobRole vs TrainingTimesLastYear (Attrition = 0)', size = 14,                               **{'horizontalalignment': 'right'})
    
    
    
    #[10,1]--------: Heatmap for the relation between the JobRole & TrainingTimesLastYear (Attrition = 1)
    axJobRoleTTLY_x1 = plt.Subplot(figPerfSatisfTran, gsPerfSatisfTran[10, 1])
    axJobRoleTTLY_x1.set_anchor((0,.5))
    figPerfSatisfTran.add_subplot(axJobRoleTTLY_x1)
    axJobRoleTTLY_x1.spines['left'].set_visible(False)
    axJobRoleTTLY_x1.spines['right'].set_visible(True)

    df_JobRoleTTLYl1 = pd.DataFrame(np.zeros(shape = (len(JobRole), len(TrainingTimesLastYear))))
    row = 0
    for JR in JobRole:
        col = 0
        for TTLY in TrainingTimesLastYear:
            df_JobRoleTTLYl1.iloc[row, col] = (df_in_1.loc[(df_in_1['JobRole'] == JR) &                                                        (df_in_1['TrainingTimesLastYear'] == TTLY)]                                                       ['EmployeeNumber'].count()) /             (df_in_1.loc[(df_in_1['JobRole'] == JR)]['EmployeeNumber'].count())*100

            col = col + 1
        row = row + 1

    sns.heatmap(df_JobRoleTTLYl1, cbar = False, annot = True, fmt = '.0f', cmap = 'YlGnBu',                vmin = 0, vmax = 100,                linewidths = .5, ax = axJobRoleTTLY_x1)
    
    axJobRoleTTLY_x1.set_aspect('equal')    
    axJobRoleTTLY_x1.get_yaxis().set_label_position('right')
    axJobRoleTTLY_x1.get_yaxis().tick_right()
    axJobRoleTTLY_x1.invert_yaxis()
    axJobRoleTTLY_x1.set_ylabel('JobRole')
    axJobRoleTTLY_x1.set_yticklabels(JobRole, **{'rotation': 0})  
    axJobRoleTTLY_x1.set_xlabel('TrainingTimesLastYear')
    axJobRoleTTLY_x1.set_xticklabels(TrainingTimesLastYear, **{'rotation': 0})  
    

    axdividerJobTTLY_x1 = make_axes_locatable(axJobRoleTTLY_x1)
    axdividerJobTTLY_x1.set_anchor((0,.5))
    caxJobRoleTTLY_x1 = axdividerJobTTLY_x1.append_axes('bottom', size = '5%', pad = '20%')
    colorbar(axJobRoleTTLY_x1.get_children()[0], cax = caxJobRoleTTLY_x1,             orientation = 'horizontal', **{'ticks': (0, 100)})
    caxJobRoleTTLY_x1.xaxis.set_ticks_position('bottom')
    caxJobRoleTTLY_x1.set_xlabel('Emp.Count by Role[%]')
    caxJobRoleTTLY_x1.get_xaxis().set_label_position('bottom')
    
    axJobRoleTTLY_x1.set_title('JobRole vs TrainingTimesLastYear (Attrition = 1)', size = 14,                                   **{'horizontalalignment': 'left'})

    
    
    #[11,0]--------: Relation between YearsAtCompany & TrainingTimesLastYear (Attrition = 0)
    axYearsAtCompanyTTLY_x0 = plt.Subplot(figPerfSatisfTran, gsPerfSatisfTran[11, 0])
    axYearsAtCompanyTTLY_x0.set_anchor((0,.5))
    figPerfSatisfTran.add_subplot(axYearsAtCompanyTTLY_x0)

    sns.boxplot(x = 'TrainingTimesLastYear', y = 'YearsAtCompany', data = df_in_0, palette = palette_training,                boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                medianprops = medianprops, flierprops = flierprops, whis = whis,                ax = axYearsAtCompanyTTLY_x0).set_title('Y.AtComp. vs Training (Attrition = 0)', size = 14)

    pos = range(len(TrainingTimesLastYear))
    for tick, label in zip(pos, axYearsAtCompanyTTLY_x0.get_xticklabels()):
        df_aux = df_in_0.loc[(df_in_0['TrainingTimesLastYear'] == int(label.get_text()))]

        stats = f_stats_boxplot(df_aux['YearsAtCompany'])
        axYearsAtCompanyTTLY_x0.text(pos[tick] - .15, stats[1] - 2.5, '{:.0f}'.format(stats[1]),                                       color = 'red', ha = 'center')
        axYearsAtCompanyTTLY_x0.text(pos[tick] + .3, stats[2] - 2.5, '{:.0f}'.format(stats[2]),                                     color = 'black', ha = 'center')
        axYearsAtCompanyTTLY_x0.text(pos[tick], stats[3], '{:.0f}'.format(stats[3]),                                       color = 'black', ha = 'center')
        axYearsAtCompanyTTLY_x0.text(pos[tick] + .3, stats[4] + 1, '{:.0f}'.format(stats[4]),                                       color = 'black', ha = 'center')
        axYearsAtCompanyTTLY_x0.text(pos[tick] - .15, stats[5] + 1, '{:.0f}'.format(stats[5]),                                       color = 'red', ha = 'center')    

    axYearsAtCompanyTTLY_x0.set_xlabel('TrainingTimesLastYear')
    axYearsAtCompanyTTLY_x0.set_ylabel('YearsAtCompany')
    axYearsAtCompanyTTLY_x0.set_ylim((-3,42))
    axYearsAtCompanyTTLY_x0.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())
    
    
    
    #[11,1]--------: Relation between YearsAtCompany & TrainingTimesLastYear (Attrition = 1)
    axYearsAtCompanyTTLY_x1 = plt.Subplot(figPerfSatisfTran, gsPerfSatisfTran[11, 1])
    axYearsAtCompanyTTLY_x1.set_anchor((0,.5))
    figPerfSatisfTran.add_subplot(axYearsAtCompanyTTLY_x1)
    axYearsAtCompanyTTLY_x1.spines['left'].set_visible(False)
    axYearsAtCompanyTTLY_x1.spines['right'].set_visible(True)

    sns.boxplot(x = 'TrainingTimesLastYear', y = 'YearsAtCompany', data = df_in_1, palette = palette_training,                boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                medianprops = medianprops, flierprops = flierprops, whis = whis,                ax = axYearsAtCompanyTTLY_x1).set_title('Y.AtComp. vs Training (Attrition = 1)', size = 14)

    pos = range(len(TrainingTimesLastYear))
    for tick, label in zip(pos, axYearsAtCompanyTTLY_x1.get_xticklabels()):
        df_aux = df_in_1.loc[(df_in_1['TrainingTimesLastYear'] == int(label.get_text()))]

        stats = f_stats_boxplot(df_aux['YearsAtCompany'])
        axYearsAtCompanyTTLY_x1.text(pos[tick] - .15, stats[1] - 2.5, '{:.0f}'.format(stats[1]),                                       color = 'red', ha = 'center')
        axYearsAtCompanyTTLY_x1.text(pos[tick] + .3, stats[2] - 2.5, '{:.0f}'.format(stats[2]),                                       color = 'black', ha = 'center')
        axYearsAtCompanyTTLY_x1.text(pos[tick], stats[3], '{:.0f}'.format(stats[3]),                                       color = 'black', ha = 'center')
        axYearsAtCompanyTTLY_x1.text(pos[tick] + .3, stats[4] + 1, '{:.0f}'.format(stats[4]),                                     color = 'black', ha = 'center')
        axYearsAtCompanyTTLY_x1.text(pos[tick] - .15, stats[5] + 1, '{:.0f}'.format(stats[5]),                                       color = 'red', ha = 'center')    

    axYearsAtCompanyTTLY_x1.set_xlabel('TrainingTimesLastYear')
    axYearsAtCompanyTTLY_x1.set_ylabel('YearsAtCompany')
    axYearsAtCompanyTTLY_x1.set_ylim((-3,42))
    axYearsAtCompanyTTLY_x1.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())    
    axYearsAtCompanyTTLY_x1.get_yaxis().set_label_position("right")
    axYearsAtCompanyTTLY_x1.get_yaxis().tick_right()
    
    return figPerfSatisfTran


# In[9]:


def f_Remuneration_analysis(df_in):
    # Define indexes
    # Relation between the JobLevel & JobRole
    min_JobLevel = df_in['JobLevel'].min()
    max_JobLevel = df_in['JobLevel'].max()
    JobLevel = range(min_JobLevel, max_JobLevel + 1)
    JobLevel_list = [JobLevel[pos] for pos in range(0, len(JobLevel))]
    
    # PerformanceRating
    df_in['PerformanceRating_str'] = df_in.apply(lambda row: '#' + str(row['PerformanceRating']), axis = 1)
    min_PerformanceRating = df_in['PerformanceRating'].min()
    max_PerformanceRating = df_in['PerformanceRating'].max()
    PerformanceRating = range(min_PerformanceRating, max_PerformanceRating + 1)

    # OverTime
    df_in['OverTime_str'] = df_in.apply(lambda row: '#' + str(row['OverTime']), axis = 1)
    
    # StockOptionLevel
    min_StockOptionLevel = df_in['StockOptionLevel'].min()
    max_StockOptionLevel = df_in['StockOptionLevel'].max()
    StockOptionLevel = range(min_StockOptionLevel, max_StockOptionLevel + 1)   
    
    
    # Create the plot
    df_in_0 = df_in[df_in['Attrition'] == 0]
    df_in_1 = df_in[df_in['Attrition'] == 1]

    figRemun = plt.figure(figsize = (10, 80), dpi = 80, facecolor = 'w', edgecolor = 'k',                                 constrained_layout = False)
    gsRemun = gridspec.GridSpec(nrows = 17, ncols = 2, hspace = .3, wspace = .08,                         height_ratios = [1, 1, .4, .6, .4, 1, 1, .4, .6, .4, 1, 1, .4, .6, .4, 1, .38],                                width_ratios = [1, 1],                          figure = figRemun)
    

    #[0,0]--------: Relation between JobRole & MonthlyIncome (Attrition = 0)
    axJobRoleIncome_x0 = plt.Subplot(figRemun, gsRemun[0, 0])
    figRemun.add_subplot(axJobRoleIncome_x0)

    sns.boxplot(x = 'MonthlyIncome', y = 'JobRole', data = df_in_0, order = JobRole[::-1],                color = '#4ec5b0',                boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                medianprops = medianprops, flierprops = flierprops, whis = whis,                ax = axJobRoleIncome_x0).set_title('JobRole vs MonthlyIncome (Attrition = 0)', size = 14)

    pos = range(len(JobRole))
    for tick, label in zip(pos, axJobRoleIncome_x0.get_yticklabels()):
        
        df_aux_0 = df_in_0[df_in_0['JobRole'] == JobRole[::-1][tick]]['MonthlyIncome']
        stats = f_stats_boxplot(df_aux_0)
        axJobRoleIncome_x0.text(stats[1] - 1200, pos[tick] - .25, '{:.1f}k'.format(stats[1] / 1000),                                       color = 'red', ha = 'center')
        axJobRoleIncome_x0.text(stats[3] + 1000, pos[tick], '{:.1f}k'.format(stats[3] / 1000),                                       color = 'black', ha = 'center')
        axJobRoleIncome_x0.text(stats[5] + 1200, pos[tick] - .25, '{:.1f}k'.format(stats[5] / 1000),                                       color = 'red', ha = 'center')    
    
    axJobRoleIncome_x0.set_xlabel('MonthlyIncome')
    axJobRoleIncome_x0.set_ylabel('JobRole')
    axJobRoleIncome_x0.set_xlim((-1000, 22000))
    axJobRoleIncome_x0.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
    axJobRoleIncome_x0.get_xaxis().set_major_formatter(mtick.FuncFormatter(lambda x, pos: '{:.0f}k'.format(x / 1000))) 
    
    
    
    #[0,1]--------: Relation between JobRole & MonthlyIncome (Attrition = 1)
    axJobRoleIncome_x1 = plt.Subplot(figRemun, gsRemun[0, 1])
    figRemun.add_subplot(axJobRoleIncome_x1)
    axJobRoleIncome_x1.spines['left'].set_visible(False)
    axJobRoleIncome_x1.spines['right'].set_visible(True)

    sns.boxplot(x = 'MonthlyIncome', y = 'JobRole', data = df_in_1, order = JobRole[::-1],                color = '#4ec5b0',                boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                medianprops = medianprops, flierprops = flierprops, whis = whis,                ax = axJobRoleIncome_x1).set_title('JobRole vs MonthlyIncome (Attrition = 1)', size = 14)

    pos = range(len(JobRole))
    for tick, label in zip(pos, axJobRoleIncome_x1.get_yticklabels()):
        
        df_aux_1 = df_in_1[df_in_1['JobRole'] == JobRole[::-1][tick]]['MonthlyIncome']
        stats = f_stats_boxplot(df_aux_1)
        axJobRoleIncome_x1.text(stats[1] - 1200, pos[tick] - .25, '{:.1f}k'.format(stats[1] / 1000),                                       color = 'red', ha = 'center')
        axJobRoleIncome_x1.text(stats[3] + 1000, pos[tick], '{:.1f}k'.format(stats[3] / 1000),                                       color = 'black', ha = 'center')
        axJobRoleIncome_x1.text(stats[5] + 1200, pos[tick] - .25, '{:.1f}k'.format(stats[5] / 1000),                                       color = 'red', ha = 'center')    
    
    axJobRoleIncome_x1.set_xlabel('MonthlyIncome')
    axJobRoleIncome_x1.set_ylabel('JobRole')
    axJobRoleIncome_x1.get_yaxis().set_label_position("right")
    axJobRoleIncome_x1.get_yaxis().tick_right()
    axJobRoleIncome_x1.set_xlim((-1000, 22000))
    axJobRoleIncome_x1.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
    axJobRoleIncome_x1.get_xaxis().set_major_formatter(mtick.FuncFormatter(lambda x, pos: '{:.0f}k'.format(x / 1000))) 
    
    
    
    #[1,0]--------: Relation between YearsAtCompany & MonthlyIncome (Attrition = 0)
    axYearsatCompRemun_x0 = plt.Subplot(figRemun, gsRemun[1, 0])
    figRemun.add_subplot(axYearsatCompRemun_x0)

    sns.scatterplot(x = 'MonthlyIncome', y = 'YearsAtCompany', hue = 'JobLevel',                    palette = sns.color_palette("RdYlGn", len(JobLevel)),                    data = df_in_0,                    edgecolor = 'black', linewidth = .3, color = '#4ec5b0', marker = 'o', s = 20, alpha = .7,                    ax = axYearsatCompRemun_x0).set_title('Y.AtCompany vs M.Income (Attrition = 0)', size = 14)

    #handles, labels = axYearsatCompRemun_x0.get_legend_handles_labels()
    #axYearsatCompRemun_x0.legend(handles, labels, title = 'JobLevel', loc = 'best', prop = dict(size = 10))

    axYearsatCompRemun_x0.set_xlabel('Income')
    axYearsatCompRemun_x0.set_xlim((-1000, 22000))
    axYearsatCompRemun_x0.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
    axYearsatCompRemun_x0.get_xaxis().set_major_formatter(mtick.FuncFormatter(lambda x,                                                                                  pos: '{:.0f}k'.format(x / 1000)))
    axYearsatCompRemun_x0.set_ylabel('YearsAtCompany')
    axYearsatCompRemun_x0.set_ylim((-3, 42))
    axYearsatCompRemun_x0.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())

    

    #[1,1]--------: Relation between YearsAtCompany & MonthlyIncome (Attrition = 1)
    axYearsatCompRemun_x1 = plt.Subplot(figRemun, gsRemun[1, 1])
    figRemun.add_subplot(axYearsatCompRemun_x1)
    axYearsatCompRemun_x1.spines['left'].set_visible(False)
    axYearsatCompRemun_x1.spines['right'].set_visible(True)

    sns.scatterplot(x = 'MonthlyIncome', y = 'YearsAtCompany', hue = 'JobLevel',                    palette = sns.color_palette("RdYlGn", len(JobLevel)),                    data = df_in_1,                    edgecolor = 'black', linewidth = .3, color = '#4ec5b0', marker = 'o', s = 20, alpha = .7,                    ax = axYearsatCompRemun_x1).set_title('Y.AtCompany vs M.Income (Attrition = 1)', size = 14)

    #handles, labels = axYearsatCompRemun_x0.get_legend_handles_labels()
    #axYearsatCompRemun_x1.legend(handles, labels, title = 'JobLevel', loc = 'best', prop = dict(size = 10))

    axYearsatCompRemun_x1.set_xlabel('Income')
    axYearsatCompRemun_x1.set_xlim((-1000, 22000))
    axYearsatCompRemun_x1.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
    axYearsatCompRemun_x1.get_xaxis().set_major_formatter(mtick.FuncFormatter(lambda x,                                                                                  pos: '{:.0f}k'.format(x / 1000)))
    axYearsatCompRemun_x1.set_ylabel('YearsAtCompany')
    axYearsatCompRemun_x1.set_ylim((-3, 42))
    axYearsatCompRemun_x1.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())
    axYearsatCompRemun_x1.get_yaxis().set_label_position("right")
    axYearsatCompRemun_x1.get_yaxis().tick_right()    
    

    
    #[2,0]--------: Relation between PerformanceRating & MonthlyIncome (Attrition = 0)
    axRemunPerf_x0 = plt.Subplot(figRemun, gsRemun[2, 0])
    figRemun.add_subplot(axRemunPerf_x0)

    sns.boxplot(x = 'MonthlyIncome', y = 'PerformanceRating_str', data = df_in_0, palette = palette_relscale_str,                order = sorted(df_in_0['PerformanceRating_str'].unique()),                color = '#4ec5b0',                boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                medianprops = medianprops, flierprops = flierprops, whis = whis,                ax = axRemunPerf_x0).set_title('Perf.Rating vs M.Income (Attrition = 0)', size = 14)
    
    pos = range(len(df_in_0['PerformanceRating_str'].unique()))
    for tick, label in zip(pos, axRemunPerf_x0.get_yticklabels()):
        df_aux_0 = df_in_0[df_in_0['PerformanceRating_str'] == label.get_text()]['MonthlyIncome']
        stats = f_stats_boxplot(df_aux_0)
        axRemunPerf_x0.text(stats[1] - 1200, pos[tick], '{:.1f}k'.format(stats[1] / 1000),                                       color = 'red', ha = 'center')
        axRemunPerf_x0.text(stats[2] - 1200, pos[tick] - .3, '{:.1f}k'.format(stats[2] / 1000),                                       color = 'black', ha = 'center')
        axRemunPerf_x0.text(stats[3] + 1000, pos[tick], '{:.1f}k'.format(stats[3] / 1000),                                       color = 'black', ha = 'center')
        axRemunPerf_x0.text(stats[4] + 1200, pos[tick] - .3, '{:.1f}k'.format(stats[4] / 1000),                                       color = 'black', ha = 'center')
        axRemunPerf_x0.text(stats[5] + 1200, pos[tick], '{:.1f}k'.format(stats[5] / 1000),                                       color = 'red', ha = 'center')    

    axRemunPerf_x0.set_xlabel('MonthlyIncome')
    axRemunPerf_x0.set_ylabel('PerformanceRating')
    axRemunPerf_x0.invert_yaxis()
    axRemunPerf_x0.set_xlim((-1000, 22000))
    axRemunPerf_x0.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
    axRemunPerf_x0.get_xaxis().set_major_formatter(mtick.FuncFormatter(lambda x, pos: '{:.0f}k'.format(x / 1000))) 
    
    
    
    #[2,1]--------: Relation between PerformanceRating & MonthlyIncome (Attrition = 1)
    axRemunPerf_x1 = plt.Subplot(figRemun, gsRemun[2, 1])
    figRemun.add_subplot(axRemunPerf_x1)
    axRemunPerf_x1.spines['left'].set_visible(False)
    axRemunPerf_x1.spines['right'].set_visible(True)

    sns.boxplot(x = 'MonthlyIncome', y = 'PerformanceRating_str', data = df_in_1, palette = palette_relscale_str,                order = sorted(df_in_1['PerformanceRating_str'].unique()),                color = '#4ec5b0',                boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                medianprops = medianprops, flierprops = flierprops, whis = whis,                ax = axRemunPerf_x1).set_title('Perf.Rating vs M.Income (Attrition = 1)', size = 14)
    
    pos = range(len(df_in_1['PerformanceRating_str'].unique()))
    for tick, label in zip(pos, axRemunPerf_x1.get_yticklabels()):
        df_aux_1 = df_in_1[df_in_1['PerformanceRating_str'] == label.get_text()]['MonthlyIncome']
        stats = f_stats_boxplot(df_aux_1)
        axRemunPerf_x1.text(stats[1] - 1200, pos[tick], '{:.1f}k'.format(stats[1] / 1000),                                       color = 'red', ha = 'center')
        axRemunPerf_x1.text(stats[2] - 1200, pos[tick] - .3, '{:.1f}k'.format(stats[2] / 1000),                                       color = 'black', ha = 'center')
        axRemunPerf_x1.text(stats[3] + 1000, pos[tick], '{:.1f}k'.format(stats[3] / 1000),                                       color = 'black', ha = 'center')
        axRemunPerf_x1.text(stats[4] + 1200, pos[tick] - .3, '{:.1f}k'.format(stats[4] / 1000),                                       color = 'black', ha = 'center')
        axRemunPerf_x1.text(stats[5] + 1200, pos[tick], '{:.1f}k'.format(stats[5] / 1000),                                       color = 'red', ha = 'center')    

    axRemunPerf_x1.set_xlabel('MonthlyIncome')
    axRemunPerf_x1.set_ylabel('PerformanceRating')
    axRemunPerf_x1.get_yaxis().set_label_position("right")
    axRemunPerf_x1.get_yaxis().tick_right()
    axRemunPerf_x1.invert_yaxis()
    axRemunPerf_x1.set_xlim((-1000, 22000))
    axRemunPerf_x1.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
    axRemunPerf_x1.get_xaxis().set_major_formatter(mtick.FuncFormatter(lambda x, pos: '{:.0f}k'.format(x / 1000)))     

    
    
    #[3,0]--------: Relation between BusinessTravel & MonthlyIncome (Attrition = 0)
    axRemunBusinessTravel_x0 = plt.Subplot(figRemun, gsRemun[3, 0])
    figRemun.add_subplot(axRemunBusinessTravel_x0)

    sns.boxplot(x = 'MonthlyIncome', y = 'BusinessTravel', data = df_in_0, palette = businesstravellevels_colors,                order = businesstravellevels_index,                color = '#4ec5b0',                boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                medianprops = medianprops, flierprops = flierprops, whis = whis,                ax = axRemunBusinessTravel_x0).set_title('BusinessTravel vs M.Income (Attrition = 0)', size = 14)
    
    pos = range(len(df_in_0['BusinessTravel'].unique()))
    for tick, label in zip(pos, axRemunBusinessTravel_x0.get_yticklabels()):
        df_aux_0 = df_in_0[df_in_0['BusinessTravel'] == label.get_text()]['MonthlyIncome']
        stats = f_stats_boxplot(df_aux_0)
        axRemunBusinessTravel_x0.text(stats[1] - 1200, pos[tick], '{:.1f}k'.format(stats[1] / 1000),                                       color = 'red', ha = 'center')
        axRemunBusinessTravel_x0.text(stats[2] - 1200, pos[tick] - .3, '{:.1f}k'.format(stats[2] / 1000),                                       color = 'black', ha = 'center')
        axRemunBusinessTravel_x0.text(stats[3] + 1000, pos[tick], '{:.1f}k'.format(stats[3] / 1000),                                       color = ('black' if tick == 2 else 'white'), ha = 'center')
        axRemunBusinessTravel_x0.text(stats[4] + 1200, pos[tick] - .3, '{:.1f}k'.format(stats[4] / 1000),                                       color = 'black', ha = 'center')
        axRemunBusinessTravel_x0.text(stats[5] + 1200, pos[tick], '{:.1f}k'.format(stats[5] / 1000),                                       color = 'red', ha = 'center')    

    axRemunBusinessTravel_x0.set_xlabel('MonthlyIncome')
    axRemunBusinessTravel_x0.set_ylabel('BusinessTravel')
    axRemunBusinessTravel_x0.invert_yaxis()
    axRemunBusinessTravel_x0.set_xlim((-1000, 22000))
    axRemunBusinessTravel_x0.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
    axRemunBusinessTravel_x0.get_xaxis().set_major_formatter(mtick.FuncFormatter(lambda x,                                                                                 pos: '{:.0f}k'.format(x / 1000))) 
    
    
    
    #[3,1]--------: Relation between BusinessTravel & MonthlyIncome (Attrition = 1)
    axRemunBusinessTravel_x1 = plt.Subplot(figRemun, gsRemun[3, 1])
    figRemun.add_subplot(axRemunBusinessTravel_x1)
    axRemunBusinessTravel_x1.spines['left'].set_visible(False)
    axRemunBusinessTravel_x1.spines['right'].set_visible(True)

    sns.boxplot(x = 'MonthlyIncome', y = 'BusinessTravel', data = df_in_1, palette = businesstravellevels_colors,                order = businesstravellevels_index,
                boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,\
                medianprops = medianprops, flierprops = flierprops, whis = whis,\
                ax = axRemunBusinessTravel_x1).set_title('BusinessTravel vs M.Income (Attrition = 1)', size = 14)
    
    pos = range(len(df_in_1['BusinessTravel'].unique()))
    for tick, label in zip(pos, axRemunBusinessTravel_x1.get_yticklabels()):
        df_aux_1 = df_in_1[df_in_1['BusinessTravel'] == label.get_text()]['MonthlyIncome']
        stats = f_stats_boxplot(df_aux_1)
        axRemunBusinessTravel_x1.text(stats[1] - 1200, pos[tick], '{:.1f}k'.format(stats[1] / 1000),                                       color = 'red', ha = 'center')
        axRemunBusinessTravel_x1.text(stats[2] - 1200, pos[tick] - .3, '{:.1f}k'.format(stats[2] / 1000),                                       color = 'black', ha = 'center')
        axRemunBusinessTravel_x1.text(stats[3] + 1000, pos[tick], '{:.1f}k'.format(stats[3] / 1000),                                       color = ('black' if tick == 2 else 'white'), ha = 'center')
        axRemunBusinessTravel_x1.text(stats[4] + 1200, pos[tick] - .3, '{:.1f}k'.format(stats[4] / 1000),                                       color = 'black', ha = 'center')
        axRemunBusinessTravel_x1.text(stats[5] + 1200, pos[tick], '{:.1f}k'.format(stats[5] / 1000),                                       color = 'red', ha = 'center')    

    axRemunBusinessTravel_x1.set_xlabel('MonthlyIncome')
    axRemunBusinessTravel_x1.set_ylabel('BusinessTravel')
    axRemunBusinessTravel_x1.get_yaxis().set_label_position("right")
    axRemunBusinessTravel_x1.get_yaxis().tick_right() 
    axRemunBusinessTravel_x1.invert_yaxis()
    axRemunBusinessTravel_x1.set_xlim((-1000, 22000))
    axRemunBusinessTravel_x1.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
    axRemunBusinessTravel_x1.get_xaxis().set_major_formatter(mtick.FuncFormatter(lambda x,                                                                                  pos: '{:.0f}k'.format(x / 1000))) 
    
    
    
    #[4,0]--------: Relation between OverTime & MonthlyIncome (Attrition = 0)
    axRemunOvertime_x0 = plt.Subplot(figRemun, gsRemun[4, 0])
    figRemun.add_subplot(axRemunOvertime_x0)

    sns.boxplot(x = 'MonthlyIncome', y = 'OverTime_str', data = df_in_0, palette = overtimelevels_colors,                order = sorted(df_in_0['OverTime_str'].unique()),                color = '#4ec5b0',                boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                medianprops = medianprops, flierprops = flierprops, whis = whis,                ax = axRemunOvertime_x0).set_title('OverTime vs M.Income (Attrition = 0)', size = 14)
    
    pos = range(len(df_in_0['OverTime_str'].unique()))
    for tick, label in zip(pos, axRemunOvertime_x0.get_yticklabels()):
        df_aux_0 = df_in_0[df_in_0['OverTime_str'] == label.get_text()]['MonthlyIncome']
        stats = f_stats_boxplot(df_aux_0)
        axRemunOvertime_x0.text(stats[1] - 1200, pos[tick], '{:.1f}k'.format(stats[1] / 1000),                                       color = 'red', ha = 'center')
        axRemunOvertime_x0.text(stats[2] - 1200, pos[tick] - .3, '{:.1f}k'.format(stats[2] / 1000),                                       color = 'black', ha = 'center')
        axRemunOvertime_x0.text(stats[3] + 1000, pos[tick], '{:.1f}k'.format(stats[3] / 1000),                                       color = ('white' if tick == 0 else 'black'), ha = 'center')
        axRemunOvertime_x0.text(stats[4] + 1200, pos[tick] - .3, '{:.1f}k'.format(stats[4] / 1000),                                       color = 'black', ha = 'center')
        axRemunOvertime_x0.text(stats[5] + 1200, pos[tick], '{:.1f}k'.format(stats[5] / 1000),                                       color = 'red', ha = 'center')    

    axRemunOvertime_x0.set_xlabel('MonthlyIncome')
    axRemunOvertime_x0.set_ylabel('OverTime')
    axRemunOvertime_x0.invert_yaxis()
    axRemunOvertime_x0.set_xlim((-1000, 22000))
    axRemunOvertime_x0.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
    axRemunOvertime_x0.get_xaxis().set_major_formatter(mtick.FuncFormatter(lambda x, pos: '{:.0f}k'.format(x / 1000))) 
    
    
    
    #[4,1]--------: Relation between OverTime & MonthlyIncome (Attrition = 1)
    axRemunOvertime_x1 = plt.Subplot(figRemun, gsRemun[4, 1])
    figRemun.add_subplot(axRemunOvertime_x1)
    axRemunOvertime_x1.spines['left'].set_visible(False)
    axRemunOvertime_x1.spines['right'].set_visible(True)

    sns.boxplot(x = 'MonthlyIncome', y = 'OverTime_str', data = df_in_1, palette = overtimelevels_colors,                order = sorted(df_in_1['OverTime_str'].unique()),
                boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,\
                medianprops = medianprops, flierprops = flierprops, whis = whis,\
                ax = axRemunOvertime_x1).set_title('OverTime vs M.Income (Attrition = 1)', size = 14)
    
    pos = range(len(df_in_1['OverTime_str'].unique()))
    for tick, label in zip(pos, axRemunOvertime_x1.get_yticklabels()):
        df_aux_1 = df_in_1[df_in_1['OverTime_str'] == label.get_text()]['MonthlyIncome']
        stats = f_stats_boxplot(df_aux_1)
        axRemunOvertime_x1.text(stats[1] - 1200, pos[tick], '{:.1f}k'.format(stats[1] / 1000),                                       color = 'red', ha = 'center')
        axRemunOvertime_x1.text(stats[2] - 1200, pos[tick] - .3, '{:.1f}k'.format(stats[2] / 1000),                                       color = 'black', ha = 'center')
        axRemunOvertime_x1.text(stats[3] + 1000, pos[tick], '{:.1f}k'.format(stats[3] / 1000),                                       color = ('white' if tick == 0 else 'black'), ha = 'center')
        axRemunOvertime_x1.text(stats[4] + 1200, pos[tick] - .3, '{:.1f}k'.format(stats[4] / 1000),                                       color = 'black', ha = 'center')
        axRemunOvertime_x1.text(stats[5] + 1200, pos[tick], '{:.1f}k'.format(stats[5] / 1000),                                       color = 'red', ha = 'center')    

    axRemunOvertime_x1.set_xlabel('MonthlyIncome')
    axRemunOvertime_x1.set_ylabel('OverTime')
    axRemunOvertime_x1.get_yaxis().set_label_position("right")
    axRemunOvertime_x1.get_yaxis().tick_right() 
    axRemunOvertime_x1.invert_yaxis()
    axRemunOvertime_x1.set_xlim((-1000, 22000))
    axRemunOvertime_x1.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
    axRemunOvertime_x1.get_xaxis().set_major_formatter(mtick.FuncFormatter(lambda x, pos: '{:.0f}k'.format(x / 1000)))   

    
    
    #[5,0]--------: Relation between JobRole & YearsSinceLastPromotion (Attrition = 0)
    axYPromoAge_x0 = plt.Subplot(figRemun, gsRemun[5, 0])
    figRemun.add_subplot(axYPromoAge_x0)

    sns.boxplot(x = 'YearsSinceLastPromotion', y = 'JobRole', data = df_in_0, order = JobRole[::-1],                color = '#4294e5',                boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                medianprops = medianprops, flierprops = flierprops, whis = whis,                ax = axYPromoAge_x0).set_title('JobRole vs Y.LastPromotion (Attrition = 0)', size = 14)

    pos = range(len(JobRole))
    for tick, label in zip(pos, axYPromoAge_x0.get_yticklabels()):
        
        df_aux_0 = df_in_0[df_in_0['JobRole'] == JobRole[::-1][tick]]['YearsSinceLastPromotion']
        stats = f_stats_boxplot(df_aux_0)
        axYPromoAge_x0.text(stats[1] - .5, pos[tick], '{:.0f}'.format(stats[1]),                                       color = 'red', ha = 'center')
        axYPromoAge_x0.text(stats[2] - .5, pos[tick] + .4, '{:.0f}'.format(stats[2]),                                       color = 'black', ha = 'center')
        axYPromoAge_x0.text(stats[3] + .25, pos[tick], '{:.0f}'.format(stats[3]),                                       color = 'black', ha = 'center')
        axYPromoAge_x0.text(stats[4] + .5, pos[tick] + .4, '{:.0f}'.format(stats[4]),                                       color = 'black', ha = 'center')
        axYPromoAge_x0.text(stats[5] + .5, pos[tick], '{:.0f}'.format(stats[5]),                                       color = 'red', ha = 'center')    
    
    axYPromoAge_x0.set_xlabel('YearsSinceLastPromotion')
    axYPromoAge_x0.set_ylabel('JobRole')
    axYPromoAge_x0.set_xlim((-1.5, 17))
    axYPromoAge_x0.get_xaxis().set_major_locator(mtick.MaxNLocator(integer = True))
    axYPromoAge_x0.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())

    
    
    
    #[5,1]--------: Relation between JobRole & MonthlyIncome (Attrition = 1)
    axYPromoAge_x1 = plt.Subplot(figRemun, gsRemun[5, 1])
    figRemun.add_subplot(axYPromoAge_x1)
    axYPromoAge_x1.spines['left'].set_visible(False)
    axYPromoAge_x1.spines['right'].set_visible(True)

    sns.boxplot(x = 'YearsSinceLastPromotion', y = 'JobRole', data = df_in_1, order = JobRole[::-1],                color = '#4294e5',                boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                medianprops = medianprops, flierprops = flierprops, whis = whis,                ax = axYPromoAge_x1).set_title('JobRole vs Y.LastPromotion (Attrition = 1)', size = 14)

    pos = range(len(JobRole))
    for tick, label in zip(pos, axYPromoAge_x1.get_yticklabels()):
        
        df_aux_1 = df_in_1[df_in_1['JobRole'] == JobRole[::-1][tick]]['YearsSinceLastPromotion']
        stats = f_stats_boxplot(df_aux_1)
        axYPromoAge_x1.text(stats[1] - .5, pos[tick], '{:.0f}'.format(stats[1]),                                       color = 'red', ha = 'center')
        axYPromoAge_x1.text(stats[2] - .5, pos[tick] + .4, '{:.0f}'.format(stats[2]),                                       color = 'black', ha = 'center')
        axYPromoAge_x1.text(stats[3] + .25, pos[tick], '{:.0f}'.format(stats[3]),                                       color = 'black', ha = 'center')
        axYPromoAge_x1.text(stats[4] + .5, pos[tick] + .4, '{:.0f}'.format(stats[4]),                                       color = 'black', ha = 'center')
        axYPromoAge_x1.text(stats[5] + .5, pos[tick], '{:.0f}'.format(stats[5]),                                       color = 'red', ha = 'center')    
    
    axYPromoAge_x1.set_xlabel('YearsSinceLastPromotion')
    axYPromoAge_x1.set_ylabel('JobRole')
    axYPromoAge_x1.get_yaxis().set_label_position("right")
    axYPromoAge_x1.get_yaxis().tick_right()
    axYPromoAge_x1.set_xlim((-1.5, 17))
    axYPromoAge_x1.get_xaxis().set_major_locator(mtick.MaxNLocator(integer = True))
    axYPromoAge_x1.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
    

    
    #[6,0]--------: Relation between YearsAtCompany & YearsSinceLastPromotion (Attrition = 0)
    axYearsatCompYPromo_x0 = plt.Subplot(figRemun, gsRemun[6, 0])
    figRemun.add_subplot(axYearsatCompYPromo_x0)

    sns.scatterplot(x = 'YearsSinceLastPromotion', y = 'YearsAtCompany', hue = 'JobLevel',                    palette = sns.color_palette("RdYlGn", len(JobLevel)),                    data = df_in_0,                    edgecolor = 'black', linewidth = .3, color = '#4ec5b0', marker = 'o', s = 20, alpha = .7,                    ax = axYearsatCompYPromo_x0).set_title('Y.AtCompany vs Y.L.Promotion (Attrition = 0)', size = 14)

    axYearsatCompYPromo_x0.set_xlabel('YearsSinceLastPromotion')
    axYearsatCompYPromo_x0.set_xlim((-1.5, 17))
    axYearsatCompYPromo_x0.get_xaxis().set_major_locator(mtick.MaxNLocator(integer = True))
    axYearsatCompYPromo_x0.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())

    axYearsatCompYPromo_x0.set_ylabel('YearsAtCompany')
    axYearsatCompYPromo_x0.set_ylim((-3, 42))
    axYearsatCompYPromo_x0.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())
    
    # Plot line Y=x
    x_lim = axYearsatCompYPromo_x0.get_xlim()[1]
    y_lim = axYearsatCompYPromo_x0.get_ylim()[1]
    axYearsatCompYPromo_x0.plot((0, min(x_lim, y_lim)), (0, min(x_lim, y_lim)),                                color = 'red', alpha = 0.75, zorder = 0)
    axYearsatCompYPromo_x0.text(5, 3, 'Y=x', ha = 'center', **dict(size = 10, color = 'red'))   

    

    #[6,1]--------: Relation between YearsAtCompany & YearsSinceLastPromotion (Attrition = 1)
    axYearsatCompYPromo_x1 = plt.Subplot(figRemun, gsRemun[6, 1])
    figRemun.add_subplot(axYearsatCompYPromo_x1)
    axYearsatCompYPromo_x1.spines['left'].set_visible(False)
    axYearsatCompYPromo_x1.spines['right'].set_visible(True)

    sns.scatterplot(x = 'YearsSinceLastPromotion', y = 'YearsAtCompany', hue = 'JobLevel',                    palette = sns.color_palette("RdYlGn", len(JobLevel)),                    data = df_in_1,                    edgecolor = 'black', linewidth = .3, color = '#4ec5b0', marker = 'o', s = 20, alpha = .7,                    ax = axYearsatCompYPromo_x1).set_title('Y.AtCompany vs Y.L.Promotion (Attrition = 1)', size = 14)

    axYearsatCompYPromo_x1.set_xlabel('YearsSinceLastPromotion')
    axYearsatCompYPromo_x1.set_xlim((-1.5, 17))
    axYearsatCompYPromo_x1.get_xaxis().set_major_locator(mtick.MaxNLocator(integer = True))
    axYearsatCompYPromo_x1.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())

    axYearsatCompYPromo_x1.set_ylabel('YearsAtCompany')
    axYearsatCompYPromo_x1.set_ylim((-3, 42))
    axYearsatCompYPromo_x1.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())
    axYearsatCompYPromo_x1.get_yaxis().set_label_position("right")
    axYearsatCompYPromo_x1.get_yaxis().tick_right()  
    
    
    # Plot line Y=x
    x_lim = axYearsatCompYPromo_x1.get_xlim()[1]
    y_lim = axYearsatCompYPromo_x1.get_ylim()[1]
    axYearsatCompYPromo_x1.plot((0, min(x_lim, y_lim)), (0, min(x_lim, y_lim)),                                   color = 'red', alpha = 0.75, zorder = 0)
    axYearsatCompYPromo_x1.text(5, 3, 'Y=x', ha = 'center', **dict(size = 10, color = 'red')) 
    
    

    #[7,0]--------: Relation between PerformanceRating & MonthlyIncome (Attrition = 0)
    axYPromoPerf_x0 = plt.Subplot(figRemun, gsRemun[7, 0])
    figRemun.add_subplot(axYPromoPerf_x0)

    sns.boxplot(x = 'YearsSinceLastPromotion', y = 'PerformanceRating_str', data = df_in_0,                palette = palette_relscale_str,                order = sorted(df_in_0['PerformanceRating_str'].unique()),                color = '#4ec5b0',                boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                medianprops = medianprops, flierprops = flierprops, whis = whis,                ax = axYPromoPerf_x0).set_title('Perf.Rating vs Y.LastPromotion (Attrition = 0)', size = 14)
    
    pos = range(len(df_in_0['PerformanceRating_str'].unique()))
    for tick, label in zip(pos, axYPromoPerf_x0.get_yticklabels()):
        df_aux_0 = df_in_0[df_in_0['PerformanceRating_str'] == label.get_text()]['YearsSinceLastPromotion']
        stats = f_stats_boxplot(df_aux_0)
        axYPromoPerf_x0.text(stats[1] - .5, pos[tick], '{:.0f}'.format(stats[1]),                                       color = 'red', ha = 'center')
        axYPromoPerf_x0.text(stats[2] - .5, pos[tick] - .3, '{:.0f}'.format(stats[2]),                                       color = 'black', ha = 'center')
        axYPromoPerf_x0.text(stats[3] + .25, pos[tick], '{:.0f}'.format(stats[3]),                                       color = 'black', ha = 'center')
        axYPromoPerf_x0.text(stats[4] + .5, pos[tick] - .3, '{:.0f}'.format(stats[4]),                                       color = 'black', ha = 'center')
        axYPromoPerf_x0.text(stats[5] + .5, pos[tick], '{:.0f}'.format(stats[5]),                                       color = 'red', ha = 'center')    

    axYPromoPerf_x0.set_xlabel('YearsSinceLastPromotion')
    axYPromoPerf_x0.set_ylabel('PerformanceRating')
    axYPromoPerf_x0.invert_yaxis()
    axYPromoPerf_x0.set_xlim((-1.5, 17))
    axYPromoPerf_x0.get_xaxis().set_major_locator(mtick.MaxNLocator(integer = True))
    axYPromoPerf_x0.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
 
    
    
    #[7,1]--------: Relation between PerformanceRating & YearsSinceLastPromotion (Attrition = 1)
    axYPromoPerf_x1 = plt.Subplot(figRemun, gsRemun[7, 1])
    figRemun.add_subplot(axYPromoPerf_x1)
    axYPromoPerf_x1.spines['left'].set_visible(False)
    axYPromoPerf_x1.spines['right'].set_visible(True)

    sns.boxplot(x = 'YearsSinceLastPromotion', y = 'PerformanceRating_str', data = df_in_1,                palette = palette_relscale_str,                order = sorted(df_in_1['PerformanceRating_str'].unique()),                color = '#4ec5b0',                boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                medianprops = medianprops, flierprops = flierprops, whis = whis,                ax = axYPromoPerf_x1).set_title('Perf.Rating vs Y.LastPromotion (Attrition = 1)', size = 14)
    
    pos = range(len(df_in_1['PerformanceRating_str'].unique()))
    for tick, label in zip(pos, axYPromoPerf_x1.get_yticklabels()):
        df_aux_1 = df_in_1[df_in_1['PerformanceRating_str'] == label.get_text()]['YearsSinceLastPromotion']
        stats = f_stats_boxplot(df_aux_1)
        axYPromoPerf_x1.text(stats[1] - .5, pos[tick], '{:.0f}'.format(stats[1]),                                       color = 'red', ha = 'center')
        axYPromoPerf_x1.text(stats[2] - .5, pos[tick] - .3, '{:.0f}'.format(stats[2]),                                       color = 'black', ha = 'center')
        axYPromoPerf_x1.text(stats[3] + .25, pos[tick], '{:.0f}'.format(stats[3]),                                       color = 'black', ha = 'center')
        axYPromoPerf_x1.text(stats[4] + .5, pos[tick] - .3, '{:.0f}'.format(stats[4]),                                       color = 'black', ha = 'center')
        axYPromoPerf_x1.text(stats[5] + .5, pos[tick], '{:.0f}'.format(stats[5]),                                       color = 'red', ha = 'center')    

    axYPromoPerf_x1.set_xlabel('YearsSinceLastPromotion')
    axYPromoPerf_x1.set_ylabel('PerformanceRating')
    axYPromoPerf_x1.get_yaxis().set_label_position("right")
    axYPromoPerf_x1.get_yaxis().tick_right()
    axYPromoPerf_x1.invert_yaxis()
    axYPromoPerf_x1.set_xlim((-1.5, 17))
    axYPromoPerf_x1.get_xaxis().set_major_locator(mtick.MaxNLocator(integer = True))
    axYPromoPerf_x1.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())

    
    
    #[8,0]--------: Relation between BusinessTravel & MonthlyYearsSinceLastPromotionIncome (Attrition = 0)
    axYPromoBusinessTravel_x0 = plt.Subplot(figRemun, gsRemun[8, 0])
    figRemun.add_subplot(axYPromoBusinessTravel_x0)

    sns.boxplot(x = 'YearsSinceLastPromotion', y = 'BusinessTravel', data = df_in_0, palette = businesstravellevels_colors,                order = businesstravellevels_index,                color = '#4ec5b0',                boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                medianprops = medianprops, flierprops = flierprops, whis = whis,                ax = axYPromoBusinessTravel_x0).set_title('BusinessTravel vs Y.L.Prom. (Attrition = 0)', size = 14)
    
    pos = range(len(df_in_0['BusinessTravel'].unique()))
    for tick, label in zip(pos, axYPromoBusinessTravel_x0.get_yticklabels()):
        df_aux_0 = df_in_0[df_in_0['BusinessTravel'] == label.get_text()]['YearsSinceLastPromotion']
        stats = f_stats_boxplot(df_aux_0)
        axYPromoBusinessTravel_x0.text(stats[1] - .5, pos[tick], '{:.0f}'.format(stats[1]),                                       color = 'red', ha = 'center')
        axYPromoBusinessTravel_x0.text(stats[2] - .5, pos[tick] - .3, '{:.0f}'.format(stats[2]),                                       color = 'black', ha = 'center')
        axYPromoBusinessTravel_x0.text(stats[3] + .25, pos[tick], '{:.0f}'.format(stats[3]),                                       color = ('black' if tick == 2 else 'white'), ha = 'center')
        axYPromoBusinessTravel_x0.text(stats[4] + .5, pos[tick] - .3, '{:.0f}'.format(stats[4]),                                       color = 'black', ha = 'center')
        axYPromoBusinessTravel_x0.text(stats[5] + .5, pos[tick], '{:.0f}'.format(stats[5]),                                       color = 'red', ha = 'center')    

    axYPromoBusinessTravel_x0.set_xlabel('YearsSinceLastPromotion')
    axYPromoBusinessTravel_x0.set_ylabel('BusinessTravel')
    axYPromoBusinessTravel_x0.invert_yaxis()
    axYPromoBusinessTravel_x0.set_xlim((-1.5, 17))
    axYPromoBusinessTravel_x0.get_xaxis().set_major_locator(mtick.MaxNLocator(integer = True))
    axYPromoBusinessTravel_x0.get_xaxis().set_minor_locator(mtick.AutoMinorLocator()) 
    
    
    
    #[8,1]--------: Relation between BusinessTravel & YearsSinceLastPromotion (Attrition = 1)
    axYPromoBusinessTravel_x1 = plt.Subplot(figRemun, gsRemun[8, 1])
    figRemun.add_subplot(axYPromoBusinessTravel_x1)
    axYPromoBusinessTravel_x1.spines['left'].set_visible(False)
    axYPromoBusinessTravel_x1.spines['right'].set_visible(True)

    sns.boxplot(x = 'YearsSinceLastPromotion', y = 'BusinessTravel', data = df_in_1,                palette = businesstravellevels_colors,                order = businesstravellevels_index,
                boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,\
                medianprops = medianprops, flierprops = flierprops, whis = whis,\
                ax = axYPromoBusinessTravel_x1).set_title('BusinessTravel vs Y.L.Prom. (Attrition = 1)', size = 14)
    
    pos = range(len(df_in_1['BusinessTravel'].unique()))
    for tick, label in zip(pos, axYPromoBusinessTravel_x1.get_yticklabels()):
        df_aux_1 = df_in_1[df_in_1['BusinessTravel'] == label.get_text()]['YearsSinceLastPromotion']
        stats = f_stats_boxplot(df_aux_1)
        axYPromoBusinessTravel_x1.text(stats[1] - .5, pos[tick], '{:.0f}'.format(stats[1]),                                       color = 'red', ha = 'center')
        axYPromoBusinessTravel_x1.text(stats[2] - .5, pos[tick] - .3, '{:.0f}'.format(stats[2]),                                       color = 'black', ha = 'center')
        axYPromoBusinessTravel_x1.text(stats[3] + .25, pos[tick], '{:.0f}'.format(stats[3]),                                       color = ('black' if tick == 2 else 'white'), ha = 'center')
        axYPromoBusinessTravel_x1.text(stats[4] + .5, pos[tick] - .3, '{:.0f}'.format(stats[4]),                                       color = 'black', ha = 'center')
        axYPromoBusinessTravel_x1.text(stats[5] + .5, pos[tick], '{:.0f}'.format(stats[5]),                                       color = 'red', ha = 'center')    

    axYPromoBusinessTravel_x1.set_xlabel('YearsSinceLastPromotion')
    axYPromoBusinessTravel_x1.set_ylabel('BusinessTravel')
    axYPromoBusinessTravel_x1.get_yaxis().set_label_position("right")
    axYPromoBusinessTravel_x1.get_yaxis().tick_right() 
    axYPromoBusinessTravel_x1.invert_yaxis()
    axYPromoBusinessTravel_x1.set_xlim((-1.5, 17))
    axYPromoBusinessTravel_x1.get_xaxis().set_major_locator(mtick.MaxNLocator(integer = True))
    axYPromoBusinessTravel_x1.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
    
    
    
    #[9,0]--------: Relation between OverTime & YearsSinceLastPromotion (Attrition = 0)
    axYPromoOvertime_x0 = plt.Subplot(figRemun, gsRemun[9, 0])
    figRemun.add_subplot(axYPromoOvertime_x0)

    sns.boxplot(x = 'YearsSinceLastPromotion', y = 'OverTime_str', data = df_in_0, palette = overtimelevels_colors,                order = sorted(df_in_0['OverTime_str'].unique()),                color = '#4ec5b0',                boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                medianprops = medianprops, flierprops = flierprops, whis = whis,                ax = axYPromoOvertime_x0).set_title('OverTime vs Y.LastPromotion (Attrition = 0)', size = 14)
    
    pos = range(len(df_in_0['OverTime_str'].unique()))
    for tick, label in zip(pos, axYPromoOvertime_x0.get_yticklabels()):
        df_aux_0 = df_in_0[df_in_0['OverTime_str'] == label.get_text()]['YearsSinceLastPromotion']
        stats = f_stats_boxplot(df_aux_0)
        axYPromoOvertime_x0.text(stats[1] - .5, pos[tick], '{:.0f}'.format(stats[1]),                                       color = 'red', ha = 'center')
        axYPromoOvertime_x0.text(stats[2] - .5, pos[tick] - .3, '{:.0f}'.format(stats[2]),                                       color = 'black', ha = 'center')
        axYPromoOvertime_x0.text(stats[3] + .25, pos[tick], '{:.0f}'.format(stats[3]),                                       color = ('white' if tick == 0 else 'black'), ha = 'center')
        axYPromoOvertime_x0.text(stats[4] + .5, pos[tick] - .3, '{:.0f}'.format(stats[4]),                                       color = 'black', ha = 'center')
        axYPromoOvertime_x0.text(stats[5] + .5, pos[tick], '{:.0f}'.format(stats[5]),                                       color = 'red', ha = 'center')    

    axYPromoOvertime_x0.set_xlabel('YearsSinceLastPromotion')
    axYPromoOvertime_x0.set_ylabel('OverTime')
    axYPromoOvertime_x0.invert_yaxis()
    axYPromoOvertime_x0.set_xlim((-1.5, 17))
    axYPromoOvertime_x0.get_xaxis().set_major_locator(mtick.MaxNLocator(integer = True))
    axYPromoOvertime_x0.get_xaxis().set_minor_locator(mtick.AutoMinorLocator()) 
    
    
    
    #[9,1]--------: Relation between OverTime & YearsSinceLastPromotion (Attrition = 1)
    axYPromoOvertime_x1 = plt.Subplot(figRemun, gsRemun[9, 1])
    figRemun.add_subplot(axYPromoOvertime_x1)
    axYPromoOvertime_x1.spines['left'].set_visible(False)
    axYPromoOvertime_x1.spines['right'].set_visible(True)

    sns.boxplot(x = 'YearsSinceLastPromotion', y = 'OverTime_str', data = df_in_1, palette = overtimelevels_colors,                order = sorted(df_in_1['OverTime_str'].unique()),
                boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,\
                medianprops = medianprops, flierprops = flierprops, whis = whis,\
                ax = axYPromoOvertime_x1).set_title('OverTime vs Y.LastPromotion (Attrition = 1)', size = 14)
    
    pos = range(len(df_in_1['OverTime_str'].unique()))
    for tick, label in zip(pos, axYPromoOvertime_x1.get_yticklabels()):
        df_aux_1 = df_in_1[df_in_1['OverTime_str'] == label.get_text()]['YearsSinceLastPromotion']
        stats = f_stats_boxplot(df_aux_1)
        axYPromoOvertime_x1.text(stats[1] - .5, pos[tick], '{:.0f}'.format(stats[1]),                                       color = 'red', ha = 'center')
        axYPromoOvertime_x1.text(stats[2] - .5, pos[tick] - .3, '{:.0f}'.format(stats[2]),                                       color = 'black', ha = 'center')
        axYPromoOvertime_x1.text(stats[3] + .25, pos[tick], '{:.0f}'.format(stats[3]),                                       color = ('white' if tick == 0 else 'black'), ha = 'center')
        axYPromoOvertime_x1.text(stats[4] + .5, pos[tick] - .3, '{:.0f}'.format(stats[4]),                                       color = 'black', ha = 'center')
        axYPromoOvertime_x1.text(stats[5] + .5, pos[tick], '{:.0f}'.format(stats[5]),                                       color = 'red', ha = 'center')    

    axYPromoOvertime_x1.set_xlabel('YearsSinceLastPromotion')
    axYPromoOvertime_x1.set_ylabel('OverTime')
    axYPromoOvertime_x1.get_yaxis().set_label_position("right")
    axYPromoOvertime_x1.get_yaxis().tick_right() 
    axYPromoOvertime_x1.invert_yaxis()
    axYPromoOvertime_x1.set_xlim((-1.5, 17))
    axYPromoOvertime_x1.get_xaxis().set_major_locator(mtick.MaxNLocator(integer = True))
    axYPromoOvertime_x1.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
    
    

    #[10,0]--------: Relation between JobRole & PercentSalaryHike (Attrition = 0)
    axPSHikeAge_x0 = plt.Subplot(figRemun, gsRemun[10, 0])
    figRemun.add_subplot(axPSHikeAge_x0)

    sns.boxplot(x = 'PercentSalaryHike', y = 'JobRole', data = df_in_0, order = JobRole[::-1],                color = '#87b4dc',                boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                medianprops = medianprops, flierprops = flierprops, whis = whis,                ax = axPSHikeAge_x0).set_title('JobRole vs %SalaryHike (Attrition = 0)', size = 14)

    pos = range(len(JobRole))
    for tick, label in zip(pos, axPSHikeAge_x0.get_yticklabels()):
        
        df_aux_0 = df_in_0[df_in_0['JobRole'] == JobRole[::-1][tick]]['PercentSalaryHike']
        stats = f_stats_boxplot(df_aux_0)
        axPSHikeAge_x0.text(stats[1] - .5, pos[tick], '{:.0f}'.format(stats[1]),                                       color = 'red', ha = 'center')
        axPSHikeAge_x0.text(stats[2] - .5, pos[tick] + .4, '{:.0f}'.format(stats[2]),                                       color = 'black', ha = 'center')
        axPSHikeAge_x0.text(stats[3] + .25, pos[tick], '{:.0f}'.format(stats[3]),                                       color = 'black', ha = 'center')
        axPSHikeAge_x0.text(stats[4] + .5, pos[tick] + .4, '{:.0f}'.format(stats[4]),                                       color = 'black', ha = 'center')
        axPSHikeAge_x0.text(stats[5] + .5, pos[tick], '{:.0f}'.format(stats[5]),                                       color = 'red', ha = 'center')    
    
    axPSHikeAge_x0.set_xlabel('PercentSalaryHike')
    axPSHikeAge_x0.set_ylabel('JobRole')
    axPSHikeAge_x0.set_xlim((9, 27))
    axPSHikeAge_x0.get_xaxis().set_major_locator(mtick.MaxNLocator(integer = True))
    axPSHikeAge_x0.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())

    
    
    #[10,1]--------: Relation between JobRole & MonthlyIncome (Attrition = 1)
    axPSHikeAge_x1 = plt.Subplot(figRemun, gsRemun[10, 1])
    figRemun.add_subplot(axPSHikeAge_x1)
    axPSHikeAge_x1.spines['left'].set_visible(False)
    axPSHikeAge_x1.spines['right'].set_visible(True)

    sns.boxplot(x = 'PercentSalaryHike', y = 'JobRole', data = df_in_1, order = JobRole[::-1],                color = '#87b4dc',                boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                medianprops = medianprops, flierprops = flierprops, whis = whis,                ax = axPSHikeAge_x1).set_title('JobRole vs %SalaryHike (Attrition = 1)', size = 14)

    pos = range(len(JobRole))
    for tick, label in zip(pos, axPSHikeAge_x1.get_yticklabels()):
        
        df_aux_1 = df_in_1[df_in_1['JobRole'] == JobRole[::-1][tick]]['PercentSalaryHike']
        stats = f_stats_boxplot(df_aux_1)
        axPSHikeAge_x1.text(stats[1] - .5, pos[tick], '{:.0f}'.format(stats[1]),                                       color = 'red', ha = 'center')
        axPSHikeAge_x1.text(stats[2] - .5, pos[tick] + .4, '{:.0f}'.format(stats[2]),                                       color = 'black', ha = 'center')
        axPSHikeAge_x1.text(stats[3] + .25, pos[tick], '{:.0f}'.format(stats[3]),                                       color = 'black', ha = 'center')
        axPSHikeAge_x1.text(stats[4] + .5, pos[tick] + .4, '{:.0f}'.format(stats[4]),                                       color = 'black', ha = 'center')
        axPSHikeAge_x1.text(stats[5] + .5, pos[tick], '{:.0f}'.format(stats[5]),                                       color = 'red', ha = 'center')    
    
    axPSHikeAge_x1.set_xlabel('PercentSalaryHike')
    axPSHikeAge_x1.set_ylabel('JobRole')
    axPSHikeAge_x1.get_yaxis().set_label_position("right")
    axPSHikeAge_x1.get_yaxis().tick_right()
    axPSHikeAge_x1.set_xlim((9, 27))
    axPSHikeAge_x1.get_xaxis().set_major_locator(mtick.MaxNLocator(integer = True))
    axPSHikeAge_x1.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
    

    
    #[11,0]--------: Relation between YearsAtCompany & PercentSalaryHike (Attrition = 0)
    axYearsatCompYPromo_x0 = plt.Subplot(figRemun, gsRemun[11, 0])
    figRemun.add_subplot(axYearsatCompYPromo_x0)

    sns.scatterplot(x = 'PercentSalaryHike', y = 'YearsAtCompany', hue = 'JobLevel',                    palette = sns.color_palette("RdYlGn", len(JobLevel)),                    data = df_in_0,                    edgecolor = 'black', linewidth = .3, color = '#4ec5b0', marker = 'o', s = 20, alpha = .7,                    ax = axYearsatCompYPromo_x0).set_title('Y.AtCompany vs %SalaryHike (Attrition = 0)', size = 14)

    axYearsatCompYPromo_x0.set_xlabel('PercentSalaryHike')
    axYearsatCompYPromo_x0.set_xlim((9, 27))
    axYearsatCompYPromo_x0.get_xaxis().set_major_locator(mtick.MaxNLocator(integer = True))
    axYearsatCompYPromo_x0.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())

    axYearsatCompYPromo_x0.set_ylabel('YearsAtCompany')
    axYearsatCompYPromo_x0.set_ylim((-3, 42))
    axYearsatCompYPromo_x0.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())
    
    

    #[11,1]--------: Relation between YearsAtCompany & PercentSalaryHike (Attrition = 1)
    axYearsatCompYPromo_x1 = plt.Subplot(figRemun, gsRemun[11, 1])
    figRemun.add_subplot(axYearsatCompYPromo_x1)
    axYearsatCompYPromo_x1.spines['left'].set_visible(False)
    axYearsatCompYPromo_x1.spines['right'].set_visible(True)

    sns.scatterplot(x = 'PercentSalaryHike', y = 'YearsAtCompany', hue = 'JobLevel',                    palette = sns.color_palette("RdYlGn", len(JobLevel)),                    data = df_in_1,                    edgecolor = 'black', linewidth = .3, color = '#4ec5b0', marker = 'o', s = 20, alpha = .7,                    ax = axYearsatCompYPromo_x1).set_title('Y.AtCompany vs %SalaryHike (Attrition = 1)', size = 14)

    axYearsatCompYPromo_x1.set_xlabel('PercentSalaryHike')
    axYearsatCompYPromo_x1.set_xlim((9, 27))
    axYearsatCompYPromo_x1.get_xaxis().set_major_locator(mtick.MaxNLocator(integer = True))
    axYearsatCompYPromo_x1.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())

    axYearsatCompYPromo_x1.set_ylabel('YearsAtCompany')
    axYearsatCompYPromo_x1.set_ylim((-3, 42))
    axYearsatCompYPromo_x1.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())
    axYearsatCompYPromo_x1.get_yaxis().set_label_position("right")
    axYearsatCompYPromo_x1.get_yaxis().tick_right()   
    
    

    #[12,0]--------: Relation between PerformanceRating & MonthlyIncome (Attrition = 0)
    axPSHikePerf_x0 = plt.Subplot(figRemun, gsRemun[12, 0])
    figRemun.add_subplot(axPSHikePerf_x0)

    sns.boxplot(x = 'PercentSalaryHike', y = 'PerformanceRating_str', data = df_in_0,                palette = palette_relscale_str,                order = sorted(df_in_0['PerformanceRating_str'].unique()),                color = '#4ec5b0',                boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                medianprops = medianprops, flierprops = flierprops, whis = whis,                ax = axPSHikePerf_x0).set_title('Perf.Rating vs %SalaryHike (Attrition = 0)', size = 14)
    
    pos = range(len(df_in_0['PerformanceRating_str'].unique()))
    for tick, label in zip(pos, axPSHikePerf_x0.get_yticklabels()):
        df_aux_0 = df_in_0[df_in_0['PerformanceRating_str'] == label.get_text()]['PercentSalaryHike']
        stats = f_stats_boxplot(df_aux_0)
        axPSHikePerf_x0.text(stats[1] - .5, pos[tick], '{:.0f}'.format(stats[1]),                                       color = 'red', ha = 'center')
        axPSHikePerf_x0.text(stats[2] - .5, pos[tick] - .3, '{:.0f}'.format(stats[2]),                                       color = 'black', ha = 'center')
        axPSHikePerf_x0.text(stats[3] + .25, pos[tick], '{:.0f}'.format(stats[3]),                                       color = 'black', ha = 'center')
        axPSHikePerf_x0.text(stats[4] + .5, pos[tick] - .3, '{:.0f}'.format(stats[4]),                                       color = 'black', ha = 'center')
        axPSHikePerf_x0.text(stats[5] + .5, pos[tick], '{:.0f}'.format(stats[5]),                                       color = 'red', ha = 'center')    

    axPSHikePerf_x0.set_xlabel('PercentSalaryHike')
    axPSHikePerf_x0.set_ylabel('PerformanceRating')
    axPSHikePerf_x0.invert_yaxis()
    axPSHikePerf_x0.set_xlim((9, 27))
    axPSHikePerf_x0.get_xaxis().set_major_locator(mtick.MaxNLocator(integer = True))
    axPSHikePerf_x0.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
 
    
    
    #[12,1]--------: Relation between PerformanceRating & PercentSalaryHike (Attrition = 1)
    axPSHikePerf_x1 = plt.Subplot(figRemun, gsRemun[12, 1])
    figRemun.add_subplot(axPSHikePerf_x1)
    axPSHikePerf_x1.spines['left'].set_visible(False)
    axPSHikePerf_x1.spines['right'].set_visible(True)

    sns.boxplot(x = 'PercentSalaryHike', y = 'PerformanceRating_str', data = df_in_1,                palette = palette_relscale_str,                order = sorted(df_in_1['PerformanceRating_str'].unique()),                color = '#4ec5b0',                boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                medianprops = medianprops, flierprops = flierprops, whis = whis,                ax = axPSHikePerf_x1).set_title('Perf.Rating vs %SalaryHike (Attrition = 1)', size = 14)
    
    pos = range(len(df_in_1['PerformanceRating_str'].unique()))
    for tick, label in zip(pos, axPSHikePerf_x1.get_yticklabels()):
        df_aux_1 = df_in_1[df_in_1['PerformanceRating_str'] == label.get_text()]['PercentSalaryHike']
        stats = f_stats_boxplot(df_aux_1)
        axPSHikePerf_x1.text(stats[1] - .5, pos[tick], '{:.0f}'.format(stats[1]),                                       color = 'red', ha = 'center')
        axPSHikePerf_x1.text(stats[2] - .5, pos[tick] - .3, '{:.0f}'.format(stats[2]),                                       color = 'black', ha = 'center')
        axPSHikePerf_x1.text(stats[3] + .25, pos[tick], '{:.0f}'.format(stats[3]),                                       color = 'black', ha = 'center')
        axPSHikePerf_x1.text(stats[4] + .5, pos[tick] - .3, '{:.0f}'.format(stats[4]),                                       color = 'black', ha = 'center')
        axPSHikePerf_x1.text(stats[5] + .5, pos[tick], '{:.0f}'.format(stats[5]),                                       color = 'red', ha = 'center')    

    axPSHikePerf_x1.set_xlabel('PercentSalaryHike')
    axPSHikePerf_x1.set_ylabel('PerformanceRating')
    axPSHikePerf_x1.get_yaxis().set_label_position("right")
    axPSHikePerf_x1.get_yaxis().tick_right()
    axPSHikePerf_x1.invert_yaxis()
    axPSHikePerf_x1.set_xlim((9, 27))
    axPSHikePerf_x1.get_xaxis().set_major_locator(mtick.MaxNLocator(integer = True))
    axPSHikePerf_x1.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())

    
    
    #[13,0]--------: Relation between BusinessTravel & MonthlyPercentSalaryHikeIncome (Attrition = 0)
    axPSHikeBusinessTravel_x0 = plt.Subplot(figRemun, gsRemun[13, 0])
    figRemun.add_subplot(axPSHikeBusinessTravel_x0)

    sns.boxplot(x = 'PercentSalaryHike', y = 'BusinessTravel', data = df_in_0, palette = businesstravellevels_colors,                order = businesstravellevels_index,                color = '#4ec5b0',                boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                medianprops = medianprops, flierprops = flierprops, whis = whis,                ax = axPSHikeBusinessTravel_x0).set_title('BusinessTravel vs %SalaryHike (Attrition = 0)', size = 14)
    
    pos = range(len(df_in_0['BusinessTravel'].unique()))
    for tick, label in zip(pos, axPSHikeBusinessTravel_x0.get_yticklabels()):
        df_aux_0 = df_in_0[df_in_0['BusinessTravel'] == label.get_text()]['PercentSalaryHike']
        stats = f_stats_boxplot(df_aux_0)
        axPSHikeBusinessTravel_x0.text(stats[1] - .5, pos[tick], '{:.0f}'.format(stats[1]),                                       color = 'red', ha = 'center')
        axPSHikeBusinessTravel_x0.text(stats[2] - .5, pos[tick] - .3, '{:.0f}'.format(stats[2]),                                       color = 'black', ha = 'center')
        axPSHikeBusinessTravel_x0.text(stats[3] + .25, pos[tick], '{:.0f}'.format(stats[3]),                                       color = ('black' if tick == 2 else 'white'), ha = 'center')
        axPSHikeBusinessTravel_x0.text(stats[4] + .5, pos[tick] - .3, '{:.0f}'.format(stats[4]),                                       color = 'black', ha = 'center')
        axPSHikeBusinessTravel_x0.text(stats[5] + .5, pos[tick], '{:.0f}'.format(stats[5]),                                       color = 'red', ha = 'center')    

    axPSHikeBusinessTravel_x0.set_xlabel('PercentSalaryHike')
    axPSHikeBusinessTravel_x0.set_ylabel('BusinessTravel')
    axPSHikeBusinessTravel_x0.invert_yaxis()
    axPSHikeBusinessTravel_x0.set_xlim((9, 27))
    axPSHikeBusinessTravel_x0.get_xaxis().set_major_locator(mtick.MaxNLocator(integer = True))
    axPSHikeBusinessTravel_x0.get_xaxis().set_minor_locator(mtick.AutoMinorLocator()) 
    
    
    
    #[13,1]--------: Relation between BusinessTravel & PercentSalaryHike (Attrition = 1)
    axPSHikeBusinessTravel_x1 = plt.Subplot(figRemun, gsRemun[13, 1])
    figRemun.add_subplot(axPSHikeBusinessTravel_x1)
    axPSHikeBusinessTravel_x1.spines['left'].set_visible(False)
    axPSHikeBusinessTravel_x1.spines['right'].set_visible(True)

    sns.boxplot(x = 'PercentSalaryHike', y = 'BusinessTravel', data = df_in_1,                palette = businesstravellevels_colors,                order = businesstravellevels_index,
                boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,\
                medianprops = medianprops, flierprops = flierprops, whis = whis,\
                ax = axPSHikeBusinessTravel_x1).set_title('BusinessTravel vs %SalaryHike (Attrition = 1)', size = 14)
    
    pos = range(len(df_in_1['BusinessTravel'].unique()))
    for tick, label in zip(pos, axPSHikeBusinessTravel_x1.get_yticklabels()):
        df_aux_1 = df_in_1[df_in_1['BusinessTravel'] == label.get_text()]['PercentSalaryHike']
        stats = f_stats_boxplot(df_aux_1)
        axPSHikeBusinessTravel_x1.text(stats[1] - .5, pos[tick], '{:.0f}'.format(stats[1]),                                       color = 'red', ha = 'center')
        axPSHikeBusinessTravel_x1.text(stats[2] - .5, pos[tick] - .3, '{:.0f}'.format(stats[2]),                                       color = 'black', ha = 'center')
        axPSHikeBusinessTravel_x1.text(stats[3] + .25, pos[tick], '{:.0f}'.format(stats[3]),                                       color = ('black' if tick == 2 else 'white'), ha = 'center')
        axPSHikeBusinessTravel_x1.text(stats[4] + .5, pos[tick] - .3, '{:.0f}'.format(stats[4]),                                       color = 'black', ha = 'center')
        axPSHikeBusinessTravel_x1.text(stats[5] + .5, pos[tick], '{:.0f}'.format(stats[5]),                                       color = 'red', ha = 'center')    

    axPSHikeBusinessTravel_x1.set_xlabel('PercentSalaryHike')
    axPSHikeBusinessTravel_x1.set_ylabel('BusinessTravel')
    axPSHikeBusinessTravel_x1.get_yaxis().set_label_position("right")
    axPSHikeBusinessTravel_x1.get_yaxis().tick_right() 
    axPSHikeBusinessTravel_x1.invert_yaxis()
    axPSHikeBusinessTravel_x1.set_xlim((9, 27))
    axPSHikeBusinessTravel_x1.get_xaxis().set_major_locator(mtick.MaxNLocator(integer = True))
    axPSHikeBusinessTravel_x1.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
    
    
    
    #[14,0]--------: Relation between OverTime & PercentSalaryHike (Attrition = 0)
    axPSHikeOvertime_x0 = plt.Subplot(figRemun, gsRemun[14, 0])
    figRemun.add_subplot(axPSHikeOvertime_x0)

    sns.boxplot(x = 'PercentSalaryHike', y = 'OverTime_str', data = df_in_0, palette = overtimelevels_colors,                order = sorted(df_in_0['OverTime_str'].unique()),                color = '#4ec5b0',                boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                medianprops = medianprops, flierprops = flierprops, whis = whis,                ax = axPSHikeOvertime_x0).set_title('OverTime vs %SalaryHike (Attrition = 0)', size = 14)
    
    pos = range(len(df_in_0['OverTime_str'].unique()))
    for tick, label in zip(pos, axPSHikeOvertime_x0.get_yticklabels()):
        df_aux_0 = df_in_0[df_in_0['OverTime_str'] == label.get_text()]['PercentSalaryHike']
        stats = f_stats_boxplot(df_aux_0)
        axPSHikeOvertime_x0.text(stats[1] - .5, pos[tick], '{:.0f}'.format(stats[1]),                                       color = 'red', ha = 'center')
        axPSHikeOvertime_x0.text(stats[2] - .5, pos[tick] - .3, '{:.0f}'.format(stats[2]),                                       color = 'black', ha = 'center')
        axPSHikeOvertime_x0.text(stats[3] + .25, pos[tick], '{:.0f}'.format(stats[3]),                                       color = ('white' if tick == 0 else 'black'), ha = 'center')
        axPSHikeOvertime_x0.text(stats[4] + .5, pos[tick] - .3, '{:.0f}'.format(stats[4]),                                       color = 'black', ha = 'center')
        axPSHikeOvertime_x0.text(stats[5] + .5, pos[tick], '{:.0f}'.format(stats[5]),                                       color = 'red', ha = 'center')    

    axPSHikeOvertime_x0.set_xlabel('PercentSalaryHike')
    axPSHikeOvertime_x0.set_ylabel('OverTime')
    axPSHikeOvertime_x0.invert_yaxis()
    axPSHikeOvertime_x0.set_xlim((9, 27))
    axPSHikeOvertime_x0.get_xaxis().set_major_locator(mtick.MaxNLocator(integer = True))
    axPSHikeOvertime_x0.get_xaxis().set_minor_locator(mtick.AutoMinorLocator()) 
    
    
    
    #[14,1]--------: Relation between OverTime & PercentSalaryHike (Attrition = 1)
    axPSHikeOvertime_x1 = plt.Subplot(figRemun, gsRemun[14, 1])
    figRemun.add_subplot(axPSHikeOvertime_x1)
    axPSHikeOvertime_x1.spines['left'].set_visible(False)
    axPSHikeOvertime_x1.spines['right'].set_visible(True)

    sns.boxplot(x = 'PercentSalaryHike', y = 'OverTime_str', data = df_in_1, palette = overtimelevels_colors,                order = sorted(df_in_1['OverTime_str'].unique()),
                boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,\
                medianprops = medianprops, flierprops = flierprops, whis = whis,\
                ax = axPSHikeOvertime_x1).set_title('OverTime vs %SalaryHike (Attrition = 1)', size = 14)
    
    pos = range(len(df_in_1['OverTime_str'].unique()))
    for tick, label in zip(pos, axPSHikeOvertime_x1.get_yticklabels()):
        df_aux_1 = df_in_1[df_in_1['OverTime_str'] == label.get_text()]['PercentSalaryHike']
        stats = f_stats_boxplot(df_aux_1)
        axPSHikeOvertime_x1.text(stats[1] - .5, pos[tick], '{:.0f}'.format(stats[1]),                                       color = 'red', ha = 'center')
        axPSHikeOvertime_x1.text(stats[2] - .5, pos[tick] - .3, '{:.0f}'.format(stats[2]),                                       color = 'black', ha = 'center')
        axPSHikeOvertime_x1.text(stats[3] + .25, pos[tick], '{:.0f}'.format(stats[3]),                                       color = ('white' if tick == 0 else 'black'), ha = 'center')
        axPSHikeOvertime_x1.text(stats[4] + .5, pos[tick] - .3, '{:.0f}'.format(stats[4]),                                       color = 'black', ha = 'center')
        axPSHikeOvertime_x1.text(stats[5] + .5, pos[tick], '{:.0f}'.format(stats[5]),                                       color = 'red', ha = 'center')    

    axPSHikeOvertime_x1.set_xlabel('PercentSalaryHike')
    axPSHikeOvertime_x1.set_ylabel('OverTime')
    axPSHikeOvertime_x1.get_yaxis().set_label_position("right")
    axPSHikeOvertime_x1.get_yaxis().tick_right() 
    axPSHikeOvertime_x1.invert_yaxis()
    axPSHikeOvertime_x1.set_xlim((9, 27))
    axPSHikeOvertime_x1.get_xaxis().set_major_locator(mtick.MaxNLocator(integer = True))
    axPSHikeOvertime_x1.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
    
    
    
    #[15,0]--------: Heatmap for the relation between the JobRole & StockOptionLevel (Attrition = 0)
    axJobRoleStock_x0 = plt.Subplot(figRemun, gsRemun[15, 0])
    axJobRoleStock_x0.set_anchor((1,.5))
    figRemun.add_subplot(axJobRoleStock_x0)

    df_JobRolePerfl0 = pd.DataFrame(np.zeros(shape = (len(JobRole), len(StockOptionLevel))))
    row = 0
    for JR in JobRole:
        col = 0
        for SOL in StockOptionLevel:
            df_JobRolePerfl0.iloc[row, col] = (df_in_0.loc[(df_in_0['JobRole'] == JR) &                                                            (df_in_0['StockOptionLevel'] == SOL)]                                                              ['EmployeeNumber'].count()) /             (df_in_0.loc[(df_in_0['JobRole'] == JR)]['EmployeeNumber'].count()) * 100 
                         
            col = col + 1
        row = row + 1

    sns.heatmap(df_JobRolePerfl0, cbar = False, annot = True, fmt = '.0f', cmap = 'YlGnBu',                vmin = 0, vmax = 100,                linewidths = .5, ax = axJobRoleStock_x0)
    
    axJobRoleStock_x0.set_aspect('equal')    
    axJobRoleStock_x0.get_yaxis().set_label_position('left')
    axJobRoleStock_x0.get_yaxis().tick_left()
    axJobRoleStock_x0.invert_yaxis()
    axJobRoleStock_x0.set_ylabel('JobRole')
    axJobRoleStock_x0.set_yticklabels(JobRole, **{'rotation': 0})  
    axJobRoleStock_x0.set_xlabel('StockOptionLevel')
    axJobRoleStock_x0.set_xticklabels(StockOptionLevel, **{'rotation': 0})  
    
    axdividerJobRolePerf_x0 = make_axes_locatable(axJobRoleStock_x0)
    axdividerJobRolePerf_x0.set_anchor((1,.5))
    caxJobRoleStock_x0 = axdividerJobRolePerf_x0.append_axes('bottom', size = '5%', pad = '20%')
    colorbar(axJobRoleStock_x0.get_children()[0], cax = caxJobRoleStock_x0, orientation = 'horizontal',             **{'ticks': (0, 100)})
    caxJobRoleStock_x0.xaxis.set_ticks_position('bottom')
    caxJobRoleStock_x0.set_xlabel('Emp.Count by Role[%]')
    caxJobRoleStock_x0.get_xaxis().set_label_position('bottom')
    
    axJobRoleStock_x0.set_title('JobRole vs StockOptionLevel (Attrition = 0)', size = 14,                               **{'horizontalalignment': 'right'})

    
    
    #[15,1]--------: Heatmap for the relation between the JobRole & StockOptionLevel (Attrition = 1)
    axJobRoleStock_x1 = plt.Subplot(figRemun, gsRemun[15, 1])
    axJobRoleStock_x1.set_anchor((0,.5))
    figRemun.add_subplot(axJobRoleStock_x1)
    axJobRoleStock_x1.spines['left'].set_visible(False)
    axJobRoleStock_x1.spines['right'].set_visible(True)

    df_JobRolePerfl1 = pd.DataFrame(np.zeros(shape = (len(JobRole), len(StockOptionLevel))))
    row = 0
    for JR in JobRole:
        col = 0
        for SOL in StockOptionLevel:
            df_JobRolePerfl1.iloc[row, col] = df_in_1.loc[(df_in_1['JobRole'] == JR) &                                                            (df_in_1['StockOptionLevel'] == SOL)]                                                                         ['EmployeeNumber'].count() /             (df_in_1.loc[(df_in_1['JobRole'] == JR)]['EmployeeNumber'].count()) * 100
            
            col = col + 1
        row = row + 1

    sns.heatmap(df_JobRolePerfl1, cbar = False, annot = True, fmt = '.0f', cmap = 'YlGnBu',                vmin = 0, vmax = 100,                linewidths = .5, ax = axJobRoleStock_x1)
    
    axJobRoleStock_x1.set_aspect('equal')    
    axJobRoleStock_x1.get_yaxis().set_label_position('right')
    axJobRoleStock_x1.get_yaxis().tick_right()
    axJobRoleStock_x1.invert_yaxis()
    axJobRoleStock_x1.set_ylabel('JobRole')
    axJobRoleStock_x1.set_yticklabels(JobRole, **{'rotation': 0})  
    axJobRoleStock_x1.set_xlabel('StockOptionLevel')
    axJobRoleStock_x1.set_xticklabels(StockOptionLevel, **{'rotation': 0})  
    

    axdividerJobRolePerf_x1 = make_axes_locatable(axJobRoleStock_x1)
    axdividerJobRolePerf_x1.set_anchor((0,.5))
    caxJobRoleStock_x1 = axdividerJobRolePerf_x1.append_axes('bottom', size = '5%', pad = '20%')
    colorbar(axJobRoleStock_x1.get_children()[0], cax = caxJobRoleStock_x1, orientation = 'horizontal',             **{'ticks': (0, 100)})
    caxJobRoleStock_x1.xaxis.set_ticks_position('bottom')
    caxJobRoleStock_x1.set_xlabel('Emp.Count by Role[%]')
    caxJobRoleStock_x1.get_xaxis().set_label_position('bottom')
    
    axJobRoleStock_x1.set_title('JobRole vs StockOptionLevel (Attrition = 1)', size = 14,                                   **{'horizontalalignment': 'left'})
    

    #[16,0]--------: Heatmap for the relation between the PerformanceRating & StockOptionLevel (Attrition = 0)
    axPerfRatingStock_x0 = plt.Subplot(figRemun, gsRemun[16, 0])
    axPerfRatingStock_x0.set_anchor((1,.5))
    figRemun.add_subplot(axPerfRatingStock_x0)

    df_JobRolePerfl0 = pd.DataFrame(np.zeros(shape = (len(PerformanceRating), len(StockOptionLevel))))
    row = 0
    for PR in PerformanceRating:
        col = 0
        for SOL in StockOptionLevel:
            df_JobRolePerfl0.iloc[row, col] = (df_in_0.loc[(df_in_0['PerformanceRating'] == PR) &                                                            (df_in_0['StockOptionLevel'] == SOL)]                                                              ['EmployeeNumber'].count()) /             (df_in_0.loc[(df_in_0['PerformanceRating'] == PR)]['EmployeeNumber'].count()) * 100 
                         
            col = col + 1
        row = row + 1

    sns.heatmap(df_JobRolePerfl0, cbar = False, annot = True, fmt = '.0f', cmap = 'YlGnBu',                vmin = 0, vmax = 100,                linewidths = .5, ax = axPerfRatingStock_x0)
    
    axPerfRatingStock_x0.set_aspect('equal')    
    axPerfRatingStock_x0.get_yaxis().set_label_position('left')
    axPerfRatingStock_x0.get_yaxis().tick_left()
    axPerfRatingStock_x0.invert_yaxis()
    axPerfRatingStock_x0.set_ylabel('Perf.Rating')
    axPerfRatingStock_x0.set_yticklabels(PerformanceRating, **{'rotation': 0})  
    axPerfRatingStock_x0.set_xlabel('StockOptionLevel')
    axPerfRatingStock_x0.set_xticklabels(StockOptionLevel, **{'rotation': 0})  
    
    axdividerJobRolePerf_x0 = make_axes_locatable(axPerfRatingStock_x0)
    axdividerJobRolePerf_x0.set_anchor((1,.5))
    caxPerfRatingStock_x0 = axdividerJobRolePerf_x0.append_axes('bottom', size = '20%', pad = '75%')
    colorbar(axPerfRatingStock_x0.get_children()[0], cax = caxPerfRatingStock_x0, orientation = 'horizontal',             **{'ticks': (0, 100)})
    caxPerfRatingStock_x0.xaxis.set_ticks_position('bottom')
    caxPerfRatingStock_x0.set_xlabel('Emp.Count by Perf.[%]')
    caxPerfRatingStock_x0.get_xaxis().set_label_position('bottom')
    
    axPerfRatingStock_x0.set_title('Perf.Rating vs StockOptionLevel (Attrition = 0)', size = 14,                               **{'horizontalalignment': 'right'})

    
    
    #[16,1]--------: Heatmap for the relation between the PerformanceRating & StockOptionLevel (Attrition = 1)
    axPerfRatingStock_x1 = plt.Subplot(figRemun, gsRemun[16, 1])
    axPerfRatingStock_x1.set_anchor((0,.5))
    figRemun.add_subplot(axPerfRatingStock_x1)
    axPerfRatingStock_x1.spines['left'].set_visible(False)
    axPerfRatingStock_x1.spines['right'].set_visible(True)

    df_JobRolePerfl1 = pd.DataFrame(np.zeros(shape = (len(PerformanceRating), len(StockOptionLevel))))
    row = 0
    for PR in PerformanceRating:
        col = 0
        for SOL in StockOptionLevel:
            df_JobRolePerfl1.iloc[row, col] = (df_in_1.loc[(df_in_1['PerformanceRating'] == PR) &                                                            (df_in_1['StockOptionLevel'] == SOL)]                                                              ['EmployeeNumber'].count()) /             (df_in_1.loc[(df_in_1['PerformanceRating'] == PR)]['EmployeeNumber'].count()) * 100 
                         
            col = col + 1
        row = row + 1

    sns.heatmap(df_JobRolePerfl1, cbar = False, annot = True, fmt = '.0f', cmap = 'YlGnBu',                vmin = 0, vmax = 100,                linewidths = .5, ax = axPerfRatingStock_x1)
    
    axPerfRatingStock_x1.set_aspect('equal')    
    axPerfRatingStock_x1.get_yaxis().set_label_position('right')
    axPerfRatingStock_x1.get_yaxis().tick_right()
    axPerfRatingStock_x1.invert_yaxis()
    axPerfRatingStock_x1.set_ylabel('Perf.Rating')
    axPerfRatingStock_x1.set_yticklabels(PerformanceRating, **{'rotation': 0})  
    axPerfRatingStock_x1.set_xlabel('StockOptionLevel')
    axPerfRatingStock_x1.set_xticklabels(StockOptionLevel, **{'rotation': 0})  
    

    axdividerJobRolePerf_x1 = make_axes_locatable(axPerfRatingStock_x1)
    axdividerJobRolePerf_x1.set_anchor((0,.5))
    caxPerfRatingStock_x1 = axdividerJobRolePerf_x1.append_axes('bottom', size = '20%', pad = '75%')
    colorbar(axPerfRatingStock_x1.get_children()[0], cax = caxPerfRatingStock_x1, orientation = 'horizontal',             **{'ticks': (0, 100)})
    caxPerfRatingStock_x1.xaxis.set_ticks_position('bottom')
    caxPerfRatingStock_x1.set_xlabel('Emp.Count by Perf.[%]')
    caxPerfRatingStock_x1.get_xaxis().set_label_position('bottom')
    
    axPerfRatingStock_x1.set_title('Perf.Rating vs StockOptionLevel (Attrition = 1)', size = 14,                                   **{'horizontalalignment': 'left'})

    # remove all _str variables
    df_in.drop(columns = df_in.columns[pd.Series(df_in.columns).str.contains('_str')], inplace = True)
    
    return figRemun


# In[10]:


def f_Rate_analysis(df_in):
    # Define indexes
    # JobLevel
    min_JobLevel = df_in['JobLevel'].min()
    max_JobLevel = df_in['JobLevel'].max()
    JobLevel = range(min_JobLevel, max_JobLevel + 1)
    JobLevel_list = [JobLevel[pos] for pos in range(0, len(JobLevel))]
    
    # PerformanceRating
    df_in['PerformanceRating_str'] = df_in.apply(lambda row: '#' + str(row['PerformanceRating']), axis = 1)
    
    
    # Create the plot
    df_in_0 = df_in[df_in['Attrition'] == 0]
    df_in_1 = df_in[df_in['Attrition'] == 1]

    figRates = plt.figure(figsize = (10, 50), dpi = 80, facecolor = 'w', edgecolor = 'k',                                 constrained_layout = False)
    gsRates = gridspec.GridSpec(nrows = 9, ncols = 2, hspace = .3, wspace = .08,                         height_ratios = [1, 1, 1, 1, 1, 1, 1, 1, .4],                                width_ratios = [1, 1],                          figure = figRates)
    
    
    #[0,0]--------: Relation between JobRole & HourlyRate (Attrition = 0)
    axJobRoleHRate_x0 = plt.Subplot(figRates, gsRates[0, 0])
    figRates.add_subplot(axJobRoleHRate_x0)

    sns.boxplot(x = 'HourlyRate', y = 'JobRole', data = df_in_0, order = JobRole[::-1],                color = '#718c6a',                boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                medianprops = medianprops, flierprops = flierprops, whis = whis,                ax = axJobRoleHRate_x0).set_title('JobRole vs HourlyRate (Attrition = 0)', size = 14)

    pos = range(len(JobRole))
    for tick, label in zip(pos, axJobRoleHRate_x0.get_yticklabels()):
        
        df_aux_0 = df_in_0[df_in_0['JobRole'] == JobRole[::-1][tick]]['HourlyRate']
        stats = f_stats_boxplot(df_aux_0)
        axJobRoleHRate_x0.text(stats[1] - 2.5, pos[tick], '{:.0f}'.format(stats[1]),                                       color = 'red', ha = 'center')
        axJobRoleHRate_x0.text(stats[2] - 2, pos[tick] + .45, '{:.0f}'.format(stats[2]),                                       color = 'black', ha = 'center')
        axJobRoleHRate_x0.text(stats[3] + 2, pos[tick], '{:.0f}'.format(stats[3]),                                       color = 'black', ha = 'center')
        axJobRoleHRate_x0.text(stats[4] + 2, pos[tick] + .45, '{:.0f}'.format(stats[4]),                                       color = 'black', ha = 'center')
        axJobRoleHRate_x0.text(stats[5] + 2.5, pos[tick], '{:.0f}'.format(stats[5]),                                       color = 'red', ha = 'center')    
    
    axJobRoleHRate_x0.set_xlabel('HourlyRate')
    axJobRoleHRate_x0.set_ylabel('JobRole')
    axJobRoleHRate_x0.set_xlim((24, 106))
    axJobRoleHRate_x0.get_xaxis().set_major_locator(mtick.MaxNLocator(integer = True))
    axJobRoleHRate_x0.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())


    
    #[0,1]--------: Relation between JobRole & HourlyRate (Attrition = 1)
    axJobRoleHRate_x1 = plt.Subplot(figRates, gsRates[0, 1])
    figRates.add_subplot(axJobRoleHRate_x1)
    axJobRoleHRate_x1.spines['left'].set_visible(False)
    axJobRoleHRate_x1.spines['right'].set_visible(True)

    sns.boxplot(x = 'HourlyRate', y = 'JobRole', data = df_in_1, order = JobRole[::-1],                color = '#718c6a',                boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                medianprops = medianprops, flierprops = flierprops, whis = whis,                ax = axJobRoleHRate_x1).set_title('JobRole vs HourlyRate (Attrition = 1)', size = 14)

    pos = range(len(JobRole))
    for tick, label in zip(pos, axJobRoleHRate_x1.get_yticklabels()):
        
        df_aux_1 = df_in_1[df_in_1['JobRole'] == JobRole[::-1][tick]]['HourlyRate']
        stats = f_stats_boxplot(df_aux_1)
        axJobRoleHRate_x1.text(stats[1] - 2.5, pos[tick], '{:.0f}'.format(stats[1]),                                       color = 'red', ha = 'center')
        axJobRoleHRate_x1.text(stats[2] - 2, pos[tick] + .45, '{:.0f}'.format(stats[2]),                                       color = 'black', ha = 'center')
        axJobRoleHRate_x1.text(stats[3] + 2, pos[tick], '{:.0f}'.format(stats[3]),                                       color = 'black', ha = 'center')
        axJobRoleHRate_x1.text(stats[4] + 2, pos[tick] + .45, '{:.0f}'.format(stats[4]),                                       color = 'black', ha = 'center')
        axJobRoleHRate_x1.text(stats[5] + 2.5, pos[tick], '{:.0f}'.format(stats[5]),                                       color = 'red', ha = 'center')    
    
    axJobRoleHRate_x1.set_xlabel('HourlyRate')
    axJobRoleHRate_x1.set_ylabel('JobRole')
    axJobRoleHRate_x1.get_yaxis().set_label_position("right")
    axJobRoleHRate_x1.get_yaxis().tick_right()
    axJobRoleHRate_x1.set_xlim((24, 106))
    axJobRoleHRate_x1.get_xaxis().set_major_locator(mtick.MaxNLocator(integer = True))
    axJobRoleHRate_x1.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
    
    
    
    #[1,0]--------: Relation between JobRole & DailyRate (Attrition = 0)
    axJobRoleDRate_x0 = plt.Subplot(figRates, gsRates[1, 0])
    figRates.add_subplot(axJobRoleDRate_x0)

    sns.boxplot(x = 'DailyRate', y = 'JobRole', data = df_in_0, order = JobRole[::-1],                color = '#ded797',                boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                medianprops = medianprops, flierprops = flierprops, whis = whis,                ax = axJobRoleDRate_x0).set_title('JobRole vs DailyRate (Attrition = 0)', size = 14)

    pos = range(len(JobRole))
    for tick, label in zip(pos, axJobRoleDRate_x0.get_yticklabels()):
        
        df_aux_0 = df_in_0[df_in_0['JobRole'] == JobRole[::-1][tick]]['DailyRate']
        stats = f_stats_boxplot(df_aux_0)
        axJobRoleDRate_x0.text(stats[1] - 80, pos[tick], '{:.0f}'.format(stats[1]),                                       color = 'red', ha = 'center')
        axJobRoleDRate_x0.text(stats[2] - 70, pos[tick] + .45, '{:.0f}'.format(stats[2]),                                       color = 'black', ha = 'center')
        axJobRoleDRate_x0.text(stats[3] + 70, pos[tick], '{:.0f}'.format(stats[3]),                                       color = 'black', ha = 'center')
        axJobRoleDRate_x0.text(stats[4] + 70, pos[tick] + .45, '{:.0f}'.format(stats[4]),                                       color = 'black', ha = 'center')
        axJobRoleDRate_x0.text(stats[5] + 80, pos[tick], '{:.0f}'.format(stats[5]),                                       color = 'red', ha = 'center')    
    
    axJobRoleDRate_x0.set_xlabel('DailyRate')
    axJobRoleDRate_x0.set_ylabel('JobRole')
    axJobRoleDRate_x0.set_xlim((-100, 1650))
    axJobRoleDRate_x0.get_xaxis().set_major_locator(mtick.MaxNLocator(integer = True))
    axJobRoleDRate_x0.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())


    
    #[1,1]--------: Relation between JobRole & DailyRate (Attrition = 1)
    axJobRoleDRate_x1 = plt.Subplot(figRates, gsRates[1, 1])
    figRates.add_subplot(axJobRoleDRate_x1)
    axJobRoleDRate_x1.spines['left'].set_visible(False)
    axJobRoleDRate_x1.spines['right'].set_visible(True)

    sns.boxplot(x = 'DailyRate', y = 'JobRole', data = df_in_1, order = JobRole[::-1],                color = '#ded797',                boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                medianprops = medianprops, flierprops = flierprops, whis = whis,                ax = axJobRoleDRate_x1).set_title('JobRole vs DailyRate (Attrition = 1)', size = 14)

    pos = range(len(JobRole))
    for tick, label in zip(pos, axJobRoleDRate_x1.get_yticklabels()):
        
        df_aux_1 = df_in_1[df_in_1['JobRole'] == JobRole[::-1][tick]]['DailyRate']
        stats = f_stats_boxplot(df_aux_1)
        axJobRoleDRate_x1.text(stats[1] - 80, pos[tick], '{:.0f}'.format(stats[1]),                                       color = 'red', ha = 'center')
        axJobRoleDRate_x1.text(stats[2] - 70, pos[tick] + .45, '{:.0f}'.format(stats[2]),                                       color = 'black', ha = 'center')
        axJobRoleDRate_x1.text(stats[3] + 70, pos[tick], '{:.0f}'.format(stats[3]),                                       color = 'black', ha = 'center')
        axJobRoleDRate_x1.text(stats[4] + 70, pos[tick] + .45, '{:.0f}'.format(stats[4]),                                       color = 'black', ha = 'center')
        axJobRoleDRate_x1.text(stats[5] + 80, pos[tick], '{:.0f}'.format(stats[5]),                                       color = 'red', ha = 'center')    
    
    axJobRoleDRate_x1.set_xlabel('DailyRate')
    axJobRoleDRate_x1.set_ylabel('JobRole')
    axJobRoleDRate_x1.get_yaxis().set_label_position("right")
    axJobRoleDRate_x1.get_yaxis().tick_right()
    axJobRoleDRate_x1.set_xlim((-100, 1650))
    axJobRoleDRate_x1.get_xaxis().set_major_locator(mtick.MaxNLocator(integer = True))
    axJobRoleDRate_x1.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())

    
    
    #[2,0]--------: Relation between JobRole & MonthlyRate (Attrition = 0)
    axJobRoleMRate_x0 = plt.Subplot(figRates, gsRates[2, 0])
    figRates.add_subplot(axJobRoleMRate_x0)

    sns.boxplot(x = 'MonthlyRate', y = 'JobRole', data = df_in_0, order = JobRole[::-1],                color = '#ffebcd',                boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                medianprops = medianprops, flierprops = flierprops, whis = whis,                ax = axJobRoleMRate_x0).set_title('JobRole vs MonthlyRate (Attrition = 0)', size = 14)

    pos = range(len(JobRole))
    for tick, label in zip(pos, axJobRoleMRate_x0.get_yticklabels()):
        
        df_aux_0 = df_in_0[df_in_0['JobRole'] == JobRole[::-1][tick]]['MonthlyRate']
        stats = f_stats_boxplot(df_aux_0)
        axJobRoleMRate_x0.text(stats[1] - 1800, pos[tick], '{:.1f}k'.format(stats[1] / 1000),                                       color = 'red', ha = 'center')
        axJobRoleMRate_x0.text(stats[2] - 1600, pos[tick] + .45, '{:.1f}k'.format(stats[2] / 1000),                                       color = 'black', ha = 'center')
        axJobRoleMRate_x0.text(stats[3] + 1600, pos[tick], '{:.1f}k'.format(stats[3] / 1000),                                       color = 'black', ha = 'center')
        axJobRoleMRate_x0.text(stats[4] + 1600, pos[tick] + .45, '{:.1f}k'.format(stats[4] / 1000),                                       color = 'black', ha = 'center')
        axJobRoleMRate_x0.text(stats[5] + 1800, pos[tick], '{:.1f}k'.format(stats[5] / 1000),                                       color = 'red', ha = 'center')    
    
    axJobRoleMRate_x0.set_xlabel('MonthlyRate')
    axJobRoleMRate_x0.set_ylabel('JobRole')
    axJobRoleMRate_x0.set_xlim((-2000, 31000))
    axJobRoleMRate_x0.get_xaxis().set_major_locator(mtick.MaxNLocator(integer = True))
    axJobRoleMRate_x0.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
    axJobRoleMRate_x0.get_xaxis().set_major_formatter(mtick.FuncFormatter(lambda x, pos: '{:.0f}k'.format(x / 1000)))

    
    
    #[2,1]--------: Relation between JobRole & MonthlyRate (Attrition = 1)
    axJobRoleMRate_x1 = plt.Subplot(figRates, gsRates[2, 1])
    figRates.add_subplot(axJobRoleMRate_x1)
    axJobRoleMRate_x1.spines['left'].set_visible(False)
    axJobRoleMRate_x1.spines['right'].set_visible(True)

    sns.boxplot(x = 'MonthlyRate', y = 'JobRole', data = df_in_1, order = JobRole[::-1],                color = '#ffebcd',                boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                medianprops = medianprops, flierprops = flierprops, whis = whis,                ax = axJobRoleMRate_x1).set_title('JobRole vs MonthlyRate (Attrition = 1)', size = 14)

    pos = range(len(JobRole))
    for tick, label in zip(pos, axJobRoleMRate_x1.get_yticklabels()):
        
        df_aux_1 = df_in_1[df_in_1['JobRole'] == JobRole[::-1][tick]]['MonthlyRate']
        stats = f_stats_boxplot(df_aux_1)
        axJobRoleMRate_x1.text(stats[1] - 1800, pos[tick], '{:.1f}k'.format(stats[1] / 1000),                                       color = 'red', ha = 'center')
        axJobRoleMRate_x1.text(stats[2] - 1600, pos[tick] + .45, '{:.1f}k'.format(stats[2] / 1000),                                       color = 'black', ha = 'center')
        axJobRoleMRate_x1.text(stats[3] + 1600, pos[tick], '{:.1f}k'.format(stats[3] / 1000),                                       color = 'black', ha = 'center')
        axJobRoleMRate_x1.text(stats[4] + 1600, pos[tick] + .45, '{:.1f}k'.format(stats[4] / 1000),                                       color = 'black', ha = 'center')
        axJobRoleMRate_x1.text(stats[5] + 1800, pos[tick], '{:.1f}k'.format(stats[5] / 1000),                                       color = 'red', ha = 'center')    
    
    axJobRoleMRate_x1.set_xlabel('MonthlyRate')
    axJobRoleMRate_x1.set_ylabel('JobRole')
    axJobRoleMRate_x1.get_yaxis().set_label_position("right")
    axJobRoleMRate_x1.get_yaxis().tick_right()
    axJobRoleMRate_x1.set_xlim((-2000, 31000))
    axJobRoleMRate_x1.get_xaxis().set_major_locator(mtick.MaxNLocator(integer = True))
    axJobRoleMRate_x1.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
    axJobRoleMRate_x1.get_xaxis().set_major_formatter(mtick.FuncFormatter(lambda x, pos: '{:.0f}k'.format(x / 1000)))

    
    
    #[3,0]--------: Relation between DailyRate & HourlyRate (Attrition = 0)
    axDRate_HRate_x0 = plt.Subplot(figRates, gsRates[3, 0])
    figRates.add_subplot(axDRate_HRate_x0)

    sns.scatterplot(x = 'HourlyRate', y = 'DailyRate', hue = 'JobLevel',                    palette = sns.color_palette("RdYlGn", len(JobLevel)),                    data = df_in_0,                    edgecolor = 'black', linewidth = .3, color = '#4ec5b0', marker = 'o', s = 20, alpha = .7,                    ax = axDRate_HRate_x0).set_title('DailyRate vs HourlyRate (Attrition = 0)', size = 14)
    
    axDRate_HRate_x0.legend(loc = "lower left")
    axDRate_HRate_x0.set_xlabel('HourlyRate')
    axDRate_HRate_x0.set_xlim((24, 106))
    axDRate_HRate_x0.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())

    axDRate_HRate_x0.set_ylabel('DailyRate')
    axDRate_HRate_x0.set_ylim((-100, 1650))
    axDRate_HRate_x0.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())
    
    # Plot line Y=x
    x_lim = axDRate_HRate_x0.get_xlim()[1]
    y_lim = axDRate_HRate_x0.get_ylim()[1]
    axDRate_HRate_x0.plot((0, min(x_lim, y_lim)), (0, min(x_lim, y_lim)),                                   color = 'red', alpha = 0.75, zorder = 0)
    axDRate_HRate_x0.text(60 + 10, 0, 'Y=x', ha = 'center', **dict(size = 10, color = 'red'))
    
    # Plot line Y=24x
    axDRate_HRate_x0.plot((24, y_lim / 24), (24 * 24, y_lim), color = 'black', alpha = 0.75, zorder = 0)
    axDRate_HRate_x0.text(65 + 7, 24 * 65, 'Y=24x', ha = 'center', **dict(size = 10, color = 'black'))


    
    #[3,1]--------: Relation between DailyRate & HourlyRate (Attrition = 1)
    axDRate_HRate_x1 = plt.Subplot(figRates, gsRates[3, 1])
    figRates.add_subplot(axDRate_HRate_x1)
    axDRate_HRate_x1.spines['left'].set_visible(False)
    axDRate_HRate_x1.spines['right'].set_visible(True)

    sns.scatterplot(x = 'HourlyRate', y = 'DailyRate', hue = 'JobLevel',                    palette = sns.color_palette("RdYlGn", len(JobLevel)),                    data = df_in_1,                    edgecolor = 'black', linewidth = .3, color = '#4ec5b0', marker = 'o', s = 20, alpha = .7,                    ax = axDRate_HRate_x1).set_title('DailyRate vs HourlyRate (Attrition = 1)', size = 14)

    axDRate_HRate_x1.legend(loc = "lower left")
    axDRate_HRate_x1.set_xlabel('HourlyRate')
    axDRate_HRate_x1.set_xlim((24, 106))
    axDRate_HRate_x1.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())

    axDRate_HRate_x1.set_ylabel('DailyRate')
    axDRate_HRate_x1.set_ylim((-100, 1650))
    axDRate_HRate_x1.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())
    axDRate_HRate_x1.get_yaxis().set_label_position("right")
    axDRate_HRate_x1.get_yaxis().tick_right()  
    
    # Plot line Y=x
    x_lim = axDRate_HRate_x1.get_xlim()[1]
    y_lim = axDRate_HRate_x1.get_ylim()[1]
    axDRate_HRate_x1.plot((0, min(x_lim, y_lim)), (0, min(x_lim, y_lim)),                                   color = 'red', alpha = 0.75, zorder = 0)
    axDRate_HRate_x1.text(60 + 10, 0, 'Y=x', ha = 'center', **dict(size = 10, color = 'red'))
    
    # Plot line Y=24x
    axDRate_HRate_x1.plot((24, y_lim / 24), (24 * 24, y_lim), color = 'black', alpha = 0.75, zorder = 0,                          label = "Y=24.X")
    axDRate_HRate_x1.text(65 + 7, 24 * 65, 'Y=24x', ha = 'center', **dict(size = 10, color = 'black'))
    

    
    #[4,0]--------: Relation between MonthlyRate & DailyRate (Attrition = 0)
    axMRate_DRate_x0 = plt.Subplot(figRates, gsRates[4, 0])
    figRates.add_subplot(axMRate_DRate_x0)

    sns.scatterplot(x = 'DailyRate', y = 'MonthlyRate', hue = 'JobLevel',                    palette = sns.color_palette("RdYlGn", len(JobLevel)),                    data = df_in_0,                    edgecolor = 'black', linewidth = .3, color = '#4ec5b0', marker = 'o', s = 20, alpha = .7,                    ax = axMRate_DRate_x0).set_title('MonthlyRate vs DailyRate (Attrition = 0)', size = 14)

    axMRate_DRate_x0.legend(loc = "lower left")
    axMRate_DRate_x0.set_xlabel('DailyRate')
    axMRate_DRate_x0.set_xlim((-100, 1650))
    axMRate_DRate_x0.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())

    axMRate_DRate_x0.set_ylabel('MonthlyRate')
    axMRate_DRate_x0.set_ylim((-2000, 31000))
    axMRate_DRate_x0.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())
    axMRate_DRate_x0.get_yaxis().set_major_formatter(mtick.FuncFormatter(lambda y, pos: '{:.0f}k'.format(y / 1000)))
    
    # Plot line Y=x
    x_lim = axMRate_DRate_x0.get_xlim()[1]
    y_lim = axMRate_DRate_x0.get_ylim()[1]
    axMRate_DRate_x0.plot((0, min(x_lim, y_lim)), (0, min(x_lim, y_lim)),                                   color = 'red', alpha = 0.75, zorder = 0)
    axMRate_DRate_x0.text(1250, 0, 'Y=x', ha = 'center', **dict(size = 10, color = 'red'))

    # Plot line Y=31x
    axMRate_DRate_x0.plot((0, y_lim / 31), (0, y_lim), color = 'black', alpha = 0.75, zorder = 0)
    axMRate_DRate_x0.text(900 + 150, 900 * 31, 'Y=31x', ha = 'center', **dict(size = 10, color = 'black'))
    
    

    #[4,1]--------: Relation between MonthlyRate & DailyRate (Attrition = 1)
    axMRate_DRate_x1 = plt.Subplot(figRates, gsRates[4, 1])
    figRates.add_subplot(axMRate_DRate_x1)
    axMRate_DRate_x1.spines['left'].set_visible(False)
    axMRate_DRate_x1.spines['right'].set_visible(True)

    sns.scatterplot(x = 'DailyRate', y = 'MonthlyRate', hue = 'JobLevel',                    palette = sns.color_palette("RdYlGn", len(JobLevel)),                    data = df_in_1,                    edgecolor = 'black', linewidth = .3, color = '#4ec5b0', marker = 'o', s = 20, alpha = .7,                    ax = axMRate_DRate_x1).set_title('MonthlyRate vs DailyRate (Attrition = 1)', size = 14)

    axMRate_DRate_x1.legend(loc = "lower left")
    axMRate_DRate_x1.set_xlabel('DailyRate')
    axMRate_DRate_x1.set_xlim((-100, 1650))
    axMRate_DRate_x1.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())

    axMRate_DRate_x1.set_ylabel('MonthlyRate')
    axMRate_DRate_x1.set_ylim((-2000, 31000))
    axMRate_DRate_x1.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())
    axMRate_DRate_x1.get_yaxis().set_label_position("right")
    axMRate_DRate_x1.get_yaxis().tick_right()  
    axMRate_DRate_x1.get_yaxis().set_major_formatter(mtick.FuncFormatter(lambda y, pos: '{:.0f}k'.format(y / 1000)))
    
    # Plot line Y=x
    x_lim = axMRate_DRate_x1.get_xlim()[1]
    y_lim = axMRate_DRate_x1.get_ylim()[1]
    axMRate_DRate_x1.plot((0, min(x_lim, y_lim)), (0, min(x_lim, y_lim)), color = 'red', alpha = 0.75, zorder = 0)
    axMRate_DRate_x1.text(1250, 0, 'Y=x', ha = 'center', **dict(size = 10, color = 'red'))   
    
    # Plot line Y=31x
    axMRate_DRate_x1.plot((0, y_lim / 31), (0, y_lim), color = 'black', alpha = 0.75, zorder = 0)
    axMRate_DRate_x1.text(900 + 150, 900 * 31, 'Y=31x', ha = 'center', **dict(size = 10, color = 'black'))
    
    
    
    #[5,0]--------: Relation between DRate_HRate_dv & MRate_DRate_dv (Attrition = 0)
    axCrossRatesCv_x0 = plt.Subplot(figRates, gsRates[5, 0])
    figRates.add_subplot(axCrossRatesCv_x0)

    sns.scatterplot(x = 'DRate_HRate_dv', y = 'MRate_DRate_dv', hue = 'JobLevel',                    palette = sns.color_palette("RdYlGn", len(JobLevel)),                    data = df_in_0,                    edgecolor = 'black', linewidth = .3, color = '#4ec5b0', marker = 'o', s = 20, alpha = .7,                    ax = axCrossRatesCv_x0).set_title('(Attrition = 0)', size = 14)

    axCrossRatesCv_x0.legend(loc = "upper right")
    axCrossRatesCv_x0.set_xlabel('DRate_HRate_dv')
    axCrossRatesCv_x0.set_xlim((-2, 52))
    axCrossRatesCv_x0.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())

    axCrossRatesCv_x0.set_ylabel('MRate_DRate_dv')
    axCrossRatesCv_x0.set_ylim((-10, 250))
    axCrossRatesCv_x0.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())


    #[5,1]--------: Relation between MonthlyRate & DailyRate (Attrition = 1)
    axCrossRatesCv_x1 = plt.Subplot(figRates, gsRates[5, 1])
    figRates.add_subplot(axCrossRatesCv_x1)
    axCrossRatesCv_x1.spines['left'].set_visible(False)
    axCrossRatesCv_x1.spines['right'].set_visible(True)

    sns.scatterplot(x = 'DRate_HRate_dv', y = 'MRate_DRate_dv', hue = 'JobLevel',                    palette = sns.color_palette("RdYlGn", len(JobLevel)),                    data = df_in_1,                    edgecolor = 'black', linewidth = .3, color = '#4ec5b0', marker = 'o', s = 20, alpha = .7,                    ax = axCrossRatesCv_x1).set_title('(Attrition = 1)', size = 14)

    axCrossRatesCv_x1.legend(loc = "upper right")
    axCrossRatesCv_x1.set_xlabel('DRate_HRate_dv')
    axCrossRatesCv_x1.set_xlim((-2, 52))
    axCrossRatesCv_x1.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())

    axCrossRatesCv_x1.set_ylabel('MRate_DRate_dv')
    axCrossRatesCv_x1.set_ylim((-10, 250))
    axCrossRatesCv_x1.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())
    axCrossRatesCv_x1.get_yaxis().set_label_position("right")
    axCrossRatesCv_x1.get_yaxis().tick_right()  
    
    
    
    #[6,0]--------: Relation between MonthlyRate & MonthlyIncome (Attrition = 0)
    axMIncomeMRate_x0 = plt.Subplot(figRates, gsRates[6, 0])
    figRates.add_subplot(axMIncomeMRate_x0)

    sns.scatterplot(x = 'MonthlyIncome', y = 'MonthlyRate', hue = 'JobLevel',                    palette = sns.color_palette("RdYlGn", len(JobLevel)),                    data = df_in_0,                    edgecolor = 'black', linewidth = .3, color = '#4ec5b0', marker = 'o', s = 20, alpha = .7,                    ax = axMIncomeMRate_x0).set_title('MonthlyRate vs MonthlyIncome (Attrition = 0)', size = 14)

    axMIncomeMRate_x0.legend(loc = "upper right")
    axMIncomeMRate_x0.set_xlabel('MonthlyIncome')
    axMIncomeMRate_x0.set_xlim((-1000, 22000))
    axMIncomeMRate_x0.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
    axMIncomeMRate_x0.get_xaxis().set_major_formatter(mtick.FuncFormatter(lambda x, pos: '{:.0f}k'.format(x / 1000)))

    axMIncomeMRate_x0.set_ylabel('MonthlyRate')
    axMIncomeMRate_x0.set_ylim((-2000, 31000))
    axMIncomeMRate_x0.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())
    axMIncomeMRate_x0.get_yaxis().set_major_formatter(mtick.FuncFormatter(lambda y, pos: '{:.0f}k'.format(y / 1000)))

    # Plot line Y=x
    x_lim = axMIncomeMRate_x0.get_xlim()[1]
    y_lim = axMIncomeMRate_x0.get_ylim()[1]
    axMIncomeMRate_x0.plot((0, min(x_lim, y_lim)), (0, min(x_lim, y_lim)), color = 'blue', alpha = 0.75, zorder = 0)
    axMIncomeMRate_x0.text(2000, 0, 'Y=x', ha = 'center', **dict(size = 10, color = 'blue'))   
    
    

    #[6,1]--------: Relation between MonthlyRate & DailyRate (Attrition = 1)
    axMIncomeMRate_x1 = plt.Subplot(figRates, gsRates[6, 1])
    figRates.add_subplot(axMIncomeMRate_x1)
    axMIncomeMRate_x1.spines['left'].set_visible(False)
    axMIncomeMRate_x1.spines['right'].set_visible(True)

    sns.scatterplot(x = 'MonthlyIncome', y = 'MonthlyRate', hue = 'JobLevel',                    palette = sns.color_palette("RdYlGn", len(JobLevel)),                    data = df_in_1,                    edgecolor = 'black', linewidth = .3, color = '#4ec5b0', marker = 'o', s = 20, alpha = .7,                    ax = axMIncomeMRate_x1).set_title('MonthlyRate vs MonthlyIncome (Attrition = 1)', size = 14)

    axMIncomeMRate_x1.legend(loc = "upper right")
    axMIncomeMRate_x1.set_xlabel('MonthlyIncome')
    axMIncomeMRate_x1.set_xlim((-1000, 22000))
    axMIncomeMRate_x1.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
    axMIncomeMRate_x1.get_xaxis().set_major_formatter(mtick.FuncFormatter(lambda x, pos: '{:.0f}k'.format(x / 1000)))

    axMIncomeMRate_x1.set_ylabel('MonthlyRate')
    axMIncomeMRate_x1.set_ylim((-2000, 31000))
    axMIncomeMRate_x1.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())
    axMIncomeMRate_x1.get_yaxis().set_major_formatter(mtick.FuncFormatter(lambda y, pos: '{:.0f}k'.format(y / 1000)))
    axMIncomeMRate_x1.get_yaxis().set_label_position("right")
    axMIncomeMRate_x1.get_yaxis().tick_right() 
    
    # Plot line Y=x
    x_lim = axMIncomeMRate_x1.get_xlim()[1]
    y_lim = axMIncomeMRate_x1.get_ylim()[1]
    axMIncomeMRate_x1.plot((0, min(x_lim, y_lim)), (0, min(x_lim, y_lim)), color = 'blue', alpha = 0.75, zorder = 0)
    axMIncomeMRate_x1.text(2000, 0, 'Y=x', ha = 'center', **dict(size = 10, color = 'blue'))   
    
    

    #[7,0]--------: Relation between JobRole & MonthlyProfit_dv (Attrition = 0)
    axJobRoleProfit_x0 = plt.Subplot(figRates, gsRates[7, 0])
    figRates.add_subplot(axJobRoleProfit_x0)

    sns.boxplot(x = 'MonthlyProfit_dv', y = 'JobRole', data = df_in_0, order = JobRole[::-1],                color = '#4ec5b0',                boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                medianprops = medianprops, flierprops = flierprops, whis = whis,                ax = axJobRoleProfit_x0).set_title('JobRole vs MonthlyProfit_dv (Attrition = 0)', size = 14)

    pos = range(len(JobRole))
    for tick, label in zip(pos, axJobRoleProfit_x0.get_yticklabels()):
        
        df_aux_0 = df_in_0[df_in_0['JobRole'] == JobRole[::-1][tick]]['MonthlyProfit_dv']
        stats = f_stats_boxplot(df_aux_0)
        axJobRoleProfit_x0.text(stats[1] - 2600, pos[tick] - .25, '{:.1f}k'.format(stats[1] / 1000),                                       color = 'red', ha = 'center')
        axJobRoleProfit_x0.text(stats[3] + 2300, pos[tick], '{:.1f}k'.format(stats[3] / 1000),                                       color = 'black', ha = 'center')
        axJobRoleProfit_x0.text(stats[5] + 2600, pos[tick] - .25, '{:.1f}k'.format(stats[5] / 1000),                                       color = 'red', ha = 'center')    
    
    axJobRoleProfit_x0.set_xlabel('MonthlyProfit_dv')
    axJobRoleProfit_x0.set_ylabel('JobRole')
    axJobRoleProfit_x0.set_xlim((-25000, 32000))
    axJobRoleProfit_x0.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
    axJobRoleProfit_x0.get_xaxis().set_major_formatter(mtick.FuncFormatter(lambda x, pos: '{:.0f}k'.format(x / 1000))) 
    
    # Plot line x=0
    y_0 = axJobRoleProfit_x0.get_ylim()[0]
    y_lim = axJobRoleProfit_x0.get_ylim()[1]
    axJobRoleProfit_x0.plot((0, 0), (y_0, y_lim), color = 'blue', alpha = 0.75, linestyle = '--', zorder = 0)
    
    
    
    #[7,1]--------: Relation between JobRole & MonthlyProfit_dv (Attrition = 1)
    axJobRoleProfit_x1 = plt.Subplot(figRates, gsRates[7, 1])
    figRates.add_subplot(axJobRoleProfit_x1)
    axJobRoleProfit_x1.spines['left'].set_visible(False)
    axJobRoleProfit_x1.spines['right'].set_visible(True)

    sns.boxplot(x = 'MonthlyProfit_dv', y = 'JobRole', data = df_in_1, order = JobRole[::-1],                color = '#4ec5b0',                boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                medianprops = medianprops, flierprops = flierprops, whis = whis,                ax = axJobRoleProfit_x1).set_title('JobRole vs MonthlyProfit_dv (Attrition = 1)', size = 14)

    pos = range(len(JobRole))
    for tick, label in zip(pos, axJobRoleProfit_x1.get_yticklabels()):
        
        df_aux_1 = df_in_1[df_in_1['JobRole'] == JobRole[::-1][tick]]['MonthlyProfit_dv']
        stats = f_stats_boxplot(df_aux_1)
        axJobRoleProfit_x1.text(stats[1] - 2600, pos[tick] - .25, '{:.1f}k'.format(stats[1] / 1000),                                       color = 'red', ha = 'center')
        axJobRoleProfit_x1.text(stats[3] + 2300, pos[tick], '{:.1f}k'.format(stats[3] / 1000),                                       color = 'black', ha = 'center')
        axJobRoleProfit_x1.text(stats[5] + 2600, pos[tick] - .25, '{:.1f}k'.format(stats[5] / 1000),                                       color = 'red', ha = 'center')    
    
    axJobRoleProfit_x1.set_xlabel('MonthlyProfit_dv')
    axJobRoleProfit_x1.set_ylabel('JobRole')
    axJobRoleProfit_x1.get_yaxis().set_label_position("right")
    axJobRoleProfit_x1.get_yaxis().tick_right()
    axJobRoleProfit_x1.set_xlim((-25000, 32000))
    axJobRoleProfit_x1.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
    axJobRoleProfit_x1.get_xaxis().set_major_formatter(mtick.FuncFormatter(lambda x, pos: '{:.0f}k'.format(x / 1000)))

    # Plot line x=0
    y_0 = axJobRoleProfit_x1.get_ylim()[0]
    y_lim = axJobRoleProfit_x1.get_ylim()[1]
    axJobRoleProfit_x1.plot((0, 0), (y_0, y_lim), color = 'blue', alpha = 0.75, linestyle = '--', zorder = 0)

    
    
    #[8,0]--------: Relation between PerformanceRating & MonthlyProfit_dv (Attrition = 0)
    axProfitPerf_x0 = plt.Subplot(figRates, gsRates[8, 0])
    figRates.add_subplot(axProfitPerf_x0)

    sns.boxplot(x = 'MonthlyProfit_dv', y = 'PerformanceRating_str', data = df_in_0, palette = palette_relscale_str,                order = sorted(df_in_0['PerformanceRating_str'].unique()),                color = '#4ec5b0',                boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                medianprops = medianprops, flierprops = flierprops, whis = whis,                ax = axProfitPerf_x0).set_title('Perf.Rating vs M. Profit (Attrition = 0)', size = 14)
    
    pos = range(len(df_in_0['PerformanceRating_str'].unique()))
    for tick, label in zip(pos, axProfitPerf_x0.get_yticklabels()):
        df_aux_0 = df_in_0[df_in_0['PerformanceRating_str'] == label.get_text()]['MonthlyProfit_dv']
        stats = f_stats_boxplot(df_aux_0)
        axProfitPerf_x0.text(stats[1] - 2600, pos[tick], '{:.1f}k'.format(stats[1] / 1000),                                       color = 'red', ha = 'center')
        axProfitPerf_x0.text(stats[2] - 2600, pos[tick] - .3, '{:.1f}k'.format(stats[2] / 1000),                                       color = 'black', ha = 'center')
        axProfitPerf_x0.text(stats[3] + 2300, pos[tick], '{:.1f}k'.format(stats[3] / 1000),                                       color = 'black', ha = 'center')
        axProfitPerf_x0.text(stats[4] + 2600, pos[tick] - .3, '{:.1f}k'.format(stats[4] / 1000),                                       color = 'black', ha = 'center')
        axProfitPerf_x0.text(stats[5] + 2600, pos[tick], '{:.1f}k'.format(stats[5] / 1000),                                       color = 'red', ha = 'center')    

    axProfitPerf_x0.set_xlabel('MonthlyProfit_dv')
    axProfitPerf_x0.set_ylabel('PerformanceRating')
    axProfitPerf_x0.invert_yaxis()
    axProfitPerf_x0.set_xlim((-25000, 32000))
    axProfitPerf_x0.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
    axProfitPerf_x0.get_xaxis().set_major_formatter(mtick.FuncFormatter(lambda x, pos: '{:.0f}k'.format(x / 1000))) 
    
    # Plot line x=0
    y_0 = axProfitPerf_x0.get_ylim()[0]
    y_lim = axProfitPerf_x0.get_ylim()[1]
    axProfitPerf_x0.plot((0, 0), (y_0, y_lim), color = 'blue', alpha = 0.75, linestyle = '--', zorder = 0)
    
    
    
    #[8,1]--------: Relation between PerformanceRating & MonthlyProfit_dv (Attrition = 1)
    axProfitPerf_x1 = plt.Subplot(figRates, gsRates[8, 1])
    figRates.add_subplot(axProfitPerf_x1)
    axProfitPerf_x1.spines['left'].set_visible(False)
    axProfitPerf_x1.spines['right'].set_visible(True)

    sns.boxplot(x = 'MonthlyProfit_dv', y = 'PerformanceRating_str', data = df_in_1, palette = palette_relscale_str,                order = sorted(df_in_1['PerformanceRating_str'].unique()),                color = '#4ec5b0',                boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                medianprops = medianprops, flierprops = flierprops, whis = whis,                ax = axProfitPerf_x1).set_title('Perf.Rating vs M.Profit (Attrition = 1)', size = 14)
    
    pos = range(len(df_in_1['PerformanceRating_str'].unique()))
    for tick, label in zip(pos, axProfitPerf_x1.get_yticklabels()):
        df_aux_1 = df_in_1[df_in_1['PerformanceRating_str'] == label.get_text()]['MonthlyProfit_dv']
        stats = f_stats_boxplot(df_aux_1)
        axProfitPerf_x1.text(stats[1] - 2600, pos[tick], '{:.1f}k'.format(stats[1] / 1000),                                       color = 'red', ha = 'center')
        axProfitPerf_x1.text(stats[2] - 2600, pos[tick] - .3, '{:.1f}k'.format(stats[2] / 1000),                                       color = 'black', ha = 'center')
        axProfitPerf_x1.text(stats[3] + 2300, pos[tick], '{:.1f}k'.format(stats[3] / 1000),                                       color = 'black', ha = 'center')
        axProfitPerf_x1.text(stats[4] + 2600, pos[tick] - .3, '{:.1f}k'.format(stats[4] / 1000),                                       color = 'black', ha = 'center')
        axProfitPerf_x1.text(stats[5] + 2600, pos[tick], '{:.1f}k'.format(stats[5] / 1000),                                       color = 'red', ha = 'center')    

    axProfitPerf_x1.set_xlabel('MonthlyProfit_dv')
    axProfitPerf_x1.set_ylabel('PerformanceRating')
    axProfitPerf_x1.get_yaxis().set_label_position("right")
    axProfitPerf_x1.get_yaxis().tick_right()
    axProfitPerf_x1.invert_yaxis()
    axProfitPerf_x1.set_xlim((-25000, 32000))
    axProfitPerf_x1.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
    axProfitPerf_x1.get_xaxis().set_major_formatter(mtick.FuncFormatter(lambda x, pos: '{:.0f}k'.format(x / 1000)))
    
    # Plot line x=0
    y_0 = axProfitPerf_x1.get_ylim()[0]
    y_lim = axProfitPerf_x1.get_ylim()[1]
    axProfitPerf_x1.plot((0, 0), (y_0, y_lim), color = 'blue', alpha = 0.75, linestyle = '--', zorder = 0)
    
    # remove all _str variables
    df_in.drop(columns = df_in.columns[pd.Series(df_in.columns).str.contains('_str')], inplace = True)

    return figRates


# In[11]:


def f_hist_box(df_in, varlist_in, n_columns_in, color_in):
    """This function performs the combined boxplot + histogram of a set of variables from an input dataframe."""
    
    n_plots = len(varlist_in)
    n_rows = math.ceil(n_plots / n_columns_in)  
    
    fig_hist_box = plt.figure(figsize = (12, 5 * n_rows), dpi = 80, facecolor = 'w', edgecolor = 'k',                             constrained_layout = False)

    gs_hist_box = gridspec.GridSpec(nrows = n_rows, ncols = n_columns_in, hspace = .25, wspace = .25,                            height_ratios = [1 for i in range(n_rows)],                            width_ratios = [1 for i in range(n_columns_in)], figure = fig_hist_box)
    
    row = 0
    column = 0
    for index, item in enumerate(varlist_in):
        sgs_hist_box_ij = gridspec.GridSpecFromSubplotSpec(nrows = 2, ncols = 1,                           subplot_spec = gs_hist_box[row, column], hspace = .05, wspace = .2,                           height_ratios = (.2, .8))

        # boxplot
        ax_hist_box_0j = plt.Subplot(fig_hist_box, sgs_hist_box_ij[0, 0])
        fig_hist_box.add_subplot(ax_hist_box_0j , sharex = True)
        ax_hist_box_0j.tick_params(left = False, bottom = False, labelleft = False)
        ax_hist_box_0j.spines['left'].set_visible(False)
        ax_hist_box_0j.spines['bottom'].set_visible(False)
        

        sns.boxplot(eval('df_in[' + '\'' + item  + '\'' + ']'), color = 'white',                    boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                    medianprops = medianprops, flierprops = flierprops, whis = whis,                    ax = ax_hist_box_0j ).set_title(item, size = 12)
        ax_hist_box_0j.tick_params(labelbottom = False) 
        ax_hist_box_0j.set_xlabel('')

        stats = f_stats_boxplot(eval('df_in[' + '\'' + item  + '\'' + ']'))
        if(stats[3] > 3000):
            ax_hist_box_0j.text(stats[1], .4, '{:.0f}k'.format(stats[1] / 1000), color = 'red', ha = 'center')
            ax_hist_box_0j.text(stats[3], .1, '{:.0f}k'.format(stats[3] / 1000), color = 'black', ha = 'center')
            ax_hist_box_0j.text(stats[5], -.3, '{:.0f}k'.format(stats[5] / 1000), color = 'red', ha = 'center')
        else:
            ax_hist_box_0j.text(stats[1], .4, '{:.1f}'.format(stats[1]), color = 'red', ha = 'center')
            ax_hist_box_0j.text(stats[3], .1, '{:.1f}'.format(stats[3]), color = 'black', ha = 'center')
            ax_hist_box_0j.text(stats[5], -.3, '{:.1f}'.format(stats[5]), color = 'red', ha = 'center')


        # histogram
        ax_hist_box_1j  = plt.Subplot(fig_hist_box, sgs_hist_box_ij[1, 0])
        fig_hist_box.add_subplot(ax_hist_box_1j)

        sns.distplot(eval('df_in[' + '\'' + item  + '\'' + ']'), kde = False, rug = False, bins = 15,                         color = color_in, hist_kws = {'alpha': 1}, ax = ax_hist_box_1j)
        
        ax_hist_box_1j .set_xlabel('')
        ax_hist_box_1j.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())  
        if(column == 0):
            ax_hist_box_1j.set_ylabel('Employee Count') 
        x_pos = ax_hist_box_1j.get_xlim()[0] + (ax_hist_box_1j.get_xlim()[1] - ax_hist_box_1j.get_xlim()[0])*.5
        y_pos = ax_hist_box_1j.get_ylim()[0] + (ax_hist_box_1j.get_ylim()[1] - ax_hist_box_1j.get_ylim()[0])*.9
        ax_hist_box_1j.text(x_pos, y_pos, 'skewness: ' + '{:.2f}'.format(stats[6]), color = 'red',                            ha = 'center', fontsize = 10)
        
        ax_hist_box_0j.get_xaxis().set_major_locator(mtick.MaxNLocator(integer = True, nbins = 6))
        ax_hist_box_1j.get_xaxis().set_major_locator(mtick.MaxNLocator(integer = True, nbins = 6))
        if(stats[3] > 3000):
            ax_hist_box_0j.get_xaxis().set_major_formatter(mtick.FuncFormatter(lambda x,                                                                               pos: '{:.0f}k'.format(x / 1000)))
            ax_hist_box_1j.get_xaxis().set_major_formatter(mtick.FuncFormatter(lambda x,                                                                               pos: '{:.0f}k'.format(x / 1000)))

        # update the position of the plot inside the grid
        if((column + 1) == n_columns_in):
            row = row + 1
            column = 0
        else:
            column = column + 1
            
    return fig_hist_box


# In[12]:


def f_cluster_analysis (vars_in, df_in, cluster_in, color_palette_in):
    axCluster = sns.pairplot(df_in, vars = vars_in, diag_kind = 'kde', diag_kws = dict(shade = True),                              hue = cluster_in, palette = color_palette_in)

    return axCluster


# In[13]:


def f_pielabels(pct, allvalues):
    """This function determines the labels to be used on a pie chart (%, value)."""
    
    absolute = int(pct / 100 * np.sum(allvalues))
    return '{:.1f}%\n({:.0f})'.format(pct, absolute)


def f_cluster_interpretation (n_clusters_in, df_in, cluster_in, bins_in = 7, lim_inf_in = .1,                              lim_sup_in = .9, ntickbins_in = 5):
    nrows = 32
    
    # Define indexes
    # Education
    min_Education = df_in['Education'].min()
    max_Education = df_in['Education'].max()
    Education = range(min_Education, max_Education + 1)
    
    # Relation between the JobLevel & JobRole
    min_JobLevel = df_in['JobLevel'].min()
    max_JobLevel = df_in['JobLevel'].max()
    JobLevel = range(min_JobLevel, max_JobLevel + 1)
    
    # JobInvolvement
    df_in['JobInvolvement_str'] = df_in.apply(lambda row: '#' + str(row['JobInvolvement']), axis = 1)
    
    # JobSatisfaction
    df_in['JobSatisfaction_str'] = df_in.apply(lambda row: '#' + str(row['JobSatisfaction']), axis = 1)    
    
    # PerformanceRating 
    df_in['PerformanceRating_str'] = df_in.apply(lambda row: '#' + str(row['PerformanceRating']), axis = 1)
    
    # EnvironmentSatisfaction
    df_in['EnvironmentSatisfaction_str'] = df_in.apply(lambda row: '#' + str(row['EnvironmentSatisfaction']), axis = 1)     
    
    # RelationshipSatisfaction
    df_in['RelationshipSatisfaction_str'] = df_in.apply(lambda row: '#' + str(row['RelationshipSatisfaction']),                                                        axis = 1)   
    
    # StockOptionLevel
    df_in['StockOptionLevel_str'] = df_in.apply(lambda row: '#' + str(row['StockOptionLevel']),                                                        axis = 1)     
    
    

    # Create the plot
    figClusterInt = plt.figure(figsize = (16 / 6 * n_clusters_in, nrows * 4.4), dpi = 80, facecolor = 'w',                                edgecolor = 'k', constrained_layout = False)
    gsClusterInt = gridspec.GridSpec(nrows = nrows, ncols = n_clusters_in, hspace = .4, wspace = .3,                        height_ratios = [1, 1, 1, 1, 1, 1.25, 1, 1.25, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,                                         1, 1, 1, 1, 1, 1, 1, 1, 1],                                     width_ratios = [1 for i in range(n_clusters_in)],                                       figure = figClusterInt)
    
    DepVar_colorsdict = {0:'#0F9149', 1:'#911A0F'}
    
    pos_cluster = []
    if(n_clusters_in ==2 ):
        if(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == 0][' + '\'' + 'EmployeeNumber' + '\'' + '].count()')>          eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == 1][' + '\'' + 'EmployeeNumber' + '\'' + '].count()')):
            pos_cluster.append(0)
            pos_cluster.append(1)
        else:
            pos_cluster.append(1)
            pos_cluster.append(0)            

    for c in range(n_clusters_in):
        if(n_clusters_in == 2):
            cluster = pos_cluster[c]
        else:
            cluster = c

        # -------- Figs: PieChart DepVar
        axClusterInt_DepVari = plt.Subplot(figClusterInt, gsClusterInt[0, cluster])
        figClusterInt.add_subplot(axClusterInt_DepVari)

        DepVari = pd.Series([eval('df_in[(df_in[' + '\'' + cluster_in + '\'' + '] == cluster) & (df_in[' +              '\'' + 'Attrition_flag' + '\'' + '] == 0)][' + '\'' + 'EmployeeNumber' + '\'' + '].count()'),                           eval('df_in[(df_in[' + '\'' + cluster_in + '\'' + '] == cluster) & (df_in[' +              '\'' + 'Attrition_flag' + '\'' + '] == 1)][' + '\'' + 'EmployeeNumber' + '\'' + '].count()')],                           name = 'Attrition')
        
        DepVar_colors = [DepVar_colorsdict[idx] for idx in DepVari.index]
        wedges, texts, autotexts = axClusterInt_DepVari.pie(DepVari, radius = .8, startangle = 90,                                            autopct = lambda pct: f_pielabels(pct, DepVari),                                            pctdistance = 1.4, textprops = dict(color = 'black'),                                            colors = DepVar_colors, wedgeprops = {'alpha': 1})
        axClusterInt_DepVari.legend(wedges, DepVari.index.tolist(), title = 'Attrition', loc = 'lower center',                    bbox_to_anchor = (0, -.5, 1, 1), prop = dict(size = 12))
        
        # add cluster record counts
        cluster_nrecords = eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'EmployeeNumber' + '\'' + '].count()')
        axClusterInt_DepVari.set_title('Cluster ' + str(cluster) + '\n' +                                        'n = {:d} ({:.0f}%)'.format(cluster_nrecords,                                                               cluster_nrecords / df_in.shape[0] * 100), size = 14)

        
        # -------- Figs:BoxPlot + Histogram for Age
        sgsClusterInt_Agei = gridspec.GridSpecFromSubplotSpec(2, 1,                                               subplot_spec = gsClusterInt[1, cluster],                                                            hspace = .05, wspace = .2, height_ratios = (.2, .8))
        
        percent = eval('df_in[df_in[' + '\'' + cluster_in +                     '\'' + '] == cluster][' + '\'' + 'Age' + '\'' + '].sum()') /  df_in['Age'].sum() * 100

        # boxplot
        axClusterInt_Agei_00 = plt.Subplot(figClusterInt, sgsClusterInt_Agei[0, 0])
        figClusterInt.add_subplot(axClusterInt_Agei_00, sharex = True)
        axClusterInt_Agei_00.tick_params(left = False, bottom = False, labelleft = False)
        axClusterInt_Agei_00.spines['left'].set_visible(False)
        axClusterInt_Agei_00.spines['bottom'].set_visible(False)
        axClusterInt_Agei_00.spines['top'].set_visible(False)
        
        if(percent > 0):
            sns.boxplot(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'Age' + '\'' + ']'),                        color = 'white',                        boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                        medianprops = medianprops, flierprops = flierprops, whis = whis,                        ax = axClusterInt_Agei_00)
            
            axClusterInt_Agei_00.tick_params(labelbottom = False) 
            
            axClusterInt_Agei_00.set_xlabel('')
            stats = f_stats_boxplot(eval('df_in[df_in[' + '\'' + cluster_in +                     '\'' + '] == cluster][' + '\'' + 'Age' + '\'' + ']'))
            if(stats[3] > 2000):
                axClusterInt_Agei_00.text(stats[1], -.35, '{:.0f}k'.format(stats[1] / 1000),                                                   color = 'red', ha = 'center')
                axClusterInt_Agei_00.text(stats[3], .1, '{:.0f}k'.format(stats[3] / 1000),                                                   color = 'black', ha = 'center')
                axClusterInt_Agei_00.text(stats[5], -.35, '{:.0f}k'.format(stats[5] / 1000),                                                   color = 'red', ha = 'center')
            else:
                axClusterInt_Agei_00.text(stats[1], -.35, '{:.1f}'.format(stats[1]), color = 'red', ha = 'center')
                axClusterInt_Agei_00.text(stats[3], .1, '{:.1f}'.format(stats[3]), color = 'black', ha = 'center')
                axClusterInt_Agei_00.text(stats[5], -.35, '{:.1f}'.format(stats[5]), color = 'red', ha = 'center')                

            if(stats[3] > lim_sup_in * max(df_in['Age'])):
                axClusterInt_Agei_00.set_xlim((min(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'Age' + '\'' + ']')),                                                 max(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'Age' + '\'' + ']'))))
                axClusterInt_Agei_00.set_title('({:.1f}%)'.format(percent), size = 10, color = 'red')
            else:
                axClusterInt_Agei_00.set_xlim((0, max(df_in['Age'])))
                axClusterInt_Agei_00.set_title('({:.1f}%)'.format(percent), size = 10)
            #axClusterInt_Agei_00.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
            axClusterInt_Agei_00.get_xaxis().set_major_locator(mtick.MaxNLocator(integer = True,                                                                                nbins = ntickbins_in - 1)) 

        # histogram
        axClusterInt_Agei_10 = plt.Subplot(figClusterInt, sgsClusterInt_Agei[1, 0])
        figClusterInt.add_subplot(axClusterInt_Agei_10)
                                  
        if(percent > 0):    
            if((stats[3] > lim_sup_in * max(df_in['Age'])) or (stats[5] <  lim_inf_in * max(df_in['Age']))):
                axClusterInt_Agei_10.set_xlim((min(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'Age' + '\'' + ']')),                                                 max(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'Age' + '\'' + ']'))))
                axClusterInt_Agei_10.tick_params(axis = 'x', colors = 'red')
            else:
                axClusterInt_Agei_10.set_xlim((0, max(df_in['Age'])))

            sns.distplot(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'Age' + '\'' + ']'), kde = False,                         rug = False, bins = bins_in,                         color = '#03366f', hist_kws = {'alpha': .85},                         ax = axClusterInt_Agei_10)
                                  
            axClusterInt_Agei_10.set_xlabel('Age')
            axClusterInt_Agei_10.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
            axClusterInt_Agei_10.get_xaxis().set_major_locator(mtick.MaxNLocator(integer = True,                                                                                 nbins = ntickbins_in - 1)) 
            
            if(stats[3] > 2000):
                axClusterInt_Agei_10.get_xaxis().set_major_formatter(mtick.FuncFormatter(lambda x,                      pos: '{:.0f}k'.format(x / 1000)))            

            if(cluster == 0):
                axClusterInt_Agei_10.set_ylabel('Emp. Count')
            else:
                axClusterInt_Agei_10.set_ylabel('')
            axClusterInt_Agei_10.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())
            axClusterInt_Agei_10.get_yaxis().set_major_locator(mtick.MaxNLocator(integer = True)) 
        

        # -------- Figs: Barplot for Gender
        axClusterInt_Genderi = plt.Subplot(figClusterInt, gsClusterInt[2, cluster])
        figClusterInt.add_subplot(axClusterInt_Genderi)

        gender_values = eval('df_in[df_in[' + '\'' + cluster_in +                                                        '\'' + '] == cluster].groupby([' + '\'' + 'Gender' +                                                        '\'' + '])[' + '\'' + 'EmployeeNumber' + '\'' +'].count()')
        
        genderlevels = pd.Series(gender_values, index = genderlevels_index, name = 'GenderLevels')
        palette = sns.color_palette(genderlevels_colors)
        
        sns.barplot(x = genderlevels.index, y = genderlevels, palette = palette, ax = axClusterInt_Genderi)
        if(cluster == 0):
            axClusterInt_Genderi.set_ylabel('Emp. Count')
        else:
            axClusterInt_Genderi.set_ylabel('')
        axClusterInt_Genderi.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())
        axClusterInt_Genderi.get_yaxis().set_major_locator(mtick.MaxNLocator(integer = True)) 
        plt.setp(axClusterInt_Genderi.get_xticklabels(), **{'rotation': 20})
        pos = 0
        total = genderlevels.sum()
        for index, value in genderlevels.iteritems():
            if(value > 0):
                axClusterInt_Genderi.text(pos, value, '{:.0f}%'.format(value / total * 100),                                             color = 'black', ha = 'center')
            pos = pos + 1      
        
        
        
        # -------- Figs: Barplot for MaritalStatus
        axClusterInt_MaritalStatusi = plt.Subplot(figClusterInt, gsClusterInt[3, cluster])
        figClusterInt.add_subplot(axClusterInt_MaritalStatusi)

        mstatuslevels = eval('df_in[df_in[' + '\'' + cluster_in +                                                        '\'' + '] == cluster].groupby([' + '\'' + 'MaritalStatus' +                                                        '\'' + '])[' + '\'' + 'EmployeeNumber' + '\'' +'].count()')
        
        mstatuslevelslevels = pd.Series(mstatuslevels, index = mstatuslevels_index, name = 'MaritalStatusLevels')
        palette = sns.color_palette(mstatuslevels_colors)
        
        sns.barplot(x = mstatuslevelslevels.index, y = mstatuslevelslevels, palette = palette,                    ax = axClusterInt_MaritalStatusi)
        if(cluster == 0):
            axClusterInt_MaritalStatusi.set_ylabel('Emp. Count')
        else:
            axClusterInt_MaritalStatusi.set_ylabel('')
        axClusterInt_MaritalStatusi.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())
        axClusterInt_MaritalStatusi.get_yaxis().set_major_locator(mtick.MaxNLocator(integer = True)) 
        plt.setp(axClusterInt_MaritalStatusi.get_xticklabels(), **{'rotation': 20})
        pos = 0
        total = mstatuslevelslevels.sum()
        for index, value in mstatuslevelslevels.iteritems():
            if(value > 0):
                axClusterInt_MaritalStatusi.text(pos, value, '{:.0f}%'.format(value / total * 100),                                             color = 'black', ha = 'center')
            pos = pos + 1 

            
        
        # -------- Figs:BoxPlot + Histogram for DistanceFromHome
        sgsClusterInt_DistanceFromHomei = gridspec.GridSpecFromSubplotSpec(2, 1,                                               subplot_spec = gsClusterInt[4, cluster],                                                            hspace = .05, wspace = .2, height_ratios = (.2, .8))
        
        percent = eval('df_in[df_in[' + '\'' + cluster_in +                     '\'' + '] == cluster][' + '\'' + 'DistanceFromHome' +                        '\'' + '].sum()') / df_in['DistanceFromHome'].sum() * 100

        # boxplot
        axClusterInt_DistanceFromHomei_00 = plt.Subplot(figClusterInt, sgsClusterInt_DistanceFromHomei[0, 0])
        figClusterInt.add_subplot(axClusterInt_DistanceFromHomei_00, sharex = True)
        axClusterInt_DistanceFromHomei_00.tick_params(left = False, bottom = False, labelleft = False)
        axClusterInt_DistanceFromHomei_00.spines['left'].set_visible(False)
        axClusterInt_DistanceFromHomei_00.spines['bottom'].set_visible(False)
        axClusterInt_DistanceFromHomei_00.spines['top'].set_visible(False)
        
        if(percent > 0):
            sns.boxplot(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'DistanceFromHome' + '\'' + ']'),                        color = 'white',                        boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                        medianprops = medianprops, flierprops = flierprops, whis = whis,                        ax = axClusterInt_DistanceFromHomei_00)
            
            axClusterInt_DistanceFromHomei_00.tick_params(labelbottom = False) 
            
            axClusterInt_DistanceFromHomei_00.set_xlabel('')
            stats = f_stats_boxplot(eval('df_in[df_in[' + '\'' + cluster_in +                     '\'' + '] == cluster][' + '\'' + 'DistanceFromHome' + '\'' + ']'))
            if(stats[3] > 2000):
                axClusterInt_DistanceFromHomei_00.text(stats[1], -.35, '{:.0f}k'.format(stats[1] / 1000),                                                   color = 'red', ha = 'center')
                axClusterInt_DistanceFromHomei_00.text(stats[3], .1, '{:.0f}k'.format(stats[3] / 1000),                                                   color = 'black', ha = 'center')
                axClusterInt_DistanceFromHomei_00.text(stats[5], -.35, '{:.0f}k'.format(stats[5] / 1000),                                                   color = 'red', ha = 'center')
            else:
                axClusterInt_DistanceFromHomei_00.text(stats[1], -.35, '{:.1f}'.format(stats[1]), color = 'red',                                                       ha = 'center')
                axClusterInt_DistanceFromHomei_00.text(stats[3], .1, '{:.1f}'.format(stats[3]), color = 'black',                                                       ha = 'center')
                axClusterInt_DistanceFromHomei_00.text(stats[5], -.35, '{:.1f}'.format(stats[5]), color = 'red',                                                       ha = 'center')                

            if(stats[3] > lim_sup_in * max(df_in['DistanceFromHome'])):
                axClusterInt_DistanceFromHomei_00.set_xlim((min(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'DistanceFromHome' + '\'' + ']')),                                                 max(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'DistanceFromHome' + '\'' + ']'))))
                axClusterInt_DistanceFromHomei_00.set_title('({:.1f}%)'.format(percent), size = 10, color = 'red')
            else:
                axClusterInt_DistanceFromHomei_00.set_xlim((0, max(df_in['DistanceFromHome'])))
                axClusterInt_DistanceFromHomei_00.set_title('({:.1f}%)'.format(percent), size = 10)
            #axClusterInt_DistanceFromHomei_00.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
            axClusterInt_DistanceFromHomei_00.get_xaxis().set_major_locator(mtick.MaxNLocator(integer = True,                                                                                          nbins = ntickbins_in - 1)) 

        # histogram
        axClusterInt_DistanceFromHomei_10 = plt.Subplot(figClusterInt, sgsClusterInt_DistanceFromHomei[1, 0])
        figClusterInt.add_subplot(axClusterInt_DistanceFromHomei_10)
                                  
        if(percent > 0):    
            if((stats[3] > lim_sup_in * max(df_in['DistanceFromHome'])) or                (stats[5] <  lim_inf_in * max(df_in['DistanceFromHome']))):
                axClusterInt_DistanceFromHomei_10.set_xlim((min(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'DistanceFromHome' + '\'' + ']')),                                                 max(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'DistanceFromHome' + '\'' + ']'))))
                axClusterInt_DistanceFromHomei_10.tick_params(axis = 'x', colors = 'red')
            else:
                axClusterInt_DistanceFromHomei_10.set_xlim((0, max(df_in['DistanceFromHome'])))

            sns.distplot(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'DistanceFromHome' + '\'' + ']'), kde = False,                         rug = False, bins = bins_in,                         color = '#4ec5b0', hist_kws = {'alpha': .85},                         ax = axClusterInt_DistanceFromHomei_10)
                                  
            axClusterInt_DistanceFromHomei_10.set_xlabel('DistanceFromHome')
            axClusterInt_DistanceFromHomei_10.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
            axClusterInt_DistanceFromHomei_10.get_xaxis().set_major_locator(mtick.MaxNLocator(integer = True,                                                                                 nbins = ntickbins_in - 1)) 
            
            if(stats[3] > 2000):
                axClusterInt_DistanceFromHomei_10.get_xaxis().set_major_formatter(mtick.FuncFormatter(lambda x,                      pos: '{:.0f}k'.format(x / 1000)))            

            if(cluster == 0):
                axClusterInt_DistanceFromHomei_10.set_ylabel('Emp. Count')
            else:
                axClusterInt_DistanceFromHomei_10.set_ylabel('')
            axClusterInt_DistanceFromHomei_10.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())
            axClusterInt_DistanceFromHomei_10.get_yaxis().set_major_locator(mtick.MaxNLocator(integer = True)) 


            
        # -------- Figs: Heatmap for the relation between the EducationField & Education
        df_EducationEducationFieldsi = pd.DataFrame(np.zeros(shape = (len(EducationField), len(Education))))
        row = 0
        for ED in EducationField:
            col = 0
            for j in Education:
                df_EducationEducationFieldsi.iloc[row, col] = eval('df_in[(df_in[' + '\'' +                        cluster_in + '\'' + '] == cluster) & (df_in[' + '\'' + 'EducationField' + '\'' +                        '] == ED) & (df_in[' + '\'' + 'Education' + '\'' + '] == j)][' + '\'' +                        'EmployeeNumber' + '\'' + '].count()') 

                col = col + 1
            row = row + 1

        axClusterInt_EducationFieldsLevelsi = plt.Subplot(figClusterInt, gsClusterInt[5, cluster])
        figClusterInt.add_subplot(axClusterInt_EducationFieldsLevelsi) 

        sns.heatmap(df_EducationEducationFieldsi,  annot = True, fmt = '.0f', cmap = 'YlGnBu', vmin = 0,                    vmax = df_EducationEducationFieldsi.values.max() , linewidths = .5, cbar = False,                    ax = axClusterInt_EducationFieldsLevelsi)        

        axClusterInt_EducationFieldsLevelsi.invert_yaxis()
        axClusterInt_EducationFieldsLevelsi.set_xticklabels(Education, **{'rotation': 0})  
        axClusterInt_EducationFieldsLevelsi.set_xlabel('Education')
        axClusterInt_EducationFieldsLevelsi.set_aspect('equal')

        axdividerEducationFieldsLevelsi = make_axes_locatable(axClusterInt_EducationFieldsLevelsi)
        axdividerEducationFieldsLevelsi.set_anchor((1,.5))
        caxEducationFieldsLevelsi = axdividerEducationFieldsLevelsi.append_axes('bottom', size = '3%', pad = '20%')
        colorbar(axClusterInt_EducationFieldsLevelsi.get_children()[0],                 cax = caxEducationFieldsLevelsi, orientation = 'horizontal',                 **{'ticks': (0, df_EducationEducationFieldsi.values.max())})
        caxEducationFieldsLevelsi.xaxis.set_ticks_position('bottom')
        caxEducationFieldsLevelsi.set_xlabel('Emp. Count')
        caxEducationFieldsLevelsi.get_xaxis().set_label_position('bottom')
            
            
        if(cluster == 0):
            axClusterInt_EducationFieldsLevelsi.set_ylabel('EducationField')   
            axClusterInt_EducationFieldsLevelsi.set_yticklabels(EducationField, **{'rotation': 0})  
        else:
            axClusterInt_EducationFieldsLevelsi.set_yticklabels([])   


        
        # -------- Figs: Barplot for Department
        axClusterInt_Departmenti = plt.Subplot(figClusterInt, gsClusterInt[6, cluster])
        figClusterInt.add_subplot(axClusterInt_Departmenti)

        departmentlevels = eval('df_in[df_in[' + '\'' + cluster_in +                                                        '\'' + '] == cluster].groupby([' + '\'' + 'Department' +                                                        '\'' + '])[' + '\'' + 'EmployeeNumber' + '\'' +'].count()')
        
        departmentlevelslevels = pd.Series(departmentlevels, index = departmentlevels_index, name = 'DepartmentLevels')
        palette = sns.color_palette(departmentlevels_colors)
        
        sns.barplot(x = departmentlevelslevels.index, y = departmentlevelslevels, palette = palette,                    ax = axClusterInt_Departmenti)
        if(cluster == 0):
            axClusterInt_Departmenti.set_ylabel('Emp. Count')
        else:
            axClusterInt_Departmenti.set_ylabel('')
        axClusterInt_Departmenti.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())
        axClusterInt_Departmenti.get_yaxis().set_major_locator(mtick.MaxNLocator(integer = True)) 
        plt.setp(axClusterInt_Departmenti.get_xticklabels(), **{'rotation': 20})
        pos = 0
        total = departmentlevelslevels.sum()
        for index, value in departmentlevelslevels.iteritems():
            if(value > 0):
                axClusterInt_Departmenti.text(pos, value, '{:.0f}%'.format(value / total * 100),                                             color = 'black', ha = 'center')
            pos = pos + 1 
        
        

        # -------- Figs: Heatmap for the relation between the JobRole & JobLevel 
        df_JobRoles_Levelsi = pd.DataFrame(np.zeros(shape = (len(JobRole), len(JobLevel))))
        row = 0
        for JR in JobRole:
            col = 0
            for j in JobLevel:
                df_JobRoles_Levelsi.iloc[row, col] = eval('df_in[(df_in[' + '\'' +                        cluster_in + '\'' + '] == cluster) & (df_in[' + '\'' + 'JobRole' + '\'' +                        '] == JR) & (df_in[' + '\'' + 'JobLevel' + '\'' + '] == j)][' + '\'' +                        'EmployeeNumber' + '\'' + '].count()') 

                col = col + 1
            row = row + 1

        axClusterInt_JobRolesLevelsi = plt.Subplot(figClusterInt, gsClusterInt[7, cluster])
        figClusterInt.add_subplot(axClusterInt_JobRolesLevelsi) 

        sns.heatmap(df_JobRoles_Levelsi,  annot = True, fmt = '.0f', cmap = 'YlGnBu', vmin = 0,                    vmax = df_JobRoles_Levelsi.values.max() , linewidths = .5, cbar = False,                    ax = axClusterInt_JobRolesLevelsi)        

        axClusterInt_JobRolesLevelsi.invert_yaxis()
        axClusterInt_JobRolesLevelsi.set_xlabel('JobLevel')
        axClusterInt_JobRolesLevelsi.set_xticklabels(JobLevel, **{'rotation': 0}) 
        axClusterInt_JobRolesLevelsi.set_aspect('equal')

        axdividerJobRolesLevelsi = make_axes_locatable(axClusterInt_JobRolesLevelsi)
        axdividerJobRolesLevelsi.set_anchor((1,.5))
        caxJobRolesLevelsi = axdividerJobRolesLevelsi.append_axes('bottom', size = '3%', pad = '20%')
        colorbar(axClusterInt_JobRolesLevelsi.get_children()[0],                 cax = caxJobRolesLevelsi, orientation = 'horizontal',                 **{'ticks': (0, df_JobRoles_Levelsi.values.max())})
        caxJobRolesLevelsi.xaxis.set_ticks_position('bottom')
        caxJobRolesLevelsi.set_xlabel('Emp. Count')
        caxJobRolesLevelsi.get_xaxis().set_label_position('bottom')
            
            
        if(cluster == 0):
            axClusterInt_JobRolesLevelsi.set_ylabel('JobRole')   
            axClusterInt_JobRolesLevelsi.set_yticklabels(JobRole, **{'rotation': 0})  
        else:
            axClusterInt_JobRolesLevelsi.set_yticklabels([])
        
        
   
        
        # -------- Figs: Barplot for BusinessTravel
        axClusterInt_WorkLifeBalancei = plt.Subplot(figClusterInt, gsClusterInt[8, cluster])
        figClusterInt.add_subplot(axClusterInt_WorkLifeBalancei)

        worklifebalance_values = eval('df_in[df_in[' + '\'' + cluster_in +                                                        '\'' + '] == cluster].groupby([' + '\'' + 'WorkLifeBalance' +                                                        '\'' + '])[' + '\'' + 'EmployeeNumber' + '\'' +'].count()')
        
        worklifebalancelevels = pd.Series(worklifebalance_values, index = worklifebalancelevels_index,                                           name = 'WorkLifeBalanceLevels')
        palette = sns.color_palette(worklifebalancelevels_colors)
        
        sns.barplot(x = worklifebalancelevels.index, y = worklifebalancelevels, palette = palette,                    ax = axClusterInt_WorkLifeBalancei)
        if(cluster == 0):
            axClusterInt_WorkLifeBalancei.set_ylabel('Emp. Count')
        else:
            axClusterInt_WorkLifeBalancei.set_ylabel('')
        axClusterInt_WorkLifeBalancei.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())
        axClusterInt_WorkLifeBalancei.get_yaxis().set_major_locator(mtick.MaxNLocator(integer = True)) 
        plt.setp(axClusterInt_WorkLifeBalancei.get_xticklabels(), **{'rotation': 0})
        axClusterInt_WorkLifeBalancei.set_xlabel('WorkLifeBalance')
        pos = 0
        total = worklifebalancelevels.sum()
        for index, value in worklifebalancelevels.iteritems():
            if(value > 0):
                axClusterInt_WorkLifeBalancei.text(pos, value, '{:.0f}%'.format(value / total * 100),                                             color = 'black', ha = 'center')
            pos = pos + 1   
            

        
        # -------- Figs: Barplot for BusinessTravel
        axClusterInt_BusinessTraveli = plt.Subplot(figClusterInt, gsClusterInt[9, cluster])
        figClusterInt.add_subplot(axClusterInt_BusinessTraveli)

        businesstravel_values = eval('df_in[df_in[' + '\'' + cluster_in +                                                        '\'' + '] == cluster].groupby([' + '\'' + 'BusinessTravel' +                                                        '\'' + '])[' + '\'' + 'EmployeeNumber' + '\'' +'].count()')
        
        businesstravellevels = pd.Series(businesstravel_values, index = businesstravellevels_index,                                         name = 'BusinessTravelLevels')
        palette = sns.color_palette(businesstravellevels_colors)
        
        sns.barplot(x = businesstravellevels.index, y = businesstravellevels, palette = palette,                    ax = axClusterInt_BusinessTraveli)
        if(cluster == 0):
            axClusterInt_BusinessTraveli.set_ylabel('Emp. Count')
        else:
            axClusterInt_BusinessTraveli.set_ylabel('')
        axClusterInt_BusinessTraveli.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())
        axClusterInt_BusinessTraveli.get_yaxis().set_major_locator(mtick.MaxNLocator(integer = True)) 
        plt.setp(axClusterInt_BusinessTraveli.get_xticklabels(), **{'rotation': 20})
        axClusterInt_BusinessTraveli.set_xlabel('BusinessTravel')
        pos = 0
        total = businesstravellevels.sum()
        for index, value in businesstravellevels.iteritems():
            if(value > 0):
                axClusterInt_BusinessTraveli.text(pos, value, '{:.0f}%'.format(value / total * 100),                                             color = 'black', ha = 'center')
            pos = pos + 1   

        
        # -------- Figs: Barplot for OverTime
        axClusterInt_OverTimei = plt.Subplot(figClusterInt, gsClusterInt[10, cluster])
        figClusterInt.add_subplot(axClusterInt_OverTimei)

        overtime_values = eval('df_in[df_in[' + '\'' + cluster_in +                                                        '\'' + '] == cluster].groupby([' + '\'' + 'OverTime_flag' +                                                        '\'' + '])[' + '\'' + 'EmployeeNumber' + '\'' +'].count()')
        
        overtimelevels = pd.Series(overtime_values, index = overtimelevels_index, name = 'OverTimeLevels')
        palette = sns.color_palette(overtimelevels_colors)
        
        sns.barplot(x = overtimelevels.index, y = overtimelevels, palette = palette,                    ax = axClusterInt_OverTimei)
        if(cluster == 0):
            axClusterInt_OverTimei.set_ylabel('Emp. Count')
        else:
            axClusterInt_OverTimei.set_ylabel('')
        axClusterInt_OverTimei.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())
        axClusterInt_OverTimei.get_yaxis().set_major_locator(mtick.MaxNLocator(integer = True)) 
        plt.setp(axClusterInt_OverTimei.get_xticklabels(), **{'rotation': 0})
        axClusterInt_OverTimei.set_xlabel('OverTime')
        pos = 0
        total = overtimelevels.sum()
        for index, value in overtimelevels.iteritems():
            if(value > 0):
                axClusterInt_OverTimei.text(pos, value, '{:.0f}%'.format(value / total * 100),                                             color = 'black', ha = 'center')
            pos = pos + 1  

        
        
        # -------- Figs:BoxPlot + Histogram for YearsAtCompany
        sgsClusterInt_YearsAtCompanyi = gridspec.GridSpecFromSubplotSpec(2, 1,                                               subplot_spec = gsClusterInt[11, cluster],                                                            hspace = .05, wspace = .2, height_ratios = (.2, .8))
        
        percent = eval('df_in[df_in[' + '\'' + cluster_in +                     '\'' + '] == cluster][' + '\'' + 'YearsAtCompany' +                        '\'' + '].sum()') / df_in['YearsAtCompany'].sum() * 100

        # boxplot
        axClusterInt_YearsAtCompanyi_00 = plt.Subplot(figClusterInt, sgsClusterInt_YearsAtCompanyi[0, 0])
        figClusterInt.add_subplot(axClusterInt_YearsAtCompanyi_00, sharex = True)
        axClusterInt_YearsAtCompanyi_00.tick_params(left = False, bottom = False, labelleft = False)
        axClusterInt_YearsAtCompanyi_00.spines['left'].set_visible(False)
        axClusterInt_YearsAtCompanyi_00.spines['bottom'].set_visible(False)
        axClusterInt_YearsAtCompanyi_00.spines['top'].set_visible(False)
        
        if(percent > 0):
            sns.boxplot(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'YearsAtCompany' + '\'' + ']'),                        color = 'white',                        boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                        medianprops = medianprops, flierprops = flierprops, whis = whis,                        ax = axClusterInt_YearsAtCompanyi_00)
            
            axClusterInt_YearsAtCompanyi_00.tick_params(labelbottom = False) 
            
            axClusterInt_YearsAtCompanyi_00.set_xlabel('')
            stats = f_stats_boxplot(eval('df_in[df_in[' + '\'' + cluster_in +                     '\'' + '] == cluster][' + '\'' + 'YearsAtCompany' + '\'' + ']'))
            if(stats[3] > 2000):
                axClusterInt_YearsAtCompanyi_00.text(stats[1], -.35, '{:.0f}k'.format(stats[1] / 1000),                                                   color = 'red', ha = 'center')
                axClusterInt_YearsAtCompanyi_00.text(stats[3], .1, '{:.0f}k'.format(stats[3] / 1000),                                                   color = 'black', ha = 'center')
                axClusterInt_YearsAtCompanyi_00.text(stats[5], -.35, '{:.0f}k'.format(stats[5] / 1000),                                                   color = 'red', ha = 'center')
            else:
                axClusterInt_YearsAtCompanyi_00.text(stats[1], -.35, '{:.1f}'.format(stats[1]), color = 'red',                                                       ha = 'center')
                axClusterInt_YearsAtCompanyi_00.text(stats[3], .1, '{:.1f}'.format(stats[3]), color = 'black',                                                       ha = 'center')
                axClusterInt_YearsAtCompanyi_00.text(stats[5], -.35, '{:.1f}'.format(stats[5]), color = 'red',                                                       ha = 'center')                

            if(stats[3] > lim_sup_in * max(df_in['YearsAtCompany'])):
                axClusterInt_YearsAtCompanyi_00.set_xlim((min(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'YearsAtCompany' + '\'' + ']')),                                                 max(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'YearsAtCompany' + '\'' + ']'))))
                axClusterInt_YearsAtCompanyi_00.set_title('({:.1f}%)'.format(percent), size = 10, color = 'red')
            else:
                axClusterInt_YearsAtCompanyi_00.set_xlim((0, max(df_in['YearsAtCompany'])))
                axClusterInt_YearsAtCompanyi_00.set_title('({:.1f}%)'.format(percent), size = 10)
            #axClusterInt_YearsAtCompanyi_00.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
            axClusterInt_YearsAtCompanyi_00.get_xaxis().set_major_locator(mtick.MaxNLocator(integer = True,                                                                                          nbins = ntickbins_in - 1)) 

        # histogram
        axClusterInt_YearsAtCompanyi_10 = plt.Subplot(figClusterInt, sgsClusterInt_YearsAtCompanyi[1, 0])
        figClusterInt.add_subplot(axClusterInt_YearsAtCompanyi_10)
                                  
        if(percent > 0):    
            if((stats[3] > lim_sup_in * max(df_in['YearsAtCompany'])) or                (stats[5] <  lim_inf_in * max(df_in['YearsAtCompany']))):
                axClusterInt_YearsAtCompanyi_10.set_xlim((min(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'YearsAtCompany' + '\'' + ']')),                                                 max(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'YearsAtCompany' + '\'' + ']'))))
                axClusterInt_YearsAtCompanyi_10.tick_params(axis = 'x', colors = 'red')
            else:
                axClusterInt_YearsAtCompanyi_10.set_xlim((0, max(df_in['YearsAtCompany'])))

            sns.distplot(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'YearsAtCompany' + '\'' + ']'), kde = False,                         rug = False, bins = bins_in,                         color = '#9ECCFF', hist_kws = {'alpha': .85},                         ax = axClusterInt_YearsAtCompanyi_10)
                                  
            axClusterInt_YearsAtCompanyi_10.set_xlabel('YearsAtCompany')
            axClusterInt_YearsAtCompanyi_10.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
            axClusterInt_YearsAtCompanyi_10.get_xaxis().set_major_locator(mtick.MaxNLocator(integer = True,                                                                                 nbins = ntickbins_in - 1)) 
            
            if(stats[3] > 2000):
                axClusterInt_YearsAtCompanyi_10.get_xaxis().set_major_formatter(mtick.FuncFormatter(lambda x,                      pos: '{:.0f}k'.format(x / 1000)))            

            if(cluster == 0):
                axClusterInt_YearsAtCompanyi_10.set_ylabel('Emp. Count')
            else:
                axClusterInt_YearsAtCompanyi_10.set_ylabel('')
            axClusterInt_YearsAtCompanyi_10.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())
            axClusterInt_YearsAtCompanyi_10.get_yaxis().set_major_locator(mtick.MaxNLocator(integer = True)) 
        
        
        
        # -------- Figs:BoxPlot + Histogram for YearsWithCurrManager
        sgsClusterInt_YearsWithCurrManageri = gridspec.GridSpecFromSubplotSpec(2, 1,                                               subplot_spec = gsClusterInt[12, cluster],                                                            hspace = .05, wspace = .2, height_ratios = (.2, .8))
        
        percent = eval('df_in[df_in[' + '\'' + cluster_in +                     '\'' + '] == cluster][' + '\'' + 'YearsWithCurrManager' +                        '\'' + '].sum()') / df_in['YearsWithCurrManager'].sum() * 100

        # boxplot
        axClusterInt_YearsWithCurrManageri_00 = plt.Subplot(figClusterInt, sgsClusterInt_YearsWithCurrManageri[0, 0])
        figClusterInt.add_subplot(axClusterInt_YearsWithCurrManageri_00, sharex = True)
        axClusterInt_YearsWithCurrManageri_00.tick_params(left = False, bottom = False, labelleft = False)
        axClusterInt_YearsWithCurrManageri_00.spines['left'].set_visible(False)
        axClusterInt_YearsWithCurrManageri_00.spines['bottom'].set_visible(False)
        axClusterInt_YearsWithCurrManageri_00.spines['top'].set_visible(False)
        
        if(percent > 0):
            sns.boxplot(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'YearsWithCurrManager' + '\'' + ']'),                        color = 'white',                        boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                        medianprops = medianprops, flierprops = flierprops, whis = whis,                        ax = axClusterInt_YearsWithCurrManageri_00)
            
            axClusterInt_YearsWithCurrManageri_00.tick_params(labelbottom = False) 
            
            axClusterInt_YearsWithCurrManageri_00.set_xlabel('')
            stats = f_stats_boxplot(eval('df_in[df_in[' + '\'' + cluster_in +                     '\'' + '] == cluster][' + '\'' + 'YearsWithCurrManager' + '\'' + ']'))
            if(stats[3] > 2000):
                axClusterInt_YearsWithCurrManageri_00.text(stats[1], -.35, '{:.0f}k'.format(stats[1] / 1000),                                                   color = 'red', ha = 'center')
                axClusterInt_YearsWithCurrManageri_00.text(stats[3], .1, '{:.0f}k'.format(stats[3] / 1000),                                                   color = 'black', ha = 'center')
                axClusterInt_YearsWithCurrManageri_00.text(stats[5], -.35, '{:.0f}k'.format(stats[5] / 1000),                                                   color = 'red', ha = 'center')
            else:
                axClusterInt_YearsWithCurrManageri_00.text(stats[1], -.35, '{:.1f}'.format(stats[1]), color = 'red',                                                       ha = 'center')
                axClusterInt_YearsWithCurrManageri_00.text(stats[3], .1, '{:.1f}'.format(stats[3]), color = 'black',                                                       ha = 'center')
                axClusterInt_YearsWithCurrManageri_00.text(stats[5], -.35, '{:.1f}'.format(stats[5]), color = 'red',                                                       ha = 'center')                

            if(stats[3] > lim_sup_in * max(df_in['YearsWithCurrManager'])):
                axClusterInt_YearsWithCurrManageri_00.set_xlim((min(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'YearsWithCurrManager' + '\'' + ']')),                                                 max(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'YearsWithCurrManager' + '\'' + ']'))))
                axClusterInt_YearsWithCurrManageri_00.set_title('({:.1f}%)'.format(percent), size = 10, color = 'red')
            else:
                axClusterInt_YearsWithCurrManageri_00.set_xlim((0, max(df_in['YearsWithCurrManager'])))
                axClusterInt_YearsWithCurrManageri_00.set_title('({:.1f}%)'.format(percent), size = 10)
            #axClusterInt_YearsWithCurrManageri_00.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
            axClusterInt_YearsWithCurrManageri_00.get_xaxis().set_major_locator(mtick.MaxNLocator(integer = True,                                                                                          nbins = ntickbins_in - 1)) 

        # histogram
        axClusterInt_YearsWithCurrManageri_10 = plt.Subplot(figClusterInt, sgsClusterInt_YearsWithCurrManageri[1, 0])
        figClusterInt.add_subplot(axClusterInt_YearsWithCurrManageri_10)
                                  
        if(percent > 0):    
            if((stats[3] > lim_sup_in * max(df_in['YearsWithCurrManager'])) or                (stats[5] <  lim_inf_in * max(df_in['YearsWithCurrManager']))):
                axClusterInt_YearsWithCurrManageri_10.set_xlim((min(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'YearsWithCurrManager' + '\'' + ']')),                                                 max(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'YearsWithCurrManager' + '\'' + ']'))))
                axClusterInt_YearsWithCurrManageri_10.tick_params(axis = 'x', colors = 'red')
            else:
                axClusterInt_YearsWithCurrManageri_10.set_xlim((0, max(df_in['YearsWithCurrManager'])))

            sns.distplot(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'YearsWithCurrManager' + '\'' + ']'), kde = False,                         rug = False, bins = bins_in,                         color = '#9ECCFF', hist_kws = {'alpha': .85},                         ax = axClusterInt_YearsWithCurrManageri_10)
                                  
            axClusterInt_YearsWithCurrManageri_10.set_xlabel('YearsWithCurrManager')
            axClusterInt_YearsWithCurrManageri_10.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
            axClusterInt_YearsWithCurrManageri_10.get_xaxis().set_major_locator(mtick.MaxNLocator(integer = True,                                                                                 nbins = ntickbins_in - 1)) 
            
            if(stats[3] > 2000):
                axClusterInt_YearsWithCurrManageri_10.get_xaxis().set_major_formatter(mtick.FuncFormatter(lambda x,                      pos: '{:.0f}k'.format(x / 1000)))            

            if(cluster == 0):
                axClusterInt_YearsWithCurrManageri_10.set_ylabel('Emp. Count')
            else:
                axClusterInt_YearsWithCurrManageri_10.set_ylabel('')
            axClusterInt_YearsWithCurrManageri_10.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())
            axClusterInt_YearsWithCurrManageri_10.get_yaxis().set_major_locator(mtick.MaxNLocator(integer = True))      
        
        
        
        # -------- Figs: Relation between YearsWithCurrManager vs TotalWorkingYears
        axClusterInt_IncomeWYears = plt.Subplot(figClusterInt, gsClusterInt[13, cluster])
        figClusterInt.add_subplot(axClusterInt_IncomeWYears)

        sns.scatterplot(x = eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'YearsAtCompany' + '\'' + ']'),                        y = eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'YearsWithCurrManager' + '\'' + ']'),                        edgecolor = 'black', linewidth = .3, color = '#303D69', marker = 'o', s = 20, alpha = .5,                        ax = axClusterInt_IncomeWYears)

        axClusterInt_IncomeWYears.set_xlabel('YearsAtCompany')
        axClusterInt_IncomeWYears.set_xlim((-2, 42))
        axClusterInt_IncomeWYears.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
        axClusterInt_IncomeWYears.get_xaxis().set_major_locator(mtick.MaxNLocator(integer = True,                                                                     nbins = ntickbins_in - 1)) 

        if(cluster == 0):
            axClusterInt_IncomeWYears.set_ylabel('YearsWithCurrManager')
        else:
            axClusterInt_IncomeWYears.set_ylabel('')
        axClusterInt_IncomeWYears.set_ylim((-2, 20))
        axClusterInt_IncomeWYears.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())
        axClusterInt_IncomeWYears.get_yaxis().set_major_locator(mtick.MaxNLocator(integer = True,                                                                             nbins = ntickbins_in - 1))

        
        # -------- Figs:BoxPlot + Histogram for YearsInCurrentRole
        sgsClusterInt_YearsInCurrentRolei = gridspec.GridSpecFromSubplotSpec(2, 1,                                               subplot_spec = gsClusterInt[14, cluster],                                                            hspace = .05, wspace = .2, height_ratios = (.2, .8))
        
        percent = eval('df_in[df_in[' + '\'' + cluster_in +                     '\'' + '] == cluster][' + '\'' + 'YearsInCurrentRole' +                        '\'' + '].sum()') / df_in['YearsInCurrentRole'].sum() * 100

        # boxplot
        axClusterInt_YearsInCurrentRolei_00 = plt.Subplot(figClusterInt, sgsClusterInt_YearsInCurrentRolei[0, 0])
        figClusterInt.add_subplot(axClusterInt_YearsInCurrentRolei_00, sharex = True)
        axClusterInt_YearsInCurrentRolei_00.tick_params(left = False, bottom = False, labelleft = False)
        axClusterInt_YearsInCurrentRolei_00.spines['left'].set_visible(False)
        axClusterInt_YearsInCurrentRolei_00.spines['bottom'].set_visible(False)
        axClusterInt_YearsInCurrentRolei_00.spines['top'].set_visible(False)
        
        if(percent > 0):
            sns.boxplot(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'YearsInCurrentRole' + '\'' + ']'),                        color = 'white',                        boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                        medianprops = medianprops, flierprops = flierprops, whis = whis,                        ax = axClusterInt_YearsInCurrentRolei_00)
            
            axClusterInt_YearsInCurrentRolei_00.tick_params(labelbottom = False) 
            
            axClusterInt_YearsInCurrentRolei_00.set_xlabel('')
            stats = f_stats_boxplot(eval('df_in[df_in[' + '\'' + cluster_in +                     '\'' + '] == cluster][' + '\'' + 'YearsInCurrentRole' + '\'' + ']'))
            if(stats[3] > 2000):
                axClusterInt_YearsInCurrentRolei_00.text(stats[1], -.35, '{:.0f}k'.format(stats[1] / 1000),                                                   color = 'red', ha = 'center')
                axClusterInt_YearsInCurrentRolei_00.text(stats[3], .1, '{:.0f}k'.format(stats[3] / 1000),                                                   color = 'black', ha = 'center')
                axClusterInt_YearsInCurrentRolei_00.text(stats[5], -.35, '{:.0f}k'.format(stats[5] / 1000),                                                   color = 'red', ha = 'center')
            else:
                axClusterInt_YearsInCurrentRolei_00.text(stats[1], -.35, '{:.1f}'.format(stats[1]), color = 'red',                                                       ha = 'center')
                axClusterInt_YearsInCurrentRolei_00.text(stats[3], .1, '{:.1f}'.format(stats[3]), color = 'black',                                                       ha = 'center')
                axClusterInt_YearsInCurrentRolei_00.text(stats[5], -.35, '{:.1f}'.format(stats[5]), color = 'red',                                                       ha = 'center')                

            if(stats[3] > lim_sup_in * max(df_in['YearsInCurrentRole'])):
                axClusterInt_YearsInCurrentRolei_00.set_xlim((min(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'YearsInCurrentRole' + '\'' + ']')),                                                 max(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'YearsInCurrentRole' + '\'' + ']'))))
                axClusterInt_YearsInCurrentRolei_00.set_title('({:.1f}%)'.format(percent), size = 10, color = 'red')
            else:
                axClusterInt_YearsInCurrentRolei_00.set_xlim((0, max(df_in['YearsInCurrentRole'])))
                axClusterInt_YearsInCurrentRolei_00.set_title('({:.1f}%)'.format(percent), size = 10)
            #axClusterInt_YearsInCurrentRolei_00.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
            axClusterInt_YearsInCurrentRolei_00.get_xaxis().set_major_locator(mtick.MaxNLocator(integer = True,                                                                                          nbins = ntickbins_in - 1)) 

        # histogram
        axClusterInt_YearsInCurrentRolei_10 = plt.Subplot(figClusterInt, sgsClusterInt_YearsInCurrentRolei[1, 0])
        figClusterInt.add_subplot(axClusterInt_YearsInCurrentRolei_10)
                                  
        if(percent > 0):    
            if((stats[3] > lim_sup_in * max(df_in['YearsInCurrentRole'])) or                (stats[5] <  lim_inf_in * max(df_in['YearsInCurrentRole']))):
                axClusterInt_YearsInCurrentRolei_10.set_xlim((min(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'YearsInCurrentRole' + '\'' + ']')),                                                 max(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'YearsInCurrentRole' + '\'' + ']'))))
                axClusterInt_YearsInCurrentRolei_10.tick_params(axis = 'x', colors = 'red')
            else:
                axClusterInt_YearsInCurrentRolei_10.set_xlim((0, max(df_in['YearsInCurrentRole'])))

            sns.distplot(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'YearsInCurrentRole' + '\'' + ']'), kde = False,                         rug = False, bins = bins_in,                         color = '#9ECCFF', hist_kws = {'alpha': .85},                         ax = axClusterInt_YearsInCurrentRolei_10)
                                  
            axClusterInt_YearsInCurrentRolei_10.set_xlabel('YearsInCurrentRole')
            axClusterInt_YearsInCurrentRolei_10.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
            axClusterInt_YearsInCurrentRolei_10.get_xaxis().set_major_locator(mtick.MaxNLocator(integer = True,                                                                                 nbins = ntickbins_in - 1)) 
            
            if(stats[3] > 2000):
                axClusterInt_YearsInCurrentRolei_10.get_xaxis().set_major_formatter(mtick.FuncFormatter(lambda x,                      pos: '{:.0f}k'.format(x / 1000)))            

            if(cluster == 0):
                axClusterInt_YearsInCurrentRolei_10.set_ylabel('Emp. Count')
            else:
                axClusterInt_YearsInCurrentRolei_10.set_ylabel('')
            axClusterInt_YearsInCurrentRolei_10.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())
            axClusterInt_YearsInCurrentRolei_10.get_yaxis().set_major_locator(mtick.MaxNLocator(integer = True))         
        
        

        # -------- Figs: Relation between YearsAtCompany vs YearsInCurrentRole
        axClusterInt_IncomeWYears = plt.Subplot(figClusterInt, gsClusterInt[15, cluster])
        figClusterInt.add_subplot(axClusterInt_IncomeWYears)

        sns.scatterplot(x = eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'YearsAtCompany' + '\'' + ']'),                        y = eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'YearsInCurrentRole' + '\'' + ']'),                        edgecolor = 'black', linewidth = .3, color = '#303D69', marker = 'o', s = 20, alpha = .5,                        ax = axClusterInt_IncomeWYears)

        axClusterInt_IncomeWYears.set_xlabel('YearsAtCompany')
        axClusterInt_IncomeWYears.set_xlim((-2, 42))
        axClusterInt_IncomeWYears.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
        axClusterInt_IncomeWYears.get_xaxis().set_major_locator(mtick.MaxNLocator(integer = True,                                                                     nbins = ntickbins_in - 1)) 

        if(cluster == 0):
            axClusterInt_IncomeWYears.set_ylabel('YearsInCurrentRole')
        else:
            axClusterInt_IncomeWYears.set_ylabel('')
        axClusterInt_IncomeWYears.set_ylim((-2, 20))
        axClusterInt_IncomeWYears.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())
        axClusterInt_IncomeWYears.get_yaxis().set_major_locator(mtick.MaxNLocator(integer = True,                                                                             nbins = ntickbins_in - 1))
        

        
        # -------- Figs:BoxPlot + Histogram for NumCompaniesWorked
        sgsClusterInt_NumCompaniesWorkedi = gridspec.GridSpecFromSubplotSpec(2, 1,                                               subplot_spec = gsClusterInt[16, cluster],                                                            hspace = .05, wspace = .2, height_ratios = (.2, .8))
        
        percent = eval('df_in[df_in[' + '\'' + cluster_in +                     '\'' + '] == cluster][' + '\'' + 'NumCompaniesWorked' +                        '\'' + '].sum()') / df_in['NumCompaniesWorked'].sum() * 100

        # boxplot
        axClusterInt_NumCompaniesWorkedi_00 = plt.Subplot(figClusterInt, sgsClusterInt_NumCompaniesWorkedi[0, 0])
        figClusterInt.add_subplot(axClusterInt_NumCompaniesWorkedi_00, sharex = True)
        axClusterInt_NumCompaniesWorkedi_00.tick_params(left = False, bottom = False, labelleft = False)
        axClusterInt_NumCompaniesWorkedi_00.spines['left'].set_visible(False)
        axClusterInt_NumCompaniesWorkedi_00.spines['bottom'].set_visible(False)
        axClusterInt_NumCompaniesWorkedi_00.spines['top'].set_visible(False)
        
        if(percent > 0):
            sns.boxplot(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'NumCompaniesWorked' + '\'' + ']'),                        color = 'white',                        boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                        medianprops = medianprops, flierprops = flierprops, whis = whis,                        ax = axClusterInt_NumCompaniesWorkedi_00)
            
            axClusterInt_NumCompaniesWorkedi_00.tick_params(labelbottom = False) 
            
            axClusterInt_NumCompaniesWorkedi_00.set_xlabel('')
            stats = f_stats_boxplot(eval('df_in[df_in[' + '\'' + cluster_in +                     '\'' + '] == cluster][' + '\'' + 'NumCompaniesWorked' + '\'' + ']'))
            if(stats[3] > 2000):
                axClusterInt_NumCompaniesWorkedi_00.text(stats[1], -.35, '{:.0f}k'.format(stats[1] / 1000),                                                   color = 'red', ha = 'center')
                axClusterInt_NumCompaniesWorkedi_00.text(stats[3], .1, '{:.0f}k'.format(stats[3] / 1000),                                                   color = 'black', ha = 'center')
                axClusterInt_NumCompaniesWorkedi_00.text(stats[5], -.35, '{:.0f}k'.format(stats[5] / 1000),                                                   color = 'red', ha = 'center')
            else:
                axClusterInt_NumCompaniesWorkedi_00.text(stats[1], -.35, '{:.1f}'.format(stats[1]), color = 'red',                                                       ha = 'center')
                axClusterInt_NumCompaniesWorkedi_00.text(stats[3], .1, '{:.1f}'.format(stats[3]), color = 'black',                                                       ha = 'center')
                axClusterInt_NumCompaniesWorkedi_00.text(stats[5], -.35, '{:.1f}'.format(stats[5]), color = 'red',                                                       ha = 'center')                

            if(stats[3] > lim_sup_in * max(df_in['NumCompaniesWorked'])):
                axClusterInt_NumCompaniesWorkedi_00.set_xlim((min(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'NumCompaniesWorked' + '\'' + ']')),                                                 max(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'NumCompaniesWorked' + '\'' + ']'))))
                axClusterInt_NumCompaniesWorkedi_00.set_title('({:.1f}%)'.format(percent), size = 10, color = 'red')
            else:
                axClusterInt_NumCompaniesWorkedi_00.set_xlim((0, max(df_in['NumCompaniesWorked'])))
                axClusterInt_NumCompaniesWorkedi_00.set_title('({:.1f}%)'.format(percent), size = 10)
            #axClusterInt_NumCompaniesWorkedi_00.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
            axClusterInt_NumCompaniesWorkedi_00.get_xaxis().set_major_locator(mtick.MaxNLocator(integer = True,                                                                                          nbins = ntickbins_in - 1)) 

        # histogram
        axClusterInt_NumCompaniesWorkedi_10 = plt.Subplot(figClusterInt, sgsClusterInt_NumCompaniesWorkedi[1, 0])
        figClusterInt.add_subplot(axClusterInt_NumCompaniesWorkedi_10)
                                  
        if(percent > 0):    
            if((stats[3] > lim_sup_in * max(df_in['NumCompaniesWorked'])) or                (stats[5] <  lim_inf_in * max(df_in['NumCompaniesWorked']))):
                axClusterInt_NumCompaniesWorkedi_10.set_xlim((min(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'NumCompaniesWorked' + '\'' + ']')),                                                 max(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'NumCompaniesWorked' + '\'' + ']'))))
                axClusterInt_NumCompaniesWorkedi_10.tick_params(axis = 'x', colors = 'red')
            else:
                axClusterInt_NumCompaniesWorkedi_10.set_xlim((0, max(df_in['NumCompaniesWorked'])))

            sns.distplot(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'NumCompaniesWorked' + '\'' + ']'), kde = False,                         rug = False, bins = bins_in,                         color = '#9ECCFF', hist_kws = {'alpha': .85},                         ax = axClusterInt_NumCompaniesWorkedi_10)
                                  
            axClusterInt_NumCompaniesWorkedi_10.set_xlabel('NumCompaniesWorked')
            axClusterInt_NumCompaniesWorkedi_10.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
            axClusterInt_NumCompaniesWorkedi_10.get_xaxis().set_major_locator(mtick.MaxNLocator(integer = True,                                                                                 nbins = ntickbins_in - 1)) 
            
            if(stats[3] > 2000):
                axClusterInt_NumCompaniesWorkedi_10.get_xaxis().set_major_formatter(mtick.FuncFormatter(lambda x,                      pos: '{:.0f}k'.format(x / 1000)))            

            if(cluster == 0):
                axClusterInt_NumCompaniesWorkedi_10.set_ylabel('Emp. Count')
            else:
                axClusterInt_NumCompaniesWorkedi_10.set_ylabel('')
            axClusterInt_NumCompaniesWorkedi_10.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())
            axClusterInt_NumCompaniesWorkedi_10.get_yaxis().set_major_locator(mtick.MaxNLocator(integer = True))  
            
            
 
        # -------- Figs: Barplot for PerformanceRating
        axClusterInt_PerformanceRatingi = plt.Subplot(figClusterInt, gsClusterInt[17, cluster])
        figClusterInt.add_subplot(axClusterInt_PerformanceRatingi)
        
        perfrating_values = eval('df_in[df_in[' + '\'' + cluster_in +                                                '\'' + '] == cluster].groupby([' + '\'' + 'PerformanceRating_str' +                                                '\'' + '])[' + '\'' + 'EmployeeNumber' + '\'' +'].count()')

        perfratinglevels = pd.Series(perfrating_values, index = sorted(df_in['PerformanceRating_str'].unique()),                                     name = 'PerformanceRatingLevels')

        sns.barplot(x = perfratinglevels.index, y = perfratinglevels, palette = palette_relscale_str,                    ax = axClusterInt_PerformanceRatingi, order = sorted(df_in['PerformanceRating_str'].unique()))
        if(cluster == 0):
            axClusterInt_PerformanceRatingi.set_ylabel('Emp. Count')
        else:
            axClusterInt_PerformanceRatingi.set_ylabel('')
        axClusterInt_PerformanceRatingi.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())
        axClusterInt_PerformanceRatingi.get_yaxis().set_major_locator(mtick.MaxNLocator(integer = True)) 
        plt.setp(axClusterInt_PerformanceRatingi.get_xticklabels(), **{'rotation': 0})
        axClusterInt_PerformanceRatingi.set_xlabel('PerformanceRating')
        pos = 0
        total = perfratinglevels.sum()
        for index, value in perfratinglevels.iteritems():
            if(value > 0):
                axClusterInt_PerformanceRatingi.text(pos, value, '{:.0f}%'.format(value / total * 100),                                             color = 'black', ha = 'center')
            pos = pos + 1  
            

            
        # -------- Figs: Barplot for JobInvolvement
        axClusterInt_JobInvolvementi = plt.Subplot(figClusterInt, gsClusterInt[18, cluster])
        figClusterInt.add_subplot(axClusterInt_JobInvolvementi)

        involvement_values = eval('df_in[df_in[' + '\'' + cluster_in +                                                '\'' + '] == cluster].groupby([' + '\'' + 'JobInvolvement_str' +                                                '\'' + '])[' + '\'' + 'EmployeeNumber' + '\'' +'].count()')
        
        involvementlevels = pd.Series(involvement_values, index = sorted(df_in['JobInvolvement_str'].unique()),                                     name = 'JobInvolvementLevels')

        sns.barplot(x = involvementlevels.index, y = involvementlevels, palette =palette_relscale_str,                    ax = axClusterInt_JobInvolvementi, order = sorted(df_in['JobInvolvement_str'].unique()))
        if(cluster == 0):
            axClusterInt_JobInvolvementi.set_ylabel('Emp. Count')
        else:
            axClusterInt_JobInvolvementi.set_ylabel('')
        axClusterInt_JobInvolvementi.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())
        axClusterInt_JobInvolvementi.get_yaxis().set_major_locator(mtick.MaxNLocator(integer = True)) 
        plt.setp(axClusterInt_JobInvolvementi.get_xticklabels(), **{'rotation': 0})
        axClusterInt_JobInvolvementi.set_xlabel('JobInvolvement')
        pos = 0
        total = involvementlevels.sum()
        for index, value in involvementlevels.iteritems():
            if(value > 0):
                axClusterInt_JobInvolvementi.text(pos, value, '{:.0f}%'.format(value / total * 100),                                             color = 'black', ha = 'center')
            pos = pos + 1  
            
            

        # -------- Figs: Barplot for JobSatisfaction
        axClusterInt_JobSatisfactioni = plt.Subplot(figClusterInt, gsClusterInt[19, cluster])
        figClusterInt.add_subplot(axClusterInt_JobSatisfactioni)

        satisfaction_values = eval('df_in[df_in[' + '\'' + cluster_in +                                                '\'' + '] == cluster].groupby([' + '\'' + 'JobSatisfaction_str' +                                                '\'' + '])[' + '\'' + 'EmployeeNumber' + '\'' +'].count()')
        
        satisfactionlevels = pd.Series(satisfaction_values, index = sorted(df_in['JobSatisfaction_str'].unique()),                                     name = 'JobSatisfactionLevels')

        sns.barplot(x = satisfactionlevels.index, y = satisfactionlevels, palette =palette_relscale_str,                    ax = axClusterInt_JobSatisfactioni, order = sorted(df_in['JobSatisfaction_str'].unique()))
        if(cluster == 0):
            axClusterInt_JobSatisfactioni.set_ylabel('Emp. Count')
        else:
            axClusterInt_JobSatisfactioni.set_ylabel('')
        axClusterInt_JobSatisfactioni.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())
        axClusterInt_JobSatisfactioni.get_yaxis().set_major_locator(mtick.MaxNLocator(integer = True)) 
        plt.setp(axClusterInt_JobSatisfactioni.get_xticklabels(), **{'rotation': 0})
        axClusterInt_JobSatisfactioni.set_xlabel('JobSatisfaction')
        pos = 0
        total = satisfactionlevels.sum()
        for index, value in satisfactionlevels.iteritems():
            if(value > 0):
                axClusterInt_JobSatisfactioni.text(pos, value, '{:.0f}%'.format(value / total * 100),                                             color = 'black', ha = 'center')
            pos = pos + 1           
            
            
            
        # -------- Figs: Barplot for EnvironmentSatisfaction
        axClusterInt_EnvironmentSatisfactioni = plt.Subplot(figClusterInt, gsClusterInt[20, cluster])
        figClusterInt.add_subplot(axClusterInt_EnvironmentSatisfactioni)

        envsatisfaction_values = eval('df_in[df_in[' + '\'' + cluster_in +                                                '\'' + '] == cluster].groupby([' + '\'' + 'EnvironmentSatisfaction_str' +                                                '\'' + '])[' + '\'' + 'EmployeeNumber' + '\'' +'].count()')
        
        envsatisfactionlevels = pd.Series(envsatisfaction_values, index = sorted(df_in['EnvironmentSatisfaction_str'].unique()),                                     name = 'EnvironmentSatisfactionLevels')

        sns.barplot(x = envsatisfactionlevels.index, y = envsatisfactionlevels, palette =palette_relscale_str,                    ax = axClusterInt_EnvironmentSatisfactioni, order = sorted(df_in['EnvironmentSatisfaction_str'].unique()))
        if(cluster == 0):
            axClusterInt_EnvironmentSatisfactioni.set_ylabel('Emp. Count')
        else:
            axClusterInt_EnvironmentSatisfactioni.set_ylabel('')
        axClusterInt_EnvironmentSatisfactioni.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())
        axClusterInt_EnvironmentSatisfactioni.get_yaxis().set_major_locator(mtick.MaxNLocator(integer = True)) 
        plt.setp(axClusterInt_EnvironmentSatisfactioni.get_xticklabels(), **{'rotation': 0})
        axClusterInt_EnvironmentSatisfactioni.set_xlabel('EnvironmentSatisfaction')
        pos = 0
        total = envsatisfactionlevels.sum()
        for index, value in envsatisfactionlevels.iteritems():
            if(value > 0):
                axClusterInt_EnvironmentSatisfactioni.text(pos, value, '{:.0f}%'.format(value / total * 100),                                             color = 'black', ha = 'center')
            pos = pos + 1              
            
            
            
        # -------- Figs: Barplot for RelationshipSatisfaction
        axClusterInt_RelationshipSatisfactioni = plt.Subplot(figClusterInt, gsClusterInt[21, cluster])
        figClusterInt.add_subplot(axClusterInt_RelationshipSatisfactioni)

        relsatisfaction_values = eval('df_in[df_in[' + '\'' + cluster_in +                                                '\'' + '] == cluster].groupby([' + '\'' + 'RelationshipSatisfaction_str' +                                                '\'' + '])[' + '\'' + 'EmployeeNumber' + '\'' +'].count()')
        
        relsatisfactionlevels = pd.Series(relsatisfaction_values, index = sorted(df_in['RelationshipSatisfaction_str'].unique()),                                     name = 'RelationshipSatisfactionLevels')

        sns.barplot(x = relsatisfactionlevels.index, y = relsatisfactionlevels, palette =palette_relscale_str,                    ax = axClusterInt_RelationshipSatisfactioni, order = sorted(df_in['RelationshipSatisfaction_str'].unique()))
        if(cluster == 0):
            axClusterInt_RelationshipSatisfactioni.set_ylabel('Emp. Count')
        else:
            axClusterInt_RelationshipSatisfactioni.set_ylabel('')
        axClusterInt_RelationshipSatisfactioni.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())
        axClusterInt_RelationshipSatisfactioni.get_yaxis().set_major_locator(mtick.MaxNLocator(integer = True)) 
        plt.setp(axClusterInt_RelationshipSatisfactioni.get_xticklabels(), **{'rotation': 0})
        axClusterInt_RelationshipSatisfactioni.set_xlabel('RelationshipSatisfaction')
        pos = 0
        total = relsatisfactionlevels.sum()
        for index, value in relsatisfactionlevels.iteritems():
            if(value > 0):
                axClusterInt_RelationshipSatisfactioni.text(pos, value, '{:.0f}%'.format(value / total * 100),                                             color = 'black', ha = 'center')
            pos = pos + 1   
            

            
        # -------- Figs:BoxPlot + Histogram for TrainingTimesLastYear
        sgsClusterInt_TrainingTimesLastYeari = gridspec.GridSpecFromSubplotSpec(2, 1,                                               subplot_spec = gsClusterInt[22, cluster],                                                            hspace = .05, wspace = .2, height_ratios = (.2, .8))
        
        percent = eval('df_in[df_in[' + '\'' + cluster_in +                     '\'' + '] == cluster][' + '\'' + 'TrainingTimesLastYear' +                        '\'' + '].sum()') / df_in['TrainingTimesLastYear'].sum() * 100

        # boxplot
        axClusterInt_TrainingTimesLastYeari_00 = plt.Subplot(figClusterInt, sgsClusterInt_TrainingTimesLastYeari[0, 0])
        figClusterInt.add_subplot(axClusterInt_TrainingTimesLastYeari_00, sharex = True)
        axClusterInt_TrainingTimesLastYeari_00.tick_params(left = False, bottom = False, labelleft = False)
        axClusterInt_TrainingTimesLastYeari_00.spines['left'].set_visible(False)
        axClusterInt_TrainingTimesLastYeari_00.spines['bottom'].set_visible(False)
        axClusterInt_TrainingTimesLastYeari_00.spines['top'].set_visible(False)
        
        if(percent > 0):
            sns.boxplot(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'TrainingTimesLastYear' + '\'' + ']'),                        color = 'white',                        boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                        medianprops = medianprops, flierprops = flierprops, whis = whis,                        ax = axClusterInt_TrainingTimesLastYeari_00)
            
            axClusterInt_TrainingTimesLastYeari_00.tick_params(labelbottom = False) 
            
            axClusterInt_TrainingTimesLastYeari_00.set_xlabel('')
            stats = f_stats_boxplot(eval('df_in[df_in[' + '\'' + cluster_in +                     '\'' + '] == cluster][' + '\'' + 'TrainingTimesLastYear' + '\'' + ']'))
            if(stats[3] > 2000):
                axClusterInt_TrainingTimesLastYeari_00.text(stats[1], -.35, '{:.0f}k'.format(stats[1] / 1000),                                                   color = 'red', ha = 'center')
                axClusterInt_TrainingTimesLastYeari_00.text(stats[3], .1, '{:.0f}k'.format(stats[3] / 1000),                                                   color = 'black', ha = 'center')
                axClusterInt_TrainingTimesLastYeari_00.text(stats[5], -.35, '{:.0f}k'.format(stats[5] / 1000),                                                   color = 'red', ha = 'center')
            else:
                axClusterInt_TrainingTimesLastYeari_00.text(stats[1], -.35, '{:.1f}'.format(stats[1]), color = 'red',                                                       ha = 'center')
                axClusterInt_TrainingTimesLastYeari_00.text(stats[3], .1, '{:.1f}'.format(stats[3]), color = 'black',                                                       ha = 'center')
                axClusterInt_TrainingTimesLastYeari_00.text(stats[5], -.35, '{:.1f}'.format(stats[5]), color = 'red',                                                       ha = 'center')                

            if(stats[3] > lim_sup_in * max(df_in['TrainingTimesLastYear'])):
                axClusterInt_TrainingTimesLastYeari_00.set_xlim((min(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'TrainingTimesLastYear' + '\'' + ']')),                                                 max(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'TrainingTimesLastYear' + '\'' + ']'))))
                axClusterInt_TrainingTimesLastYeari_00.set_title('({:.1f}%)'.format(percent), size = 10, color = 'red')
            else:
                axClusterInt_TrainingTimesLastYeari_00.set_xlim((0, max(df_in['TrainingTimesLastYear'])))
                axClusterInt_TrainingTimesLastYeari_00.set_title('({:.1f}%)'.format(percent), size = 10)
            #axClusterInt_TrainingTimesLastYeari_00.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
            axClusterInt_TrainingTimesLastYeari_00.get_xaxis().set_major_locator(mtick.MaxNLocator(integer = True,                                                                                          nbins = ntickbins_in - 1)) 

        # histogram
        axClusterInt_TrainingTimesLastYeari_10 = plt.Subplot(figClusterInt, sgsClusterInt_TrainingTimesLastYeari[1, 0])
        figClusterInt.add_subplot(axClusterInt_TrainingTimesLastYeari_10)
                                  
        if(percent > 0):    
            if((stats[3] > lim_sup_in * max(df_in['TrainingTimesLastYear'])) or                (stats[5] <  lim_inf_in * max(df_in['TrainingTimesLastYear']))):
                axClusterInt_TrainingTimesLastYeari_10.set_xlim((min(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'TrainingTimesLastYear' + '\'' + ']')),                                                 max(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'TrainingTimesLastYear' + '\'' + ']'))))
                axClusterInt_TrainingTimesLastYeari_10.tick_params(axis = 'x', colors = 'red')
            else:
                axClusterInt_TrainingTimesLastYeari_10.set_xlim((0, max(df_in['TrainingTimesLastYear'])))

            sns.distplot(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'TrainingTimesLastYear' + '\'' + ']'), kde = False,                         rug = False, bins = bins_in,                         color = '#7EA3CC', hist_kws = {'alpha': .85},                         ax = axClusterInt_TrainingTimesLastYeari_10)
                                  
            axClusterInt_TrainingTimesLastYeari_10.set_xlabel('TrainingTimesLastYear')
            axClusterInt_TrainingTimesLastYeari_10.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
            axClusterInt_TrainingTimesLastYeari_10.get_xaxis().set_major_locator(mtick.MaxNLocator(integer = True,                                                                                 nbins = ntickbins_in - 1)) 
            
            if(stats[3] > 2000):
                axClusterInt_TrainingTimesLastYeari_10.get_xaxis().set_major_formatter(mtick.FuncFormatter(lambda x,                      pos: '{:.0f}k'.format(x / 1000)))            

            if(cluster == 0):
                axClusterInt_TrainingTimesLastYeari_10.set_ylabel('Emp. Count')
            else:
                axClusterInt_TrainingTimesLastYeari_10.set_ylabel('')
            axClusterInt_TrainingTimesLastYeari_10.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())
            axClusterInt_TrainingTimesLastYeari_10.get_yaxis().set_major_locator(mtick.MaxNLocator(integer = True)) 
            
        

        # -------- Figs:BoxPlot + Histogram for MonthlyIncome
        sgsClusterInt_MonthlyIncomei = gridspec.GridSpecFromSubplotSpec(2, 1,                                               subplot_spec = gsClusterInt[23, cluster],                                                            hspace = .05, wspace = .2, height_ratios = (.2, .8))
        
        percent = eval('df_in[df_in[' + '\'' + cluster_in +                     '\'' + '] == cluster][' + '\'' + 'MonthlyIncome' +                        '\'' + '].sum()') / df_in['MonthlyIncome'].sum() * 100

        # boxplot
        axClusterInt_MonthlyIncomei_00 = plt.Subplot(figClusterInt, sgsClusterInt_MonthlyIncomei[0, 0])
        figClusterInt.add_subplot(axClusterInt_MonthlyIncomei_00, sharex = True)
        axClusterInt_MonthlyIncomei_00.tick_params(left = False, bottom = False, labelleft = False)
        axClusterInt_MonthlyIncomei_00.spines['left'].set_visible(False)
        axClusterInt_MonthlyIncomei_00.spines['bottom'].set_visible(False)
        axClusterInt_MonthlyIncomei_00.spines['top'].set_visible(False)
        
        if(percent > 0):
            sns.boxplot(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'MonthlyIncome' + '\'' + ']'),                        color = 'white',                        boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                        medianprops = medianprops, flierprops = flierprops, whis = whis,                        ax = axClusterInt_MonthlyIncomei_00)
            
            axClusterInt_MonthlyIncomei_00.tick_params(labelbottom = False) 
            
            axClusterInt_MonthlyIncomei_00.set_xlabel('')
            stats = f_stats_boxplot(eval('df_in[df_in[' + '\'' + cluster_in +                     '\'' + '] == cluster][' + '\'' + 'MonthlyIncome' + '\'' + ']'))
            if(stats[3] > 2000):
                axClusterInt_MonthlyIncomei_00.text(stats[1], -.35, '{:.0f}k'.format(stats[1] / 1000),                                                   color = 'red', ha = 'center')
                axClusterInt_MonthlyIncomei_00.text(stats[3], .1, '{:.0f}k'.format(stats[3] / 1000),                                                   color = 'black', ha = 'center')
                axClusterInt_MonthlyIncomei_00.text(stats[5], -.35, '{:.0f}k'.format(stats[5] / 1000),                                                   color = 'red', ha = 'center')
            else:
                axClusterInt_MonthlyIncomei_00.text(stats[1], -.35, '{:.1f}'.format(stats[1]), color = 'red',                                                       ha = 'center')
                axClusterInt_MonthlyIncomei_00.text(stats[3], .1, '{:.1f}'.format(stats[3]), color = 'black',                                                       ha = 'center')
                axClusterInt_MonthlyIncomei_00.text(stats[5], -.35, '{:.1f}'.format(stats[5]), color = 'red',                                                       ha = 'center')                

            if(stats[3] > lim_sup_in * max(df_in['MonthlyIncome'])):
                axClusterInt_MonthlyIncomei_00.set_xlim((min(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'MonthlyIncome' + '\'' + ']')),                                                 max(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'MonthlyIncome' + '\'' + ']'))))
                axClusterInt_MonthlyIncomei_00.set_title('({:.1f}%)'.format(percent), size = 10, color = 'red')
            else:
                axClusterInt_MonthlyIncomei_00.set_xlim((0, max(df_in['MonthlyIncome'])))
                axClusterInt_MonthlyIncomei_00.set_title('({:.1f}%)'.format(percent), size = 10)
            #axClusterInt_MonthlyIncomei_00.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
            axClusterInt_MonthlyIncomei_00.get_xaxis().set_major_locator(mtick.MaxNLocator(integer = True,                                                                                          nbins = ntickbins_in - 1)) 

        # histogram
        axClusterInt_MonthlyIncomei_10 = plt.Subplot(figClusterInt, sgsClusterInt_MonthlyIncomei[1, 0])
        figClusterInt.add_subplot(axClusterInt_MonthlyIncomei_10)
                                  
        if(percent > 0):    
            if((stats[3] > lim_sup_in * max(df_in['MonthlyIncome'])) or                (stats[5] <  lim_inf_in * max(df_in['MonthlyIncome']))):
                axClusterInt_MonthlyIncomei_10.set_xlim((min(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'MonthlyIncome' + '\'' + ']')),                                                 max(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'MonthlyIncome' + '\'' + ']'))))
                axClusterInt_MonthlyIncomei_10.tick_params(axis = 'x', colors = 'red')
            else:
                axClusterInt_MonthlyIncomei_10.set_xlim((0, max(df_in['MonthlyIncome'])))

            sns.distplot(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'MonthlyIncome' + '\'' + ']'), kde = False,                         rug = False, bins = bins_in,                         color = '#4ec5b0', hist_kws = {'alpha': .85},                         ax = axClusterInt_MonthlyIncomei_10)
                                  
            axClusterInt_MonthlyIncomei_10.set_xlabel('MonthlyIncome')
            axClusterInt_MonthlyIncomei_10.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
            axClusterInt_MonthlyIncomei_10.get_xaxis().set_major_locator(mtick.MaxNLocator(integer = True,                                                                                 nbins = ntickbins_in - 1)) 
            
            if(stats[3] > 2000):
                axClusterInt_MonthlyIncomei_10.get_xaxis().set_major_formatter(mtick.FuncFormatter(lambda x,                      pos: '{:.0f}k'.format(x / 1000)))            

            if(cluster == 0):
                axClusterInt_MonthlyIncomei_10.set_ylabel('Emp. Count')
            else:
                axClusterInt_MonthlyIncomei_10.set_ylabel('')
            axClusterInt_MonthlyIncomei_10.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())
            axClusterInt_MonthlyIncomei_10.get_yaxis().set_major_locator(mtick.MaxNLocator(integer = True)) 
        

        
        # -------- Figs:BoxPlot + Histogram for YearsSinceLastPromotion
        sgsClusterInt_YearsSinceLastPromotioni = gridspec.GridSpecFromSubplotSpec(2, 1,                                               subplot_spec = gsClusterInt[24, cluster],                                                            hspace = .05, wspace = .2, height_ratios = (.2, .8))
        
        percent = eval('df_in[df_in[' + '\'' + cluster_in +                     '\'' + '] == cluster][' + '\'' + 'YearsSinceLastPromotion' +                        '\'' + '].sum()') / df_in['YearsSinceLastPromotion'].sum() * 100

        # boxplot
        axClusterInt_YearsSinceLastPromotioni_00 = plt.Subplot(figClusterInt, sgsClusterInt_YearsSinceLastPromotioni[0, 0])
        figClusterInt.add_subplot(axClusterInt_YearsSinceLastPromotioni_00, sharex = True)
        axClusterInt_YearsSinceLastPromotioni_00.tick_params(left = False, bottom = False, labelleft = False)
        axClusterInt_YearsSinceLastPromotioni_00.spines['left'].set_visible(False)
        axClusterInt_YearsSinceLastPromotioni_00.spines['bottom'].set_visible(False)
        axClusterInt_YearsSinceLastPromotioni_00.spines['top'].set_visible(False)
        
        if(percent > 0):
            sns.boxplot(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'YearsSinceLastPromotion' + '\'' + ']'),                        color = 'white',                        boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                        medianprops = medianprops, flierprops = flierprops, whis = whis,                        ax = axClusterInt_YearsSinceLastPromotioni_00)
            
            axClusterInt_YearsSinceLastPromotioni_00.tick_params(labelbottom = False) 
            
            axClusterInt_YearsSinceLastPromotioni_00.set_xlabel('')
            stats = f_stats_boxplot(eval('df_in[df_in[' + '\'' + cluster_in +                     '\'' + '] == cluster][' + '\'' + 'YearsSinceLastPromotion' + '\'' + ']'))
            if(stats[3] > 2000):
                axClusterInt_YearsSinceLastPromotioni_00.text(stats[1], -.35, '{:.0f}k'.format(stats[1] / 1000),                                                   color = 'red', ha = 'center')
                axClusterInt_YearsSinceLastPromotioni_00.text(stats[3], .1, '{:.0f}k'.format(stats[3] / 1000),                                                   color = 'black', ha = 'center')
                axClusterInt_YearsSinceLastPromotioni_00.text(stats[5], -.35, '{:.0f}k'.format(stats[5] / 1000),                                                   color = 'red', ha = 'center')
            else:
                axClusterInt_YearsSinceLastPromotioni_00.text(stats[1], -.35, '{:.1f}'.format(stats[1]), color = 'red',                                                       ha = 'center')
                axClusterInt_YearsSinceLastPromotioni_00.text(stats[3], .1, '{:.1f}'.format(stats[3]), color = 'black',                                                       ha = 'center')
                axClusterInt_YearsSinceLastPromotioni_00.text(stats[5], -.35, '{:.1f}'.format(stats[5]), color = 'red',                                                       ha = 'center')                

            if(stats[3] > lim_sup_in * max(df_in['YearsSinceLastPromotion'])):
                axClusterInt_YearsSinceLastPromotioni_00.set_xlim((min(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'YearsSinceLastPromotion' + '\'' + ']')),                                                 max(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'YearsSinceLastPromotion' + '\'' + ']'))))
                axClusterInt_YearsSinceLastPromotioni_00.set_title('({:.1f}%)'.format(percent), size = 10, color = 'red')
            else:
                axClusterInt_YearsSinceLastPromotioni_00.set_xlim((0, max(df_in['YearsSinceLastPromotion'])))
                axClusterInt_YearsSinceLastPromotioni_00.set_title('({:.1f}%)'.format(percent), size = 10)
            #axClusterInt_YearsSinceLastPromotioni_00.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
            axClusterInt_YearsSinceLastPromotioni_00.get_xaxis().set_major_locator(mtick.MaxNLocator(integer = True,                                                                                          nbins = ntickbins_in - 1)) 

        # histogram
        axClusterInt_YearsSinceLastPromotioni_10 = plt.Subplot(figClusterInt, sgsClusterInt_YearsSinceLastPromotioni[1, 0])
        figClusterInt.add_subplot(axClusterInt_YearsSinceLastPromotioni_10)
                                  
        if(percent > 0):    
            if((stats[3] > lim_sup_in * max(df_in['YearsSinceLastPromotion'])) or                (stats[5] <  lim_inf_in * max(df_in['YearsSinceLastPromotion']))):
                axClusterInt_YearsSinceLastPromotioni_10.set_xlim((min(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'YearsSinceLastPromotion' + '\'' + ']')),                                                 max(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'YearsSinceLastPromotion' + '\'' + ']'))))
                axClusterInt_YearsSinceLastPromotioni_10.tick_params(axis = 'x', colors = 'red')
            else:
                axClusterInt_YearsSinceLastPromotioni_10.set_xlim((0, max(df_in['YearsSinceLastPromotion'])))

            sns.distplot(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'YearsSinceLastPromotion' + '\'' + ']'), kde = False,                         rug = False, bins = bins_in,                         color = '#BBCC5D', hist_kws = {'alpha': .85},                         ax = axClusterInt_YearsSinceLastPromotioni_10)
                                  
            axClusterInt_YearsSinceLastPromotioni_10.set_xlabel('YearsSinceLastPromotion')
            axClusterInt_YearsSinceLastPromotioni_10.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
            axClusterInt_YearsSinceLastPromotioni_10.get_xaxis().set_major_locator(mtick.MaxNLocator(integer = True,                                                                                 nbins = ntickbins_in - 1)) 
            
            if(stats[3] > 2000):
                axClusterInt_YearsSinceLastPromotioni_10.get_xaxis().set_major_formatter(mtick.FuncFormatter(lambda x,                      pos: '{:.0f}k'.format(x / 1000)))            

            if(cluster == 0):
                axClusterInt_YearsSinceLastPromotioni_10.set_ylabel('Emp. Count')
            else:
                axClusterInt_YearsSinceLastPromotioni_10.set_ylabel('')
            axClusterInt_YearsSinceLastPromotioni_10.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())
            axClusterInt_YearsSinceLastPromotioni_10.get_yaxis().set_major_locator(mtick.MaxNLocator(integer = True))
            

        
        # -------- Figs:BoxPlot + Histogram for PercentSalaryHike
        sgsClusterInt_PercentSalaryHikei = gridspec.GridSpecFromSubplotSpec(2, 1,                                               subplot_spec = gsClusterInt[25, cluster],                                                            hspace = .05, wspace = .2, height_ratios = (.2, .8))
        
        percent = eval('df_in[df_in[' + '\'' + cluster_in +                     '\'' + '] == cluster][' + '\'' + 'PercentSalaryHike' +                        '\'' + '].sum()') / df_in['PercentSalaryHike'].sum() * 100

        # boxplot
        axClusterInt_PercentSalaryHikei_00 = plt.Subplot(figClusterInt, sgsClusterInt_PercentSalaryHikei[0, 0])
        figClusterInt.add_subplot(axClusterInt_PercentSalaryHikei_00, sharex = True)
        axClusterInt_PercentSalaryHikei_00.tick_params(left = False, bottom = False, labelleft = False)
        axClusterInt_PercentSalaryHikei_00.spines['left'].set_visible(False)
        axClusterInt_PercentSalaryHikei_00.spines['bottom'].set_visible(False)
        axClusterInt_PercentSalaryHikei_00.spines['top'].set_visible(False)
        
        if(percent > 0):
            sns.boxplot(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'PercentSalaryHike' + '\'' + ']'),                        color = 'white',                        boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                        medianprops = medianprops, flierprops = flierprops, whis = whis,                        ax = axClusterInt_PercentSalaryHikei_00)
            
            axClusterInt_PercentSalaryHikei_00.tick_params(labelbottom = False) 
            
            axClusterInt_PercentSalaryHikei_00.set_xlabel('')
            stats = f_stats_boxplot(eval('df_in[df_in[' + '\'' + cluster_in +                     '\'' + '] == cluster][' + '\'' + 'PercentSalaryHike' + '\'' + ']'))
            if(stats[3] > 2000):
                axClusterInt_PercentSalaryHikei_00.text(stats[1], -.35, '{:.0f}k'.format(stats[1] / 1000),                                                   color = 'red', ha = 'center')
                axClusterInt_PercentSalaryHikei_00.text(stats[3], .1, '{:.0f}k'.format(stats[3] / 1000),                                                   color = 'black', ha = 'center')
                axClusterInt_PercentSalaryHikei_00.text(stats[5], -.35, '{:.0f}k'.format(stats[5] / 1000),                                                   color = 'red', ha = 'center')
            else:
                axClusterInt_PercentSalaryHikei_00.text(stats[1], -.35, '{:.1f}'.format(stats[1]), color = 'red',                                                       ha = 'center')
                axClusterInt_PercentSalaryHikei_00.text(stats[3], .1, '{:.1f}'.format(stats[3]), color = 'black',                                                       ha = 'center')
                axClusterInt_PercentSalaryHikei_00.text(stats[5], -.35, '{:.1f}'.format(stats[5]), color = 'red',                                                       ha = 'center')                

            if(stats[3] > lim_sup_in * max(df_in['PercentSalaryHike'])):
                axClusterInt_PercentSalaryHikei_00.set_xlim((min(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'PercentSalaryHike' + '\'' + ']')),                                                 max(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'PercentSalaryHike' + '\'' + ']'))))
                axClusterInt_PercentSalaryHikei_00.set_title('({:.1f}%)'.format(percent), size = 10, color = 'red')
            else:
                axClusterInt_PercentSalaryHikei_00.set_xlim((0, max(df_in['PercentSalaryHike'])))
                axClusterInt_PercentSalaryHikei_00.set_title('({:.1f}%)'.format(percent), size = 10)
            #axClusterInt_PercentSalaryHikei_00.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
            axClusterInt_PercentSalaryHikei_00.get_xaxis().set_major_locator(mtick.MaxNLocator(integer = True,                                                                                          nbins = ntickbins_in - 1)) 

        # histogram
        axClusterInt_PercentSalaryHikei_10 = plt.Subplot(figClusterInt, sgsClusterInt_PercentSalaryHikei[1, 0])
        figClusterInt.add_subplot(axClusterInt_PercentSalaryHikei_10)
                                  
        if(percent > 0):    
            if((stats[3] > lim_sup_in * max(df_in['PercentSalaryHike'])) or                (stats[5] <  lim_inf_in * max(df_in['PercentSalaryHike']))):
                axClusterInt_PercentSalaryHikei_10.set_xlim((min(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'PercentSalaryHike' + '\'' + ']')),                                                 max(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'PercentSalaryHike' + '\'' + ']'))))
                axClusterInt_PercentSalaryHikei_10.tick_params(axis = 'x', colors = 'red')
            else:
                axClusterInt_PercentSalaryHikei_10.set_xlim((0, max(df_in['PercentSalaryHike'])))

            sns.distplot(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'PercentSalaryHike' + '\'' + ']'), kde = False,                         rug = False, bins = bins_in,                         color = '#BBCC5D', hist_kws = {'alpha': .85},                         ax = axClusterInt_PercentSalaryHikei_10)
                                  
            axClusterInt_PercentSalaryHikei_10.set_xlabel('PercentSalaryHike')
            axClusterInt_PercentSalaryHikei_10.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
            axClusterInt_PercentSalaryHikei_10.get_xaxis().set_major_locator(mtick.MaxNLocator(integer = True,                                                                                 nbins = ntickbins_in - 1)) 
            
            if(stats[3] > 2000):
                axClusterInt_PercentSalaryHikei_10.get_xaxis().set_major_formatter(mtick.FuncFormatter(lambda x,                      pos: '{:.0f}k'.format(x / 1000)))            

            if(cluster == 0):
                axClusterInt_PercentSalaryHikei_10.set_ylabel('Emp. Count')
            else:
                axClusterInt_PercentSalaryHikei_10.set_ylabel('')
            axClusterInt_PercentSalaryHikei_10.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())
            axClusterInt_PercentSalaryHikei_10.get_yaxis().set_major_locator(mtick.MaxNLocator(integer = True))
            

            
        # -------- Figs: Barplot for StockOptionLevel
        axClusterInt_StockOptionLeveli = plt.Subplot(figClusterInt, gsClusterInt[26, cluster])
        figClusterInt.add_subplot(axClusterInt_StockOptionLeveli)

        stockoptn_values = eval('df_in[df_in[' + '\'' + cluster_in +                                                '\'' + '] == cluster].groupby([' + '\'' + 'StockOptionLevel_str' +                                                '\'' + '])[' + '\'' + 'EmployeeNumber' + '\'' +'].count()')

        stockoptnlevels = pd.Series(stockoptn_values, index = sorted(df_in['StockOptionLevel_str'].unique()),                                     name = 'StockOptionLevelLevels')
        
        sns.barplot(x = stockoptnlevels.index, y = stockoptnlevels, palette = palette_stockkoptn_str,                    ax = axClusterInt_StockOptionLeveli, order = sorted(df_in['StockOptionLevel_str'].unique()))
        if(cluster == 0):
            axClusterInt_StockOptionLeveli.set_ylabel('Emp. Count')
        else:
            axClusterInt_StockOptionLeveli.set_ylabel('')
        axClusterInt_StockOptionLeveli.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())
        axClusterInt_StockOptionLeveli.get_yaxis().set_major_locator(mtick.MaxNLocator(integer = True)) 
        plt.setp(axClusterInt_StockOptionLeveli.get_xticklabels(), **{'rotation': 0})
        axClusterInt_StockOptionLeveli.set_xlabel('StockOptionLevel')
        pos = 0
        total = stockoptnlevels.sum()
        for index, value in stockoptnlevels.iteritems():
            if(value > 0):
                axClusterInt_StockOptionLeveli.text(pos, value, '{:.0f}%'.format(value / total * 100),                                             color = 'black', ha = 'center')
            pos = pos + 1  

            
            
        # -------- Figs:BoxPlot + Histogram for HourlyRate
        sgsClusterInt_HourlyRatei = gridspec.GridSpecFromSubplotSpec(2, 1,                                               subplot_spec = gsClusterInt[27, cluster],                                                            hspace = .05, wspace = .2, height_ratios = (.2, .8))
        
        percent = eval('df_in[df_in[' + '\'' + cluster_in +                     '\'' + '] == cluster][' + '\'' + 'HourlyRate' +                        '\'' + '].sum()') / df_in['HourlyRate'].sum() * 100

        # boxplot
        axClusterInt_HourlyRatei_00 = plt.Subplot(figClusterInt, sgsClusterInt_HourlyRatei[0, 0])
        figClusterInt.add_subplot(axClusterInt_HourlyRatei_00, sharex = True)
        axClusterInt_HourlyRatei_00.tick_params(left = False, bottom = False, labelleft = False)
        axClusterInt_HourlyRatei_00.spines['left'].set_visible(False)
        axClusterInt_HourlyRatei_00.spines['bottom'].set_visible(False)
        axClusterInt_HourlyRatei_00.spines['top'].set_visible(False)
        
        if(percent > 0):
            sns.boxplot(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'HourlyRate' + '\'' + ']'),                        color = 'white',                        boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                        medianprops = medianprops, flierprops = flierprops, whis = whis,                        ax = axClusterInt_HourlyRatei_00)
            
            axClusterInt_HourlyRatei_00.tick_params(labelbottom = False) 
            
            axClusterInt_HourlyRatei_00.set_xlabel('')
            stats = f_stats_boxplot(eval('df_in[df_in[' + '\'' + cluster_in +                     '\'' + '] == cluster][' + '\'' + 'HourlyRate' + '\'' + ']'))
            if(stats[3] > 2000):
                axClusterInt_HourlyRatei_00.text(stats[1], -.35, '{:.0f}k'.format(stats[1] / 1000),                                                   color = 'red', ha = 'center')
                axClusterInt_HourlyRatei_00.text(stats[3], .1, '{:.0f}k'.format(stats[3] / 1000),                                                   color = 'black', ha = 'center')
                axClusterInt_HourlyRatei_00.text(stats[5], -.35, '{:.0f}k'.format(stats[5] / 1000),                                                   color = 'red', ha = 'center')
            else:
                axClusterInt_HourlyRatei_00.text(stats[1], -.35, '{:.1f}'.format(stats[1]), color = 'red',                                                       ha = 'center')
                axClusterInt_HourlyRatei_00.text(stats[3], .1, '{:.1f}'.format(stats[3]), color = 'black',                                                       ha = 'center')
                axClusterInt_HourlyRatei_00.text(stats[5], -.35, '{:.1f}'.format(stats[5]), color = 'red',                                                       ha = 'center')                

            if(stats[3] > lim_sup_in * max(df_in['HourlyRate'])):
                axClusterInt_HourlyRatei_00.set_xlim((min(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'HourlyRate' + '\'' + ']')),                                                 max(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'HourlyRate' + '\'' + ']'))))
                axClusterInt_HourlyRatei_00.set_title('({:.1f}%)'.format(percent), size = 10, color = 'red')
            else:
                axClusterInt_HourlyRatei_00.set_xlim((0, max(df_in['HourlyRate'])))
                axClusterInt_HourlyRatei_00.set_title('({:.1f}%)'.format(percent), size = 10)
            #axClusterInt_HourlyRatei_00.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
            axClusterInt_HourlyRatei_00.get_xaxis().set_major_locator(mtick.MaxNLocator(integer = True,                                                                                          nbins = ntickbins_in - 1)) 

        # histogram
        axClusterInt_HourlyRatei_10 = plt.Subplot(figClusterInt, sgsClusterInt_HourlyRatei[1, 0])
        figClusterInt.add_subplot(axClusterInt_HourlyRatei_10)
                                  
        if(percent > 0):    
            if((stats[3] > lim_sup_in * max(df_in['HourlyRate'])) or                (stats[5] <  lim_inf_in * max(df_in['HourlyRate']))):
                axClusterInt_HourlyRatei_10.set_xlim((min(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'HourlyRate' + '\'' + ']')),                                                 max(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'HourlyRate' + '\'' + ']'))))
                axClusterInt_HourlyRatei_10.tick_params(axis = 'x', colors = 'red')
            else:
                axClusterInt_HourlyRatei_10.set_xlim((0, max(df_in['HourlyRate'])))

            sns.distplot(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'HourlyRate' + '\'' + ']'), kde = False,                         rug = False, bins = bins_in,                         color = '#718c6a', hist_kws = {'alpha': .85},                         ax = axClusterInt_HourlyRatei_10)
                                  
            axClusterInt_HourlyRatei_10.set_xlabel('HourlyRate')
            axClusterInt_HourlyRatei_10.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
            axClusterInt_HourlyRatei_10.get_xaxis().set_major_locator(mtick.MaxNLocator(integer = True,                                                                                 nbins = ntickbins_in - 1)) 
            
            if(stats[3] > 2000):
                axClusterInt_HourlyRatei_10.get_xaxis().set_major_formatter(mtick.FuncFormatter(lambda x,                      pos: '{:.0f}k'.format(x / 1000)))            

            if(cluster == 0):
                axClusterInt_HourlyRatei_10.set_ylabel('Emp. Count')
            else:
                axClusterInt_HourlyRatei_10.set_ylabel('')
            axClusterInt_HourlyRatei_10.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())
            axClusterInt_HourlyRatei_10.get_yaxis().set_major_locator(mtick.MaxNLocator(integer = True))    
            
        

        # -------- Figs:BoxPlot + Histogram for DailyRate
        sgsClusterInt_DailyRatei = gridspec.GridSpecFromSubplotSpec(2, 1,                                               subplot_spec = gsClusterInt[28, cluster],                                                            hspace = .05, wspace = .2, height_ratios = (.2, .8))
        
        percent = eval('df_in[df_in[' + '\'' + cluster_in +                     '\'' + '] == cluster][' + '\'' + 'DailyRate' +                        '\'' + '].sum()') / df_in['DailyRate'].sum() * 100

        # boxplot
        axClusterInt_DailyRatei_00 = plt.Subplot(figClusterInt, sgsClusterInt_DailyRatei[0, 0])
        figClusterInt.add_subplot(axClusterInt_DailyRatei_00, sharex = True)
        axClusterInt_DailyRatei_00.tick_params(left = False, bottom = False, labelleft = False)
        axClusterInt_DailyRatei_00.spines['left'].set_visible(False)
        axClusterInt_DailyRatei_00.spines['bottom'].set_visible(False)
        axClusterInt_DailyRatei_00.spines['top'].set_visible(False)
        
        if(percent > 0):
            sns.boxplot(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'DailyRate' + '\'' + ']'),                        color = 'white',                        boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                        medianprops = medianprops, flierprops = flierprops, whis = whis,                        ax = axClusterInt_DailyRatei_00)
            
            axClusterInt_DailyRatei_00.tick_params(labelbottom = False) 
            
            axClusterInt_DailyRatei_00.set_xlabel('')
            stats = f_stats_boxplot(eval('df_in[df_in[' + '\'' + cluster_in +                     '\'' + '] == cluster][' + '\'' + 'DailyRate' + '\'' + ']'))
            if(stats[3] > 2000):
                axClusterInt_DailyRatei_00.text(stats[1], -.35, '{:.0f}k'.format(stats[1] / 1000),                                                   color = 'red', ha = 'center')
                axClusterInt_DailyRatei_00.text(stats[3], .1, '{:.0f}k'.format(stats[3] / 1000),                                                   color = 'black', ha = 'center')
                axClusterInt_DailyRatei_00.text(stats[5], -.35, '{:.0f}k'.format(stats[5] / 1000),                                                   color = 'red', ha = 'center')
            else:
                axClusterInt_DailyRatei_00.text(stats[1], -.35, '{:.1f}'.format(stats[1]), color = 'red',                                                       ha = 'center')
                axClusterInt_DailyRatei_00.text(stats[3], .1, '{:.1f}'.format(stats[3]), color = 'black',                                                       ha = 'center')
                axClusterInt_DailyRatei_00.text(stats[5], -.35, '{:.1f}'.format(stats[5]), color = 'red',                                                       ha = 'center')                

            if(stats[3] > lim_sup_in * max(df_in['DailyRate'])):
                axClusterInt_DailyRatei_00.set_xlim((min(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'DailyRate' + '\'' + ']')),                                                 max(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'DailyRate' + '\'' + ']'))))
                axClusterInt_DailyRatei_00.set_title('({:.1f}%)'.format(percent), size = 10, color = 'red')
            else:
                axClusterInt_DailyRatei_00.set_xlim((0, max(df_in['DailyRate'])))
                axClusterInt_DailyRatei_00.set_title('({:.1f}%)'.format(percent), size = 10)
            #axClusterInt_DailyRatei_00.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
            axClusterInt_DailyRatei_00.get_xaxis().set_major_locator(mtick.MaxNLocator(integer = True,                                                                                          nbins = ntickbins_in - 1)) 

        # histogram
        axClusterInt_DailyRatei_10 = plt.Subplot(figClusterInt, sgsClusterInt_DailyRatei[1, 0])
        figClusterInt.add_subplot(axClusterInt_DailyRatei_10)
                                  
        if(percent > 0):    
            if((stats[3] > lim_sup_in * max(df_in['DailyRate'])) or                (stats[5] <  lim_inf_in * max(df_in['DailyRate']))):
                axClusterInt_DailyRatei_10.set_xlim((min(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'DailyRate' + '\'' + ']')),                                                 max(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'DailyRate' + '\'' + ']'))))
                axClusterInt_DailyRatei_10.tick_params(axis = 'x', colors = 'red')
            else:
                axClusterInt_DailyRatei_10.set_xlim((0, max(df_in['DailyRate'])))

            sns.distplot(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'DailyRate' + '\'' + ']'), kde = False,                         rug = False, bins = bins_in,                         color = '#ded797', hist_kws = {'alpha': .85},                         ax = axClusterInt_DailyRatei_10)
                                  
            axClusterInt_DailyRatei_10.set_xlabel('DailyRate')
            axClusterInt_DailyRatei_10.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
            axClusterInt_DailyRatei_10.get_xaxis().set_major_locator(mtick.MaxNLocator(integer = True,                                                                                 nbins = ntickbins_in - 1)) 
            
            if(stats[3] > 2000):
                axClusterInt_DailyRatei_10.get_xaxis().set_major_formatter(mtick.FuncFormatter(lambda x,                      pos: '{:.0f}k'.format(x / 1000)))            

            if(cluster == 0):
                axClusterInt_DailyRatei_10.set_ylabel('Emp. Count')
            else:
                axClusterInt_DailyRatei_10.set_ylabel('')
            axClusterInt_DailyRatei_10.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())
            axClusterInt_DailyRatei_10.get_yaxis().set_major_locator(mtick.MaxNLocator(integer = True))  
            

        # -------- Figs:BoxPlot + Histogram for MonthlyRate
        sgsClusterInt_MonthlyRatei = gridspec.GridSpecFromSubplotSpec(2, 1,                                               subplot_spec = gsClusterInt[29, cluster],                                                            hspace = .05, wspace = .2, height_ratios = (.2, .8))
        
        percent = eval('df_in[df_in[' + '\'' + cluster_in +                     '\'' + '] == cluster][' + '\'' + 'MonthlyRate' +                        '\'' + '].sum()') / df_in['MonthlyRate'].sum() * 100

        # boxplot
        axClusterInt_MonthlyRatei_00 = plt.Subplot(figClusterInt, sgsClusterInt_MonthlyRatei[0, 0])
        figClusterInt.add_subplot(axClusterInt_MonthlyRatei_00, sharex = True)
        axClusterInt_MonthlyRatei_00.tick_params(left = False, bottom = False, labelleft = False)
        axClusterInt_MonthlyRatei_00.spines['left'].set_visible(False)
        axClusterInt_MonthlyRatei_00.spines['bottom'].set_visible(False)
        axClusterInt_MonthlyRatei_00.spines['top'].set_visible(False)
        
        if(percent > 0):
            sns.boxplot(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'MonthlyRate' + '\'' + ']'),                        color = 'white',                        boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                        medianprops = medianprops, flierprops = flierprops, whis = whis,                        ax = axClusterInt_MonthlyRatei_00)
            
            axClusterInt_MonthlyRatei_00.tick_params(labelbottom = False) 
            
            axClusterInt_MonthlyRatei_00.set_xlabel('')
            stats = f_stats_boxplot(eval('df_in[df_in[' + '\'' + cluster_in +                     '\'' + '] == cluster][' + '\'' + 'MonthlyRate' + '\'' + ']'))
            if(stats[3] > 2000):
                axClusterInt_MonthlyRatei_00.text(stats[1], -.35, '{:.0f}k'.format(stats[1] / 1000),                                                   color = 'red', ha = 'center')
                axClusterInt_MonthlyRatei_00.text(stats[3], .1, '{:.0f}k'.format(stats[3] / 1000),                                                   color = 'black', ha = 'center')
                axClusterInt_MonthlyRatei_00.text(stats[5], -.35, '{:.0f}k'.format(stats[5] / 1000),                                                   color = 'red', ha = 'center')
            else:
                axClusterInt_MonthlyRatei_00.text(stats[1], -.35, '{:.1f}'.format(stats[1]), color = 'red',                                                       ha = 'center')
                axClusterInt_MonthlyRatei_00.text(stats[3], .1, '{:.1f}'.format(stats[3]), color = 'black',                                                       ha = 'center')
                axClusterInt_MonthlyRatei_00.text(stats[5], -.35, '{:.1f}'.format(stats[5]), color = 'red',                                                       ha = 'center')                

            if(stats[3] > lim_sup_in * max(df_in['MonthlyRate'])):
                axClusterInt_MonthlyRatei_00.set_xlim((min(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'MonthlyRate' + '\'' + ']')),                                                 max(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'MonthlyRate' + '\'' + ']'))))
                axClusterInt_MonthlyRatei_00.set_title('({:.1f}%)'.format(percent), size = 10, color = 'red')
            else:
                axClusterInt_MonthlyRatei_00.set_xlim((0, max(df_in['MonthlyRate'])))
                axClusterInt_MonthlyRatei_00.set_title('({:.1f}%)'.format(percent), size = 10)
            #axClusterInt_MonthlyRatei_00.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
            axClusterInt_MonthlyRatei_00.get_xaxis().set_major_locator(mtick.MaxNLocator(integer = True,                                                                                          nbins = ntickbins_in - 1)) 

        # histogram
        axClusterInt_MonthlyRatei_10 = plt.Subplot(figClusterInt, sgsClusterInt_MonthlyRatei[1, 0])
        figClusterInt.add_subplot(axClusterInt_MonthlyRatei_10)
                                  
        if(percent > 0):    
            if((stats[3] > lim_sup_in * max(df_in['MonthlyRate'])) or                (stats[5] <  lim_inf_in * max(df_in['MonthlyRate']))):
                axClusterInt_MonthlyRatei_10.set_xlim((min(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'MonthlyRate' + '\'' + ']')),                                                 max(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'MonthlyRate' + '\'' + ']'))))
                axClusterInt_MonthlyRatei_10.tick_params(axis = 'x', colors = 'red')
            else:
                axClusterInt_MonthlyRatei_10.set_xlim((0, max(df_in['MonthlyRate'])))

            sns.distplot(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'MonthlyRate' + '\'' + ']'), kde = False,                         rug = False, bins = bins_in,                         color = '#ebd3b0', hist_kws = {'alpha': .85},                         ax = axClusterInt_MonthlyRatei_10)
                                  
            axClusterInt_MonthlyRatei_10.set_xlabel('MonthlyRate')
            axClusterInt_MonthlyRatei_10.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
            axClusterInt_MonthlyRatei_10.get_xaxis().set_major_locator(mtick.MaxNLocator(integer = True,                                                                                 nbins = ntickbins_in - 1)) 
            
            if(stats[3] > 2000):
                axClusterInt_MonthlyRatei_10.get_xaxis().set_major_formatter(mtick.FuncFormatter(lambda x,                      pos: '{:.0f}k'.format(x / 1000)))            

            if(cluster == 0):
                axClusterInt_MonthlyRatei_10.set_ylabel('Emp. Count')
            else:
                axClusterInt_MonthlyRatei_10.set_ylabel('')
            axClusterInt_MonthlyRatei_10.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())
            axClusterInt_MonthlyRatei_10.get_yaxis().set_major_locator(mtick.MaxNLocator(integer = True))          

            
            
            
        # -------- Figs:BoxPlot + Histogram for MonthlyProfit_dv
        sgsClusterInt_MonthlyProfit_dvi = gridspec.GridSpecFromSubplotSpec(2, 1,                                               subplot_spec = gsClusterInt[30, cluster],                                                            hspace = .05, wspace = .2, height_ratios = (.2, .8))
        
        percent = eval('df_in[df_in[' + '\'' + cluster_in +                     '\'' + '] == cluster][' + '\'' + 'MonthlyProfit_dv' +                        '\'' + '].sum()') / df_in['MonthlyProfit_dv'].sum() * 100

        # boxplot
        axClusterInt_MonthlyProfit_dvi_00 = plt.Subplot(figClusterInt, sgsClusterInt_MonthlyProfit_dvi[0, 0])
        figClusterInt.add_subplot(axClusterInt_MonthlyProfit_dvi_00, sharex = True)
        axClusterInt_MonthlyProfit_dvi_00.tick_params(left = False, bottom = False, labelleft = False)
        axClusterInt_MonthlyProfit_dvi_00.spines['left'].set_visible(False)
        axClusterInt_MonthlyProfit_dvi_00.spines['bottom'].set_visible(False)

        if(percent > 0):
            sns.boxplot(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'MonthlyProfit_dv' + '\'' + ']'),                        color = 'white',                        boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops,                        medianprops = medianprops, flierprops = flierprops, whis = whis,                        ax = axClusterInt_MonthlyProfit_dvi_00)
            
            axClusterInt_MonthlyProfit_dvi_00.tick_params(labelbottom = False) 
            
            axClusterInt_MonthlyProfit_dvi_00.set_xlabel('')
            stats = f_stats_boxplot(eval('df_in[df_in[' + '\'' + cluster_in +                     '\'' + '] == cluster][' + '\'' + 'MonthlyProfit_dv' + '\'' + ']'))
            if(stats[3] > 2000):
                axClusterInt_MonthlyProfit_dvi_00.text(stats[1], -.35, '{:.0f}k'.format(stats[1] / 1000),                                                   color = 'red', ha = 'center')
                axClusterInt_MonthlyProfit_dvi_00.text(stats[3], .1, '{:.0f}k'.format(stats[3] / 1000),                                                   color = 'black', ha = 'center')
                axClusterInt_MonthlyProfit_dvi_00.text(stats[5], -.35, '{:.0f}k'.format(stats[5] / 1000),                                                   color = 'red', ha = 'center')
            else:
                axClusterInt_MonthlyProfit_dvi_00.text(stats[1], -.35, '{:.1f}'.format(stats[1]), color = 'red',                                                       ha = 'center')
                axClusterInt_MonthlyProfit_dvi_00.text(stats[3], .1, '{:.1f}'.format(stats[3]), color = 'black',                                                       ha = 'center')
                axClusterInt_MonthlyProfit_dvi_00.text(stats[5], -.35, '{:.1f}'.format(stats[5]), color = 'red',                                                       ha = 'center')                

            if(stats[3] > lim_sup_in * max(df_in['MonthlyProfit_dv'])):
                axClusterInt_MonthlyProfit_dvi_00.set_xlim((min(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'MonthlyProfit_dv' + '\'' + ']')),                                                 max(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'MonthlyProfit_dv' + '\'' + ']'))))
                axClusterInt_MonthlyProfit_dvi_00.set_title('({:.1f}%)'.format(percent), size = 10, color = 'red')
            else:
                axClusterInt_MonthlyProfit_dvi_00.set_xlim((min(df_in['MonthlyProfit_dv']),                                                            max(df_in['MonthlyProfit_dv'])))
                axClusterInt_MonthlyProfit_dvi_00.set_title('({:.1f}%)'.format(percent), size = 10)
            #axClusterInt_MonthlyProfit_dvi_00.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
            axClusterInt_MonthlyProfit_dvi_00.get_xaxis().set_major_locator(mtick.MaxNLocator(integer = True,                                                                                          nbins = ntickbins_in - 1)) 

        # histogram
        axClusterInt_MonthlyProfit_dvi_10 = plt.Subplot(figClusterInt, sgsClusterInt_MonthlyProfit_dvi[1, 0])
        figClusterInt.add_subplot(axClusterInt_MonthlyProfit_dvi_10)
                                  
        if(percent > 0):    
            if((stats[3] > lim_sup_in * max(df_in['MonthlyProfit_dv'])) or                (stats[5] <  lim_inf_in * max(df_in['MonthlyProfit_dv']))):
                axClusterInt_MonthlyProfit_dvi_10.set_xlim((min(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'MonthlyProfit_dv' + '\'' + ']')),                                                 max(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'MonthlyProfit_dv' + '\'' + ']'))))
                axClusterInt_MonthlyProfit_dvi_10.tick_params(axis = 'x', colors = 'red')
            else:
                axClusterInt_MonthlyProfit_dvi_10.set_xlim((min(df_in['MonthlyProfit_dv']),                                                            max(df_in['MonthlyProfit_dv'])))

            sns.distplot(eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'MonthlyProfit_dv' + '\'' + ']'), kde = False,                         rug = False, bins = bins_in,                         color = '#4ec5b0', hist_kws = {'alpha': .85},                         ax = axClusterInt_MonthlyProfit_dvi_10)
                                  
            axClusterInt_MonthlyProfit_dvi_10.set_xlabel('MonthlyProfit_dv')
            axClusterInt_MonthlyProfit_dvi_10.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
            axClusterInt_MonthlyProfit_dvi_10.get_xaxis().set_major_locator(mtick.MaxNLocator(integer = True,                                                                                 nbins = ntickbins_in - 1)) 
            
            if(stats[3] > 2000):
                axClusterInt_MonthlyProfit_dvi_10.get_xaxis().set_major_formatter(mtick.FuncFormatter(lambda x,                      pos: '{:.0f}k'.format(x / 1000)))            

            if(cluster == 0):
                axClusterInt_MonthlyProfit_dvi_10.set_ylabel('Emp. Count')
            else:
                axClusterInt_MonthlyProfit_dvi_10.set_ylabel('')
            axClusterInt_MonthlyProfit_dvi_10.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())
            axClusterInt_MonthlyProfit_dvi_10.get_yaxis().set_major_locator(mtick.MaxNLocator(integer = True)) 
            
            
        # Plot line x=0
        y_0 = axClusterInt_MonthlyProfit_dvi_10.get_ylim()[0]
        y_lim = axClusterInt_MonthlyProfit_dvi_10.get_ylim()[1]
        axClusterInt_MonthlyProfit_dvi_10.plot((0, 0), (y_0, y_lim), color = 'blue', alpha = 0.75,                                               linestyle = '--', zorder = 1)
        
        
        # -------- Figs: Relation between MonthlyIncome vs TotalWorkingYears
        axClusterInt_IncomeWYears = plt.Subplot(figClusterInt, gsClusterInt[31, cluster])
        figClusterInt.add_subplot(axClusterInt_IncomeWYears)

        sns.scatterplot(x = eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'TotalWorkingYears' + '\'' + ']'),                        y = eval('df_in[df_in[' + '\'' + cluster_in +                                 '\'' + '] == cluster][' + '\'' + 'MonthlyIncome' + '\'' + ']'),                        edgecolor = 'black', linewidth = .3, color = '#303D69', marker = 'o', s = 20, alpha = .5,                        ax = axClusterInt_IncomeWYears)

        axClusterInt_IncomeWYears.set_xlabel('TotalWorkingYears')
        axClusterInt_IncomeWYears.set_xlim((0, 42))
        axClusterInt_IncomeWYears.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
        axClusterInt_IncomeWYears.get_xaxis().set_major_locator(mtick.MaxNLocator(integer = True,                                                                     nbins = ntickbins_in - 1)) 

        if(cluster == 0):
            axClusterInt_IncomeWYears.set_ylabel('MonthlyIncome')
        else:
            axClusterInt_IncomeWYears.set_ylabel('')
        axClusterInt_IncomeWYears.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())
        axClusterInt_IncomeWYears.get_yaxis().set_major_locator(mtick.MaxNLocator(integer = True,                                                                             nbins = ntickbins_in - 1))
        axClusterInt_IncomeWYears.get_yaxis().set_major_formatter(mtick.FuncFormatter(lambda x,                                                                                       pos: '{:.0f}k'.format(x / 1000)))
    
        
    # remove all _str variables
    df_in.drop(columns = df_in.columns[pd.Series(df_in.columns).str.contains('_str')], inplace = True)
        
    return figClusterInt


# In[14]:


def f_CorrMatrix(df_in, w_in, h_in, annot_size_in):
    mask = np.zeros_like(df_in, dtype = np.bool)
    mask[np.triu_indices_from(mask)] = True

    figcorr_all, ax_corr_all = plt.subplots(figsize = (w_in, h_in), dpi = 80, facecolor = 'w', edgecolor = 'k')
    sns.heatmap(df_in, mask = mask, annot = True, annot_kws = {'size': annot_size_in}, fmt = '.2f', cmap = 'Spectral',        vmin = -1, vmax = 1, linewidths = .5, cbar = False,         ax = ax_corr_all).set_title('Correlation Matrix', size = 14)
    ax_corr_all.set_xticklabels(df_in.columns, **{'rotation': 90}, ha = 'right') 
    
    axdivider_corr_all = make_axes_locatable(ax_corr_all)
    caxdivider_corr_all = axdivider_corr_all.append_axes('top', size = '1%', pad = '5.5%')
    colorbar(ax_corr_all.get_children()[0], cax = caxdivider_corr_all,
             orientation = 'horizontal', **{'ticks': (-1, 1)})

    return figcorr_all


# In[15]:


def f_feature_importances(model_in, X_in, title_in, w_in, h_in, max_in):
    df_features = pd.Series(model_in.feature_importances_, index = X_in.columns)
    df_features = df_features.sort_values(ascending = True)
    n_features = df_features.size

    figfeatureImp, ax_featureImp = plt.subplots(figsize = (w_in, h_in), dpi = 80, facecolor = 'w', edgecolor = 'k')
    ax_featureImp.barh(df_features.index, df_features.values, align = 'center',                       tick_label = df_features.index.tolist())
    ax_featureImp.set_title(title_in, size = 14)
    ax_featureImp.set_xlabel('Feature importance')
    ax_featureImp.set_xlim([0, max_in])
    return figfeatureImp


# In[16]:


def f_Attrition_JobRole(df_in):
    palette = sns.color_palette("Spectral", len(df_in))

    fig_Attrition_JobRole, ax_Attrition_JobRole = plt.subplots(figsize = (5, 5), dpi = 80, facecolor = 'w', edgecolor = 'k')
    sns.barplot(x = df_in.index, y = df_in.values, palette = palette)    .set_title('Relative Attrition by JobRole', size = 14)

    ax_Attrition_JobRole.set_ylabel('Relative Attrition Level')
    ax_Attrition_JobRole.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())
    ax_Attrition_JobRole.get_yaxis().set_major_formatter(mtick.FuncFormatter(lambda y, pos: '{:.0f}%'.format(y * 100)))
    ax_Attrition_JobRole.set_xlabel('')
    ax_Attrition_JobRole.set_xticklabels(df_in.index, **{'rotation': 90}) 

    pos = 0
    for value in df_in:
        ax_Attrition_JobRole.text(pos, value, '{:.1f}%'.format(value * 100),                                             color = 'black', ha = 'center')
        pos = pos + 1
    return fig_Attrition_JobRole

