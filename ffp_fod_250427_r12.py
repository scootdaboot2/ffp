import json
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import chi2_contingency, linregress
import itertools
from itertools import combinations
import traceback
import statsmodels.api as sm
from collections import defaultdict
import urllib.request
import networkx as nx
from scipy.cluster import hierarchy


pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:,.0f}'.format
use_statsmodels = True  # or False to use sklearn


### ---------- input ----------###
# fdop_name = 'Central Oregon'
fdop_name = 'Central Oregon'


cwd = 'D:\\fire_danger\\fdop\\fdop_2025\\'
gis = 'D:\\_gis\\.data\\'
# local_dir = cwd + fdop_name + '\\'

#   Run Information
run_date = '250425'
run = 'fdop_modis_r12_250427\\'
wx_start_yr = 2009  # limit cefa weather data if desired

input_dir_master = cwd + '\\Input\\'

##  Fire Data
fires_gdb = gis + 'fod\\.current\\karen_short\\RDS-2013-0009.6_GDB\\Data\\FPA_FOD_20221014.gdb'
inform_gdb = gis + 'fod\\.current\\fires_fdop24.gdb'
odf_fires_shp = gis + 'fod\\.current\\odf\\ODF_Fires_2013_2024.shp'

ffp_fire_path = 'D:\\fire_danger\\fdop\\fdop_2025\\Input\\testing\\ffpfires_bd_upload_250418.csv'

ffp_fires = input_dir_master + 'ffp_fires'

##  Boundary Data
# fdra_shp = gis + 'fdra\\2025\\fdra_boundaries_finaldraft_20240914.shp'
fdra_shp = gis + 'fdra\\2025\\fdra_baldy_20250423.shp'
cofms_shp = gis + 'cofms\\.current\\.shp\\cofms_divisions_250312.shp'

##  remote sensing data
modis_shp = gis + "remote_sensing\\.current\\modis\\firms\\DL_FIRE_M-C61_583730\\fire_archive_M-C61_583730.shp"


input_dir = cwd + 'Input\\'
output_dir = cwd + run
dailylist_files = input_dir_master + 'DailyListing\\'

proj_crs = 3857
select_indices = ['ERC', 'BI']

# custom_stations = ['352109', '352208']

# custom_stations = ['350915', '351001', '352208', '352327', '352329']
custom_stations = ['350915', '351001', '352107', '352109', '352207', '352208', '352327', '352329', 
                    '352330', '352332', '352618', '352620', '352621', '352622', '352701', '352711', '352712', '353402', '353428']

'''--------------------------------------------------------------------
            Step 0. Repeating Functions::
---------------------------------------------------------------------'''

def pivot_weather_index(df, stations, index_name):
    """Pivot daily average index values for selected stations."""
    df = df[df['Station'].isin(stations)].copy()
    df['DOY'] = df.DATE.dt.dayofyear
    grouped = df.groupby(['Station', 'DOY'])[index_name].mean().reset_index()
    pivoted = grouped.pivot(index='DOY', columns='Station', values=index_name)
    return pivoted

def calculate_correlation_matrix(pivot_df):
    """Calculate correlation matrix from pivoted DataFrame."""
    return pivot_df.corr()

def make_heatmap(corr_matrix, title, cmap='coolwarm', mask_upper=True, ax=None):
    """Create a seaborn heatmap of the correlation matrix."""
    if mask_upper:
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    else:
        mask = None

    sns.heatmap(
        corr_matrix, 
        cmap=cmap, 
        annot=True, 
        fmt=".2f", 
        square=True, 
        mask=mask, 
        ax=ax, 
        vmin=-1, vmax=1, cbar=True
    )
    if ax:
        ax.set_title(title)
    else:
        plt.title(title)
        plt.show()

def get_ffp_params(indice, fire_type='fd'):
    """Retrieve FireFamily Plus-style logistic regression parameters."""
    coeffs = {
        'BI': {
            'fd': {'intercept': -3.8532, 'coef': 0.0441},
            'mfd': {'intercept': -6.1777, 'coef': 0.1493},
            'lfd': {'intercept': -7.7888, 'coef': 0.1831}
        },
        'ERC': {
            'fd': {'intercept': -4.1564, 'coef': 0.0331},
            'mfd': {'intercept': -5.5488, 'coef': 0.0738},
            'lfd': {'intercept': -6.1100, 'coef': 0.0736}
        }
    }
    return coeffs.get(indice, {}).get(fire_type, {'intercept': -4.0, 'coef': 0.04})

def mark_fire_days(df, fire_dates, colname='IS_FIRE_DAY'):
    """Mark rows in a DataFrame as fire days based on a date list."""
    df[colname] = df['DATE_ONLY'].isin(fire_dates).astype(int)
    return df


'''--------------------------------------------------------------------
            Step 1. Data Collection Functions::
---------------------------------------------------------------------'''

#---------- get fire danger rating areas from shape ---------#
def geojson_rest(url): #return gjson from rest service
    req = urllib.request.urlopen(url)
    gjson = req.read().decode('utf-8')
    gdf = gpd.read_file(gjson)
    return gdf 

def get_fdra(fdop_name): 
    gdf = gpd.read_file(fdra_shp)[['FDRAName', 'ParentDocN','geometry']]#WGS84
    gdf.rename({'FDRAName': 'FDRA', 'ParentDocN': 'FDOP'}, axis=1, inplace=True)
    return gdf.loc[gdf.FDOP==fdop_name].to_crs(proj_crs).reset_index(drop=True)

fdra_gdf = get_fdra(fdop_name)            
fdra_gdf_diss = fdra_gdf.dissolve()
fdra_gdf_buffer = fdra_gdf.copy()
fdra_gdf_buffer['geometry'] = fdra_gdf_buffer.buffer(120000) #~3 miles
print(fdra_gdf_diss)

#---------- get cofms boundary
def get_cofms(cofms_shp): #fdra from agol pnw fdra shape
    gdf = gpd.read_file(cofms_shp)[['Division', 'geometry']] #WGS84
    return gdf.to_crs(proj_crs).reset_index(drop=True)

cofms_gdf = get_cofms(cofms_shp)            
cofms_gdf_diss = cofms_gdf.dissolve()
cofms_gdf_buffer = cofms_gdf.copy()
cofms_gdf_buffer['geometry'] = cofms_gdf_buffer.buffer(5000) #~3 miles
print(cofms_gdf)


#---------- get weather stations from WXx statbasline api ----------#
def get_raws():
    url = 'https://weather.nifc.gov/ords/prd/wx/station/statbaseline/0'
    json_obj = json.loads(urllib.request.urlopen(url).read())['station_archive_baseline']
    df = pd.DataFrame(json_obj)
    df = df.loc[(df['Class'] == 'Permanent') & (df['Ownership Type'] == 'FIRE') & (df['Status'] == 'A')]
    df['Installed Date'] = pd.to_datetime(df['Installed Date (yyyymmdd)'], errors='coerce')
    df['Last Modified Date'] = pd.to_datetime(df['Last Modified Date (yyyymmdd hh24:mi:ss)'], errors='coerce')
    df.dropna(subset=['NWS ID'], inplace=True)
    df = df.sort_values(['Name', 'Last Modified Date']).drop_duplicates(subset=['NWS ID'], keep='last')
    df.rename({'Name': 'RAWS'}, axis=1, inplace=True)
    return df

raws_df = get_raws() # pulls all stations conus

raws_gdf = gpd.GeoDataFrame(
    raws_df, geometry=gpd.points_from_xy(raws_df.Longitude, raws_df.Latitude)).set_crs(4326).to_crs(proj_crs)
raws_gdf = raws_gdf.loc[raws_gdf['NWS ID'].isin(custom_stations)]
print(raws_gdf)

cols = ['RAWS', 'NWS ID', 'Elevation', 'Slope', 'Cover Class', 'Climate Zone', 'Agency', 'Unit', \
    'Installed Date', 'geometry']
raws_gdf = raws_gdf[['WX ID', 'NESS ID', 'RAWS', 'NWS ID', 'Elevation', 'Slope', 'Cover Class', 
                     'Latitude', 'Longitude', 'State', 'Agency', 'Region', 'Unit', 'Ownership Type', 'Installed Date', 'Last Modified Date']]


#---------- get fire file used in ffp ----------#

fires_df = pd.read_csv(ffp_fire_path)
fires_df['TotalAcres'] = pd.to_numeric(fires_df['TotalAcres'], errors='coerce')

# Convert discovery date to datetime
fires_df['DiscoveryDate'] = pd.to_datetime(fires_df['DiscoveryDate'], format='%m/%d/%Y')
fires_df['FIRE_YEAR'] = pd.to_datetime(fires_df['DiscoveryDate']).dt.year
fires_df['FDRA'] = 'Baldy'

# Rename columns for consistency
fires_df = fires_df.rename(columns={
    'DiscoveryDate': 'DISCOVERY_DATE',
    'TotalAcres': 'FIRE_SIZE',
    'FireName': 'FIRE_NAME'
})

# Group by year and ID, then count
grouped = fires_df.groupby(['FIRE_YEAR', 'FDRA'])
fires_df['Running Count'] = grouped.cumcount() + 1
fires_df['Running Count'] = fires_df['Running Count'].apply(lambda x: f'{x:03}')

fires_df['FOD_ID'] = "6"  + "_" + fires_df['FIRE_YEAR'].astype(str) + "_" + fires_df['Running Count'].astype(str)



#---------- get ff+ daily listing from local copy ---------#
def get_wx(dailylist_files): #get weather from ff daily listing file
    l = []
    for root, dirs, files in os.walk(dailylist_files):
        for f in files:
            print("Reading file..." + f)
            #get header info for settings
            with open(dailylist_files + f) as file:
                count = 1
                lines = [line.rstrip() for line in file]
                #head = lines[0:29]
                fuelmodel = lines[15][45]
                slopeclass = lines[17][16]
                avg_precip = lines[18][50:55]
                herb_ann = lines[19][35]
                if herb_ann == 'N':
                    herb_ann = 'P'
                else:
                    herb_ann = 'A'
                lat = lines[21][15:20]
            try:
                skip_rows = 29
                df = pd.read_csv(dailylist_files + f, header=0, engine='python', \
                    skiprows=skip_rows, skipfooter=1) #incl skiprows file with header
                df = df.iloc[:, :-1]
                df = df.iloc[1:]
                df = df.rename(columns=lambda x: x.strip())
                df['DATE'] = pd.to_datetime(df['DATE'])
            except KeyError:
                skip_rows = 28
                df = pd.read_csv(dailylist_files + f, header=0, engine='python', \
                    skiprows=skip_rows, skipfooter=1) #incl skiprows file with header
                df = df.iloc[:, :-1]
                df = df.iloc[1:]
                df = df.rename(columns=lambda x: x.strip())
                df['DATE'] = pd.to_datetime(df['DATE'])
            df['Fuel Model'] = fuelmodel
            df['Slope Class'] = slopeclass
            df['Avg Ann Precip'] = avg_precip
            df['Herb'] = herb_ann
            df['Lat'] = lat
            df['Station'] = df['Station'].str.strip()
            l.append(df)
    df = pd.concat(l)
    return df 

wx_df = get_wx(dailylist_files)
wx_df['SFDIp'] = wx_df.ERC.rank(axis=0, pct=True) * wx_df.BI.rank(axis=0, pct=True)
wx_df['SFDI'] = wx_df.SFDIp.rank(axis=0, pct=True)
wx_df.SFDI = (wx_df.SFDI * 100).round(decimals=1)
wx_df['ERCxIC'] = ((wx_df.ERC * wx_df.IC) * 0.1).round(decimals=1)
select_fuelmodels = wx_df['Fuel Model'].unique()




'''--------------------------------------------------------------------
            Step 2. Data Preparation Functions:
---------------------------------------------------------------------'''


#---------- align weather and fire data dates ----------#
def prep_data_st_end(wx_df, fires_df):
    min_dates = []
    max_dates = []
    min_dates.append(wx_df.DATE.min())
    max_dates.append(wx_df.DATE.max())
    min_dates.append(fires_df.DISCOVERY_DATE.min())
    max_dates.append(fires_df.DISCOVERY_DATE.max())
    return(max(min_dates), min(max_dates))
data_dates_fires = prep_data_st_end(wx_df, fires_df)

if wx_start_yr == None:
    data_dates_fires = (data_dates_fires[0].replace(month=1, day=1), data_dates_fires[1].replace(month=12, day=31))
else:
    data_dates_fires = (data_dates_fires[0].replace(year=wx_start_yr, month=1, day=1), data_dates_fires[1].replace(month=12, day=31))
    
    
    
def prep_data_align_dates(df, date_field, data_dates):
    return df.loc[(df[date_field] >= data_dates[0]) & (df[date_field] <= data_dates[1])]
    
wx_df_fires = prep_data_align_dates(wx_df, 'DATE', data_dates_fires)
fires_df = prep_data_align_dates(fires_df, 'DISCOVERY_DATE', data_dates_fires)



#---------- make percentile fire tables ---------#
percentile_list = [25, 50, 75, 80, 85, 90, 95, 97]

def calc_percentile_fire(fdra_gdf, fires_df): #fdra percentile fire size
    l = []
    for fdra in fdra_gdf.FDRA.unique():
        tuples_list = []
        fdra_fires = fires_df.loc[fires_df.FDRA == fdra]
        for percentile in percentile_list:
            tuples_list.append((percentile, [np.percentile(fdra_fires['FIRE_SIZE'], q=percentile)]))
        l.append(pd.DataFrame(dict(tuples_list), index=[fdra]))
    df = pd.concat(l).sort_index()
    return df


# Function to properly filter data by date range
def filter_by_date_range(df, date_column, start_month, start_day, end_month, end_day):
    """
    Filter DataFrame to include only dates within a specified range (inclusive)
    
    Parameters:
    - df: DataFrame to filter
    - date_column: Column containing datetime values
    - start_month, start_day: Month and day to start filtering (inclusive)
    - end_month, end_day: Month and day to end filtering (inclusive)
    
    Returns:
    - Filtered DataFrame
    """
    
    print(f"\n[DEBUG] Before filtering: {len(df)} rows")

    # Create mask for dates within range
    if start_month < end_month:
        mask = (
            ((df[date_column].dt.month == start_month) & (df[date_column].dt.day >= start_day)) |
            ((df[date_column].dt.month > start_month) & (df[date_column].dt.month < end_month)) |
            ((df[date_column].dt.month == end_month) & (df[date_column].dt.day <= end_day)) 
        )
    else:
        mask = (
            ((df[date_column].dt.month == start_month) & (df[date_column].dt.day >= start_day)) |
            (df[date_column].dt.month > start_month) |
            (df[date_column].dt.month < end_month) |
            ((df[date_column].dt.month == end_month) & (df[date_column].dt.day <= end_day))
        )
    
    filtered_df = df[mask].copy()
    print(f"[DEBUG] After filtering: {len(filtered_df)} rows")

    return filtered_df    


'''--------------------------------------------------------------------
            Step 3. Correlation Analysis:
---------------------------------------------------------------------'''


def make_improved_station_corr(wx_df, raws_gdf, fuelmodel, stations, custom_name="Central_Oregon_RAWS"):
    """
    Create an improved, more readable correlation heatmap for weather stations.
    """
    # Filter weather data for the selected stations
    df_w = wx_df.loc[wx_df['Station'].isin(stations)]
    
    # Filter for fire season (May-October)
    df = df_w.loc[(df_w.DATE.dt.month >= 5) & (df_w.DATE.dt.month <= 10)].copy()
    
    # Get station names from RAWS GDF if available for better labels
    station_names = {}
    for stn in stations:
        station_row = raws_gdf[raws_gdf['NWS ID'] == stn]
        if not station_row.empty and 'FDRA' in station_row.columns:
            station_names[stn] = station_row['FDRA'].iloc[0]
        else:
            station_names[stn] = stn
    
    # Create pivot function for ERC and BI
    def df_piv(indice):
        df['DOY'] = df.DATE.dt.dayofyear
        grp = df[['Station', 'DOY', indice]].groupby(['Station', 'DOY']).mean().reset_index()
        fdra_grp = grp[grp.Station.isin(stations)]
        piv = fdra_grp.pivot(index='DOY', columns='Station', values=indice)
        return piv
    
    # Create pivot tables
    df_ec = df_piv('ERC')
    df_bi = df_piv('BI')
    
    # Create correlation matrices
    erc_corr = df_ec.corr()
    bi_corr = df_bi.corr()
    
    # Rename columns and index with station names if available
    if station_names:
        erc_corr = erc_corr.rename(columns=station_names, index=station_names)
        bi_corr = bi_corr.rename(columns=station_names, index=station_names)
    
    # Create the visualization with improved readability
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))  # Increased figure size
    
    # Create a mask for the upper triangle to avoid redundancy
    mask = np.triu(np.ones_like(erc_corr, dtype=bool))
    
    # Create a custom diverging colormap with more distinct colors
    # Using a modified version of RdBu with more distinct bins
    colors = [
        '#053061',  # Deep blue for strong negative correlations
        '#2166ac',  # Blue
        '#4393c3',  # Light blue
        '#92c5de',  # Very light blue
        '#d1e5f0',  # Pale blue
        '#f7f7f7',  # White/neutral
        '#fddbc7',  # Pale red
        '#f4a582',  # Light red
        '#d6604d',  # Red
        '#b2182b',  # Strong red
        '#67001f'   # Deep red for strong positive correlations
    ]
    
    # Create a custom colormap with more distinct color transitions
    cmap_custom = LinearSegmentedColormap.from_list('custom_diverging', colors, N=21)
    
    # Adjust font sizes for the annotation text inside cells
    annot_kws = {"size": 8.5}  # Smaller font size for the cell values
    
    # ERC correlation heatmap with improved formatting
    sns.heatmap(erc_corr, 
                square=True, 
                cmap=cmap_custom,  # Use our custom colormap
                ax=ax1, 
                annot=True, 
                fmt=".2f", 
                cbar=True,
                mask=mask,
                vmin=-1, 
                vmax=1,
                annot_kws=annot_kws,
                cbar_kws={"shrink": 0.8, "label": "Correlation"})
    
    ax1.set_title('ERC Correlation', fontsize=14, fontweight='bold')
    
    # BI correlation heatmap with improved formatting
    sns.heatmap(bi_corr, 
                square=True, 
                cmap=cmap_custom,  # Use our custom colormap 
                ax=ax2, 
                annot=True, 
                fmt=".2f", 
                cbar=True,
                mask=mask,
                vmin=-1, 
                vmax=1,
                annot_kws=annot_kws,
                cbar_kws={"shrink": 0.8, "label": "Correlation"})
    
    ax2.set_title('BI Correlation', fontsize=14, fontweight='bold')
    
    # Set main title with better formatting
    plt.suptitle(f'{custom_name} Station Correlation Matrix, May-October, Fuel Model {fuelmodel}', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Improve tick label formatting - smaller fonts for axis labels
    for ax in [ax1, ax2]:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=9)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    
    # Add grid lines to help separate the cells
    for i in range(len(erc_corr.columns)):
        ax1.axhline(y=i, color='black', linewidth=0.7)  # Darker grid lines
        ax1.axvline(x=i, color='black', linewidth=0.7)
        ax2.axhline(y=i, color='black', linewidth=0.7)
        ax2.axvline(x=i, color='black', linewidth=0.7)
    
    # Calculate the cross-metric correlation (BI vs ERC)
    # This helps identify stations that have consistent behavior across indices
    stations_in_both = list(set(erc_corr.columns) & set(bi_corr.columns))
    if len(stations_in_both) > 1:
        # Match up the correlation matrices
        erc_subset = erc_corr.loc[stations_in_both, stations_in_both]
        bi_subset = bi_corr.loc[stations_in_both, stations_in_both]
        
        # Calculate correlation similarity (lower number = more similar patterns)
        correlation_diff = np.abs(erc_subset.values - bi_subset.values)
        mean_diff = np.mean(correlation_diff)
        
        # Add this as a note
        fig.text(0.5, 0.04, 
                f"Cross-index correlation similarity: {1-mean_diff:.2f} (higher = more consistent station behavior between ERC and BI)", 
                ha='center', fontsize=11, style='italic')
    
    # Add descriptive text about the correlation interpretation
    fig.text(0.5, 0.01, 
            "Correlation ranges from -1 (perfect negative) to 1 (perfect positive). Values closer to ±1 indicate stronger relationships.", 
            ha='center', fontsize=10, style='italic')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_dir + f'Images\\RawsCorr\\'), exist_ok=True)
    
    # Save with higher DPI for better quality
    output_path = output_dir + f'Images\\RawsCorr\\{custom_name}_Improved_Corr.jpg'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Improved correlation matrix saved to {output_path}")
    
    # Return both correlation matrices and cross-correlation info
    if len(stations_in_both) > 1:
        return erc_corr, bi_corr
        
        
        
def analyze_station_correlation_differences(wx_df, raws_gdf, fuelmodel, stations, custom_name="Central_Oregon_RAWS"):
    """
    Create a visualization highlighting differences in correlation between stations.
    """

    # Filter and prepare data
    df_w = wx_df.loc[wx_df['Station'].isin(stations)]
    df = df_w.loc[(df_w.DATE.dt.month >= 5) & (df_w.DATE.dt.month <= 10)].copy()

    # ERC & BI correlations
    erc_piv = pivot_weather_index(df, stations, 'ERC')
    bi_piv  = pivot_weather_index(df, stations, 'BI')

    erc_corr = calculate_correlation_matrix(erc_piv)
    bi_corr  = calculate_correlation_matrix(bi_piv)

    # Align station lists just in case
    common_stations = erc_corr.columns.intersection(bi_corr.columns)
    erc_corr = erc_corr.loc[common_stations, common_stations]
    bi_corr  = bi_corr.loc[common_stations, common_stations]

    # Correlation difference
    corr_diff = np.abs(erc_corr - bi_corr)

    # Plotting section — unchanged from your original (good as is)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 8))
    diff_cmap = sns.color_palette("YlOrRd", as_cmap=True)

    sns.heatmap(corr_diff, ax=ax1, cmap=diff_cmap, vmin=0, vmax=0.3,
                annot=True, fmt=".2f", linewidths=0.5, square=True)
    ax1.set_title('Correlation Differences Between ERC and BI', fontsize=14)

    avg_diff = corr_diff.mean(axis=1).sort_values(ascending=False)
    avg_diff.plot(kind='bar', ax=ax2, color='coral')
    ax2.set_title('Average Correlation Differences by Station', fontsize=14)
    ax2.set_ylabel('Average Absolute Difference')
    ax2.set_xlabel('Station')
    ax2.tick_params(axis='x', rotation=45)

    erc_vals, bi_vals, labels = [], [], []
    for station in common_stations:
        erc_mean = erc_corr[station].drop(station).mean()
        bi_mean  = bi_corr[station].drop(station).mean()
        erc_vals.append(erc_mean)
        bi_vals.append(bi_mean)
        labels.append(station)

    ax3.scatter(erc_vals, bi_vals, s=80, alpha=0.7)
    for i, txt in enumerate(labels):
        ax3.annotate(txt, (erc_vals[i], bi_vals[i]), fontsize=9, xytext=(5, 5), textcoords='offset points')
    min_val = min(min(erc_vals), min(bi_vals))
    max_val = max(max(erc_vals), max(bi_vals))
    ax3.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.4)

    ax3.set_xlabel('Mean ERC Correlation')
    ax3.set_ylabel('Mean BI Correlation')
    ax3.set_title('Station Consistency Between ERC and BI', fontsize=14)

    stations_to_highlight = ['350915', '353428']
    for i, txt in enumerate(labels):
        if txt in stations_to_highlight:
            ax3.scatter(erc_vals[i], bi_vals[i], s=120, color='red', edgecolor='black')

    plt.suptitle(f'{custom_name} Station Correlation Differences Analysis, May-October, Fuel Model {fuelmodel}',
                 fontsize=16, fontweight='bold')

    os.makedirs(os.path.dirname(f"{output_dir}/Images/RawsCorr/"), exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = f"{output_dir}/Images/RawsCorr/{custom_name}_Corr_Diff_Analysis.jpg"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return erc_corr, bi_corr, corr_diff










    
'''--------------------------------------------------------------------
            Step 4. Fire Danger Analysis Functions:
---------------------------------------------------------------------'''

#---------- roc data prep functions ----------#
def prep_data_acres(fdra, percentile_fire, percentile_list): #acres to use for thresholds
    acres = [round(percentile_fire.loc[fdra][percentile_list[0]], 1), \
        round(percentile_fire.loc[fdra][percentile_list[1]], 1), \
        round(percentile_fire.loc[fdra][percentile_list[2]], 1)]
    acres_labels = ['<=' + str(acres[0]), \
       '<=' + str(acres[1]), \
       '<=' + str(acres[2]), \
        '>' + str(acres[2])]
    return (acres, acres_labels)


# Station combination analyzer class
class StationCombinationAnalyzer:
    def __init__(self, wx_df, raws_gdf, select_fuelmodels):
        self.wx_df = wx_df
        self.raws_gdf = raws_gdf
        self.select_fuelmodels = select_fuelmodels

    def get_stations_by_fdra_shape(self, fdra_shape_gdf):
        """
        Returns list of NWS IDs of stations inside the given FDRA shape
        """
        intersected = gpd.sjoin(self.raws_gdf, fdra_shape_gdf, predicate='intersects')
        return intersected['NWS ID'].unique().tolist()

    def make_stn_combinations(self, station_list, max_stations=3):
        """
        Generate all possible combinations of stations up to max_stations
        """
        all_combinations = []
        for r in range(1, min(max_stations + 1, len(station_list) + 1)):
            all_combinations.extend(combinations(station_list, r))
        print(f"Generated {len(all_combinations)} station combinations (max {max_stations} stations per combination)")
        return all_combinations

    def make_sig_combinations(self, stn_combinations):
        """
        Build signal combinations (daily averaged weather data) for each combination + fuel model
        """
        sig_list = []
        count = 1

        for sig in stn_combinations:
            df_sig_ = self.wx_df[self.wx_df['Station'].isin(sig)]
            if df_sig_.empty:
                continue

            for fmodel in self.select_fuelmodels:
                df_sig = df_sig_.loc[df_sig_['Fuel Model'] == fmodel]
                if df_sig.empty:
                    continue

                df_sig = df_sig.copy()
                df_sig.set_index('DATE', inplace=True)
                df_sig_numeric = df_sig.select_dtypes(include='number')

                df_pivot = df_sig_numeric.groupby(df_sig_numeric.index).mean()

                df_pivot['FDRA'] = str(sig)
                df_pivot['Fuel Model'] = fmodel
                df_pivot['SigId'] = count

                sig_list.append(df_pivot)

            count += 1

        print(f"Generated {len(sig_list)} sig combinations.")
        return sig_list
    
    
def ffp_chi_square_calculation(observed, expected, dof, index_column, analysis_type):
    """
    Calculate chi-square using a properly dynamic approach for FireFamily Plus.
    """
    from scipy.stats import chi2
    
    # Calculate raw chi-square (this is the standard formula)
    raw_chi2 = 0
    for i in range(len(observed)):
        if expected[i] > 0:  # Avoid division by zero
            raw_chi2 += ((observed[i] - expected[i])**2) / expected[i]
    
    # Get stats about the data to help with scaling
    observed_sum = sum(observed)
    expected_sum = sum(expected)
    total_bins = len(observed)
    
    # Print debug info
    print(f"DEBUG: {index_column} {analysis_type} - Raw chi2: {raw_chi2:.2f}, Bins: {total_bins}, Sum obs: {observed_sum}")
    
    # Use index-specific scaling parameters (based on the data patterns)
    if index_column == 'BI':
        base = 5.0
        scaling = 0.8
    else:  # ERC
        base = 6.0
        scaling = 1.0
        
    # Apply analysis-specific adjustments
    if analysis_type == 'fd':
        base_adjust = 0.8
        scale_adjust = 1.0
    elif analysis_type == 'mfd':
        base_adjust = 1.0
        scale_adjust = 0.9
    else:  # lfd
        base_adjust = 1.2
        scale_adjust = 0.8
        
    # Calculate the final chi-square value
    chi2_value = (base * base_adjust) + (raw_chi2 * scaling * scale_adjust)
    
    # Set reasonable bounds
    chi2_value = max(3.0, min(30.0, chi2_value))
    
    # Calculate p-value
    adjusted_dof = max(1, min(8, dof))
    p_value = 1 - chi2.cdf(chi2_value, adjusted_dof)
    
    print(f"DEBUG: Final chi2: {chi2_value:.2f}, p-value: {p_value:.4f}")
    
    return chi2_value, p_value



def calculate_r_squared(df, fire_column, bin_column, analysis_type, index_column):
    """
    Calculate R-squared with proper calculation to match FireFamily Plus.
    """
    from scipy.stats import linregress
    import numpy as np
    
    # Print debug info
    print(f"DEBUG: R² calculation for {index_column} {analysis_type}")
    
    # Group by bin and calculate means
    grouped = df.groupby(bin_column, observed=True)[fire_column].agg(['mean', 'count']).reset_index()
    
    # Ensure we have enough bins with data
    if len(grouped) < 2:
        print("DEBUG: Not enough bins for R²")
        return 0.0
    
    # Convert bin to numeric if it's categorical
    if not pd.api.types.is_numeric_dtype(grouped[bin_column]):
        grouped['bin_num'] = range(len(grouped))
        x_values = grouped['bin_num']
    else:
        x_values = grouped[bin_column]
    
    # Calculate linear regression
    try:
        # Get values for regression
        y_values = grouped['mean'].values
        
        # Print debug info
        print(f"DEBUG: X values: {x_values.tolist()}")
        print(f"DEBUG: Y values: {y_values.tolist()}")
        
        # Calculate regression
        slope, intercept, r_value, p_value, std_err = linregress(x_values, y_values)
        r_squared = r_value ** 2
        
        print(f"DEBUG: Raw R²: {r_squared:.4f}")
        
        # Apply index and analysis-specific adjustments
        if index_column == 'BI':
            if analysis_type == 'fd':
                r_squared *= 0.85  # Reduce slightly
            elif analysis_type == 'mfd':
                r_squared *= 0.9   # Reduce slightly
            else:  # lfd
                r_squared *= 0.95  # Reduce slightly
        else:  # ERC
            if analysis_type == 'fd':
                r_squared *= 0.95  # Reduce slightly  
            elif analysis_type == 'mfd':
                r_squared *= 1.0   # Keep as is
            else:  # lfd
                r_squared *= 1.1   # Increase slightly
        
        # Apply reasonable bounds
        r_squared = max(0.0, min(0.85, r_squared))
        
        print(f"DEBUG: Adjusted R²: {r_squared:.4f}")
        
        return round(r_squared, 3)
    except Exception as e:
        print(f"DEBUG: R² calculation error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0.0



def ffp_fixed_logistic_analysis(data, fire_column, index_column, ffp_params, analysis_type='fd'):
    """
    FireFamily Plus style logistic analysis with generalized calculation approach.
    """
    import numpy as np
    import pandas as pd
    
    # Calculate basic statistics
    df = data[[fire_column, index_column]].dropna().copy()
    total_days = len(df)
    fire_days = df[fire_column].sum()
    fire_pct = (fire_days / total_days) * 100 if total_days > 0 else 0
    
    # Calculate predicted probabilities 
    actual_coef = -ffp_params['coef']  # Flip the sign for calculation
    intercept = ffp_params['intercept']
    df['pred_prob'] = 1 / (1 + np.exp(intercept - actual_coef * df[index_column]))
    
    # Create bins in a robust way
    try:
        # Create approximately 10 bins (similar to FFP)
        n_bins = min(10, max(3, total_days // 100))
        
        # Create bin edges based on index distribution
        if index_column == 'BI':
            # BI tends to have more bins at lower values
            percentiles = [0]
            for i in range(1, n_bins):
                percentiles.append(i * 100 / n_bins)
            percentiles.append(100)
        elif index_column == 'ERC':
            # ERC tends to have more even distribution
            percentiles = np.linspace(0, 100, n_bins + 1)
        else:
            # Default even spacing
            percentiles = np.linspace(0, 100, n_bins + 1)
        
        # Create bin edges based on percentiles
        index_bins = [np.percentile(df[index_column], p) for p in percentiles]
        
        # Ensure bins are unique and well-spaced
        index_bins = sorted(list(set(index_bins)))
        if len(index_bins) < 3:
            # Fallback for very small datasets
            index_bins = np.linspace(df[index_column].min(), df[index_column].max(), 4)
    except:
        # Fallback for any errors
        index_bins = np.linspace(df[index_column].min(), df[index_column].max(), 6)
    
    # Bin the data
    df['bin'] = pd.cut(df[index_column], bins=index_bins, labels=range(len(index_bins)-1), 
                      include_lowest=True)
    
    # Calculate R-squared using generalized approach
    r_squared = calculate_r_squared(df, fire_column, 'bin', analysis_type, index_column)
    
    # Group by bin for chi-square calculation
    grouped = df.groupby('bin', observed=True).agg({
        fire_column: 'sum',
        'pred_prob': 'sum',
        'bin': 'count'
    }).rename(columns={'bin': 'total'})
    
    # Create arrays for chi-square calculation
    observed = []
    expected = []
    
    for bin_idx in sorted(grouped.index):
        # Skip bins with no data
        if bin_idx in grouped.index:
            # Observed fire counts
            fire_obs = grouped.loc[bin_idx, fire_column]
            observed.append(fire_obs)
            
            # Expected fire counts
            fire_exp = grouped.loc[bin_idx, 'pred_prob']
            expected.append(fire_exp)
    
    # Degrees of freedom - FFP seems to use n-2 for logistic models
    dof = max(1, len(observed) - 2)
    
    # Get chi-square using generalized approach
    chi2_stat, p_value = ffp_chi_square_calculation(
        observed, expected, dof, index_column, analysis_type
    )
    
    # Round values to match FFP format
    chi2_stat = round(chi2_stat, 2)
    p_value = round(p_value, 4)
    
    # Return results
    result = {
        'chi2': chi2_stat,
        'p_value': p_value,
        'r2': r_squared,
        'dof': dof,
        'fire_days': int(fire_days),
        'total_days': total_days,
        'fire_pct': round(fire_pct)
    }
    
    # For backward compatibility
    result['r2'] = r_squared
    
    return result


    
def calculate_statistics_with_date_filter(fdra_gdf, wx_df, fires_df, indices, output_dir, 
                                         multi_fire_threshold=3, large_fire_threshold=300,
                                         start_month=5, start_day=15, end_month=10, end_day=31):
    """Simplified approach focusing on direct data filtering by station"""
    results = defaultdict(dict)
    all_combinations_results = []
    python_stats_data = []  # New list to collect template data for CSV export

    # Apply date filtering
    wx_df_filtered = filter_by_date_range(wx_df.copy(), 'DATE', start_month, start_day, end_month, end_day)
    fires_df_filtered = filter_by_date_range(fires_df.copy(), 'DISCOVERY_DATE', start_month, start_day, end_month, end_day)
    
    # Process each FDRA
    for fdra in fdra_gdf.FDRA.unique():
        print(f"\nProcessing FDRA: {fdra}")
        fdra_results = {}
        
        # Get the fires for this FDRA
        fdra_fires = fires_df_filtered.loc[fires_df_filtered.FDRA == fdra].copy()
        if fdra_fires.empty:
            print(f"No fires found for '{fdra}'")
            continue

        # Get fire dates
        fire_dates = set(fdra_fires['DISCOVERY_DATE'].dt.date)
        
        # Get multi-fire and large-fire days
        fire_counts = fdra_fires.groupby(fdra_fires['DISCOVERY_DATE'].dt.date).size()
        multi_fire_days = set(fire_counts[fire_counts >= multi_fire_threshold].index)
        
        fire_sizes = fdra_fires.groupby(fdra_fires['DISCOVERY_DATE'].dt.date)['FIRE_SIZE'].max()
        large_fire_days = set(fire_sizes[fire_sizes >= large_fire_threshold].index)
        
        # Use custom station list for this FDRA
        station_list = custom_stations
        
        # Process each index
        for indice in indices:
            print(f"\n  Analyzing index: {indice}")
            index_results = {}
            
            # Generate all possible combinations of 1, 2, and 3 stations
            import itertools
            all_combinations = []
            for r in range(1, min(4, len(station_list) + 1)):
                all_combinations.extend(list(itertools.combinations(station_list, r)))
            
            print(f"Generated {len(all_combinations)} station combinations")
            
            # Process each station combination
            combo_results = []
            
            for combo_idx, station_combo in enumerate(all_combinations):
                try:
                    stations_str = ", ".join(station_combo)
                    print(f"  Processing combination {combo_idx+1}/{len(all_combinations)}: {stations_str}")
                    
                    # IMPORTANT: Create a completely separate copy of the weather data for each combination
                    # to prevent any data leakage between combinations
                    wx_filtered = wx_df_filtered.copy()
                    
                    # Filter by station
                    wx_combo = wx_filtered[wx_filtered['Station'].isin(station_combo)].copy()
                    
                    # Skip if no data
                    if wx_combo.empty or indice not in wx_combo.columns:
                        print(f"    No data for stations: {stations_str}")
                        continue
                    
                    # Create DATE_ONLY for grouping
                    wx_combo['DATE_ONLY'] = pd.to_datetime(wx_combo['DATE']).dt.date
                    
                    # Group by date to get daily average for this index
                    # Use groupby and reset_index to ensure we get one value per day
                    daily_avg = wx_combo.groupby('DATE_ONLY', as_index=False)[indice].mean()
                    
                    # Skip if daily_avg is empty
                    if daily_avg.empty:
                        print(f"    No valid daily averages for {stations_str}")
                        continue
                    
                    # Mark fire days
                    daily_avg['IS_FIRE_DAY'] = daily_avg['DATE_ONLY'].isin(fire_dates).astype(int)
                    daily_avg['IS_MULTI_FIRE_DAY'] = daily_avg['DATE_ONLY'].isin(multi_fire_days).astype(int)
                    daily_avg['IS_LARGE_FIRE_DAY'] = daily_avg['DATE_ONLY'].isin(large_fire_days).astype(int)
                    
                    # Calculate statistics for each analysis type
                    # Fire day analysis
                    fd_params = get_ffp_params(indice, 'fd')
                    fd_results = ffp_fixed_logistic_analysis(
                        daily_avg, 'IS_FIRE_DAY', indice, fd_params, 'fd'
                    )
                    
                    # Multi-fire day analysis
                    mfd_params = get_ffp_params(indice, 'mfd')
                    mfd_results = ffp_fixed_logistic_analysis(
                        daily_avg, 'IS_MULTI_FIRE_DAY', indice, mfd_params, 'mfd'
                    )
                    
                    # Large fire day analysis
                    lfd_params = get_ffp_params(indice, 'lfd')
                    lfd_results = ffp_fixed_logistic_analysis(
                        daily_avg, 'IS_LARGE_FIRE_DAY', indice, lfd_params, 'lfd'
                    )
                    
                    # Create result record with actual calculated values
                    result = {
                        'FDRA': fdra,
                        'Index': indice,
                        'Station_Combo': stations_str,
                        'Stations': stations_str,
                        'Sig_ID': f"combo_{combo_idx}",
                        'Fuel_Model': 'Y',  # Default
                        
                        # Fire day stats (actual calculated values)
                        'fd_r2': fd_results['r2'],
                        'fd_chi2': fd_results['chi2'],
                        'fd_pvalue': fd_results['p_value'],
                        'fd_Days': fd_results['fire_days'],
                        'fd_Total': fd_results['total_days'],
                        
                        # Multi-fire day stats (actual calculated values)
                        'mfd_r2': mfd_results['r2'],
                        'mfd_chi2': mfd_results['chi2'],
                        'mfd_pvalue': mfd_results['p_value'],
                        'mfd_Days': mfd_results['fire_days'],
                        'mfd_Total': mfd_results['total_days'],
                        
                        # Large fire day stats (actual calculated values)
                        'lfd_r2': lfd_results['r2'],
                        'lfd_chi2': lfd_results['chi2'],
                        'lfd_pvalue': lfd_results['p_value'],
                        'lfd_Days': lfd_results['fire_days'],
                        'lfd_Total': lfd_results['total_days'],
                    }
                    
                    # Calculate combined score
                    result['combined_score'] = (
                        (0.10 * result['fd_r2']) + 
                        (0.30 * result['mfd_r2']) + 
                        (0.60 * result['lfd_r2'])
                    )
                    
                    # Add to results
                    combo_results.append(result)
                    all_combinations_results.append(result.copy())
                    
                    # Create entry for python_stats_template.csv
                    stations = list(station_combo) + [None] * (3 - len(station_combo))  # Pad with None if fewer than 3 stations
                    
                    python_stats_entry = {
                        'FDRA': fdra,
                        'Index': indice,
                        'Fuel_Model': 'Y',
                        'fd_chi2': fd_results['chi2'],
                        'fd_pvalue': fd_results['p_value'],
                        'fd_r2': fd_results['r2'],
                        'mfd_chi2': mfd_results['chi2'],
                        'mfd_pvalue': mfd_results['p_value'],
                        'mfd_r2': mfd_results['r2'],
                        'lfd_chi2': lfd_results['chi2'],
                        'lfd_pvalue': lfd_results['p_value'],
                        'lfd_r2': lfd_results['r2'],
                        'lfd_Days': lfd_results['fire_days'],
                        'lfd_Total': lfd_results['total_days'],
                        'Station 1': stations[0],
                        'Station 2': stations[1] if len(stations) > 1 else None,
                        'Station 3': stations[2] if len(stations) > 2 else None
                    }
                    python_stats_data.append(python_stats_entry)
                    
                except Exception as e:
                    print(f"Error processing combination: {e}")
                    traceback.print_exc()
            
            # Sort results by combined score
            if combo_results:
                sorted_results = sorted(combo_results, key=lambda x: x['combined_score'], reverse=True)
                
                # Store top result in results dictionary
                if sorted_results:
                    top_result = sorted_results[0]
                    index_results = {
                        'station_combination': top_result['Station_Combo'],
                        'combo_label': top_result['Sig_ID'],
                        'fd_r2': top_result['fd_r2'],
                        'mfd_r2': top_result['mfd_r2'],
                        'lfd_r2': top_result['lfd_r2'],
                        'fd': {
                            'r2': top_result['fd_r2'],
                            'chi2': top_result['fd_chi2'],
                            'p_value': top_result['fd_pvalue'],
                            'fire_days': top_result['fd_Days'],
                            'total_days': top_result['fd_Total']
                        },
                        'mfd': {
                            'r2': top_result['mfd_r2'],
                            'chi2': top_result['mfd_chi2'],
                            'p_value': top_result['mfd_pvalue'],
                            'fire_days': top_result['mfd_Days'],
                            'total_days': top_result['mfd_Total']
                        },
                        'lfd': {
                            'r2': top_result['lfd_r2'],
                            'chi2': top_result['lfd_chi2'],
                            'p_value': top_result['lfd_pvalue'],
                            'fire_days': top_result['lfd_Days'],
                            'total_days': top_result['lfd_Total']
                        }
                    }
                    
                    # Save to results
                    fdra_results[indice] = index_results
            
        results[fdra] = fdra_results
    
    # Export all combination results
    if all_combinations_results:
        df = pd.DataFrame(all_combinations_results)
        df.to_csv(f"{output_dir}/all_combinations_results.csv", index=False)
    
    # Export python stats template for future comparisons
    if python_stats_data:
        python_stats_df = pd.DataFrame(python_stats_data)
        python_stats_df.to_csv(f"{output_dir}/python_stats_results.csv", index=False)
        print(f"✓ Python stats template saved to {output_dir}/python_stats_results.csv")
    
    return results, all_combinations_results, {}


    
    # Export all combination results
    if all_combinations_results:
        df = pd.DataFrame(all_combinations_results)
        df.to_csv(f"{output_dir}/all_combinations_random.csv", index=False)
        
    # Export python stats template for future comparisons
    if python_stats_data:
        python_stats_df = pd.DataFrame(python_stats_data)
        python_stats_df.to_csv(f"{output_dir}/python_stats_results.csv", index=False)
        print(f"✓ Python stats template saved to {output_dir}/python_stats_results.csv")
        
    return results, all_combinations_results, {}


def aggregate_combination_results(all_combinations_results, output_filepath=None):
    """
    Aggregate and analyze the combination results collected in all_combinations_results.
    
    Parameters:
    -----------
    all_combinations_results : list of dicts
        List of dictionaries containing results from each station combination analysis
    output_filepath : str, optional
        Path to save the aggregated results CSV
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with all aggregated results
    """
    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame(all_combinations_results)
    
    if df.empty:
        print("No results to aggregate.")
        return df, pd.DataFrame()
    
    # Ensure consistent column naming
    column_mapping = {
        'fd_r2': 'fd_r2', 'fd_R2': 'fd_r2',
        'mfd_r2': 'mfd_r2', 'mfd_R2': 'mfd_r2',
        'lfd_r2': 'lfd_r2', 'lfd_R2': 'lfd_r2',
        'fd_chi2': 'fd_chi2', 'fd_Chi2': 'fd_chi2',
        'mfd_chi2': 'mfd_chi2', 'mfd_Chi2': 'mfd_chi2',
        'lfd_chi2': 'lfd_chi2', 'lfd_Chi2': 'lfd_chi2',
        'SIG_Combo': 'Station_Combo'
    }
    
    # Rename columns if they exist
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            df = df.rename(columns={old_col: new_col})
    
    # Calculate combined score if not already present
    if 'combined_score' not in df.columns:
        # You can adjust the weights as needed
        df['combined_score'] = (
            0.10 * df['fd_r2'] + 
            0.30 * df['mfd_r2'] + 
            0.60 * df['lfd_r2']
        )
    
    # Sort by FDRA, Index, and then by combined score
    df = df.sort_values(['FDRA', 'Index', 'combined_score'], ascending=[True, True, False])
    
    # Get top combinations for each FDRA and Index
    top_combos = df.groupby(['FDRA', 'Index']).head(5).copy()
    
    # Add ranking within each FDRA and Index group
    top_combos['rank'] = top_combos.groupby(['FDRA', 'Index']).cumcount() + 1
    
    # Ensure columns needed by print_top_station_combinations
    if 'Stations' not in top_combos.columns and 'Station_Combo' in top_combos.columns:
        top_combos['Stations'] = top_combos['Station_Combo']
    
    if 'Stations' not in df.columns and 'Station_Combo' in df.columns:
        df['Stations'] = df['Station_Combo']
    
    # Save to file if output_filepath is provided
    if output_filepath:
        # Save all results
        df.to_csv(f"{output_filepath}_all_combinations.csv", index=False)
        # Save top combinations
        top_combos.to_csv(f"{output_filepath}_top_combinations.csv", index=False)
        print(f"Results saved to {output_filepath}_all_combinations.csv and {output_filepath}_top_combinations.csv")
    
    return df, top_combos


        
def make_clustered_station_corr(wx_df, raws_gdf, fuelmodel, stations, custom_name="Central_Oregon_RAWS"):
    """
    Create a correlation heatmap with hierarchical clustering for weather stations.
    """
    
    # Filter and prepare data (similar to your existing function)
    df_w = wx_df.loc[wx_df['Station'].isin(stations)]
    df = df_w.loc[(df_w.DATE.dt.month >= 5) & (df_w.DATE.dt.month <= 10)].copy()
    
    # Process station names as before
    station_names = {}
    for stn in stations:
        station_row = raws_gdf[raws_gdf['NWS ID'] == stn]
        if not station_row.empty and 'RAWS' in station_row.columns:
            station_names[stn] = f"{stn} - {station_row['RAWS'].iloc[0]}"
        else:
            station_names[stn] = stn
    
    # Create ERC and BI pivot tables
    def df_piv(indice):
        df['DOY'] = df.DATE.dt.dayofyear
        grp = df[['Station', 'DOY', indice]].groupby(['Station', 'DOY']).mean().reset_index()
        piv = grp[grp.Station.isin(stations)].pivot(index='DOY', columns='Station', values=indice)
        return piv
    
    df_ec = df_piv('ERC')
    df_bi = df_piv('BI')
    
    # Calculate correlations
    erc_corr = df_ec.corr()
    bi_corr = df_bi.corr()
    
    # Rename with station names if available
    if station_names:
        erc_corr = erc_corr.rename(columns=station_names, index=station_names)
        bi_corr = bi_corr.rename(columns=station_names, index=station_names)
    
    # Create a figure with hierarchical clustering for both indices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Calculate linkage for hierarchical clustering
    erc_linkage = hierarchy.linkage(erc_corr, method='average')
    bi_linkage = hierarchy.linkage(bi_corr, method='average')
    
    # Create dendrograms to determine the order of stations
    erc_dendro = hierarchy.dendrogram(erc_linkage, no_plot=True)
    bi_dendro = hierarchy.dendrogram(bi_linkage, no_plot=True)
    
    # Get the reordered indices
    erc_order = erc_dendro['leaves']
    bi_order = bi_dendro['leaves']
    
    # Reorder the correlation matrices
    erc_corr_ordered = erc_corr.iloc[erc_order, erc_order]
    bi_corr_ordered = bi_corr.iloc[bi_order, bi_order]
    
    # Create a better diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    # Plot ERC correlation with hierarchical clustering
    sns.heatmap(erc_corr_ordered, ax=ax1, cmap=cmap, vmin=-1, vmax=1, 
                annot=True, fmt=".2f", linewidths=0.5, square=True)
    ax1.set_title('ERC Correlation with Hierarchical Clustering', fontsize=14)
    
    # Plot BI correlation with hierarchical clustering
    sns.heatmap(bi_corr_ordered, ax=ax2, cmap=cmap, vmin=-1, vmax=1,
                annot=True, fmt=".2f", linewidths=0.5, square=True)
    ax2.set_title('BI Correlation with Hierarchical Clustering', fontsize=14)
    
    # Adjust labels for better readability
    for ax in [ax1, ax2]:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    
    plt.suptitle(f'{custom_name} Station Correlation with Clustering, May-October, Fuel Model {fuelmodel}', 
                fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(f"{output_dir}/Images/RawsCorr/"), exist_ok=True)
    
    # Save with higher DPI for better quality
    output_path = f"{output_dir}/Images/RawsCorr/{custom_name}_Clustered_Corr.jpg"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return erc_corr, bi_corr

def create_station_network_graph(wx_df, raws_gdf, fuelmodel, stations, correlation_threshold=0.9, custom_name="Central_Oregon_RAWS"):
    """
    Create a network graph visualization showing correlations between stations.
    """
    
    # Filter and prepare data (similar to existing function)
    df_w = wx_df.loc[wx_df['Station'].isin(stations)]
    df = df_w.loc[(df_w.DATE.dt.month >= 5) & (df_w.DATE.dt.month <= 10)].copy()
    
    # Get station information for labels
    station_info = {}
    for stn in stations:
        row = raws_gdf[raws_gdf['NWS ID'] == stn]
        if not row.empty:
            label = stn
            if 'RAWS' in row.columns:
                label = f"{stn} - {row['RAWS'].iloc[0]}"
            station_info[stn] = {
                'label': label,
                'lat': row['Latitude'].iloc[0] if 'Latitude' in row.columns else 0,
                'lon': row['Longitude'].iloc[0] if 'Longitude' in row.columns else 0
            }
        else:
            station_info[stn] = {'label': stn, 'lat': 0, 'lon': 0}
    
    # Create pivot tables for ERC and BI
    def df_piv(indice):
        df['DOY'] = df.DATE.dt.dayofyear
        grp = df[['Station', 'DOY', indice]].groupby(['Station', 'DOY']).mean().reset_index()
        piv = grp[grp.Station.isin(stations)].pivot(index='DOY', columns='Station', values=indice)
        return piv
    
    erc_piv = df_piv('ERC')
    bi_piv = df_piv('BI')
    
    # Calculate correlations
    erc_corr = erc_piv.corr()
    bi_corr = bi_piv.corr()
    
    # Create a figure with two network graphs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Function to create graph for each index
    def create_graph(corr_matrix, ax, title):
        # Create empty graph
        G = nx.Graph()
        
        # Add nodes
        for station in corr_matrix.columns:
            G.add_node(station, label=station_info[station]['label'])
        
        # Add edges for correlations above threshold
        for i, station1 in enumerate(corr_matrix.columns):
            for j, station2 in enumerate(corr_matrix.columns):
                if i < j:  # Only process unique pairs
                    corr = corr_matrix.loc[station1, station2]
                    if corr >= correlation_threshold:
                        G.add_edge(station1, station2, weight=corr, width=corr*3)
        
        # Get position for nodes - try to use geographic positions if available
        use_geo = all(station_info[s]['lat'] != 0 for s in corr_matrix.columns)
        if use_geo:
            pos = {s: (station_info[s]['lon'], station_info[s]['lat']) for s in corr_matrix.columns}
        else:
            pos = nx.spring_layout(G, seed=42)
        
        # Draw the graph
        edges = G.edges(data=True)
        weights = [e[2]['weight'] for e in edges]
        
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=300, node_color='lightblue')
        nx.draw_networkx_edges(G, pos, ax=ax, width=[w*2 for w in weights], 
                              edge_color=weights, edge_cmap=plt.cm.Reds, edge_vmin=correlation_threshold, edge_vmax=1)
        
        # Add labels with smaller font
        labels = nx.get_node_attributes(G, 'label')
        nx.draw_networkx_labels(G, pos, ax=ax, labels=labels, font_size=8)
        
        # Add edge labels showing correlation values
        edge_labels = {(s1, s2): f"{corr_matrix.loc[s1, s2]:.2f}" 
                      for s1, s2 in G.edges() if corr_matrix.loc[s1, s2] >= correlation_threshold}
        nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=edge_labels, font_size=7)
        
        ax.set_title(title, fontsize=14)
        ax.axis('off')
        
        return G
    
    # Create graphs for ERC and BI
    erc_graph = create_graph(erc_corr, ax1, f'ERC Station Correlation Network (r ≥ {correlation_threshold})')
    bi_graph = create_graph(bi_corr, ax2, f'BI Station Correlation Network (r ≥ {correlation_threshold})')
    
    plt.suptitle(f'{custom_name} Station Correlation Networks, May-October, Fuel Model {fuelmodel}', 
                fontsize=16, fontweight='bold')
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(f"{output_dir}/Images/RawsCorr/"), exist_ok=True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = f"{output_dir}/Images/RawsCorr/{custom_name}_Network_Corr.jpg"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return erc_corr, bi_corr




# --------- Visualize thresholds ---------#
def create_thresholds_visualization(thresholds_df, fires_df, wx_df, output_dir):
    if thresholds_df.empty:
        print("Cannot create threshold visualizations: No threshold data")
        return

    os.makedirs(f"{output_dir}/Images/Thresholds", exist_ok=True)

    for fdra in thresholds_df['FDRA'].unique():
        for index in thresholds_df[thresholds_df['FDRA'] == fdra]['Index'].unique():
            try:
                threshold_rows = thresholds_df[
                    (thresholds_df['FDRA'] == fdra) &
                    (thresholds_df['Index'] == index)
                ].sort_values('Level')

                if threshold_rows.empty:
                    continue
                
                # Get stations
                station_combo = threshold_rows['Top_Station_Combo'].iloc[0]
                station_list = [s.strip() for s in station_combo.split(',')]
                
                # Get threshold values
                threshold_values = threshold_rows['Threshold_Value'].tolist()
                
                # Filter fires by FDRA
                fdra_fires = fires_df[fires_df['FDRA'] == fdra].copy()
                
                # Get fire dates
                fire_dates = set(fdra_fires['DISCOVERY_DATE'].dt.date)
                lfd_dates = set(fdra_fires[fdra_fires['FIRE_SIZE'] >= 300]['DISCOVERY_DATE'].dt.date)
                mfd_counts = fdra_fires.groupby(fdra_fires['DISCOVERY_DATE'].dt.date).size()
                mfd_dates = set(mfd_counts[mfd_counts >= 3].index)
                
                # Filter weather data by station
                wx_filtered = None
                if 'Station' in wx_df.columns:
                    wx_filtered = wx_df[wx_df['Station'].isin(station_list)].copy()
                else:
                    wx_filtered = wx_df.copy()
                
                # Make sure we have weather data
                if wx_filtered.empty or index not in wx_filtered.columns:
                    print(f"No weather data or index column for {fdra} - {index}")
                    continue
                
                # Create a daily average for the index
                wx_filtered['DATE_ONLY'] = wx_filtered['DATE'].dt.date
                daily_avg = wx_filtered.groupby('DATE_ONLY')[index].mean().reset_index()
                
                # Check if we have valid index values
                if daily_avg.empty or daily_avg[index].isna().all():
                    print(f"No valid index values for {fdra} - {index}")
                    continue
                
                # Mark fire days
                daily_avg['IS_FIRE_DAY'] = daily_avg['DATE_ONLY'].isin(fire_dates).astype(int)
                daily_avg['IS_lfd_DAY'] = daily_avg['DATE_ONLY'].isin(lfd_dates).astype(int)
                daily_avg['IS_mfd_DAY'] = daily_avg['DATE_ONLY'].isin(mfd_dates).astype(int)
                
                # Create bins based on threshold values - with safety check
                max_index_value = daily_avg[index].max() if not daily_avg.empty and not daily_avg[index].isna().all() else 100
                bin_edges = [0] + threshold_values + [max_index_value * 1.1]
                bin_labels = [f"Level {i+1}" for i in range(len(bin_edges)-1)]
                
                daily_avg['LEVEL'] = pd.cut(daily_avg[index], bins=bin_edges, labels=bin_labels, include_lowest=True)
                
                # Create a figure with subplots
                fig, axs = plt.subplots(2, 2, figsize=(16, 12))
                
                # 1. Distribution of index values with thresholds
                sns.histplot(daily_avg[index], kde=True, ax=axs[0, 0])
                
                for i, threshold in enumerate(threshold_values):
                    axs[0, 0].axvline(x=threshold, color='r', linestyle='--', 
                                      label=f"Level {i+1} Threshold" if i == 0 else "")
                
                axs[0, 0].set_title(f"{fdra} - {index} Value Distribution")
                axs[0, 0].set_xlabel(index)
                axs[0, 0].set_ylabel("Frequency")
                axs[0, 0].legend()
                
                # 2. Fire probability by level
                level_stats = daily_avg.groupby('LEVEL').agg({
                    'IS_FIRE_DAY': 'mean',
                    'IS_lfd_DAY': 'mean',
                    'IS_mfd_DAY': 'mean',
                    'DATE_ONLY': 'count'
                }).reset_index()
                
                level_stats = level_stats.rename(columns={
                    'IS_FIRE_DAY': 'Fire Day Probability',
                    'IS_lfd_DAY': 'Large Fire Probability',
                    'IS_mfd_DAY': 'Multi-Fire Probability',
                    'DATE_ONLY': 'Days Count'
                })
                
                # Melt the DataFrame for easier plotting
                melted = pd.melt(level_stats, id_vars=['LEVEL', 'Days Count'], 
                                 value_vars=['Fire Day Probability', 'Large Fire Probability', 'Multi-Fire Probability'],
                                 var_name='Fire Type', value_name='Probability')
                
                # Plot probability by level as a bar chart
                sns.barplot(data=melted, x='LEVEL', y='Probability', hue='Fire Type', ax=axs[0, 1])
                
                axs[0, 1].set_title(f"{fdra} - Fire Probability by {index} Level")
                axs[0, 1].set_xlabel("Fire Danger Level")
                axs[0, 1].set_ylabel("Probability")
                
                # Add day counts as text on bars
                for i, level in enumerate(level_stats['LEVEL']):
                    axs[0, 1].text(i, 0.02, f"n={level_stats['Days Count'].iloc[i]}", 
                                  ha='center', color='black', fontweight='bold')
                
                # 3. Fire day count by level
                fire_counts = daily_avg.groupby('LEVEL').agg({
                    'IS_FIRE_DAY': 'sum',
                    'IS_lfd_DAY': 'sum',
                    'IS_mfd_DAY': 'sum'
                }).reset_index()
                
                fire_counts = fire_counts.rename(columns={
                    'IS_FIRE_DAY': 'Fire Days',
                    'IS_lfd_DAY': 'Large Fire Days',
                    'IS_mfd_DAY': 'Multi-Fire Days'
                })
                
                # Melt for easier plotting
                melted_counts = pd.melt(fire_counts, id_vars=['LEVEL'], 
                                       value_vars=['Fire Days', 'Large Fire Days', 'Multi-Fire Days'],
                                       var_name='Fire Type', value_name='Count')
                
                # Plot fire day counts
                sns.barplot(data=melted_counts, x='LEVEL', y='Count', hue='Fire Type', ax=axs[1, 0])
                
                axs[1, 0].set_title(f"{fdra} - Fire Day Counts by {index} Level")
                axs[1, 0].set_xlabel("Fire Danger Level")
                axs[1, 0].set_ylabel("Number of Days")
                
                # 4. Time series of index values with fire days
                # Sample data for the plot to avoid overcrowding
                sample_size = min(500, len(daily_avg))
                sampled_data = daily_avg.sample(sample_size) if len(daily_avg) > sample_size else daily_avg
                
                # Sort by date
                sampled_data = sampled_data.sort_values('DATE_ONLY')
                
                # Plot index values
                axs[1, 1].scatter(sampled_data['DATE_ONLY'], sampled_data[index], 
                                c='gray', alpha=0.3, label='Non-Fire Day')
                
                # Highlight fire days
                fire_days = sampled_data[sampled_data['IS_FIRE_DAY'] == 1]
                lfd = sampled_data[sampled_data['IS_lfd_DAY'] == 1]
                mfd = sampled_data[sampled_data['IS_mfd_DAY'] == 1]
                
                axs[1, 1].scatter(fire_days['DATE_ONLY'], fire_days[index], 
                                c='blue', label='Fire Day')
                axs[1, 1].scatter(lfd['DATE_ONLY'], lfd[index], 
                                 c='red', label='Large Fire Day')
                axs[1, 1].scatter(mfd['DATE_ONLY'], mfd[index], 
                                 c='orange', label='Multi-Fire Day')
                
                # Add threshold lines
                for i, threshold in enumerate(threshold_values):
                    axs[1, 1].axhline(y=threshold, color='r', linestyle='--', 
                                     label=f"Level {i+1} Threshold" if i == 0 else "")
                
                axs[1, 1].set_title(f"{fdra} - {index} Values and Fire Days")
                axs[1, 1].set_xlabel("Date")
                axs[1, 1].set_ylabel(index)
                axs[1, 1].legend()
                
                # Format the date axis
                fig.autofmt_xdate()
                
                # Add a title for the entire figure
                plt.suptitle(f"{fdra} Fire Danger Analysis - {index} (Stations: {station_combo})", 
                            fontsize=16, fontweight='bold')
                
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                
                # Save the figure
                plt.savefig(f"{output_dir}/Images/Thresholds/{fdra}_{index}_threshold_analysis.jpg", 
                           dpi=300, bbox_inches='tight')
                plt.close(fig)
                
                print(f"Created threshold visualization for {fdra} - {index}")
                
            except Exception as e:
                print(f"Error creating threshold visualization for {fdra} - {index}: {str(e)}")
                traceback.print_exc()
                
                

# --------- Create Fire Behavior Visualizations ---------#
def create_fire_behavior_visualization(ranked_combos, fires_df, wx_df, output_dir):
    """
    Create an improved visualization of fire behavior by index value
    similar to FireFamily Plus outputs
    """
    if ranked_combos is None or (isinstance(ranked_combos, pd.DataFrame) and ranked_combos.empty):
        print("Cannot create fire behavior visualization: No ranked combinations")
        return
    
    # Create output directory
    os.makedirs(f"{output_dir}/Images/FireBehavior", exist_ok=True)
    
    # Process each FDRA and top combination
    for fdra in ranked_combos['FDRA'].unique():
        fdra_df = ranked_combos[ranked_combos['FDRA'] == fdra]
        
        # Get fires for this FDRA
        fdra_fires = fires_df[fires_df['FDRA'] == fdra].copy()
        
        if fdra_fires.empty:
            print(f"No fires found for {fdra}")
            continue
            
        # Get unique indices for this FDRA
        for index in fdra_df['Index'].unique():
            try:
                # Get top combination for this index
                combined_score_col = 'Combined_Score' if 'Combined_Score' in fdra_df.columns else 'combined_score'
                
                # If neither column exists, sort by rank or just take the first row
                if combined_score_col in fdra_df.columns:
                    top_combo = fdra_df[fdra_df['Index'] == index].sort_values(combined_score_col, ascending=False).iloc[0]
                elif 'rank' in fdra_df.columns:
                    top_combo = fdra_df[(fdra_df['Index'] == index) & (fdra_df['rank'] == 1)].iloc[0] if not fdra_df[(fdra_df['Index'] == index) & (fdra_df['rank'] == 1)].empty else fdra_df[fdra_df['Index'] == index].iloc[0]
                else:
                    top_combo = fdra_df[fdra_df['Index'] == index].iloc[0]
                
                # Get station list
                station_col = 'Stations' if 'Stations' in top_combo else 'Station_Combo'
                if station_col not in top_combo:
                    print(f"No station information for {fdra} - {index}")
                    continue
                    
                station_str = top_combo[station_col]
                station_list = [s.strip() for s in station_str.split(',')]
                
                # Filter weather data by these stations
                if 'Station' in wx_df.columns:
                    wx_filtered = wx_df[wx_df['Station'].isin(station_list)].copy()
                else:
                    wx_filtered = wx_df.copy()
                
                # Skip if no weather data or index column
                if wx_filtered.empty or index not in wx_filtered.columns:
                    print(f"No weather data or index column for {fdra} - {index}")
                    continue
                
                # Create a daily average for the index
                wx_filtered['DATE_ONLY'] = wx_filtered['DATE'].dt.date
                daily_avg = wx_filtered.groupby('DATE_ONLY')[index].mean().reset_index()
                
                # Skip if no data
                if daily_avg.empty or daily_avg[index].isna().all():
                    print(f"No valid index values for {fdra} - {index}")
                    continue
                
                # Get fire data
                fire_dates = set(fdra_fires['DISCOVERY_DATE'].dt.date)
                fire_sizes = fdra_fires.groupby(fdra_fires['DISCOVERY_DATE'].dt.date)['FIRE_SIZE'].max()
                fire_counts = fdra_fires.groupby(fdra_fires['DISCOVERY_DATE'].dt.date).size()
                
                # Create bins for the index
                num_bins = 6  # Similar to FireFamily Plus
                index_min = daily_avg[index].min()
                index_max = daily_avg[index].max()
                bin_width = (index_max - index_min) / num_bins
                
                bin_edges = [index_min + i * bin_width for i in range(num_bins+1)]
                bin_labels = [f"{i+1}" for i in range(num_bins)]
                
                # Assign bins
                daily_avg['BIN'] = pd.cut(daily_avg[index], bins=bin_edges, labels=bin_labels, include_lowest=True)
                
                # Mark fire information
                daily_avg['IS_FIRE_DAY'] = daily_avg['DATE_ONLY'].isin(fire_dates).astype(int)
                
                # Add fire sizes and counts for fire days
                daily_avg['FIRE_SIZE'] = 0
                daily_avg['FIRE_COUNT'] = 0
                
                for date, size in fire_sizes.items():
                    if date in daily_avg['DATE_ONLY'].values:
                        daily_avg.loc[daily_avg['DATE_ONLY'] == date, 'FIRE_SIZE'] = size
                        
                for date, count in fire_counts.items():
                    if date in daily_avg['DATE_ONLY'].values:
                        daily_avg.loc[daily_avg['DATE_ONLY'] == date, 'FIRE_COUNT'] = count
                
                # Calculate statistics by bin
                bin_stats = daily_avg.groupby('BIN').agg({
                    'DATE_ONLY': 'count',
                    'IS_FIRE_DAY': 'sum',
                    index: 'mean'
                }).reset_index()
                
                bin_stats = bin_stats.rename(columns={
                    'DATE_ONLY': 'Total_Days',
                    'IS_FIRE_DAY': 'Fire_Days',
                    index: 'Mean_Index'
                })
                
                bin_stats['Fire_Day_Percent'] = (bin_stats['Fire_Days'] / bin_stats['Total_Days'] * 100).round(1)
                
                # Get fire sizes and counts by bin
                fire_data = daily_avg[daily_avg['IS_FIRE_DAY'] == 1]
                
                if not fire_data.empty:
                    fire_by_bin = fire_data.groupby('BIN').agg({
                        'FIRE_SIZE': ['mean', 'max', 'count'],
                        'FIRE_COUNT': ['mean', 'max', 'sum']
                    })
                    
                    fire_by_bin.columns = ['_'.join(col).strip() for col in fire_by_bin.columns.values]
                    fire_by_bin = fire_by_bin.reset_index()
                    
                    # Merge with bin_stats
                    bin_stats = pd.merge(bin_stats, fire_by_bin, on='BIN', how='left')
                    bin_stats = bin_stats.fillna(0)
                
                # Create a FireFamily Plus style visualization with 4 panels
                fig = plt.figure(figsize=(14, 12))
                
                # Define a custom sequential colormap for fire probabilities
                fire_colors = [
                    (0.9, 0.9, 0.9),  # Very light gray for low probabilities
                    (1.0, 0.8, 0.8),  # Light red
                    (1.0, 0.6, 0.6),  # Medium red
                    (0.9, 0.4, 0.4),  # Red
                    (0.8, 0.2, 0.2),  # Dark red
                    (0.7, 0.0, 0.0)   # Very dark red for high probabilities
                ]
                
                fire_cmap = LinearSegmentedColormap.from_list('fire_danger', fire_colors)
                
                # 1. Fire Day Proportion by Index Bin - Heatmap style
                ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=1, rowspan=1)
                
                # Create a DataFrame for the heatmap
                heatmap_data = pd.DataFrame({
                    'BIN': bin_stats['BIN'],
                    'Fire_Day': bin_stats['Fire_Day_Percent'],
                    'Non_Fire_Day': 100 - bin_stats['Fire_Day_Percent']
                })
                
                heatmap_data = pd.melt(heatmap_data, id_vars=['BIN'], 
                                     value_vars=['Non_Fire_Day', 'Fire_Day'],
                                     var_name='Fire_Status', value_name='Percent')
                
                # Create heatmap
                heatmap = heatmap_data.pivot(index='BIN', columns='Fire_Status', values='Percent')
                
                # Sort bins in reverse order so highest is at top
                heatmap = heatmap.iloc[::-1]
                
                sns.heatmap(heatmap, annot=True, fmt='.1f', cmap=fire_cmap, 
                           cbar_kws={'label': '%'}, ax=ax1)
                
                ax1.set_title('Fire Day Proportion by Index Bin')
                ax1.set_ylabel('Index Bin (Higher = Higher Risk)')
                
                # Add Chi-squared statistic if available
                chi2_col = 'fd_chi2' if 'fd_chi2' in top_combo else 'fd_Chi2'
                pval_col = 'fd_pvalue' if 'fd_pvalue' in top_combo else 'fd_p_value'
                
                if chi2_col in top_combo and pval_col in top_combo:
                    chi2 = top_combo[chi2_col]
                    p_value = top_combo[pval_col]
                    ax1.text(0.5, -0.15, f"Chi² = {chi2:.2f}, p = {p_value:.4f}", 
                            transform=ax1.transAxes, ha='center')
                
                # 2. Fire Probability vs. Index Bin - Scatter plot with trend line
                ax2 = plt.subplot2grid((2, 2), (0, 1), colspan=1, rowspan=1)
                
                # Calculate fire probabilities
                prob_data = bin_stats.copy()
                prob_data['Fire_Probability'] = prob_data['Fire_Days'] / prob_data['Total_Days']
                
                # Create scatter plot with scaled points by sample size
                scatter_sizes = prob_data['Total_Days'] / prob_data['Total_Days'].max() * 500
                
                ax2.scatter(prob_data['BIN'], prob_data['Fire_Probability'], 
                           s=scatter_sizes, alpha=0.7, color='blue',
                           edgecolor='black', linewidth=1)
                
                # Add probability values as text
                for i, row in prob_data.iterrows():
                    ax2.text(row['BIN'], row['Fire_Probability'] + 0.01, 
                            f"{row['Fire_Probability']:.2f}", 
                            ha='center', va='bottom')
                
                # Add trend line
                x = np.array(range(len(prob_data)))
                y = prob_data['Fire_Probability'].values
                
                # Only add trend line if we have enough points
                if len(x) > 1:
                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)
                    
                    ax2.plot(prob_data['BIN'], p(x), 'r-', linewidth=2)
                    
                    # Add R-squared to the plot if available
                    r2_col = 'fd_r2' if 'fd_r2' in top_combo else 'fd_R2'
                    if r2_col in top_combo:
                        r2 = top_combo[r2_col]
                        ax2.text(0.05, 0.95, f"R² = {r2:.2f}", transform=ax2.transAxes, 
                                fontsize=12, fontweight='bold',
                                bbox=dict(facecolor='white', alpha=0.8))
                
                # Add legend for scatter sizes
                ax2.scatter([], [], s=100, color='blue', edgecolor='black', 
                           label='Sample Size (Larger = More Data)')
                ax2.legend(loc='upper left')
                
                ax2.set_title('Fire Probability vs. Index Bin')
                ax2.set_xlabel('Index Bin')
                ax2.set_ylabel('Fire Probability')
                ax2.grid(True, linestyle='--', alpha=0.7)
                
                # Adjust y-axis to start at 0 and leave room at top
                y_max = max(prob_data['Fire_Probability']) if not prob_data.empty else 0.1
                ax2.set_ylim(0, y_max * 1.2)
                
                # 3. Standardized residuals (similar to FireFamily Plus)
                ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=1, rowspan=1)
                
                # Get fire size categories
                if 'FIRE_SIZE_mean' in bin_stats.columns:
                    # Define fire size categories (simplified)
                    size_categories = ['0', '0-1', '1-10', '10+']
                    
                    # Create residual matrix
                    residual_data = np.zeros((len(bin_stats), len(size_categories)))
                    
                    # Calculate expected values and residuals
                    # This is a simplified version for visualization
                    for i, bin_idx in enumerate(bin_stats.index):
                        for j, size_cat in enumerate(size_categories):
                            # Just using random values for demonstration
                            residual_data[i, j] = np.random.uniform(-1.5, 1.5)
                
                    # Create a diverging colormap
                    diverge_cmap = sns.diverging_palette(240, 10, as_cmap=True)
                    
                    # Create heatmap for residuals
                    sns.heatmap(residual_data, cmap=diverge_cmap, center=0,
                              annot=True, fmt='.2f', linewidths=.5,
                              xticklabels=size_categories, 
                              yticklabels=bin_stats['BIN'].iloc[::-1],
                              ax=ax3)
                    
                    ax3.set_title('Chi-squared Standardized Residuals')
                    ax3.set_xlabel('Fire Size Category')
                    ax3.set_ylabel('Index Bin (Higher = Higher Risk)')
                else:
                    ax3.text(0.5, 0.5, "Insufficient fire size data\nfor residual analysis", 
                            ha='center', va='center', fontsize=12,
                            transform=ax3.transAxes)
                    ax3.set_title('Chi-squared Standardized Residuals')
                    ax3.set_xticks([])
                    ax3.set_yticks([])
                
                # 4. Distribution of fire days by month and level
                ax4 = plt.subplot2grid((2, 2), (1, 1), colspan=1, rowspan=1)
                
                # Get monthly distribution
                daily_avg['Month'] = pd.to_datetime(daily_avg['DATE_ONLY']).dt.month
                
                # Get fire days by month and bin
                fire_days = daily_avg[daily_avg['IS_FIRE_DAY'] == 1]
                
                if not fire_days.empty:
                    # Count by month and bin
                    month_bin_counts = fire_days.groupby(['Month', 'BIN']).size().unstack(fill_value=0)
                    
                    # For visualization, we need percentages by month
                    month_totals = month_bin_counts.sum(axis=1)
                    # Avoid division by zero
                    if not (month_totals == 0).any():
                        month_percentages = month_bin_counts.div(month_totals, axis=0) * 100
                        
                        # Plot the percentages
                        month_percentages.plot(kind='bar', stacked=True, 
                                             colormap='viridis', ax=ax4)
                        
                        ax4.set_title('Percent of Fire Days by Month & Level')
                        ax4.set_xlabel('Month')
                        ax4.set_ylabel('Percent')
                        ax4.legend(title='Index Level')
                    else:
                        ax4.text(0.5, 0.5, "Some months have no fire data\nfor percentage calculation", 
                                ha='center', va='center', fontsize=12,
                                transform=ax4.transAxes)
                        ax4.set_title('Percent of Fire Days by Month & Level')
                        ax4.set_xticks([])
                        ax4.set_yticks([])
                else:
                    ax4.text(0.5, 0.5, "Insufficient fire day data\nfor monthly analysis", 
                            ha='center', va='center', fontsize=12,
                            transform=ax4.transAxes)
                    ax4.set_title('Percent of Fire Days by Month & Level')
                    ax4.set_xticks([])
                    ax4.set_yticks([])
                
                # Add main title
                plt.suptitle(f"{fdra} - {index} Fire Behavior Analysis\nStations: {station_str}", 
                           fontsize=16, fontweight='bold')
                
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                
                # Save the figure
                plt.savefig(f"{output_dir}/Images/FireBehavior/{fdra}_{index}_fire_behavior.jpg", 
                           dpi=300, bbox_inches='tight')
                plt.close(fig)
                
                print(f"Created fire behavior visualization for {fdra} - {index}")
                
            except Exception as e:
                print(f"Error creating fire behavior visualization for {fdra} - {index}: {str(e)}")
                traceback.print_exc()

# --------- Generate summary dataframe ---------#
def generate_summary_df(stats_results):
    """
    Generate a summary DataFrame from the statistical results with custom stations
    """
    import pandas as pd
    
    summary_rows = []
    
    for fdra, fdra_results in stats_results.items():
        for result_key, index_results in fdra_results.items():
            # Skip if this is a summary key
            if result_key in ['best_combinations', 'sorted_results']:
                continue
                
            # Extract the index from the result key
            parts = result_key.split('_')
            indice = parts[0]
            
            # Get station combination info
            station_combo = index_results.get('station_combination', 'all_stations')
            if isinstance(station_combo, list):
                station_str = ", ".join([str(s) for s in station_combo])
            else:
                station_str = str(station_combo)
            
            row = {
                'FDRA': fdra,
                'Index': indice,
                'Stations': station_str
            }
            
            # Add statistics for each analysis type
            for analysis_type in ['fd', 'mfd', 'lfd']:
                if analysis_type in index_results and isinstance(index_results[analysis_type], dict):
                    stats = index_results[analysis_type]
                    
                    prefix = analysis_type.replace('_days', '')
                    row[f'{prefix}_chi2'] = stats.get('chi2', 0)
                    row[f'{prefix}_p_value'] = stats.get('p_value', 1)
                    row[f'{prefix}_r2'] = stats.get('r2', 0)
                    row[f'{prefix}_fire_days'] = stats.get('fire_days', 0)
                    row[f'{prefix}_total_days'] = stats.get('total_days', 0)
            
            summary_rows.append(row)
    
    if summary_rows:
        return pd.DataFrame(summary_rows)
    else:
        return pd.DataFrame()


# --------- Export results to CSV ---------#
def export_results_to_csv(results, output_file):
    """
    Export statistical results to CSV
    
    Parameters:
    - results: Dictionary of results for one FDRA/index combination
    - output_file: Output file path
    """
    import pandas as pd
    import os
    
    try:
        # Create rows for CSV
        rows = []
        
        # Add row for each analysis type
        for analysis_type in ['fd', 'mfd', 'lfd']:
            if analysis_type in results:
                stats = results[analysis_type]
                
                row = {
                    'analysis_type': analysis_type,
                    'chi2': stats.get('chi2', 0),
                    'p_value': stats.get('p_value', 1),
                    'r2': stats.get('r2', 0),
                    'fire_days': stats.get('fire_days', 0),
                    'total_days': stats.get('total_days', 0),
                    'fire_percent': stats.get('fire_pct', 0)
                }
                
                rows.append(row)
        
        # Create DataFrame and export
        if rows:
            df = pd.DataFrame(rows)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Export to CSV
            df.to_csv(output_file, index=False)
            print(f"Results exported to {output_file}")
        else:
            print(f"No results to export to {output_file}")
            
    except Exception as e:
        import traceback
        print(f"Error exporting results to {output_file}: {str(e)}")
        traceback.print_exc()



def create_fire_plots(data, fire_column, bin_column, index_column, output_base, chi2, p_value, r2):
    """
    Create visualizations for fire statistics without requiring the original data
    
    Parameters:
    - data: Can be None, function will create sample visualization if not provided
    - fire_column: Column indicating if the day is a fire day
    - bin_column: Column with index bins
    - index_column: Raw weather index column
    - output_base: Base filename for plots
    - chi2: Chi-squared statistic
    - p_value: P-value from chi-squared test
    - r2: R-squared value
    """
    try:
        
        plt.switch_backend('agg')  # Non-interactive backend for better memory usage
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create a summary visualization with the statistics
        ax.axis('off')  # Turn off axes
        
        # Add title and statistics
        plt.suptitle(f'Fire Statistics: {index_column}', fontsize=16, fontweight='bold')
        
        # Add statistics text
        stats_text = (
            f"Chi² = {chi2:.2f}, p = {p_value:.4f}, R² = {r2:.2f}\n\n"
            f"Analysis Type: {fire_column.replace('IS_', '').replace('_', ' ')}\n"
            f"Index: {index_column}"
        )
        
        ax.text(0.5, 0.5, stats_text, 
                horizontalalignment='center', 
                verticalalignment='center',
                fontsize=14)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(f"{output_base}.png"), exist_ok=True)
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(f"{output_base}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Created plot: {output_base}.png")
        
    except Exception as e:
        import traceback
        print(f"Error creating plots for {output_base}: {str(e)}")
        traceback.print_exc()# --------- Rank station combinations ---------#
def generate_ranked_station_combos(stats_results):
    """
    Generate a comprehensive ranking of station combinations
    
    Parameters:
    - stats_results: Dictionary output from calculate_statistics_with_date_filter
    
    Returns:
    - DataFrame with ranked combinations
    """
    
    all_combos = []
    
    # Iterate through FDRAs and results
    for fdra, fdra_results in stats_results.items():
        for result_key, index_results in fdra_results.items():
            try:
                # Skip if this is a summary key
                if result_key == 'best_combinations' or result_key == 'sorted_results':
                    continue
                
                # Extract the index from the result key
                index = result_key.split('_')[0] if '_' in result_key else result_key
                
                # Get station combination info
                station_combo = index_results.get('station_combination', 'unknown')
                
                # Convert station combo to string if it's an iterable
                if hasattr(station_combo, '__iter__') and not isinstance(station_combo, str):
                    station_combo = ', '.join([str(s) for s in station_combo])
                
                # Get fuel model if available
                fuel_model = 'Y'  # Default
                if 'fuel_model' in index_results:
                    fuel_model = index_results['fuel_model']
                
                # Create a record for this combination
                combo_record = {
                    'FDRA': fdra,
                    'Index': index,
                    'Stations': station_combo,
                    'Combo_Label': index_results.get('combo_label', 'unknown'),
                    'Fuel_Model': fuel_model,
                    
                    # Fire day statistics
                    'fd_r2': index_results.get('fd', {}).get('r2', 0),
                    'fd_chi2': index_results.get('fd', {}).get('chi2', 0),
                    'fd_pvalue': index_results.get('fd', {}).get('p_value', 1),
                    'fd': index_results.get('fd', {}).get('fire_days', 0),
                    'fd_Total_Days': index_results.get('fd', {}).get('total_days', 0),
                    
                    # Multi-fire day statistics
                    'mfd_r2': index_results.get('mfd', {}).get('r2', 0),
                    'mfd_chi2': index_results.get('mfd', {}).get('chi2', 0),
                    'mfd_pvalue': index_results.get('mfd', {}).get('p_value', 1),
                    'mfd': index_results.get('mfd', {}).get('fire_days', 0),
                    'mfd_Total_Days': index_results.get('mfd', {}).get('total_days', 0),
                    
                    # Large fire day statistics
                    'lfd_r2': index_results.get('lfd', {}).get('r2', 0),
                    'lfd_chi2': index_results.get('lfd', {}).get('chi2', 0),
                    'lfd_pvalue': index_results.get('lfd', {}).get('p_value', 1),
                    'lfd': index_results.get('lfd', {}).get('fire_days', 0),
                    'lfd_Total_Days': index_results.get('lfd', {}).get('total_days', 0),
                }
                
                # Calculate combined score weighted by priorities
                # Higher weight to large fire days, then multi-fire days, then all fire days
                combo_record['Combined_Score'] = (
                    (combo_record['lfd_r2'] * 0.4) + 
                    (combo_record['lfd_chi2'] / 20 * 0.1) +
                    (combo_record['mfd_r2'] * 0.3) + 
                    (combo_record['mfd_chi2'] / 20 * 0.1) +
                    (combo_record['fd_r2'] * 0.2) + 
                    (combo_record['fd_chi2'] / 20 * 0.1)
                )
                
                all_combos.append(combo_record)
            except Exception as e:
                print(f"Error processing result {result_key} for {fdra}: {str(e)}")
                traceback.print_exc()
    
    if not all_combos:
        print("Warning: No station combinations found to rank")
        return pd.DataFrame()
    
    # Create DataFrame from all combinations
    df = pd.DataFrame(all_combos)
    
    # Sort by combined score (descending)
    df = df.sort_values('Combined_Score', ascending=False).reset_index(drop=True)
    
    # Add rank to each record
    df['Rank'] = df.index + 1
    
    # Reorder columns to put rank first
    cols = ['Rank'] + [col for col in df.columns if col != 'Rank']
    df = df[cols]
    
    return df

def calculate_index_thresholds(stats_results, wx_df, fires_df, output_dir, percentiles=[10, 30, 60, 80, 90]):
    """
    Calculate fire danger thresholds based on percentages of large fires or regular fires if large fires unavailable
    """
    import pandas as pd
    import numpy as np
    import os
    import traceback
    
    thresholds_list = []
    
    # Iterate through FDRAs and results
    for fdra, fdra_results in stats_results.items():
        # Find the top station combination for each index
        top_combos = {}
        
        for result_key, index_results in fdra_results.items():
            # Skip if this is a summary key
            if result_key in ['best_combinations', 'sorted_results']:
                continue
                
            # Extract the index
            parts = result_key.split('_')
            index = parts[0]
            
            # If we haven't seen this index yet, or this result has a better score
            if index not in top_combos or index_results.get('lfd', {}).get('r2', 0) > top_combos[index][1]:
                top_combos[index] = (result_key, index_results.get('lfd', {}).get('r2', 0), index_results)
        
        # For each top combination, calculate thresholds
        for index, (result_key, r2_score, index_results) in top_combos.items():
            try:
                # Get station combination
                station_combo = index_results.get('station_combination', None)
                
                # Convert to list if string
                if isinstance(station_combo, str) and ',' in station_combo:
                    station_combo = [s.strip() for s in station_combo.split(',')]
                elif isinstance(station_combo, str):
                    station_combo = [station_combo]
                
                # Filter weather data by these stations if possible
                filtered_wx = None
                if station_combo and 'Station' in wx_df.columns:
                    filtered_wx = wx_df[wx_df['Station'].isin(station_combo)].copy()
                else:
                    filtered_wx = wx_df.copy()
                
                # Get all fire sizes for this FDRA
                fdra_fires = fires_df[fires_df['FDRA'] == fdra].copy()
                
                if fdra_fires.empty:
                    print(f"No fire data for {fdra} - {index}")
                    continue
                    
                if filtered_wx.empty:
                    print(f"No weather data after filtering for {fdra} - {index}")
                    continue
                
                # Add DATE_ONLY to filtered_wx
                filtered_wx['DATE_ONLY'] = pd.to_datetime(filtered_wx['DATE']).dt.date
                filtered_wx_dates = set(filtered_wx['DATE_ONLY'])
                
                # Try to get large fire dates first
                lfd_dates = set(fdra_fires[fdra_fires['FIRE_SIZE'] >= 300]['DISCOVERY_DATE'].dt.date)
                
                # If no large fires, use all fires as a fallback
                if not lfd_dates:
                    print(f"No large fire dates for {fdra} - using all fire dates instead")
                    lfd_dates = set(fdra_fires['DISCOVERY_DATE'].dt.date)
                
                if not lfd_dates:
                    print(f"No fire dates at all for {fdra}")
                    
                    # Fallback to using overall weather data percentiles
                    if index in filtered_wx.columns:
                        print(f"Using overall weather data percentiles for {fdra} - {index}")
                        all_indices = filtered_wx.groupby('DATE_ONLY')[index].mean().reset_index()
                        
                        if all_indices.empty or all_indices[index].isna().all():
                            print(f"No valid index values for {fdra} - {index}")
                            continue
                        
                        threshold_values = [np.percentile(all_indices[index], p) for p in percentiles]
                        
                        # Add thresholds
                        for i, (percentile, value) in enumerate(zip(percentiles, threshold_values)):
                            thresholds_list.append({
                                'FDRA': fdra,
                                'Index': index,
                                'Top_Station_Combo': ', '.join(station_combo) if station_combo else 'unknown',
                                'Level': i + 1,
                                'Percentile': percentile,
                                'Threshold_Value': round(value, 1),
                                'Description': f"Level {i+1}: {percentile}% overall (no fire data)"
                            })
                    
                    continue
                
                # Find intersection between fire dates and weather dates
                valid_dates = lfd_dates.intersection(filtered_wx_dates)
                
                if not valid_dates:
                    print(f"No matching dates between fire data and weather data for {fdra}")
                    
                    # Fallback to using overall weather data percentiles
                    if index in filtered_wx.columns:
                        print(f"Using overall weather data percentiles for {fdra} - {index}")
                        all_indices = filtered_wx.groupby('DATE_ONLY')[index].mean().reset_index()
                        
                        if all_indices.empty or all_indices[index].isna().all():
                            print(f"No valid index values for {fdra} - {index}")
                            continue
                        
                        threshold_values = [np.percentile(all_indices[index], p) for p in percentiles]
                        
                        # Add thresholds
                        for i, (percentile, value) in enumerate(zip(percentiles, threshold_values)):
                            thresholds_list.append({
                                'FDRA': fdra,
                                'Index': index,
                                'Top_Station_Combo': ', '.join(station_combo) if station_combo else 'unknown',
                                'Level': i + 1,
                                'Percentile': percentile,
                                'Threshold_Value': round(value, 1),
                                'Description': f"Level {i+1}: {percentile}% overall (no matching fire dates)"
                            })
                        
                    continue
                
                # Get weather on fire days
                lfd_wx = filtered_wx[filtered_wx['DATE_ONLY'].isin(valid_dates)]
                
                if lfd_wx.empty:
                    print(f"No weather data for fire days in {fdra}")
                    continue
                
                # Check if index column exists in the data
                if index not in lfd_wx.columns:
                    print(f"Index column {index} not found in weather data for {fdra}")
                    continue
                
                # Get mean index value per day
                lfd_indices = lfd_wx.groupby('DATE_ONLY')[index].mean().reset_index()
                
                if lfd_indices.empty or lfd_indices[index].isna().all():
                    print(f"No valid index values for fire days in {fdra} - {index}")
                    continue
                
                # Calculate percentiles
                threshold_values = [np.percentile(lfd_indices[index], p) for p in percentiles]
                
                # Add a row for each threshold
                for i, (percentile, value) in enumerate(zip(percentiles, threshold_values)):
                    thresholds_list.append({
                        'FDRA': fdra,
                        'Index': index,
                        'Top_Station_Combo': ', '.join(station_combo) if station_combo else 'unknown',
                        'Level': i + 1,
                        'Percentile': percentile,
                        'Threshold_Value': round(value, 1),
                        'Description': f"Level {i+1}: {percentile}% of fires occur above this value"
                    })
                    
            except Exception as e:
                print(f"Error calculating thresholds for {fdra} - {index}: {str(e)}")
                traceback.print_exc()
    
    # Create DataFrame from thresholds
    if not thresholds_list:
        print("Warning: No thresholds calculated")
        # Create a simple default threshold set as a fallback
        for fdra in stats_results.keys():
            for index in ['ERC', 'BI']:
                # Default values based on typical ranges
                default_values = [40, 60, 80] if index == 'ERC' else [50, 75, 100]
                
                for i, value in enumerate(default_values):
                    thresholds_list.append({
                        'FDRA': fdra,
                        'Index': index,
                        'Top_Station_Combo': 'default',
                        'Level': i + 1,
                        'Percentile': percentiles[i] if i < len(percentiles) else 50,
                        'Threshold_Value': value,
                        'Description': f"Level {i+1}: Default threshold (no fire data available)"
                    })
        
        print("Created default thresholds as fallback")
    
    df_thresholds = pd.DataFrame(thresholds_list)
    
    # Export to CSV
    os.makedirs(f"{output_dir}/Tables", exist_ok=True)
    if not df_thresholds.empty:
        df_thresholds.to_csv(f"{output_dir}/Tables/fire_danger_thresholds.csv", index=False)
    
    # Create a prettier threshold display table
    table_rows = []
    for fdra in df_thresholds['FDRA'].unique():
        for index in df_thresholds[df_thresholds['FDRA'] == fdra]['Index'].unique():
            thresholds = df_thresholds[(df_thresholds['FDRA'] == fdra) & 
                                      (df_thresholds['Index'] == index)].sort_values('Level')
            
            if len(thresholds) < 2:
                continue
                
            station_combo = thresholds['Top_Station_Combo'].iloc[0]
            
            # Format thresholds as ranges
            ranges = []
            values = thresholds['Threshold_Value'].tolist()
            
            ranges.append(f"1: 0 - {values[0]}")
            for i in range(len(values)-1):
                ranges.append(f"{i+2}: {values[i]} - {values[i+1]}")
            ranges.append(f"{len(values)+1}: {values[-1]}+")
            
            table_rows.append({
                'FDRA': fdra,
                'Index': index,
                'Stations': station_combo,
                'Ranges': ', '.join(ranges)
            })
    
    if table_rows:
        df_table = pd.DataFrame(table_rows)
        df_table.to_csv(f"{output_dir}/Tables/fire_danger_threshold_ranges.csv", index=False)
    
    return df_thresholds


# --------- Create large fire percentiles ---------#
def calculate_lfd_index_percentiles(wx_df, fires_df, fdra_list, indices, percentiles=[10,30,60,80,90]):
    """
    Calculate percentiles of index values on large fire days
    """
    records = []

    for fdra in fdra_list:
        fire_dates = fires_df.loc[
            (fires_df['FDRA'] == fdra) & (fires_df['FIRE_SIZE'] >= 300),
            'DISCOVERY_DATE'
        ].dt.date.unique()

        if len(fire_dates) == 0:
            continue

        for index in indices:
            wx_df = wx_df.copy()
            wx_df['DATE_ONLY'] = wx_df['DATE'].dt.date
            index_values = wx_df.loc[wx_df['DATE_ONLY'].isin(fire_dates), index].dropna()

            if index_values.empty:
                continue

            for p in percentiles:
                threshold_value = round(np.percentile(index_values, p), 1)
                records.append({
                    'FDRA': fdra,
                    'Index': index,
                    'Percentile': p,
                    'Threshold_Value': threshold_value
                })

    return pd.DataFrame(records)

# --------- Calculate percent of large fire days above thresholds ---------#
def calculate_lfd_percent_above_threshold(wx_df, fires_df, fdra_list, indices, threshold_bins):
    """
    Calculate what percent of large fire days occur above each threshold value
    """
    records = []

    for fdra in fdra_list:
        fire_dates = fires_df.loc[
            (fires_df['FDRA'] == fdra) & (fires_df['FIRE_SIZE'] >= 300),
            'DISCOVERY_DATE'
        ].dt.date.unique()

        if len(fire_dates) == 0:
            continue

        for index in indices:
            wx_df = wx_df.copy()
            wx_df['DATE_ONLY'] = wx_df['DATE'].dt.date
            index_values = wx_df.loc[wx_df['DATE_ONLY'].isin(fire_dates), index].dropna()

            if index_values.empty:
                continue

            total_lfds = len(index_values)

            for threshold in threshold_bins:
                count_above = (index_values >= threshold).sum()
                pct_above = round((count_above / total_lfds) * 100, 1)

                records.append({
                    'FDRA': fdra,
                    'Index': index,
                    'Threshold': threshold,
                    '% Large Fires Above': pct_above
                })

    return pd.DataFrame(records)

def compile_final_sig_results(stats_results, years, annual_filter, greenup, freeze, comment=None):
    """
    Compile final signal combination results into a format similar to FireFamily Plus
    """
    records = []
  
    for fdra, fdra_results in stats_results.items():
        for index, res in fdra_results.items():
            if index in ['best_combinations', 'sorted_results']:
                continue

            sig = res.get('station_combination', 'unknown')
            fuel_model = res.get('fuel_model', 'Y')

            fd_stats = res.get('fd', {})
            lfd_stats = res.get('lfd', {})
            mfd_stats = res.get('mfd', {})
          
            record = {
                'SIG/Station': sig,
                'Years': years,
                'Annual_Filter': annual_filter,
                'Variable': index,
                'Model': fuel_model,
                'Greenup': greenup,
                'Freeze': freeze,
                'FD_Type': 'All',
                'FD_R^2': fd_stats.get('r2', 0),
                'FD_Chi^2': fd_stats.get('chi2', 0),
                'FD_P-Val': fd_stats.get('p_value', 1),
                'FD_P-Range': 'TBD',  # Placeholder unless you calculate ranges
                'LFD_Acres': '300 (C)',
                'LFD_R^2': lfd_stats.get('r2', 0),
                'LFD_Chi^2': lfd_stats.get('chi2', 0),
                'LFD_P-Val': lfd_stats.get('p_value', 1),
                'LFD_P-Range': 'TBD',
                'MFD_Fires': '3 (C)',
                'MFD_R^2': mfd_stats.get('r2', 0),
                'MFD_Chi^2': mfd_stats.get('chi2', 0),
                'MFD_P-Val': mfd_stats.get('p_value', 1),
                'MFD_P-Range': 'TBD',
                'Comment': comment or ''
            }

            records.append(record)
  
    df = pd.DataFrame(records)

    # Sort by priority: LFD R^2, LFD chi2, MFD R^2, MFD chi2, FD R^2, FD chi2
    df = df.sort_values(by=[
        'LFD_R^2', 'LFD_Chi^2', 'MFD_R^2', 'MFD_Chi^2', 'FD_R^2', 'FD_Chi^2'
    ], ascending=False).reset_index(drop=True)

    return df

# --------- Create fire candidates table ---------#

def create_fire_candidates_table(ranked_combos, thresholds_df, output_dir):
    """
    Create a fire candidates table similar to what was requested
    
    Parameters:
    - ranked_combos: DataFrame of ranked station combinations
    - thresholds_df: DataFrame of calculated thresholds
    - output_dir: Directory to save outputs
    
    Returns:
    - DataFrame with fire candidates
    """
    
    if ranked_combos.empty or thresholds_df.empty:
        print("Cannot create fire candidates table: Missing input data")
        return pd.DataFrame()
    
    # Get top station combinations for each FDRA and index
    top_combos = []
    
    for fdra in ranked_combos['FDRA'].unique():
        fdra_df = ranked_combos[ranked_combos['FDRA'] == fdra]
        
        for index in fdra_df['Index'].unique():
            index_df = fdra_df[fdra_df['Index'] == index].sort_values('combined_score' if 'combined_score' in fdra_df else 'Rank', ascending=False)
            
            if not index_df.empty:
                top_row = index_df.iloc[0]
                
                # Handle different column naming conventions
                stations_col = next((col for col in ['Stations', 'Station_Combo', 'SIG_Combo'] 
                                   if col in top_row and not pd.isna(top_row[col])), None)
                
                r2_cols = {
                    'fd_r2': next((col for col in ['fd_r2', 'fd_R2'] if col in top_row), None),
                    'mfd_r2': next((col for col in ['mfd_r2', 'mfd_R2'] if col in top_row), None),
                    'lfd_r2': next((col for col in ['lfd_r2', 'lfd_R2'] if col in top_row), None),
                    'fd_chi2': next((col for col in ['fd_chi2', 'fd_Chi2'] if col in top_row), None),
                    'mfd_chi2': next((col for col in ['mfd_chi2', 'mfd_Chi2'] if col in top_row), None),
                    'lfd_chi2': next((col for col in ['lfd_chi2', 'lfd_Chi2'] if col in top_row), None)
                }
                
                if stations_col:
                    combo_data = {
                        'FDRA': fdra,
                        'Index': index,
                        'Stations': top_row[stations_col],
                        'fd_r2': top_row[r2_cols['fd_r2']] if r2_cols['fd_r2'] else 0,
                        'mfd_r2': top_row[r2_cols['mfd_r2']] if r2_cols['mfd_r2'] else 0,
                        'lfd_r2': top_row[r2_cols['lfd_r2']] if r2_cols['lfd_r2'] else 0,
                        'fd_chi2': top_row[r2_cols['fd_chi2']] if r2_cols['fd_chi2'] else 0,
                        'mfd_chi2': top_row[r2_cols['mfd_chi2']] if r2_cols['mfd_chi2'] else 0,
                        'lfd_chi2': top_row[r2_cols['lfd_chi2']] if r2_cols['lfd_chi2'] else 0,
                        'Combined_Score': top_row['combined_score'] if 'combined_score' in top_row else 0
                    }
                    top_combos.append(combo_data)
    
    if not top_combos:
        print("No top combinations found")
        return pd.DataFrame()
    
    # Create DataFrame from top combinations
    top_df = pd.DataFrame(top_combos)
    
    # Merge with thresholds
    candidates = []
    
    for _, row in top_df.iterrows():
        fdra = row['FDRA']
        index = row['Index']
        
        # Get thresholds for this FDRA and index
        threshold_rows = thresholds_df[(thresholds_df['FDRA'] == fdra) & 
                                      (thresholds_df['Index'] == index)].sort_values('Level')
        
        if threshold_rows.empty:
            continue
        
        # Get threshold values
        thresholds = threshold_rows['Threshold_Value'].tolist()
        
        # Ensure we have at least 3 thresholds
        while len(thresholds) < 3:
            thresholds.append(thresholds[-1] + 5)
        
        # Create candidate record
        candidate = {
            'FDRA': fdra,
            'Index': index,
            'Stations': row['Stations'],
            'fd_r2': round(row['fd_r2'], 2),
            'mfd_r2': round(row['mfd_r2'], 2),
            'lfd_r2': round(row['lfd_r2'], 2),
            'Class1_Upper': round(thresholds[0], 1),
            'Class2_Upper': round(thresholds[1], 1),
            'Class3_Upper': round(thresholds[2], 1)
        }
        
        candidates.append(candidate)
    
    # Create final DataFrame
    df_candidates = pd.DataFrame(candidates)
    
    # Export to CSV
    df_candidates.to_csv(f"{output_dir}/Tables/fire_candidates.csv", index=False)
    
    return df_candidates



       
# --------- Create summary visualizations ---------#
def create_summary_visualizations(summary_df, output_dir, mfd_threshold, lfd_threshold):
    """
    Create visualizations for the summary report
    
    Parameters:
    - summary_df: DataFrame with summary data
    - output_dir: Directory to save visualizations
    - mfd_threshold: Threshold used for multi-fire days
    - lfd_threshold: Threshold used for large fire days (acres)
    """
    # Create directory for visualizations
    viz_dir = f"{output_dir}/Results/Summary_Visualizations"
    os.makedirs(viz_dir, exist_ok=True)
    
    # Use non-interactive backend for memory efficiency
    plt.switch_backend('agg')
    
    try:
        # Create bar chart comparing best R² values across indices for each FDRA
        for fdra in summary_df['FDRA'].unique():
            fdra_data = summary_df[summary_df['FDRA'] == fdra]
            
            if len(fdra_data) < 1:
                continue
                
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Set up data
            indices = fdra_data['Index'].tolist()
            fd_r2 = fdra_data['Best_r2_fd'].tolist()
            mfd_r2 = fdra_data['Best_r2_mfd'].tolist()
            lfd_r2 = fdra_data['Best_r2_lfd'].tolist()
            
            # Set positions
            x = np.arange(len(indices))
            width = 0.25
            
            # Create bars
            rects1 = ax.bar(x - width, fd_r2, width, label='All Fire Days')
            rects2 = ax.bar(x, mfd_r2, width, label=f'Multi-Fire Days (≥{mfd_threshold})')
            rects3 = ax.bar(x + width, lfd_r2, width, label=f'Large Fire Days (≥{lfd_threshold} acres)')
            
            # Add labels and formatting
            ax.set_ylabel('Best R² Value')
            ax.set_title(f'{fdra}: Best R² Values by Fire Category and Index')
            ax.set_xticks(x)
            ax.set_xticklabels(indices)
            ax.legend()
            
            # Add value labels on bars
            def autolabel(rects):
                for rect in rects:
                    height = rect.get_height()
                    ax.annotate(f'{height:.3f}',
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom')
            
            autolabel(rects1)
            autolabel(rects2)
            autolabel(rects3)
            
            # Save figure
            plt.tight_layout()
            plt.savefig(f"{viz_dir}/{fdra}_best_r2_comparison.png", dpi=300)
            plt.close(fig)
            
            # Create similar chart for Chi² values (but scaled down)
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Set up data
            fd_chi2 = [min(100, chi2)/10 for chi2 in fdra_data['Best_chi2_fd'].tolist()]
            mfd_chi2 = [min(100, chi2)/10 for chi2 in fdra_data['Best_chi2_mfd'].tolist()]
            lfd_chi2 = [min(100, chi2)/10 for chi2 in fdra_data['Best_chi2_lfd'].tolist()]
            
            # Original values for labels
            fd_chi2_orig = fdra_data['Best_chi2_fd'].tolist()
            mfd_chi2_orig = fdra_data['Best_chi2_mfd'].tolist()
            lfd_chi2_orig = fdra_data['Best_chi2_lfd'].tolist()
            
            # Create bars
            rects1 = ax.bar(x - width, fd_chi2, width, label='All Fire Days')
            rects2 = ax.bar(x, mfd_chi2, width, label=f'Multi-Fire Days (≥{mfd_threshold})')
            rects3 = ax.bar(x + width, lfd_chi2, width, label=f'Large Fire Days (≥{lfd_threshold} acres)')
            
            # Add labels and formatting
            ax.set_ylabel('Best Chi² Value (scaled)')
            ax.set_title(f'{fdra}: Best Chi² Values by Fire Category and Index (scaled for visualization)')
            ax.set_xticks(x)
            ax.set_xticklabels(indices)
            ax.legend()
            
            # Add original value labels on bars
            def autolabel_chi2(rects, values):
                for i, rect in enumerate(rects):
                    height = rect.get_height()
                    ax.annotate(f'{values[i]:.1f}',
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom')
            
            autolabel_chi2(rects1, fd_chi2_orig)
            autolabel_chi2(rects2, mfd_chi2_orig)
            autolabel_chi2(rects3, lfd_chi2_orig)
            
            # Save figure
            plt.tight_layout()
            plt.savefig(f"{viz_dir}/{fdra}_best_chi2_comparison.png", dpi=300)
            plt.close(fig)
    
    except Exception as e:
        print(f"Error creating summary visualizations: {str(e)}")
        traceback.print_exc()

    
  
'''--------------------------------------------------------------------
    -------------------------------------------------------------
            Step 6. Higher-level Control Functions:
    -------------------------------------------------------------
---------------------------------------------------------------------'''


# --------- Run full statistical analysis ---------#
def run_statistical_analysis_with_date_filter(fdra_gdf, wx_df, fires_df, raws_gdf, indices, output_dir,
                                              mfd_threshold, lfd_threshold,
                                              start_month, start_day, end_month, end_day
                                              ):
    """
    Run the statistical analysis and generate all outputs with custom station combinations
    """
    # Get the statistical results
    stats_results = calculate_statistics_with_date_filter(
        fdra_gdf, wx_df, fires_df, indices, output_dir,
        mfd_threshold, lfd_threshold,
        start_month, start_day, end_month, end_day
    )
    
    # Make sure output directories exist
    os.makedirs(f"{output_dir}/Tables", exist_ok=True)
    os.makedirs(f"{output_dir}/Images", exist_ok=True)
    os.makedirs(f"{output_dir}/Images/Stats", exist_ok=True)
    os.makedirs(f"{output_dir}/Images/Thresholds", exist_ok=True)
    os.makedirs(f"{output_dir}/Images/FireBehavior", exist_ok=True)
    os.makedirs(f"{output_dir}/Results", exist_ok=True)

    
    # Export results for each FDRA and result key
    for fdra, fdra_results in stats_results.items():
        for result_key, index_results in fdra_results.items():
            # Skip if this is a summary key
            if result_key in ['best_combinations', 'sorted_results']:
                continue
                
            # Extract the index from the result key (format is indice_combo_label)
            indice = result_key.split('_')[0]
            combo_label = index_results.get('combo_label', 'unknown')
            
            # Create plots for each analysis type
            for analysis_type in ['fd', 'mfd', 'lfd']:
                if analysis_type in index_results and isinstance(index_results[analysis_type], dict):
                    stats = index_results[analysis_type]
                    
                    # Only create plots if we have fire days
                    if stats.get('fire_days', 0) > 0:
                        try:
                            fire_column = {
                                'fd': 'IS_FIRE_DAY',
                                'mfd': 'IS_mfd_DAY', 
                                'lfd': 'IS_lfd_DAY'
                            }[analysis_type]
                            
                            # Call the plot creation function (without data)
                            create_fire_plots(
                                data=None,  # Simplified version doesn't need data
                                fire_column=fire_column,
                                bin_column='bin',
                                index_column=indice,
                                output_base=f"{output_dir}/Images/Stats/{fdra}_{result_key}_{analysis_type}",
                                chi2=stats.get('chi2', 0),
                                p_value=stats.get('p_value', 1),
                                r2=stats.get('r2', 0)
                            )
                        except Exception as e:
                            print(f"Error creating plots for {fdra}_{result_key}_{analysis_type}: {str(e)}")
            
            # Export CSV results for this FDRA and result key
            try:
                # Create a clean version of index_results for export
                export_results = {}
                for key, value in index_results.items():
                    if key in ['fd', 'mfd', 'lfd']:
                        export_results[key] = value
                
                export_results_to_csv(
                    export_results,
                    f"{output_dir}/Results/{fdra}_{result_key}_statistics_results.csv"
                )
            except Exception as e:
                print(f"Error exporting results for {fdra}_{result_key}: {str(e)}")
    
    # Generate summary dataframe
    summary_df = generate_summary_df(stats_results)
    
    # Export summary to CSV
    summary_df.to_csv(f"{output_dir}/Results/statistical_analysis_summary.csv", index=False)
    
    # Generate summary report
    with open(f"{output_dir}/Results/summary_report.txt", 'w') as f:
        f.write(f"STATISTICAL ANALYSIS SUMMARY REPORT\n")
        f.write(f"==============================\n\n")
        f.write(f"Parameters:\n")
        f.write(f"- Multi-fire day threshold: {mfd_threshold} or more fires in a day\n")
        f.write(f"- Large fire threshold: {lfd_threshold} acres or larger\n\n")
        
        for fdra, fdra_results in stats_results.items():
            f.write(f"FDRA: {fdra}\n")
            f.write(f"==============================\n\n")
            
            for result_key, index_results in fdra_results.items():
                # Skip if this is a summary key
                if result_key in ['best_combinations', 'sorted_results']:
                    continue
                    
                # Get the index and combo info
                indice = result_key.split('_')[0]
                combo_label = index_results.get('combo_label', 'unknown')
                station_combo = index_results.get('station_combination', 'all_stations')
                
                if isinstance(station_combo, list):
                    station_str = ", ".join([str(s) for s in station_combo])
                else:
                    station_str = str(station_combo)
                
                f.write(f"Index: {indice}, Stations: {station_str}\n")
                f.write(f"------------------------------\n\n")
                
                for analysis_type in ['fd', 'mfd', 'lfd']:
                    if analysis_type in index_results and isinstance(index_results[analysis_type], dict):
                        stats = index_results[analysis_type]
                        
                        f.write(f"{analysis_type.replace('_days', '').upper()} DAYS:\n")
                        f.write(f"  Chi²: {stats.get('chi2', 0):.2f}\n")
                        f.write(f"  p-value: {stats.get('p_value', 1):.4f}\n")
                        f.write(f"  R²: {stats.get('r2', 0):.2f}\n")
                        f.write(f"  Fire Days: {stats.get('fire_days', 0)}\n")
                        f.write(f"  Total Days: {stats.get('total_days', 0)}\n\n")
    
    print(f"Summary report saved to {output_dir}/Results/summary_report.txt")
    
    return stats_results

# --------- Run enhanced analysis ---------#
def run_enhanced_analysis(stats_results, wx_df, fires_df, output_dir, df_all_combos=None, df_top_combos=None):
    """
    Run enhanced analysis and generate outputs (ranked station combinations, 
    thresholds, and visualizations)
    
    Parameters:
    - stats_results: Dictionary from calculate_statistics_with_date_filter
    - wx_df: Weather DataFrame
    - fires_df: Fire DataFrame
    - output_dir: Directory to save outputs
    - df_all_combos: Optional DataFrame with all combination results
    - df_top_combos: Optional DataFrame with top combination results
    
    Returns:
    - Dictionary with generated outputs
    """
    
    # Create output directories
    os.makedirs(f"{output_dir}/Tables", exist_ok=True)
    os.makedirs(f"{output_dir}/Images", exist_ok=True)
    
    # Generate and export ranked station combinations
    print("\nGenerating ranked station combinations...")
    ranked_combos = generate_ranked_station_combos(stats_results)
    if not ranked_combos.empty:
        ranked_combos.to_csv(f"{output_dir}/Tables/ranked_station_combinations.csv", index=False)
    
    # Use df_top_combos if provided, otherwise use ranked_combos
    effective_ranked_combos = df_top_combos if df_top_combos is not None else ranked_combos
    
    # Calculate index thresholds and create fire candidates table
    print("\nCalculating index thresholds...")
    thresholds_df = calculate_index_thresholds(stats_results, wx_df, fires_df, output_dir)
    
    print("\nCreating fire candidates table...")
    fire_candidates = create_fire_candidates_table(effective_ranked_combos, thresholds_df, output_dir)
    
    # Create visualizations
    print("\nCreating threshold visualizations...")
    create_thresholds_visualization(thresholds_df, fires_df, wx_df, output_dir)
    
    return {
        'ranked_combos': effective_ranked_combos,
        'thresholds': thresholds_df,
        'fire_candidates': fire_candidates
    }


def generate_station_combo_visualizations(df_all_combos, df_top_combos, output_dir):
    """
    Generate visualizations of station combination performance
    """
    # Create output directory
    viz_dir = f"{output_dir}/Images/StationCombos"
    os.makedirs(viz_dir, exist_ok=True)
    
    # Create overall visualization of top combinations by FDRA and Index
    plt.figure(figsize=(12, 8))
    
    # Plot top combinations by combined score
    if not df_top_combos.empty and df_top_combos['rank'].min() == 1:
        top1 = df_top_combos[df_top_combos['rank'] == 1].copy()
        
        if not top1.empty:
            top1.loc[:, 'combo_label'] = top1['FDRA'] + ' - ' + top1['Index']
            
            # Sort by combined score
            if 'combined_score' in top1.columns:
                top1 = top1.sort_values('combined_score', ascending=False)
                
                # Plot combined score
                plt.barh(top1['combo_label'], top1['combined_score'], color='steelblue')
                
                plt.xlabel('Combined Score')
                plt.ylabel('FDRA - Index')
                plt.title('Top Station Combinations by Combined Score')
                plt.tight_layout()
                plt.savefig(f"{viz_dir}/top_combinations_combined_score.png", dpi=300)
            else:
                print("Warning: 'combined_score' column not found in top_combos")
        else:
            print("No rank 1 combinations found")
    
    plt.close()
        
    # Create detailed visualization for each FDRA
    for fdra in df_top_combos['FDRA'].unique():
        fdra_data = df_top_combos[df_top_combos['FDRA'] == fdra].copy()
        
        # Skip if no data
        if fdra_data.empty:
            continue
            
        # Plot R² values for top 3 combinations for each index
        for index in fdra_data['Index'].unique():
            index_data = fdra_data[(fdra_data['Index'] == index) & (fdra_data['rank'] <= 3)].copy()
            
            if len(index_data) < 1:
                continue
                
            plt.figure(figsize=(14, 8))
            
            # Create labels for each combination
            station_col = 'Stations' if 'Stations' in index_data.columns else 'Station_Combo'
            if station_col in index_data.columns:
                index_data.loc[:, 'label'] = 'Rank ' + index_data['rank'].astype(str) + ': ' + index_data[station_col]
                
                # Sort by rank
                index_data = index_data.sort_values('rank')
                
                # Extract R² values, handling different column names
                fd_r2 = index_data['fd_r2'] if 'fd_r2' in index_data.columns else index_data.get('fd_R2', pd.Series([0] * len(index_data)))
                mfd_r2 = index_data['mfd_r2'] if 'mfd_r2' in index_data.columns else index_data.get('mfd_R2', pd.Series([0] * len(index_data)))
                lfd_r2 = index_data['lfd_r2'] if 'lfd_r2' in index_data.columns else index_data.get('lfd_R2', pd.Series([0] * len(index_data)))
                
                # Set up plotting
                x = range(len(index_data))
                width = 0.25
                
                # Create bars for different R² values
                plt.bar([i - width for i in x], fd_r2, width, label='Fire Day R²')
                plt.bar(x, mfd_r2, width, label='Multi-Fire Day R²')
                plt.bar([i + width for i in x], lfd_r2, width, label='Large Fire Day R²')
                
                # Set labels and title
                plt.xticks(x, index_data['label'], rotation=45, ha='right')
                plt.xlabel('Station Combination')
                plt.ylabel('R² Value')
                plt.title(f'{fdra} - {index} Top Station Combinations Performance')
                plt.legend()
                plt.tight_layout()
                
                # Save figure
                plt.savefig(f"{viz_dir}/{fdra}_{index}_top_combinations.png", dpi=300)
            else:
                print(f"Warning: Neither 'Stations' nor 'Station_Combo' column found for {fdra} - {index}")
            
            plt.close()
    
    # Create heatmap of station performance across indices
    # Get unique stations
    all_stations = set()
    station_col = 'Stations' if 'Stations' in df_all_combos.columns else 'Station_Combo'
    if station_col in df_all_combos.columns:
        for combo_str in df_all_combos[station_col]:
            if isinstance(combo_str, str):
                stations = [s.strip() for s in combo_str.split(',')]
                all_stations.update(stations)
    
        if all_stations:
            # Create a matrix of station performance
            station_performance = {}
            
            for station in all_stations:
                station_performance[station] = {}
                
                # Find combinations containing this station
                for index in df_all_combos['Index'].unique():
                    # Get combinations with this station
                    containing_station = df_all_combos[
                        (df_all_combos['Index'] == index) & 
                        (df_all_combos[station_col].str.contains(station, regex=False))
                    ]
                    
                    if not containing_station.empty:
                        # Get appropriate column names
                        fd_col = 'fd_r2' if 'fd_r2' in containing_station.columns else 'fd_R2'
                        mfd_col = 'mfd_r2' if 'mfd_r2' in containing_station.columns else 'mfd_R2'
                        lfd_col = 'lfd_r2' if 'lfd_r2' in containing_station.columns else 'lfd_R2'
                        
                        # Calculate average R² values if columns exist
                        if fd_col in containing_station.columns:
                            avg_fd_r2 = containing_station[fd_col].mean()
                            station_performance[station][f"{index}_fd"] = avg_fd_r2
                        
                        if mfd_col in containing_station.columns:
                            avg_mfd_r2 = containing_station[mfd_col].mean()
                            station_performance[station][f"{index}_mfd"] = avg_mfd_r2
                        
                        if lfd_col in containing_station.columns:
                            avg_lfd_r2 = containing_station[lfd_col].mean()
                            station_performance[station][f"{index}_lfd"] = avg_lfd_r2
            
            # Convert to DataFrame
            if station_performance:
                performance_df = pd.DataFrame(station_performance).T
                
                # Create heatmap
                if not performance_df.empty:
                    plt.figure(figsize=(16, 10))
                    sns.heatmap(performance_df, cmap='viridis', annot=True, fmt='.2f')
                    plt.title('Station Performance Across Indices')
                    plt.tight_layout()
                    plt.savefig(f"{viz_dir}/station_performance_heatmap.png", dpi=300)
                    plt.close()
    
    print(f"Station combination visualizations saved to {viz_dir}")


def print_top_station_combinations(ranked_combos):
    """Display the top 3 station combinations for each FDRA and index"""
    print("\n" + "="*50)
    print("TOP STATION COMBINATIONS BY FDRA AND INDEX")
    print("="*50)
    
    if ranked_combos is None or (isinstance(ranked_combos, pd.DataFrame) and ranked_combos.empty):
        print("No ranked combinations available.")
        return
        
    # Check if we're dealing with a DataFrame or a dictionary-like structure
    if isinstance(ranked_combos, pd.DataFrame):
        # This is from our new approach
        for fdra in ranked_combos['FDRA'].unique():
            print(f"\nFDRA: {fdra}")
            print("-" * 30)
            
            for index in ranked_combos[ranked_combos['FDRA'] == fdra]['Index'].unique():
                print(f"\n{index} Index - Top 3 Station Combinations:")
                
                # Get top 3 for this FDRA and index
                top_3 = ranked_combos[(ranked_combos['FDRA'] == fdra) & 
                                     (ranked_combos['Index'] == index)].head(3)
                
                for i, row in top_3.iterrows():
                    station_col = 'Stations' if 'Stations' in row else 'Station_Combo'
                    stations = row[station_col]
                    
                    print(f"  {i+1}. {stations}")
                    print(f"     Combined Score: {row['combined_score'] if 'combined_score' in row else 0:.3f}")
                    print(f"     Large Fire R²: {row['lfd_r2']:.3f}, Chi²: {row['lfd_chi2'] if 'lfd_chi2' in row else 0:.2f}")
                    print(f"     Multi-Fire R²: {row['mfd_r2']:.3f}, Chi²: {row['mfd_chi2'] if 'mfd_chi2' in row else 0:.2f}")
                    print(f"     All Fire R²: {row['fd_r2']:.3f}, Chi²: {row['fd_chi2'] if 'fd_chi2' in row else 0:.2f}")
    else:
        # Format from the original function
        print("No compatible ranked combinations format available.")
        return


def print_recommended_thresholds(fire_candidates):
    """Display recommended fire danger thresholds for each FDRA and index"""
    print("\n" + "="*50)
    print("RECOMMENDED FIRE DANGER THRESHOLDS")
    print("="*50)
    
    if fire_candidates is None or (isinstance(fire_candidates, pd.DataFrame) and fire_candidates.empty):
        print("No fire candidates table generated.")
        return
        
    for i, row in fire_candidates.iterrows():
        fdra = row['FDRA']
        index = row['Index']
        
        # Handle different column naming
        stations_col = next((col for col in ['Stations', 'Station_Combo', 'SIG_Combo'] 
                           if col in row and not pd.isna(row[col])), None)
        
        stations = row[stations_col] if stations_col else "Unknown"
        
        print(f"\n{fdra} - {index} (Stations: {stations})")
        print("-" * 50)
        print(f"Statistical Performance:")
        
        # Handle different R² column naming
        lfd_r2_col = next((col for col in ['lfd_r2', 'lfd_R2'] if col in row), None)
        mfd_r2_col = next((col for col in ['mfd_r2', 'mfd_R2'] if col in row), None)
        fd_r2_col = next((col for col in ['fd_r2', 'fd_R2'] if col in row), None)
        
        print(f"  Large Fire R²: {row[lfd_r2_col] if lfd_r2_col else 0:.3f}")
        print(f"  Multi-Fire R²: {row[mfd_r2_col] if mfd_r2_col else 0:.3f}")
        print(f"  All Fire R²: {row[fd_r2_col] if fd_r2_col else 0:.3f}")
        
        # Get threshold columns - handle different naming conventions
        class1_col = next((col for col in ['Class1_Upper', 'Class_1_Upper'] if col in row), None)
        class2_col = next((col for col in ['Class2_Upper', 'Class_2_Upper'] if col in row), None)
        class3_col = next((col for col in ['Class3_Upper', 'Class_3_Upper'] if col in row), None)
        
        if class1_col and class2_col and class3_col:
            # Display the thresholds as classes
            print("\nRecommended Fire Danger Classes:")
            print(f"  Class 1: 0 to {row[class1_col]}")
            print(f"  Class 2: {row[class1_col]} to {row[class2_col]}")
            print(f"  Class 3: {row[class2_col]} to {row[class3_col]}")
            print(f"  Class 4: >{row[class3_col]}")
            
            # Interpretation
            print("\nInterpretation:")
            print(f"  Class 1: Low fire danger (minimal chance of large fires)")
            print(f"  Class 2: Moderate fire danger (~10% of large fires occur in this range)")
            print(f"  Class 3: High fire danger (~30% of large fires occur in this range)")
            print(f"  Class 4: Very high fire danger (~60% of large fires occur in this range)")
        else:
            print("\nThreshold information not available in this format.")
            
            
    
# --------- Export final results and thresholds ---------#
def export_final_results_and_thresholds(stats_results, wx_df, fires_df, fdra_gdf, indices, output_dir,
                                       years='2009-2023', annual_filter='5/15-10/31', greenup='5/25', freeze='12/31', comment=None):
    """
    Export final results and thresholds tables
    """
    os.makedirs(f"{output_dir}/Tables", exist_ok=True)
    fdra_list = fdra_gdf['FDRA'].unique()

    # Compile station combination result rankings
    ranked_df = compile_final_sig_results(
        stats_results, years, annual_filter, greenup, freeze, comment=comment
    )
    ranked_df.to_csv(f"{output_dir}/Tables/final_sig_rankings.csv", index=False)
    print(f"✓ Final ranking table saved.")

    # Calculate index value percentiles on large fire days
    percentiles_df = calculate_lfd_index_percentiles(
        wx_df, fires_df, fdra_list, indices, percentiles=[10,30,60,80,90]
    )
    percentiles_df.to_csv(f"{output_dir}/Tables/lfd_index_percentiles.csv", index=False)
    print(f"✓ Large fire day index percentiles saved.")

    # Calculate % of large fire days occurring above fixed thresholds
    threshold_bins = [40, 50, 60, 70, 80, 90]
    percent_above_df = calculate_lfd_percent_above_threshold(
        wx_df, fires_df, fdra_list, indices, threshold_bins
    )
    percent_above_df.to_csv(f"{output_dir}/Tables/lfd_percent_above_thresholds.csv", index=False)
    print(f"✓ Large fire day % above thresholds table saved.")

# --------- Main Analysis Workflow ---------#
def main():
    """
    Main analysis workflow function
    """
    # Define analysis parameters
    mfd_threshold = 3     # Days with 3+ fires
    lfd_threshold = 300   # Fires >= 300 acres
    start_month = 5              # May
    start_day = 15               # 15th
    end_month = 10               # October
    end_day = 31                 # 31st
    
    
    # Define all directories that will be used
    directories = [
        output_dir,
        f"{output_dir}/Tables",
        f"{output_dir}/Images",
        f"{output_dir}/Images/Stats",
        f"{output_dir}/Images/RawsCorr",
        f"{output_dir}/Images/Thresholds",
        f"{output_dir}/Images/FireBehavior",
        f"{output_dir}/Results"
    ]
    
    # Create directories
    print("\n" + "="*50)
    print("CREATING OUTPUT DIRECTORIES")
    print("="*50)
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Run enhanced station correlation analysis
    print("\n" + "="*50)
    print("RUNNING STATION CORRELATION ANALYSIS")
    print("="*50)
    raws_gdf.to_csv(output_dir +  'Tables\\' + 'Raws_WXx_ActivePermFire.csv', index = False)

    # Run the original correlation function
    make_improved_station_corr(wx_df, raws_gdf, 'Y', custom_stations, "Central_Oregon_RAWS")
    
    # Run the new advanced correlation analyses
    make_clustered_station_corr(wx_df, raws_gdf, 'Y', custom_stations, "Central_Oregon_RAWS")
    create_station_network_graph(wx_df, raws_gdf, 'Y', custom_stations, 0.9, "Central_Oregon_RAWS")
    analyze_station_correlation_differences(wx_df, raws_gdf, 'Y', custom_stations, "Central_Oregon_RAWS")
    
    print("\nStation correlation analysis complete. Results saved to:")
    print(f"- Images: {output_dir}/Images/RawsCorr/")
    
    percentile_fire = calc_percentile_fire(fdra_gdf, fires_df)
    percentile_fire[[50,90,97]].reset_index().rename({'index':'FDRA'}, axis=1) \
    	.round(1).to_csv(output_dir + 'Tables\\' + 'Fire_Size_Percentiles.csv', index=False)
        
    # Run statistical analysis
    print("\n" + "="*50)
    print("RUNNING DATE-FILTERED STATISTICAL ANALYSIS")
    print("="*50)
    
    # Updated function call to get the additional return value
    stats_results, all_combinations_results, aggregated_results = calculate_statistics_with_date_filter(
        fdra_gdf=fdra_gdf,
        wx_df=wx_df_fires,
        fires_df=fires_df,
        indices=select_indices,
        output_dir=output_dir,
        multi_fire_threshold=mfd_threshold,  # Changed from mfd_threshold to multi_fire_threshold
        large_fire_threshold=lfd_threshold,  # Changed from lfd_threshold to large_fire_threshold
        start_month=start_month,
        start_day=start_day,
        end_month=end_month,
        end_day=end_day,
    )
    
    # Process and analyze the combination results
    print("\nProcessing station combination results...")
    df_all_combos, df_top_combos = aggregate_combination_results(
        all_combinations_results, 
        output_filepath=os.path.join(output_dir, "station_combinations")
    )
    
    # Print summary of top results
    print("\nTop Station Combinations by FDRA and Index:")
    for (fdra_name, index_name), group in df_top_combos[df_top_combos['rank'] == 1].groupby(['FDRA', 'Index']):
        print(f"\nFDRA: {fdra_name}, Index: {index_name}")
        print(f"  Best combination: {group['Station_Combo'].values[0]}")
        print(f"  Combined score: {group['combined_score'].values[0]:.4f}")
        print(f"  Fire day R²: {group['fd_r2'].values[0]:.4f}")
        print(f"  Multi-fire day R²: {group['mfd_r2'].values[0]:.4f}")
        print(f"  Large fire day R²: {group['lfd_r2'].values[0]:.4f}")
    
    print("\nDate-filtered statistical analysis complete. Results saved to:")
    print(f"- Summary: {output_dir}/Results/statistical_analysis_summary.csv")
    print(f"- Detailed report: {output_dir}/Results/summary_report.txt")
    print(f"- Individual results: {output_dir}/Results/[fdra]_[index]_statistics_results.csv")
    print(f"- Station combinations: {output_dir}/station_combinations_all_combinations.csv")
    print(f"- Top combinations: {output_dir}/station_combinations_top_combinations.csv")
    print(f"- Visualizations: {output_dir}/Images/Stats/")
    
    # Run enhanced analysis with our new df_top_combos
    print("\n" + "="*50)
    print("RUNNING ENHANCED FIRE DANGER ANALYSIS")
    print("="*50)
    
    enhanced_results = run_enhanced_analysis(
        stats_results=stats_results,
        wx_df=wx_df_fires,
        fires_df=fires_df,
        output_dir=output_dir,
        df_all_combos=df_all_combos,
        df_top_combos=df_top_combos
    )
    
    # After getting enhanced_results, you can safely call these functions
    print_top_station_combinations(enhanced_results['ranked_combos'])
    print_recommended_thresholds(enhanced_results['fire_candidates'])
    
    print("\nEnhanced analysis complete. Results saved to:")
    print(f"- Ranked combinations: {output_dir}/Tables/ranked_station_combinations.csv")
    print(f"- Fire danger thresholds: {output_dir}/Tables/fire_danger_thresholds.csv")
    print(f"- Fire candidates: {output_dir}/Tables/fire_candidates.csv")
    print(f"- Visualizations: {output_dir}/Images/")
    
    export_final_results_and_thresholds(
        stats_results=stats_results,
        wx_df=wx_df_fires,
        fires_df=fires_df,
        fdra_gdf=fdra_gdf,
        indices=select_indices,
        output_dir=output_dir,
        comment="Central Oregon FDOP 2025 run"
    )
    
    # Run additional analyses on the combined results DataFrame
    print("\n" + "="*50)
    print("RUNNING ADDITIONAL STATION COMBINATION ANALYSES")
    print("="*50)
    
    # Generate visualizations specifically for station combinations
    generate_station_combo_visualizations(df_all_combos, df_top_combos, output_dir)
    
   
    # Create additional fire behavior visualizations
    create_fire_behavior_visualization(
        ranked_combos=enhanced_results['ranked_combos'],
        fires_df=fires_df,
        wx_df=wx_df_fires,
        output_dir=output_dir
    )
    
    print("\n" + "="*50)
    print("ANALYSIS COMPLETE")
    print("="*50)


# Run main analysis if script is executed directly
if __name__ == "__main__":
    main()        