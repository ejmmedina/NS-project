# Network Analysis of the Primary Healthcare Facility Referral System in the Philippines
##### By: Elijah Justin Medina and Jomilynn Rebanal

This project aims to analyze to referral system among health units in the Philippines in order to identify the characteristics and consequently, the robustness of this system. The main data used in this analysis are geospatial coordinates of different health units in the network &mdash; barangay health stations (BHS), rural health units (RHU), and hospitals. The network is supplemented with bed capacity information of the different health units. The aforementioned data was obtained from the DOH Data Collect app v2.1 and the National Health Facility Registry.

##### Preliminaries


```python
import pandas as pd
import numpy as np
from glob import glob
import sys
import locale
from geopy.distance import vincenty
import warnings

warnings.filterwarnings("ignore")
np.set_printoptions(threshold=sys.maxsize, precision=2)
pd.set_option('float_format', '{:,.2f}'.format)
pd.set_option('display.max_rows', 1000)
```

## Data Processing and Exploratory Data Analysis
Before performing the analysis, the data must be cleaned to standardize text and minimize effect of errors (e.g. typographical, encoding, outliers). The goal of this processing is to prepare the data for network analysis.

### Merging of multiple, separated files

##### Geographic Coordinates (Health Facility names and coordinates)


```python
files = glob('Geographic Coordinates/*.xlsx')
cols = ['Health Facility Code Short', 'Facility Name', 'Health Facility Type',
        'Region Name                                             ',
        'Province Name ', 'City/Municipality Name', 'Barangay Name',
        'Latitude', 'Longitude', 'Service Capability', 'Licensing Status',
        'Bed Capacity']

HF_list = pd.DataFrame()

for f in files:
    data = pd.read_excel(f, usecols=cols)
    HF_list = HF_list.append(data)
```


```python
HF_list.isnull().sum() # Verify mismatched fields across different excel files
```


```python
HF_list.columns = ['SHORT_HFCODE', 'HF_NAME', 'HF_TYPE', 'REGION', 'PROVINCE',
                  'MUNI_CITY', 'BRGY', 'LAT', 'LONG', 'SERVICE_CAP',
                  'LICENSING', 'BED_CAP']

str_cols = ['HF_NAME', 'HF_TYPE', 'REGION', 'PROVINCE', 'MUNI_CITY', 'BRGY',
            'SERVICE_CAP', 'LICENSING', 'BED_CAP']

HF_list[str_cols] = HF_list[str_cols].fillna('UNKNOWN').apply(lambda x: x.str.upper().str.strip())
HF_list['SHORT_HFCODE'] = HF_list['SHORT_HFCODE'].astype(int)

HF_list.to_excel('cleaned/HFList_cleaned.xlsx') #Store the combined dataframe
```

##### Rural Health Unit


```python
rhu = pd.read_excel('rhu2018.xlsx', sheet_name='MAIN', na_values='None')
```


```python
str_cols = ['HF_NAME', 'REGION', 'PROVINCE', 'MUNI_CITY', 'BRGY',
            'STREET_NAME', 'BUILDING', 'FACILITY_HEAD', 'DETACHED', 'BRGYS',
            'SDN', 'SDN_NAME', 'REF1_NAME', 'REF1_SAMEPROV',
            'REF1_REF1A', 'REF1A_SAMEPROV', 'REF2_NAME', 'REF3_NAME',  
            'AMB_ACCESS', 'AMB_OWN', 'PHIC_ACC', 'PHIC_PACKAGES', 'PHIC_MCP',
            'PHIC_PCB1', 'PHIC_MALARIA', 'PHIC_TBDOTS', 'PHIC_ABTC',
            'PHIC_NCP', 'PHIC_OTH']
code_cols = ['id', 'REF1_CODE', 'REF2_CODE', 'REF3_CODE']
float_cols = ['REF1_DIST', 'REF1_TRAVEL', 'REF2_DIST', 'REF2_TRAVEL',
             'REF3_DIST', 'REF3_TRAVEL']
# int_cols = ['id', 'BHS_COUNT','CATCHMENT', 'REF1_CODE', 'REF2_CODE',
#             'REF3_CODE',  'MD_NO', 'MD_AUG', 'MD_TOTAL','MD_FT', 'MD_PT',
#             'MD_VISIT', 'RN_NO', 'RN_AUG', 'RN_TOTAL', 'RN_FT', 'RN_PT',
#             'RN_VISIT', 'MW_NO', 'MW_AUG', 'MW_TOTAL', 'MW_FT', 'MW_PT',
#             'MW_VISIT']

rhu[str_cols] = rhu[str_cols].apply(lambda x: x.str.upper().str.strip())
rhu[code_cols] = rhu[code_cols].fillna(0).astype(int)
rhu[float_cols] = rhu[float_cols].astype(float)

rhu[str_cols] = rhu[str_cols].fillna('UNKNOWN')

# Extract short code for merging with the Geographic Coordinates files
rhu['SHORT_HFCODE'] = rhu['HF_CODE'].apply(lambda x: int(x[-6:]))

rhu.to_excel('cleaned/rhu_cleaned.xlsx')
```

### Impute missing information from other tables
As the data is being processed from different tables, the other tables can be used to fill some missing information. Aside from imputing missing information, coordinates outside the Philippines are identified.


```python
# Bounding box for the Philippines (manually extracted from Google).
long_min, lat_min, long_max, lat_max = (117.17427453, 5.58100332277, 126.537423944, 18.5052273625)
```


```python
HF_list = pd.read_excel('cleaned/HFList_cleaned.xlsx')

# Groupby the data to account for duplicate names for different codes
HF_dict = HF_list[['HF_NAME', 'SHORT_HFCODE']].groupby('HF_NAME')['SHORT_HFCODE'].apply(set).to_dict()
latlong_dict = HF_list[['SHORT_HFCODE', 'LAT', 'LONG']].set_index('SHORT_HFCODE').to_dict()

```

#### RHU


```python
rhu = pd.read_excel('cleaned/rhu_cleaned.xlsx')

# Create copies of the dataframe for later use
rhu2 = rhu.copy()
rhu3 = rhu.copy()
```

##### Fill missing REF1 Codes


```python
cols = ['id', 'HF_CODE', 'SHORT_HFCODE', 'HF_NAME', 'REGION', 'PROVINCE', 'MUNI_CITY', 'BRGY',
            'STREET_NAME', 'BUILDING', 'FACILITY_HEAD', 'DETACHED', 'BRGYS',
            'SDN', 'SDN_NAME', 'REF1_NAME', 'REF1_SAMEPROV',
            'REF1_REF1A', 'REF1A_SAMEPROV', 'REF2_NAME', 'REF3_NAME',  
            'AMB_ACCESS', 'AMB_OWN', 'PHIC_ACC', 'PHIC_PACKAGES', 'PHIC_MCP',
            'PHIC_PCB1', 'PHIC_MALARIA', 'PHIC_TBDOTS', 'PHIC_ABTC',
            'PHIC_NCP', 'PHIC_OTH', 'REF1_CODE', 'REF1_DIST', 'REF1_TRAVEL',
            'REF2_CODE', 'REF2_DIST', 'REF2_TRAVEL',
            'REF3_CODE', 'REF3_DIST', 'REF3_TRAVEL']

rhu = rhu[cols]

# Using the health facility list, complete the RHU data
rhu.loc[rhu['REF1_CODE']==0, 'REF_CODE'] = rhu[rhu['REF1_CODE']==0]['REF1_NAME'].map(HF_dict)
```


```python
temp = rhu[['SHORT_HFCODE', 'REF_CODE']].dropna().copy()

# This dataframe contains the HF codes of one health facility to other facilities.
temp.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SHORT_HFCODE</th>
      <th>REF_CODE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>2228</td>
      <td>{3313}</td>
    </tr>
    <tr>
      <th>20</th>
      <td>6698</td>
      <td>{3313}</td>
    </tr>
    <tr>
      <th>29</th>
      <td>147</td>
      <td>{2703}</td>
    </tr>
    <tr>
      <th>36</th>
      <td>2033</td>
      <td>{3667}</td>
    </tr>
    <tr>
      <th>37</th>
      <td>2184</td>
      <td>{2703}</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Out of all the mapped referred facilities, the closest facility is used as the actual referred facility
# Slight pre-processing: "place" the health facilities without coordinates to southwest of southwest Philippine boundary (lat_min - 10, long_min - 20) or northeast of northeast Philippine boundary
temp_dict = pd.DataFrame(temp.apply(lambda x: min([(vincenty((latlong_dict['LAT'][x['SHORT_HFCODE']] if latlong_dict['LAT'][x['SHORT_HFCODE']]==latlong_dict['LAT'][x['SHORT_HFCODE']] else lat_min-10,
                                                  latlong_dict['LONG'][x['SHORT_HFCODE']] if latlong_dict['LONG'][x['SHORT_HFCODE']]==latlong_dict['LONG'][x['SHORT_HFCODE']] else long_min-20),
                                                 (latlong_dict['LAT'][i] if latlong_dict['LAT'][i]==latlong_dict['LAT'][i] else lat_max+10,
                                                  latlong_dict['LONG'][i] if latlong_dict['LONG'][i]==latlong_dict['LONG'][i] else long_max+20)).km, i, x['SHORT_HFCODE']) for i in x['REF_CODE']], key=lambda x: x[0]), axis=1).tolist()).set_index(2).to_dict()

rhu['REF_CODE'] = rhu['SHORT_HFCODE'].map(temp_dict[1])
rhu.loc[rhu['REF1_CODE']!=0, 'REF_CODE'] = rhu.loc[rhu['REF1_CODE']!=0, 'REF1_CODE']
```

##### Fill missing REF2 Codes

The data contains up to three referred facilities so the same processing is performed for the second and third HF code.


```python
rhu2.loc[rhu2['REF2_CODE']==0, 'REF_CODE'] = rhu2[rhu2['REF2_CODE']==0]['REF2_NAME'].map(HF_dict)

temp = rhu2[['SHORT_HFCODE', 'REF_CODE']].dropna().copy()
temp_dict = pd.DataFrame(temp.apply(lambda x: min([(vincenty((latlong_dict['LAT'][x['SHORT_HFCODE']] if latlong_dict['LAT'][x['SHORT_HFCODE']]==latlong_dict['LAT'][x['SHORT_HFCODE']] else lat_min-10,
                                                  latlong_dict['LONG'][x['SHORT_HFCODE']] if latlong_dict['LONG'][x['SHORT_HFCODE']]==latlong_dict['LONG'][x['SHORT_HFCODE']] else long_min-20),
                                                 (latlong_dict['LAT'][i] if latlong_dict['LAT'][i]==latlong_dict['LAT'][i] else lat_max+10,
                                                  latlong_dict['LONG'][i] if latlong_dict['LONG'][i]==latlong_dict['LONG'][i] else long_max+20)).km, i, x['SHORT_HFCODE']) for i in x['REF_CODE']], key=lambda x: x[0]), axis=1).tolist()).set_index(2).to_dict()

rhu2['REF_CODE'] = rhu2['SHORT_HFCODE'].map(temp_dict[1])
rhu2.loc[rhu2['REF2_CODE']!=0, 'REF_CODE'] = rhu2.loc[rhu2['REF2_CODE']!=0, 'REF2_CODE']
```

##### Fill missing REF3 Codes


```python
rhu3.loc[rhu3['REF3_CODE']==0, 'REF_CODE'] = rhu3[rhu3['REF3_CODE']==0]['REF3_NAME'].map(HF_dict)

temp = rhu3[['SHORT_HFCODE', 'REF_CODE']].dropna().copy()
temp_dict = pd.DataFrame(temp.apply(lambda x: min([(vincenty((latlong_dict['LAT'][x['SHORT_HFCODE']] if latlong_dict['LAT'][x['SHORT_HFCODE']]==latlong_dict['LAT'][x['SHORT_HFCODE']] else lat_min-10,
                                                  latlong_dict['LONG'][x['SHORT_HFCODE']] if latlong_dict['LONG'][x['SHORT_HFCODE']]==latlong_dict['LONG'][x['SHORT_HFCODE']] else long_min-20),
                                                 (latlong_dict['LAT'][i] if latlong_dict['LAT'][i]==latlong_dict['LAT'][i] else lat_max+10,
                                                  latlong_dict['LONG'][i] if latlong_dict['LONG'][i]==latlong_dict['LONG'][i] else long_max+20)).km, i, x['SHORT_HFCODE']) for i in x['REF_CODE']], key=lambda x: x[0]), axis=1).tolist()).set_index(2).to_dict()

rhu3['REF_CODE'] = rhu3['SHORT_HFCODE'].map(temp_dict[1])
rhu3.loc[rhu3['REF3_CODE']!=0, 'REF_CODE'] = rhu3.loc[rhu3['REF3_CODE']!=0, 'REF3_CODE']
```


```python
rhu.dropna(subset=['REF_CODE'], inplace=True)
rhu2.dropna(subset=['REF_CODE'], inplace=True)
rhu3.dropna(subset=['REF_CODE'], inplace=True)
```

##### Combine the processed dataframes


```python
rhu.rename({'REF1_DIST':'REF_DIST', 'REF1_TRAVEL':'REF_TRAVEL', 'REF1_NAME':'REF_NAME'}, axis=1, inplace=True)
rhu2.rename({'REF2_DIST':'REF_DIST', 'REF2_TRAVEL':'REF_TRAVEL', 'REF1_NAME':'REF_NAME'}, axis=1, inplace=True)
rhu3.rename({'REF3_DIST':'REF_DIST', 'REF3_TRAVEL':'REF_TRAVEL', 'REF1_NAME':'REF_NAME'}, axis=1, inplace=True)
```


```python
cols2 = ['SHORT_HFCODE', 'REF_CODE']
rhu_edges = rhu[cols2].append(rhu2[cols2]).append(rhu3[cols2])
```


```python
# Add a column identifying the type of facility for the later network analysis

rhu_edges['HF_TYPE'] = 'RHU'
```


```python
HF_list[['SHORT_HFCODE', 'LAT', 'LONG']].describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SHORT_HFCODE</th>
      <th>LAT</th>
      <th>LONG</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>29,424.00</td>
      <td>25,752.00</td>
      <td>25,752.00</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>19,505.19</td>
      <td>12.27</td>
      <td>122.49</td>
    </tr>
    <tr>
      <th>std</th>
      <td>11,203.47</td>
      <td>7.11</td>
      <td>4.42</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>9,724.75</td>
      <td>9.52</td>
      <td>121.02</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>19,799.50</td>
      <td>12.73</td>
      <td>122.38</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>29,542.25</td>
      <td>14.81</td>
      <td>124.16</td>
    </tr>
    <tr>
      <th>max</th>
      <td>38,002.00</td>
      <td>1,000.00</td>
      <td>126.58</td>
    </tr>
  </tbody>
</table>
</div>



##### Map lat-long to the HF codes


```python
rhu_edges['source_lat'] = rhu_edges['SHORT_HFCODE'].map(latlong_dict['LAT'])
rhu_edges['source_long'] = rhu_edges['SHORT_HFCODE'].map(latlong_dict['LONG'])
rhu_edges['target_lat'] = rhu_edges['REF_CODE'].map(latlong_dict['LAT'])
rhu_edges['target_long'] = rhu_edges['REF_CODE'].map(latlong_dict['LONG'])
```

##### Set lat-long outside PH to NaN


```python
rhu_edges.loc[~((rhu_edges['source_lat'].between(lat_min, lat_max)) & (rhu_edges['source_long'].between(
    long_min, long_max))), ['source_lat', 'source_long']] = np.nan
rhu_edges.loc[~((rhu_edges['target_lat'].between(lat_min, lat_max)) & (rhu_edges['target_long'].between(
    long_min, long_max))), ['target_lat', 'target_long']] = np.nan
```

##### Measure distance using lat-long for non-NaN HF pairs


```python
missing_latlong = ~rhu_edges[['source_lat', 'source_long', 'target_lat', 'target_long']].isnull().sum(axis=1).astype(bool)
rhu_edges.loc[missing_latlong, 'DIST'] = rhu_edges.loc[missing_latlong, ['source_lat', 'source_long', 'target_lat', 'target_long']].apply(lambda x: vincenty((x['source_lat'], x['source_long']), (x['target_lat'], x['target_long'])).km, axis=1)

rhu_edges.loc[rhu_edges['DIST']==0, 'DIST'] = 1
```

##### Check for outliers
In this case, referral between facilities that are too high are classified as outliers. The threshold to be used for filtering out these outliers is subject to the researcher's choice. The unit of distance is in kilometers. Note that since the distance is extracted from coordinates, errors in encoding will change the actual coordinates of the facilities, therefore making changes to the distance (e.g. dividing by a constant factor) cannot be done. Instead, other imputing techinques are used to fill for missing distance data.


```python
B = rhu_edges.boxplot('DIST', return_type='both')
```


![png](output_img/output_38_0.png)



```python
outliers = [i.get_ydata()[1] for i in B.lines['whiskers']]
rhu_edges.loc[rhu_edges['DIST'] > outliers[1], 'DIST'] = np.nan
```


```python
outliers
```




    [0.0006040641840882818, 88.56724901828711]




```python
muni_dict = HF_list[['SHORT_HFCODE', 'MUNI_CITY', 'PROVINCE']].set_index('SHORT_HFCODE').to_dict()
```

##### Impute distance from municipality
With the municipality information of the facilities, the distance of different health units are imputed using the median. The assumption is that within the same municipality, the distance of referring facilities are more or less similar. For tracking, the source of the distance information is stored.


```python
rhu_edges['muni_city'] = rhu_edges['SHORT_HFCODE'].map(muni_dict['MUNI_CITY'])
mean_dist_city = rhu_edges.groupby('muni_city')['DIST'].median().to_dict()

imputed_muni = ~(rhu_edges.loc[rhu_edges['DIST'].isnull(), 'muni_city'].map(mean_dist_city).isnull())
imputed_muni = imputed_muni[imputed_muni].index
rhu_edges.loc[imputed_muni, "IMPUTED"] = "MUNI"
rhu_edges.loc[rhu_edges['DIST'].isnull(), 'DIST'] = rhu_edges.loc[rhu_edges['DIST'].isnull(), 'muni_city'].map(mean_dist_city)
```

##### Impute distance from province
For those facilities without municipality information, the province is used.


```python
rhu_edges['province'] = rhu_edges['SHORT_HFCODE'].map(muni_dict['PROVINCE'])
mean_dist_prov = rhu_edges.groupby('province')['DIST'].median().to_dict()

imputed_prov = ~(rhu_edges.loc[rhu_edges['DIST'].isnull(), 'province'].map(mean_dist_prov).isnull())
imputed_prov = imputed_prov[imputed_prov].index

rhu_edges.loc[imputed_prov, "IMPUTED"] = "PROV"
rhu_edges.loc[rhu_edges['DIST'].isnull(), 'DIST'] = rhu_edges.loc[rhu_edges['DIST'].isnull(), 'province'].map(mean_dist_prov)
```


```python
rhu_edges['DIST'].isnull().sum()
```




    17



If after all these, the referring facilities still do not have distance information, the connection between the facilities is dropped.


```python
rhu_edges.dropna(subset=['DIST'], inplace=True)
```


```python
prov_HF_dict = rhu_edges.groupby('province')[['SHORT_HFCODE', 'REF_CODE']].agg(set).rename({'SHORT_HFCODE':'RHU', 'REF_CODE':'HOSP'}, axis=1).to_dict()
```


```python
rhu_edges['REF_CODES'] = [set(rhu_edges[~rhu_edges[['target_lat', 'target_long']].isnull().sum(axis=1).astype(bool)]['REF_CODE'])] * len(rhu_edges)
```


```python
rhu_edges = rhu_edges[rhu_edges['REF_CODE']!=rhu_edges['SHORT_HFCODE']]
```

##### Connect nearest neighbors
The referrals above are based on actual data, i.e. actual referrals from facility to facility. This list is supplemented with the three nearest facilities, regardless of being the same facilities as the actual data.


```python
n = 3 #num of nearest neighbors to connect
temp = rhu_edges[['SHORT_HFCODE', 'REF_CODES']].dropna().drop_duplicates(subset='SHORT_HFCODE').copy()
df_neighbors = pd.DataFrame(temp.apply(lambda x: sorted([(vincenty((latlong_dict['LAT'][x['SHORT_HFCODE']] if latlong_dict['LAT'][x['SHORT_HFCODE']]==latlong_dict['LAT'][x['SHORT_HFCODE']] else lat_min-10,
                                                  latlong_dict['LONG'][x['SHORT_HFCODE']] if latlong_dict['LONG'][x['SHORT_HFCODE']]==latlong_dict['LONG'][x['SHORT_HFCODE']] else long_min-20),
                                                 (latlong_dict['LAT'][i] if latlong_dict['LAT'][i]==latlong_dict['LAT'][i] else lat_max+10,
                                                  latlong_dict['LONG'][i] if latlong_dict['LONG'][i]==latlong_dict['LONG'][i] else long_max+20)).km, i, x['SHORT_HFCODE']) for i in x['REF_CODES'] if i!=x['SHORT_HFCODE']], key=lambda x: x[0])[:n], axis=1).tolist())#.set_index(2).to_dict()
```


```python
df_neighbors_edges = pd.DataFrame(df_neighbors[0].append(df_neighbors[1]).append(df_neighbors[2]).tolist(), columns=['DIST', 'REF_CODE', 'SHORT_HFCODE'])
```


```python
df_neighbors_edges['IMPUTED'] = 'NEAREST_NEIGHBOR'
```


```python
rhu_edges[['SHORT_HFCODE', 'REF_CODE', 'DIST', 'IMPUTED']]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SHORT_HFCODE</th>
      <th>REF_CODE</th>
      <th>DIST</th>
      <th>IMPUTED</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>25</td>
      <td>32,170.00</td>
      <td>0.00</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>105</td>
      <td>5,940.00</td>
      <td>0.41</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>106</td>
      <td>3,313.00</td>
      <td>11.13</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>137</td>
      <td>3,313.00</td>
      <td>11.29</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1760</td>
      <td>6,513.00</td>
      <td>5.98</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2296</th>
      <td>6814</td>
      <td>273.00</td>
      <td>51.12</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2297</th>
      <td>7045</td>
      <td>273.00</td>
      <td>15.07</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2298</th>
      <td>7696</td>
      <td>273.00</td>
      <td>29.09</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2299</th>
      <td>8861</td>
      <td>273.00</td>
      <td>16.35</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2300</th>
      <td>27898</td>
      <td>237.00</td>
      <td>41.68</td>
      <td>MUNI</td>
    </tr>
  </tbody>
</table>
<p>5925 rows × 4 columns</p>
</div>




```python
rhu_edges = rhu_edges[['SHORT_HFCODE', 'REF_CODE', 'DIST', 'IMPUTED']].append(df_neighbors_edges)
```


```python
rhu_edges.loc[rhu_edges['DIST'] > outliers[1], 'DIST'] = np.nan
rhu_edges.loc[rhu_edges['DIST']==0, 'DIST'] = 1
```


```python
rhu_edges.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DIST</th>
      <th>REF_CODE</th>
      <th>SHORT_HFCODE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>12,157.00</td>
      <td>12,330.00</td>
      <td>12,330.00</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>16.11</td>
      <td>4,600.62</td>
      <td>5,816.23</td>
    </tr>
    <tr>
      <th>std</th>
      <td>16.55</td>
      <td>7,159.54</td>
      <td>6,469.21</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.00</td>
      <td>1.00</td>
      <td>10.00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3.39</td>
      <td>622.00</td>
      <td>2,307.00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>11.07</td>
      <td>2,850.00</td>
      <td>4,234.00</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>22.87</td>
      <td>5,139.00</td>
      <td>7,062.00</td>
    </tr>
    <tr>
      <th>max</th>
      <td>88.57</td>
      <td>261,101.00</td>
      <td>36,628.00</td>
    </tr>
  </tbody>
</table>
</div>



#### BHS
The same processing done for the rural health units is performed for the barangay health stations.


```python
bhs = pd.read_excel('bhs2018.xlsx', sheet_name='MAIN', na_values='None')
```


```python
str_cols = ['HF_NAME', 'REGION', 'PROVINCE', 'MUNI_CITY', 'BRGY',
            'STREET_NAME', 'BUILDING', 'FACILITY_HEAD', 'DETACHED',
            'BRGYS', 'RHU_NAME', 'RHU_SAME_CTY', 'RHU_NOTSAME_CTY',
            'AMB_ACCESS']
code_cols = ['id', 'RHU_CODE']
float_cols = ['RHU_DIST', 'RHU_TRAVEL']
# int_cols = ['CATCHMENT', 'MD_NO', 'MD_AUG', 'MD_TOTAL',
#        'MD_FT', 'MD_PT', 'MD_VISIT', 'RN_NO', 'RN_AUG', 'RN_TOTAL', 'RN_FT',
#        'RN_PT', 'RN_VISIT', 'MW_NO', 'MW_AUG', 'MW_TOTAL', 'MW_FT', 'MW_PT',
#        'MW_VISIT', 'BHW_NO']

bhs[str_cols] = bhs[str_cols].apply(lambda x: x.str.upper().str.strip())
bhs[code_cols] = bhs[code_cols].fillna(0).astype(int)
bhs[float_cols] = bhs[float_cols].astype(float)

bhs[str_cols] = bhs[str_cols].fillna('UNKNOWN')
bhs['SHORT_HFCODE'] = bhs['HF_CODE'].apply(lambda x: int(x[-6:]))
bhs.to_excel('cleaned/bhs_cleaned.xlsx')
```

##### Fill missing RHU Codes


```python
bhs = pd.read_excel('cleaned/bhs_cleaned.xlsx')
```


```python
bhs.loc[bhs['RHU_CODE']==0, 'REF_CODE'] = bhs[bhs['RHU_CODE']==0]['RHU_NAME'].map(HF_dict)
```


```python
temp = bhs[['SHORT_HFCODE', 'REF_CODE']].dropna().copy()
temp_dict = pd.DataFrame(temp.apply(lambda x: min([(vincenty((latlong_dict['LAT'][x['SHORT_HFCODE']] if latlong_dict['LAT'][x['SHORT_HFCODE']]==latlong_dict['LAT'][x['SHORT_HFCODE']] else lat_min-10,
                                                  latlong_dict['LONG'][x['SHORT_HFCODE']] if latlong_dict['LONG'][x['SHORT_HFCODE']]==latlong_dict['LONG'][x['SHORT_HFCODE']] else long_min-20),
                                                 (latlong_dict['LAT'][i] if latlong_dict['LAT'][i]==latlong_dict['LAT'][i] else lat_max+10,
                                                  latlong_dict['LONG'][i] if latlong_dict['LONG'][i]==latlong_dict['LONG'][i] else long_max+20)).km, i, x['SHORT_HFCODE']) for i in x['REF_CODE']], key=lambda x: x[0]), axis=1).tolist()).set_index(2).to_dict()

bhs['REF_CODE'] = bhs['SHORT_HFCODE'].map(temp_dict[1])
```


```python
bhs.dropna(subset=['REF_CODE'], inplace=True)
```


```python
cols = ['SHORT_HFCODE', 'REF_CODE']
```


```python
bhs = bhs[cols]
bhs['HF_TYPE'] = 'BHS'
```


```python
bhs['source_lat'] = bhs['SHORT_HFCODE'].map(latlong_dict['LAT'])
bhs['source_long'] = bhs['SHORT_HFCODE'].map(latlong_dict['LONG'])
bhs['target_lat'] = bhs['REF_CODE'].map(latlong_dict['LAT'])
bhs['target_long'] = bhs['REF_CODE'].map(latlong_dict['LONG'])
```


```python
bhs.loc[~((bhs['source_lat'].between(lat_min, lat_max)) & (bhs['source_long'].between(
    long_min, long_max))), ['source_lat', 'source_long']] = np.nan
bhs.loc[~((bhs['target_lat'].between(lat_min, lat_max)) & (bhs['target_long'].between(
    long_min, long_max))), ['target_lat', 'target_long']] = np.nan
```


```python
missing_latlong = ~bhs[['source_lat', 'source_long', 'target_lat', 'target_long']].isnull().sum(axis=1).astype(bool)
bhs.loc[missing_latlong, 'DIST'] = bhs.loc[missing_latlong, ['source_lat', 'source_long', 'target_lat', 'target_long']].apply(lambda x: vincenty((x['source_lat'], x['source_long']), (x['target_lat'], x['target_long'])).km, axis=1)
```


```python
bhs.loc[bhs['DIST']==0, 'DIST'] = 1
```


```python
B = bhs.boxplot('DIST', return_type='both')
```


![png](output_img/output_74_0.png)



```python
outliers = [i.get_ydata()[1] for i in B.lines['whiskers']]
```


```python
outliers
```




    [0.0024470114519412256, 16.988283224538524]




```python
outliers = [i.get_ydata()[1] for i in B.lines['whiskers']]
bhs.loc[bhs['DIST'] > outliers[1], 'DIST'] = np.nan
```


```python
muni_dict = HF_list[['SHORT_HFCODE', 'MUNI_CITY', 'PROVINCE']].set_index('SHORT_HFCODE').to_dict()
```


```python
bhs['muni_city'] = bhs['SHORT_HFCODE'].map(muni_dict['MUNI_CITY'])
mean_dist_city = bhs.groupby('muni_city')['DIST'].mean().to_dict()

imputed_muni = ~(bhs.loc[bhs['DIST'].isnull(), 'muni_city'].map(mean_dist_city).isnull())
imputed_muni = imputed_muni[imputed_muni].index
bhs.loc[imputed_muni, "IMPUTED"] = "MUNI"
bhs.loc[bhs['DIST'].isnull(), 'DIST'] = bhs.loc[bhs['DIST'].isnull(), 'muni_city'].map(mean_dist_city)
```


```python
bhs['province'] = bhs['SHORT_HFCODE'].map(muni_dict['PROVINCE'])
mean_dist_prov = bhs.groupby('province')['DIST'].mean().to_dict()

imputed_prov = ~(bhs.loc[bhs['DIST'].isnull(), 'province'].map(mean_dist_prov).isnull())
imputed_prov = imputed_prov[imputed_prov].index

bhs.loc[imputed_prov, "IMPUTED"] = "PROV"
bhs.loc[bhs['DIST'].isnull(), 'DIST'] = bhs.loc[bhs['DIST'].isnull(), 'province'].map(mean_dist_prov)
```


```python
bhs['DIST'].isnull().sum()
```




    6




```python
bhs.dropna(subset=['DIST'], inplace=True)
```


```python
bhs = bhs[['SHORT_HFCODE', 'REF_CODE', 'DIST', 'IMPUTED']]
```


```python
bhs = bhs[bhs['REF_CODE'] != bhs['SHORT_HFCODE']]
```


```python
edge_list = rhu_edges.append(bhs)
edge_list
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DIST</th>
      <th>IMPUTED</th>
      <th>REF_CODE</th>
      <th>SHORT_HFCODE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00</td>
      <td>NaN</td>
      <td>32,170.00</td>
      <td>25</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.41</td>
      <td>NaN</td>
      <td>5,940.00</td>
      <td>105</td>
    </tr>
    <tr>
      <th>2</th>
      <td>11.13</td>
      <td>NaN</td>
      <td>3,313.00</td>
      <td>106</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11.29</td>
      <td>NaN</td>
      <td>3,313.00</td>
      <td>137</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5.98</td>
      <td>NaN</td>
      <td>6,513.00</td>
      <td>1760</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>19297</th>
      <td>8.88</td>
      <td>NaN</td>
      <td>7,207.00</td>
      <td>17805</td>
    </tr>
    <tr>
      <th>19325</th>
      <td>3.56</td>
      <td>NaN</td>
      <td>79.00</td>
      <td>17893</td>
    </tr>
    <tr>
      <th>19345</th>
      <td>0.65</td>
      <td>NaN</td>
      <td>27,529.00</td>
      <td>27530</td>
    </tr>
    <tr>
      <th>19367</th>
      <td>1.68</td>
      <td>NaN</td>
      <td>169.00</td>
      <td>29236</td>
    </tr>
    <tr>
      <th>19368</th>
      <td>12.67</td>
      <td>NaN</td>
      <td>169.00</td>
      <td>29238</td>
    </tr>
  </tbody>
</table>
<p>12793 rows × 4 columns</p>
</div>




```python
edge_list = edge_list.groupby(['REF_CODE', 'SHORT_HFCODE'])[['IMPUTED', 'DIST']].agg({'DIST':'mean', 'IMPUTED':'first'}).reset_index()
```


```python
edge_list.to_excel('edge_list.xlsx', index=False)
```

## Network Analysis

Using the facilities as nodes and the referral as edges, a tripartite network of BHS, RHU, and hospitals is created. The characteristics of the healthcare provider network for different regions are characterized and explored in this section. The main metric used in the analysis is degree and path length.


```python
import pandas as pd
import networkx as nx
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import itertools
import random
```


```python
nodes = pd.read_excel("nodeslist.xlsx")
edges = pd.read_excel("edge_list.xlsx")
```

For the succeeding analysis, the imputed distances using the nearest neighbors are not included.


```python
edges = edges[(edges['IMPUTED']!='NEAREST_NEIGHBOR') | (edges['SHORT_HFCODE']!=6108)]
```


```python
nodes.drop(['LAT', 'LONG'], axis=1, inplace=True)
```


```python
G = nx.from_pandas_edgelist(edges, source='SHORT_HFCODE', target='REF_CODE', edge_attr='DIST', create_using=nx.DiGraph)
```


```python
nodes.index = nodes['SHORT_HFCODE']
```


```python
node_attrib = nodes.to_dict()
```


```python
for col in nodes.columns:
    nx.set_node_attributes(G, node_attrib[col], col)
```


```python
bhs_nodes = [i for i, j in G.nodes(data=True) if j.get('HF_TYPE')=='BARANGAY HEALTH STATION']
rhu_nodes = [i for i, j in G.nodes(data=True) if j.get('HF_TYPE')=='RURAL HEALTH UNIT']
hosp_nodes = [i for i, j in G.nodes(data=True) if ((j.get('HF_TYPE')=='HOSPITAL') or (j.get('HF_TYPE')=='INFIRMARY'))]
```

### Degree (Hospital)


```python
import seaborn as sns
```


```python
region_deg = {}
for region in nodes['REGION'].unique():
    deg = list(dict(G.in_degree(nodes[((nodes['HF_TYPE']=='INFIRMARY') | (nodes['HF_TYPE']=='HOSPITAL')) & (nodes['REGION']==region)]['SHORT_HFCODE'])).values())
    region_deg[region] = deg
```


```python
df_deg = pd.DataFrame(list(itertools.chain(*[list(zip(len(j) * [i], j)) for i, j in region_deg.items()]))).rename({0:'Region', 1:'Degree'}, axis=1)#.boxplot(by=0)
```


```python
region_map = {'AUTONOMOUS REGION IN MUSLIM MINDANAO (ARMM)':'ARMM',
       'CORDILLERA ADMINISTRA TIVE REGION (CAR)':'CAR',
       'REGION IV-B (MIMAROPA)':'IV-B', 'NATIONAL CAPITAL REGION (NCR)':'NCR',
       'REGION X (NORTHERN MINDANAO)':'X', 'REGION XI (DAVAO REGION)':'XI',
       'REGION XII (SOCCSKSA RGEN)':'XII', 'REGION XIII (CARAGA)':'XIII',
       'REGION I (ILOCOS REGION)':'I', 'REGION II (CAGAYAN VALLEY)':'II',
       'REGION III (CENTRAL LUZON)':'III', 'REGION IV-A (CALABAR ZON)':'IV-A',
       'REGION V (BICOL REGION)':'V', 'REGION VI (WESTERN VISAYAS)':'VI',
       'REGION VII (CENTRAL VISAYAS)':'VII', 'REGION VIII (EASTERN VISAYAS)':'VIII',
       'REGION IX (ZAMBOANGA PENINSULA)':'IX'}
```


```python
df_deg['Region'] = df_deg['Region'].map(region_map)
```


```python
(df_deg.groupby('Region')['Degree']
 .agg(['min', 'max', 'mean', 'median', 'count']).sort_values(by='Region')
 [['max', 'mean', 'median', 'count']])
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>max</th>
      <th>mean</th>
      <th>median</th>
      <th>count</th>
    </tr>
    <tr>
      <th>Region</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ARMM</th>
      <td>24</td>
      <td>6.89</td>
      <td>4.50</td>
      <td>18</td>
    </tr>
    <tr>
      <th>CAR</th>
      <td>32</td>
      <td>9.42</td>
      <td>6.50</td>
      <td>36</td>
    </tr>
    <tr>
      <th>I</th>
      <td>68</td>
      <td>15.97</td>
      <td>10.00</td>
      <td>36</td>
    </tr>
    <tr>
      <th>II</th>
      <td>58</td>
      <td>10.26</td>
      <td>7.00</td>
      <td>34</td>
    </tr>
    <tr>
      <th>III</th>
      <td>153</td>
      <td>18.53</td>
      <td>13.00</td>
      <td>57</td>
    </tr>
    <tr>
      <th>IV-A</th>
      <td>144</td>
      <td>12.16</td>
      <td>8.00</td>
      <td>57</td>
    </tr>
    <tr>
      <th>IV-B</th>
      <td>59</td>
      <td>8.52</td>
      <td>6.00</td>
      <td>33</td>
    </tr>
    <tr>
      <th>IX</th>
      <td>53</td>
      <td>14.33</td>
      <td>11.00</td>
      <td>21</td>
    </tr>
    <tr>
      <th>NCR</th>
      <td>73</td>
      <td>31.84</td>
      <td>29.00</td>
      <td>45</td>
    </tr>
    <tr>
      <th>V</th>
      <td>62</td>
      <td>11.69</td>
      <td>8.00</td>
      <td>39</td>
    </tr>
    <tr>
      <th>VI</th>
      <td>96</td>
      <td>10.76</td>
      <td>7.00</td>
      <td>50</td>
    </tr>
    <tr>
      <th>VII</th>
      <td>59</td>
      <td>10.47</td>
      <td>8.00</td>
      <td>45</td>
    </tr>
    <tr>
      <th>VIII</th>
      <td>146</td>
      <td>15.79</td>
      <td>11.00</td>
      <td>39</td>
    </tr>
    <tr>
      <th>X</th>
      <td>47</td>
      <td>8.50</td>
      <td>6.00</td>
      <td>36</td>
    </tr>
    <tr>
      <th>XI</th>
      <td>30</td>
      <td>7.90</td>
      <td>6.00</td>
      <td>21</td>
    </tr>
    <tr>
      <th>XII</th>
      <td>46</td>
      <td>9.17</td>
      <td>5.50</td>
      <td>18</td>
    </tr>
    <tr>
      <th>XIII</th>
      <td>37</td>
      <td>7.72</td>
      <td>6.00</td>
      <td>25</td>
    </tr>
  </tbody>
</table>
</div>



### Degree Outliers (Hospital)


```python
upper_fence = df_deg.groupby('Region')['Degree'].quantile(0.75) + 1.5 * (df_deg.groupby('Region')['Degree'].quantile(0.75) - df_deg.groupby('Region')['Degree'].quantile(0.25))
```


```python
fig, ax = plt.subplots(figsize=(10,8))

# fig = plt.figure()
sns.boxplot(data=df_deg, x='Region', y='Degree', palette='Blues',
            order=upper_fence.sort_values(ascending=False).index)
labels = [l.get_text() for l in ax.get_xticklabels()]
ax.set_xticklabels(labels, ha='right');
```


![png](output_img/output_108_0.png)



```python
df_deg['Outlierness'] = df_deg['Degree'] - df_deg['Region'].map(upper_fence)
deg_outlier = df_deg[df_deg['Outlierness'] > 0].groupby('Region')['Outlierness'].agg(['min', 'max', 'mean', 'median', 'count'])
deg_outlier
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>min</th>
      <th>max</th>
      <th>mean</th>
      <th>median</th>
      <th>count</th>
    </tr>
    <tr>
      <th>Region</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ARMM</th>
      <td>4.88</td>
      <td>4.88</td>
      <td>4.88</td>
      <td>4.88</td>
      <td>1</td>
    </tr>
    <tr>
      <th>CAR</th>
      <td>1.62</td>
      <td>12.62</td>
      <td>7.38</td>
      <td>7.62</td>
      <td>4</td>
    </tr>
    <tr>
      <th>I</th>
      <td>1.38</td>
      <td>39.38</td>
      <td>26.62</td>
      <td>32.88</td>
      <td>4</td>
    </tr>
    <tr>
      <th>II</th>
      <td>3.75</td>
      <td>37.75</td>
      <td>16.75</td>
      <td>8.75</td>
      <td>3</td>
    </tr>
    <tr>
      <th>III</th>
      <td>6.50</td>
      <td>125.50</td>
      <td>40.17</td>
      <td>28.00</td>
      <td>6</td>
    </tr>
    <tr>
      <th>IV-A</th>
      <td>8.00</td>
      <td>123.00</td>
      <td>39.50</td>
      <td>13.50</td>
      <td>4</td>
    </tr>
    <tr>
      <th>IV-B</th>
      <td>2.00</td>
      <td>45.00</td>
      <td>14.25</td>
      <td>5.00</td>
      <td>4</td>
    </tr>
    <tr>
      <th>IX</th>
      <td>17.00</td>
      <td>17.00</td>
      <td>17.00</td>
      <td>17.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>NCR</th>
      <td>4.50</td>
      <td>4.50</td>
      <td>4.50</td>
      <td>4.50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>V</th>
      <td>12.75</td>
      <td>42.75</td>
      <td>32.75</td>
      <td>42.75</td>
      <td>3</td>
    </tr>
    <tr>
      <th>VI</th>
      <td>7.62</td>
      <td>76.62</td>
      <td>27.38</td>
      <td>12.62</td>
      <td>4</td>
    </tr>
    <tr>
      <th>VII</th>
      <td>2.00</td>
      <td>38.00</td>
      <td>14.00</td>
      <td>8.00</td>
      <td>4</td>
    </tr>
    <tr>
      <th>VIII</th>
      <td>0.75</td>
      <td>123.75</td>
      <td>29.95</td>
      <td>4.75</td>
      <td>5</td>
    </tr>
    <tr>
      <th>X</th>
      <td>17.00</td>
      <td>32.00</td>
      <td>24.50</td>
      <td>24.50</td>
      <td>2</td>
    </tr>
    <tr>
      <th>XI</th>
      <td>11.50</td>
      <td>13.50</td>
      <td>12.50</td>
      <td>12.50</td>
      <td>2</td>
    </tr>
    <tr>
      <th>XII</th>
      <td>25.12</td>
      <td>25.12</td>
      <td>25.12</td>
      <td>25.12</td>
      <td>1</td>
    </tr>
    <tr>
      <th>XIII</th>
      <td>2.00</td>
      <td>23.00</td>
      <td>12.50</td>
      <td>12.50</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



### Degree (RHU)


```python
import seaborn as sns
```


```python
region_deg = {}
for region in nodes['REGION'].unique():
    deg = list(dict(G.in_degree(nodes[(nodes['HF_TYPE']=='RURAL HEALTH UNIT') & (nodes['REGION']==region)]['SHORT_HFCODE'])).values())
    region_deg[region] = deg
```


```python
df_deg = pd.DataFrame(list(itertools.chain(*[list(zip(len(j) * [i], j)) for i, j in region_deg.items()]))).rename({0:'Region', 1:'Degree'}, axis=1)#.boxplot(by=0)
```


```python
region_map = {'AUTONOMOUS REGION IN MUSLIM MINDANAO (ARMM)':'ARMM',
       'CORDILLERA ADMINISTRA TIVE REGION (CAR)':'CAR',
       'REGION IV-B (MIMAROPA)':'IV-B', 'NATIONAL CAPITAL REGION (NCR)':'NCR',
       'REGION X (NORTHERN MINDANAO)':'X', 'REGION XI (DAVAO REGION)':'XI',
       'REGION XII (SOCCSKSA RGEN)':'XII', 'REGION XIII (CARAGA)':'XIII',
       'REGION I (ILOCOS REGION)':'I', 'REGION II (CAGAYAN VALLEY)':'II',
       'REGION III (CENTRAL LUZON)':'III', 'REGION IV-A (CALABAR ZON)':'IV-A',
       'REGION V (BICOL REGION)':'V', 'REGION VI (WESTERN VISAYAS)':'VI',
       'REGION VII (CENTRAL VISAYAS)':'VII', 'REGION VIII (EASTERN VISAYAS)':'VIII',
       'REGION IX (ZAMBOANGA PENINSULA)':'IX'}
```


```python
df_deg['Region'] = df_deg['Region'].map(region_map)
```


```python
(df_deg.groupby('Region')['Degree']
 .agg(['min', 'max', 'mean', 'median', 'count']).sort_values(by='Region')
 [['max', 'mean', 'median', 'count']])
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>max</th>
      <th>mean</th>
      <th>median</th>
      <th>count</th>
    </tr>
    <tr>
      <th>Region</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ARMM</th>
      <td>8</td>
      <td>0.13</td>
      <td>0</td>
      <td>75</td>
    </tr>
    <tr>
      <th>CAR</th>
      <td>13</td>
      <td>0.40</td>
      <td>0</td>
      <td>96</td>
    </tr>
    <tr>
      <th>I</th>
      <td>13</td>
      <td>0.35</td>
      <td>0</td>
      <td>144</td>
    </tr>
    <tr>
      <th>II</th>
      <td>5</td>
      <td>0.24</td>
      <td>0</td>
      <td>82</td>
    </tr>
    <tr>
      <th>III</th>
      <td>15</td>
      <td>0.15</td>
      <td>0</td>
      <td>260</td>
    </tr>
    <tr>
      <th>IV-A</th>
      <td>24</td>
      <td>0.65</td>
      <td>0</td>
      <td>164</td>
    </tr>
    <tr>
      <th>IV-B</th>
      <td>6</td>
      <td>0.38</td>
      <td>0</td>
      <td>79</td>
    </tr>
    <tr>
      <th>IX</th>
      <td>22</td>
      <td>0.89</td>
      <td>0</td>
      <td>95</td>
    </tr>
    <tr>
      <th>NCR</th>
      <td>25</td>
      <td>0.25</td>
      <td>0</td>
      <td>382</td>
    </tr>
    <tr>
      <th>V</th>
      <td>13</td>
      <td>0.82</td>
      <td>0</td>
      <td>129</td>
    </tr>
    <tr>
      <th>VI</th>
      <td>11</td>
      <td>0.56</td>
      <td>0</td>
      <td>141</td>
    </tr>
    <tr>
      <th>VII</th>
      <td>18</td>
      <td>0.90</td>
      <td>0</td>
      <td>129</td>
    </tr>
    <tr>
      <th>VIII</th>
      <td>16</td>
      <td>0.54</td>
      <td>0</td>
      <td>160</td>
    </tr>
    <tr>
      <th>X</th>
      <td>13</td>
      <td>0.62</td>
      <td>0</td>
      <td>81</td>
    </tr>
    <tr>
      <th>XI</th>
      <td>11</td>
      <td>0.44</td>
      <td>0</td>
      <td>54</td>
    </tr>
    <tr>
      <th>XII</th>
      <td>25</td>
      <td>1.34</td>
      <td>0</td>
      <td>35</td>
    </tr>
    <tr>
      <th>XIII</th>
      <td>8</td>
      <td>0.73</td>
      <td>0</td>
      <td>56</td>
    </tr>
  </tbody>
</table>
</div>



### Degree Outliers (RHU)


```python
upper_fence = df_deg.groupby('Region')['Degree'].quantile(0.75) + 1.5 * (df_deg.groupby('Region')['Degree'].quantile(0.75) - df_deg.groupby('Region')['Degree'].quantile(0.25))
```


```python
df_deg[df_deg['Region']=='CAR']
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Region</th>
      <th>Degree</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>75</th>
      <td>CAR</td>
      <td>1</td>
    </tr>
    <tr>
      <th>76</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>77</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>78</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>79</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>80</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>81</th>
      <td>CAR</td>
      <td>3</td>
    </tr>
    <tr>
      <th>82</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>83</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>84</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>85</th>
      <td>CAR</td>
      <td>1</td>
    </tr>
    <tr>
      <th>86</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>87</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>88</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>89</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>90</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>91</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>92</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>93</th>
      <td>CAR</td>
      <td>5</td>
    </tr>
    <tr>
      <th>94</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>96</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>97</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>98</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>99</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>100</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>101</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>102</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>103</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>104</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>105</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>106</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>107</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>108</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>109</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>110</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>111</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>112</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>113</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>114</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>115</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>116</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>117</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>118</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>119</th>
      <td>CAR</td>
      <td>1</td>
    </tr>
    <tr>
      <th>120</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>121</th>
      <td>CAR</td>
      <td>1</td>
    </tr>
    <tr>
      <th>122</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>123</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>124</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>125</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>126</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>127</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>128</th>
      <td>CAR</td>
      <td>13</td>
    </tr>
    <tr>
      <th>129</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>130</th>
      <td>CAR</td>
      <td>1</td>
    </tr>
    <tr>
      <th>131</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>132</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>133</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>134</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>135</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>136</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>137</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>138</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>139</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>140</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>141</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>142</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>143</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>144</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>145</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>146</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>147</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>148</th>
      <td>CAR</td>
      <td>7</td>
    </tr>
    <tr>
      <th>149</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>150</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>151</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>152</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>153</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>154</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>155</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>156</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>157</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>158</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>159</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>160</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>161</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>162</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>163</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>164</th>
      <td>CAR</td>
      <td>5</td>
    </tr>
    <tr>
      <th>165</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>166</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>167</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>168</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>169</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>170</th>
      <td>CAR</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots(figsize=(10,8))

# fig = plt.figure()
sns.boxplot(data=df_deg, x='Region', y='Degree', palette='Blues',
            order=upper_fence.sort_values(ascending=False).index)
labels = [l.get_text() for l in ax.get_xticklabels()]
ax.set_xticklabels(labels, ha='right');
```


![png](output_img/output_120_0.png)



```python
df_deg['Outlierness'] = df_deg['Degree'] - df_deg['Region'].map(upper_fence)
deg_outlier_rhu = df_deg[df_deg['Outlierness'] > 0].groupby('Region')['Outlierness'].agg(['min', 'max', 'mean', 'median', 'count'])
deg_outlier_rhu.loc['CAR'] = 0
```

### Average path length


```python
sp = list(nx.all_pairs_dijkstra_path_length(G, weight='DIST'))
```


```python
sp = pd.DataFrame(sp).set_index(0)[1].to_dict()
```

##### BHS -> RHU path


```python
bhs_rhu_ave_sp = {}
for rhu_node in rhu_nodes:
    ave_sp = 0
    N = 0
    for bhs_node in bhs_nodes:
        if sp.get(bhs_node) and sp.get(bhs_node, {}).get(rhu_node):
            ave_sp += sp[bhs_node][rhu_node]
            N += 1
    if N:
        ave_sp /= N
        bhs_rhu_ave_sp[rhu_node] = ave_sp
```


```python
df_bhs_rhu = pd.DataFrame(bhs_rhu_ave_sp.items(), columns=['HF_CODE', 'AVE_SPL'])
df_bhs_rhu['REGION'] = df_bhs_rhu['HF_CODE'].map(nodes['REGION']).map(region_map)
```


```python
bhs_rhu_spl = df_bhs_rhu.groupby('REGION')['AVE_SPL'].agg(['min', 'max', 'mean', 'median', 'count'])
```


```python
bhs_rhu_spl
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>min</th>
      <th>max</th>
      <th>mean</th>
      <th>median</th>
      <th>count</th>
    </tr>
    <tr>
      <th>REGION</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ARMM</th>
      <td>4.29</td>
      <td>4.29</td>
      <td>4.29</td>
      <td>4.29</td>
      <td>1</td>
    </tr>
    <tr>
      <th>CAR</th>
      <td>2.61</td>
      <td>12.68</td>
      <td>7.60</td>
      <td>8.42</td>
      <td>6</td>
    </tr>
    <tr>
      <th>I</th>
      <td>0.98</td>
      <td>20.15</td>
      <td>5.96</td>
      <td>3.36</td>
      <td>8</td>
    </tr>
    <tr>
      <th>II</th>
      <td>2.36</td>
      <td>16.81</td>
      <td>10.96</td>
      <td>10.32</td>
      <td>10</td>
    </tr>
    <tr>
      <th>III</th>
      <td>0.35</td>
      <td>9.25</td>
      <td>3.64</td>
      <td>3.11</td>
      <td>5</td>
    </tr>
    <tr>
      <th>IV-A</th>
      <td>0.83</td>
      <td>15.99</td>
      <td>4.22</td>
      <td>4.24</td>
      <td>15</td>
    </tr>
    <tr>
      <th>IV-B</th>
      <td>6.16</td>
      <td>21.42</td>
      <td>13.79</td>
      <td>13.79</td>
      <td>2</td>
    </tr>
    <tr>
      <th>IX</th>
      <td>3.83</td>
      <td>42.95</td>
      <td>16.33</td>
      <td>8.21</td>
      <td>6</td>
    </tr>
    <tr>
      <th>NCR</th>
      <td>1.52</td>
      <td>83.05</td>
      <td>53.08</td>
      <td>63.88</td>
      <td>4</td>
    </tr>
    <tr>
      <th>V</th>
      <td>1.22</td>
      <td>57.83</td>
      <td>16.51</td>
      <td>7.84</td>
      <td>14</td>
    </tr>
    <tr>
      <th>VI</th>
      <td>0.00</td>
      <td>32.29</td>
      <td>8.31</td>
      <td>5.91</td>
      <td>17</td>
    </tr>
    <tr>
      <th>VII</th>
      <td>0.37</td>
      <td>12.33</td>
      <td>4.32</td>
      <td>3.99</td>
      <td>21</td>
    </tr>
    <tr>
      <th>VIII</th>
      <td>0.00</td>
      <td>49.12</td>
      <td>13.45</td>
      <td>5.65</td>
      <td>10</td>
    </tr>
    <tr>
      <th>X</th>
      <td>3.17</td>
      <td>22.97</td>
      <td>9.34</td>
      <td>7.30</td>
      <td>9</td>
    </tr>
    <tr>
      <th>XI</th>
      <td>3.79</td>
      <td>12.32</td>
      <td>6.77</td>
      <td>4.19</td>
      <td>3</td>
    </tr>
    <tr>
      <th>XII</th>
      <td>4.33</td>
      <td>33.07</td>
      <td>10.53</td>
      <td>5.78</td>
      <td>6</td>
    </tr>
    <tr>
      <th>XIII</th>
      <td>0.65</td>
      <td>24.45</td>
      <td>7.07</td>
      <td>5.99</td>
      <td>13</td>
    </tr>
  </tbody>
</table>
</div>



##### Comparison with Degree Outliers


```python
fig, ax = plt.subplots(figsize=(10,8))
ax.plot(deg_outlier_rhu['mean'], bhs_rhu_spl['mean'], 'o', color='deepskyblue',
        markersize='15')
for i,j,k in zip(deg_outlier_rhu['mean'].values, bhs_rhu_spl['mean'].values, list(deg_outlier_rhu.index)):
    ax.text(i,j,k, fontsize=14)
ax.set_xlabel("Degree Outlierness (RHU)", fontsize=14)
ax.set_ylabel("Average shortest path length (BHS->RHU)", fontsize=14);
```


![png](output_img/output_131_0.png)


##### RHU -> HOSP path


```python
rhu_hosp_ave_sp = {}
for hosp_node in hosp_nodes:
    ave_sp = 0
    N = 0
    for rhu_node in rhu_nodes:
        if sp.get(rhu_node) and sp.get(rhu_node, {}).get(hosp_node):
            ave_sp += sp[rhu_node][hosp_node]
            N += 1
    if N:
        ave_sp /= N
        rhu_hosp_ave_sp[hosp_node] = ave_sp
```


```python
df_rhu_hosp = pd.DataFrame(rhu_hosp_ave_sp.items(), columns=['HF_CODE', 'AVE_SPL'])
df_rhu_hosp['REGION'] = df_rhu_hosp['HF_CODE'].map(nodes['REGION']).map(region_map)
```


```python
rhu_hosp_spl = df_rhu_hosp.groupby('REGION')['AVE_SPL'].agg(['min', 'max', 'mean', 'median', 'count'])
rhu_hosp_spl
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>min</th>
      <th>max</th>
      <th>mean</th>
      <th>median</th>
      <th>count</th>
    </tr>
    <tr>
      <th>REGION</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ARMM</th>
      <td>13.96</td>
      <td>41.71</td>
      <td>27.51</td>
      <td>25.55</td>
      <td>18</td>
    </tr>
    <tr>
      <th>CAR</th>
      <td>6.52</td>
      <td>30.51</td>
      <td>15.63</td>
      <td>14.41</td>
      <td>35</td>
    </tr>
    <tr>
      <th>I</th>
      <td>5.73</td>
      <td>32.67</td>
      <td>16.12</td>
      <td>12.81</td>
      <td>36</td>
    </tr>
    <tr>
      <th>II</th>
      <td>0.22</td>
      <td>43.62</td>
      <td>18.39</td>
      <td>14.65</td>
      <td>32</td>
    </tr>
    <tr>
      <th>III</th>
      <td>3.53</td>
      <td>33.86</td>
      <td>12.97</td>
      <td>10.79</td>
      <td>57</td>
    </tr>
    <tr>
      <th>IV-A</th>
      <td>3.13</td>
      <td>72.94</td>
      <td>15.79</td>
      <td>14.95</td>
      <td>57</td>
    </tr>
    <tr>
      <th>IV-B</th>
      <td>5.94</td>
      <td>54.63</td>
      <td>27.00</td>
      <td>25.51</td>
      <td>27</td>
    </tr>
    <tr>
      <th>IX</th>
      <td>0.42</td>
      <td>56.19</td>
      <td>23.47</td>
      <td>21.24</td>
      <td>21</td>
    </tr>
    <tr>
      <th>NCR</th>
      <td>1.20</td>
      <td>33.93</td>
      <td>7.21</td>
      <td>3.62</td>
      <td>45</td>
    </tr>
    <tr>
      <th>V</th>
      <td>10.95</td>
      <td>52.01</td>
      <td>22.81</td>
      <td>19.48</td>
      <td>39</td>
    </tr>
    <tr>
      <th>VI</th>
      <td>7.50</td>
      <td>40.18</td>
      <td>16.84</td>
      <td>15.35</td>
      <td>47</td>
    </tr>
    <tr>
      <th>VII</th>
      <td>5.43</td>
      <td>45.77</td>
      <td>20.58</td>
      <td>18.71</td>
      <td>45</td>
    </tr>
    <tr>
      <th>VIII</th>
      <td>7.55</td>
      <td>55.53</td>
      <td>25.25</td>
      <td>23.95</td>
      <td>39</td>
    </tr>
    <tr>
      <th>X</th>
      <td>1.53</td>
      <td>53.28</td>
      <td>20.09</td>
      <td>16.63</td>
      <td>36</td>
    </tr>
    <tr>
      <th>XI</th>
      <td>7.14</td>
      <td>48.44</td>
      <td>26.17</td>
      <td>23.32</td>
      <td>21</td>
    </tr>
    <tr>
      <th>XII</th>
      <td>13.03</td>
      <td>41.43</td>
      <td>20.74</td>
      <td>17.68</td>
      <td>17</td>
    </tr>
    <tr>
      <th>XIII</th>
      <td>7.66</td>
      <td>40.07</td>
      <td>19.27</td>
      <td>18.85</td>
      <td>25</td>
    </tr>
  </tbody>
</table>
</div>



##### Comparison with Degree Outliers


```python
fig, ax = plt.subplots(figsize=(10,8))
ax.plot(deg_outlier['mean'], rhu_hosp_spl['mean'], 'o', color='deepskyblue',
        markersize='15')
for i,j,k in zip(deg_outlier['mean'].values, rhu_hosp_spl['mean'].values,
                 list(deg_outlier.index)):
    ax.text(i,j,k, fontsize=14)
ax.set_xlabel("Degree Outlierness (Hosp)", fontsize=14)
ax.set_ylabel("Average shortest path length (RHU->Hosp)", fontsize=14);

#I changed the ylabel to RHU->Hosp
```


![png](output_img/output_137_0.png)


##### BHS -> HOSP path


```python
bhs_hosp_ave_sp = {}
for hosp_node in hosp_nodes:
    ave_sp = 0
    N = 0
    for bhs_node in bhs_nodes:
        if sp.get(bhs_node) and sp.get(bhs_node, {}).get(hosp_node):
            ave_sp += sp[bhs_node][hosp_node]
            N += 1
    if N:
        ave_sp /= N
        bhs_hosp_ave_sp[hosp_node] = ave_sp
```


```python
df_bhs_hosp = pd.DataFrame(bhs_hosp_ave_sp.items(), columns=['HF_CODE', 'AVE_SPL'])
df_bhs_hosp['REGION'] = df_bhs_hosp['HF_CODE'].map(nodes['REGION']).map(region_map)
```


```python
bhs_hosp_spl = df_bhs_hosp.groupby('REGION')['AVE_SPL'].agg(['min', 'max', 'mean', 'median', 'count'])
bhs_hosp_spl
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>min</th>
      <th>max</th>
      <th>mean</th>
      <th>median</th>
      <th>count</th>
    </tr>
    <tr>
      <th>REGION</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ARMM</th>
      <td>12.17</td>
      <td>38.93</td>
      <td>25.24</td>
      <td>24.61</td>
      <td>3</td>
    </tr>
    <tr>
      <th>CAR</th>
      <td>15.29</td>
      <td>62.75</td>
      <td>27.42</td>
      <td>22.69</td>
      <td>16</td>
    </tr>
    <tr>
      <th>I</th>
      <td>1.74</td>
      <td>69.47</td>
      <td>22.28</td>
      <td>17.81</td>
      <td>18</td>
    </tr>
    <tr>
      <th>II</th>
      <td>9.06</td>
      <td>75.55</td>
      <td>39.42</td>
      <td>36.04</td>
      <td>19</td>
    </tr>
    <tr>
      <th>III</th>
      <td>1.13</td>
      <td>63.65</td>
      <td>18.09</td>
      <td>13.53</td>
      <td>15</td>
    </tr>
    <tr>
      <th>IV-A</th>
      <td>1.45</td>
      <td>49.21</td>
      <td>18.65</td>
      <td>17.41</td>
      <td>25</td>
    </tr>
    <tr>
      <th>IV-B</th>
      <td>11.32</td>
      <td>23.08</td>
      <td>17.20</td>
      <td>17.20</td>
      <td>2</td>
    </tr>
    <tr>
      <th>IX</th>
      <td>10.07</td>
      <td>52.89</td>
      <td>33.75</td>
      <td>39.08</td>
      <td>9</td>
    </tr>
    <tr>
      <th>NCR</th>
      <td>3.09</td>
      <td>86.34</td>
      <td>53.10</td>
      <td>65.95</td>
      <td>10</td>
    </tr>
    <tr>
      <th>V</th>
      <td>16.30</td>
      <td>63.85</td>
      <td>35.28</td>
      <td>31.63</td>
      <td>18</td>
    </tr>
    <tr>
      <th>VI</th>
      <td>7.13</td>
      <td>60.55</td>
      <td>25.50</td>
      <td>22.54</td>
      <td>28</td>
    </tr>
    <tr>
      <th>VII</th>
      <td>2.91</td>
      <td>64.14</td>
      <td>20.35</td>
      <td>20.19</td>
      <td>27</td>
    </tr>
    <tr>
      <th>VIII</th>
      <td>13.70</td>
      <td>73.88</td>
      <td>38.24</td>
      <td>35.28</td>
      <td>17</td>
    </tr>
    <tr>
      <th>X</th>
      <td>15.62</td>
      <td>41.22</td>
      <td>25.33</td>
      <td>21.74</td>
      <td>14</td>
    </tr>
    <tr>
      <th>XI</th>
      <td>8.32</td>
      <td>55.47</td>
      <td>28.33</td>
      <td>26.42</td>
      <td>6</td>
    </tr>
    <tr>
      <th>XII</th>
      <td>10.54</td>
      <td>41.89</td>
      <td>26.88</td>
      <td>25.69</td>
      <td>7</td>
    </tr>
    <tr>
      <th>XIII</th>
      <td>7.16</td>
      <td>44.44</td>
      <td>23.76</td>
      <td>22.36</td>
      <td>18</td>
    </tr>
  </tbody>
</table>
</div>



##### Comparison with Degree Outliers


```python
fig, ax = plt.subplots(figsize=(10,8))
ax.plot(deg_outlier['mean'], bhs_hosp_spl['mean'], 'o', color='deepskyblue',
        markersize='15')
for i,j,k in zip(deg_outlier['mean'].values, bhs_hosp_spl['mean'].values, list(deg_outlier.index)):
    ax.text(i,j,k, fontsize=14)
ax.set_xlabel("Degree Outlierness (Hosp)", fontsize=14)
ax.set_ylabel("Average shortest path length (BHS->Hosp)", fontsize=14);


## I changed the ylabel to BHS -> Hosp
```


![png](output_img/output_143_0.png)


## Simulations
After characterizing the network, further analysis were performed by simulating what would happen to the network as higher percentages of the population enters the system. That is, the facilities that would overload are identified based on the bed capacity.


```python
nodes.groupby('HF_TYPE')['CATCHMENT'].describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>HF_TYPE</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>BARANGAY HEALTH STATION</th>
      <td>17,435.00</td>
      <td>3,800.18</td>
      <td>6,013.85</td>
      <td>0.00</td>
      <td>1,302.00</td>
      <td>2,444.00</td>
      <td>4,408.50</td>
      <td>123,708.00</td>
    </tr>
    <tr>
      <th>HOSPITAL</th>
      <td>0.00</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
    </tr>
    <tr>
      <th>INFIRMARY</th>
      <td>0.00</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
    </tr>
    <tr>
      <th>RURAL HEALTH UNIT</th>
      <td>2,172.00</td>
      <td>38,663.14</td>
      <td>44,947.38</td>
      <td>0.00</td>
      <td>16,981.75</td>
      <td>30,224.00</td>
      <td>46,929.25</td>
      <td>903,309.00</td>
    </tr>
  </tbody>
</table>
</div>




```python
intake = nodes.loc[(nodes['HF_TYPE']=='RURAL HEALTH UNIT') | (nodes['HF_TYPE']=='BARANGAY HEALTH STATION'), ['REGION', 'CATCHMENT', 'HF_TYPE']]
```


```python
intake.groupby('REGION')['CATCHMENT'].agg(['mean','median', 'min', 'max'])
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>median</th>
      <th>min</th>
      <th>max</th>
    </tr>
    <tr>
      <th>REGION</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AUTONOMOUS REGION IN MUSLIM MINDANAO (ARMM)</th>
      <td>9,252.73</td>
      <td>2,682.00</td>
      <td>0.00</td>
      <td>222,885.00</td>
    </tr>
    <tr>
      <th>CORDILLERA ADMINISTRA TIVE REGION (CAR)</th>
      <td>3,688.14</td>
      <td>1,187.00</td>
      <td>0.00</td>
      <td>134,461.00</td>
    </tr>
    <tr>
      <th>NATIONAL CAPITAL REGION (NCR)</th>
      <td>30,745.29</td>
      <td>25,084.00</td>
      <td>0.00</td>
      <td>290,977.00</td>
    </tr>
    <tr>
      <th>REGION I (ILOCOS REGION)</th>
      <td>6,133.94</td>
      <td>2,556.00</td>
      <td>0.00</td>
      <td>204,135.00</td>
    </tr>
    <tr>
      <th>REGION II (CAGAYAN VALLEY)</th>
      <td>4,528.67</td>
      <td>1,826.00</td>
      <td>0.00</td>
      <td>158,245.00</td>
    </tr>
    <tr>
      <th>REGION III (CENTRAL LUZON)</th>
      <td>9,866.27</td>
      <td>3,700.50</td>
      <td>0.00</td>
      <td>260,691.00</td>
    </tr>
    <tr>
      <th>REGION IV-A (CALABAR ZON)</th>
      <td>9,040.42</td>
      <td>3,122.50</td>
      <td>0.00</td>
      <td>463,727.00</td>
    </tr>
    <tr>
      <th>REGION IV-B (MIMAROPA)</th>
      <td>5,363.06</td>
      <td>2,104.50</td>
      <td>0.00</td>
      <td>272,190.00</td>
    </tr>
    <tr>
      <th>REGION IX (ZAMBOANGA PENINSULA)</th>
      <td>9,427.20</td>
      <td>3,773.50</td>
      <td>0.00</td>
      <td>903,309.00</td>
    </tr>
    <tr>
      <th>REGION V (BICOL REGION)</th>
      <td>7,685.29</td>
      <td>3,480.50</td>
      <td>0.00</td>
      <td>210,047.00</td>
    </tr>
    <tr>
      <th>REGION VI (WESTERN VISAYAS)</th>
      <td>7,620.31</td>
      <td>3,293.00</td>
      <td>0.00</td>
      <td>597,740.00</td>
    </tr>
    <tr>
      <th>REGION VII (CENTRAL VISAYAS)</th>
      <td>5,827.44</td>
      <td>2,131.00</td>
      <td>0.00</td>
      <td>414,265.00</td>
    </tr>
    <tr>
      <th>REGION VIII (EASTERN VISAYAS)</th>
      <td>9,092.89</td>
      <td>4,577.00</td>
      <td>0.00</td>
      <td>387,813.00</td>
    </tr>
    <tr>
      <th>REGION X (NORTHERN MINDANAO)</th>
      <td>6,047.59</td>
      <td>2,369.00</td>
      <td>0.00</td>
      <td>197,890.00</td>
    </tr>
    <tr>
      <th>REGION XI (DAVAO REGION)</th>
      <td>5,155.10</td>
      <td>1,818.50</td>
      <td>0.00</td>
      <td>175,935.00</td>
    </tr>
    <tr>
      <th>REGION XII (SOCCSKSA RGEN)</th>
      <td>8,722.27</td>
      <td>2,168.00</td>
      <td>0.00</td>
      <td>453,585.00</td>
    </tr>
    <tr>
      <th>REGION XIII (CARAGA)</th>
      <td>7,212.89</td>
      <td>2,530.00</td>
      <td>0.00</td>
      <td>137,527.00</td>
    </tr>
  </tbody>
</table>
</div>




```python
num_iter=1000
#check changes in the system if 0.001% to 1% of population enters the HCPN;
pop_prop = 0.00001
hosp_zero_cap = nodes[((nodes['HF_TYPE']=='HOSPITAL') | (nodes['HF_TYPE']=='INFIRMARY')) & (nodes['BED_CAP']==0)].SHORT_HFCODE.values
```


```python
len(hosp_zero_cap)
```




    0




```python
fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(9, 6))

intake_t = intake['CATCHMENT'] * pop_prop
df_isolates = pd.DataFrame()
df_rem_hosps = pd.DataFrame()
# ax2 = ax1.twinx()
for region in ['NATIONAL CAPITAL REGION (NCR)', 'REGION VIII (EASTERN VISAYAS)']:
    rhu_reg = nodes[(nodes['REGION']==region) & (nodes['HF_TYPE']=='RURAL HEALTH UNIT')]['SHORT_HFCODE']
    hosp_reg = nodes[(nodes['REGION']==region) & ((nodes['HF_TYPE']=='HOSPITAL') | (nodes['HF_TYPE']=='INFIRMARY'))]['SHORT_HFCODE']
    G_reg = G.subgraph(rhu_reg.tolist() + hosp_reg.tolist()).copy()
    hosp_reg_connected = [i for i in hosp_reg if ((i not in nx.isolates(G_reg)) and (i not in hosp_zero_cap))]
    remaining_hosps = [len(hosp_reg_connected)]
    G_reg.remove_nodes_from(list(nx.isolates(G_reg)))
    G_reg.remove_nodes_from(hosp_zero_cap)
    num_isolates = [0]
    remaining_hosp = [i for i, j in G_reg.nodes(data=True) if ((j['HF_TYPE']=='HOSPITAL') or (j['HF_TYPE']=='INFIRMARY'))]
    # t=0
    # while remaining_hosp:
    for t in range(num_iter):
        removed_hosps = []
        for rhu in rhu_reg:
            try:
                N = len(list(G_reg.neighbors(rhu)))
            except nx.NetworkXError:
                continue
            nearby_hosps = list(G_reg.neighbors(rhu))
#             dist_sum = G_reg.out_degree(rhu, weight='DIST')
            dist_sum = sum([G_reg[rhu][i]['DIST'] for i in nearby_hosps if G_reg[rhu][i]['DIST']==G_reg[rhu][i]['DIST']])
            for hosp in nearby_hosps:
                if len(nearby_hosps)==1:
                    prob_hosp = 1
                else:
                    prob_hosp = (1 - G_reg[rhu][hosp]['DIST'] / dist_sum) if G_reg[rhu][hosp]['DIST']==G_reg[rhu][hosp]['DIST'] else 0
                new_patient = intake_t[rhu] * prob_hosp if intake_t[rhu]==intake_t[rhu] else 0
                add_one = 1 if random.random()<new_patient%1 else 0
                if (G_reg.nodes()[hosp].get('patients', 0) + int(new_patient) + add_one) > G_reg.nodes()[hosp]['BED_CAP']:
                    G_reg.remove_node(hosp)
                    removed_hosps.append(hosp)
                else:
                    nx.set_node_attributes(G_reg, {hosp: G_reg.nodes()[hosp].get('patients', 0) + int(new_patient) + add_one}, 'patients')
        remaining_hosp = [i for i, j in G_reg.nodes(data=True) if ((j['HF_TYPE']=='HOSPITAL') or (j['HF_TYPE']=='INFIRMARY'))]
        remaining_hosps.append(len(remaining_hosp))
        num_isolates.append(len(list(nx.isolates(G_reg))))
    #     print(t, removed_hosps)
    df_rem_hosps.loc[region, "0.05%"] = remaining_hosps[np.where(pop_prop * np.arange(num_iter+1)==0.0005)[0][0]] / remaining_hosps[0]
    df_rem_hosps.loc[region, "0.5%"] = remaining_hosps[np.where(pop_prop * np.arange(num_iter+1)==0.005)[0][0]] / remaining_hosps[0]
    df_isolates.loc[region, "0.05%"] = num_isolates[np.where(pop_prop * np.arange(num_iter+1)==0.0005)[0][0]] / num_isolates[-1]
    df_isolates.loc[region, "0.5%"] = num_isolates[np.where(pop_prop * np.arange(num_iter+1)==0.005)[0][0]] / num_isolates[-1]

#     print("At 0.05%, remaining hosps are", remaining_hosps[np.where(pop_prop * np.arange(num_iter+1)==0.0005)[0][0]] / remaining_hosps[0], region)
#     print("At 0.5%, remaining hosps are", remaining_hosps[np.where(pop_prop * np.arange(num_iter+1)==0.005)[0][0]] / remaining_hosps[0], region)
#     print("At 0.05%, the isolated RHUs are", num_isolates[np.where(pop_prop * np.arange(num_iter+1)==0.0005)[0][0]] / num_isolates[-1], region)
#     print("At 0.5%, the isolated RHUs are", num_isolates[np.where(pop_prop * np.arange(num_iter+1)==0.005)[0][0]] / num_isolates[-1], region)
    ax1.plot(pop_prop * np.arange(num_iter+1), [i/remaining_hosps[0] for i in remaining_hosps], label=region)
    ax2.plot(pop_prop * np.arange(num_iter+1), [i/num_isolates[-1] for i in num_isolates])
fig.legend()
ax1.set_xlabel('Percent of population in HCPN')
ax2.set_xlabel('Percent of population in HCPN')

ax1.set_xticklabels(['-0.2%', '0.0%', '0.2%', '0.4%', '0.6%', '0.8%', '1.0%'])
ax2.set_xticklabels(['-0.2%', '0.0%', '0.2%', '0.4%', '0.6%', '0.8%', '1.0%'])
ax1.set_ylabel('Percentage of remaining hospitals')
ax2.set_ylabel('Percentage of isolated RHUs')
    #     t+=1
    #     break
```




    Text(0, 0.5, 'Percentage of isolated RHUs')




![png](output_img/output_150_1.png)



```python
fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(9, 6))

intake_t = intake['CATCHMENT'] * pop_prop
df_isolates = pd.DataFrame()
df_rem_hosps = pd.DataFrame()
# ax2 = ax1.twinx()
for region in nodes['REGION'].unique():
    rhu_reg = nodes[(nodes['REGION']==region) & (nodes['HF_TYPE']=='RURAL HEALTH UNIT')]['SHORT_HFCODE']
    hosp_reg = nodes[(nodes['REGION']==region) & ((nodes['HF_TYPE']=='HOSPITAL') | (nodes['HF_TYPE']=='INFIRMARY'))]['SHORT_HFCODE']
    G_reg = G.subgraph(rhu_reg.tolist() + hosp_reg.tolist()).copy()
    hosp_reg_connected = [i for i in hosp_reg if ((i not in nx.isolates(G_reg)) and (i not in hosp_zero_cap))]
    remaining_hosps = [len(hosp_reg_connected)]
    G_reg.remove_nodes_from(list(nx.isolates(G_reg)))
    G_reg.remove_nodes_from(hosp_zero_cap)
    num_isolates = [0]
    remaining_hosp = [i for i, j in G_reg.nodes(data=True) if ((j['HF_TYPE']=='HOSPITAL') or (j['HF_TYPE']=='INFIRMARY'))]
    # t=0
    # while remaining_hosp:
    for t in range(num_iter):
        removed_hosps = []
        for rhu in rhu_reg:
            try:
                N = len(list(G_reg.neighbors(rhu)))
            except nx.NetworkXError:
                continue
            nearby_hosps = list(G_reg.neighbors(rhu))
#             dist_sum = G_reg.out_degree(rhu, weight='DIST')
            dist_sum = sum([G_reg[rhu][i]['DIST'] for i in nearby_hosps if G_reg[rhu][i]['DIST']==G_reg[rhu][i]['DIST']])
            for hosp in nearby_hosps:
                if len(nearby_hosps)==1:
                    prob_hosp = 1
                else:
                    prob_hosp = (1 - G_reg[rhu][hosp]['DIST'] / dist_sum) if G_reg[rhu][hosp]['DIST']==G_reg[rhu][hosp]['DIST'] else 0
                new_patient = intake_t[rhu] * prob_hosp if intake_t[rhu]==intake_t[rhu] else 0
                add_one = 1 if random.random()<new_patient%1 else 0
                if (G_reg.nodes()[hosp].get('patients', 0) + int(new_patient) + add_one) > G_reg.nodes()[hosp]['BED_CAP']:
                    G_reg.remove_node(hosp)
                    removed_hosps.append(hosp)
                else:
                    nx.set_node_attributes(G_reg, {hosp: G_reg.nodes()[hosp].get('patients', 0) + int(new_patient) + add_one}, 'patients')
        remaining_hosp = [i for i, j in G_reg.nodes(data=True) if ((j['HF_TYPE']=='HOSPITAL') or (j['HF_TYPE']=='INFIRMARY'))]
        remaining_hosps.append(len(remaining_hosp))
        num_isolates.append(len(list(nx.isolates(G_reg))))
    #     print(t, removed_hosps)
    df_rem_hosps.loc[region, "0.05%"] = remaining_hosps[np.where(pop_prop * np.arange(num_iter+1)==0.0005)[0][0]] / remaining_hosps[0]
    df_rem_hosps.loc[region, "0.5%"] = remaining_hosps[np.where(pop_prop * np.arange(num_iter+1)==0.005)[0][0]] / remaining_hosps[0]
    df_isolates.loc[region, "0.05%"] = num_isolates[np.where(pop_prop * np.arange(num_iter+1)==0.0005)[0][0]] / num_isolates[-1]
    df_isolates.loc[region, "0.5%"] = num_isolates[np.where(pop_prop * np.arange(num_iter+1)==0.005)[0][0]] / num_isolates[-1]

#     print("At 0.05%, remaining hosps are", remaining_hosps[np.where(pop_prop * np.arange(num_iter+1)==0.0005)[0][0]] / remaining_hosps[0], region)
#     print("At 0.5%, remaining hosps are", remaining_hosps[np.where(pop_prop * np.arange(num_iter+1)==0.005)[0][0]] / remaining_hosps[0], region)
#     print("At 0.05%, the isolated RHUs are", num_isolates[np.where(pop_prop * np.arange(num_iter+1)==0.0005)[0][0]] / num_isolates[-1], region)
#     print("At 0.5%, the isolated RHUs are", num_isolates[np.where(pop_prop * np.arange(num_iter+1)==0.005)[0][0]] / num_isolates[-1], region)
    ax1.plot(pop_prop * np.arange(num_iter+1), [i/remaining_hosps[0] for i in remaining_hosps], label=region)
    ax2.plot(pop_prop * np.arange(num_iter+1), [i/num_isolates[-1] for i in num_isolates])
fig.legend()
ax1.set_xlabel('Percent of population in HCPN')
ax2.set_xlabel('Percent of population in HCPN')

ax1.set_xticklabels(['-0.2%', '0.0%', '0.2%', '0.4%', '0.6%', '0.8%', '1.0%'])
ax2.set_xticklabels(['-0.2%', '0.0%', '0.2%', '0.4%', '0.6%', '0.8%', '1.0%'])
ax1.set_ylabel('Percentage of remaining hospitals')
ax2.set_ylabel('Percentage of isolated RHUs')
    #     t+=1
    #     break
```




    Text(0, 0.5, 'Percentage of isolated RHUs')




![png](output_img/output_151_1.png)



```python
print("Remaining hospitals")
(df_rem_hosps*100)
```

    Remaining hospitals





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0.05%</th>
      <th>0.5%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AUTONOMOUS REGION IN MUSLIM MINDANAO (ARMM)</th>
      <td>22.22</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>CORDILLERA ADMINISTRA TIVE REGION (CAR)</th>
      <td>38.89</td>
      <td>2.78</td>
    </tr>
    <tr>
      <th>REGION IV-B (MIMAROPA)</th>
      <td>26.47</td>
      <td>2.94</td>
    </tr>
    <tr>
      <th>NATIONAL CAPITAL REGION (NCR)</th>
      <td>27.08</td>
      <td>8.33</td>
    </tr>
    <tr>
      <th>REGION X (NORTHERN MINDANAO)</th>
      <td>22.22</td>
      <td>5.56</td>
    </tr>
    <tr>
      <th>REGION XI (DAVAO REGION)</th>
      <td>26.92</td>
      <td>19.23</td>
    </tr>
    <tr>
      <th>REGION XII (SOCCSKSA RGEN)</th>
      <td>28.57</td>
      <td>19.05</td>
    </tr>
    <tr>
      <th>REGION XIII (CARAGA)</th>
      <td>26.92</td>
      <td>11.54</td>
    </tr>
    <tr>
      <th>REGION I (ILOCOS REGION)</th>
      <td>12.82</td>
      <td>5.13</td>
    </tr>
    <tr>
      <th>REGION II (CAGAYAN VALLEY)</th>
      <td>8.11</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>REGION III (CENTRAL LUZON)</th>
      <td>4.84</td>
      <td>1.61</td>
    </tr>
    <tr>
      <th>REGION IV-A (CALABAR ZON)</th>
      <td>8.20</td>
      <td>1.64</td>
    </tr>
    <tr>
      <th>REGION V (BICOL REGION)</th>
      <td>4.88</td>
      <td>2.44</td>
    </tr>
    <tr>
      <th>REGION VI (WESTERN VISAYAS)</th>
      <td>6.00</td>
      <td>2.00</td>
    </tr>
    <tr>
      <th>REGION VII (CENTRAL VISAYAS)</th>
      <td>10.42</td>
      <td>4.17</td>
    </tr>
    <tr>
      <th>REGION VIII (EASTERN VISAYAS)</th>
      <td>12.82</td>
      <td>5.13</td>
    </tr>
    <tr>
      <th>REGION IX (ZAMBOANGA PENINSULA)</th>
      <td>17.39</td>
      <td>13.04</td>
    </tr>
  </tbody>
</table>
</div>




```python
print("RHU isolates")
(df_isolates*100)
```

    RHU isolates





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0.05%</th>
      <th>0.5%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AUTONOMOUS REGION IN MUSLIM MINDANAO (ARMM)</th>
      <td>67.92</td>
      <td>100.00</td>
    </tr>
    <tr>
      <th>CORDILLERA ADMINISTRA TIVE REGION (CAR)</th>
      <td>12.33</td>
      <td>100.00</td>
    </tr>
    <tr>
      <th>REGION IV-B (MIMAROPA)</th>
      <td>58.70</td>
      <td>100.00</td>
    </tr>
    <tr>
      <th>NATIONAL CAPITAL REGION (NCR)</th>
      <td>45.27</td>
      <td>92.59</td>
    </tr>
    <tr>
      <th>REGION X (NORTHERN MINDANAO)</th>
      <td>52.63</td>
      <td>100.00</td>
    </tr>
    <tr>
      <th>REGION XI (DAVAO REGION)</th>
      <td>10.53</td>
      <td>100.00</td>
    </tr>
    <tr>
      <th>REGION XII (SOCCSKSA RGEN)</th>
      <td>46.67</td>
      <td>100.00</td>
    </tr>
    <tr>
      <th>REGION XIII (CARAGA)</th>
      <td>68.57</td>
      <td>100.00</td>
    </tr>
    <tr>
      <th>REGION I (ILOCOS REGION)</th>
      <td>96.12</td>
      <td>100.00</td>
    </tr>
    <tr>
      <th>REGION II (CAGAYAN VALLEY)</th>
      <td>82.05</td>
      <td>100.00</td>
    </tr>
    <tr>
      <th>REGION III (CENTRAL LUZON)</th>
      <td>94.42</td>
      <td>100.00</td>
    </tr>
    <tr>
      <th>REGION IV-A (CALABAR ZON)</th>
      <td>91.11</td>
      <td>100.00</td>
    </tr>
    <tr>
      <th>REGION V (BICOL REGION)</th>
      <td>82.35</td>
      <td>100.00</td>
    </tr>
    <tr>
      <th>REGION VI (WESTERN VISAYAS)</th>
      <td>79.79</td>
      <td>100.00</td>
    </tr>
    <tr>
      <th>REGION VII (CENTRAL VISAYAS)</th>
      <td>97.47</td>
      <td>100.00</td>
    </tr>
    <tr>
      <th>REGION VIII (EASTERN VISAYAS)</th>
      <td>92.59</td>
      <td>100.00</td>
    </tr>
    <tr>
      <th>REGION IX (ZAMBOANGA PENINSULA)</th>
      <td>100.00</td>
      <td>100.00</td>
    </tr>
  </tbody>
</table>
</div>
