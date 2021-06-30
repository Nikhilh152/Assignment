# importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from mpl_toolkits.basemap import Basemap


# question 1

# import the dataset
df = pd.read_excel ('2019-20-australian-government-contract-data.xlsx')

# check the head
df.head()

# check the shape of the data
df.shape	

# check to make sure the data type of the columns is accurate
df.dtypes

# selecting the contracts for which the end date is greater than 2021
# store this subset data in a new variable
subset_df = df[df['End Date']>'2021-12-31']
subset_df.head()

# reference - https://stackoverflow.com/questions/49735683/python-removing-rows-on-count-condition
# set a threshold - select agencies & suppliers having contract > 8
subset_agency = subset_df.groupby('Agency Name').filter(lambda x : len(x)>8)
subset_supplier = subset_df.groupby('Supplier Name').filter(lambda x : len(x)>8)

# consider the contracts in which the duration of years > 3 and the amount is greater than 10 million
subset_year_amount = subset_df.loc[(subset_df['Duration Years'] > 3) | (subset_df['Value'] > 10000000) ]

# append the subset_year_amount in both the dataframes subset_agency and subset_supplier
subset_agency = subset_agency.append(subset_year_amount)
subset_supplier = subset_supplier.append(subset_year_amount)

# drop the records which might be duplicate
subset_agency = subset_agency.drop_duplicates()
subset_supplier = subset_supplier.drop_duplicates()


#the above list will be the list of suppliers and the Agencies

#######################################################################################################

#question 2) a)

# to get the location
# Find the list of location of the agencies
list_locations_agencies = subset_agency['Office Postcode']
count_location = list_locations_agencies.value_counts()

# Find the list of location of the suppliers
list_locations_suppliers = subset_supplier['Supplier Postcode']
count_location_suppliers = list_locations_suppliers.value_counts()

# append the location suppliers list and location agencies list to get the total value counts of the locations
total_location_list = list_locations_agencies.append(list_locations_suppliers)
total_loc = total_location_list.value_counts()

# map the postcodes to longitude and latitude
# reference - https://www.matthewproctor.com/australian_postcodes 
# accessed a dataset from the above link 
cordinates_df = pd.read_csv('australian_postcodes.csv')

# only specific columns are requied form the dataset
cordinates_df_subset = cordinates_df[['postcode', 'lat', 'long', 'state']]
# drop the duplicate values
cordinates_df_subset = cordinates_df_subset.drop_duplicates(subset='postcode', keep='first')

# merge the two dataframes location and cordintaes
complete_data = pd.merge(location_counts, cordinates_df_subset, on = 'postcode')

# find the value counts for each state to get an approximate value
complete_data['state'].value_counts

# to plot the bar Graph
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
states = ['NSW','ACT','VIC','QLD','SA' ,'WA' ,'TAS','NT']
count_values = data['state'].value_counts()
ax.bar(states,count_values)
plt.title('Number of PostCodes in each state for Agency and Supplier Office')
plt.show()

# cordinates for basemap of australia
OZMINLAT = -44+1     # (South East Cape Tasmania 43° 38' 40" S 146° 49' 30" E)
OZMAXLAT = -9+1.5    # (Boigu Island 09° 16' S 142° 13' E)
OZMINLON = 112.5-4.1 # (Steep point 26° 09' 5" S 113° 09' 18" E
OZMAXLON = 154-2.5   # (Cape Byron 28° 38' 15" S 153° 38' 14" E)
OZMIDLAT = -25.6
OZMIDLON = 134.35

fig = plt.figure(figsize = (12,9))

m = Basemap(llcrnrlon=OZMINLON, llcrnrlat=OZMINLAT, 
    urcrnrlon=OZMAXLON, urcrnrlat=OZMAXLAT, resolution='i',
    lat_0=OZMIDLAT, lon_0=OZMIDLON, 
    area_thresh=100.,projection='lcc', width = 8E6, height = 8E6)
 
m.drawcoastlines()

m.drawparallels(np.arange(-90,90,10), labels=[True, False, False, False])
m.drawmeridians(np.arange(-180,180,30), labels=[0,0,0,1])

office_lat_y = location_office_subset['lat'].tolist()
office_lon_x = location_office_subset['long'].tolist()

# alpha sets the transparency and s = size
# set k to highlight the edge color
# use zorder to stack - works like 3d
m.scatter(office_lon_x, office_lat_y,latlon=True, c='blue', alpha= 1, edgecolor='k', s=100)
plt.title('Office Locations for Bank X', fontsize= 20)
plt.show()

# to find the distance between two cordinates
# reference - https://stackoverflow.com/questions/19412462/getting-distance-between-two-points-based-on-latitude-longitude

from math import sin, cos, sqrt, atan2, radians

# approximate radius of earth in km
R = 6373.0

lat1 = radians(-35.4197)
lon1 = radians(149.069)
lat2 = radians(-35.431011)
lon2 = radians(149.110025)

dlon = lon2 - lon1
dlat = lat2 - lat1

a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
c = 2 * atan2(sqrt(a), sqrt(1 - a))

distance = R * c

print("Result:", distance, 'km' )

####################################################################################################################

# question 2) b) 
import pgeocode

nomi = pgeocode.Nominatim('au')

# keep incrementing the postcode until county_name changes to a city which is not a capital city
def exclude_capital_city(postcode):
  for i in range(postcode,postcode + 100):
    res = nomi.query_postal_code(i)
    if 'MELB' not in res['county_name']:
      return res

result = exclude_capital_city(3053)
result

###########################################################################################################################

# Question 3) 

df = pd.read_excel('2019-20-australian-government-contract-data.xlsx')

# selecting only 2 Agencies and 2 UNSPSC Titles 
data_df = df[(df['Agency Name'] == 'Department of Defence') | (df['Agency Name'] == 'Department of Home Affairs')]

data_df = data_df[(data_df['UNSPSC Title'] == 'Computer services') | (data_df['UNSPSC Title'] == 'Market research') ]

columns_li = ['Agency Name', 'Value', 'UNSPSC Title', 'Duration Years' ]

# select only the above specific columns from the dataset
data = data_df[columns_li]

# one  hot encoding for agency
data['Agency Name'] = pd.get_dummies(data['Agency Name'], drop_first=True)

# one  hot encoding for UNSPSC
data['UNSPSC Title'] = pd.get_dummies(data['UNSPSC Title'], drop_first=True)

# find the euclidean distance from all the data point to the query point
euclidean_dist =  np.sqrt(((1 - df['Agency Name'])**2) + ((0 - df['UNSPSC Title'])**2)  + (3 - df['Duration Years']**2))

dist_sort_ascending = euclidean_dist.sort_values()

# displaying the 2 least euclidean distances
dist_sort_ascending[0:2]

# use slicing to find the index of the 5 instances with the least euclidean distance
index_points = dist_sort_ascending[0:2].index
index_points

# add all the carat values present at the corresponding indexes of the dataframe df
contract_value = 0
for i in index_points:
    contract_value = (data.iloc[i]['Value']) + contract_value

# diving by 2 to find the avg value of Contract for the query point
contract_value = round((contract_value/2),3)

print(contract_value)



































