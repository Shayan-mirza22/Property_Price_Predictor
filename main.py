import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib

df = pd.read_csv("Bengaluru_House_Data.csv")
#print(df.head())

#print(df.shape)

df1 = df.drop(['area_type', 'society', 'balcony', 'availability'], axis= 'columns')
#print(df1.head())

#print(df1.isnull().sum())

df2 = df1.dropna()
#print(df2.isnull().sum())

#print(df2['size'].unique())

df2['bhk'] = df2['size'].apply(lambda x: int(x.split(' ')[0]))
#print(df2.head())

#print(df2[df2.bhk > 20])            # unrealistic data wrt total_sqft

def is_float(x):
    try:
        float(x)
    except:
        return False
    return True

#print(df2[~df2['total_sqft'].apply(is_float)].head())     # Filtering the normal values from the ranges. By using the negation (~) sign, only ranges can be seen. If it is not used, only floating values will be seen. The ~ sign can only be used with boolian variables.


def sqft_to_num(x):       # Calculating and using the avgs of ranges and if not ranges then only return the float value and if anything like 2300ounce etc return None
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None
    
df3 = df2.copy()

df3['total_sqft'] = df3['total_sqft'].apply(sqft_to_num)
#print(df3.head())


#print(df3.loc[410])

#df.duplicated()            # Boolean series showing duplicates
#print(df3[df3.duplicated(keep=False)])        # Show duplicate rows
#df.drop_duplicates(inplace=True)  # Remove duplicates
# No ned of checking for duplicates

df4 = df3.copy()
df4['price_per_sqft'] = df4['price']*100000 / df4['total_sqft']      # Adding a new column in df4
#print(df4.head())

#print((df4.location.unique()))     # Printing the unique location names, if they are many in number, this can be a huge problem!

df4.location = df4.location.apply(lambda x: x.strip())
location_stats = df4.groupby('location')['location'].agg('count').sort_values(ascending= False)     # Returns the number of houses (Data points) in each location and arranges them in descdeing order
#print(location_stats)

location_stats_less_than_10 = location_stats[location_stats <= 10]     # Only those locations where number of houses is less than or equal to 10
#print(location_stats_less_than_10)

df4.location = df4.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
#print(len(df4.location.unique()))

######################
# Outlier Detection
######################

# In this portion, we will detect any training example that doesn't seem realistc! Like a 1500 sqft house having 1 Bedroom. These are called as outliers. We will set a certain threshold for sqft/BHK then accordingly will remove the ones which do no come within the threshold!

df5 = df4[~(df4.total_sqft/df4.bhk<300)]     # Below the normal ones now removed
#print(df5.shape)  

# Use .describe and see the results of max, min, std and mean to have an idea of more outliers

#print(df5.price_per_sqft.describe())


def remove_outliers(df):
    # key is the location name.
    # subdf contains only the rows of that particular location.
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):        # df.groupby('location') splits your dataset into multiple small DataFrames (subdf) — one for each location.
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]     # core filtering line
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)      
    return df_out

df6 = remove_outliers(df5)
#print(df6.shape)


def plot_scatter_chart(df, location):
    bhk2 = df[(df.location == location) & (df.bhk == 2)]
    bhk3 = df[(df.location == location) & (df.bhk == 3)]
    matplotlib.rcParams['figure.figsize'] = (15, 10)      # Setting the size of the frame
    plt.scatter(bhk2.total_sqft, bhk2.price, color = 'blue', label='2 bhk', s=50)     # s is the size of each marker
    plt.scatter(bhk3.total_sqft, bhk3.price, color = 'green', label='3 bhk', s=50)
    plt.xlabel("Total sqft area")
    plt.ylabel("Price per sqft")
    plt.title(location)
    plt.legend()
    plt.show()

""" #  What this plot tells you:
Whether 2 BHK and 3 BHK properties follow a similar pricing pattern per square foot.

Whether larger properties (3 BHK) are more expensive per sqft or cheaper per sqft.

You can spot outliers (e.g., a 3 BHK that's far too cheap or expensive compared to 2 BHKs).

Useful for understanding pricing inconsistency in specific locations — which is very important for building a better model. """

#plot_scatter_chart(df6, "Rajaji Nagar")
#plot_scatter_chart(df6, "Hebbal")

def remove_bhk_outliers(df):
    exclude_indices = np.array([])      # Hold the indices (row numbers) of the outliers to be removed from the dataset.
    for location, location_df in df.groupby('location'):       # location_df is a smaller DataFrame that only includes data from one location (e.g., "Indira Nagar").
        bhk_stats = {}      # A dictionary to store the mean, std and count for usage later
        for bhk, bhk_df in location_df.groupby('bhk'):        # Similar to location_df, bhk_Df is smaller datadrame here
            bhk_stats[bhk] = {
                'mean' : np.mean(bhk_df.price_per_sqft),
                'std' : np.std(bhk_df.price_per_sqft),
                'count' : bhk_df.shape[0]
            }
        
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)      #Comparing the bhk group with the bhk-1 group (e.g., 3 BHK compared with 2 BHK).
            if stats and stats['count'] > 5:       # Ensure having more than 5 rows for proper comparison
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)      # Is this larger BHK (bhk) cheaper per sqft than the average of smaller BHK (bhk-1)? If yes, it's an outlier.
    return df.drop(exclude_indices, axis='index')      # all rows with indices in exclude_indices are dropped from the DataFrame. axis='index' makes sure we’re dropping rows, not columns.

df7 = remove_bhk_outliers(df6)
print(df7.shape)

#plot_scatter_chart(df6, "Rajaji Nagar")
#plot_scatter_chart(df6, "Hebbal")

df8 = df7[df7.bath<df7.bhk+2]      # A 2 BHK with 4 or more bathrooms is likely a luxury or incorrect entry, so filtered from df7 and new dataframe is df8
print(df8.shape)

df9 = df8.drop(['size', 'price_per_sqft'])
