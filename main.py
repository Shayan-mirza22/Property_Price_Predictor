import numpy as np
import pandas as pd
import matplotlib as mp

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

print((df4.location.unique()))     # Printing the unique location names, if they are many in number, this can be a huge problem!

