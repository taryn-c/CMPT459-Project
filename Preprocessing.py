#!/usr/bin/env python
# coding: utf-8
# %%

# # Phase 1: Exploratory data analysis and data pre-processing
#
# ### Taryn Chung, ZeYu Zhu, Kearro Chow

# %%


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from scipy import nanmean
import math
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
color = sns.color_palette()


# %%


train_df = pd.read_json("train.json")


# ## Hour-wise listing trend

# %%


train_df["created"] = pd.to_datetime(train_df["created"])
train_df["hour_created"] = train_df["created"].dt.hour


# ## Handling missing values

# ### Created


# %%


whichrow = 0
count = 0
for row in train_df['display_address']:
    if any(c.isalpha() for c in row) == False:
        train_df.loc[train_df.index[whichrow], 'display_address'] = np.nan
        count+=1
        whichrow+=1
    else:
        whichrow += 1
##print ("number of missing values: ", count)


# Since there are relatively few missing values and an address is important to have, we removed these values by changing them to NAN and dropping every row with a NAN.

# ### Street Address

# %%


whichrow = 0
count = 0
for row in train_df['street_address']:
    if any(c.isalpha() for c in row) == False :
        train_df.loc[train_df.index[whichrow], 'street_address'] = np.nan
        count+=1
        whichrow+=1
    else:
        whichrow += 1
##print ("number of missing values: ", count)


# Since there are relatively few missing values and an address is important to have, we removed these values by changing them to NAN and dropping every row with a NAN.

# ### Latitude

# %%


whichrow = 0
count = 0
for row in train_df['latitude']:
    if row==0:
        train_df.loc[train_df.index[whichrow], 'latitude'] = np.nan
        count+=1
        whichrow+=1
    else:
        whichrow += 1

# ### Longitude

# %%


whichrow = 0
count = 0
for row in train_df['longitude']:
    if row==0:
        train_df.loc[train_df.index[whichrow], 'longitude'] = np.nan
        count+=1
        whichrow+=1
    else:
        whichrow += 1


#remove selected rows with few missing values
train_df = train_df.dropna()


# ## Handling outliers
#
# ### Prices

# %%


q25, q75 = np.percentile(train_df['price'].values, 25), np.percentile(train_df['price'].values, 75)
iqr = q75-q25
cutoff = iqr * 1.5
lower, upper = q25 - cutoff, q75 + cutoff
outliers = [x for x in train_df['price'].values if x > upper or x < lower]
#print ("number of mild outliers:", len(outliers))
##print(lower,upper)


# #### After Handling

# %%

whichrow = 0
for row in train_df['price']:
    if row>upper or row<lower:
        train_df.loc[train_df.index[whichrow], 'price'] = np.nan
        whichrow+=1
    else:
        whichrow += 1

mean_val =round(nanmean(train_df['price']),2)

whichrow = 0
for row in train_df['price']:
    if math.isnan(row):
        train_df.loc[train_df.index[whichrow],'price'] = mean_val
        whichrow+=1
    else:
        whichrow+=1


# We found the Tukey inner and outer fences to find the mild and extreme outliers. Extreme fences included a cutoff that left listings with a negative price range. We decided to only remove the mild outliers instead.  We replaced them with the mean value 3272.00 of so we have more data points to train with. The upper fence cutoff was 6,500 and the lower cutoff was 100. This dealt with outliers such as listings prices at 4,490,000, 43, and negative numbers, which are unrealistic. The new minimum price is 401 and the maximum is now 6,500.

# ### Latitude

# #### Before Handling

# %%


q25, q75 = np.percentile(train_df['latitude'].values, 25), np.percentile(train_df['latitude'].values, 75)
iqr = q75-q25
cutoff = iqr * 3
lower, upper = q25 - cutoff, q75 + cutoff
outliers = [x for x in train_df['latitude'].values if x > upper or x < lower]
#print ("number of extreme outliers:", len(outliers))
##print(lower, upper)


# #### After Handling

# %%


whichrow = 0
for row in train_df['latitude']:
    if row>upper or row<lower:
        train_df.loc[train_df.index[whichrow], 'latitude'] = np.nan
        whichrow+=1
    else:
        whichrow += 1

mean_val = nanmean(train_df['latitude'])
##print(mean_val)
whichrow = 0
for row in train_df['latitude']:
    if math.isnan(row):
        train_df.loc[train_df.index[whichrow],'latitude'] = mean_val
        whichrow+=1
    else:
        whichrow+=1



# We found the Tukey inner and outer fences to find the mild and extreme outliers. The inner fences still included listings that were within the NYC bounds, accoding to Google Maps. So we decided to only deal with the extreme outliers that were a little further out than NYC vicinity. We replaced with the mean value  of 40.7513098874893 so we have more data points to train with. The fence cutoffs we used were 40.5903 and 40.912299999999995. This dealt with outliers such as listings located in LA because we wanted to keep the listings within NYC. The new maximum and minimum longitude and latitude are now 40.9121 and 40.5904, respectively.

# ### Longitude

# %%


q25, q75 = np.percentile(train_df['longitude'].values, 25), np.percentile(train_df['longitude'].values, 75)
iqr = q75-q25
cutoff = iqr * 1.5
lower, upper = q25 - cutoff, q75 + cutoff
outliers = [x for x in train_df['longitude'].values if x > upper or x < lower]
#print ("number of mild outliers:", len(outliers))
##print(lower, upper)


# #### After Handling

# %%


whichrow = 0
for row in train_df['longitude']:
    if row>upper or row<lower:
        train_df.loc[train_df.index[whichrow], 'longitude'] = np.nan
        whichrow+=1
    else:
        whichrow += 1

mean_val = nanmean(train_df['longitude'])
#print(mean_val)

whichrow = 0
for row in train_df['longitude']:
    if math.isnan(row):
        train_df.loc[train_df.index[whichrow],'longitude'] = mean_val
        whichrow+=1
    else:
        whichrow+=1



# We found the Tukey inner and outer fences to find the mild and extreme outliers. The outer fences still included listings that were within the NYC borough bounds, accoding to Google Maps. The very bottom tip of New York the island is around -74.01 and -73.88 is in the Bronx. So we decided to only deal with the mild outliers that encompassed this. We replaced with the mean value  of 40.7513098874893 so we have more data points to train with. The fence cutoffs we used were -74.04704999999998 and -73.89945000000003. This dealt with outliers such as listings located in LA because we wanted to keep the listings within NYC. The new minimum and maximum longitude and latitude are now -74.0454 and -73.8995, respectively.



# ## Text feature extraction

# Here we expore the text in two meaningful features of the listings: features and description. We use the TFIDF to tokenize the words found in each feature. Common stop words were excluded and we defined each word 2+ characters because there were some words found such as 'xx' and other random strings. Since the features are actually lists in this column, we concatenated them to string before tokenize. After tokenization, we vectorize each feature into new columns.

# %%


# vectorize each description into a new column by frequency
# common stop words are excluded like a, the, in, etc.
# matches tokens of length 2+

vectorizer = TfidfVectorizer(analyzer='word', stop_words='english', token_pattern=r'^[a-zA-Z][a-zA-Z]+')
vectorizer.fit(train_df['description'].values);
train_df['desc_vect'] = train_df['description'].apply(lambda x: vectorizer.transform([x]))
#vectorizer.vocabulary_  #shows all tokens


# %%


# convert features list to string
train_df['features'] = train_df['features'].apply(lambda x: ' '.join(x))


# %%


#features
vectorizer = TfidfVectorizer(analyzer='word', stop_words='english', token_pattern=r'^[a-zA-Z][a-zA-Z]+')
vectorizer.fit(train_df['features'].values);
train_df['feat_vect'] = train_df['features'].apply(lambda x: vectorizer.transform([x]))
# vectorizer.vocabulary_  #shows all tokens
