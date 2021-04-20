# Step 3: Select and train a model 

## End-to-end Machine Learning Project

This project is obtained from Aurelien Geron's ML book Chapter 2. The aim to predict median house values in Californian districts, given a number of features from these districts.

Main steps we will go through: 
1. Formulate the problem
2. Get the data
3. Discover and visualize data / Data exploration to gain insight 
4. Prep data for ML algorithm testing 
5. Select model and train it
6. Fine-tuning the model 

https://github.com/pgg1610/handson-ml/blob/master/02_end_to_end_machine_learning_project.ipynb


### Step 1: Formulate the problem 

Prediction of district's median housing price given all other metrics. A supervised learning task is where we are given 'labelled' data for training purpose. Regression model to predict a continuous variable i.e. district median housing price. Given multiple features, this is a multi-class regression type problem. Univariate regression since a single output is estimated. 

### Step 2: Get the data


```python
import pandas as pd
import numpy as np 
%config InlineBackend.figure_format = 'retina'
import matplotlib as mpl
import matplotlib.pyplot as plt

#Load the dataset 
housing = pd.read_csv('housing.csv')

#Or defining a function to download the latest copy of the dataset is generally a good practice if the dataset changes 
'''
import os
import tarfile
from six.moves import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
'''

```




    '\nimport os\nimport tarfile\nfrom six.moves import urllib\n\nDOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"\nHOUSING_PATH = os.path.join("datasets", "housing")\nHOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"\n\ndef fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):\n    if not os.path.isdir(housing_path):\n        os.makedirs(housing_path)\n    tgz_path = os.path.join(housing_path, "housing.tgz")\n    urllib.request.urlretrieve(housing_url, tgz_path)\n    housing_tgz = tarfile.open(tgz_path)\n    housing_tgz.extractall(path=housing_path)\n    housing_tgz.close()\n'



```python
housing.sample(7)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
      <th>ocean_proximity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3471</th>
      <td>-118.47</td>
      <td>34.29</td>
      <td>18.0</td>
      <td>4256.0</td>
      <td>987.0</td>
      <td>3401.0</td>
      <td>955.0</td>
      <td>4.2935</td>
      <td>190000.0</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>15628</th>
      <td>-122.40</td>
      <td>37.80</td>
      <td>52.0</td>
      <td>2094.0</td>
      <td>568.0</td>
      <td>920.0</td>
      <td>503.0</td>
      <td>4.2015</td>
      <td>412500.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>17985</th>
      <td>-121.95</td>
      <td>37.30</td>
      <td>25.0</td>
      <td>5641.0</td>
      <td>1517.0</td>
      <td>3786.0</td>
      <td>1410.0</td>
      <td>3.3958</td>
      <td>267500.0</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>16242</th>
      <td>-121.30</td>
      <td>37.99</td>
      <td>38.0</td>
      <td>2375.0</td>
      <td>494.0</td>
      <td>1167.0</td>
      <td>471.0</td>
      <td>2.6673</td>
      <td>87500.0</td>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>13176</th>
      <td>-117.75</td>
      <td>34.01</td>
      <td>4.0</td>
      <td>22128.0</td>
      <td>3522.0</td>
      <td>10450.0</td>
      <td>3258.0</td>
      <td>6.1287</td>
      <td>289600.0</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>5507</th>
      <td>-118.43</td>
      <td>33.99</td>
      <td>42.0</td>
      <td>2558.0</td>
      <td>558.0</td>
      <td>1328.0</td>
      <td>541.0</td>
      <td>4.8750</td>
      <td>301300.0</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>18611</th>
      <td>-122.05</td>
      <td>37.11</td>
      <td>39.0</td>
      <td>1065.0</td>
      <td>248.0</td>
      <td>497.0</td>
      <td>208.0</td>
      <td>4.5972</td>
      <td>146300.0</td>
      <td>NEAR OCEAN</td>
    </tr>
  </tbody>
</table>
</div>



Each row presents one district. Each of these districts has 10 attributes (features). 

```python
housing.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 20640 entries, 0 to 20639
    Data columns (total 10 columns):
     #   Column              Non-Null Count  Dtype  
    ---  ------              --------------  -----  
     0   longitude           20640 non-null  float64
     1   latitude            20640 non-null  float64
     2   housing_median_age  20640 non-null  float64
     3   total_rooms         20640 non-null  float64
     4   total_bedrooms      20433 non-null  float64
     5   population          20640 non-null  float64
     6   households          20640 non-null  float64
     7   median_income       20640 non-null  float64
     8   median_house_value  20640 non-null  float64
     9   ocean_proximity     20640 non-null  object 
    dtypes: float64(9), object(1)
    memory usage: 1.6+ MB


One thing to notice in this dataset is the number of total_bedroom entries is different from other entries. This suggests there are some missing entries or null in the dataset. 

```python
#To show the null elements (if any in the total_bedroom entries)
housing[housing.total_bedrooms.isnull()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
      <th>ocean_proximity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>290</th>
      <td>-122.16</td>
      <td>37.77</td>
      <td>47.0</td>
      <td>1256.0</td>
      <td>NaN</td>
      <td>570.0</td>
      <td>218.0</td>
      <td>4.3750</td>
      <td>161900.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>341</th>
      <td>-122.17</td>
      <td>37.75</td>
      <td>38.0</td>
      <td>992.0</td>
      <td>NaN</td>
      <td>732.0</td>
      <td>259.0</td>
      <td>1.6196</td>
      <td>85100.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>538</th>
      <td>-122.28</td>
      <td>37.78</td>
      <td>29.0</td>
      <td>5154.0</td>
      <td>NaN</td>
      <td>3741.0</td>
      <td>1273.0</td>
      <td>2.5762</td>
      <td>173400.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>563</th>
      <td>-122.24</td>
      <td>37.75</td>
      <td>45.0</td>
      <td>891.0</td>
      <td>NaN</td>
      <td>384.0</td>
      <td>146.0</td>
      <td>4.9489</td>
      <td>247100.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>696</th>
      <td>-122.10</td>
      <td>37.69</td>
      <td>41.0</td>
      <td>746.0</td>
      <td>NaN</td>
      <td>387.0</td>
      <td>161.0</td>
      <td>3.9063</td>
      <td>178400.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>20267</th>
      <td>-119.19</td>
      <td>34.20</td>
      <td>18.0</td>
      <td>3620.0</td>
      <td>NaN</td>
      <td>3171.0</td>
      <td>779.0</td>
      <td>3.3409</td>
      <td>220500.0</td>
      <td>NEAR OCEAN</td>
    </tr>
    <tr>
      <th>20268</th>
      <td>-119.18</td>
      <td>34.19</td>
      <td>19.0</td>
      <td>2393.0</td>
      <td>NaN</td>
      <td>1938.0</td>
      <td>762.0</td>
      <td>1.6953</td>
      <td>167400.0</td>
      <td>NEAR OCEAN</td>
    </tr>
    <tr>
      <th>20372</th>
      <td>-118.88</td>
      <td>34.17</td>
      <td>15.0</td>
      <td>4260.0</td>
      <td>NaN</td>
      <td>1701.0</td>
      <td>669.0</td>
      <td>5.1033</td>
      <td>410700.0</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>20460</th>
      <td>-118.75</td>
      <td>34.29</td>
      <td>17.0</td>
      <td>5512.0</td>
      <td>NaN</td>
      <td>2734.0</td>
      <td>814.0</td>
      <td>6.6073</td>
      <td>258100.0</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>20484</th>
      <td>-118.72</td>
      <td>34.28</td>
      <td>17.0</td>
      <td>3051.0</td>
      <td>NaN</td>
      <td>1705.0</td>
      <td>495.0</td>
      <td>5.7376</td>
      <td>218600.0</td>
      <td>&lt;1H OCEAN</td>
    </tr>
  </tbody>
</table>
<p>207 rows × 10 columns</p>
</div>



For categorical entries (here, ocean_proximity entries) we can find out the entries and their number using the value_counts(). We can do this for any entry we wish but makes more sense for categorical entries. 

```python
housing["ocean_proximity"].value_counts()
```




    <1H OCEAN     9136
    INLAND        6551
    NEAR OCEAN    2658
    NEAR BAY      2290
    ISLAND           5
    Name: ocean_proximity, dtype: int64



```python
housing.describe().round(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>20640.00</td>
      <td>20640.00</td>
      <td>20640.00</td>
      <td>20640.00</td>
      <td>20433.00</td>
      <td>20640.00</td>
      <td>20640.00</td>
      <td>20640.00</td>
      <td>20640.00</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-119.57</td>
      <td>35.63</td>
      <td>28.64</td>
      <td>2635.76</td>
      <td>537.87</td>
      <td>1425.48</td>
      <td>499.54</td>
      <td>3.87</td>
      <td>206855.82</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.00</td>
      <td>2.14</td>
      <td>12.59</td>
      <td>2181.62</td>
      <td>421.39</td>
      <td>1132.46</td>
      <td>382.33</td>
      <td>1.90</td>
      <td>115395.62</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-124.35</td>
      <td>32.54</td>
      <td>1.00</td>
      <td>2.00</td>
      <td>1.00</td>
      <td>3.00</td>
      <td>1.00</td>
      <td>0.50</td>
      <td>14999.00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-121.80</td>
      <td>33.93</td>
      <td>18.00</td>
      <td>1447.75</td>
      <td>296.00</td>
      <td>787.00</td>
      <td>280.00</td>
      <td>2.56</td>
      <td>119600.00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-118.49</td>
      <td>34.26</td>
      <td>29.00</td>
      <td>2127.00</td>
      <td>435.00</td>
      <td>1166.00</td>
      <td>409.00</td>
      <td>3.53</td>
      <td>179700.00</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>-118.01</td>
      <td>37.71</td>
      <td>37.00</td>
      <td>3148.00</td>
      <td>647.00</td>
      <td>1725.00</td>
      <td>605.00</td>
      <td>4.74</td>
      <td>264725.00</td>
    </tr>
    <tr>
      <th>max</th>
      <td>-114.31</td>
      <td>41.95</td>
      <td>52.00</td>
      <td>39320.00</td>
      <td>6445.00</td>
      <td>35682.00</td>
      <td>6082.00</td>
      <td>15.00</td>
      <td>500001.00</td>
    </tr>
  </tbody>
</table>
</div>



Describe is powerful subroutine since that allows us to check the stat summary of numerical attributes

The 25%-50%-75% entries for each column show corresponding percentiles. It indicates the value below which a given percentage of observations in a group of observations fall. For example, 25% of observation have median income below 2.56, 50% observations have median income below 3.53, and 75% observations have median income below 4.74. 25% --> 1st Quartile, 50% --> Median, 75% --> 3rd Quartile

```python
housing.hist(bins=50,figsize=(20,20))
```




    array([[<AxesSubplot:title={'center':'longitude'}>,
            <AxesSubplot:title={'center':'latitude'}>,
            <AxesSubplot:title={'center':'housing_median_age'}>],
           [<AxesSubplot:title={'center':'total_rooms'}>,
            <AxesSubplot:title={'center':'total_bedrooms'}>,
            <AxesSubplot:title={'center':'population'}>],
           [<AxesSubplot:title={'center':'households'}>,
            <AxesSubplot:title={'center':'median_income'}>,
            <AxesSubplot:title={'center':'median_house_value'}>]],
          dtype=object)




![png](end2endML_housing_files/output_12_1.png)


Few observations from the Histogram plots, again remember each row is an entry for an ENTIRE district: 
NOTICE: From the dataset's source disclaimer: The housing_median_value, housing_median_age, median_income_value are capped at an arbitrary value.
1. From latitute and longitude plots there seems to be lots of district in four particular locations (34,37 -- latitude) and (-120,-118 -- longitude). We cannot comment on the exact location but only one on these pairs giving most data. 
2. We see a tighter distribution for total_rooms, total_bedrooms, and population but spread for house_value and an intresting spike at its end. 
3. Small spike at the end of median_income plot suggests presence of small group of affluent families but interestingly that spike does not correlate with the spike in the house_value (More high-end property entries than more "rich" people in a district) 

Finally, the dataset is tail-heavy that is they extend further to the right from the median which might make modeling using some ML algorithm a bit chanellenging. Few entries should be scaled such that the distribution is more normal. 

### Create a test-set

This ensures that this is the data on which training, testing occurs and we do not try overfitting to account for all the variance in the data. Typical 20% of data-points are randomly chosen. 

```python
def split_train_test(data,test_ratio):
    shuffled_indices=np.random.permutation(len(data))
    test_set_size=int(len(data)*test_ratio)
    test_indices=shuffled_indices[:test_set_size]
    train_indices=shuffled_indices[test_set_size:]
    return(data.iloc[train_indices],data.iloc[test_indices])

```

```python
#To ensure we get similar results at each run -- if not initiated every successive will give more random 
#shuffled indices risking the possibility of the algo seeing the entire dataset! 

np.random.seed(42)

#Random seed set to 42 for no particular reason 
#but just cause its the answer to the Ultimate Question of Life, The Universe, and Everything

train_set, test_set = split_train_test(housing, 0.2)
print(len(train_set), "train +", len(test_set), "test")
```

    16512 train + 4128 test


Better way is to have an instance identifier (like id) for each entry to distingusih each entry and see if its sampled or not. 

```python
from zlib import crc32

def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]
```

The dataset currently doesnt have inherent id. We could use the row index as id. Or we could use an ad-hoc unique identifier as an interim id. 

```python
#HOUSING DATA WITH ID as ROW INDEX
housing_with_id = housing.reset_index()   # adds an `index` column
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

#HOUSING DATA WITH ID AS COMBO OF LAT AND LONG. 
housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")

#SCIKIT-LEARN IMPLEMENTATION
#from sklearn.model_selection import train_test_split
#train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
```

```python
test_set.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
      <th>ocean_proximity</th>
      <th>id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>59</th>
      <td>59</td>
      <td>-122.29</td>
      <td>37.82</td>
      <td>2.0</td>
      <td>158.0</td>
      <td>43.0</td>
      <td>94.0</td>
      <td>57.0</td>
      <td>2.5625</td>
      <td>60000.0</td>
      <td>NEAR BAY</td>
      <td>-122252.18</td>
    </tr>
    <tr>
      <th>60</th>
      <td>60</td>
      <td>-122.29</td>
      <td>37.83</td>
      <td>52.0</td>
      <td>1121.0</td>
      <td>211.0</td>
      <td>554.0</td>
      <td>187.0</td>
      <td>3.3929</td>
      <td>75700.0</td>
      <td>NEAR BAY</td>
      <td>-122252.17</td>
    </tr>
    <tr>
      <th>61</th>
      <td>61</td>
      <td>-122.29</td>
      <td>37.82</td>
      <td>49.0</td>
      <td>135.0</td>
      <td>29.0</td>
      <td>86.0</td>
      <td>23.0</td>
      <td>6.1183</td>
      <td>75000.0</td>
      <td>NEAR BAY</td>
      <td>-122252.18</td>
    </tr>
    <tr>
      <th>62</th>
      <td>62</td>
      <td>-122.29</td>
      <td>37.81</td>
      <td>50.0</td>
      <td>760.0</td>
      <td>190.0</td>
      <td>377.0</td>
      <td>122.0</td>
      <td>0.9011</td>
      <td>86100.0</td>
      <td>NEAR BAY</td>
      <td>-122252.19</td>
    </tr>
    <tr>
      <th>67</th>
      <td>67</td>
      <td>-122.29</td>
      <td>37.80</td>
      <td>52.0</td>
      <td>1027.0</td>
      <td>244.0</td>
      <td>492.0</td>
      <td>147.0</td>
      <td>2.6094</td>
      <td>81300.0</td>
      <td>NEAR BAY</td>
      <td>-122252.20</td>
    </tr>
  </tbody>
</table>
</div>



However the sampling we have considered here or the one used in Scikit-learn is random sampling by default. This is fine for large dataset however for smaller dataset it is utmost important that the sampled data is representative of the main population data or else we will introduce sampling bias. 

This is an important bias that could be introduced without prior knowledge and could be overlooked at multiple occassion leading to wrong conclusions. To ensure the sampled dataset is representative of the population set we  use stratified sampling (pseudo-random sampling). To make the stratified sampling tractable we first divide the main data into multiple 'stratas' based on an variable which we feel is an feature that should be replicated in our test set. The sample is divided into strata and right number of instances are chosen from each strata. We must not have too many stratas and the each strate must have appropriate number of instances.

For the case of property pricing in the district, median_income variable is chosen as the variable whose distribution in the main population and the randomly chosen test sample is same. This attribute is an important attribute to predict the final median housing price. So we can think of converting the continuous variable of median_variable into categorical variable -- that is stratas.

#### Stratified sampling using median income

```python
housing["median_income"].hist()
```




    <AxesSubplot:>




![png](end2endML_housing_files/output_22_1.png)


From the median_income histogram it is seen that most of the entries are clustered in the range of 2-5 (arbitrary units). We can then use this information to make stratas around these instances. Cut routine in the pandas is used for this purpose. This function is also useful for going from a continuous variable to a categorical variable. For example, cut could convert ages to groups of age ranges.

```python
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf], #bins around 2-5 income bracket
                               labels=[1, 2, 3, 4, 5])
housing["income_cat"].value_counts()
```




    3    7236
    2    6581
    4    3639
    5    2362
    1     822
    Name: income_cat, dtype: int64



```python
housing["income_cat"].hist()
```




    <AxesSubplot:>




![png](end2endML_housing_files/output_25_1.png)


Now with the population categorised into various median income groups we can use stratified sampling routine (as implemented in scikit-learn) to make our test-set. As an additional proof let's compare this to a randomly sampled test_case. We will redo the random sampling we did prviously but with the new population with categorised median_income. 

```python
#Stratified sampling from scikit-learn 
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

#Using random sampling
rand_train_set, rand_test_set = train_test_split(housing, test_size=0.2, random_state=42)
```

Let's check the distribution of the income_cat variable in the strat_test, random_test, and the main sample. 

```python
housing["income_cat"].value_counts()/len(housing["income_cat"])
```




    3    0.350581
    2    0.318847
    4    0.176308
    5    0.114438
    1    0.039826
    Name: income_cat, dtype: float64



```python
rand_test_set["income_cat"].value_counts()/len(rand_test_set["income_cat"])
```




    3    0.358527
    2    0.324370
    4    0.167393
    5    0.109496
    1    0.040213
    Name: income_cat, dtype: float64



```python
strat_test_set["income_cat"].value_counts()/len(strat_test_set["income_cat"])
```




    3    0.350533
    2    0.318798
    4    0.176357
    5    0.114583
    1    0.039729
    Name: income_cat, dtype: float64



```python
def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)

compare_props = pd.DataFrame({
    "Overall": income_cat_proportions(housing),
    "Stratified": income_cat_proportions(strat_test_set),
    "Random": income_cat_proportions(rand_test_set),
}).sort_index()
compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100
```

```python
compare_props
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Overall</th>
      <th>Stratified</th>
      <th>Random</th>
      <th>Rand. %error</th>
      <th>Strat. %error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.039826</td>
      <td>0.039729</td>
      <td>0.040213</td>
      <td>0.973236</td>
      <td>-0.243309</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.318847</td>
      <td>0.318798</td>
      <td>0.324370</td>
      <td>1.732260</td>
      <td>-0.015195</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.350581</td>
      <td>0.350533</td>
      <td>0.358527</td>
      <td>2.266446</td>
      <td>-0.013820</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.176308</td>
      <td>0.176357</td>
      <td>0.167393</td>
      <td>-5.056334</td>
      <td>0.027480</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.114438</td>
      <td>0.114583</td>
      <td>0.109496</td>
      <td>-4.318374</td>
      <td>0.127011</td>
    </tr>
  </tbody>
</table>
</div>



Now, we can remove the income_cat column

```python
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
```

### Preliminary visualization of the data
Let's now dive a bit deeper into the data visualization and analysis. Before we do so, copy the strat_train_set as that would be the data-set we would be playing around and make sure the main data-set is not touched. 

```python
housing=strat_train_set.copy()
housing.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 16512 entries, 17606 to 15775
    Data columns (total 10 columns):
     #   Column              Non-Null Count  Dtype  
    ---  ------              --------------  -----  
     0   longitude           16512 non-null  float64
     1   latitude            16512 non-null  float64
     2   housing_median_age  16512 non-null  float64
     3   total_rooms         16512 non-null  float64
     4   total_bedrooms      16354 non-null  float64
     5   population          16512 non-null  float64
     6   households          16512 non-null  float64
     7   median_income       16512 non-null  float64
     8   median_house_value  16512 non-null  float64
     9   ocean_proximity     16512 non-null  object 
    dtypes: float64(9), object(1)
    memory usage: 1.4+ MB


#### Geographical visualization

Let's now plot the data entries in the housing data as per the latitude and longitude. 

```python
housing.plot(kind='scatter',x='longitude',y='latitude')
```




    <AxesSubplot:xlabel='longitude', ylabel='latitude'>




![png](end2endML_housing_files/output_39_1.png)


This look's like California however, we cannot infer anything more out of this. Let's play around a little bit more...

1. Playing with the alpha value in the plotting routine allows us to see the frequency of THAT datapoint in the plot

```python
housing.plot(kind='scatter',x='longitude',y='latitude',alpha=0.1)
```




    <AxesSubplot:xlabel='longitude', ylabel='latitude'>




![png](end2endML_housing_files/output_41_1.png)


From here, we can see the high density of listings in the Bay area and LA also around Sacramento and Fresco. 
> ## [TODO] Overlay actual map of California on this plot

```python
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    s=housing["population"]/100, label="population", figsize=(15,10),
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
    sharex=False)
plt.legend()
```




    <matplotlib.legend.Legend at 0x7f9f6a5b85c0>




![png](end2endML_housing_files/output_43_1.png)


This is more interesting! We have now plotted the data with more information. Each data-point has two additional set of info apart of frequency of occurence. First being the color of the point is the median_house_value entry (option c). The radius of the data-point is the population of that district (option s). It can be seen that the housing prices are very much related to the location. The ones closer to the bay area are more expensive but need not be densely populated.

#### Looking for simple correlations 
In addition to looking at the plot of housing price, we can check for simpler correaltions. Pearson's correlation matrix is something which is in-built in pandas and can be directly used. It checks for correlation between every pair of feature provided in the data-set. It estimates the covariance of the two features and estimates whether the correlation is inverse, direct, or none. 
#### [Formula of Pearson's correlation]

```python
corr_matrix=housing.corr()
corr_matrix.style.background_gradient(cmap='coolwarm').set_precision(2)
```




<style  type="text/css" >
#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row0_col0,#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row1_col1,#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row2_col2,#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row3_col3,#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row4_col4,#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row5_col5,#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row6_col6,#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row7_col7,#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row8_col8{
            background-color:  #b40426;
            color:  #f1f1f1;
        }#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row0_col1,#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row1_col0,#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row1_col8,#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row2_col3,#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row2_col4,#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row2_col5,#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row2_col6,#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row2_col7,#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row3_col2{
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row0_col2,#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row1_col4{
            background-color:  #779af7;
            color:  #000000;
        }#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row0_col3,#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row0_col4{
            background-color:  #9fbfff;
            color:  #000000;
        }#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row0_col5{
            background-color:  #a3c2fe;
            color:  #000000;
        }#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row0_col6,#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row8_col4,#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row8_col6{
            background-color:  #98b9ff;
            color:  #000000;
        }#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row0_col7,#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row0_col8{
            background-color:  #5470de;
            color:  #000000;
        }#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row1_col2{
            background-color:  #94b6ff;
            color:  #000000;
        }#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row1_col3{
            background-color:  #89acfd;
            color:  #000000;
        }#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row1_col5{
            background-color:  #6788ee;
            color:  #000000;
        }#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row1_col6{
            background-color:  #7295f4;
            color:  #000000;
        }#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row1_col7{
            background-color:  #445acc;
            color:  #f1f1f1;
        }#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row2_col0{
            background-color:  #c7d7f0;
            color:  #000000;
        }#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row2_col1{
            background-color:  #d8dce2;
            color:  #000000;
        }#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row2_col8{
            background-color:  #84a7fc;
            color:  #000000;
        }#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row3_col0{
            background-color:  #dedcdb;
            color:  #000000;
        }#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row3_col1{
            background-color:  #d2dbe8;
            color:  #000000;
        }#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row3_col4,#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row4_col3{
            background-color:  #c73635;
            color:  #f1f1f1;
        }#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row3_col5{
            background-color:  #d95847;
            color:  #000000;
        }#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row3_col6,#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row6_col3{
            background-color:  #ca3b37;
            color:  #f1f1f1;
        }#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row3_col7{
            background-color:  #97b8ff;
            color:  #000000;
        }#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row3_col8,#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row7_col6{
            background-color:  #8badfd;
            color:  #000000;
        }#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row4_col0{
            background-color:  #e2dad5;
            color:  #000000;
        }#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row4_col1{
            background-color:  #cdd9ec;
            color:  #000000;
        }#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row4_col2{
            background-color:  #4358cb;
            color:  #f1f1f1;
        }#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row4_col5{
            background-color:  #d55042;
            color:  #000000;
        }#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row4_col6,#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row6_col4{
            background-color:  #b8122a;
            color:  #f1f1f1;
        }#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row4_col7{
            background-color:  #5673e0;
            color:  #000000;
        }#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row4_col8{
            background-color:  #6f92f3;
            color:  #000000;
        }#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row5_col0{
            background-color:  #e6d7cf;
            color:  #000000;
        }#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row5_col1{
            background-color:  #c6d6f1;
            color:  #000000;
        }#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row5_col2{
            background-color:  #4961d2;
            color:  #f1f1f1;
        }#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row5_col3{
            background-color:  #d85646;
            color:  #000000;
        }#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row5_col4{
            background-color:  #d44e41;
            color:  #000000;
        }#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row5_col6,#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row6_col5{
            background-color:  #cd423b;
            color:  #f1f1f1;
        }#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row5_col7{
            background-color:  #5a78e4;
            color:  #000000;
        }#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row5_col8{
            background-color:  #5977e3;
            color:  #000000;
        }#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row6_col0{
            background-color:  #e0dbd8;
            color:  #000000;
        }#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row6_col1,#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row7_col1{
            background-color:  #ccd9ed;
            color:  #000000;
        }#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row6_col2{
            background-color:  #465ecf;
            color:  #f1f1f1;
        }#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row6_col7{
            background-color:  #5d7ce6;
            color:  #000000;
        }#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row6_col8{
            background-color:  #7597f6;
            color:  #000000;
        }#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row7_col0{
            background-color:  #d5dbe5;
            color:  #000000;
        }#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row7_col2{
            background-color:  #7699f6;
            color:  #000000;
        }#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row7_col3{
            background-color:  #c4d5f3;
            color:  #000000;
        }#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row7_col4{
            background-color:  #88abfd;
            color:  #000000;
        }#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row7_col5{
            background-color:  #86a9fc;
            color:  #000000;
        }#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row7_col8{
            background-color:  #f6a385;
            color:  #000000;
        }#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row8_col0{
            background-color:  #d1dae9;
            color:  #000000;
        }#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row8_col1{
            background-color:  #c1d4f4;
            color:  #000000;
        }#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row8_col2{
            background-color:  #afcafc;
            color:  #000000;
        }#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row8_col3{
            background-color:  #b5cdfa;
            color:  #000000;
        }#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row8_col5{
            background-color:  #7ea1fa;
            color:  #000000;
        }#T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row8_col7{
            background-color:  #f7a688;
            color:  #000000;
        }</style><table id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >longitude</th>        <th class="col_heading level0 col1" >latitude</th>        <th class="col_heading level0 col2" >housing_median_age</th>        <th class="col_heading level0 col3" >total_rooms</th>        <th class="col_heading level0 col4" >total_bedrooms</th>        <th class="col_heading level0 col5" >population</th>        <th class="col_heading level0 col6" >households</th>        <th class="col_heading level0 col7" >median_income</th>        <th class="col_heading level0 col8" >median_house_value</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6level0_row0" class="row_heading level0 row0" >longitude</th>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row0_col0" class="data row0 col0" >1.00</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row0_col1" class="data row0 col1" >-0.92</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row0_col2" class="data row0 col2" >-0.11</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row0_col3" class="data row0 col3" >0.05</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row0_col4" class="data row0 col4" >0.08</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row0_col5" class="data row0 col5" >0.11</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row0_col6" class="data row0 col6" >0.06</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row0_col7" class="data row0 col7" >-0.02</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row0_col8" class="data row0 col8" >-0.05</td>
            </tr>
            <tr>
                        <th id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6level0_row1" class="row_heading level0 row1" >latitude</th>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row1_col0" class="data row1 col0" >-0.92</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row1_col1" class="data row1 col1" >1.00</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row1_col2" class="data row1 col2" >0.01</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row1_col3" class="data row1 col3" >-0.04</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row1_col4" class="data row1 col4" >-0.07</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row1_col5" class="data row1 col5" >-0.12</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row1_col6" class="data row1 col6" >-0.08</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row1_col7" class="data row1 col7" >-0.08</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row1_col8" class="data row1 col8" >-0.14</td>
            </tr>
            <tr>
                        <th id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6level0_row2" class="row_heading level0 row2" >housing_median_age</th>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row2_col0" class="data row2 col0" >-0.11</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row2_col1" class="data row2 col1" >0.01</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row2_col2" class="data row2 col2" >1.00</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row2_col3" class="data row2 col3" >-0.36</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row2_col4" class="data row2 col4" >-0.33</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row2_col5" class="data row2 col5" >-0.30</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row2_col6" class="data row2 col6" >-0.31</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row2_col7" class="data row2 col7" >-0.11</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row2_col8" class="data row2 col8" >0.11</td>
            </tr>
            <tr>
                        <th id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6level0_row3" class="row_heading level0 row3" >total_rooms</th>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row3_col0" class="data row3 col0" >0.05</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row3_col1" class="data row3 col1" >-0.04</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row3_col2" class="data row3 col2" >-0.36</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row3_col3" class="data row3 col3" >1.00</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row3_col4" class="data row3 col4" >0.93</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row3_col5" class="data row3 col5" >0.86</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row3_col6" class="data row3 col6" >0.92</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row3_col7" class="data row3 col7" >0.20</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row3_col8" class="data row3 col8" >0.14</td>
            </tr>
            <tr>
                        <th id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6level0_row4" class="row_heading level0 row4" >total_bedrooms</th>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row4_col0" class="data row4 col0" >0.08</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row4_col1" class="data row4 col1" >-0.07</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row4_col2" class="data row4 col2" >-0.33</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row4_col3" class="data row4 col3" >0.93</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row4_col4" class="data row4 col4" >1.00</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row4_col5" class="data row4 col5" >0.88</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row4_col6" class="data row4 col6" >0.98</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row4_col7" class="data row4 col7" >-0.01</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row4_col8" class="data row4 col8" >0.05</td>
            </tr>
            <tr>
                        <th id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6level0_row5" class="row_heading level0 row5" >population</th>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row5_col0" class="data row5 col0" >0.11</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row5_col1" class="data row5 col1" >-0.12</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row5_col2" class="data row5 col2" >-0.30</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row5_col3" class="data row5 col3" >0.86</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row5_col4" class="data row5 col4" >0.88</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row5_col5" class="data row5 col5" >1.00</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row5_col6" class="data row5 col6" >0.90</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row5_col7" class="data row5 col7" >0.00</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row5_col8" class="data row5 col8" >-0.03</td>
            </tr>
            <tr>
                        <th id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6level0_row6" class="row_heading level0 row6" >households</th>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row6_col0" class="data row6 col0" >0.06</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row6_col1" class="data row6 col1" >-0.08</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row6_col2" class="data row6 col2" >-0.31</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row6_col3" class="data row6 col3" >0.92</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row6_col4" class="data row6 col4" >0.98</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row6_col5" class="data row6 col5" >0.90</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row6_col6" class="data row6 col6" >1.00</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row6_col7" class="data row6 col7" >0.01</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row6_col8" class="data row6 col8" >0.06</td>
            </tr>
            <tr>
                        <th id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6level0_row7" class="row_heading level0 row7" >median_income</th>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row7_col0" class="data row7 col0" >-0.02</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row7_col1" class="data row7 col1" >-0.08</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row7_col2" class="data row7 col2" >-0.11</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row7_col3" class="data row7 col3" >0.20</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row7_col4" class="data row7 col4" >-0.01</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row7_col5" class="data row7 col5" >0.00</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row7_col6" class="data row7 col6" >0.01</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row7_col7" class="data row7 col7" >1.00</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row7_col8" class="data row7 col8" >0.69</td>
            </tr>
            <tr>
                        <th id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6level0_row8" class="row_heading level0 row8" >median_house_value</th>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row8_col0" class="data row8 col0" >-0.05</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row8_col1" class="data row8 col1" >-0.14</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row8_col2" class="data row8 col2" >0.11</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row8_col3" class="data row8 col3" >0.14</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row8_col4" class="data row8 col4" >0.05</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row8_col5" class="data row8 col5" >-0.03</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row8_col6" class="data row8 col6" >0.06</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row8_col7" class="data row8 col7" >0.69</td>
                        <td id="T_147a08a8_87d1_11eb_8ee8_f218983eaaf6row8_col8" class="data row8 col8" >1.00</td>
            </tr>
    </tbody></table>



```python
corr_matrix['median_house_value'].sort_values(ascending=True)
```




    latitude             -0.142724
    longitude            -0.047432
    population           -0.026920
    total_bedrooms        0.047689
    households            0.064506
    housing_median_age    0.114110
    total_rooms           0.135097
    median_income         0.687160
    median_house_value    1.000000
    Name: median_house_value, dtype: float64



The correlation matrix suggests the amount of correlation between a pair of variables. When close to 1 it means a strong +ve correlation whereas, -1 means an inverse correlation. Looking at the correlation between median_house_values and other variables, we can see that there's some correlation with median_income (0.68 -- so +ve), and with the latitude (-0.14 -- so an inverse relation).

Another to check this relation is to plot scatter plots for each pair of variables in the dataset. Below we plot this for few potential/interesting variables 

```python
# from pandas.tools.plotting import scatter_matrix # For older versions of Pandas
from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
```




    array([[<AxesSubplot:xlabel='median_house_value', ylabel='median_house_value'>,
            <AxesSubplot:xlabel='median_income', ylabel='median_house_value'>,
            <AxesSubplot:xlabel='total_rooms', ylabel='median_house_value'>,
            <AxesSubplot:xlabel='housing_median_age', ylabel='median_house_value'>],
           [<AxesSubplot:xlabel='median_house_value', ylabel='median_income'>,
            <AxesSubplot:xlabel='median_income', ylabel='median_income'>,
            <AxesSubplot:xlabel='total_rooms', ylabel='median_income'>,
            <AxesSubplot:xlabel='housing_median_age', ylabel='median_income'>],
           [<AxesSubplot:xlabel='median_house_value', ylabel='total_rooms'>,
            <AxesSubplot:xlabel='median_income', ylabel='total_rooms'>,
            <AxesSubplot:xlabel='total_rooms', ylabel='total_rooms'>,
            <AxesSubplot:xlabel='housing_median_age', ylabel='total_rooms'>],
           [<AxesSubplot:xlabel='median_house_value', ylabel='housing_median_age'>,
            <AxesSubplot:xlabel='median_income', ylabel='housing_median_age'>,
            <AxesSubplot:xlabel='total_rooms', ylabel='housing_median_age'>,
            <AxesSubplot:xlabel='housing_median_age', ylabel='housing_median_age'>]],
          dtype=object)




![png](end2endML_housing_files/output_48_1.png)


The diagonal entries show the histogram for each variable. We saw this previously for some variables. The most promising variable from this analysis seems to be the median_income. 

```python
housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1)
```




    <AxesSubplot:xlabel='median_income', ylabel='median_house_value'>




![png](end2endML_housing_files/output_50_1.png)


Plotting it shows the stronger correlation with the target variable i.e. median_house_value however we can see horizontal lines (especially at USD 500k, 450k 350k) these could be due to some stratifying done in the dataset implicitly. It would be better to remove those to ensure our model does not spuriously fit for those since they are some of the quirks in the data. 

### Experimenting with attributes

Before we began proposing models for the data. We can play around with the variables and try different combinations of them to see if we get better trends. Let's look at a few. First, the total_room and/or total_bedroom variable could be changed to average_bedroom_per_house to better for bedrooms rather than looking for total bedroom in that district we would be looking at avg_bedroom per district and similarly we would do it for rooms. 

```python
#Average bedroom per house-holds in the district 
housing['avg_bedroom']=housing['total_bedrooms']/housing['households']

#Average room per house-holds in the district 
housing['avg_room']=housing['total_rooms']/housing['households']

#Average bedrooms per rooms in a given district
housing['bedroom_per_room']=housing['total_bedrooms']/housing['total_rooms']

#Average population per household in a given district
housing['population_per_household']=housing['population']/housing['households']

#Average room per population in a given district
housing['room_per_popoulation']=housing['total_rooms']/housing['population']

#Average room per population in a given district
housing['room_per_popoulation']=housing['total_rooms']/housing['population']
```

```python
#Making the correlation matrix again 
corr_matrix=housing.corr()
corr_matrix.style.background_gradient(cmap='coolwarm').set_precision(2)

```




<style  type="text/css" >
#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row0_col0,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row1_col1,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row2_col2,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row3_col3,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row4_col4,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row5_col5,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row6_col6,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row7_col7,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row8_col8,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row9_col9,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row10_col10,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row11_col11,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row12_col12,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row13_col13{
            background-color:  #b40426;
            color:  #f1f1f1;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row0_col1,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row1_col0,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row2_col3,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row2_col4,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row2_col5,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row2_col6,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row2_col9,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row3_col2,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row7_col11,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row11_col7,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row11_col8,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row11_col10,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row11_col13,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row13_col12{
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row0_col2,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row1_col4,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row9_col6,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row12_col8{
            background-color:  #779af7;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row0_col3,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row0_col4,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row3_col13{
            background-color:  #9fbfff;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row0_col5,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row1_col11,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row2_col7,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row3_col8{
            background-color:  #a3c2fe;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row0_col6,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row8_col4,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row8_col6,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row13_col4{
            background-color:  #98b9ff;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row0_col7{
            background-color:  #b7cff9;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row0_col8,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row10_col6{
            background-color:  #7093f3;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row0_col9{
            background-color:  #536edd;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row0_col10{
            background-color:  #92b4fe;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row0_col11,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row4_col1,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row13_col0{
            background-color:  #cdd9ec;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row0_col12,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row5_col2{
            background-color:  #4961d2;
            color:  #f1f1f1;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row0_col13{
            background-color:  #6a8bef;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row1_col2{
            background-color:  #94b6ff;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row1_col3{
            background-color:  #89acfd;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row1_col5{
            background-color:  #6788ee;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row1_col6,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row9_col8,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row10_col5{
            background-color:  #7295f4;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row1_col7{
            background-color:  #abc8fd;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row1_col8{
            background-color:  #5673e0;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row1_col9{
            background-color:  #6485ec;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row1_col10,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row10_col3{
            background-color:  #b3cdfb;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row1_col12,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row11_col12{
            background-color:  #4a63d3;
            color:  #f1f1f1;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row1_col13{
            background-color:  #a5c3fe;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row2_col0{
            background-color:  #c7d7f0;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row2_col1,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row12_col1{
            background-color:  #d8dce2;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row2_col8{
            background-color:  #9ebeff;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row2_col10{
            background-color:  #7597f6;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row2_col11,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row10_col0{
            background-color:  #d4dbe6;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row2_col12{
            background-color:  #4c66d6;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row2_col13{
            background-color:  #6180e9;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row3_col0,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row3_col7,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row7_col10{
            background-color:  #dedcdb;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row3_col1{
            background-color:  #d2dbe8;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row3_col4,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row4_col3{
            background-color:  #c73635;
            color:  #f1f1f1;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row3_col5{
            background-color:  #d95847;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row3_col6,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row6_col3{
            background-color:  #ca3b37;
            color:  #f1f1f1;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row3_col9,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row5_col13{
            background-color:  #5875e1;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row3_col10,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row4_col7{
            background-color:  #b9d0f9;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row3_col11{
            background-color:  #93b5fe;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row3_col12,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row8_col12{
            background-color:  #4257c9;
            color:  #f1f1f1;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row4_col0{
            background-color:  #e2dad5;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row4_col2,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row8_col9{
            background-color:  #4358cb;
            color:  #f1f1f1;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row4_col5{
            background-color:  #d55042;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row4_col6,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row6_col4{
            background-color:  #b8122a;
            color:  #f1f1f1;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row4_col8,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row4_col13,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row7_col6,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row10_col4{
            background-color:  #8badfd;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row4_col9{
            background-color:  #5d7ce6;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row4_col10,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row9_col3,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row11_col6,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row12_col5{
            background-color:  #9abbff;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row4_col11,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row6_col1,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row7_col1{
            background-color:  #ccd9ed;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row4_col12,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row6_col12{
            background-color:  #4055c8;
            color:  #f1f1f1;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row5_col0,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row10_col1{
            background-color:  #e6d7cf;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row5_col1,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row11_col1{
            background-color:  #c6d6f1;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row5_col3{
            background-color:  #d85646;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row5_col4{
            background-color:  #d44e41;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row5_col6,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row6_col5{
            background-color:  #cd423b;
            color:  #f1f1f1;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row5_col7{
            background-color:  #bbd1f8;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row5_col8,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row7_col2{
            background-color:  #7699f6;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row5_col9{
            background-color:  #3c4ec2;
            color:  #f1f1f1;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row5_col10,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row7_col5{
            background-color:  #86a9fc;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row5_col11{
            background-color:  #c3d5f4;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row5_col12,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row11_col9,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row13_col5{
            background-color:  #5f7fe8;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row6_col0{
            background-color:  #e0dbd8;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row6_col2,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row9_col12,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row10_col12{
            background-color:  #465ecf;
            color:  #f1f1f1;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row6_col7{
            background-color:  #bed2f6;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row6_col8{
            background-color:  #8fb1fe;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row6_col9{
            background-color:  #3f53c6;
            color:  #f1f1f1;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row6_col10,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row8_col11,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row12_col4,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row13_col11{
            background-color:  #84a7fc;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row6_col11{
            background-color:  #c9d7f0;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row6_col13,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row9_col5{
            background-color:  #7396f5;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row7_col0{
            background-color:  #d5dbe5;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row7_col3{
            background-color:  #c4d5f3;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row7_col4{
            background-color:  #88abfd;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row7_col8{
            background-color:  #f4987a;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row7_col9{
            background-color:  #3e51c5;
            color:  #f1f1f1;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row7_col12{
            background-color:  #4f69d9;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row7_col13,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row8_col10,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row12_col11{
            background-color:  #bcd2f7;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row8_col0{
            background-color:  #d1dae9;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row8_col1{
            background-color:  #c1d4f4;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row8_col2,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row9_col7{
            background-color:  #afcafc;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row8_col3,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row8_col13,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row13_col8{
            background-color:  #b5cdfa;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row8_col5,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row13_col6{
            background-color:  #7ea1fa;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row8_col7{
            background-color:  #ec8165;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row9_col0{
            background-color:  #d9dce1;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row9_col1{
            background-color:  #e1dad6;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row9_col2,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row12_col6{
            background-color:  #80a3fa;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row9_col4,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row12_col2,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row12_col10{
            background-color:  #97b8ff;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row9_col10{
            background-color:  #d65244;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row9_col11{
            background-color:  #c5d6f2;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row9_col13,#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row10_col9{
            background-color:  #de614d;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row10_col2{
            background-color:  #6c8ff1;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row10_col7{
            background-color:  #eed0c0;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row10_col8{
            background-color:  #a6c4fe;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row10_col11{
            background-color:  #6687ed;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row10_col13{
            background-color:  #d1493f;
            color:  #f1f1f1;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row11_col0{
            background-color:  #e4d9d2;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row11_col2{
            background-color:  #b6cefa;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row11_col3{
            background-color:  #6282ea;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row11_col4{
            background-color:  #a2c1ff;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row11_col5{
            background-color:  #90b2fe;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row12_col0{
            background-color:  #d7dce3;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row12_col3{
            background-color:  #8caffe;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row12_col7{
            background-color:  #bfd3f6;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row12_col9{
            background-color:  #4e68d8;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row12_col13{
            background-color:  #6f92f3;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row13_col1{
            background-color:  #ead5c9;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row13_col2{
            background-color:  #799cf8;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row13_col3{
            background-color:  #b1cbfc;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row13_col7{
            background-color:  #e3d9d3;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row13_col9{
            background-color:  #e36c55;
            color:  #000000;
        }#T_317b7448_87d1_11eb_9c2c_f218983eaaf6row13_col10{
            background-color:  #cf453c;
            color:  #f1f1f1;
        }</style><table id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >longitude</th>        <th class="col_heading level0 col1" >latitude</th>        <th class="col_heading level0 col2" >housing_median_age</th>        <th class="col_heading level0 col3" >total_rooms</th>        <th class="col_heading level0 col4" >total_bedrooms</th>        <th class="col_heading level0 col5" >population</th>        <th class="col_heading level0 col6" >households</th>        <th class="col_heading level0 col7" >median_income</th>        <th class="col_heading level0 col8" >median_house_value</th>        <th class="col_heading level0 col9" >avg_bedroom</th>        <th class="col_heading level0 col10" >avg_room</th>        <th class="col_heading level0 col11" >bedroom_per_room</th>        <th class="col_heading level0 col12" >population_per_household</th>        <th class="col_heading level0 col13" >room_per_popoulation</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6level0_row0" class="row_heading level0 row0" >longitude</th>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row0_col0" class="data row0 col0" >1.00</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row0_col1" class="data row0 col1" >-0.92</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row0_col2" class="data row0 col2" >-0.11</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row0_col3" class="data row0 col3" >0.05</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row0_col4" class="data row0 col4" >0.08</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row0_col5" class="data row0 col5" >0.11</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row0_col6" class="data row0 col6" >0.06</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row0_col7" class="data row0 col7" >-0.02</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row0_col8" class="data row0 col8" >-0.05</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row0_col9" class="data row0 col9" >0.01</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row0_col10" class="data row0 col10" >-0.03</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row0_col11" class="data row0 col11" >0.10</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row0_col12" class="data row0 col12" >-0.00</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row0_col13" class="data row0 col13" >-0.07</td>
            </tr>
            <tr>
                        <th id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6level0_row1" class="row_heading level0 row1" >latitude</th>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row1_col0" class="data row1 col0" >-0.92</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row1_col1" class="data row1 col1" >1.00</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row1_col2" class="data row1 col2" >0.01</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row1_col3" class="data row1 col3" >-0.04</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row1_col4" class="data row1 col4" >-0.07</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row1_col5" class="data row1 col5" >-0.12</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row1_col6" class="data row1 col6" >-0.08</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row1_col7" class="data row1 col7" >-0.08</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row1_col8" class="data row1 col8" >-0.14</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row1_col9" class="data row1 col9" >0.07</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row1_col10" class="data row1 col10" >0.11</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row1_col11" class="data row1 col11" >-0.12</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row1_col12" class="data row1 col12" >0.01</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row1_col13" class="data row1 col13" >0.14</td>
            </tr>
            <tr>
                        <th id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6level0_row2" class="row_heading level0 row2" >housing_median_age</th>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row2_col0" class="data row2 col0" >-0.11</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row2_col1" class="data row2 col1" >0.01</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row2_col2" class="data row2 col2" >1.00</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row2_col3" class="data row2 col3" >-0.36</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row2_col4" class="data row2 col4" >-0.33</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row2_col5" class="data row2 col5" >-0.30</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row2_col6" class="data row2 col6" >-0.31</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row2_col7" class="data row2 col7" >-0.11</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row2_col8" class="data row2 col8" >0.11</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row2_col9" class="data row2 col9" >-0.08</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row2_col10" class="data row2 col10" >-0.15</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row2_col11" class="data row2 col11" >0.14</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row2_col12" class="data row2 col12" >0.02</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row2_col13" class="data row2 col13" >-0.10</td>
            </tr>
            <tr>
                        <th id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6level0_row3" class="row_heading level0 row3" >total_rooms</th>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row3_col0" class="data row3 col0" >0.05</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row3_col1" class="data row3 col1" >-0.04</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row3_col2" class="data row3 col2" >-0.36</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row3_col3" class="data row3 col3" >1.00</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row3_col4" class="data row3 col4" >0.93</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row3_col5" class="data row3 col5" >0.86</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row3_col6" class="data row3 col6" >0.92</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row3_col7" class="data row3 col7" >0.20</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row3_col8" class="data row3 col8" >0.14</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row3_col9" class="data row3 col9" >0.03</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row3_col10" class="data row3 col10" >0.13</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row3_col11" class="data row3 col11" >-0.19</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row3_col12" class="data row3 col12" >-0.02</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row3_col13" class="data row3 col13" >0.12</td>
            </tr>
            <tr>
                        <th id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6level0_row4" class="row_heading level0 row4" >total_bedrooms</th>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row4_col0" class="data row4 col0" >0.08</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row4_col1" class="data row4 col1" >-0.07</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row4_col2" class="data row4 col2" >-0.33</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row4_col3" class="data row4 col3" >0.93</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row4_col4" class="data row4 col4" >1.00</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row4_col5" class="data row4 col5" >0.88</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row4_col6" class="data row4 col6" >0.98</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row4_col7" class="data row4 col7" >-0.01</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row4_col8" class="data row4 col8" >0.05</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row4_col9" class="data row4 col9" >0.04</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row4_col10" class="data row4 col10" >0.00</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row4_col11" class="data row4 col11" >0.09</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row4_col12" class="data row4 col12" >-0.03</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row4_col13" class="data row4 col13" >0.05</td>
            </tr>
            <tr>
                        <th id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6level0_row5" class="row_heading level0 row5" >population</th>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row5_col0" class="data row5 col0" >0.11</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row5_col1" class="data row5 col1" >-0.12</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row5_col2" class="data row5 col2" >-0.30</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row5_col3" class="data row5 col3" >0.86</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row5_col4" class="data row5 col4" >0.88</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row5_col5" class="data row5 col5" >1.00</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row5_col6" class="data row5 col6" >0.90</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row5_col7" class="data row5 col7" >0.00</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row5_col8" class="data row5 col8" >-0.03</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row5_col9" class="data row5 col9" >-0.07</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row5_col10" class="data row5 col10" >-0.07</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row5_col11" class="data row5 col11" >0.04</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row5_col12" class="data row5 col12" >0.08</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row5_col13" class="data row5 col13" >-0.14</td>
            </tr>
            <tr>
                        <th id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6level0_row6" class="row_heading level0 row6" >households</th>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row6_col0" class="data row6 col0" >0.06</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row6_col1" class="data row6 col1" >-0.08</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row6_col2" class="data row6 col2" >-0.31</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row6_col3" class="data row6 col3" >0.92</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row6_col4" class="data row6 col4" >0.98</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row6_col5" class="data row6 col5" >0.90</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row6_col6" class="data row6 col6" >1.00</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row6_col7" class="data row6 col7" >0.01</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row6_col8" class="data row6 col8" >0.06</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row6_col9" class="data row6 col9" >-0.06</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row6_col10" class="data row6 col10" >-0.08</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row6_col11" class="data row6 col11" >0.07</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row6_col12" class="data row6 col12" >-0.03</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row6_col13" class="data row6 col13" >-0.04</td>
            </tr>
            <tr>
                        <th id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6level0_row7" class="row_heading level0 row7" >median_income</th>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row7_col0" class="data row7 col0" >-0.02</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row7_col1" class="data row7 col1" >-0.08</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row7_col2" class="data row7 col2" >-0.11</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row7_col3" class="data row7 col3" >0.20</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row7_col4" class="data row7 col4" >-0.01</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row7_col5" class="data row7 col5" >0.00</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row7_col6" class="data row7 col6" >0.01</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row7_col7" class="data row7 col7" >1.00</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row7_col8" class="data row7 col8" >0.69</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row7_col9" class="data row7 col9" >-0.06</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row7_col10" class="data row7 col10" >0.31</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row7_col11" class="data row7 col11" >-0.62</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row7_col12" class="data row7 col12" >0.02</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row7_col13" class="data row7 col13" >0.23</td>
            </tr>
            <tr>
                        <th id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6level0_row8" class="row_heading level0 row8" >median_house_value</th>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row8_col0" class="data row8 col0" >-0.05</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row8_col1" class="data row8 col1" >-0.14</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row8_col2" class="data row8 col2" >0.11</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row8_col3" class="data row8 col3" >0.14</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row8_col4" class="data row8 col4" >0.05</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row8_col5" class="data row8 col5" >-0.03</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row8_col6" class="data row8 col6" >0.06</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row8_col7" class="data row8 col7" >0.69</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row8_col8" class="data row8 col8" >1.00</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row8_col9" class="data row8 col9" >-0.04</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row8_col10" class="data row8 col10" >0.15</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row8_col11" class="data row8 col11" >-0.26</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row8_col12" class="data row8 col12" >-0.02</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row8_col13" class="data row8 col13" >0.20</td>
            </tr>
            <tr>
                        <th id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6level0_row9" class="row_heading level0 row9" >avg_bedroom</th>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row9_col0" class="data row9 col0" >0.01</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row9_col1" class="data row9 col1" >0.07</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row9_col2" class="data row9 col2" >-0.08</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row9_col3" class="data row9 col3" >0.03</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row9_col4" class="data row9 col4" >0.04</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row9_col5" class="data row9 col5" >-0.07</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row9_col6" class="data row9 col6" >-0.06</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row9_col7" class="data row9 col7" >-0.06</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row9_col8" class="data row9 col8" >-0.04</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row9_col9" class="data row9 col9" >1.00</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row9_col10" class="data row9 col10" >0.86</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row9_col11" class="data row9 col11" >0.05</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row9_col12" class="data row9 col12" >-0.01</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row9_col13" class="data row9 col13" >0.84</td>
            </tr>
            <tr>
                        <th id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6level0_row10" class="row_heading level0 row10" >avg_room</th>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row10_col0" class="data row10 col0" >-0.03</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row10_col1" class="data row10 col1" >0.11</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row10_col2" class="data row10 col2" >-0.15</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row10_col3" class="data row10 col3" >0.13</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row10_col4" class="data row10 col4" >0.00</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row10_col5" class="data row10 col5" >-0.07</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row10_col6" class="data row10 col6" >-0.08</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row10_col7" class="data row10 col7" >0.31</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row10_col8" class="data row10 col8" >0.15</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row10_col9" class="data row10 col9" >0.86</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row10_col10" class="data row10 col10" >1.00</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row10_col11" class="data row10 col11" >-0.40</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row10_col12" class="data row10 col12" >-0.01</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row10_col13" class="data row10 col13" >0.90</td>
            </tr>
            <tr>
                        <th id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6level0_row11" class="row_heading level0 row11" >bedroom_per_room</th>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row11_col0" class="data row11 col0" >0.10</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row11_col1" class="data row11 col1" >-0.12</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row11_col2" class="data row11 col2" >0.14</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row11_col3" class="data row11 col3" >-0.19</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row11_col4" class="data row11 col4" >0.09</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row11_col5" class="data row11 col5" >0.04</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row11_col6" class="data row11 col6" >0.07</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row11_col7" class="data row11 col7" >-0.62</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row11_col8" class="data row11 col8" >-0.26</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row11_col9" class="data row11 col9" >0.05</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row11_col10" class="data row11 col10" >-0.40</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row11_col11" class="data row11 col11" >1.00</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row11_col12" class="data row11 col12" >0.00</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row11_col13" class="data row11 col13" >-0.26</td>
            </tr>
            <tr>
                        <th id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6level0_row12" class="row_heading level0 row12" >population_per_household</th>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row12_col0" class="data row12 col0" >-0.00</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row12_col1" class="data row12 col1" >0.01</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row12_col2" class="data row12 col2" >0.02</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row12_col3" class="data row12 col3" >-0.02</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row12_col4" class="data row12 col4" >-0.03</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row12_col5" class="data row12 col5" >0.08</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row12_col6" class="data row12 col6" >-0.03</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row12_col7" class="data row12 col7" >0.02</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row12_col8" class="data row12 col8" >-0.02</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row12_col9" class="data row12 col9" >-0.01</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row12_col10" class="data row12 col10" >-0.01</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row12_col11" class="data row12 col11" >0.00</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row12_col12" class="data row12 col12" >1.00</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row12_col13" class="data row12 col13" >-0.05</td>
            </tr>
            <tr>
                        <th id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6level0_row13" class="row_heading level0 row13" >room_per_popoulation</th>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row13_col0" class="data row13 col0" >-0.07</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row13_col1" class="data row13 col1" >0.14</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row13_col2" class="data row13 col2" >-0.10</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row13_col3" class="data row13 col3" >0.12</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row13_col4" class="data row13 col4" >0.05</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row13_col5" class="data row13 col5" >-0.14</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row13_col6" class="data row13 col6" >-0.04</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row13_col7" class="data row13 col7" >0.23</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row13_col8" class="data row13 col8" >0.20</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row13_col9" class="data row13 col9" >0.84</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row13_col10" class="data row13 col10" >0.90</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row13_col11" class="data row13 col11" >-0.26</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row13_col12" class="data row13 col12" >-0.05</td>
                        <td id="T_317b7448_87d1_11eb_9c2c_f218983eaaf6row13_col13" class="data row13 col13" >1.00</td>
            </tr>
    </tbody></table>



```python
corr_matrix['median_house_value'].sort_values(ascending=True)
```




    bedroom_per_room           -0.259984
    latitude                   -0.142724
    longitude                  -0.047432
    avg_bedroom                -0.043343
    population                 -0.026920
    population_per_household   -0.021985
    total_bedrooms              0.047689
    households                  0.064506
    housing_median_age          0.114110
    total_rooms                 0.135097
    avg_room                    0.146285
    room_per_popoulation        0.199429
    median_income               0.687160
    median_house_value          1.000000
    Name: median_house_value, dtype: float64



This is interesting! We see that `bedroom_per_room` is another potential descriptor with negative corelation moreover we get `room_per_population` and `avg_room` to be decent new descriptors for the `median_house_value`. Not bad for a simple math manipulation to better represent the data. This is a crucial step and where domain knowledge and intuition would come handy. 

### Data cleaning and prepping

1. Separate the predictors and the target values 
2. Write functions to conduct various data transformations ensuring consistency and ease 
3. Make sure the data is devoid of any NaN values since that would raise warning and errors. We have three strategies we can implement here:
    a. Get rid of those points (districts) entirely 
    b. Get rid of whole attribute 
    c. Set missing values to either zero or one of the averages (mean, median, or mode) 

In our case, total bedrooms had some missing values. 
```python
# Option a:
housing.dropna(subset=["total_bedrooms"])
# Option b:
housing.drop("total_bedrooms", axis=1)
# Option c:
median = housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median, inplace=True) # option 3
```

Before we do any of this let's first separate the predictor and target_values

```python
housing = strat_train_set.drop("median_house_value", axis=1) # drop labels for training set
housing_labels = strat_train_set["median_house_value"].copy()
```

```python
#Checking the NULL enties in the dataset
sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()
sample_incomplete_rows
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>ocean_proximity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4629</th>
      <td>-118.30</td>
      <td>34.07</td>
      <td>18.0</td>
      <td>3759.0</td>
      <td>NaN</td>
      <td>3296.0</td>
      <td>1462.0</td>
      <td>2.2708</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>6068</th>
      <td>-117.86</td>
      <td>34.01</td>
      <td>16.0</td>
      <td>4632.0</td>
      <td>NaN</td>
      <td>3038.0</td>
      <td>727.0</td>
      <td>5.1762</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>17923</th>
      <td>-121.97</td>
      <td>37.35</td>
      <td>30.0</td>
      <td>1955.0</td>
      <td>NaN</td>
      <td>999.0</td>
      <td>386.0</td>
      <td>4.6328</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>13656</th>
      <td>-117.30</td>
      <td>34.05</td>
      <td>6.0</td>
      <td>2155.0</td>
      <td>NaN</td>
      <td>1039.0</td>
      <td>391.0</td>
      <td>1.6675</td>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>19252</th>
      <td>-122.79</td>
      <td>38.48</td>
      <td>7.0</td>
      <td>6837.0</td>
      <td>NaN</td>
      <td>3468.0</td>
      <td>1405.0</td>
      <td>3.1662</td>
      <td>&lt;1H OCEAN</td>
    </tr>
  </tbody>
</table>
</div>



```python
sample_incomplete_rows.dropna(subset=["total_bedrooms"])    # option 1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>ocean_proximity</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



```python
sample_incomplete_rows.drop("total_bedrooms", axis=1)   #option 2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>ocean_proximity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4629</th>
      <td>-118.30</td>
      <td>34.07</td>
      <td>18.0</td>
      <td>3759.0</td>
      <td>3296.0</td>
      <td>1462.0</td>
      <td>2.2708</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>6068</th>
      <td>-117.86</td>
      <td>34.01</td>
      <td>16.0</td>
      <td>4632.0</td>
      <td>3038.0</td>
      <td>727.0</td>
      <td>5.1762</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>17923</th>
      <td>-121.97</td>
      <td>37.35</td>
      <td>30.0</td>
      <td>1955.0</td>
      <td>999.0</td>
      <td>386.0</td>
      <td>4.6328</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>13656</th>
      <td>-117.30</td>
      <td>34.05</td>
      <td>6.0</td>
      <td>2155.0</td>
      <td>1039.0</td>
      <td>391.0</td>
      <td>1.6675</td>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>19252</th>
      <td>-122.79</td>
      <td>38.48</td>
      <td>7.0</td>
      <td>6837.0</td>
      <td>3468.0</td>
      <td>1405.0</td>
      <td>3.1662</td>
      <td>&lt;1H OCEAN</td>
    </tr>
  </tbody>
</table>
</div>



```python
median = housing["total_bedrooms"].median()
sample_incomplete_rows["total_bedrooms"].fillna(median, inplace=True) # option 3
sample_incomplete_rows
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>ocean_proximity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4629</th>
      <td>-118.30</td>
      <td>34.07</td>
      <td>18.0</td>
      <td>3759.0</td>
      <td>433.0</td>
      <td>3296.0</td>
      <td>1462.0</td>
      <td>2.2708</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>6068</th>
      <td>-117.86</td>
      <td>34.01</td>
      <td>16.0</td>
      <td>4632.0</td>
      <td>433.0</td>
      <td>3038.0</td>
      <td>727.0</td>
      <td>5.1762</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>17923</th>
      <td>-121.97</td>
      <td>37.35</td>
      <td>30.0</td>
      <td>1955.0</td>
      <td>433.0</td>
      <td>999.0</td>
      <td>386.0</td>
      <td>4.6328</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>13656</th>
      <td>-117.30</td>
      <td>34.05</td>
      <td>6.0</td>
      <td>2155.0</td>
      <td>433.0</td>
      <td>1039.0</td>
      <td>391.0</td>
      <td>1.6675</td>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>19252</th>
      <td>-122.79</td>
      <td>38.48</td>
      <td>7.0</td>
      <td>6837.0</td>
      <td>433.0</td>
      <td>3468.0</td>
      <td>1405.0</td>
      <td>3.1662</td>
      <td>&lt;1H OCEAN</td>
    </tr>
  </tbody>
</table>
</div>



##### Scikit-learn imputer class 
This is a handy class to take of missing values. First, we create an instance of that class with specifying what is to be replaced and what strategy is used. Before doing so, we need to make srue the entire data-set has ONLY numerical entries and Imputer will evaluate the given average for all the dataset and store it in the `statistics_` instance

What the imputer will do is, 
1. Evaluate an specified type of average. 
2. For a given numerical data-set look for NaN or Null entires in a given attribute and replace it with the computed avearge for that attribute 

```python
try:
    from sklearn.impute import SimpleImputer # Scikit-Learn 0.20+
except ImportError:
    from sklearn.preprocessing import Imputer as SimpleImputer

imputer = SimpleImputer(strategy="median") #We define the strategy here 
housing_num = housing.drop('ocean_proximity', axis=1)
# alternatively: housing_num = housing.select_dtypes(include=[np.number])
```

```python
imputer.fit(housing_num)
```




    SimpleImputer(strategy='median')



```python
imputer.statistics_
```




    array([-118.51  ,   34.26  ,   29.    , 2119.5   ,  433.    , 1164.    ,
            408.    ,    3.5409])



```python
housing_num.median().values
```




    array([-118.51  ,   34.26  ,   29.    , 2119.5   ,  433.    , 1164.    ,
            408.    ,    3.5409])



We can now use this as a training variables set for our model

```python
X = imputer.transform(housing_num)
```

We convert the Pandas dataframe entries to a numpy array which is transformed with appropriately replacing the missing entries with median. 

```python
print(type(X), type(housing_num))
```

    <class 'numpy.ndarray'> <class 'pandas.core.frame.DataFrame'>


```python
print(np.shape(X), housing_num.shape)
'''
If we need the data-frame back
housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing.index)
'''
```

    (16512, 8) (16512, 8)





    '\nIf we need the data-frame back\nhousing_tr = pd.DataFrame(X, columns=housing_num.columns,\n                          index=housing.index)\n'



### Handling Text and Categorical Attribute 

Now let's preprocess the categorical input feature, `ocean_proximity`:

```python
housing_cat = housing[['ocean_proximity']]
type(housing_cat)
```




    pandas.core.frame.DataFrame



Converting the categorical entries to integers

```python
try:
    from sklearn.preprocessing import OrdinalEncoder
except ImportError:
    from future_encoders import OrdinalEncoder # Scikit-Learn < 0.20
    
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:10]
```




    array([[0.],
           [0.],
           [4.],
           [1.],
           [0.],
           [1.],
           [0.],
           [1.],
           [0.],
           [0.]])



```python
housing["ocean_proximity"].value_counts()
```




    <1H OCEAN     7276
    INLAND        5263
    NEAR OCEAN    2124
    NEAR BAY      1847
    ISLAND           2
    Name: ocean_proximity, dtype: int64



Now, `housing_cat_encoded` has converted the categorical entries to purely numerical values for each category

```python
ordinal_encoder.categories_
```




    [array(['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'],
           dtype=object)]



Now, this is helpful with the string categories getting converted to numerical categories. However, there is still an small issue. The categorical numbering may introduce some bias in the final model. **ML algorithms will assume that two nearby values are more similar than two distant values.** 

In the above case, '<1H OCEAN' and 'INLAND'have category values as 0 and 1 however '<1H OCEAN' is more closer to 'NEAR OCEAN' with category value 4. 

To fix this issue, one solution is to create one binary attribute per category. This is called **One-hot encoding** as ONLY one of the attribute in the vector is 1 (hot) and others are 0 (cold). 

Scikit-learn provides a `OneHotEncoder` encoder to convert integer categorical values to one-hot vectors.

```python
try:
    from sklearn.preprocessing import OrdinalEncoder # just to raise an ImportError if Scikit-Learn < 0.20
    from sklearn.preprocessing import OneHotEncoder
except ImportError:
    from future_encoders import OneHotEncoder # Scikit-Learn < 0.20

cat_encoder = OneHotEncoder()
#1-Hot encoded vector for the housing training data-set 
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
type(housing_cat_1hot)
```




    scipy.sparse.csr.csr_matrix



A sparse array is declared in this case which has the position of the non-zero value and not necessarily the entire numpy matrix. This is helpful in the cases where there are too many categories and also many datapoints. For examples, if we have 4 categories and 1000 datapoints the final one-hot matrix would be 1000x4 size. Most of that would be full of 0s, with only one 1 per row for a particular category. 

The `housing_cat_1hot` can be converted to numpy array by using the `housing_cat_1hot.toarray()` 

Alternatively, you can set `sparse=False` when creating the `OneHotEncoder`

```python
cat_encoder.categories_
```




    [array(['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'],
           dtype=object)]



Let's create a custom transformer to add extra attributes:

```python
housing.columns
```




    Index(['longitude', 'latitude', 'housing_median_age', 'total_rooms',
           'total_bedrooms', 'population', 'households', 'median_income',
           'ocean_proximity'],
          dtype='object')



```python
from sklearn.base import BaseEstimator, TransformerMixin
# get the right column indices: safer than hard-coding indices
rooms_ix, bedrooms_ix, population_ix, household_ix = [
    list(housing.columns).index(col) for col in ("total_rooms", "total_bedrooms", "population", "households")]

#Here we convert the housing.columns to list and 
#then find the index for the entry which matches the string in the loop 

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kwargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=True)
housing_extra_attribs = attr_adder.transform(housing.values)
```

From the above class, there's only one hyperparameter in the class. `add_bedrooms_per_room` is the only option and is set True by default. Let's check the new feature space by converting it to a `Pandas Dataframe`

```python
housing_extra_attribs = pd.DataFrame(
    housing_extra_attribs,
    columns=list(housing.columns)+["bedrooms_per_room","rooms_per_household", "population_per_household"],
    index=housing.index)
housing_extra_attribs.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>ocean_proximity</th>
      <th>bedrooms_per_room</th>
      <th>rooms_per_household</th>
      <th>population_per_household</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>17606</th>
      <td>-121.89</td>
      <td>37.29</td>
      <td>38</td>
      <td>1568</td>
      <td>351</td>
      <td>710</td>
      <td>339</td>
      <td>2.7042</td>
      <td>&lt;1H OCEAN</td>
      <td>4.62537</td>
      <td>2.0944</td>
      <td>0.223852</td>
    </tr>
    <tr>
      <th>18632</th>
      <td>-121.93</td>
      <td>37.05</td>
      <td>14</td>
      <td>679</td>
      <td>108</td>
      <td>306</td>
      <td>113</td>
      <td>6.4214</td>
      <td>&lt;1H OCEAN</td>
      <td>6.00885</td>
      <td>2.70796</td>
      <td>0.159057</td>
    </tr>
    <tr>
      <th>14650</th>
      <td>-117.2</td>
      <td>32.77</td>
      <td>31</td>
      <td>1952</td>
      <td>471</td>
      <td>936</td>
      <td>462</td>
      <td>2.8621</td>
      <td>NEAR OCEAN</td>
      <td>4.22511</td>
      <td>2.02597</td>
      <td>0.241291</td>
    </tr>
    <tr>
      <th>3230</th>
      <td>-119.61</td>
      <td>36.31</td>
      <td>25</td>
      <td>1847</td>
      <td>371</td>
      <td>1460</td>
      <td>353</td>
      <td>1.8839</td>
      <td>INLAND</td>
      <td>5.23229</td>
      <td>4.13598</td>
      <td>0.200866</td>
    </tr>
    <tr>
      <th>3555</th>
      <td>-118.59</td>
      <td>34.23</td>
      <td>17</td>
      <td>6592</td>
      <td>1525</td>
      <td>4459</td>
      <td>1463</td>
      <td>3.0347</td>
      <td>&lt;1H OCEAN</td>
      <td>4.50581</td>
      <td>3.04785</td>
      <td>0.231341</td>
    </tr>
  </tbody>
</table>
</div>



### Feature scaling 

Feaature scaling is an important transformation needed to be applied to the data. With some exceptions, ML algorithms dont perform well when the input numerical entries have very different scales. Eg: One variable has range 0-1 but other variable has range 1-1000. This is the case in our data-base where the total number of rooms range from 6 to 39,320 while the objective variable i.e. median income only ranges from 0-15. Two common ways of scaling: 

1. Min-max scaling (also called Normalization) 
Values are shifted such that they are normalized. They are rescaled in the range of 0-1. We do this by subtracting the min value and dividing by the range in the data
\begin{equation}
x_{i} = \frac{X_{i}-min}{max-min}
\end{equation}

2. Standardization 
This is when the mean of the dataset is subtracted from each entry so that the data has mean as 0 and then divided by the standard deviation so that the resulting distribution has a unit variance. Unlike min-max scaling, standardization does not bound to a particular range like 0-1. However, **standardisation is much less affected by outliers.** If a particular values is extremely high or low that could affect the other inputs in the case of min-max scaling. However that effect is reduced in the case of standardization given it does not directly account for the range in the scaling but the mean and variance. 
\begin{equation} 
x_{i} = \frac{X_{i}-\mu}{\sigma}
\end{equation} 
> **NOTE:** It is important that these scaling operations are performed on the training data only and not on the full dataset 

### Transformation pipelines

`Pipeline` class in scikit-learn can help with sequences of transformations. 
Now let's build a pipeline for preprocessing the numerical attributes (note that we could use `CombinedAttributesAdder()` instead of `FunctionTransformer(...)` if we preferred):


```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")), #Fill in missing values using the median of each entry 
        ('attribs_adder', CombinedAttributesAdder(add_bedrooms_per_room=True)), #Add additional columns entrys
        ('std_scaler', StandardScaler()), #Feature scaling -- using standardisation here 
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)
```

```python
housing_num_tr
```




    array([[-1.15604281,  0.77194962,  0.74333089, ..., -0.31205452,
            -0.08649871,  0.15531753],
           [-1.17602483,  0.6596948 , -1.1653172 , ...,  0.21768338,
            -0.03353391, -0.83628902],
           [ 1.18684903, -1.34218285,  0.18664186, ..., -0.46531516,
            -0.09240499,  0.4222004 ],
           ...,
           [ 1.58648943, -0.72478134, -1.56295222, ...,  0.3469342 ,
            -0.03055414, -0.52177644],
           [ 0.78221312, -0.85106801,  0.18664186, ...,  0.02499488,
             0.06150916, -0.30340741],
           [-1.43579109,  0.99645926,  1.85670895, ..., -0.22852947,
            -0.09586294,  0.10180567]])



This `Pipeline` constructor takes a list of name/estimator pairs defining a sequence of steps. **All but the last step/estimator must be the trasnformers (like feature scaling).** 

```python
try:
    from sklearn.compose import ColumnTransformer
except ImportError:
    from future_encoders import ColumnTransformer # Scikit-Learn < 0.20
```

**Now let's join all these components into a big pipeline that will preprocess both the numerical and the categorical features (again, we could use `CombinedAttributesAdder()` instead of `FunctionTransformer(...)` if we preferred):**

```python
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(housing)
```

```python
housing_prepared
```




    array([[-1.15604281,  0.77194962,  0.74333089, ...,  0.        ,
             0.        ,  0.        ],
           [-1.17602483,  0.6596948 , -1.1653172 , ...,  0.        ,
             0.        ,  0.        ],
           [ 1.18684903, -1.34218285,  0.18664186, ...,  0.        ,
             0.        ,  1.        ],
           ...,
           [ 1.58648943, -0.72478134, -1.56295222, ...,  0.        ,
             0.        ,  0.        ],
           [ 0.78221312, -0.85106801,  0.18664186, ...,  0.        ,
             0.        ,  0.        ],
           [-1.43579109,  0.99645926,  1.85670895, ...,  0.        ,
             1.        ,  0.        ]])



```python
housing_prepared.shape
```




    (16512, 16)



Finally. 

### Training and evaluation on the training set 
Given the prioir transformations, the features are scaled, categories are converted to one-hot vectors, and the missing variables are taken account of. Let's train a Linear Regression model first

```python
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression() 
lin_reg.fit(housing_prepared, housing_labels)
```




    LinearRegression()



```python
#Test some data out on the model 
trial_input = housing.iloc[:5] #First 5 entries 
trial_label = housing_labels.iloc[:5] #First 5 labels corresponding to the entries 
prep_trial_input = full_pipeline.transform(trial_input) #Transforming the entries to suit the trained input 
print('Predictions:',lin_reg.predict(prep_trial_input))
```

    Predictions: [210644.60459286 317768.80697211 210956.43331178  59218.98886849
     189747.55849879]


```python
print('Labels:',list(trial_label))
```

    Labels: [286600.0, 340600.0, 196900.0, 46300.0, 254500.0]


```python
from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse
```




    68628.19819848923



```python
housing_labels.describe()
```




    count     16512.000000
    mean     206990.920724
    std      115703.014830
    min       14999.000000
    25%      119800.000000
    50%      179500.000000
    75%      263900.000000
    max      500001.000000
    Name: median_house_value, dtype: float64



As seen the RMSE is 68628 which is better than nothing but still it is quite high when the range of the `median_house_values` range from 15000 to 500000

```python
from sklearn.metrics import mean_absolute_error

lin_mae = mean_absolute_error(housing_labels, housing_predictions)
lin_mae
```




    49439.89599001897



Here the model is underfitting the data since the RMSE is so high.
> When this happens either the features do not provide enough information to make good predictions or the model is not powerful enough. 

Let's try using a more complex model, `DecisionTreeRegressor` which is capable of finding non-linear relationships in the data

### Decision Tree Regressor

```python
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)
```




    DecisionTreeRegressor(random_state=42)



```python
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse
```




    0.0



# LOL! 

This model is badly overfitting the data! Let's proof this hypothesis using cross-validation schemes. We dont want to touch the test set, JUST WORK WITH THE TRAINING SET. Only touch the test set when the model we are using is good enough. We will use the part of the training set for training and other part for model validation. 

### Fine-tune the model

**Cross-validation** 

Cross-validation is a method of getting reliable estimate of model performance using only the training data 10-fold cross-validation --- breaking training data in 10 equal parts creating 10 miniature test/train splits. Out of the 10 folds, train data on 9 and test on 10th. Do this 10 times each time holding out different fold.

> Scikit-learn cross-validation feature expects a utility function (greater the better) rather than a cost function (lower the better), so to score the functions we use opposite of MSE, which is why we again compute `-scores` before calculating the square root

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
```

```python
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(tree_rmse_scores)
```

    Scores: [70194.33680785 66855.16363941 72432.58244769 70758.73896782
     71115.88230639 75585.14172901 70262.86139133 70273.6325285
     75366.87952553 71231.65726027]
    Mean: 71407.68766037929
    Standard deviation: 2439.4345041191004


Cross-validation not only allows us to get an estimate of the performance of the model but also the measure of how precise this estimate is (i.e. standard deviation). The Decision tree has high std-dev. This information could not be obtained with just one validation set. However, caveat is that cross-validation comes at the cost of training the model several times, so it is not always possible.

Let's compute the same score for `LinearRegresson` model. 

```python
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)
```

    Scores: [66782.73843989 66960.118071   70347.95244419 74739.57052552
     68031.13388938 71193.84183426 64969.63056405 68281.61137997
     71552.91566558 67665.10082067]
    Mean: 69052.46136345083
    Standard deviation: 2731.674001798349


Here it can be seen that DecisionTree model performs much worse than the LinearRegression model. 

### Random Forrest Regressor

Random forrest works by employing many decision trees on random subsets of the features, then averaging out their predictions. Building a model on top of many other models is called `Ensemble Learning` and it is often great way to push ML algorithms even further. 

```python
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators=10, random_state=42)
forest_reg.fit(housing_prepared, housing_labels)
```




    RandomForestRegressor(n_estimators=10, random_state=42)



> **Note**:we specify `n_estimators=10` to avoid a warning about the fact that the default value is going to change to 100 in Scikit-Learn 0.22.

```python
housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse
```




    21933.31414779769



```python
from sklearn.model_selection import cross_val_score

forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)
```

    Scores: [51646.44545909 48940.60114882 53050.86323649 54408.98730149
     50922.14870785 56482.50703987 51864.52025526 49760.85037653
     55434.21627933 53326.10093303]
    Mean: 52583.72407377466
    Standard deviation: 2298.353351147122


Random forest regressor looks better than `DecisionTree` and `LinearRegression`. The RMSE is still quite high for production quality code and could be due to overfitting. We can try other algorithms before spending time on a particular algorithm tweaking the hyperparameters. The goal is to shortlist 2-3 methods that are promising then fine-tune the model. Before we move ahead we can take a look at one more ML algorithm which is commonly employed for such supervised learning cases: Support Vector Regression

```python
from sklearn.svm import SVR

svm_reg = SVR(kernel="linear")
svm_reg.fit(housing_prepared, housing_labels)
housing_predictions = svm_reg.predict(housing_prepared)
svm_mse = mean_squared_error(housing_labels, housing_predictions)
svm_rmse = np.sqrt(svm_mse)
svm_rmse
```




    111094.6308539982



# Step 4: Fine-tune the model

Once we settle for an algorithm we can fine-tune them efficiently using some of the in-built scikit-learn routines. 

### Grid Search

`GridSearchCV` is a faster way of tweaking the hyper-parameters for a given algorithm. It needs the hyper-parameters you want to experiment with, what values to try out, and it will evaluate possible combination of hyperparameters values using cross-validation. We can do that step for `RandomForrestRegressor` which we found to have lowesst RMSE of the three methods we tried 

```python
from sklearn.model_selection import GridSearchCV

param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)
```




    GridSearchCV(cv=5, estimator=RandomForestRegressor(random_state=42),
                 param_grid=[{'max_features': [2, 4, 6, 8],
                              'n_estimators': [3, 10, 30]},
                             {'bootstrap': [False], 'max_features': [2, 3, 4],
                              'n_estimators': [3, 10]}],
                 return_train_score=True, scoring='neg_mean_squared_error')



The `param_grid` tells Scikit-learn to:
1. First evaluate 3x4 combinations of n_estimators and max_features with bootstrap `True` which is the default. 
2. Then with bootstrap set as `False` we look for 2x3 combinations of n_estimators and max_featurs for the random forest 
3. Finally, both these models are trained 5 times for the cross validation purposes in a 5-fold cross-validation fashion. 

Total of (12+6)x5=90 round of training are conducted. Finally when it is done we get the best model parameters which give lowest RMSE. 

```python
grid_search.best_params_
```




    {'max_features': 8, 'n_estimators': 30}



```python
grid_search.best_estimator_
```




    RandomForestRegressor(max_features=8, n_estimators=30, random_state=42)



```python
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
```

    63669.11631261028 {'max_features': 2, 'n_estimators': 3}
    55627.099719926795 {'max_features': 2, 'n_estimators': 10}
    53384.57275149205 {'max_features': 2, 'n_estimators': 30}
    60965.950449450494 {'max_features': 4, 'n_estimators': 3}
    52741.04704299915 {'max_features': 4, 'n_estimators': 10}
    50377.40461678399 {'max_features': 4, 'n_estimators': 30}
    58663.93866579625 {'max_features': 6, 'n_estimators': 3}
    52006.19873526564 {'max_features': 6, 'n_estimators': 10}
    50146.51167415009 {'max_features': 6, 'n_estimators': 30}
    57869.25276169646 {'max_features': 8, 'n_estimators': 3}
    51711.127883959234 {'max_features': 8, 'n_estimators': 10}
    49682.273345071546 {'max_features': 8, 'n_estimators': 30}
    62895.06951262424 {'bootstrap': False, 'max_features': 2, 'n_estimators': 3}
    54658.176157539405 {'bootstrap': False, 'max_features': 2, 'n_estimators': 10}
    59470.40652318466 {'bootstrap': False, 'max_features': 3, 'n_estimators': 3}
    52724.9822587892 {'bootstrap': False, 'max_features': 3, 'n_estimators': 10}
    57490.5691951261 {'bootstrap': False, 'max_features': 4, 'n_estimators': 3}
    51009.495668875716 {'bootstrap': False, 'max_features': 4, 'n_estimators': 10}


```python
pd.DataFrame(grid_search.cv_results_)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean_fit_time</th>
      <th>std_fit_time</th>
      <th>mean_score_time</th>
      <th>std_score_time</th>
      <th>param_max_features</th>
      <th>param_n_estimators</th>
      <th>param_bootstrap</th>
      <th>params</th>
      <th>split0_test_score</th>
      <th>split1_test_score</th>
      <th>...</th>
      <th>mean_test_score</th>
      <th>std_test_score</th>
      <th>rank_test_score</th>
      <th>split0_train_score</th>
      <th>split1_train_score</th>
      <th>split2_train_score</th>
      <th>split3_train_score</th>
      <th>split4_train_score</th>
      <th>mean_train_score</th>
      <th>std_train_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.054771</td>
      <td>0.003706</td>
      <td>0.003080</td>
      <td>0.000260</td>
      <td>2</td>
      <td>3</td>
      <td>NaN</td>
      <td>{'max_features': 2, 'n_estimators': 3}</td>
      <td>-3.837622e+09</td>
      <td>-4.147108e+09</td>
      <td>...</td>
      <td>-4.053756e+09</td>
      <td>1.519591e+08</td>
      <td>18</td>
      <td>-1.064113e+09</td>
      <td>-1.105142e+09</td>
      <td>-1.116550e+09</td>
      <td>-1.112342e+09</td>
      <td>-1.129650e+09</td>
      <td>-1.105559e+09</td>
      <td>2.220402e+07</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.172647</td>
      <td>0.006104</td>
      <td>0.009256</td>
      <td>0.001300</td>
      <td>2</td>
      <td>10</td>
      <td>NaN</td>
      <td>{'max_features': 2, 'n_estimators': 10}</td>
      <td>-3.047771e+09</td>
      <td>-3.254861e+09</td>
      <td>...</td>
      <td>-3.094374e+09</td>
      <td>1.327062e+08</td>
      <td>11</td>
      <td>-5.927175e+08</td>
      <td>-5.870952e+08</td>
      <td>-5.776964e+08</td>
      <td>-5.716332e+08</td>
      <td>-5.802501e+08</td>
      <td>-5.818785e+08</td>
      <td>7.345821e+06</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.502750</td>
      <td>0.005662</td>
      <td>0.023173</td>
      <td>0.000660</td>
      <td>2</td>
      <td>30</td>
      <td>NaN</td>
      <td>{'max_features': 2, 'n_estimators': 30}</td>
      <td>-2.689185e+09</td>
      <td>-3.021086e+09</td>
      <td>...</td>
      <td>-2.849913e+09</td>
      <td>1.626875e+08</td>
      <td>9</td>
      <td>-4.381089e+08</td>
      <td>-4.391272e+08</td>
      <td>-4.371702e+08</td>
      <td>-4.376955e+08</td>
      <td>-4.452654e+08</td>
      <td>-4.394734e+08</td>
      <td>2.966320e+06</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.081864</td>
      <td>0.000842</td>
      <td>0.003041</td>
      <td>0.000063</td>
      <td>4</td>
      <td>3</td>
      <td>NaN</td>
      <td>{'max_features': 4, 'n_estimators': 3}</td>
      <td>-3.730181e+09</td>
      <td>-3.786886e+09</td>
      <td>...</td>
      <td>-3.716847e+09</td>
      <td>1.631510e+08</td>
      <td>16</td>
      <td>-9.865163e+08</td>
      <td>-1.012565e+09</td>
      <td>-9.169425e+08</td>
      <td>-1.037400e+09</td>
      <td>-9.707739e+08</td>
      <td>-9.848396e+08</td>
      <td>4.084607e+07</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.270752</td>
      <td>0.002434</td>
      <td>0.008389</td>
      <td>0.000097</td>
      <td>4</td>
      <td>10</td>
      <td>NaN</td>
      <td>{'max_features': 4, 'n_estimators': 10}</td>
      <td>-2.666283e+09</td>
      <td>-2.784511e+09</td>
      <td>...</td>
      <td>-2.781618e+09</td>
      <td>1.268607e+08</td>
      <td>8</td>
      <td>-5.097115e+08</td>
      <td>-5.162820e+08</td>
      <td>-4.962893e+08</td>
      <td>-5.436192e+08</td>
      <td>-5.160297e+08</td>
      <td>-5.163863e+08</td>
      <td>1.542862e+07</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.807219</td>
      <td>0.004721</td>
      <td>0.023184</td>
      <td>0.000534</td>
      <td>4</td>
      <td>30</td>
      <td>NaN</td>
      <td>{'max_features': 4, 'n_estimators': 30}</td>
      <td>-2.387153e+09</td>
      <td>-2.588448e+09</td>
      <td>...</td>
      <td>-2.537883e+09</td>
      <td>1.214614e+08</td>
      <td>3</td>
      <td>-3.838835e+08</td>
      <td>-3.880268e+08</td>
      <td>-3.790867e+08</td>
      <td>-4.040957e+08</td>
      <td>-3.845520e+08</td>
      <td>-3.879289e+08</td>
      <td>8.571233e+06</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.109511</td>
      <td>0.004092</td>
      <td>0.003060</td>
      <td>0.000083</td>
      <td>6</td>
      <td>3</td>
      <td>NaN</td>
      <td>{'max_features': 6, 'n_estimators': 3}</td>
      <td>-3.119657e+09</td>
      <td>-3.586319e+09</td>
      <td>...</td>
      <td>-3.441458e+09</td>
      <td>1.893056e+08</td>
      <td>14</td>
      <td>-9.245343e+08</td>
      <td>-8.886939e+08</td>
      <td>-9.353135e+08</td>
      <td>-9.009801e+08</td>
      <td>-8.624664e+08</td>
      <td>-9.023976e+08</td>
      <td>2.591445e+07</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.368655</td>
      <td>0.004250</td>
      <td>0.008506</td>
      <td>0.000319</td>
      <td>6</td>
      <td>10</td>
      <td>NaN</td>
      <td>{'max_features': 6, 'n_estimators': 10}</td>
      <td>-2.549663e+09</td>
      <td>-2.782039e+09</td>
      <td>...</td>
      <td>-2.704645e+09</td>
      <td>1.471569e+08</td>
      <td>6</td>
      <td>-4.980344e+08</td>
      <td>-5.045869e+08</td>
      <td>-4.994664e+08</td>
      <td>-4.990325e+08</td>
      <td>-5.055542e+08</td>
      <td>-5.013349e+08</td>
      <td>3.100456e+06</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1.112361</td>
      <td>0.006665</td>
      <td>0.023521</td>
      <td>0.000904</td>
      <td>6</td>
      <td>30</td>
      <td>NaN</td>
      <td>{'max_features': 6, 'n_estimators': 30}</td>
      <td>-2.370010e+09</td>
      <td>-2.583638e+09</td>
      <td>...</td>
      <td>-2.514673e+09</td>
      <td>1.285080e+08</td>
      <td>2</td>
      <td>-3.838538e+08</td>
      <td>-3.804711e+08</td>
      <td>-3.805218e+08</td>
      <td>-3.856095e+08</td>
      <td>-3.901917e+08</td>
      <td>-3.841296e+08</td>
      <td>3.617057e+06</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.140054</td>
      <td>0.001447</td>
      <td>0.003148</td>
      <td>0.000092</td>
      <td>8</td>
      <td>3</td>
      <td>NaN</td>
      <td>{'max_features': 8, 'n_estimators': 3}</td>
      <td>-3.353504e+09</td>
      <td>-3.348552e+09</td>
      <td>...</td>
      <td>-3.348850e+09</td>
      <td>1.241939e+08</td>
      <td>13</td>
      <td>-9.228123e+08</td>
      <td>-8.553031e+08</td>
      <td>-8.603321e+08</td>
      <td>-8.881964e+08</td>
      <td>-9.151287e+08</td>
      <td>-8.883545e+08</td>
      <td>2.750227e+07</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.489684</td>
      <td>0.022850</td>
      <td>0.008984</td>
      <td>0.000885</td>
      <td>8</td>
      <td>10</td>
      <td>NaN</td>
      <td>{'max_features': 8, 'n_estimators': 10}</td>
      <td>-2.571970e+09</td>
      <td>-2.718994e+09</td>
      <td>...</td>
      <td>-2.674041e+09</td>
      <td>1.392777e+08</td>
      <td>5</td>
      <td>-4.932416e+08</td>
      <td>-4.815238e+08</td>
      <td>-4.730979e+08</td>
      <td>-5.155367e+08</td>
      <td>-4.985555e+08</td>
      <td>-4.923911e+08</td>
      <td>1.459294e+07</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1.455418</td>
      <td>0.022925</td>
      <td>0.023957</td>
      <td>0.000626</td>
      <td>8</td>
      <td>30</td>
      <td>NaN</td>
      <td>{'max_features': 8, 'n_estimators': 30}</td>
      <td>-2.357390e+09</td>
      <td>-2.546640e+09</td>
      <td>...</td>
      <td>-2.468328e+09</td>
      <td>1.091662e+08</td>
      <td>1</td>
      <td>-3.841658e+08</td>
      <td>-3.744500e+08</td>
      <td>-3.773239e+08</td>
      <td>-3.882250e+08</td>
      <td>-3.810005e+08</td>
      <td>-3.810330e+08</td>
      <td>4.871017e+06</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.080981</td>
      <td>0.002565</td>
      <td>0.003790</td>
      <td>0.000212</td>
      <td>2</td>
      <td>3</td>
      <td>False</td>
      <td>{'bootstrap': False, 'max_features': 2, 'n_est...</td>
      <td>-3.785816e+09</td>
      <td>-4.166012e+09</td>
      <td>...</td>
      <td>-3.955790e+09</td>
      <td>1.900964e+08</td>
      <td>17</td>
      <td>-0.000000e+00</td>
      <td>-0.000000e+00</td>
      <td>-0.000000e+00</td>
      <td>-0.000000e+00</td>
      <td>-0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.258792</td>
      <td>0.001661</td>
      <td>0.010075</td>
      <td>0.000710</td>
      <td>2</td>
      <td>10</td>
      <td>False</td>
      <td>{'bootstrap': False, 'max_features': 2, 'n_est...</td>
      <td>-2.810721e+09</td>
      <td>-3.107789e+09</td>
      <td>...</td>
      <td>-2.987516e+09</td>
      <td>1.539234e+08</td>
      <td>10</td>
      <td>-6.056477e-02</td>
      <td>-0.000000e+00</td>
      <td>-0.000000e+00</td>
      <td>-0.000000e+00</td>
      <td>-2.967449e+00</td>
      <td>-6.056027e-01</td>
      <td>1.181156e+00</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.106489</td>
      <td>0.001943</td>
      <td>0.003729</td>
      <td>0.000180</td>
      <td>3</td>
      <td>3</td>
      <td>False</td>
      <td>{'bootstrap': False, 'max_features': 3, 'n_est...</td>
      <td>-3.618324e+09</td>
      <td>-3.441527e+09</td>
      <td>...</td>
      <td>-3.536729e+09</td>
      <td>7.795057e+07</td>
      <td>15</td>
      <td>-0.000000e+00</td>
      <td>-0.000000e+00</td>
      <td>-0.000000e+00</td>
      <td>-0.000000e+00</td>
      <td>-6.072840e+01</td>
      <td>-1.214568e+01</td>
      <td>2.429136e+01</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.358375</td>
      <td>0.005906</td>
      <td>0.010570</td>
      <td>0.000934</td>
      <td>3</td>
      <td>10</td>
      <td>False</td>
      <td>{'bootstrap': False, 'max_features': 3, 'n_est...</td>
      <td>-2.757999e+09</td>
      <td>-2.851737e+09</td>
      <td>...</td>
      <td>-2.779924e+09</td>
      <td>6.286720e+07</td>
      <td>7</td>
      <td>-2.089484e+01</td>
      <td>-0.000000e+00</td>
      <td>-0.000000e+00</td>
      <td>-0.000000e+00</td>
      <td>-5.465556e+00</td>
      <td>-5.272080e+00</td>
      <td>8.093117e+00</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.138064</td>
      <td>0.005215</td>
      <td>0.004228</td>
      <td>0.000597</td>
      <td>4</td>
      <td>3</td>
      <td>False</td>
      <td>{'bootstrap': False, 'max_features': 4, 'n_est...</td>
      <td>-3.134040e+09</td>
      <td>-3.559375e+09</td>
      <td>...</td>
      <td>-3.305166e+09</td>
      <td>1.879165e+08</td>
      <td>12</td>
      <td>-0.000000e+00</td>
      <td>-0.000000e+00</td>
      <td>-0.000000e+00</td>
      <td>-0.000000e+00</td>
      <td>-0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.444252</td>
      <td>0.014321</td>
      <td>0.010609</td>
      <td>0.000467</td>
      <td>4</td>
      <td>10</td>
      <td>False</td>
      <td>{'bootstrap': False, 'max_features': 4, 'n_est...</td>
      <td>-2.525578e+09</td>
      <td>-2.710011e+09</td>
      <td>...</td>
      <td>-2.601969e+09</td>
      <td>1.088048e+08</td>
      <td>4</td>
      <td>-0.000000e+00</td>
      <td>-1.514119e-02</td>
      <td>-0.000000e+00</td>
      <td>-0.000000e+00</td>
      <td>-0.000000e+00</td>
      <td>-3.028238e-03</td>
      <td>6.056477e-03</td>
    </tr>
  </tbody>
</table>
<p>18 rows × 23 columns</p>
</div>



### Randomised search

Grid search approach is fine when we are exploring relatively few combinations. But when hyperparameter space is large it is often preferrable to use `RandomizedSearchCV` instead. Here instead of doing all the possible combinationes of hyperparameters, it evaluates a given number of random combinations by selecting a random value for each hyper parameter at every iteration. 

### Ensemble search

Combine models that perform best. The group or 'ensemble' will often perform better than the best individual model just like RandomForest peforms better than Decision Trees especially if we have individual models make different types of errors. 

# Step 5: Analyze the Best Models and their Errors

`RandomForestRegressor` can indicate the relative importance of each attribute for making the accurate predictions. 


```python
feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances
```




    array([7.33442355e-02, 6.29090705e-02, 4.11437985e-02, 1.46726854e-02,
           1.41064835e-02, 1.48742809e-02, 1.42575993e-02, 3.66158981e-01,
           5.64191792e-02, 1.08792957e-01, 5.33510773e-02, 1.03114883e-02,
           1.64780994e-01, 6.02803867e-05, 1.96041560e-03, 2.85647464e-03])



```python
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)
```




    [(0.36615898061813423, 'median_income'),
     (0.16478099356159054, 'INLAND'),
     (0.10879295677551575, 'pop_per_hhold'),
     (0.07334423551601243, 'longitude'),
     (0.06290907048262032, 'latitude'),
     (0.056419179181954014, 'rooms_per_hhold'),
     (0.053351077347675815, 'bedrooms_per_room'),
     (0.04114379847872964, 'housing_median_age'),
     (0.014874280890402769, 'population'),
     (0.014672685420543239, 'total_rooms'),
     (0.014257599323407808, 'households'),
     (0.014106483453584104, 'total_bedrooms'),
     (0.010311488326303788, '<1H OCEAN'),
     (0.0028564746373201584, 'NEAR OCEAN'),
     (0.0019604155994780706, 'NEAR BAY'),
     (6.0280386727366e-05, 'ISLAND')]



# Step 6: Evaluate the model on the Test Set

```python
final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)
```

```python
fig, ax = plt.subplots(1,1, figsize=(8,8))
ax.scatter(y_test, final_predictions, s=100)

lims = [np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
       ]

ax.plot(lims, lims, 'k--', linewidth=2.0, alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)

ax.set_xlabel('ML Prediction')
ax.set_ylabel('Actual Value')
```




    Text(0, 0.5, 'Actual Value')




![png](end2endML_housing_files/output_135_1.png)


```python
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_rmse)
```

    47730.22690385927


We can compute a 95% confidence interval for the test RMSE:

```python
from scipy import stats
confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
mean = squared_errors.mean()
m = len(squared_errors)

np.sqrt(stats.t.interval(confidence, m - 1,
                         loc=np.mean(squared_errors),
                         scale=stats.sem(squared_errors)))
```




    array([45685.10470776, 49691.25001878])



Alternatively, we could use a z-scores rather than t-scores:

```python
zscore = stats.norm.ppf((1 + confidence) / 2)
zmargin = zscore * squared_errors.std(ddof=1) / np.sqrt(m)
np.sqrt(mean - zmargin), np.sqrt(mean + zmargin)
```




    (45685.717918136594, 49690.68623889426)


