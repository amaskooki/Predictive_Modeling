
# Introduction:
The objective is to develop predictive models for Spasm in patients. The folloqing provides the results for analysis of spike rate in different regions of the brain vs. patients state divided into 3 category: rest, voluntary movement, spasm, obtained through analysis of the real time video recordings. The data set contains 185193 spikes. 

Data Dictionary:

* PatientID: Identification number assigned to each subject for privacy 
* ChannelName: The file name assigned to the recording channel
* NeuronID: An integer assigned to every neuron identified at each channel starting from 1 and increasing
* SpikeTime: The time of recording of the spike. (Marco can elaborate further on the time format)
* SpikeRate: Calculated as the inverse time between every spike and subsequent spike
* SpikeEventType: is a categorical variable indicating the state of the patient at the time of spike recording. The variable takes the following value:
    + ‘REST’: Indicates patient in resting or sleeping state (no observable movements)
    + 'SPASM_B': Indicates patient has visible spasm on BOTH sides
    + 'SPASM_L': Indicates patient has visible spasm on the LEFT sides
    + 'SPASM_R': Indicates patient has visible spasm on the RIGHT sides
    + 'Voluntary_B': Indicates patient has visible voluntary movement on BOTH sides
    + 'Voluntary_L': Indicates patient has visible voluntary movement on the LEFT sides
    + 'Voluntary_R’: Indicates patient has visible voluntary movement on the RIGHT sides



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
import re

import warnings
warnings.filterwarnings('ignore')
```

    C:\ProgramData\Anaconda3\lib\site-packages\statsmodels\compat\pandas.py:56: FutureWarning: The pandas.core.datetools module is deprecated and will be removed in a future version. Please use the pandas.tseries module instead.
      from pandas.core import datetools
    


```python
# Load Database
orig_df = pd.read_csv('redacted_spike_data.csv')
len(orig_df)
```




    185193




```python
# Inspect it
orig_df.head()
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
      <th>Index</th>
      <th>PatientID</th>
      <th>ChannelName</th>
      <th>NeuronID</th>
      <th>SpikeTime</th>
      <th>SpikeRate</th>
      <th>EventTypeSide</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>microVIM_L_1</td>
      <td>1</td>
      <td>10.583409</td>
      <td>0.000000</td>
      <td>REST</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>microVIM_L_1</td>
      <td>1</td>
      <td>24.655727</td>
      <td>0.071061</td>
      <td>REST</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1</td>
      <td>microVIM_L_1</td>
      <td>1</td>
      <td>66.921773</td>
      <td>0.023660</td>
      <td>SPASM_B</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>1</td>
      <td>microVIM_L_1</td>
      <td>1</td>
      <td>67.628909</td>
      <td>1.414154</td>
      <td>SPASM_B</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>1</td>
      <td>microVIM_L_1</td>
      <td>1</td>
      <td>68.318864</td>
      <td>1.449371</td>
      <td>SPASM_B</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = orig_df.copy()
df.drop('Index', axis=1,inplace=True)
```

# Feature Extraction

### Get brain region, lateral and contact number


```python
def getRegion(e):
    match = re.search(r'micro([a-zA-Z]+)_',e)
    if match:
        return match.group(1)
    else:
        return
def getLateral(e):
    match = re.search(r'micro([a-zA-Z]+)_([LR])_',e)
    if match:
        return match.group(2)
    else:
        return
def getCono(e):
    match = re.search(r'micro([a-zA-Z]+)_([LR])_(\d+)',e)
    if match:
        return match.group(3)
    else:
        return

df['Recording_Region'] = df['ChannelName'].map(getRegion)
df['Recording_Lateral'] = df['ChannelName'].map(getLateral)
df['Recording_ContactNo'] = df['ChannelName'].map(getCono)
df.drop('ChannelName', axis=1, inplace=True)
```

### Extract event type and lateral 


```python
def getEvent(e):
    match = re.search(r'([a-zA-Z]+)_*',e)
    if match:
        return match.group(1)
    else:
        return
def getEventLateral(e):
    match = re.search(r'[a-zA-Z]+_*([LRB]*)',e)
    if match:
        return match.group(1)
    else:
        return
    
df['Event'] = df['EventTypeSide'].map(getEvent)
df['Event_Lateral'] = df['EventTypeSide'].map(getEventLateral)

df = df[(~df['Event'].isnull()) & ~(df['Event'] == 'None')]
```


```python
df.head()
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
      <th>PatientID</th>
      <th>NeuronID</th>
      <th>SpikeTime</th>
      <th>SpikeRate</th>
      <th>EventTypeSide</th>
      <th>Recording_Region</th>
      <th>Recording_Lateral</th>
      <th>Recording_ContactNo</th>
      <th>Event</th>
      <th>Event_Lateral</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>10.583409</td>
      <td>0.000000</td>
      <td>REST</td>
      <td>VIM</td>
      <td>L</td>
      <td>1</td>
      <td>REST</td>
      <td></td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>24.655727</td>
      <td>0.071061</td>
      <td>REST</td>
      <td>VIM</td>
      <td>L</td>
      <td>1</td>
      <td>REST</td>
      <td></td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>66.921773</td>
      <td>0.023660</td>
      <td>SPASM_B</td>
      <td>VIM</td>
      <td>L</td>
      <td>1</td>
      <td>SPASM</td>
      <td>B</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>67.628909</td>
      <td>1.414154</td>
      <td>SPASM_B</td>
      <td>VIM</td>
      <td>L</td>
      <td>1</td>
      <td>SPASM</td>
      <td>B</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>68.318864</td>
      <td>1.449371</td>
      <td>SPASM_B</td>
      <td>VIM</td>
      <td>L</td>
      <td>1</td>
      <td>SPASM</td>
      <td>B</td>
    </tr>
  </tbody>
</table>
</div>




```python
p = sns.FacetGrid(df, hue='Event', aspect=4)
p.map(sns.distplot, 'SpikeRate')
plt.xlim([0,100])
p.add_legend()

```




    <seaborn.axisgrid.FacetGrid at 0x229552d1b70>




![png](output_11_1.png)


The spike rates distribution is very skewed. Also, we consider values over 80 as artifact due to the nature of the phenomenon.
First, values over 80 will be removed,


```python
df = df[ (1 < df['SpikeRate'])]
```


```python
df['logRate'] = np.log(df['SpikeRate'])
```


```python
p = sns.FacetGrid(df, aspect=3)
p.map(sns.distplot, 'logRate')
plt.xlim([0,6])
p.add_legend()
plt.tight_layout()
```


![png](output_15_0.png)


Even logRate is still skewed. There might be noise in lower rates as well.

# Data Exploration 


```python
df_L = df[(df['Recording_Lateral']=='L') & (df['Event_Lateral'].isin(['R','','B']))]
df_R = df[(df['Recording_Lateral']=='R') & (df['Event_Lateral'].isin(['L','','B']))]



fig, (axis1,axis2) = plt.subplots(1,2,figsize=(18,6))

rateplot = sns.barplot(x='Recording_Region', y='SpikeRate', hue='Event',data=df_L, capsize=.1, ax=axis1)
axis1.set_title('Left Brain')
axis1.legend(loc='upper right')

rateplot = sns.barplot(x='Recording_Region', y='SpikeRate', hue='Event',data=df_R, capsize=.1, ax=axis2)
axis2.set_title('Right Brain')


```




    Text(0.5,1,'Right Brain')




![png](output_18_1.png)


Left brain seems to be more active during rest in VIM, VoaVop and VPLa. VoaVop is very active during muscle SPASM. (These two patients have injuries at left brain)

# Predictive models


```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC,LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
```


```python
map_region = {'STN':0, 'VIM':1, 'VPLa':2, 'VoaVop':3}
X_train = df_L[['Recording_Region', 'logRate']]
X_train.loc[:,'Recording_Region'] = X_train['Recording_Region'].map(map_region)

y_train = df_L['Event'].map({'REST':0, 'SPASM':1, 'Voluntary':2})

```


```python
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_train['Recording_Region'], X_train['logRate'], y_train, c=y_train)
ax.set_xlabel('Region')
ax.set_ylabel('log rate')
ax.set_zlabel('State')
ax.view_init(30, 185)
plt.show()

```


![png](output_23_0.png)



```python
kfold = StratifiedKFold(n_splits=10)
# Modeling step Test differents algorithms 
random_state = 2
classifiers = []
classifiers.append(LogisticRegression(random_state = random_state))
classifiers.append(SVC(random_state=random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(GaussianNB())
classifiers.append(LinearSVC(random_state=random_state))
classifiers.append(DecisionTreeClassifier(random_state = random_state))
classifiers.append(RandomForestClassifier(random_state = random_state))

cv_results = []
for classifier in classifiers :
    cv_results.append(cross_val_score(classifier, X_train, y = y_train, scoring = "accuracy", cv = kfold, n_jobs=4))

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":['Logistic Regression','SVMC', 'KNN', 
     'Gaussian', 'linear SVC', 'Decision Tree', 'Random Forest']})
```


```python
cv_res
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
      <th>Algorithm</th>
      <th>CrossValMeans</th>
      <th>CrossValerrors</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Logistic Regression</td>
      <td>0.695025</td>
      <td>0.086504</td>
    </tr>
    <tr>
      <th>1</th>
      <td>SVMC</td>
      <td>0.587468</td>
      <td>0.190190</td>
    </tr>
    <tr>
      <th>2</th>
      <td>KNN</td>
      <td>0.655968</td>
      <td>0.137251</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Gaussian</td>
      <td>0.624109</td>
      <td>0.209573</td>
    </tr>
    <tr>
      <th>4</th>
      <td>linear SVC</td>
      <td>0.690725</td>
      <td>0.074735</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Decision Tree</td>
      <td>0.594778</td>
      <td>0.202726</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Random Forest</td>
      <td>0.594312</td>
      <td>0.205178</td>
    </tr>
  </tbody>
</table>
</div>




```python
#g = sns.barplot(x='CrossValMeans',y='Algorithm',data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})

cv_res[['Algorithm', 'CrossValMeans']].plot.bar(x='Algorithm',y='CrossValMeans', colormap='plasma');
plt.xlabel("Mean Accuracy")
plt.title("Cross validation scores")
plt.tight_layout()
```


![png](output_26_0.png)


It seems linear models are doing a good job predicting state of the patient. 
