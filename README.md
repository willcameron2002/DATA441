# DATA441

## Hi! I'm Will Cameron
I am a senior at William & Mary majoring in Data Science with a minor in Mathematics. I enjoy machine learning and have an interest in sports statistics. When I'm not programming, I love to play guitar and hang out with friends.

## Homework/Projects
[Homework 1](https://github.com/willcameron2002/DATA441/blob/main/Homework/Project1_WillCameron.ipynb)

Below is my project 2:

```python
# Libraries of functions need to be imported
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist
from sklearn.utils.validation import check_is_fitted
```

## Question 1

Create your class that implements the Gradient Boosting concept, based on the locally weighted regression method (Lowess class), and that allows a user-prescribed number of boosting steps. The class you develop should have all the mainstream useful options, including “fit,” “is_fitted”,  and “predict,” methods.  Show applications with real data for regression, 10-fold cross-validations and compare the effect of different scalers, such as the “StandardScaler”, “MinMaxScaler”, and the “QuantileScaler”.  In the case of the “Concrete” data set, determine a choice of hyperparameters that yield lower MSEs for your method when compared to the eXtream Gradient Boosting library.


```python
# Gaussian Kernel
def Gaussian(w):
  return np.where(w>4,0,1/(np.sqrt(2*np.pi))*np.exp(-1/2*w**2))

# Tricubic Kernel
def Tricubic(w):
  return np.where(w>1,0,70/81*(1-w**3)**3)

# Quartic Kernel
def Quartic(w):
  return np.where(w>1,0,15/16*(1-w**2)**2)

# Epanechnikov Kernel
def Epanechnikov(w):
  return np.where(w>1,0,3/4*(1-w**2))
```


```python
class GradBoostedLowess():

  def __init__(self, kernel = Gaussian, tau = .02):
    self.tau = tau
    self.kernel = kernel

  def fit(self, x, y):
      kernel = self.kernel
      tau = self.tau
      self.xtrain_ = x
      self.yhat_ = y

  def is_fitted(self):
    if self.xtrain_ != None:
      return True
    else:
      return False

  def predict(self, x_new, boosts = 0):
      if boosts == 0:
        return self.single_predict(x_new)
      else:
        final_preds = np.zeros(x_new.shape[0])
        resids = self.yhat_
        for i in range(boosts):
          #model = GradBoostedLowess()
          #model.fit(self.xtrain_, resids)
          self.fit(self.xtrain_, resids)
          #new_preds = model.single_predict(self.xtrain_)
          new_preds = self.single_predict(self.xtrain_)
          #final_preds = final_preds + model.single_predict(x_new)
          final_preds = final_preds + self.single_predict(x_new)
          resids = resids - new_preds

        #model = GradBoostedLowess()
        #model.fit(self.xtrain_, resids)
        self.fit(self.xtrain_, resids)
        #new_preds = model.single_predict(x_new)
        new_preds = self.single_predict(x_new)
        final_preds = final_preds + new_preds
        return final_preds

  def single_predict(self, x_new):
      check_is_fitted(self)
      x = self.xtrain_
      y = self.yhat_
      lm = linear_model.Ridge(alpha=0.001)
      w = self.kernel(cdist(x, x_new, metric='euclidean')/(2*self.tau))

      if np.isscalar(x_new):
        lm.fit(np.diag(w)@(x.reshape(-1,1)),np.diag(w)@(y.reshape(-1,1)))
        yest = lm.predict([[x_new]])[0][0]
      else:
        n = len(x_new)
        yest_test = []
        #Looping through all x-points
        for i in range(n):
          lm.fit(np.diag(w[:,i])@x,np.diag(w[:,i])@y)
          yest_test.append(lm.predict([x_new[i]]))
      return np.array(yest_test).flatten()

```


```python
data = pd.read_csv('drive/MyDrive/Adv. App. Machine Learning/concrete.csv')
```


```python
data.head()
```





  <div id="df-88f91ebe-8f1b-4119-872f-7f23e48a7449" class="colab-df-container">
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
      <th>cement</th>
      <th>slag</th>
      <th>ash</th>
      <th>water</th>
      <th>superplastic</th>
      <th>coarseagg</th>
      <th>fineagg</th>
      <th>age</th>
      <th>strength</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>540.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>162.0</td>
      <td>2.5</td>
      <td>1040.0</td>
      <td>676.0</td>
      <td>28</td>
      <td>79.99</td>
    </tr>
    <tr>
      <th>1</th>
      <td>540.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>162.0</td>
      <td>2.5</td>
      <td>1055.0</td>
      <td>676.0</td>
      <td>28</td>
      <td>61.89</td>
    </tr>
    <tr>
      <th>2</th>
      <td>332.5</td>
      <td>142.5</td>
      <td>0.0</td>
      <td>228.0</td>
      <td>0.0</td>
      <td>932.0</td>
      <td>594.0</td>
      <td>270</td>
      <td>40.27</td>
    </tr>
    <tr>
      <th>3</th>
      <td>332.5</td>
      <td>142.5</td>
      <td>0.0</td>
      <td>228.0</td>
      <td>0.0</td>
      <td>932.0</td>
      <td>594.0</td>
      <td>365</td>
      <td>41.05</td>
    </tr>
    <tr>
      <th>4</th>
      <td>198.6</td>
      <td>132.4</td>
      <td>0.0</td>
      <td>192.0</td>
      <td>0.0</td>
      <td>978.4</td>
      <td>825.5</td>
      <td>360</td>
      <td>44.30</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-88f91ebe-8f1b-4119-872f-7f23e48a7449')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-88f91ebe-8f1b-4119-872f-7f23e48a7449 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-88f91ebe-8f1b-4119-872f-7f23e48a7449');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-b71a4410-7786-4762-ad60-9559287c11bb">
  <button class="colab-df-quickchart" onclick="quickchart('df-b71a4410-7786-4762-ad60-9559287c11bb')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-b71a4410-7786-4762-ad60-9559287c11bb button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
x = data.drop(columns = ['strength']).values
y = data['strength'].values
```


```python
mse_SScale = []
mse_MMScale = []
mse_QScale = []
mse_XGBoost = []
SScale = StandardScaler()
MMScale = MinMaxScaler()
QScale = QuantileTransformer(n_quantiles = 300)
kf = KFold(n_splits = 10, shuffle = True, random_state = 7)

model_XG = XGBRegressor(objective ='reg:squarederror',n_estimators=100,reg_lambda=20,alpha=1,gamma=10,max_depth=3)
model_SS = GradBoostedLowess(tau=.3)
model_MM = GradBoostedLowess(tau=.3)
model_QS = GradBoostedLowess(tau=.3)

for idxtrain, idxtest in kf.split(x):
  xtrain = x[idxtrain]
  ytrain = y[idxtrain].ravel()
  ytest = y[idxtest].ravel()
  xtest = x[idxtest]

  xtrain_S = SScale.fit_transform(xtrain)
  xtest_S = SScale.transform(xtest)
  xtrain_M = MMScale.fit_transform(xtrain)
  xtest_M = MMScale.transform(xtest)
  xtrain_Q = QScale.fit_transform(xtrain)
  xtest_Q = QScale.transform(xtest)

  model_SS.fit(xtrain_S, ytrain)
  S_pred = model_SS.predict(xtest_S, boosts = 3)
  model_MM.fit(xtrain_M, ytrain)
  M_pred = model_MM.predict(xtest_M, boosts = 3)
  model_QS.fit(xtrain_Q, ytrain)
  Q_pred = model_QS.predict(xtest_Q, boosts = 3)
  model_XG.fit(xtrain_S, ytrain)
  X_pred = model_XG.predict(xtest_S)

  mse_SScale.append(mse(ytest, S_pred))
  mse_MMScale.append(mse(ytest,M_pred))
  mse_QScale.append(mse(ytest,Q_pred))
  mse_XGBoost.append(mse(ytest,X_pred))

print('The Cross-validated Mean Squared Error for StandardScaler is: '+str(np.mean(mse_SScale)))
print('The Cross-validated Mean Squared Error for MinMaxScaler is: '+str(np.mean(mse_MMScale)))
print('The Cross-validated Mean Squared Error for QuantileTransformer is: '+str(np.mean(mse_QScale)))
print('The Cross-validated Mean Squared Error for XGBoost method: '+str(np.mean(mse_XGBoost)))
```

    The Cross-validated Mean Squared Error for StandardScaler is: 40.28905266637976
    The Cross-validated Mean Squared Error for MinMaxScaler is: 47.60993849907447
    The Cross-validated Mean Squared Error for QuantileTransformer is: 27.37296985378024
    The Cross-validated Mean Squared Error for XGBoost method: 21.823977435435875
    


```python
mse_GBLow = []
mse_XGBoost = []
scale = StandardScaler()

kf = KFold(n_splits = 10, shuffle = True, random_state = 7)
x = data.drop(columns = ['strength']).values
y = data['strength'].values

model_XG = XGBRegressor(objective ='reg:squarederror',n_estimators=100,reg_lambda=20,alpha=1,gamma=10,max_depth=3)
model = GradBoostedLowess(tau=5, kernel = Gaussian)

for idxtrain, idxtest in kf.split(x):
  xtrain = x[idxtrain]
  ytrain = y[idxtrain].ravel()
  ytest = y[idxtest].ravel()
  xtest = x[idxtest]

  xtrain = scale.fit_transform(xtrain)
  xtest = scale.transform(xtest)

  model.fit(xtrain_S, ytrain)
  pred = model.predict(xtest, boosts = 10)
  model_XG.fit(xtrain_S, ytrain)
  X_pred = model_XG.predict(xtest)

  mse_GBLow.append(mse(ytest, pred))
  mse_XGBoost.append(mse(ytest,X_pred))

print('The Cross-validated Mean Squared Error for our class is: '+str(np.mean(mse_GBLow)))
print('The Cross-validated Mean Squared Error for XGBoost method: '+str(np.mean(mse_XGBoost)))
```

    The Cross-validated Mean Squared Error for our class is: 139.94009515907564
    The Cross-validated Mean Squared Error for XGBoost method: 147.83071136774723
    

Hooray! We found hyperparameters for our model that yield a lower MSE than the XGBoost method

## Question 2

Based on the Usearch library, create your own class that computes the k_Nearest Neighbors for Regression.


```python
!pip install usearch
```

    Requirement already satisfied: usearch in /usr/local/lib/python3.10/dist-packages (2.9.0)
    Requirement already satisfied: numpy in /usr/local/lib/python3.10/dist-packages (from usearch) (1.23.5)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.10/dist-packages (from usearch) (4.66.2)
    


```python
import usearch
from usearch.index import search, MetricKind, Matches, BatchMatches
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
```


```python
class K_Nearest_Neighbors():

  def __init__(self, k):
    self.k = k

  def fit(self, x, y):
    self.x = x
    self.y = y
    self.full_set =  np.array(zip(x, y))

  def predict(self, x_new):
    predictions = []
    for row in x_new:
      neighbors = self._get_neighbors(row)
      row_preds = []
      for index in neighbors:
        row_preds.append(self.y[index])
      predictions.append(Counter(row_preds).most_common(1)[0][0])
    return predictions

  def _dist_calc(self, row):
    distances: Matches = search(self.x, row, self.x.shape[0], MetricKind.L2sq, exact=True)
    return distances

  def _get_neighbors(self, new_row):
    neighbors = []
    distances = self._dist_calc(new_row)
    for i in range(self.k):
      neighbors.append(distances.to_list()[i][0])
    return neighbors
```


```python
data = pd.read_csv('drive/MyDrive/Adv. App. Machine Learning/mobile.csv')
```


```python
data
```





  <div id="df-2ecb860d-1fe1-4eb0-a122-832a5d38fffa" class="colab-df-container">
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
      <th>battery_power</th>
      <th>blue</th>
      <th>clock_speed</th>
      <th>dual_sim</th>
      <th>fc</th>
      <th>four_g</th>
      <th>int_memory</th>
      <th>m_dep</th>
      <th>mobile_wt</th>
      <th>n_cores</th>
      <th>...</th>
      <th>px_height</th>
      <th>px_width</th>
      <th>ram</th>
      <th>sc_h</th>
      <th>sc_w</th>
      <th>talk_time</th>
      <th>three_g</th>
      <th>touch_screen</th>
      <th>wifi</th>
      <th>price_range</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>842</td>
      <td>0</td>
      <td>2.2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>7</td>
      <td>0.6</td>
      <td>188</td>
      <td>2</td>
      <td>...</td>
      <td>20</td>
      <td>756</td>
      <td>2549</td>
      <td>9</td>
      <td>7</td>
      <td>19</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1021</td>
      <td>1</td>
      <td>0.5</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>53</td>
      <td>0.7</td>
      <td>136</td>
      <td>3</td>
      <td>...</td>
      <td>905</td>
      <td>1988</td>
      <td>2631</td>
      <td>17</td>
      <td>3</td>
      <td>7</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>563</td>
      <td>1</td>
      <td>0.5</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>41</td>
      <td>0.9</td>
      <td>145</td>
      <td>5</td>
      <td>...</td>
      <td>1263</td>
      <td>1716</td>
      <td>2603</td>
      <td>11</td>
      <td>2</td>
      <td>9</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>615</td>
      <td>1</td>
      <td>2.5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>10</td>
      <td>0.8</td>
      <td>131</td>
      <td>6</td>
      <td>...</td>
      <td>1216</td>
      <td>1786</td>
      <td>2769</td>
      <td>16</td>
      <td>8</td>
      <td>11</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1821</td>
      <td>1</td>
      <td>1.2</td>
      <td>0</td>
      <td>13</td>
      <td>1</td>
      <td>44</td>
      <td>0.6</td>
      <td>141</td>
      <td>2</td>
      <td>...</td>
      <td>1208</td>
      <td>1212</td>
      <td>1411</td>
      <td>8</td>
      <td>2</td>
      <td>15</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
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
      <td>...</td>
    </tr>
    <tr>
      <th>1995</th>
      <td>794</td>
      <td>1</td>
      <td>0.5</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0.8</td>
      <td>106</td>
      <td>6</td>
      <td>...</td>
      <td>1222</td>
      <td>1890</td>
      <td>668</td>
      <td>13</td>
      <td>4</td>
      <td>19</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1996</th>
      <td>1965</td>
      <td>1</td>
      <td>2.6</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>39</td>
      <td>0.2</td>
      <td>187</td>
      <td>4</td>
      <td>...</td>
      <td>915</td>
      <td>1965</td>
      <td>2032</td>
      <td>11</td>
      <td>10</td>
      <td>16</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1997</th>
      <td>1911</td>
      <td>0</td>
      <td>0.9</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>36</td>
      <td>0.7</td>
      <td>108</td>
      <td>8</td>
      <td>...</td>
      <td>868</td>
      <td>1632</td>
      <td>3057</td>
      <td>9</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1998</th>
      <td>1512</td>
      <td>0</td>
      <td>0.9</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>46</td>
      <td>0.1</td>
      <td>145</td>
      <td>5</td>
      <td>...</td>
      <td>336</td>
      <td>670</td>
      <td>869</td>
      <td>18</td>
      <td>10</td>
      <td>19</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1999</th>
      <td>510</td>
      <td>1</td>
      <td>2.0</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>45</td>
      <td>0.9</td>
      <td>168</td>
      <td>6</td>
      <td>...</td>
      <td>483</td>
      <td>754</td>
      <td>3919</td>
      <td>19</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>2000 rows × 21 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-2ecb860d-1fe1-4eb0-a122-832a5d38fffa')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-2ecb860d-1fe1-4eb0-a122-832a5d38fffa button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-2ecb860d-1fe1-4eb0-a122-832a5d38fffa');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-83025f11-400a-43a3-bfef-1bccc623076d">
  <button class="colab-df-quickchart" onclick="quickchart('df-83025f11-400a-43a3-bfef-1bccc623076d')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-83025f11-400a-43a3-bfef-1bccc623076d button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

  <div id="id_c90b3781-3d54-48bc-8ef7-19de20d43d0c">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('data')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_c90b3781-3d54-48bc-8ef7-19de20d43d0c button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('data');
      }
      })();
    </script>
  </div>

    </div>
  </div>





```python
x = data.drop(columns = ['price_range']).values
y = data['price_range'].values
```


```python
scale = StandardScaler()

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, random_state = 7)
xtrain = scale.fit_transform(xtrain)
xtest = scale.transform(xtest)

model = K_Nearest_Neighbors(k=250)
model.fit(xtrain, ytrain)
pred = model.predict(xtest)
print("The accuracy is: " + str(accuracy_score(ytest, pred)))
```

    The accuracy is: 0.72
    

## Question 3

Host your project on your GitHub page.

https://willcameron2002.github.io/DATA441/
