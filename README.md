<h1 align="center">
    <font size="30">
        <u>LSTM-XGBoost Hybrid Stock Forecasting
</u>
    </font>
</h1>

<p align="center">
  <img src="https://media.giphy.com/media/CtYFOdVbvTfgZunPEA/giphy.gif" alt="animated" width=7600", height="400"/>
</p>


<ins>**Whom of you has not thought about being a step ahead of the stock market**</ins>, using the information to gain accurate predictions for the next trading day??. Well, I am sorry to say that I am **not** going to provide the new cheatsheet of how to gain millions in the stock market through this startegy. Nonetheless, you might gain some ideas that get you into the right path in the usage of the newest technologies in time series data.
                                                                                                                
The core idea behind doing this work was **to take my knowledge one step further**. The hybrid combinations of Deep Learning models together with Decision trees or Linear Regressions are fascinating new ways to extract much more information of the raw inputs. Therefore, I took the things learned throughout the past years related to coding, statistics, ML models, DL models, Business perspectives and squeezed those into an actual _deployable_ model for real-time stock price predictions.

Note,  consider this to be the final draft of what has been very intense research on every topic treated in this work. Since it would be boring to explain every drawback and handicap faced during this time, send me a message and we can have a nice conversation, sharing our experiences on ML or DL deployments.

# Abstract
This work contains an overall analysis of the takeaways on applying a hybrid Long Short Term Memory Deep Learning Model together with an XGBoost Regressor to predict the price of the AAPL (Apple Inc.) stock for the next day (**t+1**). Throughout this work, some assumptions are made regarding the optimal number of features, some of the hyperparameter tuning (even though backtesting and tunning were done till a certain point). Consider that the expected outcome of this model should not be used as an indicator for investment decisions since the model could be refined much more and since the scope of this work was more on learning rather than profitability. I keep the priviledge to hold the full version of this trading startegy :wink:.

**Keywords: XGBoost, LSTM, Windowing, Feature Engineering, Window Optimization, Hyperparameter Tuning, Mean Absolute Error, Predictions.**

# Table of Contents
1. [Introduction](#introduction)
1. [Data](#data)
1. [Feature Engineering](#feature_engineering)
4. [LSTM-XGBoost](#lstm-xgboost)
5. [Alternatives](#alternatives)
6. [Conclusion](#conclusion)

<h1 align="center">
    <font size="30">
        <u>Introduction
</u>
    </font>
</h1>
                   
#### Introduction
                   
Is there a way to predict the unpredictable?. Certainly not, either if stock data is discrete random, the probability of exactly predicting the correct price in the future is near 0%. Nonetheless, the spread of the future price can be shrinked down into a _confidenece interval_ that tries to reduce the exposure to the risk (volatility) of the price.
                   
<h1 align="center">
    <font size="30">
        <u>Data
</u>
    </font>
</h1>

#### Data 
Apple Inc. is a publicly-traded company on the tech index NASDAQ 100. Nowadays, it is the highest valued company worldwide, with a capitalization of over 3 Billion $. To understand the selection of this asset, there is a need to point out what role the volatility of an asset plays when it comes to trading. Volatility is the standard deviation of stock prices, so if someone is pretending to trade, a higher fluctuation increases the probability of gaining more opportunities in the market, either _in the market_ or _out of the market_. Since this work faces a technical analysis, the risk of deviation of a stock (volatility) needs to be a bit higher than normal. However, this strategy could also be applied to assets that do not suffer from such high fluctuations.

<p align="center">
    <img src= "https://user-images.githubusercontent.com/67901472/152142744-c6f4a510-bbf7-4f61-98b7-14a8408d0712.png" >
</p>
<h4 align="center">
                  <u>Correlation between Technology | Health | Energy Sector & Correlation between companies (2010-2020)</u>
</h4>
<p align="center">
    <img src= "https://user-images.githubusercontent.com/67901472/152694011-837e781f-36cd-40d7-9747-f61a535f3679.png" width="800" height="400">
</p>
                                                                                                                     
To better understand the behavior of stock price data, it is necessary to get some plottings, so that we can face how the returns, the price, and the outliers perform for this specific stock.
                                                                                                                     
As seen in the histogram, we can observe that the distribution of the returns does not follow a normal distribution, represented as a black line in the plot, even though it might seem to be one (a revealing indication is the higher kurtosis and fatter tails). The good thing is that the algorithms that are going to be used in this work make no assumptions according to the [distribution of the data](https://codowd.com/bigdata/misc/Taleb_Statistical_Consequences_of_Fat_Tails.pdf). Regarding the Box Plot, we can observe a significant amount of outliers that might harm our model. Since this can later turn into an issue, it must be considered while dealing with the features. Finally, it is also interesting how the stock performed in terms of cumulative returns, as seen in the line chart, where we can observe the evolution of the stock concerning other tech giants (appended you find the annualized returns).Finally, it is always interestng to see how correlation behaves between assets and its sectors, in case one wanted to dig deeper into the fascinating world of portfolio managent.
                                                                                                                                              
<h4 align="center">
                  <u>Cumulative Distribution Functions in and out of a crash period (i.e. 2008)</u>
</h4>                                                                                                                                              
<p align="center">
    <img src= "https://user-images.githubusercontent.com/67901472/152831718-6e2755d7-0b07-4d52-98cc-f91674f34ed1.png" width="600" height = "400">
</p>


<h1 align="center">
    <font size="30">
        <u>Feature Engineering
</u>
    </font>
</h1>                                                                                                                  

#### Feature_Engineering

In this section, we will discuss the new features created in order to tackle a good performance in our model.
However, this shall be the densest section of the whole work, not only because the optimization of a model follows a cycle where you continuously adjust the features and see which one really does add value to it (entropy), this part will only cover the final outcome of what has been numerous hours of trying to optimize the input for my algorithm, avoiding so GIGO.

Since stock prices behave also according to the time of the year, a focus of interest while generating new features was to include the day, month, quarter, etc of that specific moment as there could have been patterns in the past. A simple example is that, by the end of January, a lot of *blue chips* release their quarterly earnings and since AAPL consistently does a good job on this, beating the expectations of the analysts, the stock tends to rise in a short time period (for one day to another). This is quite interesting since depending on the window optimization used for the analysis this pattern was captured or not.

Another focus of interest is the rolling window. These are useful methods when estimating the historical mean, standard deviation, quantile, or even the maximum/minimum. All these can be found in the code below.

```python
def feature_engineering(data, SPY, predictions=np.array([None])):
    
    """
    The function applies future engineering to the data in order to get more information out of the inserted data. 
    The commented code below is used when we are trying to append the predictions of the model as a new input feature to train it again. In this case it performed slightli better, however depending on the parameter optimization this gain can be vanished.
    
    """
    #if predictions.any() ==  True:
        #data = yf.download("AAPL", start="2001-11-30")
        #SPY = yf.download("SPY", start="2001-11-30")["Close"]
        #data = features(data, SPY)
        #print(data.shape)
        #data["Predictions"] = predictions
        #data["Close"] = data["Close_y"]
        #data.drop("Close_y",1,  inplace=True)
        #data.dropna(0, inplace=True)
    else:
        print("No model yet")
        data = features(data, SPY)
    return data

def features(data, SPY):
    
    for i in [2, 3, 4, 5, 6, 7]:
                            
        # Rolling Mean
        data[f"Adj_Close{i}"] = data["Adj Close"].rolling(i).mean()
        data[f"Volume{i}"] = data["Volume"].rolling(i).mean()
        
        # Rolling Standart Deviation                               
        data[f"Low_std{i}"] = data["Low"].rolling(i).std()
        data[f"High_std{i}"] = data["High"].rolling(i).std()
        data[f"Adj_CLose{i}"] = data["Adj Close"].rolling(i).std()
        
        # Stock return for the next i days
        data[f"Close{i}"] = data["Close"].shift(i)
        
        # Rolling Maximum and Minimum
        data[f"Adj_Close{i}"] = data["Adj Close"].rolling(i).max()
        data[f"Adj_Close{i}"] = data["Adj Close"].rolling(i).min()
        
        # Rolling Quantile
        data[f"Adj_Close{i}"] = data["Adj Close"].rolling(i).quantile(1)
    
    
    
    data["SPY"] = SPY
    #Decoding the time of the year
    data["Day"] = data.index.day
    data["Month"] = data.index.day
    data["Year"] = data.index.day
    data["day_year"] = data.index.day_of_year
    data["Weekday"] = data.index.weekday
                  
    #Upper and Lower shade
    data["Upper_Shape"] = data["High"]-np.maximum(data["Open"], data["Close"])
    data["Lower_Shape"] = np.minimum(data["Open"], data["Close"])-data["Low"]
    
                                                                            
    data["Close_y"] = data["Close"]
    data.drop("Close",1,  inplace=True)
    data.dropna(0, inplace=True)
    return data
```
Notice that even though this is a very small amount of features, there was some filtering applyied to it. Of course, when deploying this method in a real case scenario, it is recomendable to regularize the features and observe which ones clearly add value into the model.

<h1 align="center">
    <font size="30">
        <u>LSTM-XGBoost
</u>
    </font>
</h1> 

#### LSTM-XGBoost

Said this, lets deep into the core part of this project, where the combination between algorithms will (hopefully) provide us with reliable estimations of the Apple stock price for tomorrow.

The way to go went through an LSTM Network ensemble with a XGBoost Regressor. 

## Content:
- [UDF](#udf)
- [XGBoost](#xgboost)
- [LSTM](#lstm)
- [Hybrid Approach](#hybrid_approach)


## UDF

Main user defined functions:

The first one is used for windowing the data. Although it is explained as comments inside the function, it could be fine to go over the main functionality of this function.
Basically, this function slices the data into windows. This means that starting from a two-dimensional table having time as rows and the features as columns, this function enables to divide the data into fractions (3 dimensions). These fractions are considered windows. 
                   
Imagine you want to use the information of the last 7 days to see if they are able to predict accurately the future, so you will need to train your regressor using the input features to get the prediction for t+1. 
                   
_Windowing_ does exactly this:

 <h4 align="center">
    <font size="6">
        <u>Windowing Procedure
</u>
    </font>
</h4>
<p align="center">
  <img src="https://user-images.githubusercontent.com/67901472/152634903-84c77af7-2a5e-4f3a-8a83-4e17732a7330.gif" alt="animated" width=6000", height="500"/>
</p>

Where the larger rectangle represents the input data, using a window of two, and the smaller rectangle is the output data that we are trying to predict.
Notice that in this study the test set will be the green big rectangle since we want to estimate the unknown future value.

The other functions basically fulfill the need of splitting the data into the train and test set. However this could have been done using scikit-learn's predefined function, in order to implement the window, it was easier to code a UDF. This is also the same case for the validation split.

Moreover, since the human brain feels much more comfortable when visualizing things, it was good practice to develop a function that is able to plot, the validation set, the test set, and the prediction for t+? (**?** since you can predict for more than one period ahead). Additionally, this plot includes written conclusions, the expected price of the stock, and the spread intervals, taken from the validation performance.

```python
def windowing(train, val, WINDOW, PREDICTION_SCOPE):
    
    """
    Divides the inserted data into a list of lists. Where the shape of the data becomes and additional axe, which is time.
    Basically gets as an input shape of (X, Y) and gets returned a list which contains 3 dimensions (X, Z, Y) being Z, time.
    
    Input:
        - Train Set
        - Validation Set
        - WINDOW: the desired window
        - PREDICTION_SCOPE: The period in the future you want to analyze
        
    Output:
        - X_train: Explanatory variables for training set
        - y_train: Target variable training set
        - X_test: Explanatory variables for validation set
        - y_test:  Target variable validation set
    """  
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    for i in range(len(train)-(WINDOW+PREDICTION_SCOPE)):
        X, y = np.array(train[i:i+WINDOW, :-1]), np.array(train[i+WINDOW+PREDICTION_SCOPE, -1])
        X_train.append(X)
        y_train.append(y)

    for i in range(len(val)-(WINDOW+PREDICTION_SCOPE)):
        X, y = np.array(val[i:i+WINDOW, :-1]), np.array(val[i+WINDOW+PREDICTION_SCOPE, -1])
        X_test.append(X)
        y_test.append(y)
        
    return X_train, y_train, X_test, y_test
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
def train_test_split(data, WINDOW):
    """
    Divides the training set into train and validation set depending on the percentage indicated.
    Note this could also be done through the sklearn traintestsplit() function.
    
    Input:
        - The data to be splitted (stock data in this case)
        - The size of the window used that will be taken as an input in order to predict the t+1
        
    Output:
        - Train/Validation Set
        - Test Set
    """
    train = stock_prices.iloc[:-WINDOW]
    test = stock_prices.iloc[-WINDOW:]
    
    return train, test
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
def train_validation_split(train, percentage):
    """
    Divides the training set into train and validation set depending on the percentage indicated
    """
    train_set = np.array(train.iloc[:int(len(train)*percentage)])
    validation_set = np.array(train.iloc[int(len(train)*percentage):])
    
    
    return train_set, validation_set
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
def plotting(y_val, y_test, pred_test, mae, WINDOW, PREDICTION_SCOPE):
    
    """This function returns a graph where:
        - Validation Set
        - Test Set
        - Future Prediction
        - Upper Bound
        - Lower Bound
    """
    
    ploting_pred = [y_test[-1], pred_test]
    ploting_test = [y_val[-1]]+list(y_test)

    time = (len(y_val)-1)+(len(ploting_test)-1)+(len(ploting_pred)-1)

    test_time_init = time-(len(ploting_test)-1)-(len(ploting_pred)-1)
    test_time_end = time-(len(ploting_pred)-1)+1

    pred_time_init = time-(len(ploting_pred)-1)
    pred_time_end = time+1

    x_ticks = list(stock_prices.index[-time:])+[stock_prices.index[-1]+timedelta(PREDICTION_SCOPE+1)]

    values_for_bounds = list(y_val)+list(y_test)+list(pred_test)
    upper_band = values_for_bounds+mae
    lower_band = values_for_bounds-mae
    
    print(f"For used windowed data: {WINDOW}")
    print(f"Prediction scope for date {x_ticks[-1]} / {PREDICTION_SCOPE+1} days")
    print(f"The predicted price is {str(round(ploting_pred[-1][0],2))}$")
    print(f"With a spread of MAE is {round(mae,2)}")
    print()
    
    plt.figure(figsize=(16, 8))

    plt.plot(list(range(test_time_init, test_time_end)),ploting_test, marker="$m$", color="orange")
    plt.plot(list(range(pred_time_init, pred_time_end)),ploting_pred,marker="$m$", color="red")
    plt.plot(y_val, marker="$m$")

    plt.plot(upper_band, color="grey", alpha=.3)
    plt.plot(lower_band, color="grey", alpha=.3)

    plt.fill_between(list(range(0, time+1)),upper_band, lower_band, color="grey", alpha=.1)

    plt.xticks(list(range(0-1, time)), x_ticks, rotation=45)
    plt.text(time-0.5, ploting_pred[-1]+2, str(round(ploting_pred[-1][0],2))+"$", size=11, color='red')
    plt.title(f"Target price for date {x_ticks[-1]} / {PREDICTION_SCOPE+1} days, with used past data of {WINDOW} days and a MAE of {round(mae,2)}", size=15)
    plt.legend(["Testing Set (input for Prediction)", "Prediction", "Validation"])
    plt.show()
    
    print()
    print("-----------------------------------------------------------------------------")
    print()
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
def inverse_transformation(X, y, y_hat):
    
    """
    This function serves to inverse the rescaled data. 
    There are two ways in which this can happen:
        - There could be the conversion for the validation data to see it on the plotting.
        - There could be the conversion for the testing data, to see it plotted.
    """
    if X.shape[1]>1:
        new_X = []

        for i in range(len(X)):
            new_X.append(X[i][0])
            
        new_X = np.array(new_X)
        y = np.expand_dims(y, 1)
        
        new_X = pd.DataFrame(new_X)
        y = pd.DataFrame(y)
        y_hat = pd.DataFrame(y_hat)

        real_val = np.array(pd.concat((new_X, y), 1))
        pred_val = np.array(pd.concat((new_X, y_hat), 1))
        
        real_val = pd.DataFrame(scaler.inverse_transform(real_val))
        pred_val = pd.DataFrame(scaler.inverse_transform(pred_val))
        
    else:       
        X = X.reshape(X.shape[0], X.shape[1]*X.shape[2])
        
        new_X = pd.DataFrame(X)
        y = pd.DataFrame(y)
        y_hat = pd.DataFrame(y_hat)
        y_hat = pd.concat((y, y_hat))
        y_hat.index = range(len(y_hat))
        
        real_val = np.array(pd.concat((new_X, y), 1))
        pred_val = np.array(pd.concat((new_X, y_hat), 1))
        
        pred_val = pd.DataFrame(scaler.inverse_transform(pred_val))
        real_val = pd.DataFrame(scaler.inverse_transform(real_val))
        
    return real_val, pred_val
```

<h1 align="center">
    <font size="30">
        <u>XGBoost
</u>
    </font>
</h1> 

##### XGBoost
XGBoost, is one of the most highly used supervised ML algorithms nowadays, as it uses a more optimized way to implement a tree-based algorithm, and it is also able to efficiently manage large and complex datasets.

The methodology followed by this algorithm is the following. XGBoost uses a Greedy algorithm for the building of its tree, meaning it uses a simple intuitive way to optimize the algorithm. The algorithm combines its best model, with previous ones, and so minimizes the error. So, in order to constantly select the models that are actually improving its performance, a target is settled. This target will depend on how much the next model has decreased the error, if there has been no change in the error the target will be set to zero, otherwise, it will set really high as surpassing its performance can be difficult for the next model.

For more insights into how this algorithm works, check out this video from [StatQuest](https://www.youtube.com/watch?v=OtD8wVaFm6E&t=649s)

#### Training the Model

For training the model with the best hyperparameters and with the optimal windowing (use of past input data), a time series cross-validation was done on the data. The difference between the conventional cross-validation method is that you must ensure that the algorithm does not randomly take samples of the data to see its performance, since past data is somehow related to future events.
So, there was a need to code a _user defined GridsearchCV_, this could be done through [_ParameterGrid_](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ParameterGrid.html), where you insert a dictionary of parameters, and this function makes subsets including all combinations of the different parameters.
Notice that the more parameters you insert and depending on how you cross-validate (backtest) the data, it is computationally expensive, therefore when implementing this, take into account what are the benefits and the drawbacks of every approach.

###### Hyperparamter_optimization
<p align="center">
    <img src= "https://user-images.githubusercontent.com/67901472/152366099-d36fe0ba-483c-4a63-9b4e-8a314ed32be7.png" width ="500" height="350">
    <img src= "https://user-images.githubusercontent.com/67901472/152366615-d2ca6258-f522-49d0-8685-f9933faf8eff.png", width="500" height="350">
</p>

Lets dig into the code:

```python
PERCENTAGE = .995
WINDOW = 2
PREDICTION_SCOPE = 0
```
The algorithm will use the past two trading days (**WINDOW=2**) in order to predict the next day (**PREDICTION_SCOPE=0**), using almost the whole data for training, but the last month (except the last _WINDOW_ days), which is used for validation.

```python
stock_prices = feature_engineering(stock_prices, SPY)
```
```
Output:
-->No model yet
```
Load the data with its features.

After that, it is time to settle the train, validation, and test set. As said before the division will be done according to the selected percentage. Regarding the test set, it will make use of the settled _WINDOW_ to get the **X_test** values.

```python
train, test = train_test_split(stock_prices, WINDOW)
train_set, validation_set = train_validation_split(train, PERCENTAGE)

print(f"train_set shape: {train_set.shape}")
print(f"validation_set shape: {validation_set.shape}")
print(f"test shape: {test.shape}")
```
```
Output:
-->(5047, 50)
-->(26, 50)
-->(2, 50)
```
We apply the windowing [udf](#udf)
```python
X_train, y_train, X_val, y_val = windowing(train_set, validation_set, WINDOW, PREDICTION_SCOPE)

#Convert the returned list into arrays
X_train = np.array(X_train)
y_train = np.array(y_train)
X_val = np.array(X_val)
y_val = np.array(y_val)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"y_val shape: {y_val.shape}")
```
```
Output:
-->X_train shape: (5045, 2, 49)
-->y_train shape: (5045,)
-->X_val shape: (24, 2, 49)
-->y_val shape: (24,)
```

Since the XGBoost algorithm does not allow a three-dimensional input, there is a need to reshape the data into two dimensions. The idea behind the reshape is to join the rows of the windowed days into one big input. In this example, since our **WINDOW=2**, we are going to return the same amount of rows, but instead of only having 49 columns we will multiply this quantity by the **WINDOW** size.

```python
X_train = X_train.reshape(X_train.shape[0], -1)
X_val = X_val.reshape(X_val.shape[0], -1)

print(f"X_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")
```
```
Output:
-->X_train shape: (5045, 98)
-->X_val shape: (24, 98)
```

Finally we only have to train the algorithm with the organized data:
(The hyperparameters where optimized using the approach located to the left of the training model [image](#hyperparamter_optimization)
```python
xgb_model = xgb.XGBRegressor(gamma=1, n_estimators=200)
xgb_model.fit(X_train,y_train)
```


Condensed into a function:
```python
def xgb_model(X_train, y_train, X_val, y_val, plotting=False):

    """
    Trains a preoptimized XGBoost model and returns the Mean Absolute Error an a plot if needed
    """     
    xgb_model = xgb.XGBRegressor(gamma=1, n_estimators=200)
    xgb_model.fit(X_train,y_train)
    
    pred_val = xgb_model.predict(X_val)
    mae = mean_absolute_error(y_val, pred_val)

    if plotting == True:
        
        plt.figure(figsize=(15, 6))
        
        sns.set_theme(style="white")
        sns.lineplot(range(len(y_val)), y_val, color="grey", alpha=.4)
        sns.lineplot(range(len(y_val)),pred_val, color="red")

        plt.xlabel("Time")
        plt.ylabel("AAPL stock price")
        plt.title(f"The MAE for this period is: {round(mae, 3)}")
    
    return  mae, xgb_model
```

<p align="center">
    <img src= "https://user-images.githubusercontent.com/67901472/152218072-d0bbd0b0-7f59-449d-87c3-04dfb711763f.png" >
</p>
<p align="center">
    <img src= "https://user-images.githubusercontent.com/67901472/152699067-ce7e94ad-f25b-46ba-a5cb-727c754a1769.png" width="600" height="400">
</p>

<h6>
    <font size=1>*Since the window size os 2, the feauture performance considers twice the features, meaning, if there are 50 features, <u>f97 == f47</u> or <u>likewise f73 == f23</u>.</font>
</h6>

#### Add the predictions (if needed)

Sometimes, it is also interesting, to use the training/validation predictions as a new feature, this will be used to be more accurate while passing to the test set. in this case, there was no clear evidence that this approach improves the performance, maybe more tryouts are needed in order to get nice outcomes. Nonetheless, find attached the code of how a prediction feature could be added to your model:
```python
#try:
    #y_hat_train = np.expand_dims(xgb_model.predict(X_train), 1)
    #array = np.empty((stock_prices.shape[0]-y_hat_train.shape[0], 1))
    #array[:] = np.nan
    #predictions = np.concatenate((array, y_hat_train))
#except NameError:
    #print("No Model")
    
#new_stock_prices = feature_engineering(stock_prices, SPY, predictions=predictions)

#train, test = train_test_split(new_stock_prices, WINDOW)

#train_set, validation_set = train_validation_split(train, PERCENTAGE)
#X_train, y_train, X_val, y_val = windowing(train_set, validation_set, WINDOW, PREDICTION_SCOPE)

#Reshaping the data
#X_train = np.array(X_train)
#y_train = np.array(y_train)

#X_val = np.array(X_val)
#y_val = np.array(y_val)

#X_train = X_train.reshape(X_train.shape[0], -1)
#X_val = X_val.reshape(X_val.shape[0], -1)

#new_mae, new_xgb_model = xgb_model(X_train, y_train, X_val, y_val, plotting=True)

#print(new_mae)
```

#### Testing the Model

Of course, even if all the obtained results seem to be nice, it is important to see the model performing in a real-life situation. For this, let us obtain the prediction for the next day **t+1**.

To get the predictions, the same approach as for the train and validation is required.

```ptyhon
X_test = np.array(test.iloc[:, :-1])
y_test = np.array(test.iloc[:, -1])
X_test = X_test.reshape(1, -1)

print(f"X_test shape: {X_test.shape}")
```
```
Output:
-->X_test shape: (1, 98)
```

Lets predict and plot the results:
```python
#Apply the xgboost model on the Test Data
pred_test_xgb = xgb_model.predict(X_test)
plotting(y_val, y_test, pred_test_xgb, mae, WINDOW, PREDICTION_SCOPE)
```
<p align="center">
    <img src= "https://user-images.githubusercontent.com/67901472/152218913-d46d6ed6-3623-4720-a8af-84d75054e0d2.png">
</p>

#### Saving the XGBoost parameters for future usage
```python
#joblib.dump(xgb_model, "XGBoost.pkl")
```
#### Window and Percentage Optimization

```python
plots = {}
for window in [1, 2, 3, 4, 5, 6, 7, 10, 20, 25, 30, 35]:
    
    for percentage in [.92, .95, .97, .98, .99, .995]:

        WINDOW = window
        PREDICTION_SCOPE = 0
        PERCENTAGE = percentage

        train = stock_prices.iloc[:int(len(stock_prices))-WINDOW]
        test = stock_prices.iloc[-WINDOW:]
        
        train_set, validation_set = train_validation_split(train, PERCENTAGE)

        X_train, y_train, X_val, y_val = windowing(train_set, validation_set, WINDOW, PREDICTION_SCOPE)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        X_val = np.array(X_val)
        y_val = np.array(y_val)

        X_test = np.array(test.iloc[:, :-1])
        y_test = np.array(test.iloc[:, -1])

        X_train = X_train.reshape(X_train.shape[0], -1)
        try:
            X_val = X_val.reshape(X_val.shape[0], -1)
            X_test = X_test.reshape(1, -1)
        except ValueError:
            break

        xgb_model = xgb.XGBRegressor(gamma=1)
        xgb_model.fit(X_train, y_train)

        pred_val = xgb_model.predict(X_val)

        mae = mean_absolute_error(y_val, pred_val)

        pred_test = xgb_model.predict(X_test)
        plotii= [y_test[-1], pred_test]

        plots[str(window)+str(pred_scope)] = [y_val, y_test, pred_test, mae, WINDOW, PREDICTION_SCOPE, PERCENTAGE]
        
print()
print(plots["20"])
```
```
Output:
-->[array([179.28999329, 179.38000488, 178.19999695, 177.57000732,
       182.00999451, 179.69999695, 174.91999817, 172.        ,
       172.16999817, 172.19000244, 175.08000183, 175.52999878,
       172.19000244, 173.07000732, 169.80000305, 166.22999573,
       164.50999451, 162.41000366, 161.61999512, 159.77999878,
       159.69000244, 159.22000122, 170.33000183, 174.77999878]), 
       array([174.61000061, 174.47009277]), array([171.02782], dtype=float32), 3.8374627431233725, 2, #0, 0.995]
```
Extract the most optimized paramters:
```python
window_optimization(plots)
```
See the multiple perfromances:
```python
for key in list(plots.keys())[5:9]:
    plotting(plots[key][0], plots[key][1], plots[key][2], plots[key][3], plots[key][4], plots[key][5])
```
<p align="center">
    <img src= "https://user-images.githubusercontent.com/67901472/152638506-babdd1e9-0fb6-404e-8428-8d37298f0cec.png" width ="500" height="350">
    <img src= "https://user-images.githubusercontent.com/67901472/152638510-48ae4a91-9e09-4640-ad43-c02ccf21164b.png", width="500" height="350">
    <img src= "https://user-images.githubusercontent.com/67901472/152638525-619d0bf0-89a9-46ae-bfb2-4254b9917ff8.png", width="500" height="350">
    <img src= "https://user-images.githubusercontent.com/67901472/152638547-82b303e3-5933-444c-b1f7-9da44ac7b00e.png", width="500" height="350">
</p>

<h1 align="center">
    <font size="30">
        <u>Long-Short term Memory
</u>
    </font>
</h1> 

###### LSTM

Long Short Term Memory or LSTM is a type of Neural Network, which was developed on the basis provided by the Recurrent Neural Networks, or **RNN**. The structure of the LSTM layer can be visualized in the image below:

<p align="center">
    <img src= "https://user-images.githubusercontent.com/67901472/152638045-fe9c9538-ee48-4908-bd34-bbd258cac7ef.png" width ="670" height="470">
    <img src= "https://user-images.githubusercontent.com/67901472/152218220-1010ad55-4342-410d-b795-442db442cdb6.png", width="600" height="330">
</p>

To get more clarifications on the syntax and the math behind this algorithm, I encourage you to do this course of [DeepLearning.AI](https://www.coursera.org/specializations/deep-learning).

The algorithms main usage falls into NLPs or Time Series, and the main idea behind this is that instead of only processing the information they receive from the previous neuron and applying the activation function from scratch (as the RNN does), they actually divide the neuron into three main parts from which to set up the input from the next layer of neurons: Learn, Unlearn and Retain gate.

This method ensures that you are using the information given from previous data and the data returned from a neural that is in the same layer, to get the input for the next neuron.

While training the Apple series, several combinations of algorithms were used, whether RNNs, CNNs, or NNs, however when it comes to time series, the **LSTM** has a significant advantage over its predecessor the **RNNs**. For those of you who might be familiar with these Neural Networks, **RNNs** had a scaling effect on the gradients when the weights (W) were either very low or very high, leading to no change in the loss or an extreme change. In order to fix this, the LSTM was created, which basically, thanks to the different _gates_ that are used in each node, they are able to omit this radical change making the difference more stable (reducing the likelihood of vanishing gradients). If there is an interest to dig further in the update from an **RNN** to **LSTM**, visit [GeeksforGeeks](https://www.geeksforgeeks.org/understanding-of-lstm-networks/).

Nonetheless, there was the need to go from a simpler model to a more complex one, in the end, the LSTM returned the most optimal performance. 

Notice that using the LSTM implies more computation costs, slower training, etc

For the sake of optimization, parameter tunning was needed, this entailed: the input and hidden layer size, the batch_size, the number of epochs, and the rolling window size for the analysis.


```python
#Parameters for the LSTM
PERCENTAGE = .98 #Split train/val and test set
CALLBACK = .031 #Used to stop training the Network when the MAE from the validation set reached a perormance below 3.1%
BATCH_SIZE = 20 #Number of samples that will be propagated through the network. I chose almost a trading month
EPOCH = 50 #Settled to train the model
WINDOW_LSTM = 30 #The window used for the input data
PREDICTION_SCOPE = 0 #How many period to predict, being 0=1
```
Once settled the optimal values, the next step is to split the dataset:
```ptyhon
train_lstm, test_lstm = train_test_split(stock_prices, WINDOW_LSTM)
train_split_lstm, validation_split_lstm = train_validation_split(train_lstm, PERCENTAGE)

train_split_lstm = np.array(train_split_lstm)
validation_split_lstm = np.array(validation_split_lstm)
```

##### Rescaling to train the LSTM

To improve the performance of the network, the data has to be rescaled. This is mainly due to the fact that when the data is in its original format, the loss function might adopt a shape that is far difficult to achieve its minimum, whereas, after rescaling the global minimum is easier achievable (moreover you avoid stagnation in local minimums). For this study, the MinMax Scaler was used. The algorithm rescales the data into a range from 0 to 1. The drawback is that it is sensitive to outliers.

What is important to consider is that the fitting of the scaler has to be done on the training set only since it will allow transforming the validation and the test set compared to the train set, without including it in the rescaling. This is especially helpful in time series as several values do increase in value over time.

```python
scaler = MinMaxScaler()
scaler.fit(train_split_lstm)

train_scale_lstm = scaler.transform(train_split_lstm)
val_scale_lstm = scaler.transform(validation_split_lstm)
test_scale_lstm = scaler.transform(test_lstm)

print(train_scale_lstm.shape)
print(val_scale_lstm.shape)
print(test_scale_lstm.shape)
```
```
Output:
-->(4938, 50)
-->(101, 50)
-->(30, 50)
```

Now lets window the data for further procedure. Since NN allow to ingest multidimensional input, there is no need to rescale the data before training the net.
```python
model_lstm = lstm_model(X_train_lstm, y_train_lstm, X_val_lstm, y_val_lstm, EPOCH, BATCH_SIZE, CALLBACK, plotting=True)

X_train_lstm = np.array(X_train_lstm)
y_train_lstm = np.array(y_train_lstm)

X_val_lstm = np.array(X_val_lstm)
y_val_lstm = np.array(y_val_lstm)

X_test_lstm = np.array(test_scale_lstm[:, :-1])
y_test_lstm = np.array(test_scale_lstm[:, -1])

print(X_train_lstm.shape)
print(X_val_lstm.shape)
print(X_test_lstm.shape)
```
```
Output:
-->(4908, 30, 49)
-->(71, 30, 49)
-->(30, 49)
```

Now is the moment where our data is prepared to be trained by the algorithm:
Some comments:
* The first lines of code are used to clear the memory of the Keras API, being especially useful when training several times a model, adjusting the hyperparameters so that one training is not influenced by the other.
* There was a need to create a _callback_ class, which stops the iteration over the epochs when the loss function achieves a certain level of performance
* The optimal approach for this time series was through a neural network of one input layer, two LSTM hidden layers, and an output layer or Dense layer.
- Each hidden layer has 32 neurons, which tends to be defined as related to the number of observations in our dataset.
- For the input layer, it was necessary to define the input shape, which basically considers the window size and the number of features
* For the sake of optimization a **Stochastic Gradient Descent** was used with a momentum of .85. Moreover a **learning rate scheduler** was coded, that aimed to return the best performing learning rate for this series. Before settling down the final value, the neural net was trained on a small number of epochs, in order to see was shall be the best number (the last stable value of the loss curve, is a reference).
* For the compiler, the Huber loss function was used to not punish the outliers excessively and the metrics, through which the entire analysis is based is the Mean Absolute Error.
* Finally, when fitting the model: 
- A batch size of 20 was used, as it represents approximately one trading month. The batch size is the subset of the data that is taken from the training data to run the neural network.
- The number of epochs sum up to 50, as it equals the number of exploratory variables.
- The callback was settled to 3.1%, which indicates that the algorithm will stop running when the loss for the validation set undercut this predefined value. This means that the data has been trained with a spread of below 3%.


```python
tf.keras.backend.clear_session()
tf.random.set_seed(51)
np.random.seed(51)

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get("val_mae")<CALLBACK):
            print("\n Accuracy reached %, so cancelling training")
            self.model.stop_training=True

callbacks = myCallback()

 model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1)
])

#lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    #lambda epoch: 0.228 * 10**(epoch / 20))
optimizer = tf.keras.optimizers.SGD(learning_rate=0.228, momentum =.85)
model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer, metrics="mae")
history = model.fit(X_train, y_train,batch_size=BATCH_SIZE, epochs=EPOCH,callbacks=[callbacks],  validation_data=[X_val, y_val], verbose=1)
```
```
Output:
Epoch 1/30
328/328 [==============================] - 6s 13ms/step - loss: 5.1591e-04 - mae: 0.0161 - val_loss: 0.0046 - val_mae: 0.0819
Epoch 2/30
328/328 [==============================] - 4s 11ms/step - loss: 1.1767e-04 - mae: 0.0099 - val_loss: 0.0036 - val_mae: 0.0696
Epoch 3/30
328/328 [==============================] - 4s 11ms/step - loss: 9.9146e-05 - mae: 0.0093 - val_loss: 0.0024 - val_mae: 0.0530
Epoch 4/30
328/328 [==============================] - 4s 12ms/step - loss: 9.9358e-05 - mae: 0.0092 - val_loss: 0.0019 - val_mae: 0.0452
Epoch 5/30
328/328 [==============================] - 4s 11ms/step - loss: 7.3892e-05 - mae: 0.0077 - val_loss: 0.0012 - val_mae: 0.0355
Epoch 6/30
328/328 [==============================] - 4s 11ms/step - loss: 7.3492e-05 - mae: 0.0078 - val_loss: 0.0012 - val_mae: 0.0343
Epoch 7/30
328/328 [==============================] - 4s 12ms/step - loss: 6.5693e-05 - mae: 0.0073 - val_loss: 0.0025 - val_mae: 0.0554
Epoch 8/30
328/328 [==============================] - 6s 17ms/step - loss: 6.7699e-05 - mae: 0.0074 - val_loss: 0.0017 - val_mae: 0.0418
Epoch 9/30
328/328 [==============================] - 7s 21ms/step - loss: 6.3419e-05 - mae: 0.0074 - val_loss: 0.0012 - val_mae: 0.0361
Epoch 10/30
328/328 [==============================] - 5s 15ms/step - loss: 5.7910e-05 - mae: 0.0068 - val_loss: 0.0012 - val_mae: 0.0347
Epoch 11/30
328/328 [==============================] - 5s 15ms/step - loss: 5.7053e-05 - mae: 0.0068 - val_loss: 0.0024 - val_mae: 0.0539
Epoch 12/30
328/328 [==============================] - 5s 14ms/step - loss: 5.3417e-05 - mae: 0.0065 - val_loss: 0.0011 - val_mae: 0.0335
Epoch 13/30
328/328 [==============================] - 5s 15ms/step - loss: 5.8176e-05 - mae: 0.0069 - val_loss: 0.0014 - val_mae: 0.0377
Epoch 14/30
328/328 [==============================] - 5s 14ms/step - loss: 5.0530e-05 - mae: 0.0063 - val_loss: 0.0020 - val_mae: 0.0504
Epoch 15/30
327/328 [============================>.] - ETA: 0s - loss: 5.0619e-05 - mae: 0.0063
 Accuracy reached %, so cancelling training
328/328 [==============================] - 5s 15ms/step - loss: 5.0593e-05 - mae: 0.0063 - val_loss: 8.6126e-04 - val_mae: 0.0298
```
![image](https://user-images.githubusercontent.com/67901472/152655042-135f8f4e-788f-4678-94e2-e32c12107cc3.png)

Notice that the loss curve is pretty stable after the initial sharp decrease at the very beginnign (first epochs), showing that there is no evidence the data is overfitted

Nonetheless the loss function seems extraordinarily low, one has to consider that the data was rescaled. In order to defined the real loss on the data one has to **inverse transform** the input into its original shape. This is done with the _inverse_transformation_ [UDF](#udf).
```python
#Set up predictions for train and validation set
y_hat_lstm = model_lstm.predict(X_val_lstm)
y_hat_train_lstm = model_lstm.predict(X_train_lstm)

#Validation Transormation
mae_lstm = mean_absolute_error(y_hat_lstm, y_hat_lstm)
real_val, pred_val = inverse_transformation(X_val_lstm, y_val_lstm, y_hat_lstm)
mae_lstm = mean_absolute_error(real_val.iloc[:, 49], pred_val.iloc[:, 49])
```
Validation Set:
<p align="center">
    <img src= "https://user-images.githubusercontent.com/67901472/152655326-f1e32a49-b2a6-46ad-a38d-11a1d375e220.png">
</p>

```python
real_train, pred_train = inverse_transformation(X_train_lstm, y_train_lstm, y_hat_train_lstm)
```
Training Set:
<p align="center">
    <img src= "https://user-images.githubusercontent.com/67901472/152655365-74743dca-69ce-48b5-a233-a30f36ddd006.png">
</p>

```python
tf.keras.backend.clear_session()
tf.random.set_seed(51)
np.random.seed(51)

def lstm_model(X_train, y_train, X_val, y_val, EPOCH,BATCH_SIZE,CALLBACK,  plotting=False):
    
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get("val_mae")<CALLBACK):
                print("\n Accuracy % so cancelling training")
                self.model.stop_training=True

    callbacks = myCallback()
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1)
    ])

    lr_schedule = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: 0.228 * 10**(epoch / 20))
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.228, momentum =.85)
    model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer, metrics="mae")
    history = model.fit(X_train, y_train,batch_size=BATCH_SIZE, epochs=EPOCH,callbacks=[callbacks],  validation_data=[X_val, y_val], verbose=1)
    
    if plotting == True:
        plt.figure(figsize=(18, 6))

        lrs = 1e-5 * (10 ** (np.arange(len(history.history["loss"])) / 20))
        plt.semilogx(lrs, history.history["loss"])
        plt.xticks(size=14)
        plt.show()
    
    return model
```

#### Testing the Model

Lets obtain the prediction fot **t+1**:

```python
X_test_formula = X_test_lstm.reshape(X_test_lstm.shape[0], 1, X_test_lstm.shape[1])
print(X_test_formula.shape)
```
```
Output:
-->(30, 1, 49)# 30 rows (window=30) | 1 Cluster of Rows | 49 features
```

See that the shape is not what we want, since there should only be 1 row, that entails a window of 30 days with 49 features. That is why there is a neeed to reshape this array.

```python
X_test_lstm = X_test_formula.reshape(1, X_test_formula.shape[0], X_test_formula.shape[2])
print(X_test_lstm.shape)
```
```
Output:
-->(30, 1, 49)# 30 rows (window=30) | 1 Cluster of Rows | 49 features
```

Now we can predict:

```python
y_hat_test_lstm = model_lstm.predict(X_test_lstm)
real_test, pred_test = inverse_transformation(X_test_lstm, y_test_lstm, y_hat_test_lstm)

#For plotting purposes
y_val_lstm = np.array(real_val.iloc[-30:, 49])
y_test_lstm = np.array(real_test.iloc[:, 49])
pred_test = np.array(pred_test.iloc[-1:, 49])
mae_lstm = mean_absolute_error(real_val.iloc[:, 49], pred_val.iloc[:, 49])

#UDF
plotting(y_val_lstm, y_test_lstm, pred_test, mae_lstm, WINDOW_LSTM, PREDICTION_SCOPE)
```

<p align="center">
    <img src= "https://user-images.githubusercontent.com/67901472/152655690-0c72b4e0-4a1e-4906-921f-874e7547d482.png">
</p>

#### Saving the LSTM parameters for transfer learning

```python
#model_lstm.save('./LSTM')
#lstm_model = tf.keras.models.load_model("LSTM") //in case you want to load it
```

<h1 align="center">
    <font size="30">
        <u>Ensemble Modelling
</u>
    </font>
</h1> 

###### Hybrid_Approach

In order to get the most out of the two models, a good practice is to combine those two and apply a higher weight on the model which got a lower loss function (mean absolute error). The reason is mainly that sometimes a neural network performs really well on the loss function, but when it comes to a real-life situation, the algorithm only learns the shape of the original data and copies this with one delay (+1 lag). Combining this with a decision tree regressor might mitigate this duplicate effect.

![image](https://user-images.githubusercontent.com/67901472/152656757-33f6745e-9406-4789-8d74-f1af0837b2a1.png)

In our case we saw that the MAE of the LSTM was lower than the one from the XGBoost, therefore we will give a higher weight on the predictions returned from the LSTM model.

```python
mae_xgboost = mae
print(f"The LSTM prediction is: {pred_test})
print(f"The XGBoost prediction is: {pred_test_xgb})
```
```
Output:
-->The LSTM prediction is: array([158.05817538])
-->The XGBoost prediction is: array([157.07529])
```

Lets apply the above defined formula formula:

```python
prediction_ensemble = predictions(mae_lstm, mae_xgboost, pred_test_xgb, pred_test)
avg_mae = (mae_lstm + mae_xgboost)/2
plotting(y_val, y_test, prediction_ensemble, avg_mae, WINDOW, PREDICTION_SCOPE)
```
<p align="center">
    <img src= "https://user-images.githubusercontent.com/67901472/152657105-c66ce817-dfb3-4c42-88b5-94cf43ed1172.png">
</p>



<h1 align="center">
    <font size="30">
        <u>Alternatives
</u>
    </font>
</h1> 

##### Alternatives

Driving into the end of this work, you might ask why don't use simpler models in order to see if there is a way to benchmark the selected algorithms in this study.
So, for this reason, several _simpler_ machine learning models were applied to the stock data, and the results might be a bit confusing.

```python
#Linear Regression
lr = LinearRegression()
lr.fit(X_train_reg, y_train_reg)
y_hat_lr= lr.predict(X_val_reg)
mae_lr = mean_absolute_error(y_val_reg, y_hat_lr)

print("MSE: {}".format(np.mean((y_hat_lr-y_val_reg)**2)))
print("MAE: {}".format(mae_lr))
```
```
Output: 
-->MSE:14.15..
-->MAE:2.97...
```
```python
#Random Forest Regressor
rf = RandomForestRegressor()
rf.fit(X_train_reg, y_train_reg)
y_hat_rf= rf.predict(X_val_reg)
mae_rf= mean_absolute_error(y_val_reg, y_hat_rf)

print("MSE: {}".format(np.mean((y_hat_rf-y_val_reg)**2)))
print("MAE: {}".format(mae_rf))
```
```
Output: 
-->MSE:94.60..
-->MAE:8.59...
```

Focusing just on the results obtained, you should question why on earth using a more complex algorithm as LSTM or XGBoost it is. Well, the answer can be seen when plotting the predictions:

<p align="center">
    <img src= "https://user-images.githubusercontent.com/67901472/152657200-2003893d-49e3-4a33-952a-9daa6dca81a5.png">
</p>

```python
pred_test_lr = lr.predict(X_test_reg)
plotting(y_val_reg, y_test_reg, pred_test_lr, mae_lr, WINDOW, PREDICTION_SCOPE)
```

<p align="center">
    <img src= "https://user-images.githubusercontent.com/67901472/152657325-87e3ed8f-0b9f-454c-9f05-a716c6e89e4a.png">
</p>

See that the outperforming algorithm is the Linear Regression, with a very small error rate. Nonetheless, as seen in the graph the predictions seem to replicate the validation values but with a lag of one (remeber this happened also in the LSTM for small batch sizes). So, if we wanted to proceed with this one, a good approach would also be to embed the algorithm with a different one. This would be good practice as you do not further rely on a unique methodology.


# Conclusion

Reaching the end of this work, there are some key points that should be mentioned in the wrap up:

The first thing is that this work has more about self development, and the posibility to connect with people who might work on similar projects and want to enagage with, than to obtain skyrocketing profits. Of course, if the algorithm would have work nearly without an error, I would not share my results publicy. Anayway, one can build up really interesting stuff on the code provided in this work.

The second thing is that the selection of the embedding algorithms might not be the optimal choice, but as said in point one, the indention was to learn, not to get the highest returns. Lerning about the most used tree-based regressor and Neural Networks are two very interesting topics that will help me in future projects, those will have more a focus on computer vision and image recognition.

Regarding hyperparameter optimzation, someone has to face sometimes the limits of its hardware while trying to estimate the best performing parameters for its machine learning algorithm. Nonetheless, I pushed the limits to balance my resources for a good performing model.

When it comes to feature engineering, I was able to play around with the data and see if there is more information to extract, and as I said in the study, this is in most of the cases where ML Engineers and Data Scientists probably spend the most of their time. Whether it is because of outlier processing, missing values, encoders or just model performance optimization, one can spend several weeks/months trying to identify the best possible combination.











