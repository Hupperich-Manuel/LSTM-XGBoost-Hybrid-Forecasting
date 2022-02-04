<h1 align="center">
    <font size="30">
        <u>LSTM-XGBoost Hybrid Stock Forecasting
</u>
    </font>
</h1>

<p align="center">
  <img src="https://media.giphy.com/media/CtYFOdVbvTfgZunPEA/giphy.gif" alt="animated" width=7600", height="400"/>
</p>


Whom of you has not thought about being a step ahead the stock market, using information in a way that it returns accurate predictions for the next trading day. Well, since this might be true in some cases this is really far from the scope of this work.
The idea behind this work was to take my knowledge one step further. The hybrid combinations of Deep Learning models together with Decision trees or Linear Regressions are fascinating new ways to extract much more information of the raw inputs. Therefore, I took the things learned throughout the past years related to coding, statistics, ML models, DL models, Business Perspective, and squeezed those into an actual _deployable_ model for a real time stock price predictions.

Note that this can be considered to be the final draft of what has been a very intenses research on every topic trated in this work. Since, it would be boring to explain every drawback and handicap faced during this time, send me a message and we can have a nice conversation, sharing our experiences on ML or DL deployments.

# Abstract
This work contains an overall analysis of the takeaways on applying a hybrid Long Short Term Memory Deep Learning Model together with an XGBoost Regressor to predict the price of the AAPL (Apple Inc.) stock for the next day (**t+1**). Throughout this work, some assumptions are made regarding the optimal number of features, some of the hyperparameter tuning (even though backtesting and tunning was done till a certain point). Notice that the expected outcome of this model should not be used as an indicator for investment decisions since the model could be refined much more, and since the scope of this work was more on learning rather than profitability.

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
                   
Is there a way to predict the unpredictable?. Certainly not, either if stock data is discrete random, the probability of exactly predicting the correct price in the future is near 0%. Nonetheless, the spread of the future price can be shrinked down into a _confidenece interval_ that tries to reduce the risk (volatility) of the price.
                   
<h1 align="center">
    <font size="30">
        <u>Data
</u>
    </font>
</h1>

#### Data 
Apple Inc. is a publicy traded company on the tech index NASDAQ 100. Nowadays it is the highest valued company worldwide, with a capitalization of over 3 Billion $. To justify the selection of this stock, there is a need to point out what role the volatility of an asset plays when it come to trading. Volatility os the standart deviation of a stock, so, if someone is pretending to trade on a price, a higher fluctuation increases the probability of gaining more opportunities in the market, either _in the market_ or _out the market_. Since this work faces a technical analysis, the risk of deviation of a stock (volatitlity) needs to be a bit higher than normal. However, this startegy could also be applyed on assets that do not suffer from such high fluctuations.

<p align="center">
    <img src= "https://user-images.githubusercontent.com/67901472/152142744-c6f4a510-bbf7-4f61-98b7-14a8408d0712.png" >
</p>
To get an idea of how stock data behaves, it is necesary to get some plottings, so that we can face how the returns, the price and the outliers perform for this specific stock.
                                                                                                                     
As seen in the histogram, we can observe that the distribution of the returns does not follow a normal distribution, represented as a black line in the plot, even though it might seem to be one (a revealing indication is the higher kurtosis and fatter tails). The good thing is that the algorithms that are going to be used in this work, make no assumptions according to the [distribution of the data](https://codowd.com/bigdata/misc/Taleb_Statistical_Consequences_of_Fat_Tails.pdf). Regarding the Box Plot, we can observe that there is a significant amount of outliers that might harm our model. This is an issue which musst be considered while dealing with the features. Finally, it is also interesting how the stock performed in terms of cumulative returns, as seen in the line chart, where we can observe the evolution of the stock repsect to other tech gigants (appended you find the annualized returns).


<h1 align="center">
    <font size="30">
        <u>Feature Engineering
</u>
    </font>
</h1>                                                                                                                  

#### Feature_Engineering

In this section we will discuss the new features created in order to tackle a good performance in our model.
However this might be the longest section of the whole work, not only because the optimization of a model follows a cycle where you continuously adjust the features and see which one really do add value to it (entropy), this part will only cover a small part of this process, focusing only on some of the features that where used for the training process.

Since stock prices behave also according to the time of the year, a focus of interest while generating new features was to include the day, month , quarter, etc of that specific moment as there could have been patterns in the past. An simple example is that, by the end of January, a lot of *blue chips* release their quarterly earnings, and since AAPL consistently does a good job on this, beating the expectations of the analysts, the stock tend to rise oin a short time period (for one day to another). This is quite interesting, since depending on the window optimization used for the analysis this pattern was captured or not.

Another focus of interest are the rolling windows. These are methods that are usefull when estimating the historical mean, standart deviation, quantile, or even the maximum/minimum. All these can be found in the code below.

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
    
    
    #FEATURE ENGINEERING
    data["SPY"] = SPY
    data["Day"] = data.index.day
    data["Month"] = data.index.day
    data["Year"] = data.index.day
    data["day_year"] = data.index.day_of_year
    data["Weekday"] = data.index.weekday
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

Said this, let sdive deep into the core part of this project, where the combination between algorithms will (hopefully) provide us with reliable estimations of the Apple stock price for tomorrow.

In this sections we will make use of some user defined functions that mainly try to automitize the optimization, interpretation of the applyied models whether through a windowiing functions, graphs or comparisons.

## Content:
- [UDF](#udf)
- [XGBoost](#xgboost)
- [LSTM](#lstm)
- [Hybrid Approach](#hybrid_approach)


## UDF

Main user defined functions:

The first one is used for windowing the data. Although it is explained as comments inside the function, it could be fine to go over the main functionality of this function.
Basically, this function slices the data into windows. This means that starting from a two dimensional table having time as rows and the features as columns, thanks to this method we are able to get only fractions of this data. These fractions are the considered windows. 
Imagin you want to use the information of the last 7 days to see if they are able to predict accurately the future, so you will need to train your regressor using the input features to get the prediction for t+1. _Windowing_ does exactly this:

 <h4 align="center">
    <font size="6">
        <u>Windowing Procedure
</u>
    </font>
</h4>
<p align="center">
    <img src= "https://user-images.githubusercontent.com/67901472/152358117-f1e77e90-6fec-452a-92ec-2e6be6c05c22.png" width="420", height="470">
</p>

Where the larger rectangle represent the input data, using a eindow of two, and the smaller rectangle is the output data which we are trying to predict.
Notice that in this study the test set will be the green big rectangle, since we want to estimate the unknown future value.

The other functions basically fulfill the need of splitting the data into the train and test set. However this could have been dne using scikit-learn's predefined function, in order to implement the window, it was easier to code a udf. This is also the same case for the validation split.

Moreover, since the human brain feels much more comfortable when visualizing things, it was good practice to develop a function which is able to plot, the validation set, the test set and the prediction for t+? (**?** since you can predict for more than one period ahead). Aditionally, this plot includes written conclusions, the expected price of the stock and the spread intervals, taken from the validation performance.

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
```



## XGBoost
XGBoost, is one of the most highly used supervised ML algorithms nowadays, as it uses a more optimized way to implement a tree based algorithm, and it is also able to efficiently manage large and complex datasets.

The methodology followed by this algorithm is the following. XGBoost uses a Greedy algorithm for the building of its tree, meaning it uses a simple intuitive way to optimze the algorithm. The algorithm combines its best model, with previous ones, and so minimizes the error. So, in order to constantly select the models that are actually imporving its performance, a target is settled. and this target will depend on how much the next model has decreased the error, if there was an increase or no change in the error ythe target will be set to zero, otherwise it will set really high since it is difficult to surpas the performance of the previous model.

For more insights into how this algorithm works, check out this video from [StatQuest](https://www.youtube.com/watch?v=OtD8wVaFm6E&t=649s)

The approach to train the model started by settling some assumptions:

- The windowed data uses 2

#### Training the Model

For training the model with the best hyperparameters and with the optimal windowing (use of past input data), a time series cross validation was doen on the data. The difference to conventional cross validation method is that you must ensure that the algorithm does not randomly take samples of the data to see its performance, since past data is somehow related to future events.
So, there was a need to code a _user defined GridsearchCV_, thhis could be done through [_ParameterGrid_](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ParameterGrid.html), where you insert a dictiory of parameters, and this function makes subsets including all combination of the different parameters.
Notice that the more paramnters you insert and dpeending on how you crossvalidate (backtest) the data, it is computationally expensive, therefore when implementing this, take into account what are the benefits and the drawbacks of every approach.

<p align="center">
    <img src= "https://user-images.githubusercontent.com/67901472/152366099-d36fe0ba-483c-4a63-9b4e-8a314ed32be7.png" width ="500" height="350">
    <img src= "https://user-images.githubusercontent.com/67901472/152366615-d2ca6258-f522-49d0-8685-f9933faf8eff.png", width="500" height="350">
</p>



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

#### Testing the Model

<p align="center">
    <img src= "https://user-images.githubusercontent.com/67901472/152218913-d46d6ed6-3623-4720-a8af-84d75054e0d2.png">
</p>


## LSTM

```python
def lstm_model(X_train, y_train, X_val, y_val, EPOCH,BATCH_SIZE,CALLBACK,  plotting=False):
    
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get("val_mae")<CALLBACK):
                print("\n Accuracy % so cancelling training")
                self.model.stop_training=True

    callbacks = myCallback()
    
    tf.keras.backend.clear_session()
    tf.random.set_seed(51)
    np.random.seed(51)
    
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

<p align="center">
    <img src= "https://user-images.githubusercontent.com/67901472/152218220-1010ad55-4342-410d-b795-442db442cdb6.png" width="550", height="250">
</p>



### Assumptions



## Hybrid_Approach

# Alternatives

Driving into the end of this work, you might ask why dont using simpler models in order to see if there is a way to benchmark the selected algorithms in this study.
So, for this reason several _simpler_ machine learning models where applyied on the stock data, and the results might be a bit confusing.

```python
#Linear Regression
lr = LinearRegression()
lr.fit(X_train_reg, y_train_reg)
y_hat_lr= lr.predict(X_val_reg)
mae_lr = mean_absolute_error(y_val_reg, y_hat_lr)

print("MSE: {}".format(np.mean((y_hat_lr-y_val_reg)**2)))
print("MAE: {}".format(mae_lr))

#Output: 
#>MSE:14.15..
#>MAE:2.97...

#Random Forest Regressor
rf = RandomForestRegressor()
rf.fit(X_train_reg, y_train_reg)
y_hat_rf= rf.predict(X_val_reg)
mae_rf= mean_absolute_error(y_val_reg, y_hat_rf)

print("MSE: {}".format(np.mean((y_hat_rf-y_val_reg)**2)))
print("MAE: {}".format(mae_rf))

#Output: 
#>MSE:94.60..
#>MAE:8.59...
```

Focusing just on the results obtained, you should question why on earth using a more complex algorithm as LSTM or XGBoost it is. Well the answer can be seen when plotting the predictions:





# Conclusion



