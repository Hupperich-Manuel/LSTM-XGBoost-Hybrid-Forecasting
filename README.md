# LSTM-XGBoost Hybrid Stock Forecasting

Whom of you has not thought about making money training a model that returns accurate predictions on t+1, so that one can gain from this hail grail of information?. Well, since this might be true in some cases this is really far from the scope of this work.
The idea behind this work is to take my knowledge one step further. The hybrid combinations of Deep Learning models together with Decision trees or Linear Regressions are fascinating new ways to extract much more information of the raw inputs one is dealing with. Therefore, I found my opportunity to combine things learned throughout the past years related to coding, statistics, ML models, DL models, Business Perspective, and squeeze those to return an actual _deployable_ model for a real time application.


# Abstract
This work contains an overall analysis of the takeaways on applying a hybrid Long Short Term Memory Deep Learning Model together with an XGBoost Regressor to predict the price of the AAPL (Apple Inc.) stock for the next day (**t+1**). Throughout this work, some assumptions are made regarding the optimal number of features, some of the hyperparameter tuning (even though backtesting and tunning was done till a certain point). Notice that the expected outcome of this model should not be used as an indicator for investment decisions since the model could be refined much more, and since the scope of this work was more on learning rather than profitability.

Keywords: XGBoost, LSTM, Windowing, Feature Engineering, Window Optimization, Hyperparameter Tuning, Mean Absolute Error, Predictions.


# Introduction
Is there a way to predict the unpredictable?. Certainly not, either if stock data is discrete random, the probability of exactly predicting the correct price in the future is near 0%. Nonetheless, the spread of the future price can be shrinked down into a _confidenece interval_ that tries to reduce the risk (volatility) of the price.


# Data 
Apple Inc. is a publicy traded company on the tech index NASDAQ 100. Nowadays it is the highest valued company worldwide, with a capitalization of over 3 Billion $. Tu justify the selection of this stock, there is a need to point out the standart deviation that tech stocks suffer on a daily basis. Since this work faces a technical analysis, the risk of deviation of a stock (volatitlity) needs to be a bit higher than normal. What does this mean?, basically when someone expects the price to go up, if the historical fluctuation of the price has been exacerbated, there is a higher probability that if a certain price is reached, the price will move into that direction since more people will be dealing with that stock.

<p align="center">
    <img src= "https://user-images.githubusercontent.com/67901472/152142744-c6f4a510-bbf7-4f61-98b7-14a8408d0712.png" >
</p>

As seen in the histogram, we can observe that the distribution of the returns does not follow a normal distribution (represented as a black line in the plot), which can be observed in a higher kurtosis and fatter tails. Regarding the Box Plot, we can observe that there is a significant amount of outliers that might harm our model, which is an issue which musst be considered while dealing with the features. Finally, it is also interesting how the stock performed in terms of cumulative returns, as seen in the line chart, where we can observe the evolution of the stock repsect to other tech gigants (append you find the annualized returns).


# Feature Engineering

In this section we will discuss the new features created in order to tackle a good performance in our model.

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
        data[f"Adj_Close{i}"] = data["Adj Close"].rolling(i).mean()
        data[f"Volume{i}"] = data["Volume"].rolling(i).mean()

        data[f"Low_std{i}"] = data["Low"].rolling(i).std()
        data[f"High_std{i}"] = data["High"].rolling(i).std()
        data[f"Adj_CLose{i}"] = data["Adj Close"].rolling(i).std()

        data[f"Close{i}"] = data["Close"].shift(i)

        data[f"Adj_Close{i}"] = data["Adj Close"].rolling(i).max()
        data[f"Adj_Close{i}"] = data["Adj Close"].rolling(i).min()

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


# LSTM-XGBoost

Said this, let sdive deep into the core part of this project, where the combination between algorithms will (hopefully) provide us with reliable estimations of the Apple stock price for tomorrow.

In this sections we will make use of some user defined functions that mainly try to automitize the optimization, interpretation of the applyied models whether through a windowiing functions, graphs or comparisons.

### Main User Defined Fucntions:
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
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
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



## XGBoost
XGBoost, is one of the most highly used supervised ML algorithms nowadays, as it uses a more optimized way to implement a tree based algorithm, and it is also able to efficiently manage large and complex datasets.

The methodology followed by this algorithm is the following. XGBoost uses a Greedy algorithm for the building of its tree, meaning it uses a simple intuitive way to optimze the algorithm. The algorithm combines its best model, with previous ones, and so minimizes the error. So, in order to constantly select the models that are actually imporving its performance, a target is settled. and this target will depend on how much the next model has decreased the error, if there was an increase or no change in the error ythe target will be set to zero, otherwise it will set really high since it is difficult to surpas the performance of the previous model.

For more insights into how this algorithm works, check out this video from [StatQuest](https://www.youtube.com/watch?v=OtD8wVaFm6E&t=649s)

#### Training the Model

<p align="center">
    <img src= "https://user-images.githubusercontent.com/67901472/152218072-d0bbd0b0-7f59-449d-87c3-04dfb711763f.png" >
</p>

#### Testing the Model

<p align="center">
    <img src= "https://user-images.githubusercontent.com/67901472/152218913-d46d6ed6-3623-4720-a8af-84d75054e0d2.png">
</p>


## LSTM


<p align="center">
    <img src= "https://user-images.githubusercontent.com/67901472/152218220-1010ad55-4342-410d-b795-442db442cdb6.png" width="550", height="250">
</p>



### Assumptions



## Hybrid Approach

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





# Conclusions and Clarifications



