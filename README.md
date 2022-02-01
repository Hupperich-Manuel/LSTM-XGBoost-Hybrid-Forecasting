# LSTM-XGBoost Hybrid Stock Forecasting

Whom of you has not thought about making money training a model that returns accurate predictions on t+1, so that one can gain from this hail grail of information?. Well, since this might be true in some cases this is really far from the scope of this work.
The idea behind this work is to take my knowledge one step further. The hybrid combinations of Deep Learning models together with Decision trees or Linear Regressions are fascinating new ways to extract much more information of the raw inputs one is dealing with. Therefore, I found my opportunity to combine things learned throughout the past years related to coding, statistics, ML models, DL models, Business Perspective, and squeeze those to return an actual _deployable_ model for a real time application.


# Abstract
This work contains an overall analysis of the takeaways on applying a hybrid Long Short Term Memory Deep Learning Model together with an XGBoost Regressor to predict the price of the AAPL (Apple Inc.) stock for the next day (**t+1**). Throughout this work, some assumptions are made regarding the optimal number of features, some of the hyperparameter tuning (even though backtesting and tunning was done till a certain point). Notice that the expected outcome of this model should not be used as an indicator for investment decisions since the model could be refined much more, and since the scope of this work was more on learning rather than profitability. 


# Introduction
Is there a way to predict the unpredictable?. Certainly not, either if stock data is discrete random, the probability of exactly predicting the correct price in the future is near 0%. Nonetheless, the spread of the future price can be shrinked down into a _confidenece interval_ that tries to reduce the risk (volatility) of the price.


# Data 
Apple Inc. is a publicy traded company on the tech index NASDAQ 100. Nowadays it is the highest valued company worldwide, with a capitalization of over 3 Billion $. Tu justify the selection of this stock, there is a need to point out the standart deviation that tech stocks suffer on a daily basis. Since this work faces a technical analysis, the risk of deviation of a stock (volatitlity) needs to be a bit higher than normal. What does this mean?, basically when someone expects the price to go up, if the historical fluctuation of the price has been exacerbated, there is a higher probability that if a certain price is reached, the price will move into that direction since more people will be dealing with that stock.

<p align="center">
    <img src= "https://user-images.githubusercontent.com/67901472/152019252-b834155f-44e0-469c-84dd-926231c46dde.png" >
</p>

As seen in the histogram, we can observe that the distribution of the returns does not follow a normal distribution (represented as a black line in the plot), which can be observed in a higher kurtosis and fatter tails (also Shapiro was applyed and we can say that the residuals do not follow a normal distribution). Said this, considering to use ML models which assume Normality might be struggle unless we perform differentiations or more logs on the data. 


# Feature Engineering

# LSTM-XGBoost

## XGBoost

### Assumptions

## LSTM

### Assumptions

## Hybrid Approach

# Alternatives

# Conclusions and Clarifications



