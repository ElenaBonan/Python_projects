# Hotel Booking Demand
In this project we are using data coming from two hotels, more precisely every row corrispond to a reservation. Using the available information at the moment of the booking (expected arrival date, deposit type, type of room etc.),
we want to predict if a customer is going to cancel or not the reservation. 

<b>Problem:</b> binary classification <br>
<b>Target variable:</b> is_canceled <br>
<b>Independent variables:</b> 31 (numerical and categorical) <br>
<b>Data source</b>: https://www.kaggle.com/jessemostipak/hotel-booking-demand <br>

## Content of the Repository
- 
- 
- 
## Structure of the Code
- <b>Exploration of the variables</b> We get familiar with the data and we look at the distributions of each variable.
- <b>Data cleanining and Feature Engineering:</b> We take care of outliers and null values. We group the variables where it is needed.
- <b>Correlection and Association of the variables</b> We look at the releationship between the variables using Pearson correlation, Eta correlation and Cramer V association.
- <b>Encoding of the categorical variables</b> We treat the numerical variables using the stratified mean encoding.
- <b>Features importance</b> We look at the importance of the feature using a recursive feature elimination with the random forest.
- <b>Modelling Part 1 </b> We use xgboost and randomforest with the default parameters and we look at the results on the test set. 
- <b>Modelling Part 2 </b> We tune the hyperparameters of the xgboost using first the Hyperopt package to find the most promising regions of the parameter space and then a grid search. 


## Possible improvements 
- Going more deepen with data exploration
- Try to have  two separates models for the two hotels.
- Try the ANN. 
