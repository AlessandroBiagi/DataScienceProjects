# Predict Future Sales

This project is a classical Data Science problem regarding numerical regression, in particular prediction of future sales (you can find it on Kaggle, see the link below). Given some qualitative and historical features about the items, the goal is to predict how many times an item will be sold on the next month.

**Kind of project**: Regression
**Number of features**: 32
**Target label**: item_cnt_day, numerical
**Target metric**: RMSE as requested on the Kaggle web page
**Final model**: models/saved_models/tuned_models/xgb_manual.sav, with RMSE of 2.14894%
**Main code**: scripts/Predict_Future_Sales.ipynb

Reference: This DataSet is public on Kaggle and you can find more details looking at https://www.kaggle.com/c/competitive-data-science-predict-future-sales

## Table of contents of this repository
Additional info: we provide a requirements.txt with some particular versions of xgboost and matplotlib that we require


## Structure of the main code
- First of all we merge the datasets we have in order to build a proper fact table that we can analyze. Basically, we merge the shop information with the item information
- Then it follows a first **EDA** phase where we study in depth the statistical properties of every feature. Some relevant discovered facts were that almost the 90% of our observations has item_cnt_day equal to 1 and that there is an item_id more popular than the others by far (item_id 20949 is about the 1% of our observations). Moreover, all the names of the shops are in Cyrillic
- After this, we proceed with a first **feature engineering** phase: we treated properly incorrect values, outliers etc.
- Now we perform an **EDA** on the pairs (shop_id, item_id): we discover that we will need to predict item_cnt_day for a lot of pairs (shop_id, item_id) that our model will not be trained on
- This implies another **feature engineering** phase where we treat properly the absent months for the pairs (shop_id, item_id).
- So the first **strategy** that we apply is the naive one: we say that a given item will be sold in the same amount of the month before. Actually this strategy did not work so well when we tried to predict the month number 33 using the month 32 but we obtain a wonderful RMSE of 2.37068 when we try to predict the items of the test set (we filled the unknown values with 0)
- Then the goal is to apply some statistical models such as ARIMA or SARIMA to solve our problem. We extract randomly some observations in order to find some seasonalities. Since we find none of them, we discard SARIMA and we decide to try ARIMA for a fixed pair (shop_id, item_id). We build an ARIMA(1,1,1) model. Trying to predict the month number 33 using the previous 4 months, We got an RMSE of 9.372 which we consider high. Because of this, we do not feel confident to apply ARIMA(1,1,1) for other pairs for the moment and we proceed with another method
- The second strategy is to build a XGBoost model where for every observation we add new features containing the item_cnt_day of the months before, i.e. we use some lagged data. In particular, we build lag-columns for the twelve months before. We also insert as new feature the item_price of the previous month. Before applying the model, we perform a RFECV study using a Random Forest to get an idea about the most relevant features. Due to our limited computational resources, we perform the RFECV only to the 5% of our dataset. Thanks to this we observe that the features about 1 and 12 months before are selected among the most important features, which totally makes sense. Using this information we make a selection of the columns to use as input for XGBoost. After the training, we submit the results on Kaggle obtaining an RMSE of 2.32279, which is better than the first strategy but we were hoping for better results
- At this point we decide to use the same dataset of the previous step but we make a couple of hyperparameter tunings of the model, hoping to get better results this time. Firstly, we try to find the best parameters in a given space using a Coarse-To-Fine approach: we provide a space to the hyperopt library which selected some promising parameters of that region and after that we fine our research looking at a neighbourhood of these parameters with GidSearchCV. Even this case, our research is limited due to our poor computational resources. Probably because of this, actually we got a worse result: 2.34264. At this point, we simply try to tune the hyperparameters manually providing a larger number of trees: finally thins got better. Indeed, we have a final RMSE of 2.14894. The metric itself has improved comparing to the previous strategies (even if not so much), but in a production context we may consider this model as much more robust compared to the first approach because of the tree-based boosting approach of XGBoost and because in any case the model using the information both the month before and the year before, which is quite reasonable.

## Ideas for the future to improve the results
For the following ideas we assume that we want to try again XGBoost in order to apply on the the dataset with the lagged data:
- We could train XGBoost using additional custom features, for example for every row we could compute the average item_cnt_day until that month by shop_id, item_id or category_id
- We could apply a different feature selection, for example taking into account the lags number 1,2,3,6 and 12 and re-inserting the item_price of the month before
- We could try to apply XGBoost without the features selection since we apply RFECV only to a 5% of our dataset and maybe we lost some leve of reliability
- The RFECV algorithm and the coarse-to-fine approach have been limited a lot because of the computational resources. A possibility to get better results would be to re-launch the algorithms leaving the pc to work for several days