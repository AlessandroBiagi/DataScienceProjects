# Predict Future Sales

This project is a classical Data Science problem regarding numerical regression, in particular prediction of future sales (you can find it on Kaggle, see the link below). Given some qualitative and historical features about the items, the goal is to predict how many times an item will be sold on the next month.

Kind of project: Regression<br/> 
Number of features: 32<br/>
Target label: item_cnt_day, numerical<br/>
target metric: RMSE as requested on the Kaggle web page<br/>
Final Model: models/saved_models/tuned_models/xgb_manual.sav, with RMSE of 2.14894%<br/>
main code: scripts/Predict_Future_Sales.ipynb

Reference: This DataSet is public on Kaggle and you can find more details looking at https://www.kaggle.com/c/competitive-data-science-predict-future-sales

Additional info: we provide a requirements.txt with some particular versions of xgboost and matplotlib that we require


## Structure of the main code
- First of all we merge the datasets we have in order to build a proper fact table that we can analyze. Basically, we merge the shop information with the item information
- Then it follows a first **EDA** phase where we study in depth the statistical properties of every feature. Some relevant discovered facts were that almost the 90% of our observations has item_cnt_day equal to 1 and that there is an item_id more popular than the others by far (item_id 20949 is about the 1% of our observations). Moreover, all the names of the shops are in Cyrillic
- After this, we proceed with a first **feature engineering** phase: we treated properly incorrect values, outliers etc.
- Now we perform an **EDA** on the pairs (shop_id, item_id): we discover that we will need to predict item_cnt_day for a lot of pairs (shop_id, item_id) that our model will not be trained on
- This implies another **feature engineering** phase where we treat properly the absent months for the pairs (shop_id, item_id).
- So the first **strategy** that we apply is the naive one: we say that a given item will be sold in the same amount of the month before. Actually this strategy did not work so well when we tried to predict the month number 33 using the month 32 but we obtain a wonderful RMSE of 2.37068 when we try to predict the items of the test set (we filled the unknown values with 0)
- Then the goal is to apply some statistical models such as ARIMA or SARIMA to solve our problem.