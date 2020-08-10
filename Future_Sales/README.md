# Predict Future Sales

This project is a classical Data Science problem regarding numerical regression, in particular prediction of future sales (you can find it on Kaggle, see the link below). Given some qualitative and historical features about the items, the goal is to predict how many times an item will be sold on the next month.

Kind of project: Regression<br/> 
Number of features: 32<br/>
Target label: item_cnt_day, numerical<br/>
target metric: RMSE as requested on the Kaggle web page<br/>
Final Model: Models/saved_models/xgboost_tuned.sav, with accuracy 86,7%<br/>

Reference: This DataSet is public on Kaggle and you can find more details looking at https://www.kaggle.com/c/competitive-data-science-predict-future-sales

Additional info: we provide a requirements.txt with some particular versions of xgboost and matplotlib that we require.