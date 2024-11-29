# MACHINE LEARNING MODEL TO PREDICT CAR PRICES

# PROBLEM STATEMENT
The dataset which contains over 400k rows of car adverts is provided by Autotrader a well known British automotive online marketplace and classified advertising business. Autotrader wants to develop a robust feature on their website that will help a customer value their car (predict the price). The dataset contains the following features **publish_reference, year_of_registration, mileage, standard_make, standard_model,body_type,fuel_type, vehicle_condition, and crossover_car_and_van**. The main task at hand is to clean, explore and analyse the data to identify the best predictors of the price of cars and also build a machine learning model that can accurately predict the prices

# KEY INSIGHTS FROM ANALYSIS
* The best machine leaarning for that fit the data with **92%** accuracy was the Gradient Boosting model
* The MAE(mean absolute error) of the gradient boosting model was 2756.51 meaning that on average our predictions are off by £2765.51
* Using sequential feature selection,**year of registration, mileage, model of car and make of car** were the best predictors of price of car

# BUSINESS RECOMMENDATION
* The input fields on the website's forms should use dropdowns with auto-suggestions or auto-completion to minimize errors from customer entries
* Customer feedback should be collected to determine if they feel the valuation provided is accurate and satisfactory. This can be implemented by redirecting customers to a brief form (to minimize bounce rates), featuring a dropdown with options such as 'Satisfied' and 'Not Satisfied,' along with an optional text field for additional comments..  


# DATA CLEANING
The data cleaning process involved analysis of missing of missing values and outliers to detect anomality and erroneous values.
![Missing valuess](https://github.com/user-attachments/assets/7134d359-19ff-405e-b61f-3973628b7cf9)
* Erroneous value was detected and removed from the reg_code column
* The year of registration and regcode had over 30K missing values. Over 90% of the missing values were for new vehicles and thus their year of registration was imputed by extracting the year the car advert was published from the public_reference feature. The remaining missing values were imputed  by scraping the corresponding year of registration for their reg_code from wikipedia.
* The age of the car was then engineered by subtracting year car was published from the year it was registered.
* The missing values in body_type, fuel_type, standard_colour will be imputed with the mode in the machine learning preprocessing pipeline  and the mileage will be imputed with the mean as well.
  
Descriptive statistics for the numerical features
![description](https://github.com/user-attachments/assets/fbcbb58e-6f6b-416c-8ab7-a727572f5417)
* The mileage has a max value of 999,999miles, which is highly impossible. This was analysed further using a boxplot as shown below
![mileage](https://github.com/user-attachments/assets/16c2bf74-4c87-4ac5-9cf7-fd9163dc9d64)
* Mileage was capped at 200k because most of the values above 200k are looking unrealistic and according to an article from caranalytics.com, The typical annual miles of a vehicle is between 10,000 and 15,000 miles which translates to 100k-150k miles per 10years. Capping it at 200k will enable the model to generalise to high values of mileage considering taxis and other heavy use vehicle might hit 150k to 200k miles per 10years
* The outliers in price are due to the price of luxurious and vinatge cars. Dropping this might reduce the robustness of the Machine Learning model and because we want our model to be able to predict both expensive and less expensive car prices these rows were not dropped

# EXPLORATORY DATA ANALYSIS
## What really affects the price of acar? 
* Exploring the relationship between year of registration indicates that as the year of registration increases the price reduces. This is an early indication of the year of registration might be a good predictor of car prices
![yor and price](https://github.com/user-attachments/assets/4823a0d4-83db-4068-9ccf-e9812291037f)
* The colour of a car might not necessarily affect the price of a car too much as inidicated by the distribution of price for each car below. The price range for each colour are not far from each other. Gold though has a wide range while magneta and indigo have a really small range, but considering this colours have little obseration, we can not really conclude. **Black and white are the most popular colours which occur in over 80% of the data and they have similar distribution which is an indication that colour is does not have a strong relationship with the price of car**.
![colour](https://github.com/user-attachments/assets/2312118b-3cc1-4bed-bede-846f2976336c)

* Sampling the mean price for each make and model of a car indicates that they are good predictors of the price of car due to the variation observed as shown in the plots below
![standard model](https://github.com/user-attachments/assets/59817174-f64d-47e3-ad78-36fc0ef1d04d)

![standard make](https://github.com/user-attachments/assets/44c2d188-d755-420b-9ef2-cc92b6967090)

*Exploring body type we can observe that the distribution of price varies for each body type but not by much except in some few cases, for example Limousine
![body_type](https://github.com/user-attachments/assets/db996206-db32-4471-8162-02fc29037cb4)

*Exploring the distribution of price for each fuel type we observe that petrol and diesel cars have very similar distribution and considering the occur in over 80% of the data, the fuel type might not have a strong enough relationship with price but it also worth noting that the Petrol plug-in hybrid and diesel hybrid fuel type are on the high side of price.
![fuel type](https://github.com/user-attachments/assets/d53ceff7-ea4e-4d95-9892-95b88ca63169)

* Vehicle condition was dropped due to the fact that over 93% of the dataset contained old cars and thus this feature was dropped due to the high level of imbalance.


# MACHINE LEARNING
The following preprocessing steps were carried out to prepare the data for machine learning.
* Imputation of missing values using mean for numerical features and mode for categorical features.
* A custom class was created to group low frequency cateogories in high cardinality features such as the standard_make and standard_model.
* Target encoding was carried out to encode the categorical variables.
* Features were scaled for linear models but was not scaled for tree based models - scaling for tree based models has little to no effect on performance and is not neccessary
* Sequential Feature Selection was carried out to select best features. The selection process returned **year of registration, mileage, standard_make and standard_model**
 as best predictors of car price.

Pipeline can be seen below
![gbr pipe](https://github.com/user-attachments/assets/d5b7af17-b13b-416f-9627-3be7ca58d524)

Experiments were carried out with Linear Regression, Random Forest, Gradient Boosting and an ensemble model of Random Forest and Gradient Boosting.

# BEST MACHINE LEARNING MODEL
![results](https://github.com/user-attachments/assets/a5f9077f-2de4-423d-bb7d-80f352589310)
The **MAE (Mean absolute error) and rsquared metric** was used to measure the accuracy of the model.The linear regression model underfits. The random forest and gradient boosting models fits the data well and generalise to new data just fine with an MAE and rsquared of £2933.39, 91% and £2756.59, 92% respectively. An ensemble of both tree models results in MAE of £2794 indicating that the ensembling of both the random forest and gradient boost models results in a reduction of the MAE from the gradient boosting model. Thus the gradient boosting model will be the choosen as the best model

Visualing the gradient boosting model's peformance
![gbr ](https://github.com/user-attachments/assets/4539383c-0a6f-4c69-b5ca-f220400cf041)

# DRILLING DEEPER USING SHAP
![shap](https://github.com/user-attachments/assets/ce91743e-0d2e-46d5-81b9-f0dd33b5f3aa)


