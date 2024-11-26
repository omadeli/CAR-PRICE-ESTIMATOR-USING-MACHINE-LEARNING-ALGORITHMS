# MACHINE LEARNING MODEL TO PREDICT CAR PRICES

# PROBLEM STATEMENT
The dataset which contains over 400k rows of car adverts is provided by Autotrader a well known British automotive online marketplace and classified advertising business. Autotrader wants to develop a robust feature on their website that will help a customer value their car (predict the price). The dataset contains the following features **publish_reference, year_of_registration, mileage, standard_make, standard_model,body_type,fuel_type, vehicle_condition, and crossover_car_and_van**. The main task at hand is to clean, explore and analyse the data to identify the best predictors of the price of cars and also build a machine learning model that can accurately predict the prices

# KEY INSIGHTS FROM ANALYSIS
* Car model is a significant predictor of car price.
* Colour of car does not offer any help in predicting car prices.
* Mileage impacts price of cars, low mileage results in a higher price while high mileage results in a lower price.
* Old cars with high prices where identified as vintage/luxurious cars.
* The make of the car is also a significant predictor of car price.

# BUSINESS RECOMMENDATIONS




# DATA CLEANING
The data cleaning process involved analysis of missing of missing values and outliers to detect anomality and erroneous values.
![missing values](https://github.com/user-attachments/assets/0f8e2383-dcdb-4c1d-bf62-712a14e80705)
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
## What really affects the price of car? 
* Exploring the relationship between year of registration indicates that as the year of registration increases the price reduces. This is an early indication of the year of registration might be a good predictor of car prices
  ![yor and price](https://github.com/user-attachments/assets/4823a0d4-83db-4068-9ccf-e9812291037f)
* The colour of a car might not necessarily affect the price of a car too much as inidicated by the distribution of price for each car below. The price range for each colour are not far from each other. Gold though has a wide range while magneta and indigo have a really small range, but considering this colours have little obseration, we can not really conclude. **Black and white are the most popular colours which occur in over 80% of the data and they have similar distribution which is an indication that colour is does not have a strong relationship with the price of car**.
  ![colour](https://github.com/user-attachments/assets/2312118b-3cc1-4bed-bede-846f2976336c)




# MACHINE LEARNING
The best perfoming model for this machine learning task was the gradient boosting algorithm. The pipeline for the model can be seen below.
Sequential feature selection was carried out with linear regression specified as the estimator
![image](https://github.com/user-attachments/assets/8c24b22a-ab93-42f0-94c5-83dbc2e6ea30)

![image](https://github.com/user-attachments/assets/d6e5b5cc-4788-4902-b855-e32b0ecefaee)


# DRILLING DEEPER USING SHAP
The SHAP summary plot indicates the effect of each feature on the model’s output. Standard_model shows a fairly wide range of impact on predictions, with high values (in red) increasing predictions and lower values (in blue) slightly decreasing them. Age significantly impacts predictions, with higher ages (in red) generally lowering the model's output, while younger ages (in blue) increase it. Mileage has a moderate negative influence, where higher mileage reduces the prediction and lower mileage has a smaller positive effect. Standard_make has a mixed but minor influence, with some specific values contributing slightly to positive predictions. Finally, body_type shows minimal variation, with a very small effect overall on the model’s output
![image](https://github.com/user-attachments/assets/54716ad6-e8d5-4d86-bbfe-21ba50e3c2f3)


