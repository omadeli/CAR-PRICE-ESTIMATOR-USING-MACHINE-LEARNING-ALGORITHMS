# CAR-PRICE-ESTIMATOR-USING-MACHINE-LEARNING-ALGORITHMS
The aim of this project is to build a Machine Learning model to predict car prices. The dataset is provided by AutoTrader- a well known British automotive online marketplace and classified advertising business.

In the future I will deploy the Machine Learning Model so it's output can be rendered in the frontend of an application.

# KEY INSIGHTS
* Car model and age affects the price of cars postively and negatively
* Colour of car does not impact price.
* Mileage impacts price of cars, low mileage results in a higher price while high mileage results in a lower price
* Old cars with high prices where indetified as vintage/luxurious cars
* Body type of car moderately affects the price of cars.
  
**A technical report can be found among the files above**

# EXPLORATORY DATA ANALYSIS
## What affects car prices ?
We can observe below that the colour of a car might not necessarily affect the price of the car in any significant way. The median price for a colors are all averagely the same.
![image](https://github.com/user-attachments/assets/822beb57-b221-47bb-9145-182286869b5d)

As the mileage of of car increases the price tends to reduce. This is inline with the trend that as a car gets older, its price tends to drop. The exception to this is unless the car is a vintage/luxurious car e.g Buggati.
![image](https://github.com/user-attachments/assets/d42c6ab0-f087-4dc5-8349-1d1f384540b9)

Observing the correlation matrix below, mileage is moderately negatively correlated with price , while the year of registration is positively correlated with price indicating that as year of registration increases price as increases as newer cars will be expected to be more expensive that older cars. The reverse can be said for age
![image](https://github.com/user-attachments/assets/44b6a867-8bd7-4d5b-9521-65757203936e)

The variation observed in the mean price of each model of cars indicate that the model of a car has some effect on car prices.
![image](https://github.com/user-attachments/assets/9fc1c40f-9646-4bf7-b5b1-6d0c2c88cde7)


# BEST MACHINE LEARNING MODEL
The best perfoming model for this machine learning task was the gradient boosting algorithm. The pipeline for the model can be seen below.
Sequential feature selection was carried out with linear regression specified as the estimator
![image](https://github.com/user-attachments/assets/8c24b22a-ab93-42f0-94c5-83dbc2e6ea30)

**Train score - 96%**
**Test score - 95%**

# DRILLING DEEPER WITH USING SHAP
Using SHAP (SHapley Additive exPlanations) to visualise feature impact of price, we observe that standard model affects price the most. The plot indicates that standard model can a affect car price positively by up to £50,000 and negatively by about £10,000. The standard make is also another discovery, as the plots show that it can affect car price positively almost £40,000 and negatively by £8,000 - £ 9,000
![image](https://github.com/user-attachments/assets/580a89e0-3f2c-4f28-9d14-7a6e61081f6c)

