# CAR-PRICE-ESTIMATOR-USING-MACHINE-LEARNING-ALGORITHMS
The aim of this project was to build a Machine Learning model to predict car prices. The dataset is provided by AutoTrader- a well known British automotive online marketplace and classified advertising business.

In the future I will deploy the Machine Learning Model so it's output can be rendered in the frontend of a web app.

# KEY INSIGHTS
* Car model is a significant predictor of car price.
* Colour of car does not offer any help in predicting car prices.
* Mileage impacts price of cars, low mileage results in a higher price while high mileage results in a lower price.
* Old cars with high prices where identified as vintage/luxurious cars.
* The make of the car is also a significant predictor of car price.
  
**A technical report can be found among the files above**

# EXPLORATORY DATA ANALYSIS
## What affects car prices ?
We can observe below that the colour of a car might not necessarily affect the price of the car in any significant way. The median price for a colors are all averagely the same.
![image](https://github.com/user-attachments/assets/b23101d1-25fa-4c86-8ed9-e26b058201fc)

As the mileage of of car increases the price tends to reduce. This is inline with the trend that as a car gets older, its price tends to drop. The exception to this is unless the car is a vintage/luxurious car e.g Buggati.
![image](https://github.com/user-attachments/assets/23bff01f-4da7-4f41-9c64-a00adbc8181b)

Observing the correlation matrix below, mileage is moderately negatively correlated with price , while the year of registration is positively correlated with price indicating that as year of registration increases price as increases as newer cars will be expected to be more expensive that older cars. The reverse can be said for age
![image](https://github.com/user-attachments/assets/44b6a867-8bd7-4d5b-9521-65757203936e)

The variation observed in the mean price of each model and make of cars indicate that the model of a car has some effect on car prices.
![image](https://github.com/user-attachments/assets/22d371f8-2cb1-4594-b61b-e7aa534d34bb)
![image](https://github.com/user-attachments/assets/6e6d519a-d2b8-4eba-9061-49c830354c65)



# BEST MACHINE LEARNING MODEL
The best perfoming model for this machine learning task was the gradient boosting algorithm. The pipeline for the model can be seen below.
Sequential feature selection was carried out with linear regression specified as the estimator
![image](https://github.com/user-attachments/assets/8c24b22a-ab93-42f0-94c5-83dbc2e6ea30)

![image](https://github.com/user-attachments/assets/d6e5b5cc-4788-4902-b855-e32b0ecefaee)


# DRILLING DEEPER WITH USING SHAP
The SHAP summary plot indicates the effect of each feature on the model’s output. Standard_model shows a fairly wide range of impact on predictions, with high values (in red) increasing predictions and lower values (in blue) slightly decreasing them. Age significantly impacts predictions, with higher ages (in red) generally lowering the model's output, while younger ages (in blue) increase it. Mileage has a moderate negative influence, where higher mileage reduces the prediction and lower mileage has a smaller positive effect. Standard_make has a mixed but minor influence, with some specific values contributing slightly to positive predictions. Finally, body_type shows minimal variation, with a very small effect overall on the model’s output
![image](https://github.com/user-attachments/assets/54716ad6-e8d5-4d86-bbfe-21ba50e3c2f3)


