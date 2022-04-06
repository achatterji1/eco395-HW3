---
title: "HW3 04.06.2022"
author: "Arpan Chatterji, Rajsitee Dhavale"
date: "4/6/2022"
output:
  pdf_document: default
  html_document: default
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```



```{r load-packages, include = FALSE}
library(tidyverse)
library(ggplot2)
library(mosaic)
library (knitr)
library(caret)
library(foreach)
library(FNN)
library(rsample)
library(modelr)
library(class)
library(quantmod)
```

## What causes what?

``` 
1.
You cannot just get data from a few different cities and run the regression of crime on police to understand how more cops in the streets affect crime. This is because you need an example of a city where the number of (increase in) the police force is unrelated to crime levels in the city. Only if you have this will you have an effective control group, thus giving you accurate unbiased results for the policy question “if you hire more cops, will crime go down?”
Most cities tend to have a direct relationship between the number of police and the crime level because cities with high levels of crime have a higher incentive to hire more police. 

2. The researchers wanted a clever way to establish a causal relationship between more police and less crime. They were looking for a natural experiment. 

The approach they used was as follows: They looked for a city with a large number of police for reasons unrelated to crime. Thus, they picked Washington D.C. as it had a terrorism alert system (with color-coded alerts). Here, people would relate the increase in the police force to a potential terror threat and not to an increase in crime. Thus, people (tourists i.e., potential victims of criminals) and criminals wouldn’t alter their behavior. The researchers then checked Metro ridership data to see if fewer tourists had ventured out (fewer victims) and found that ridership was largely unchanged. 

Results: 
For (1): Only One Variable used 
When there is 1 more cop on the street, the crime rate falls by 7.316 units (inverse relationship), all other things constant. 
The R squared number 0.14 indicates that 14% of the data fit the model. However, this by itself doesn’t explain the causation relationship between the independent and dependent variables. 

For (2): Control for Metro Ridership
Where there is one more police on the street, crime falls by 6.316 units. Here the coefficient is significant at 1%, indicating better results as 1% level of significance lowers the chance of a false positive, all other things constant. This further proves the hypothesis that increasing the police force in a city lowers the level of crime. The R squared number 0.17 indicates that 17% of the data fit the model. However, this by itself doesn’t explain the causation relationship between the independent and dependent variables. 

Overall, the table indicates that holding everything else constant, an increase in the number of police leads to a decline in crime levels. 

3. The researchers had to control for Metro ridership because they wanted to check whether there were any other reasons for the crime rate to fall. For example, did the crime rate fall because there were fewer victims on the street due to the terror threat? 

4. In table 4, the researchers are trying to estimate whether the effect of high alerts days on crime was the same across all parts of the city. This is a more refined analysis than the one in table 2. When the researchers used interactions between high alert days and location, they found the effect to be clear only in District 1. This is probably because that part of the city has the highest number of important monuments and so is the highest on the terrorism alert. Thus, this area will also have the highest number of cops. 
The effect in the other districts is still negative, indicating that increasing the number of cops does lower crime, but the effect is very small. When we look at this and then at the standard error in the brackets, we could conclude the effect to be zero too because of the 5% level of significance. 
```
## Tree Modeling Dengue Cases


```{r, warning=FALSE, error=FALSE, include=TRUE}
library(knitr)
library(readr)
library(rmarkdown)
install.packages('plyr', repos = "http://cran.us.r-project.org")

install.packages("kableExtra", repos = "http://cran.us.r-project.org")
install.packages("gbm", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(ggplot2)
library(rpart)
library(rpart.plot)
library(rsample)
library(randomForest)
library(lubridate)
library(modelr)
library("devtools")
library(gbm)
library(kableExtra)
```

```{r, echo=FALSE, warning=FALSE, error=FALSE}
dengue <- read_csv("https://raw.githubusercontent.com/jgscott/ECO395M/master/data/dengue.csv") %>% drop_na()
```

```{r, echo=FALSE, warning=FALSE, error=FALSE}
## Tree Model: Dengue Cases
lapply(dengue, class)

dengue$city <- dengue$city %>% factor()
dengue$season <- dengue$season %>% factor()

# Train-Test Split
dengue_split <- initial_split(dengue, 0.8)
dengue_train <- training(dengue_split)
dengue_test <- testing(dengue_split)
```

```{r, echo=FALSE, warning=FALSE, error=FALSE}
# CART (Classification & Regression Trees)
cart_dengue <- rpart(total_cases ~ season + specific_humidity + precipitation_amt + tdtr_k,
                     data = dengue_train, control = rpart.control(cp = 0.002, minsplit = 30))

# Plot
plotcp(cart_dengue, main = "Cross_Validated Error by CP")
```

```{r, echo=FALSE, warning=FALSE, error=FALSE}
# Pick the smallest tree whose CV error is within 1 std error of the minimum
cp_1se = function(my_tree) {
  out = as.data.frame(my_tree$cptable)
  thresh = min(out$xerror + out$xstd)
  cp_opt = max(out$CP[out$xerror <= thresh])
  cp_opt
}

cp_1se(cart_dengue)

# Pruning the tree at that level
prune_1se = function(my_tree) {
  out = as.data.frame(my_tree$cptable)
  thresh = min(out$xerror + out$xstd)
  cp_opt = max(out$CP[out$xerror <= thresh])
  prune(my_tree, cp=cp_opt)
}

cart_dengue_prune <- prune_1se(cart_dengue)
```

## Random Forest Model
```{r, echo=FALSE, warning=FALSE, error=FALSE}
# Random Forest Model
dengue_randomforest <- randomForest(total_cases ~ season + specific_humidity + precipitation_amt + tdtr_k,
                              data = dengue_train, importance = TRUE, na.action = na.omit)

# Out-of-bag MSE as a function of the number of trees used (we don't use the test set here)gh
plot(dengue_randomforest, main = "Out-of-Bag MSE by No. of Trees")

# Gradient Boosting
dengue_gradient_boosting <- gbm(total_cases ~ season + specific_humidity + tdtr_k + precipitation_amt,
                                data = dengue_train, distribution = "gaussian",n.trees = 10000, shrinkage = 0.01, interaction.depth = 4)

# Error curve
gbm.perf(dengue_gradient_boosting)
```

```{r, echo=FALSE, warning=FALSE, error=FALSE}
# RMSE
modelr::rmse(cart_dengue, dengue_test)
modelr::rmse(cart_dengue_prune, dengue_test)
modelr::rmse(dengue_randomforest, dengue_test)
modelr::rmse(dengue_gradient_boosting, dengue_test)

rmsemodels <- c("RMSE CART" = rmse(cart_dengue, dengue_test),
                "RMSE CART Prune" = rmse(cart_dengue_prune, dengue_test),
                "RMSE Random Forest" = rmse(dengue_randomforest, dengue_test),
                "RMSE Gradient Boosting" = rmse(dengue_gradient_boosting, dengue_test))

kable(rmsemodels, col.names = c("RMSE"), caption = "*Table 2.1 : RMSE per Model*") %>%
  kable_styling(bootstrap_options = "striped", full_width = F)
```

## Final Result
```{r, echo=FALSE, warning=FALSE, error=FALSE}
# Final Result: We now see that the Random Forest Model is the best
partialPlot(dengue_randomforest, as.data.frame(dengue_test), 'specific_humidity', las = 1)
partialPlot(dengue_randomforest, as.data.frame(dengue_test), 'precipitation_amt', las = 1)
partialPlot(dengue_randomforest, as.data.frame(dengue_test), 'season', las = 1, ) #(Diagram: Bar graph)
```


## 3) Predictive model building: green certification 

The goal is to build the best predictive model possible for revenue per square foot per calendar year, and to use this model to quantify the average change in rental income per square foot associated with green certification, controlling for other features of the building.


```{r include=FALSE}
rm(list = ls())
greenbuildings <- read.csv("~/Desktop/Data Mining/data/greenbuildings.csv")
greenbuildings = greenbuildings %>%
  mutate(
    revenue = Rent * leasing_rate
  )
## Split the data for testing
greenbuildings_initial = initial_split(greenbuildings)
n = nrow(greenbuildings)
greenbuildings_train = training(greenbuildings_initial)
greenbuildings_test = testing(greenbuildings_initial)
# What is more relevant for green rating?
forest = randomForest(revenue ~ . - Rent - leasing_rate - green_rating,
                      na.action = na.omit,
                      data=greenbuildings_train)
greenbuildings_casetest = predict(forest, greenbuildings_test)
plot(greenbuildings_casetest, greenbuildings_test$revenue)
rmseforest = rmse(forest, greenbuildings_test)
plot(forest)
```
### Methods  
  
For this report, the goal was to detect the change in rent on houses with a green certificate, such that an architect would choose to construct if there is more revenue that can be generated when renting a "green" house. To create a predictive model, we created a variable for the rental income per square foot, or the average rent per square foot multiplied by the percentage of occupancy of the house. Following this, we had to define whether it was more relevant to include the variables for green rating as two separate controls or whether to remove those variables and use the general "green_rating" variable. To do so, we ran two different random forests, one with both variables and one with the general variable only.  
  

```{r echo=FALSE}
varImpPlot(forest)
```
  
  
```{r include=FALSE}
forest1 = randomForest(revenue ~. - Rent - leasing_rate - LEED - Energystar,
                      na.action = na.omit,
                      data=greenbuildings_train)
greenbuildings_casetest = predict(forest1, greenbuildings_test)
plot(greenbuildings_casetest, greenbuildings_test$revenue)
rmseforest1 = rmse(forest1, greenbuildings_test)
plot(forest1)
```
  
  
```{r echo=FALSE}
varImpPlot(forest1)
```
  
With those graphs, we saw that there was no real difference between the set of variables to use, so we decided to create four different predictive models using the general green rating instead of two separate models.  
  
```{r include=FALSE}
# After choosing best variable, compare to boosted
boost = gbm(revenue ~. - Rent - leasing_rate - LEED - Energystar, 
             data = greenbuildings_train,
             interaction.depth=4, n.trees=500, shrinkage=.05)
gbm.perf(boost)
yhat_test_gbm = predict(boost, greenbuildings_test, n.trees=350)
rmsegradient = rmse(boost, greenbuildings_test)
# Compare to linears
lm2 = lm(revenue ~. - Rent - leasing_rate - LEED - Energystar, data=greenbuildings_train)
lm3 = lm(revenue ~ (. - Rent - leasing_rate - LEED - Energystar)^2, data=greenbuildings_train)
rmsemedium = rmse(lm2, greenbuildings_test)
rmselarge = rmse(lm3, greenbuildings_test)
```
  
```{r echo=FALSE} 
tab1 <- matrix(c(rmseforest1, rmsegradient, rmsemedium, rmselarge), ncol=1, byrow=TRUE)
colnames(tab1) <- c("RMSE")
rownames(tab1) <- c("Random Forest", "Boosted Forest", "Medium Model", "Large Model")
tab1 <- as.table(tab1)
kable(tab1, align="lr", caption="Table with prediction model's RMSE")
```
  
The results, which are shown on the above table convinced us to use the random forest model to predict the rental income per square foot, since it had the smallest RMSE.  
  
```{r echo=FALSE}
p4 = pdp::partial(forest1, pred.var = 'green_rating')
ggplot(p4) +
  geom_line(mapping=aes(x=green_rating, y=yhat))
```
  
### Conclusion  
  
After predicting the value with our model, we decided to graph the average rental income per square foot associated with the green certificate. The graph shows that, on average, there is approximately $50-60 difference between having a green certificate or not. That means it is not very significant on the rental income the house having a green certificate.  


# Problem 4 - California Housing

The goal of this problem is to build the best possible model for predicting the median house value. We start by building the best possible linear, KNN, CART, Random Forest, and Boosted models. After the initial train-test split, we use just the training data to again split for each models building and optimization process.

```{r echo=FALSE,results=FALSE,warning=FALSE,message=FALSE}
# Loading Data
CAhousing <- read_csv("~/Desktop/Data Mining/data/CAhousing.csv")
# Adding averages for rooms and bedrooms
CAhousing = CAhousing%>%
  mutate(avg_room = totalRooms/households,
         avg_bed = totalBedrooms/households)
# Initial train-test split
CAhousing_split = initial_split(CAhousing,0.8)
ca_train = training(CAhousing_split)
ca_test = testing(CAhousing_split)
```

The first step for each model is to make a new train-test split from our global train-test split. Then use these new sets to build and test iterations of different models.


```{r echo=FALSE,results=FALSE,warning=FALSE,message=FALSE}
##### Linear Model
library(dplyr)
library(purrr)
library(modelr)
library(broom)
library(tidyr)
ca_linear_folds = crossv_kfold(ca_train, k=5)
linear_models = map(ca_linear_folds$train, ~ lm(medianHouseValue ~. - totalBedrooms - totalRooms + medianIncome*housingMedianAge + longitude*latitude, data = ., ))
linear_errs = map2_dbl(linear_models, ca_linear_folds$test, modelr::rmse)
linear_errs = mean(linear_errs)
##### KNN Model
ca_knn_split = initial_split(ca_train,.8)
ca_knn_train = training(ca_knn_split)
ca_knn_test = testing(ca_knn_split)
X = ca_knn_train[,-(10:11)]
X = scale(X, center=TRUE, scale=TRUE)
mu = attr(X,"scaled:center")
sigma = attr(X,"scaled:scale")
test = ca_knn_test[,-(10:11)]
Y = (test[,]-mu)/sigma
ca_knn = knnreg(medianHouseValue ~ ., data = X, k=15)
knn_errs = rmse(ca_knn, Y)
#### CART Model
prune_1se = function(my_tree) {
  out = as.data.frame(my_tree$cptable)
  thresh = min(out$xerror + out$xstd)
  cp_opt = max(out$CP[out$xerror <= thresh])
  prune(my_tree, cp=cp_opt)
}
ca_cart_folds = crossv_kfold(ca_train, k=5)
library(rpart)
cart_models = map(ca_cart_folds$train, ~prune_1se(rpart(medianHouseValue ~ .- totalRooms - totalBedrooms,
                                             data =., 
                                             control = rpart.control(cp = .0001, minsplit = 10))))
cart_errs = map2_dbl(cart_models, ca_cart_folds$test, modelr::rmse)
cart_errs = mean(cart_errs)
#### Random Forest
ca_forest_split = initial_split(ca_train, 0.8)
ca_forest_train = training(ca_forest_split)
ca_forest_test = testing(ca_forest_split)
library(randomForest)
ca_forest = randomForest(medianHouseValue ~ . - totalBedrooms - totalRooms,
                         data = ca_forest_train,
                         importance = TRUE)
forest_errs = rmse(ca_forest, ca_forest_test)
#### Boosting
ca_boost_folds = crossv_kfold(ca_train, k=5)
library(gbm)
boost_models = map(ca_boost_folds$train, ~ gbm(medianHouseValue ~ . - totalRooms - totalBedrooms,
                                               data =.,
                                               interaction.depth = 6, n.trees = 600, shrinkage = 0.05))
boost_errs = map2_dbl(boost_models, ca_boost_folds$test, modelr::rmse)
boost_errs = mean(boost_errs)
```

We can see that the best model from each category has an out-of-sample error based on their own train-test splits. For example, we have an average RMSE from the linear model of `r linear_errs` or an average RMSE of `r boost_errs` from the boosted model. However, these RMSE's are from train-test splits built from the original train-test split. In order to compare RMSE's across model categories, we return to our first train-test split to recover RMSE's and determine the best predictive model.

```{r echo=FALSE,results=FALSE,warning=FALSE,message=FALSE}
#### Comparing to full train-test split
# Linear
final_linear = lm(medianHouseValue ~. - totalBedrooms - totalRooms + medianIncome*housingMedianAge + longitude*latitude,
                  data = ca_train)
linear_rmse = rmse(final_linear, data = ca_test)
# KNN
X = ca_train[,-(10:11)]
X = scale(X, center=TRUE, scale=TRUE)
mu = attr(X,"scaled:center")
sigma = attr(X,"scaled:scale")
Y = ca_test[,-(10:11)]
Y = (Y[,]-mu)/sigma
final_knn = knnreg(medianHouseValue ~ ., data = X, k=15)
knn_rmse = rmse(final_knn, Y)
# CART
final_cart = rpart(medianHouseValue ~ .- totalRooms - totalBedrooms, data =ca_train, 
                   control = rpart.control(cp = .0001, minsplit = 10))
final_cart = prune_1se(final_cart)
cart_rmse = rmse(final_cart, data=ca_test)
# Random Forest
final_forest = randomForest(medianHouseValue ~ . - totalBedrooms - totalRooms,
                         data = ca_train,
                         importance = TRUE)
forest_rmse = rmse(final_forest, ca_test)
# Boosting
final_boost = gbm(medianHouseValue ~ . - totalRooms - totalBedrooms,
                  data =ca_train,
                  interaction.depth = 6, n.trees = 600, shrinkage = 0.05)
boost_rmse = rmse(final_boost, ca_test)
```

Comparing out-of-sample errors across models, we see that the boosting model and the random forest perform quite similarly, with RMSE's of `r boost_rmse` and `r forest_rmse` respectively. This is compared to something like the linear model which yields `r linear_rmse` and we see that both the boosting and random forests are considerable improvements. The CART and KNN models fall somewhere in between and we include their RMSE values for completeness, CART has an RMSE of `r cart_rmse` and KNN yields `r knn_rmse`. Finally, we will move forward with the boosting as the best predictive model and create our plots.



```{r echo=FALSE,results=FALSE,warning=FALSE,message=FALSE}
#### Plots for winner
ca_winner = final_boost
test = CAhousing%>%
  mutate(predictedHouseValue = predict(ca_winner, CAhousing))%>%
  mutate(Residuals = medianHouseValue - predictedHouseValue)
library(maps)
library(ggmap)
qmplot(longitude, latitude, color = medianHouseValue,
       data = CAhousing)+
  scale_color_continuous(type = 'viridis')
qmplot(longitude, latitude, color = predictedHouseValue,
       data = test)+
  scale_color_continuous(type = 'viridis')
qmplot(longitude, latitude, color = Residuals,
       data = test)+
  #scale_color_gradient2(low = 'black',mid = 'red')
  scale_color_continuous(type = 'viridis')
```
tinytex::reinstall_tinytex()