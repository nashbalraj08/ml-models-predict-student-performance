############################## Data Exploration #####################################

#install dependencies 
# for reading excel files
install.packages("readxl")
library(readxl)
# for encoding
install.packages("dummy") 
library(dummy)
install.packages("ggplot2") 
# for creating visuals
library(ggplot2)
install.packages("caTools") 
# for splitting of training/testing set
library(caTools) 
install.packages("dplyr")
# for data pre-processing and manipulation
library(dplyr) 
install.packages("glmnet") 
# for regularization methods
library(glmnet) 
install.packages("corrplot")
install.packages("ggcorrplot")
# for creating enhanced visuals e.g. heat map
library(corrplot)
library(ggcorrplot)
# compare the performance metrics of model
install.packages("MLmetrics")
library(MLmetrics)
install.packages("caret")
#this package allows us to perform both classification and regression. Elastic model
library(caret) 
#Recursive Partitioning and Regression Trees
install.packages("rpart")
library(rpart) 
# We can visualize our model with rpart.plot
install.packages("rpart.plot")
library(rpart.plot)
# this package will allow us to visualize the NN
install.packages("NeuralNetTools")
library(NeuralNetTools)  

# import the excel file
student_data <- read_excel("Students Performance Data.xlsx") # your directory
str(student_data)
"OUTPUT:
649 observations
23 attributes : 10 numerical attributes and 13 categorical attributes
"

#explore the distribution of the target attribute (Final_Grade)
# Summary statistics
summary(student_data$Final_grade)
sd(student_data$Final_grade)
# Create a histogram for the 'Final Grade' attribute 
ggplot(student_data, aes(x = Final_grade)) +
  geom_histogram(binwidth = 1, fill = "steelblue", color = "black") +
  geom_text(stat = "count", aes(label = ..count..), vjust = -0.1) + # Add the counts above the bars
  labs(title = "Distribution of Student Final Grades", x = "Final Grade", y = "Frequency") +
  theme_minimal()

"OUTPUT:
The summary statistics suggest a reasonable spread of final grades with some 
moderate skew due to the extreme values(students who performed either very poorly 
or exceptionally well),with grades ranging from 0 to 19.

The mean (11.91) and median (12) are quite close, which indicates a symmetric
distribution. However, the fact that the min is 0 and the max is 19, and the 
1st quartile (10) and 3rd quartile (14) points span a 
reasonable range, but there may be a slight skew.

The standard deviation (3.2) indicates a moderate spread around the mean 
(11.91). This means most grades tend to be within (+-)3.2 of the average.

The interquartile range (4) suggests that 
the middle 50% of students' grades are relatively tightly packed, which indicates 
a moderate concentration of grades around the median.

According to histogram
The mean, median, and the most common grades cluster around 9 to 12, 
indicating that most students performed in this range.

Slight Left Skew: The slight left skew in the distribution indicates that there 
are some lower outliers (students with grades of 0-5 points, but the bulk of students 
performed around the middle range.

High Achievers: A smaller group of students scored near the maximum (19 points), 
showing that some students performed exceptionally well.
"

#boxplot for target attribute
boxplot(student_data$Final_grade, main="Box plot of Final Grades", 
        ylab="Final Grade",xlab="Students", col="steelblue")

"OUTPUT:
The black line within the box represents the median final grade,This indicates 
that half of the students scored less than or equal to 12, and other half scored higher.

Q1: The lower edge of the box corresponds to the first quartile (Q1), 
meaning 25% of students scored below 10 points.
Q3: The upper edge of the box represents the third quartile (Q3), 
meaning 75% of students scored below 14 points

Min (0): The minimum grade is 0 points, which could represent students who 
either failed or did not participate.
Max (19): The maximum grade is 19, indicating that the highest-performing 
students achieved perfect scores. 
 
what level of prediction error you deem to be acceptable for the response variable 

Given this distribution, most student grades are clustered between 10 and 14. 
The mean is approximately 11.93, and the median is 12. Also, the standard deviation(3.2)
indicates a moderate spread around the mean (11.91).This means most grades tend 
to be within (+-)3.2 points of the average.
Errors larger than 4 points, as this is higher than IQR range is unacceptable as they 
could represent a considerable gap relative to the performance indicators within 
this dataset.
Therefore, 2 points of prediction error is acceptable for the response variable,
Final Grade.


"


#boxplot for target attribute Final grade vs Student Health
ggplot(student_data, aes(x = factor(health), y = Final_grade)) +
  geom_boxplot(fill = "steelblue") +
  labs(title = "Box plot of Student Health", x = "Health Status", y = "Final Grade") +
  scale_x_discrete(breaks = 1:5, labels = c("1(Very bad)", "2", "3", "4", "5(very good)")) +  # Adjust x-axis labels
  theme_minimal()

#scatter plot
ggplot(student_data, aes(x = freetime, y = Final_grade)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +  # Add a trend line
  labs(title = "Scatter plot of Student freetime vs Final Grade",
       x = "freetime",
       y = "Final Grade")+
  scale_x_continuous(breaks = 1:5, labels = c("1(Very low)", "2", "3", "4", "5(very high)")) +  # Adjust x-axis labels
  theme_minimal()

"
How complex do the relationships in your data appear to be?

After exploring the dataset, it is clear that the relationships between the 
variables are complex. This complexity arises from the combination of both 
categorical and numerical variables, and how they relate to the target 
variable(final grade). While comparing each attribute individually 
to the final grade shows some degree of relationship, 
evaluating the attributes one by one doesn’t provide a comprehensive picture.

For example, one might assume that students with good health would perform 
better academically. However, the box plot analysis reveals that there is no 
clear correlation between better health and higher grades. 
In fact, students with poorer health statuses (rated 1 and 2) 
show competitive median grades compared to those with better health statuses.
This suggests that although health may influence academic performance, 
it may not be as strong or direct a predictor as initially expected. 
Other factors such as student freetime, extra academic support, 
or studytime might play more significant roles.

Another assumption could be that students with more free time after school 
would dedicate this time to their studies and achieve higher grades. However, 
the scatter plot with a trend line indicates a slight negative slope. 
This suggests that students with more free time may, in fact, have slightly 
lower final grades, though the trend is subtle. Therefore, analyzing 
variables in isolation makes it difficult to draw meaningful insights, and 
other factors likely contribute to student performance.

In conclusion, the relationships in my dataset appear to be highly complex, 
and it is challenging to derive clear interpretations by examining each 
variable individually. Hopefully, using machine learning models, may provide 
deeper insights into what influences student performance.

"


############################## Fitting OLS Models #####################################
#no data pre-processing techniques are needed to fit ols models
#we use the same seed so we can use same train and test split
#ensure that we can compare models
set.seed(123)
split=sample.split(student_data$Final_grade,SplitRatio = 0.7)
"OUTPUT:
use 70-30 split, since medium size dataset with many attributes(complex),overfitting is a risk.
This means that 70% of the data is used for training the model, and 30% is set aside 
to test the model's performance on unseen data.
Since, we be using the regularization methods help by penalizing large coefficients 
and reducing overfitting.
Including additional steps like cross-validation and feature selection will 
improve model accuracy and reduce bias/variance issues.

"
#all true values --> training set 
training_set=subset(student_data,split==TRUE) 
#all false --> test set
test_set=subset(student_data,split==FALSE)
"output
Trainining set is used to predict the model
Test set is used to evalaute the model
"

summary(training_set$Final_grade)
sd(training_set$Final_grade)

summary(test_set$Final_grade)
sd(test_set$Final_grade)
"output:
Hardly any difference, which is good indicator both training and test set 
distribution are similar.Also, given that we have medium size dataset, values
are similar.

Understanding distribution of target variable the average final grade is around 
11 for both training and test set.

"
#baseline model
OLS_model <- lm(Final_grade~.,training_set) 
#no penalties, no regularization, no lambda, no hyperparamters,no need to perform cross validation
summary(OLS_model)
"OUTPUT:
Since lm model is from STATS packagae we can perform inference but in regularization methods
we cannot perform inference since doesnt output p-values
We dont need to standardised attributes in OLS model

Do the relationships between the predictors and your target appear to be linear or 
non-linear?

 the R-squared value of 0.37421 in the OLS model. This indicates a reasonably strong non-linear 
 relationship between the attributes and the target variable. High bias,resulting in undefitting
 OLS model is not the ideal model. Therefore, performing LASSO, RIDGE and Elastic net is not ideal
 since there try to improve on the linear model. 

"
plot(training_set$Final_grade, resid(OLS_model), 
     main = "Residual Plot", 
     xlab = "Final Grade (Target Variable)", 
     ylab = "Residuals")
"OUTPUT:
Based on the residual plot, there seems to be a non-linear relationship between
the predictors and the target variable (Final Grade).
Ideally, in a linear model, the points should be scattered randomly around the zero 
line. In your plot, the points show clusters and patterns, which means the linear 
model might not be capturing the true relationship between variables
"

#use ols model to make predictions 
# find the MAE RMSE, which compare the predicted target value to actual target values
OLS_pred_train=predict(OLS_model, newdata =training_set) 
OLS_pred_test=predict(OLS_model, newdata = test_set)

## Performance on training set:
# MAE - Mean Absolute Error (predicted values, actual values)
MAE(OLS_pred_train,training_set$Final_grade)   
# RMSE - Root Mean Square Error
RMSE(OLS_pred_train,training_set$Final_grade) 
# R2_Score (unadjusted though!)
R2_Score(OLS_pred_train,training_set$Final_grade) 

## Performance on test set:
#length(OLS_pred_test)
#length(test_set$Final_grade)

MAE(OLS_pred_test,test_set$Final_grade)
RMSE(OLS_pred_test,test_set$Final_grade)
R2_Score(OLS_pred_test,test_set$Final_grade)

"OUTPUT:
We use the test set to see if the data generalizes well with the data by 
predicting the value on test set
Since the r2_score in test set(0.115) is lower than training set(0.37), we expect this due 
to generalization error. 

MAE as we expect the MAE error on test set 2.13 is higher training test set 1.87,
This indicates that the model performs slightly better on the training data 
compared to unseen data (test set). 


Bias: training performance (with an MAE of 1.8) is relatively close to the 
test performance (MAE of 2.13), indicating that the model doesn't have high bias.

Variance: training and test errors (MAE) are reasonably close, 
which suggests that the model doesn't suffer from high variance.we can conclude 
that the model generalizes well to unseen data.

"
############################## Data preprocessing #####################################
#check if there is missing values
sum(is.na(student_data)) 
"OUTPUT:
NO missing values
"

# create data frame for cleaned data, so doesn't overwrite existing data 
student_data_cleaned <- student_data 
#convert categorical to factors 
student_data_cleaned$sex<- as.factor(student_data_cleaned$sex)
student_data_cleaned$address <- as.factor(student_data_cleaned$address )
student_data_cleaned$famsize <- as.factor(student_data_cleaned$famsize )
student_data_cleaned$Pstatus <- as.factor(student_data_cleaned$Pstatus )
student_data_cleaned$Medu <- as.factor(student_data_cleaned$Medu )
student_data_cleaned$Fedu <- as.factor(student_data_cleaned$Fedu)
student_data_cleaned$Mjob <- as.factor(student_data_cleaned$Mjob )
student_data_cleaned$Fjob <- as.factor(student_data_cleaned$Fjob )
student_data_cleaned$schoolsup  <- as.factor(student_data_cleaned$schoolsup  )
student_data_cleaned$activities <- as.factor(student_data_cleaned$activities )
student_data_cleaned$nursery <- as.factor(student_data_cleaned$nursery )
student_data_cleaned$higher <- as.factor(student_data_cleaned$higher )
student_data_cleaned$internet <- as.factor(student_data_cleaned$internet )
str(student_data_cleaned)
"OUTPUT:
converted 13 categorical attributes into Factors before any dummy recording.
"
#converting numerical attributes into factors, since each number represents a qualitative category
#This will give meaningful names to each level and improve the interpretability
#reduce the levels since of high dimensional in my data
student_data_cleaned$traveltime
student_data_cleaned$traveltime <- factor(student_data_cleaned$traveltime, 
                                          levels = c(1, 2, 3, 4),
                                          labels = c("<30 min", "<30 min", "30 min to 1 hour", "1 hour"))
student_data_cleaned$traveltime

student_data_cleaned$studytime
student_data_cleaned$studytime <- factor(student_data_cleaned$studytime, 
                                         levels = c(1, 2, 3, 4),
                                         labels = c("<5 hours", "<5 hours", "5 to 10 hours", ">10 hours"))
student_data_cleaned$studytime

student_data_cleaned$failures
student_data_cleaned$failures <- factor(student_data_cleaned$failures, 
                                        levels = c(0, 1, 2, 3),
                                        labels = c("no class failures", "1 class", "2 classes", "3 classes"))
student_data_cleaned$failures
"output: did not reduce the levels since, this is valuable information and reducing the 
levels will create loss of information, which could weaken the predictive power 
of the models that depend on this variable."
student_data_cleaned$famrel 
student_data_cleaned$famrel <- factor(student_data_cleaned$famrel, 
                                      levels = c(1, 2, 3, 4, 5),
                                      labels = c("Bad", "Bad", "Average", "Excellent", "Excellent"))
student_data_cleaned$famrel 

student_data_cleaned$freetime 
student_data_cleaned$freetime <- factor(student_data_cleaned$freetime, 
                                        levels = c(1, 2, 3, 4, 5),
                                        labels = c("Low", "Low", "Average", "High", "High"))

student_data_cleaned$freetime 

student_data_cleaned$goout 
student_data_cleaned$goout <- factor(student_data_cleaned$goout, 
                                     levels = c(1, 2, 3, 4, 5),
                                     labels = c("Low", "Low", "Average", "High", "High"))

student_data_cleaned$goout 

student_data_cleaned$health 
student_data_cleaned$health <- factor(student_data_cleaned$health, 
                                      levels = c(1, 2, 3, 4, 5),
                                      labels = c("Bad", "Bad", "Average", "Good", "Good"))
student_data_cleaned$health 
"OUTPUT: converted above 8 numeric attributes into factors and adjusted levels to reduce
high dimensionality in my data for the appopriate variables"

# fixing inconsistencies 
# #1 Sex : attributes groups into M and F, 
student_data_cleaned$sex
levels(student_data_cleaned$sex)
# Change the levels of the sex attribute
levels(student_data_cleaned$sex) <- c("Female", "Male")
# Verify the changes
levels(student_data_cleaned$sex)
student_data_cleaned$sex

"OUTPUT:
Adjust m->male and f->female
Changing M to Male and F to Female helps to make  data more 
interpretability, there is no loss of information.
"
# #2 Mother and Father job status
levels(student_data_cleaned$Mjob)
levels(student_data_cleaned$Mjob) <- c("at home", "health","other","services","teacher")
levels(student_data_cleaned$Mjob)

levels(student_data_cleaned$Fjob)
levels(student_data_cleaned$Fjob) <- c("at home", "health","other","services","teacher")
levels(student_data_cleaned$Fjob)
"
at_home--> at home
it makes  data more consistent and readable, especially since R can 
handle attribute names with spaces in strings
"
str(student_data_cleaned)
"OUTPUT:
649 observations
23 attributes : 3 numerical attributes and 20 categorical/factor attributes
"


################################ Ridge Regression #############################

set.seed(123)
split=sample.split(student_data_cleaned$Final_grade,SplitRatio = 0.7)
#all true values --> training set 
training_set=subset(student_data_cleaned,split==TRUE) 
#all false --> test set
test_set=subset(student_data_cleaned,split==FALSE)
"output
Trainining set is used to predict the model
Test set is used to evalaute the model
"

# attributes be incorporated in the model in the form of a matrix, and that the target be in the form of a vector 
X_train <- as.matrix(select(training_set,-Final_grade))
class(X_train)
Y_train <- training_set$Final_grade

X_test <- as.matrix(select(test_set,-Final_grade))
class(X_test)
Y_test <- test_set$Final_grade

#determine the optimal value of lambda by cross validation
#cross validation error is the average value generated from all  validation sets
#-3 lower limit (as long its close to 0)
#5 upper limit
lambdas_to_try <- 10^seq(-3, 5, length.out = 100) 
set.seed(123) #Also need to set the seed before running each cross-validation to ensure comparability of results between models by eliminating sampling bias during cross-validation
#any postie value of lambda is feasible 
#we set alpha=0 for ridge regression
#ridge_cv <- cv.glmnet(X_train, Y_train, type.measure="rmse",alpha = 0, lambda = lambdas_to_try, standardize = TRUE, nfolds = 5)
#standardized=TRUE, perform standardization on the attributes except target variable 
ridge_cv <- cv.glmnet(X_train, Y_train,alpha = 0, lambda = lambdas_to_try, standardize = TRUE, nfolds = 10)
# Plot cross-validation results
plot(ridge_cv)
ridge_cv  # see what the value of lambda is that produces a min error

"Output:
Used 10 fold cross validation since dataset is medium size and gives a more 
accurate estimate of the model’s performance by using more data for training.

1.17 optimal value of lambda that minimizes cross validation error

"

# Extract the best cross-validated lambda value in an object called lambda_cv
lambda_cv <- ridge_cv$lambda.min
# Fit final model on FULL training set with optimal lambda value:
ridge_reg <- glmnet(X_train, Y_train, alpha = 0, lambda = lambda_cv, standardize = TRUE)
# Look at the estimated regression coefficients for ridge regression:
ridge_reg$beta
coef(ridge_reg)
summary(OLS_model)

"OUTPUT:
Alot of attributes in the model have dots, zero (or near-zero) coefficients 
have been shrunk to zero, meaning they are not contributing significantly to 
the prediction.

OLS MODEL --> absences -0.218
RIDGE MODEL --> absences 0.0408
absences have a small negative coefficient (-0.0408), 
meaning more absences are associated with a slightly lower final grade.


OLS MODEL --> age 0.14626
RIDGE MODEL --> age -0.2237 
Huge change and cannot perform inference on this model.
the coefficient for age is -0.2237, which indicates that age has a negative
association with the target variable, meaning as age increases, the target
variable decreases slightly.


However, can identify important variables (high estimates) and the ones closer to
zero have been penalised are less important in predicting Final_grades_
"

# Evaluate Ridge model 
# make predictions for training and test sets:
length(ridge_reg)
length(X_train)
Y_ridge_pred_train=predict(ridge_reg,X_train)
length(ridge_reg)
length(X_test)
Y_ridge_pred_test=predict(ridge_reg,X_test)

## Performance on training set:

MAE(Y_ridge_pred_train,Y_train)   # MAE - Mean Absolute Error 
RMSE(Y_ridge_pred_train,Y_train)  # RMSE - Root Mean Square Error
R2_Score(Y_ridge_pred_train,Y_train) # R-Squared (Coefficient of Determination) Regression Score (unadjusted)

## Performance on test set:

MAE(Y_ridge_pred_test,Y_test)
RMSE(Y_ridge_pred_test,Y_test)
R2_Score(Y_ridge_pred_test,Y_test)

"OUTPUT:

MAE as we expect the MAE error on test set 2.41 is higher than training set 1.87,
This indicates that the model performs slightly better on the training data 
compared to unseen data (test set). 


Bias: training performance (with an MAE of 1.87) is relatively close to the 
test performance (MAE of 2.41), indicating that the model doesn't have high bias.
In other words, the model captures the relationship between the features and the 
target variable fairly well.

The close MAE values between the training and test sets suggest that the model 
does not suffer from high variance.we can conclude that the model generalizes 
well to unseen data. 


  "

################################ LASSO Regression#############################
set.seed(123)
split=sample.split(student_data_cleaned$Final_grade,SplitRatio = 0.7)
#all true values --> training set 
training_set=subset(student_data_cleaned,split==TRUE) 
#all false --> test set
test_set=subset(student_data_cleaned,split==FALSE)

# attributes be incorporated in the model in the form of a matrix, and that the target be in the form of a vector 
X_train <- as.matrix(select(training_set,-Final_grade))
class(X_train)
Y_train <- training_set$Final_grade

X_test <- as.matrix(select(test_set,-Final_grade))
class(X_test)
Y_test <- test_set$Final_grade
#determine the optimal value of lambda by cross validation
#cross validation error is the average value generated from all  validation sets
#-3 lower limit (as long its close to 0)
#5 upper limit
lambdas_to_try <- 10^seq(-3, 5, length.out = 100) 
#any postie value of lambda is feasible 
set.seed(123) #Also need to set the seed before running each cross-validation to ensure comparability of results between models by eliminating sampling bias during cross-validation
#Aplha --> 1 ---> mixing parameter
#Cross validation for lasso to determine optimal value for lambda
lasso_cv <- cv.glmnet(X_train, Y_train, alpha = 1, lambda = lambdas_to_try, standardize = TRUE, nfolds = 10)
plot(lasso_cv)
lambda_cv_lasso <- lasso_cv$lambda.min
# Fit final model on FULL training set with optimal lambda value:
lasso_reg <- glmnet(X_train, Y_train, alpha = 1, lambda = lambda_cv_lasso, standardize = TRUE)
lasso_reg
# Look at the estimated regression coefficients for LASSO regression:
coef(lasso_reg)
"OUTPUT:
. --> attributes THERE been completely zero, meaning some variables are highly 
      correlated to one another(multicolinearity) and which varaibles are not important 
      in predicting the target variable (Final_grade)
LASSO used for variable selection
"


# make predictions for training and test sets:

Y_lasso_pred_train=predict(lasso_reg,X_train)
Y_lasso_pred_test=predict(lasso_reg,X_test)

## Performance on training set:

MAE(Y_lasso_pred_train,Y_train)   # MAE - Mean Absolute Error 
RMSE(Y_lasso_pred_train,Y_train)  # RMSE - Root Mean Square Error
R2_Score(Y_lasso_pred_train,Y_train) # R-Squared (Coefficient of Determination) Regression Score

## Performance on test set:

MAE(Y_lasso_pred_test,Y_test)
RMSE(Y_lasso_pred_test,Y_test)
R2_Score(Y_lasso_pred_test,Y_test)

################################## Elastic Net ################################
set.seed(123)
split=sample.split(student_data_cleaned$Final_grade,SplitRatio = 0.7)
#all true values --> training set 
training_set=subset(student_data_cleaned,split==TRUE) 
#all false --> test set
test_set=subset(student_data_cleaned,split==FALSE)
set.seed(123) #Also need to set the seed before running each cross-validation to ensure comparability of results between models by eliminating sampling bias during cross-validation
# Performing Cross validation 
train_control <- trainControl(method = "repeatedcv",
                              number = 10, # no. of folds
                              repeats = 1,
                              search = "random",
                              verboseIter = TRUE)

elastic_net <- train(Final_grade ~ .,
                     data = training_set,
                     method = "glmnet",
                     preProcess = c("center", "scale"),#standardized attributes
                     tuneLength = 25,
                     trControl = train_control) 

"output:
aplha=0.49 value is closer to 0, so its similar to ridge Model

"
# make predictions for training and test sets:

Y_elastic_pred_train=predict(elastic_net,training_set)
Y_elastic_pred_test=predict(elastic_net,test_set)
## Performance on training set:

MAE(Y_elastic_pred_train,training_set$Final_grade)   # MAE - Mean Absolute Error 
RMSE(Y_elastic_pred_train,training_set$Final_grade)  # RMSE - Root Mean Square Error
R2_Score(Y_elastic_pred_train,training_set$Final_grade) # R-Squared (Coefficient of Determination) Regression Score

## Performance on test set:

MAE(Y_elastic_pred_test,test_set$Final_grade)
RMSE(Y_elastic_pred_test,test_set$Final_grade)
R2_Score(Y_elastic_pred_test,test_set$Final_grade)


##################### Compare Parametric Models #######################

# The following code combines the results of OLS (using glmnet), ridge, lasso and elastic net for the training
# and test set into a data frame so that it can be graphed

rows_OLS= rbind(MAE(OLS_pred_train,Y_train),   
                RMSE(OLS_pred_train,Y_train),  
                R2_Score(OLS_pred_train,Y_train),
                
                MAE(OLS_pred_test,Y_test),
                RMSE(OLS_pred_test,Y_test),
                R2_Score(OLS_pred_test,Y_test))

OLS_results=cbind(rep("OLS",6),c("MAE","RMSE","R2","MAE","RMSE","R2"),rows_OLS,c(rep("Training",3),rep("Test",3)))

rows_ridge= rbind(MAE(Y_ridge_pred_train,Y_train),   
                  RMSE(Y_ridge_pred_train,Y_train),  
                  R2_Score(Y_ridge_pred_train,Y_train),
                  
                  MAE(Y_ridge_pred_test,Y_test),
                  RMSE(Y_ridge_pred_test,Y_test),
                  R2_Score(Y_ridge_pred_test,Y_test))


ridge_results=cbind(rep("Ridge",6),c("MAE","RMSE","R2","MAE","RMSE","R2"),rows_ridge,c(rep("Training",3),rep("Test",3)))

rows_lasso= rbind(MAE(Y_lasso_pred_train,Y_train),   
                  RMSE(Y_lasso_pred_train,Y_train),  
                  R2_Score(Y_lasso_pred_train,Y_train),
                  
                  MAE(Y_lasso_pred_test,Y_test),
                  RMSE(Y_lasso_pred_test,Y_test),
                  R2_Score(Y_lasso_pred_test,Y_test))


lasso_results=cbind(rep("Lasso",6),c("MAE","RMSE","R2","MAE","RMSE","R2"),rows_lasso,c(rep("Training",3),rep("Test",3)))

#Note: the mpg values in the vector Y_train will be the same as the mpg values in training_set$mpg

rows_elastic= rbind(MAE(Y_elastic_pred_train,Y_train),   
                    RMSE(Y_elastic_pred_train,Y_train),  
                    R2_Score(Y_elastic_pred_train,Y_train),
                    
                    MAE(Y_elastic_pred_test,Y_test),
                    RMSE(Y_elastic_pred_test,Y_test),
                    R2_Score(Y_elastic_pred_test,Y_test))

elastic_results=cbind(rep("Elastic_Net",6),c("MAE","RMSE","R2","MAE","RMSE","R2"),rows_elastic,c(rep("Training",3),rep("Test",3)))

results=data.frame(rbind(OLS_results, ridge_results,lasso_results, elastic_results))
colnames(results) <- c("Method","Measure","Result","Data_Set")

results$Method <- factor(results$Method, levels = c("OLS", "Ridge", "Lasso", "Elastic_Net"), ordered = TRUE)
results$Measure <- factor(results$Measure)
results$Data_Set <- factor(results$Data_Set,levels = c("Training", "Test"), ordered = TRUE)
results$Result <- as.numeric(results$Result)




ggplot(results, aes(y=Result,x=Measure,fill=Data_Set))+scale_fill_manual(values=c("steelblue", "slategrey"))+ geom_col(position = "dodge") +
  geom_text(
    aes(label = round(Result,2)),
    size = 2.8,
    vjust = 0.0, position = position_dodge(.9)
  )+  facet_wrap(~Method)

"OUTPUT:
OLS: Training (1.82) and Test (2.12) both lie within the 2-point criterion. 
  The model performs reasonably well in predicting with minimal deviation.
Ridge: Both Training (2.36) and Test (2.42) exceed the 2-point criterion, 
  showing a slightly higher prediction error, but are still relatively close 
  to each other.
Lasso: Training (2.36) and Test (2.42) also exceed the criterion, with
  minimal difference between them, showing consistent errors across both sets.
Elastic Net: Training (1.88) is within the criterion, but Test (2.09) 
  slightly exceeds it. The model performs well on training data but shows a 
  bit higher error on unseen data.

The models slightly exceed the acceptance level of 2 points. 
OLS and Elastic Net perform better than Ridge and Lasso, but none are 
ideal for this dataset. complex models (e.g., non-linear models) could help 
improve performance.

"



##################### Regression Tree #######################
set.seed(123)
split=sample.split(student_data_cleaned$Final_grade,SplitRatio = 0.7)
#all true values --> training set 
training_set=subset(student_data_cleaned,split==TRUE) 
#all false --> test set
test_set=subset(student_data_cleaned,split==FALSE)
#dont need to convert categorical variables to numeric
#rpart automatically runs cross-validation in the background
#By default, rpart function runs 10-fold cross validation but you can change this using the rpart.control function
set.seed(123)
RegTree <- rpart(Final_grade ~.,  data = training_set,  method  = "anova")
RegTree
"output
# output : n=451 --> number of observations in training set
# output: 1) root 451 128663.4000 11.89 
# Node --> 1), 2)...42)
# split --> the attrribute that has been split --> failures,asbsences...
# n --> 451
# deviance --> 128663.4000
# y-value --> 11.89 (the average grade per student for all 451 obser is 11.89 points )

# output : failures=1,2,3 classes 70  1000.7 8.3
# question being asked about the attribute is how many classes a student failed has influence on target variable
# n --> 70
# deviance --> 1000.7
# y-value --> 8.3 (the average grade per student for all 70 obser is 8.3 points )
"

rpart.plot(RegTree)
rpart.plot(RegTree, yesno=1,type=2,fallen.leaves = FALSE) # add additional options to change the appearance.
"output:
TOP value 12 --> is the average grade
100% --> 451 observations 
Since regression trees only allow for binary splits : 
e.g. if failures categories are 1 class, 2 classes, 3 classes and no failures
The regression tree will group this into two (binary splits) = 
0(1 class, 2 classes, 3 classes ) and 1 (no failures)
yes --> 0(1 class, 2 classes, 3 classes )
no --> 1 (no failures)

Darker the blue the higher average of student grades

The most influential factor seems to be failures, indicating that students who 
have failed multiple classes tend to have lower final grades. 
This is a reasonable outcome as past failures likely indicate struggles in 
academic performance.

Absences: The second split on the left branch considers absences as an 
important predictor. Students with fewer absences (<1) are likely to achieve a 
final grade of around 9.5 or 9.8, which is higher than students with more absences.

MJOb:On the left-most side, the split considers the mothers's job as a relevant 
variable (specifically the other category). It indicates that students whose mothers 
have a job classified as other might be linked to slightly lower grade

Medu (Mother's Education): On the right branch, Medu (the education level of the mother) 
also plays a significant role. Students whose mothers have a higher level of education 
tend to score higher final grades.

Study time also comes up as a split point, with students studying fewer than 5 
hours have lower grades compared to students with more than 5 hours have higher average grades.

Mjob=at home indicates mother at home help stdunets achieve higher average grades

"

plotcp(RegTree)
# Add custom axis labels using title()
title(xlab = "Complexity Parameter (CP)", ylab = "Relative Error")
RegTree$cptable
"OUTPUT:
THE LAST VALUE ON TOP AXIS SHOW you how many nodes tree has, terminal nodes: 10
the cross validation error started off high but soon as we split the cross validation
error dropped
Size of tree is inversely related to cp complexity penality
dashed line --> reprsents 1 standard deviation of tree with smallest cross 
validation error(0.01000)

CP Table 
  of the first column is the hyperparamter
  second column nsplit --> number of splits will always be one less than you have
  relerror 
  xerror --> cross validation error average on each fold
  Node 8 0.01000000 --> CP=0.01000000, 0.7579982 --> CROSS validation error was minimised by 0.5877696

Pruning : If tree is large and has alot of nodes,
so we move from the left of the smallest node, and see which attributes are 
within 1 standard deviation from error
Since the node 4 is within 1 standard deviation belowe the line
we take its cp value 0.01739184,to fit the alternative prune tree

"
RegTree_pruned <- rpart(Final_grade ~ ., data=training_set,  method="anova",cp= 0.01739184 ) 
RegTree_pruned
RegTree_pruned$cptable
plotcp(RegTree_pruned)

rpart.plot(RegTree_pruned, yesno=1,type=2,fallen.leaves = FALSE)
"OUTPUT:
identify Important attributes by position of attributes 
Seeking  higher education and how many failures 

The entire dataset is divided into students who had class failures (1-3 classes)
and those who did not. This is the most significant split in predicting final grades.

Class Failures and Absences are the most influential variables in predicting a student’s 
final grade. Students who had failures and many absences tend to perform worse, 
with final grades ranging from 3.3 to 7.2.

Students whose mothers have jobs categorized as other have lower predicted 
final grades of 5.6, 3.3, or 7.2

 Students who do not intend to pursue higher education are still performing 
 relatively well with a predicted grade of around 13.

Higher = yes: These students also have a predicted grade of 13, showing a 
generally higher level of performance.

Mother's Job (Mjob = other) is relevant for students who have experienced 
class failures and have minimal absences. This suggests that socioeconomic 
factors or the work status of the mother may influence student outcomes.

Desire to Pursue Higher Education is a strong predictor of better performance,
which aligns with the notion that students aiming for higher education may be 
more motivated or better prepared academically.
"

#Evalaute the model of tree and pruned tree
pred_train_RegTree = predict(RegTree,newdata=training_set) 
pred_test_RegTree = predict(RegTree,newdata=test_set) #Reminder: new data sets must always use consistent variable names and formatting to data sets on which models were trained

pred_train_RegTree_pruned = predict(RegTree_pruned,newdata=training_set) 
pred_test_RegTree_pruned = predict(RegTree_pruned,newdata=test_set)

## Performance on training set:

MAE(pred_train_RegTree,training_set$Final_grade)   # MAE - Mean Absolute Error 
RMSE(pred_train_RegTree,training_set$Final_grade)  # RMSE - Root Mean Square Error

MAE(pred_train_RegTree_pruned,training_set$Final_grade)   # MAE - Mean Absolute Error 
RMSE(pred_train_RegTree_pruned,training_set$Final_grade)  # RMSE - Root Mean Square Error

## Performance on testing set:

MAE(pred_test_RegTree,test_set$Final_grade)   # MAE - Mean Absolute Error 
RMSE(pred_test_RegTree,test_set$Final_grade)  # RMSE - Root Mean Square Error

MAE(pred_test_RegTree_pruned,test_set$Final_grade)   # MAE - Mean Absolute Error 
RMSE(pred_test_RegTree_pruned,test_set$Final_grade)  # RMSE - Root Mean Square Error


rows_RegTree= rbind(round(MAE(pred_train_RegTree,training_set$Final_grade),3),    
                    round(RMSE(pred_train_RegTree,training_set$Final_grade),3),round(MAE(pred_test_RegTree,test_set$Final_grade),3),
                    round(RMSE(pred_test_RegTree,test_set$Final_grade),3))

RegTree_results=cbind(rep("RegTree",4),c("MAE","RMSE","MAE","RMSE"),rows_RegTree,c(rep("Training",2),rep("Test",2)))

rows_RegTree2= rbind(round(MAE(pred_train_RegTree_pruned,training_set$Final_grade),3),    
                     round(RMSE(pred_train_RegTree_pruned,training_set$Final_grade),3),round(MAE(pred_test_RegTree_pruned,test_set$Final_grade),3),
                     round(RMSE(pred_test_RegTree_pruned,test_set$Final_grade),3))

RegTree2_results=cbind(rep("Pruned RegTree",4),c("MAE","RMSE","MAE","RMSE"),rows_RegTree2,c(rep("Training",2),rep("Test",2)))

results=data.frame(rbind(RegTree_results, RegTree2_results))
colnames(results) <- c("Tree","Measure","Result","Data_Set")

results$Tree <- factor(results$Tree, levels = c("RegTree", "Pruned RegTree"), ordered = TRUE)
results$Measure <- factor(results$Measure)
results$Data_Set <- factor(results$Data_Set,levels = c("Training", "Test"), ordered = TRUE)
results$Result <- as.numeric(results$Result)


ggplot(results, aes(y=Result,x=Measure,fill=Data_Set))+scale_fill_manual(values=c("steelblue", "slategrey"))+ geom_col(position = "dodge") +
  geom_text(
    aes(label = round(Result,2)),
    size = 2.8,
    vjust = 0.0, position = position_dodge(.9)
  )+  facet_wrap(~Tree)

"OUTPUT:
WHICH Tree performs better on training set --> The unpruned regression tree
RMSE for test set in unpruned tree there is high variance

The model shows relatively low MAE values on both the training and test sets, 
indicating good performance. The difference between training and test MAE is small,
suggesting that the model is not overfitting significantly.

Pruning has led to slightly higher error values, but it reduces the complexity of the model. 
"

varImp(RegTree_pruned) # we use the VarImp function to extract the overall variable importance.  

#Returns results alphabetically
#It is possible that not all variables/attributes may feature in a tree, particularly if it has been pruned











############################## Support Vector Models #####################################

#convert factors to numeric
dummies = dummy(student_data_cleaned, int=TRUE, p = "all")
#combine datasets 
student_data_cleaned_model<-data.frame(student_data_cleaned,dummies)

#remove the categorical/factor attributes
student_data_cleaned_model <- select(student_data_cleaned_model,-sex,-address,-famsize,-Pstatus,-Medu,-Fedu,
                                     -Mjob,-Fjob,-traveltime,-studytime,-failures,-freetime,
                                     -schoolsup,-activities,-nursery,-higher,-internet,-famrel,-goout,-health)


set.seed(123) #we use the same seed so we can use same train and test split
#ensure that we can compare models
#use 70-30 split, since small dataset with many attributes(complex),avoid overfitting
#This gives a larger portion of data for testing, allowing you to evaluate the model’s performance more thoroughly.
split=sample.split(student_data_cleaned_model$Final_grade,SplitRatio = 0.7)

#all true values --> training set 
training_set=subset(student_data_cleaned_model,split==TRUE) 
#all false --> test set
test_set=subset(student_data_cleaned_model,split==FALSE)

set.seed(123)# set seed for reproducity and everytime we splitting and for every model
# Run algorithms using 10-fold cross-testing (for tuning or optimizing the hyperparameters)
control <-  trainControl(method = "cv", number = 10)
svm_linear <- train(Final_grade~., data = training_set, method = "svmLinear2", preProcess=c("center", "scale"), trControl = control)

svm_linear #Prints the model output. The hyperparameter that is tuned here is C (cost parameter)
"Cost parameter(c)--> the larger c the more tolerance we have for further
outlying points and the more influence we allow them to have
on the model we use 9 folds and sample size is 406. The appropriate EPSILON was selected to be 0.5
This suggests that a moderate cost value of 0.5 strikes a good balance between 
model complexity and accuracy.
0.5 offers the best performance in terms of minimizing RMSE. It allows enough 
flexibility for margin violations without overfitting, thus improving 
generalization to unseen data
RMSE was used 

"
plot(svm_linear) #Visualise the CV error for different values of hyperparameter
"output:
The optimal value of C should give lowest cross validation error which is 0.5
"
#evaluate the model
summary(svm_linear)
pred__train_svm_linear = predict(svm_linear,newdata=training_set)
pred_test_svm_linear = predict(svm_linear,newdata=test_set)
MAE(pred__train_svm_linear,training_set$Final_grade)
RMSE(pred__train_svm_linear,training_set$Final_grade)
MAE(pred_test_svm_linear,test_set$Final_grade)
RMSE(pred_test_svm_linear,test_set$Final_grade)
"output:
402 support vectors 
TRAINING SET:
MAE --> 1.76
RMSE-->2.5

TEST SET 
MAE-->2.11
RMSE-->3.08

Cross-validation helps to find the optimal model that generalizes well to 
unseen data. By trying different epsilon values, I found that 0.5 offered 
better generalization (lower MSE and RMSE) because it maintained a good balance 
between fitting the training data closely without overfitting.


"


#increase epsilon error
set.seed(123)# set seed for reproducity and everytime we splitting and for every model
svm_linear_adjusted <- train(Final_grade~., data = training_set, method = "svmLinear2", preProcess=c("center", "scale"), trControl = control, epsilon=0.25)
svm_linear_adjusted
summary(svm_linear_adjusted)
plot(svm_linear_adjusted) #Visualise the CV error for different values of hyperparameter
pred__train_svm_linear_adjusted = predict(svm_linear_adjusted,newdata=training_set)
pred_test_svm_linear_adjusted = predict(svm_linear_adjusted,newdata=test_set)
MAE(pred__train_svm_linear_adjusted,training_set$Final_grade)
RMSE(pred__train_svm_linear_adjusted,training_set$Final_grade)
MAE(pred_test_svm_linear_adjusted,test_set$Final_grade)
RMSE(pred_test_svm_linear_adjusted,test_set$Final_grade)
"output:
EPsilon value of 0.5
402 support vectors 
TRAINING SET:
MAE --> 1.76
RMSE-->2.5

TEST SET 
MAE-->2.11
RMSE-->3.08

Epsilon value of : 0.25
327  support vectors 
TRAINING SET:
MAE --> 1.78
RMSE-->2.50

TEST SET 
MAE-->2.13
RMSE-->3.07

model with an epsilon of 0.5 is allowing more flexibility, which may help it
handle the noise in the data better than the model with a lower epsilon of 0.25.
The slightly lower prediction error with the epsilon of 0.5 suggests that this 
level of tolerance to errors is better suited for your data, striking a balance 
between bias and variance
"

#test other svm models
set.seed(123)
svm_rbf <- train(Final_grade~., data = training_set, method = "svmRadial",preProcess=c("center", "scale"),trControl = control,epsilon=0.5)
svm_rbf

pred__train_svm_rbf = predict(svm_rbf,newdata=training_set)
pred_test_svm_rbf = predict(svm_rbf,newdata=test_set)
MAE(pred__train_svm_rbf ,training_set$Final_grade)
RMSE(pred__train_svm_rbf ,training_set$Final_grade)
MAE(pred_test_svm_rbf,test_set$Final_grade)
RMSE(pred_test_svm_rbf,test_set$Final_grade)

set.seed(123)
svm_poly <- train(Final_grade~., data = training_set, method = "svmPoly", preProcess=c("center", "scale"), trControl = control, epsilon=0.5)
svm_poly # The hyperparameters that are tuned here are C and d (the degree)

pred__train_svm_poly = predict(svm_poly,newdata=training_set)
pred_test_svm_poly = predict(svm_poly,newdata=test_set)
MAE(pred__train_svm_poly ,training_set$Final_grade)
RMSE(pred__train_svm_poly ,training_set$Final_grade)
MAE(pred_test_svm_poly,test_set$Final_grade)
RMSE(pred_test_svm_poly,test_set$Final_grade)

"
Output:
polynomial of degree 1 --> linear
polynomial of degree 2 --> x^2
"

#Analyze the residuals: To visually check how well the model handled outliers, you can plot the residuals of the predictions:
plot(residuals(svm_linear), main="Residuals of SVR Model", xlab="Index", ylab="Residuals")
abline(h=0, col="red")


##################### Compare SVM Models #######################
#svmLinear2
MAE(pred__train_svm_linear,training_set$Final_grade)
RMSE(pred__train_svm_linear,training_set$Final_grade)
MAE(pred_test_svm_linear,test_set$Final_grade)
RMSE(pred_test_svm_linear,test_set$Final_grade)

pred__train_svm_rbf = predict(svm_rbf,newdata=training_set)
pred_test_svm_rbf = predict(svm_rbf,newdata=test_set)
MAE(pred__train_svm_rbf ,training_set$Final_grade)
RMSE(pred__train_svm_rbf ,training_set$Final_grade)
MAE(pred_test_svm_rbf,test_set$Final_grade)
RMSE(pred_test_svm_rbf,test_set$Final_grade)

pred__train_svm_poly = predict(svm_poly,newdata=training_set)
pred_test_svm_poly = predict(svm_poly,newdata=test_set)
MAE(pred__train_svm_poly ,training_set$Final_grade)
RMSE(pred__train_svm_poly ,training_set$Final_grade)
MAE(pred_test_svm_poly,test_set$Final_grade)
RMSE(pred_test_svm_poly,test_set$Final_grade)

#epsilon 0.25
pred__train_svm_linear_adjusted = predict(svm_linear_adjusted,newdata=training_set)
pred_test_svm_linear_adjusted = predict(svm_linear_adjusted,newdata=test_set)
MAE(pred__train_svm_linear_adjusted,training_set$Final_grade)
RMSE(pred__train_svm_linear_adjusted,training_set$Final_grade)
MAE(pred_test_svm_linear_adjusted,test_set$Final_grade)
RMSE(pred_test_svm_linear_adjusted,test_set$Final_grade)

rows_svm_linear= rbind(
  MAE(pred__train_svm_linear,training_set$Final_grade),
  RMSE(pred__train_svm_linear,training_set$Final_grade),
  MAE(pred_test_svm_linear,test_set$Final_grade),
  RMSE(pred_test_svm_linear,test_set$Final_grade)
)

svm_linear_results = cbind(rep("SVM_Linear",4),
                           c("MAE","RMSE","MAE","RMSE"),
                           rows_svm_linear,
                           c(rep("Training",2), rep("Test",2)))

rows_svm_rbf = rbind(
  MAE(pred__train_svm_rbf ,training_set$Final_grade),
  RMSE(pred__train_svm_rbf ,training_set$Final_grade),
  MAE(pred_test_svm_rbf,test_set$Final_grade),
  RMSE(pred_test_svm_rbf,test_set$Final_grade)
)

svm_rbf_results = cbind(rep("svm_rbf",4),
                        c("MAE","RMSE","MAE","RMSE"),
                        rows_svm_rbf,
                        c(rep("Training",2), rep("Test",2)))

rows_svm_poly = rbind(
  MAE(pred__train_svm_poly ,training_set$Final_grade),
  RMSE(pred__train_svm_poly ,training_set$Final_grade),
  MAE(pred_test_svm_poly,test_set$Final_grade),
  RMSE(pred_test_svm_poly,test_set$Final_grade)
)

svm_poly_results = cbind(rep("svm_poly",4),
                         c("MAE","RMSE","MAE","RMSE"),
                         rows_svm_poly,
                         c(rep("Training",2), rep("Test",2)))

# Combine all results into a single data frame
results = data.frame(rbind(svm_linear_results, svm_rbf_results, svm_poly_results))
colnames(results) <- c("Method","Measure","Result","Data_Set")

# Set the correct types for plotting
results$Method <- factor(results$Method, levels = c("SVM_Linear", "svm_rbf", "svm_poly"), ordered = TRUE)
results$Measure <- factor(results$Measure)
results$Data_Set <- factor(results$Data_Set, levels = c("Training", "Test"), ordered = TRUE)
results$Result <- as.numeric(results$Result)



ggplot(results, aes(y=Result, x=Measure, fill=Data_Set)) + scale_fill_manual(values=c("steelblue", "slategrey"))+
  geom_col(position = "dodge") +
  geom_text(aes(label = round(Result, 2)), size = 2.8, vjust = 0.0, position = position_dodge(.9)) + 
  facet_wrap(~Method)
"output:
SVM_Linear:The difference between the training and test errors indicates a slight 
overfitting, as the test errors are noticeably higher. However, this model 
performs relatively well compared to the others, especially in terms of MAE.

SVM_RBF:This model has a better balance between the training and test set errors, 
suggesting that the RBF kernel provides better generalization compared to the linear model.

SVM_Polynomial:The test set errors are higher than the training set errors, but 
the overall performance is similar to the RBF kernel, indicating moderate generalization.

The SVM_Linear model has the highest test error (3.09 RMSE), indicating 
overfitting compared to the others.
SVM_RBF seems to have the best balance between training and test errors,
suggesting it generalizes well.
SVM_Polynomial performs similarly to SVM_RBF but with slightly higher test errors.
Given your acceptance criterion of 2 points for MAE, the models slightly 
exceed this threshold, but the SVM_RBF appears to provide the most stable 
performance. You might consider tuning the parameters further to improve 
generalization across models.
"

################################ K Nearest Neighbour ###########################

# We now simply change the method to "knn"

set.seed(123)
knn <- train(Final_grade~.,data = training_set, method = "knn", preProcess=c("center", "scale"), trControl = control)
knn # the hyperparameter that is tuned here is the number of nearest neighbours (k)
plot(knn) # visualize the error for different values of hyperparameter 

"OUTPUT:
WHEN k is large  you risk overfitting
when k is small you risk underfiiting
Optimal value of k is 9 
"

# Prediction of KNN:

pred__train_knn = predict(knn,newdata=training_set)
pred_test_knn = predict(knn,newdata=test_set)

#Performance on training set:

rbind("knn Train",c("MAE","RMSE"),c(round(MAE(pred__train_knn,training_set$Final_grade),3),  
                                    round(RMSE(pred__train_knn,training_set$Final_grade),3)))

## Performance on test set:

rbind("knn Test", c("MAE","RMSE"), c(round(MAE(pred_test_knn,test_set$Final_grade),3),
                                     round(RMSE(pred_test_knn,test_set$Final_grade),3)))

# Variable importance cannot be obtained using KNN

"output:
The RMSE and MAE on the training set are lower than those on the test set, 
indicating that the model performs slightly better on the training data. 
However, the difference is not large, suggesting that the model generalizes 
reasonably well to unseen data.
The difference in MAE and RMSE between the training and test sets is relatively 
small, meaning that the model is neither overfitting nor underfitting severely.
"

#################################### Neural Networks ###########################
set.seed(123)
#linout = FALSE will result in classification output and nonsensical analysis
NN <- train(Final_grade~., data=training_set, method="nnet", preProcess=c("center", "scale"), trControl=control, linout=TRUE,maxit=500)
NN   # the hyper-parameters that are tuned here is the number of nodes in the hidden 
# layer and the learning rate (lambda) also known as the decay
plot(NN) # visualise the error for different values of hyperparameters
summary(NN) # output the network topology 
plotnet(NN,rel_rsc = 2,cex_circle = 1,cex_val = 0.4) # visualise the NN
#This will adjust the scaling of the connections (rel_rsc), 
#the size of the neurons (cex_circle)
#the font size of the values on the graph (cex_val), making it easier to interpret the structure.


# Prediction of NN:
pred__train_NN = predict(NN,newdata=training_set)
pred_test_NN = predict(NN,newdata=test_set)

#Performance on training set:

rbind("NN Train",c("MAE","RMSE"),c(round(MAE(pred__train_NN,training_set$Final_grade),3),  
                                   round(RMSE(pred__train_NN,training_set$Final_grade),3)))


## Performance on test set:

rbind("NN Test", c("MAE","RMSE"), c(round(MAE(pred_test_NN,test_set$Final_grade),3),
                                    round(RMSE(pred_test_NN,test_set$Final_grade),3)))


# Variable importances for the NN:

varImp(NN) 


#################################### Compare nn,knn,svm Models ###########################



rows_knn = rbind(
  MAE(pred__train_knn,training_set$Final_grade),
  RMSE(pred__train_knn,training_set$Final_grade),
  MAE(pred_test_knn,test_set$Final_grade),
  RMSE(pred_test_knn,test_set$Final_grade)
)

knn_results = cbind(rep("KNN",4),
                   c("MAE","RMSE","MAE","RMSE"),
                   rows_knn ,
                   c(rep("Training",2), rep("Test",2)))


rows_nn = rbind(
  MAE(pred__train_NN,training_set$Final_grade),
  RMSE(pred__train_NN,training_set$Final_grade),
  MAE(pred_test_NN,test_set$Final_grade),
  RMSE(pred_test_NN,test_set$Final_grade)
)

nn_results = cbind(rep("Neural Network",4),
                         c("MAE","RMSE","MAE","RMSE"),
                         rows_nn ,
                         c(rep("Training",2), rep("Test",2)))

# Combine all results into a single data frame
results = data.frame(rbind(svm_linear_results, svm_rbf_results, svm_poly_results,nn_results,knn_results))
colnames(results) <- c("Method","Measure","Result","Data_Set")

# Set the correct types for plotting
results$Method <- factor(results$Method, levels = c("SVM_Linear", "svm_rbf", "svm_poly","Neural Network","KNN"), ordered = TRUE)
results$Measure <- factor(results$Measure)
results$Data_Set <- factor(results$Data_Set, levels = c("Training", "Test"), ordered = TRUE)
results$Result <- as.numeric(results$Result)

# Plot the results
ggplot(results, aes(y=Result, x=Measure, fill=Data_Set)) + 
  geom_col(position = "dodge") + scale_fill_manual(values=c("steelblue", "slategrey"))+
  geom_text(aes(label = round(Result, 2)), size = 2.8, vjust = 0.0, position = position_dodge(.9)) + 
  facet_wrap(~Method)

