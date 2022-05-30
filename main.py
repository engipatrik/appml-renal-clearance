from classes import Data
data = Data("Data/varma_PC_properties.txt", "Data/Varma2009_SI.xls", "Data/transporterData.xlsx")

# Testing the two separate methods which don't run upon instantiation 
print(data.processed_descriptors.describe())

# Class sizes plots binary
data.summarise_predictors(True)

# Three class sizes
data.summarise_predictors(False)

# Correlation heatmap
data.find_correlation()

from classes import Model

# Creating the model object 
model = Model(data.X, data.y, data.X_train, data.y_train, data.X_test, data.y_test)

# Training and evaluating a random forest model with default parameters
model.train_predict_rfc()

# Leave one out cross validation score 
model.leave_one_out()

# Feature elimination
model.rfecv()

# Optimisation for random forest using random serach
model.random_cv_optimise()

# Showing the new results 
model.evaluate()

# Reset the model to get rid of set hyperparameters 
model2 = Model(data.X, data.y, data.X_train, data.y_train, data.X_test, data.y_test)

# Bayesian optimisation 
model2.bayesian_optimisation()

# Showing the new results 
model2.evaluate()

# Part of evaluation
model2.leave_one_out()

# Plot showing feature importances
model.plot_feature_importance()

model3 = Model(data.X, data.y, data.X_train, data.y_train, data.X_test, data.y_test)

# Creating support vector machine model
model3.train_predict_svm()

# Checking normalised and standardised data performance
model3.compare_datasets_svm(data.X_train_norm, data.X_test_norm, data.X_train_stand, data.X_test_stand)

# Showing performance for different kernels
model3.svm_kernel_select(data.X_train_norm, data.X_test_norm)

# Parameter optimisation
model3.svm_optimise(data.X_train_norm, data.X_test_norm)

# Ridge classifier test 
model.ridge_classifier()
model.evaluate()

