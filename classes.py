# A starting list of the clases that will be included 
import pandas as pd

class Data: 
    

    """
    A class used for reading the Varma renal clearance dataset as well as two sets of physchem properties.
    To instentiate just provide the path for the three files, this should handle the basic preprocessing 
    of the data. 
    The class is designed so that by instantiating the class, all of the callable attribues are created automatically. 
    This greatly reduces the effort required to set up a dataset for modelling, also different versions of the data are 
    generated (normalised/standardised). So, these can easily be compared.
    """
    
    # As mentioned there are three necessary inputs 
    def __init__(self, raw_descriptor_xlsx_path, raw_predictor_xlsx_path, transporter_path):
        
        # Storing the paths
        self.descriptor_path = raw_descriptor_xlsx_path
        self.predictor_path = raw_predictor_xlsx_path
        self.transporter_path = transporter_path

        # Processes the descriptors, creates processed_descriptors
        self.process_descriptors()

        # Processes the predictors, creates processed_predictors
        self.process_predictors()

        # Performs the join between the descriptors and predictors, working_data produced in data frame format
        self.generate_full_dataset()

        # Splitting into descriptor and predictor classifiers and using common X,y notation 
        self.x_y_notation()

        # Splitting for training and testing data, still keeping with X and y notation to produce: X_train, X_test, y_train, y_test
        self.train_test_split()

        # Creating the same train/test variables but with normalised data: X_train_norm etc.
        self.create_normalised_data()

        # Same as above but standardised: X_train_stand etc.
        self.create_standardised_data()



    def process_predictors(self):
        
        # Reading the excel file as a data frame
        classDf = pd.DataFrame(pd.read_excel(self.predictor_path))

        # The goal is to prepare both datasets for an eventual join so differences need to be fixed
        classDf = (
            classDf.drop(['Reference', 'Therapeutic Area','No.','CAS #' ], axis=1) #we don't need the values from this column
            .drop([63,304,311,313,372])   #we don't have the physchem properties for these 
            .replace(['Amiodarone '],'Amiodarone') #the space at the end messes up the join 
        )

        # Spacing and symbols in the column names make them hard to work with so they are changed
        classDf.rename(columns = {'CLtotal (mL/min/kg)':'CLtotal','CLr  (mL/min/kg)':'CLr', 'Renal Class Secretion/reabsorption':'Class'}, inplace = True)

        # Assinging the processed data frame as an attibute
        self.processed_predictors = classDf
        print("CHECK - processed predictors")


    def process_descriptors(self):
        
        # This file is in a .txt format, read as data frame   
        physchemDf = pd.DataFrame(pd.read_csv(self.descriptor_path ,delimiter="\t"))

        # Necessary to rename columns as it'll be used in merge later on
        physchemDf.rename(columns = {'Identifier':'Name'}, inplace = True)

        # Renaming chemicals which have a descrepancy, removing ones with no predictor data
        physchemDf = (
        physchemDf.replace([' Fluorouracil, 5-'], 'Fluorouracil, 5-')
        .replace([' Hydroxyimipramine, 2-'], 'Hydroxyimipramine, 2-')
        .replace([' Mercaptopurine, 6-'], 'Mercaptopurine, 6-')
        .replace(['Clavulanic Acid'], ['Clavulanic acid'])
        .replace(['Valproic Acid'], ['Valproic acid'])
        .drop([309])
        )

        # Reading the transporter, as a data frame 
        transporterDf = pd.DataFrame(pd.read_excel(self.transporter_path))

        # SQL left join, on the names of the chemicals
        physchemDf = pd.merge(physchemDf,transporterDf,on = "Name",how = 'left')

        # Creating the attribute
        self.processed_descriptors = physchemDf
        print("CHECK - processed descriptors")


    def generate_full_dataset(self):
        
        # Taking the separately processed data frames and joining them
        joinedDf = pd.merge(self.processed_predictors, self.processed_descriptors, on = "Name", how = 'left')

        # Again cycling out some more unnecassary columns, this time in preparation for modelling purposes 
        joinedDf = joinedDf.drop(['Name','CLtotal', 'CLr','fu*GFR','Canonical SMILES','S+Acidic_pKa','S+Basic_pKa','ECCS_Class'], axis=1)

        # These are the columns which have slightly strange formatting, need to be converted to binary 
        yesNoCols = ['Pgp_Substr','Pgp_Inh','OATP1B1_Inh','S+MDCK-LE','S+CL_Renal','S+CL_Uptake','S+CL_Mech','S+CL_Metab','OATP1B1', 'OATP1B3', 'Oct-01', 'Oct-02', 'OAT1', 'OAT3', 'Pgp','BCRP']

        # Looping through those columns
        for col in yesNoCols:

            # Creating the bools for where the yes and no's are located
            if col == 'S+MDCK-LE':
                YesBool = joinedDf[col].str.contains('High')
                NoBool = joinedDf[col].str.contains('Low')
            elif col == 'S+CL_Mech':
                YesBool = joinedDf[col].str.contains('Renal')
                NoBool = joinedDf[col].str.contains('Metabolism') | joinedDf[col].str.contains('HepUptake')
            else:
                YesBool = joinedDf[col].str.contains('Yes')
                NoBool = joinedDf[col].str.contains('No')

            #replacing them with 1's and 0's
            joinedDf[col] = joinedDf[col].where(YesBool, other = False)
            joinedDf[col] = joinedDf[col].where(NoBool, other = True)
            
            joinedDf = joinedDf.astype({col: bool})
        # Working data means it can be used in modelling
        self.working_data = joinedDf
        print("CHECK - joined data")


    def x_y_notation(self):
        
        # Making a copy of the current working data
        data = self.working_data

        # Setting the classes to be numerical 
        data['Class'] = data['Class'].replace({'S': 1,'G':0,'R':0})

        # These are the descriptors 
        X = data.iloc[:,1:36]  

        # The predictor 
        y = data.iloc[:,0]

        # Set as attributes 
        self.X = X
        self.y = y 


    def train_test_split(self):

        # Sklearn library handles this very nicely
        from sklearn.model_selection import train_test_split

        # Setting the attributes for easy calling, chosen size is 80/20 split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,random_state = 2)
        

    def create_normalised_data(self):
        from sklearn.preprocessing import MinMaxScaler

        norm = MinMaxScaler().fit(self.X_train)

        # transform training data
        self.X_train_norm = norm.transform(self.X_train)

        # transform testing dataabs
        self.X_test_norm = norm.transform(self.X_test)


    def create_standardised_data(self):
        from sklearn.preprocessing import StandardScaler
        # numerical features
        num_cols = ['DiffCoef','MlogP','S+logP','S+logD','logHLC','S+Peff','S+MDCK','S+Sw','S+pH_Satd','S+S_Intrins','SolFactor','S+S_pH','hum_fup%','Vd','RBP','S+fumic','MWt','MolVol','VMcGowan']

        self.X_train_stand = self.X_train.copy()
        self.X_test_stand = self.X_test.copy()
        # apply standardization on numerical features
        for i in num_cols:
            
            # fit on training data column
            scale = StandardScaler().fit(self.X_train[[i]])
            
            # transform the training data column
            self.X_train_stand[i] = scale.transform(self.X_train[[i]])
            
            # transform the testing data column
            self.X_test_stand[i] = scale.transform(self.X_test[[i]])



    def summarise_predictors(self, binary_class_bool):
        """
        To choose between the breakdown of all three classes or the binary classes 
        use the binary_class_bool. 
        True - to show just the binary classes 
        False - to show all three classes
        """
        import matplotlib.pyplot as plt

        # Count each occurace in the class column and plot it as a bar chart
        if binary_class_bool:
            self.working_data["Class"].value_counts().plot(kind='bar')
        else:
            self.processed_predictors["Class"].value_counts().plot(kind='bar')
        
        # Show the plot 
        plt.xlabel("Count")
        plt.ylabel("Class")
        plt.show()


    def find_correlation(self):
        """
        To analyse the descriptors and identify which might be highly correlated. 
        Simply run this function and it'll generate a heatmap of all the descriptors 
        and their corresponding correlations with one another. 
        
        """
        import seaborn as sns
        import matplotlib.pyplot as plt

        # Built in pandas function for finding the correlation matrix of a data frame
        corrmat = self.X.corr()

        # This is used to get the axes/what to show in the heatmap
        top_corr_features = corrmat.index

        sns.set(rc = {'figure.figsize':(14,8)})
        # Generating the plot 
        g=sns.heatmap(self.X[top_corr_features].corr(),annot=True,cmap="RdYlGn")

        plt.show()






class Model():

    """
    Outlining the methods needed for training and optimising a 
    machine learning models using sklearn. 
    """
    def __init__(self, X, y, X_train, y_train, X_test, y_test):
        self.X = X
        self.y = y
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.params = None 
        self.features = None

    def train_predict_rfc(self) :
        from sklearn.ensemble import RandomForestClassifier
        from sklearn import metrics
        from sklearn import model_selection
        import numpy as np
        if self.params != None:
            self.clf=RandomForestClassifier(random_state = 2, **self.params)    
        else:
            self.clf=RandomForestClassifier(random_state = 2)
        self.clf.fit(self.X_train,self.y_train)
        self.y_pred = self.clf.predict(self.X_test)

        print(metrics.confusion_matrix(self.y_test, self.y_pred))
        print("Accuracy:",metrics.accuracy_score(self.y_test, self.y_pred))
        print(metrics.classification_report(self.y_test, self.y_pred))
        print(np.mean(model_selection.cross_val_score(self.clf, self.X_train, self.y_train, cv=5)))

    def leave_one_out(self):
        import numpy as np
        from sklearn import model_selection
        from sklearn.ensemble import RandomForestClassifier
        #Leave one out cross validation 
        cv = model_selection.LeaveOneOut()

        if self.params != None:
            new_clf=RandomForestClassifier(random_state = 2, **self.params)    
        else:
            new_clf=RandomForestClassifier(random_state = 2)
        #essentially getting the scores for each datapoint and comparing it against the rest 
        scores = model_selection.cross_val_score(new_clf, self.X, self.y, scoring='accuracy', cv=cv)
        
        #print the performance 
        print("Leave one out")
        print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
        
        
    def plot_feature_importance(self):
        import seaborn as sns
        import pandas as pd
        import matplotlib.pyplot as plt
        feature_imp = pd.Series(self.clf.feature_importances_, index=(list(self.X))).sort_values(ascending=False)
        feature_imp
        sns.barplot(x=feature_imp, y=feature_imp.index)
        plt.xlabel('Feature Importance Score')
        plt.ylabel('Features')
        plt.title("Visualizing Important Features")
        #plt.legend()
        plt.show()
        
    def rfecv(self):
        from sklearn.feature_selection import RFECV
        from sklearn.model_selection import StratifiedKFold
        from sklearn.ensemble import RandomForestClassifier
        import matplotlib.pyplot as plt
        import numpy as np
        cv_estimator = RandomForestClassifier(random_state = 2)

        #fitting it to the data 
        cv_estimator.fit(self.X_train, self.y_train)
        
        #creating another classifiier which will select the features using cross validation 
        cv_selector = RFECV(cv_estimator,cv= StratifiedKFold(3), step=1,scoring='accuracy')
        
        #fitting that to our data as well 
        cv_selector = cv_selector.fit(self.X_train, self.y_train)
        
        #get booleans for which features as helpful 
        rfecv_mask = cv_selector.get_support() #list of booleans
        
        #create a list for those features
        rfecv_features = [] 
        
        #add all of them to the list
        for bool, feature in zip(rfecv_mask, self.X_train.columns):
            if bool:
                rfecv_features.append(feature)
        
        #displaying the results
        print('Optimal number of features :', cv_selector.n_features_)
        print('Best features :', rfecv_features)
        
        self.features = rfecv_features
        
        #showing the results as a graph
        n_features = self.X_train.shape[1]
        plt.figure(figsize=(8,8))
        plt.barh(range(n_features), cv_estimator.feature_importances_, align='center') 
        plt.yticks(np.arange(n_features), self.X_train.columns.values) 
        plt.xlabel('Feature importance')
        plt.ylabel('Feature')
        plt.show()

    def remove_features(self):
        self.X.columns
        set_difference = set(self.X.columns).symmetric_difference(set(self.features))
        list_difference = list(set_difference)
        self.X.drop(list_difference, axis=1)
        self.X_train(list_difference, axis=1)
        self.X_test(list_difference, axis=1)
        

    def train_predict_svm(self):
        from sklearn.svm import SVC

        self.clf = SVC()
        self.clf.fit(self.X_train,self.y_train)
        self.y_pred = self.clf.predict(self.X_test)
        

    def compare_datasets_svm(self, X_train_norm, X_test_norm, X_train_stand, X_test_stand):
        from sklearn.svm import SVC
        import numpy as np
        from sklearn import metrics
        # Creating the support vector machine classifier 
        svc = SVC(kernel='poly', degree =8)
        
        rmse = []
        
        acc = []
        
        # raw, normalized and standardized training and testing data
        trainX = [self.X_train, X_train_norm, X_train_stand]
        testX = [self.X_test, X_test_norm, X_test_stand]
        
        # model fitting and measuring RMSE
        for i in range(len(trainX)):
            
            # fit
            svc.fit(trainX[i],self.y_train)
            # predict
            pred = svc.predict(testX[i])
            # RMSE
            rmse.append(np.sqrt(metrics.mean_squared_error(self.y_test,pred)))
            
            acc.append(metrics.accuracy_score(self.y_test, pred))
        
        # visualizing the result    
        df_svc = pd.DataFrame({'RMSE':rmse,"Accuracy":acc},index=['Original','Normalized','Standardized'])
        print(df_svc)

    def svm_kernel_select(self, X_train_norm, X_test_norm):
        import matplotlib.pyplot as plt
        from sklearn import metrics
        from sklearn.svm import SVC
        #testing the different kernels for SVM 
        kernel_list = ('linear', 'rbf', 'sigmoid', 'poly')
        
        #since polynomial kernels can be of different degrees testing those as well 
        degree_list = range(1,15)
        scores = {}
        scores_list = []
        poly_scores_list = []
        
        #looping through the kernel list and generating an accuracy score for all 
        for k in kernel_list:
            classifier = SVC(kernel = k)
            classifier.fit(X_train_norm, self.y_train)
            y_pred = classifier.predict(X_test_norm)
            scores[k] = metrics.accuracy_score(self.y_test,y_pred)
            scores_list.append(metrics.accuracy_score(self.y_test,y_pred))
            if k == 'poly':
                for i in degree_list:
                    classifier = SVC(kernel = 'poly', degree = i)
                    classifier.fit(X_train_norm, self.y_train)
                    y_pred = classifier.predict(X_test_norm)
                    poly_scores_list.append(metrics.accuracy_score(self.y_test,y_pred))


        df_svm = pd.DataFrame({'Accuracy':scores},index=kernel_list)
        print(df_svm)
        
        plt.plot(degree_list,poly_scores_list)
        plt.xlabel("Degree of polynomial")
        plt.ylabel("Accuracy")
        
        
    def svm_optimise(self, X_train_norm, X_test_norm):
        from sklearn.model_selection import RepeatedStratifiedKFold
        from sklearn.model_selection import GridSearchCV
        from sklearn.svm import SVC
        
        # define model and parameters
        model = SVC()
        kernel = ['poly', 'rbf', 'sigmoid']
        C = [50, 10, 1.0, 0.1, 0.01]
        gamma = ['scale']
        # define grid search
        grid = dict(kernel=kernel,C=C,gamma=gamma)
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
        grid_result = grid_search.fit(X_train_norm, self.y_train)
        
        # summarize results
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))
            
        svmc = SVC(**grid_result.best_params_)
        svmc.fit(X_train_norm, self.y_train)
        self.y_pred = svmc.predict(X_test_norm)
        from sklearn import metrics
        from sklearn import model_selection
        import numpy as np
        print(metrics.confusion_matrix(self.y_test, self.y_pred))
        print("Accuracy:",metrics.accuracy_score(self.y_test, self.y_pred))
        print(metrics.classification_report(self.y_test, self.y_pred))
        print(np.mean(model_selection.cross_val_score(svmc, X_train_norm, self.y_train, cv=3)))
        
    def ridge_classifier(self):
        from sklearn.linear_model import RidgeClassifier
        from sklearn import metrics
        from sklearn import model_selection
        #from numpy import mean
        rc = RidgeClassifier()
        print(rc)
        
        RidgeClassifier(alpha=1.0, class_weight=None, copy_X=True, fit_intercept=True,
                        max_iter=None, normalize=True, random_state=None, solver='auto',
                        tol=0.001)
        rc.fit(self.X_train, self.y_train)
        score = rc.score(self.X_train, self.y_train)
        print("Score: ", score)
        
        cv_scores = model_selection.cross_val_score(rc, self.X_train, self.y_train, cv=5)
        print("CV average score: %.2f" % cv_scores.mean())
        ypred = rc.predict(self.X_test)
        
        cm = metrics.confusion_matrix(self.y_test, ypred)
        print(cm)
        
        cr = metrics.classification_report(self.y_test, ypred)
        print(cr)
                
        
    def train_predict_knn(self):
        from sklearn.neighbors import KNeighborsClassifier
        self.knn = KNeighborsClassifier(n_neighbors=15)
        self.knn.fit(self.trainX,self.y_train)
        self.y_pred = self.knn.predict(self.X_test)

    def evaluate(self):
        from sklearn import metrics
        from sklearn import model_selection
        import numpy as np
        print(metrics.confusion_matrix(self.y_test, self.y_pred))
        print("Accuracy:",metrics.accuracy_score(self.y_test, self.y_pred))
        print(metrics.classification_report(self.y_test, self.y_pred))
        print(np.mean(model_selection.cross_val_score(self.clf, self.X_train, self.y_train, cv=3)))

    def set_params(self, parameters):
        self.params = parameters

    def random_cv_optimise(self):
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import RandomizedSearchCV

        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.append(None)

        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]

        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]

        # Method of selecting samples for training each tree
        bootstrap = [True, False]

        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'bootstrap': bootstrap}
        print(random_grid)


        # Use the random grid to search for best hyperparameters
        # First create the base model to tune
        rf = RandomForestClassifier()
        # Random search of parameters, using 3 fold cross validation, 
        # search across 100 different combinations, and use all available cores
        rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=1, random_state=2, n_jobs = -1)
        # Fit the random search model
        rf_random.fit(self.X_train, self.y_train)

        self.y_pred = rf_random.predict(self.X_test)

        print("The best parameters found are:")
        print(rf_random.best_params_)
        self.params = rf_random.best_params_

    def bayesian_optimisation(self):
        from skopt import BayesSearchCV
        from sklearn.ensemble import RandomForestClassifier
        from skopt.space import Categorical, Integer


        search_space = {"bootstrap": Categorical([True, False]), # values for boostrap can be either True or False
                "max_depth": Integer(6, 20), # values of max_depth are integers from 6 to 20
                "min_samples_leaf": Integer(2, 10),
                "min_samples_split": Integer(2, 10),
                "n_estimators": Integer(100, 1500)
            }


        search = BayesSearchCV(estimator=RandomForestClassifier(), search_spaces=search_space, cv=3)
        
        # perform the search
        search.fit(self.X_train, self.y_train)
        # report the best result
        print(search.best_score_)
        print(search.best_params_)
        self.params = search.best_params_
        
        
    def xgboost_train(self):
        
        # First XGBoost model for Pima Indians dataset

        from xgboost import XGBClassifier

        from sklearn.metrics import accuracy_score

        # split data into train and test sets

        # fit model no training data
        model = XGBClassifier()
        model.fit(self.X_train, self.y_train)
        # make predictions for test data
        self.y_pred = model.predict(self.X_test)
        predictions = [round(value) for value in self.y_pred]
        # evaluate predictions
        accuracy = accuracy_score(self.y_test, predictions)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))

