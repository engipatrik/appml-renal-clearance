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
            joinedDf[col] = joinedDf[col].where(YesBool, other = 0)
            joinedDf[col] = joinedDf[col].where(NoBool, other = 1)

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

        # Generating the plot 
        g=sns.heatmap(self.X[top_corr_features].corr(),annot=True,cmap="RdYlGn")
        plt.show()






class Model(Data):

    """
    Outlining the methods needed for training and optimising a 
    random forest classifier. 
    """
    def __init__(self, raw_descriptor_xlsx_path, raw_predictor_xlsx_path, transporter_path):
        Data.__init__(self, raw_descriptor_xlsx_path, raw_predictor_xlsx_path, transporter_path)

        self.params = None 

    def train_predict_rfc(self):
        from sklearn.ensemble import RandomForestClassifier

        if self.params != None:
            self.rfc=RandomForestClassifier(random_state = 2, **self.params)    
        else:
            self.rfc=RandomForestClassifier(random_state = 2)
        self.rfc.fit(self.X_train,self.y_train)
        self.y_pred = self.rfc.predict(self.X_test)

    def train_predict_svm(self):
        from sklearn.svm import SVC
        self.svc = SVC()
        self.svc.fit(self.X_train,self.y_train)
        self.y_pred = self.svc.predict(self.X_test)


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
        print(np.mean(model_selection.cross_val_score(self.rfc, self.X_train, self.y_train, cv=5)))

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





class Model_Metric():

    """
    Outlining the methods needed for training and optimising a 
    random forest classifier. 
    """
    def __init__(self, y_test, pred):
        self.y_test = y_test
        self.pred = pred

    def confusion_matrix(self):
        from sklearn.metrics import confusion_matrix
        confusion_matrix(self.y_test, self.y_pred)

    def predict():
        pass

    def optimise():
        pass


class KNN():

    """
    Outlining the methods needed for training and optimising a 
    random forest classifier. 
    """
    def __init__(self):
        pass

    def train_model():
        pass

    def predict():
        pass

    def optimise():
        pass


class Evaluation():
    """
    A set of methods comparing the predicted outputs from each model 
    with the chosen test data (likely to include cross validation methods).
    """
    def __init__(self):
        pass

    def confusion_matrix():
        pass
    
    def accuracy_breakdown():
        pass

    def auc_roc():
        pass

    def cross_validation():
        pass


class Visualisation():
    """
    There are various visualisations which can be generated, and therefore 
    this object can handle the creation and manipulation of these. 
    """
    def __init__(self):
        pass

    def auc_roc_plot():
        pass

    def confusion_matrix_heatmap():
        pass

    def class_balance_barplot():
        pass

    def correlation_heatmap():
        pass


class Optimisation():
    """
    Capturing the several different methods for optimising each of the models.
    Might need to create slightly different methods for each model, or 
    the inputs for each algorithm could be arguments. 
    """
    def __init__(self):
        pass

    def random_search():
        pass

    def grid_search():
        pass

    def bayesian_optimise():
        pass


class Pipeline():
    """
    This class shuld collate most of the methods from above and join into a 
    logical sequence to somewhat automatically build a model from scratch. 
    """
    def __init__(self):
        pass

    def load_data():
        pass

    def train_base_model():
        pass

    def select_features():
        pass

    def optimise_parameters():
        pass

