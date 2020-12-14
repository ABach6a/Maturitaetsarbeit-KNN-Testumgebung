import pandas
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

class DataManager:

    scaler = None

    # original data in panda DataFrame 
    odf = None
    # cleaned and scaled panda DataFrame
    odf_cleansed = None

    X_train = None
    X_validate = None
    X_test = None
    y_train = None
    y_validate = None
    y_test = None

    def __init__(self): 
        print("DataManager initializing")
        print("** loading Pima Indians data from file")
        self.odf = pandas.read_csv('data/diabetes.csv')
        self.preprocessData(self.odf)
        #self._splitDataIntoTrainingValidationAndTestingSets(self.odf_cleansed)

    def preprocessData(self, odf):
        self._handleMissingValues(odf)
        self._standardizeScales(odf)

    def _handleMissingValues(self, odf):
        print("*** handling missing values")
        # replace 0 values with NaN
        odf["Glucose"] = odf["Glucose"].replace(0, np.nan)
        odf['BloodPressure'] = odf['BloodPressure'].replace(0, np.nan)
        odf['SkinThickness'] = odf['SkinThickness'].replace(0, np.nan)
        odf['Insulin'] = odf['Insulin'].replace(0, np.nan)
        odf['BMI'] = odf['BMI'].replace(0, np.nan)

        # replace NaN according to https://www.kaggle.com/vincentlugat/pima-indians-diabetes-eda-prediction-0-906/notebook

        medianTargetTable = self._median_target('Insulin')
        odf.loc[(odf['Outcome'] == 0 ) & (odf['Insulin'].isnull()), 'Insulin'] = medianTargetTable['Insulin'][0]
        odf.loc[(odf['Outcome'] == 1 ) & (odf['Insulin'].isnull()), 'Insulin'] = medianTargetTable['Insulin'][1]

        medianTargetTable = self._median_target('Glucose')
        odf.loc[(odf['Outcome'] == 0 ) & (odf['Glucose'].isnull()), 'Glucose'] = medianTargetTable['Glucose'][0]
        odf.loc[(odf['Outcome'] == 1 ) & (odf['Glucose'].isnull()), 'Glucose'] = medianTargetTable['Glucose'][1]

        medianTargetTable = self._median_target('SkinThickness')
        odf.loc[(odf['Outcome'] == 0 ) & (odf['SkinThickness'].isnull()), 'SkinThickness'] = medianTargetTable['SkinThickness'][0]
        odf.loc[(odf['Outcome'] == 1 ) & (odf['SkinThickness'].isnull()), 'SkinThickness'] = medianTargetTable['SkinThickness'][1]

        medianTargetTable = self._median_target('BloodPressure')
        odf.loc[(odf['Outcome'] == 0 ) & (odf['BloodPressure'].isnull()), 'BloodPressure'] = medianTargetTable['BloodPressure'][0]
        odf.loc[(odf['Outcome'] == 1 ) & (odf['BloodPressure'].isnull()), 'BloodPressure'] = medianTargetTable['BloodPressure'][1]

        medianTargetTable = self._median_target('BMI')
        odf.loc[(odf['Outcome'] == 0 ) & (odf['BMI'].isnull()), 'BMI'] = medianTargetTable['BMI'][0]
        odf.loc[(odf['Outcome'] == 1 ) & (odf['BMI'].isnull()), 'BMI'] = medianTargetTable['BMI'][1]

        print(odf.describe())

    def _standardizeScales(self, odf):
        print("*** standardize scales")
        self.scaler = preprocessing.StandardScaler()
        odf_cleansed = self.scaler.fit_transform(odf)
        self.odf_cleansed = pandas.DataFrame(odf_cleansed, columns=odf.columns)
        self.odf_cleansed['Outcome'] = odf['Outcome']
        #print(self.odf_cleansed.describe())

    def _median_target(self, var):   
        temp = self.odf[self.odf[var].notnull()]
        temp = temp[[var, 'Outcome']].groupby(['Outcome'])[[var]].median().reset_index()
        #print median target table
        #print(temp)
        return temp

    def splitDataIntoTrainingValidationAndTestingSets(self):
        self._splitDataIntoTrainingValidationAndTestingSets(self.odf_cleansed)

    def _splitDataIntoTrainingValidationAndTestingSets(self, dataset):
        print("** splitting data into training, validation and testing sets")
        # seperate into input (x) and outcome (y)
        X = dataset.loc[:, dataset.columns != "Outcome"]
        y = dataset.loc[:, "Outcome"]

        # first split: training(90%) and testing set(10%)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y)
        # self.X_train = X_train
        # self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        # second split: final training(80%) and validation set(20%)
        X_train_final, X_validation, y_train_final, y_validation = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train)
        self.X_train = X_train_final
        self.y_train = y_train_final
        self.X_validate = X_validation
        self.y_validate = y_validation
        
        #print(self.X_train.describe())

