import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import joblib
df = pd.read_csv("weatherAUS.csv")
pd.set_option("display.max_columns", None)#pd.set_option('max_colwidth', 800) sets max.width to 800 pixels per column. So we use 'None'
numerical_feature = [feature for feature in df.columns if df[feature].dtypes != 'O']#Datatype='O' (if 'Object'), 'f' (if 'Float')
#print("Numerical Features Count {}".format(len(numerical_feature)))
discrete_feature=[feature for feature in numerical_feature if len(df[feature].unique())<25]#Here we find the Columns which contain CLASSES not REAL NUMBERs out of ONLY Numerical Feature columns but this is WRONG
#print("Discrete feature Count {}".format(len(discrete_feature)))
continuous_feature = [feature for feature in numerical_feature if feature not in discrete_feature]
categorical_feature = [feature for feature in df.columns if feature not in numerical_feature]
#print("Categorical feature Count {}".format(len(categorical_feature)))
#print(df.dtypes)#The columns Cloud9am/3pm only contain classes 0-9/NA, still its dtype is FLOAT
#print(df.isnull().sum())
def randomsampleimputation(df, variable):
    random_sample=df[variable].dropna().sample(df[variable].isnull().sum(),random_state=0)#'dropna' REMOVES the cells having Null values. And '.samples'returns 'n' samples out of these NON-NULL cells, where n=SUM OF NULL CELLS. Random Sampling used to maintain the variance
    random_sample.index=df[df[variable].isnull()].index#It returns the INDEX values where Cell Value==Null is True
    df.loc[df[variable].isnull(),variable]=random_sample #'.loc' filters the ROWS (where null values are present) and Columns(here, variable) and replaces them with random_sample
randomsampleimputation(df, "Cloud9am")
randomsampleimputation(df, "Cloud3pm")
randomsampleimputation(df, "Evaporation")
randomsampleimputation(df, "Sunshine")
# plt.figure(figsize = (18,18))
# sns.heatmap(df.corr(), annot = True, cmap = "RdYlGn")#'Rain Today' is Yes ONLY IF rain>0.6 mm
# plt.show()
# for feature in continuous_feature:
#     sns.distplot(df[feature])
#     plt.xlabel(feature)
#     plt.ylabel("Count")
#     plt.title(feature)
#     plt.figure(figsize=(15,15))
#     plt.show()
# for feature in continuous_feature:
#     sns.boxplot(df[feature])
#     plt.title(feature)
#     plt.figure(figsize=(15,15))
#     plt.show()
for feature in continuous_feature:
    if(df[feature].isnull().sum()*100/len(df))>0:
        df[feature].fillna(df[feature].median(), inplace=True)#MinTemp/MaxTemp/Rainfall/Windgustspeed9am/WindSpeed3pm/Humidity9am/Humidity3pm/Pressure9am/Pressure3pm/Temp9am/Temp3pm
#While, Evaporation/Sunshine/Cloud9am/Cloud3pm, Nan filled with RANDOM values
# def mode_nan(df,variable):
#     mode=df[variable].value_counts().index[0]
#     df[variable].fillna(mode,inplace=True)
# mode_nan(df,"Cloud9am")
# mode_nan(df,"Cloud3pm")
df["RainToday"] = pd.get_dummies(df["RainToday"], drop_first = True)#When 'drop_first'=False, NO is taken as 1. When 'drop_first'=True, NO is taken as 0
df["RainTomorrow"] = pd.get_dummies(df["RainTomorrow"], drop_first = True)
for feature in categorical_feature:
    print(feature, (df.groupby([feature])["RainTomorrow"].mean().sort_values(ascending = False)).index)#'groupby' puts the first column(here feature) as the INDEX and aggregates 'Rain Tomorrow' values under each Index by way of MEAN
#Subsequently, 'sort_values' SORTs the means of raintomorrow in DEscending order. Then, 'index' prints INDICES(feature)
#Now the groupby arranged the data in decreasing order of raintomorrow. We have basically RANKED the Categorical Factors which lead to raintomorrow.
#This way we have converted into Ordinal Data which will be encoded using Lable Encoder rather than One-hot encoding
windgustdir= {'NNW':0, 'NW':1, 'WNW':2, 'N':3, 'W':4, 'WSW':5, 'NNE':6, 'S':7, 'SSW':8, 'SW':9, 'SSE':10,'NE':11, 'SE':12, 'ESE':13, 'ENE':14, 'E':15}
winddir9am= {'NNW':0, 'N':1, 'NW':2, 'NNE':3, 'WNW':4, 'W':5, 'WSW':6, 'SW':7, 'SSW':8, 'NE':9, 'S':10,'SSE':11, 'ENE':12, 'SE':13, 'ESE':14, 'E':15}
winddir3pm= {'NW':0, 'NNW':1, 'N':2, 'WNW':3, 'W':4, 'NNE':5, 'WSW':6, 'SSW':7, 'S':8, 'SW':9, 'SE':10, 'NE':11, 'SSE':12, 'ENE':13, 'E':14, 'ESE':15}
df["WindGustDir"] = df["WindGustDir"].map(windgustdir)#'Map' is used for Lable encoding
df["WindDir9am"] = df["WindDir9am"].map(winddir9am)
df["WindDir3pm"] = df["WindDir3pm"].map(winddir3pm)
df["WindGustDir"] = df["WindGustDir"].fillna(df["WindGustDir"].value_counts().index[0])#'value_counts().index[0] ==Mode
df["WindDir9am"] = df["WindDir9am"].fillna(df["WindDir9am"].value_counts().index[0])#Mode can also be found using MODE fn.It, however, gives output prefixed by '0' So use Mode()[0]
df["WindDir3pm"] = df["WindDir3pm"].fillna(df["WindDir3pm"].value_counts().index[0])
#print((df.isnull().sum())*100/len(df))#Gives of Null values in Percentage
df1 = df.groupby(["Location"])["RainTomorrow"].value_counts().sort_values().unstack()#groupby segregates raintomorrow total counts 0 &1 . Unstack creates separate columns for 0 and one
df1[1].sort_values(ascending = False)#Note that [1] represents the TRUE value column, it is NOT ininverted commos since it is not Key but value
df1[1].sort_values(ascending = False).index#.index returns a NUMPY ARRAY of Locaions
#print(len(df1[1].sort_values(ascending = False).index))
location = {'Portland':1, 'Cairns':2, 'Walpole':3, 'Dartmoor':4, 'MountGambier':5,
       'NorfolkIsland':6, 'Albany':7, 'Witchcliffe':8, 'CoffsHarbour':9, 'Sydney':10,
       'Darwin':11, 'MountGinini':12, 'NorahHead':13, 'Ballarat':14, 'GoldCoast':15,
       'SydneyAirport':16, 'Hobart':17, 'Watsonia':18, 'Newcastle':19, 'Wollongong':20,
       'Brisbane':21, 'Williamtown':22, 'Launceston':23, 'Adelaide':24, 'MelbourneAirport':25,
       'Perth':26, 'Sale':27, 'Melbourne':28, 'Canberra':29, 'Albury':30, 'Penrith':31,
       'Nuriootpa':32, 'BadgerysCreek':33, 'Tuggeranong':34, 'PerthAirport':35, 'Bendigo':36,
       'Richmond':37, 'WaggaWagga':38, 'Townsville':39, 'PearceRAAF':40, 'SalmonGums':41,
       'Moree':42, 'Cobar':43, 'Mildura':44, 'Katherine':45, 'AliceSprings':46, 'Nhil':47,
       'Woomera':48, 'Uluru':49}#Note that 0 not used in Lable encoding
df["Location"] = df["Location"].map(location)
df["Date_month"]=pd.to_datetime(df["Date"], format = "%d-%m-%Y").dt.month
df["Date_day"]=pd.to_datetime(df["Date"], format = "%d-%m-%Y").dt.day
# plt.figure(figsize = (20,20))
# sns.heatmap(df.corr(), annot = True, cmap = "RdYlGn")#'Rain Today' is Yes ONLY IF rain>0.6 mm
# plt.show()
IQR=df.MinTemp.quantile(0.75)-df.MinTemp.quantile(0.25)#This gives upper/lower bounds of BoxPlot
lower_bridge=df.MinTemp.quantile(0.25)-(IQR*1.5)
upper_bridge=df.MinTemp.quantile(0.75)+(IQR*1.5)
#print(lower_bridge, upper_bridge)
df.loc[df['MinTemp']>=30.45,'MinTemp']=30.45#replaces outliers(>30.45) with 30.45
df.loc[df['MinTemp']<=-5.95,'MinTemp']=-5.95#replaces outliers(<-5.95) with -5.95
IQR=df.MaxTemp.quantile(0.75)-df.MaxTemp.quantile(0.25)
lower_bridge=df.MaxTemp.quantile(0.25)-(IQR*1.5)
upper_bridge=df.MaxTemp.quantile(0.75)+(IQR*1.5)
#print(lower_bridge, upper_bridge)
df.loc[df['MaxTemp']>=43.5,'MaxTemp']=43.5
df.loc[df['MaxTemp']<=2.7,'MaxTemp']=2.7
IQR=df.Rainfall.quantile(0.75)-df.Rainfall.quantile(0.25)
lower_bridge=df.Rainfall.quantile(0.25)-(IQR*1.5)
upper_bridge=df.Rainfall.quantile(0.75)+(IQR*1.5)
#print(lower_bridge, upper_bridge)
df.loc[df['Rainfall']>=1.5,'Rainfall']=1.5
df.loc[df['Rainfall']<=-0.89,'Rainfall']=-0.89
IQR=df.Evaporation.quantile(0.75)-df.Evaporation.quantile(0.25)
lower_bridge=df.Evaporation.quantile(0.25)-(IQR*1.5)
upper_bridge=df.Evaporation.quantile(0.75)+(IQR*1.5)
#print(lower_bridge, upper_bridge)
df.loc[df['Evaporation']>=14.6,'Evaporation']=14.6
df.loc[df['Evaporation']<=-4.6,'Evaporation']=-4.6
IQR=df.WindGustSpeed.quantile(0.75)-df.WindGustSpeed.quantile(0.25)
lower_bridge=df.WindGustSpeed.quantile(0.25)-(IQR*1.5)
upper_bridge=df.WindGustSpeed.quantile(0.75)+(IQR*1.5)
#print(lower_bridge, upper_bridge)
df.loc[df['WindGustSpeed']>=68.5,'WindGustSpeed']=68.5
df.loc[df['WindGustSpeed']<=8.5,'WindGustSpeed']=8.5
IQR=df.WindSpeed9am.quantile(0.75)-df.WindSpeed9am.quantile(0.25)
lower_bridge=df.WindSpeed9am.quantile(0.25)-(IQR*1.5)
upper_bridge=df.WindSpeed9am.quantile(0.75)+(IQR*1.5)
#print(lower_bridge, upper_bridge)
df.loc[df['WindSpeed9am']>=37,'WindSpeed9am']=37
df.loc[df['WindSpeed9am']<=-11,'WindSpeed9am']=-11
IQR=df.WindSpeed3pm.quantile(0.75)-df.WindSpeed3pm.quantile(0.25)
lower_bridge=df.WindSpeed3pm.quantile(0.25)-(IQR*1.5)
upper_bridge=df.WindSpeed3pm.quantile(0.75)+(IQR*1.5)
#print(lower_bridge, upper_bridge)
df.loc[df['WindSpeed3pm']>40.5,'WindSpeed3pm']=40.5
df.loc[df['WindSpeed3pm']<=-3.5,'WindSpeed3pm']=-3.5
IQR=df.Humidity9am.quantile(0.75)-df.Humidity9am.quantile(0.25)
lower_bridge=df.Humidity9am.quantile(0.25)-(IQR*1.5)
upper_bridge=df.Humidity9am.quantile(0.75)+(IQR*1.5)
#print(lower_bridge, upper_bridge)
df.loc[df['Humidity9am']>=122,'Humidity9am']=122
df.loc[df['Humidity9am']<=18,'Humidity9am']=18
IQR=df.Pressure9am.quantile(0.75)-df.Pressure9am.quantile(0.25)
lower_bridge=df.Pressure9am.quantile(0.25)-(IQR*1.5)
upper_bridge=df.Pressure9am.quantile(0.75)+(IQR*1.5)
#print(lower_bridge, upper_bridge)
df.loc[df['Pressure9am']>=1034.25,'Pressure9am']=1034.25
df.loc[df['Pressure9am']<=1001.05,'Pressure9am']=1001.05
IQR=df.Pressure3pm.quantile(0.75)-df.Pressure3pm.quantile(0.25)
lower_bridge=df.Pressure3pm.quantile(0.25)-(IQR*1.5)
upper_bridge=df.Pressure3pm.quantile(0.75)+(IQR*1.5)
#print(lower_bridge, upper_bridge)
df.loc[df['Pressure3pm']>=1031.85,'Pressure3pm']=1031.85
df.loc[df['Pressure3pm']<=998.65,'Pressure3pm']=998.65
IQR=df.Temp9am.quantile(0.75)-df.Temp9am.quantile(0.25)
lower_bridge=df.Temp9am.quantile(0.25)-(IQR*1.5)
upper_bridge=df.Temp9am.quantile(0.75)+(IQR*1.5)
#print(lower_bridge, upper_bridge)
df.loc[df['Temp9am']>=35.3,'Temp9am']=35.3
df.loc[df['Temp9am']<=-1.49,'Temp9am']=-1.49
IQR=df.Temp3pm.quantile(0.75)-df.Temp3pm.quantile(0.25)
lower_bridge=df.Temp3pm.quantile(0.25)-(IQR*1.5)
upper_bridge=df.Temp3pm.quantile(0.75)+(IQR*1.5)
#print(lower_bridge, upper_bridge)
df.loc[df['Temp3pm']>=40.45,'Temp3pm']=40.45
df.loc[df['Temp3pm']<=2.45,'Temp3pm']=2.45
# for feature in continuous_feature:
#     print(feature)
#     plt.figure(figsize=(15,6))
#     plt.subplot(1, 2, 1)
#     plt.gca().set_title(feature)#Notice this step
#     df[feature].hist()
#     plt.subplot(1, 2, 2)
#     stats.probplot(df[feature], dist="norm", plot=plt)#refer 'scipy' module
#     plt.show()
"""Linear models assume that the independent variables are normally distributed. 
Failure to meet this assumption may produce algorithms that perform poorly. We 
can determine whether a variable is normally distributed with histograms and probability plots. 
In a probability plot, the quantiles of the independent variable are plotted against the expected quantiles of the normal distribution. 
If the variable is normally distributed, the dots in the probability plot should fall along a 45 degree diagonal. It also catches discontinuities
in data, if any (see rainfall plot)"""
#df.to_csv("C:/Users/Ashish/Desktop/Rainfall Prediction/preprocessed_1.csv", index=True)
X = df.drop(["RainTomorrow", "Date"], axis=1)
Y = df["RainTomorrow"]
#print(Y.value_counts(normalize=True))#It gives proportion of 1s and 0s
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size =0.2, stratify = Y, random_state = 0)#'stratify' maintains same proportion of 0s/1s in y_train as well as y_test as in Y dataset
sm=SMOTE(random_state=0)
"""Synthetic Minority Oversampling TEchnique (SMOTE) is a part of 'IMBalanced classification' which involves developing predictive models on classification datasets that have a severe class imbalance (probably 0/1 ration of RainTomorrow).
Due to imbalanced datasets most machine learning techniques ignore, and in turn have poor performance on, the minority class, which sometimes is more important than majority class. 
There are too few examples of the minority class for a model to effectively learn the decision boundary. So, in SMOTE, oversample the minority class"""
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)#In y_train 0/1 ratio=2, in y_train_res 0/1 ratio=1
cat = CatBoostClassifier(iterations=2000, eval_metric = "AUC")#CatBoost is an algorithm for gradient boosting on decision trees.It is designed for categorical data(maybe strictly classification problems)
"""It is recommended to check that there is no obvious underfitting or overfitting before tuning any other parameters. This can be done by setting the number of iterations to a large value
AUC refers to Area Under the Curve(Receiver Operator Characteristic Curve). The higher the AUC, the better the performance of the model"""
cat.fit(X_train_res, y_train_res)
y_pred = cat.predict(X_test)
print(cohen_kappa_score(y_test, y_pred))
# print(confusion_matrix(y_test,y_pred))
# print(accuracy_score(y_test,y_pred))
# print(classification_report(y_test,y_pred))
metrics.plot_roc_curve(cat, X_test, y_test)#AUC (area under RoC curve)is the measure of the ability of a classifier to distinguish using different THRESHOLDs.ranges from 0 to 1. A model whose predictions are 100% wrong has an AUC of 0.0; while the one with 100% correct prediction  has an AUC of 1.0.
plt.show()#Note that hyperparameter tuning not done here
#print(metrics.roc_auc_score(y_test, y_pred, average=None))#Roc_AUC Score has Y_test and Y_pred on x and y axis, but AUC has TP rate and FP rates. Both calculate area under curve
# rf=RandomForestClassifier()#2nd Method for Classification
# rf.fit(X_train_res,y_train_res)
# y_pred1 = rf.predict(X_test)
# print(cohen_kappa_score(y_test, y_pred1))
# # print(confusion_matrix(y_test,y_pred1))
# # print(accuracy_score(y_test,y_pred1))
# # print(classification_report(y_test,y_pred1))
# metrics.plot_roc_curve(rf, X_test, y_test)
# plt.show()
#print(metrics.roc_auc_score(y_test, y_pred1, average=None))
# logreg = LogisticRegression()#3rd Method for Classification
# logreg.fit(X_train_res, y_train_res)
# y_pred2 = logreg.predict(X_test)
# print(cohen_kappa_score(y_test, y_pred2))
# # print(confusion_matrix(y_test,y_pred2))
# # print(accuracy_score(y_test,y_pred2))
# # print(classification_report(y_test,y_pred2))
# metrics.plot_roc_curve(logreg, X_test, y_test)
# plt.show()
# #print(metrics.roc_auc_score(y_test, y_pred2, average=None))
# gnb = GaussianNB()#4th Method for Classification
# gnb.fit(X_train_res, y_train_res)
# y_pred3 = gnb.predict(X_test)
# print(cohen_kappa_score(y_test, y_pred3))
# # print(confusion_matrix(y_test,y_pred3))
# # print(accuracy_score(y_test,y_pred3))
# # print(classification_report(y_test,y_pred3))
# metrics.plot_roc_curve(gnb, X_test, y_test)
# plt.show()
# #print(metrics.roc_auc_score(y_test, y_pred3, average=None))
# knn = KNeighborsClassifier(n_neighbors=3)#5th Method for Classification
# knn.fit(X_train_res, y_train_res)
# y_pred4 = knn.predict(X_test)
# print(cohen_kappa_score(y_test, y_pred4))
# # print(confusion_matrix(y_test,y_pred4))
# # print(accuracy_score(y_test,y_pred4))
# # print(classification_report(y_test,y_pred4))
# metrics.plot_roc_curve(knn, X_test, y_test)
# plt.show()
# #print(metrics.roc_auc_score(y_test, y_pred4, average=None))
# xgb = XGBClassifier()#6th Method for Classification
# xgb.fit(X_train_res, y_train_res)
# y_pred6 = xgb.predict(X_test)
# print(cohen_kappa_score(y_test, y_pred6))
# # print(confusion_matrix(y_test,y_pred6))
# # print(accuracy_score(y_test,y_pred6))
# # print(classification_report(y_test,y_pred6))
# metrics.plot_roc_curve(xgb, X_test, y_test)
# plt.show()
# #print(metrics.roc_auc_score(y_test, y_pred6, average=None))
# svc = SVC()#8th Method for Classification
# svc.fit(X_train_res, y_train_res)
# y_pred5 = svc.predict(X_test)
# print(cohen_kappa_score(y_test, y_pred5))
# # print(confusion_matrix(y_test,y_pred5))
# # print(accuracy_score(y_test,y_pred5))
# # print(classification_report(y_test,y_pred5))
# metrics.plot_roc_curve(svc, X_test, y_test)
# plt.show()
# #print(metrics.roc_auc_score(y_test, y_pred5, average=None))
# #we chose “Cohen’s Kappa” as a metric to decide on the best model, especially in case of unbalanced datasets
# #Now, we use joblib to create pickle file; in comparison to 'pickle', 'joblib' is faster(creates pickle file in ONE step)
#joblib.dump(cat, "C:/Users/Ashish/Desktop/Rainfall Prediction/cat.pkl")
