# import related libraries
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# load the entire original labelled dataset into a pandas dataframe
dataset = pd.read_csv('F:/training.csv')

# cleaning the noise from the data
df_clean = dataset.replace('#N/A', np.nan)
df_clean=df_clean.dropna()
df_clean = dataset.replace('#', np.nan)
df_clean=df_clean.dropna()
df_clean = df_clean.replace('0', np.nan)
df_clean=df_clean.drop('Instance', axis=1)
df_clean = df_clean.dropna()

# creating a dataframe consisting of independent variables (X) and dependent variables (y)
X=df_clean.drop('Income in EUR',axis=1)
y = df_clean[['Income in EUR']]

# Implementing hot enconding
X = pd.get_dummies(X, columns=["Profession", "University Degree","Country","Year of Record","Gender","Hair Color"], prefix=["prof","edu","country","year","gender","hair"],drop_first=True)

# Splitting the labelled data into training & validation data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=1)

# getting all relevant encoded columns
prof_col = [col for col in X if col.startswith('prof')]
edu_col = [col for col in X if col.startswith('edu')]
country_col = [col for col in X if col.startswith('country')]
year_col = [col for col in X if col.startswith('year')]
gender_col = [col for col in X if col.startswith('gender')]
hair_col = [col for col in X if col.startswith('hair')]
size_city_col = ['Size of City']
age_col = ['Age']
glass_col = ['Wears Glasses']
height_col = ['Body Height [cm]']
combined_col = edu_col + country_col + year_col + gender_col + size_city_col + age_col + height_col + hair_col + prof_col

# building a linear regression model and running a prediction on the validation data
reg = LinearRegression()
reg.fit(X_train[combined_col], y_train)
y_predicted = reg.predict(X_test[combined_col])
print("rms: %.2f" % sqrt(mean_squared_error(y_test, y_predicted)))

# extracting the coefficients of the linear regression model for observation & analysis
coeff_df = pd.DataFrame(np.transpose(reg.coef_), combined_col, columns=['Coefficient'])
export_csv = coeff_df.to_csv ("F:/coeff.csv", index = True, header=False)


# loading the unlabelled data into pandas dataframe
test_data = pd.read_csv('F:/test.csv')

# turning the unlabelled data into usable format with noise cleaning
test_data = test_data.drop('Income', axis=1)
test_data = test_data.replace('#N/A', np.nan)
test_data = test_data.replace('0', np.nan)
test_data.fillna(method='ffill', inplace=True)

# applying hot-encoding onto test-data & capturing relevant columns
test_data = pd.get_dummies(test_data, columns=["Profession","University Degree","Country","Year of Record","Gender", "Hair Color"], prefix=["prof","edu","country","year","gender","hair"],drop_first=True)
test_data.shape
test_country_col = [col for col in test_data if col.startswith('country')]
test_edu_col = [col for col in test_data if col.startswith('edu')]
test_year_col = [col for col in test_data if col.startswith('year')]
test_gender_col = [col for col in test_data if col.startswith('gender')]
test_hair_col = [col for col in test_data if col.startswith('hair')]
test_prof_col = [col for col in test_data if col.startswith('prof')]
test_size_city_col = ['Size of City']
test_age_col = ['Age']
test_height_col = ['Body Height [cm]']
test_combined_col = test_edu_col + test_country_col + test_year_col + test_gender_col + test_size_city_col + test_age_col + test_height_col + test_hair_col + test_prof_col

# handling the differences in the columns between labelled and unlabelled data
missing_list = np.setdiff1d(combined_col,test_combined_col)
print(len(missing_list))
for i in range(len(missing_list)):
    test_data[missing_list[i]] = np.nan
test_data = test_data.fillna(0)

# prediciting the income using the regression model
test_y_predicted = reg.predict(test_data[combined_col])

# saving the output data to csv file for reference
np.savetxt("F:/predicted.csv", test_y_predicted, delimiter=",")

# uplaod the generated csv to kaggle