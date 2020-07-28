# libraries
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
data = pd.read_csv('loan.csv')
# we do not need the loan_ID column for prediction
data.drop("Loan_ID", axis=1, inplace=True)
# in dependent part 3+ replace with 3
data['Dependents'].replace('3+', 3, inplace=True)
# replacing Y and N with 0 and 1
data['Loan_Status'].replace('N', 0, inplace=True)
data['Loan_Status'].replace('Y', 1, inplace=True)
# replacing female and male with 0 and 1
data['Gender'].replace('Female', 0, inplace=True)
data['Gender'].replace('Male', 1, inplace=True)
# replacing Yes and No with 1 and o
data['Married'].replace('No', 0, inplace=True)
data['Married'].replace('Yes', 1, inplace=True)
# data.Education.unique()
# replacing Graduate and Not Graduate with 1 and o
data['Education'].replace('Graduate', 1, inplace=True)
data['Education'].replace('Not Graduate', 0, inplace=True)
# replacing Yes and No with 1 and o for Self_Employed
data['Self_Employed'].replace('No', 0, inplace=True)
data['Self_Employed'].replace('Yes', 1, inplace=True)
# data.Property_Area.unique()
Property_Area = pd.get_dummies(data.Property_Area)
data[['Urban', 'Rural', 'Semiurban']] = Property_Area
data.drop('Property_Area', axis=1, inplace=True)
# missing value handelling

# replace missing values with the mode
data['Gender'].fillna(data['Gender'].mode()[0], inplace=True)
data['Married'].fillna(data['Married'].mode()[0], inplace=True)
data['Dependents'].fillna(data['Dependents'].mode()[0], inplace=True)
data['Self_Employed'].fillna(data['Self_Employed'].mode()[0], inplace=True)
data['Credit_History'].fillna(data['Credit_History'].mode()[0], inplace=True)
data['LoanAmount'].fillna(data['LoanAmount'].mode()[0], inplace=True)
data['Loan_Amount_Term'].fillna(
    data['Loan_Amount_Term'].mode()[0], inplace=True)
Loan_Status1 = data['Loan_Status']
data.drop("Loan_Status", axis=1, inplace=True)
data['Loan_Status'] = Loan_Status1
# divide X and y
X = data.iloc[:, 0:13].values
y = data.iloc[:, 13].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)
# Scaling data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# builduing ann

bclfr = Sequential()
bclfr.add(Dense(units=10, kernel_initializer='uniform',
                activation='relu', input_dim=13))
bclfr.add(Dense(units=10, kernel_initializer='uniform', activation='relu'))
bclfr.add(Dense(units=10, kernel_initializer='uniform', activation='sigmoid'))
bclfr.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

print(X_train.shape)
print(y_train.shape)

bclfr.fit(X_train, y_train, batch_size=10, epochs=100)
