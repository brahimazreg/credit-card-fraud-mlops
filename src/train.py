import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Read the file 
df=pd.read_csv(r"C:\Users\brahi\Desktop\credit-card-fraud-mlops\data\raw\creditcard.csv")

# we use stratify in order to have both x_train and x_test with the smae percentage of fraud
x=df.drop('Class',axis=1)
y=df['Class']
x_train,x_test,y_train,y_test=train_test_split(x,y,stratify=y,test_size=0.2,random_state=42)

# We need to scale Amount only , because other features are already scaled


scaler = StandardScaler()

x_train["Amount"]=scaler.fit_transform(x_train[["Amount"]])
x_test["Amount"]=scaler.transform(x_test[["Amount"]])

#    Verify data first (quick check for x_train and x_test  to confirm the scale for Amount feature)
x_train.head(3)

x_test.head(3)

# chek for Nan value in both x_train and x_test
x_train.isnull().sum()
x_test.isnull().sum()

# Handle class imbalance Because the  dataset is highly imbalanced (~0.17% fraud)

smote = SMOTE(random_state=42)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)

#    Logistic Regression

model = LogisticRegression()
model.fit(x_train_res, y_train_res)

y_pred = model.predict(x_test)


print(classification_report(y_test, y_pred))

roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])

# We Try better other models :
# - Random Forest
# - XGBoost

###   ---------        Random Forest   ------------------
from sklearn.ensemble import RandomForestClassifier

RFC =RandomForestClassifier(criterion='gini',max_depth=6,min_samples_leaf=10,random_state=42 )
RFC.fit(x_train,y_train)

y_pred_rfc=RFC.predict(x_test)
print(classification_report(y_test, y_pred_rfc))

#### --------------           XGBoost  -------------------

from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

model.fit(x_train, y_train)
y_pred_xgb = model.predict(x_test)

print(classification_report(y_test, y_pred_xgb))

# using  scale_pos_weight
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=100
)

model.fit(x_train, y_train)

# Changing the threshold  from 0.5 to 0.3
y_proba = model.predict_proba(x_test)[:, 1]
y_pred_th = (y_proba > 0.3).astype(int)

print(classification_report(y_test, y_pred_th))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))

# Save the model 
import joblib
joblib.dump(model, "xgboost_model.pkl")

# Save threshold
joblib.dump(0.3, "threshold.pkl")

# save scalar
joblib.dump(scaler, "scaler.pkl")