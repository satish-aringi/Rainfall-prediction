
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

data = pd.read_csv('Rainfall_Cleaned.csv')

data['temp_diff'] = data['maxtemp'] - data['mintemp']

for lag in range(1, 4):
    data[f'rainfall_prev{lag}'] = (data['rainfall'].shift(lag) == 'yes').fillna(0).astype(int)
    data[f'temp_prev{lag}'] = data['temparature'].shift(lag).bfill()

data['temp_3day_avg'] = data['temparature'].rolling(3).mean().bfill()

y_class = data['rainfall'].astype(int)

X = data.drop(['rainfall'], axis=1)
X = X.fillna(X.mean(numeric_only=True)) 

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_class, test_size=0.3, random_state=42, stratify=y_class
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=1/3, random_state=42, stratify=y_temp
)

print("Train class distribution:\n", y_train.value_counts())
print("Validation class distribution:\n", y_val.value_counts())
print("Test class distribution:\n", y_test.value_counts())

smote = SMOTE(random_state=42)

if y_train.nunique() > 1:
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
else:
    print("Warning: y_train contains only one class. SMOTE cannot be applied. Proceeding without oversampling.")
    X_train_bal, y_train_bal = X_train.copy(), y_train.copy() 

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_bal)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

rf = RandomForestClassifier(
    n_estimators=1200, max_depth=25, min_samples_split=4, min_samples_leaf=2,
    class_weight='balanced', random_state=42, n_jobs=-1
)

if y_train_bal.nunique() > 1:
    positive_class_count_bal = y_train_bal.value_counts().get(1, 0)
    negative_class_count_bal = y_train_bal.value_counts().get(0, 0)
    scale_pos_weight_val = negative_class_count_bal / positive_class_count_bal
    xgb = XGBClassifier(
        n_estimators=1200, max_depth=7, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight_val,
        use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1
    )
else:
   
    xgb = None 
    print("Warning: XGBoost will not be trained due to single class in y_train_bal.")

lgbm = LGBMClassifier(
    n_estimators=1000, max_depth=12, learning_rate=0.05,
    class_weight='balanced', random_state=42, n_jobs=-1
)

lr = LogisticRegression(max_iter=2000, class_weight='balanced', solver='liblinear')

trained_models = {}

if y_train_bal.nunique() > 1:
    rf.fit(X_train_scaled, y_train_bal)
    trained_models['rf'] = rf
    if xgb:
        xgb.fit(X_train_bal, y_train_bal)
        trained_models['xgb'] = xgb
    lgbm.fit(X_train_scaled, y_train_bal)
    trained_models['lgbm'] = lgbm
    lr.fit(X_train_scaled, y_train_bal)
    trained_models['lr'] = lr
else:
    print("Warning: No models trained due to single class in y_train_bal.")

y_pred_rf = np.full(len(y_test), y_train_bal.iloc[0]) if 'rf' not in trained_models else trained_models['rf'].predict(X_test_scaled)
y_pred_xgb = np.full(len(y_test), y_train_bal.iloc[0]) if 'xgb' not in trained_models else trained_models['xgb'].predict(X_test) 
y_pred_lgbm = np.full(len(y_test), y_train_bal.iloc[0]) if 'lgbm' not in trained_models else trained_models['lgbm'].predict(X_test_scaled)
y_pred_lr = np.full(len(y_test), y_train_bal.iloc[0]) if 'lr' not in trained_models else trained_models['lr'].predict(X_test_scaled)

if y_train_bal.nunique() > 1:
    print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
    print(f"XGBoost Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
    print(f"LightGBM Accuracy: {accuracy_score(y_test, y_pred_lgbm):.4f}")
    print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")

    print("\n--- Random Forest Report ---")
    print(classification_report(y_test, y_pred_rf))
    print("\n--- XGBoost Report ---")
    print(classification_report(y_test, y_pred_xgb))
    print("\n--- LightGBM Report ---")
    print(classification_report(y_test, y_pred_lgbm))
    print("\n--- Logistic Regression Report ---")
    print(classification_report(y_test, y_pred_lr))
else:
    print("No individual model accuracy or reports displayed due to single class in y_train_bal.")

ensemble_estimators = [(name, model) for name, model in trained_models.items()]

if len(ensemble_estimators) > 0:
    ensemble = VotingClassifier(
        estimators=ensemble_estimators,
        voting='soft', n_jobs=-1
    )
    ensemble.fit(X_train_scaled, y_train_bal)
    y_pred_ensemble = ensemble.predict(X_test_scaled)

    print(f"\nVoting Ensemble Accuracy: {accuracy_score(y_test, y_pred_ensemble):.4f}")
    print("\n--- Ensemble Report ---")
    print(classification_report(y_test, y_pred_ensemble))
else:
    print("No models were trained for the ensemble due to single class in y_train_bal.")