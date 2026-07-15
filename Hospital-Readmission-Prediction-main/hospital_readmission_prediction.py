import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("dataset/hospital_readmission.csv")

# Drop patient ID (not useful)
data.drop("patient_id", axis=1, inplace=True)

# Split blood pressure into systolic and diastolic
bp_split = data["blood_pressure"].str.split("/", expand=True)
data["bp_systolic"] = bp_split[0].astype(int)
data["bp_diastolic"] = bp_split[1].astype(int)
data.drop("blood_pressure", axis=1, inplace=True)

# Convert Yes/No columns to numeric
yes_no_cols = ["diabetes", "hypertension", "readmitted_30_days"]
for col in yes_no_cols:
    data[col] = data[col].map({"Yes": 1, "No": 0})

# Encode categorical columns
label_encoder = LabelEncoder()
categorical_cols = ["gender", "discharge_destination"]

for col in categorical_cols:
    data[col] = label_encoder.fit_transform(data[col])

# Features and target
X = data.drop("readmitted_30_days", axis=1)
y = data["readmitted_30_days"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
