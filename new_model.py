# Create the Python script for Random Forest training and evaluation

rf_script = """
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
ATL_RE_Data_encoded = pd.read_csv('/mnt/data/ATL_RE_Data.csv')

# Preprocess the data
# Drop the 'State' and 'Tags' columns
ATL_RE_Data_encoded.drop(columns=['State', 'Tags'], inplace=True)

# One-hot encode 'City', 'County', and 'Property Type' columns
ATL_RE_Data_encoded = pd.get_dummies(ATL_RE_Data_encoded, columns=['City', 'County', 'Property Type'], drop_first=True)

# Convert 'Sold Date' to datetime and extract year, month, and day as new features
ATL_RE_Data_encoded['Sold Date'] = pd.to_datetime(ATL_RE_Data_encoded['Sold Date'])
ATL_RE_Data_encoded['Sold Year'] = ATL_RE_Data_encoded['Sold Date'].dt.year
ATL_RE_Data_encoded['Sold Month'] = ATL_RE_Data_encoded['Sold Date'].dt.month
ATL_RE_Data_encoded['Sold Day'] = ATL_RE_Data_encoded['Sold Date'].dt.day

# Drop the 'Sold Date' column
ATL_RE_Data_encoded.drop(columns=['Sold Date'], inplace=True)

# Split the data into training and testing sets
X = ATL_RE_Data_encoded.drop(columns=['Sold Price', 'List Date', 'Unnamed: 0'])
y = ATL_RE_Data_encoded['Sold Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

# Predict on the test set
y_pred_rf = model_rf.predict(X_test)

# Calculate and print metrics
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Mean Squared Error (MSE): {mse_rf}")
print(f"Root Mean Squared Error (RMSE): {rmse_rf}")
print(f"R-squared (R2): {r2_rf}")

"""

# Save the script to a file
file_path = "/mnt/data/random_forest_script.py"
with open(file_path, "w") as file:
    file.write(rf_script)

file_path
