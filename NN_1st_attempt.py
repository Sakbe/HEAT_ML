import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor

# Load Filtered_ShadowMasks.npz
filtered_data = np.load('Filtered_ShadowMasks.npz')

# Print variables and their shapes
print("Variables in Filtered_ShadowMasks.npz:")
for var_name in filtered_data.files:
    var_shape = filtered_data[var_name].shape
    print(f"- {var_name}: {var_shape}")

# Load efits_all.npz
efits_data = np.load('efits_all.npz')

# Print variables and their shapes
print("\nVariables in efits_all.npz:")
for var_name in efits_data.files:
    var_shape = efits_data[var_name].shape
    print(f"- {var_name}: {var_shape}")

####### Never changing points ###########

# Count the number of points that don't change for all 1212 cases
unchanged_points = np.sum(np.all(filtered_data['ShadowMasks'] == filtered_data['ShadowMasks'][0], axis=0))

print("Number of points that don't change for all 1212 cases:", unchanged_points)



#### ML training #########################


# Extract input features and target variable
X = np.column_stack((efits_data['Bt0s'], efits_data['Ips'], efits_data['q95s'], efits_data['alpha1s'], efits_data['alpha2s']))
y = filtered_data['ShadowMasks']

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# Find the indices where the values change in the ShadowMasks array
changed_indices = np.where(np.any(filtered_data['ShadowMasks'] != filtered_data['ShadowMasks'][0], axis=0))[0]

# Create new  output using the indices of the elements that change

y_changed = filtered_data['ShadowMasks'][:, changed_indices]

# Print the shapes of the new variables

print("Shape of y_changed:", y_changed.shape)



# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_changed, test_size=0.2, random_state=42)

# Initialize MLP Regressor
mlp_regressor = MLPRegressor(hidden_layer_sizes=(100,100,100), activation='relu', solver='adam', random_state=42)

# Train the MLP Regressor
mlp_regressor.fit(X_train, y_train)

# Evaluate the model
y_pred = mlp_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared:", r2)



