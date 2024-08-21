import numpy as np
import glob
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import timeit

class BinaryClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layer_sizes=(100, 200, 300,400)):
        super(BinaryClassifier, self).__init__()
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_layer_sizes[0]))
        layers.append(nn.ReLU())

        # Hidden layers
        for i in range(1, len(hidden_layer_sizes)):
            layers.append(nn.Linear(hidden_layer_sizes[i - 1], hidden_layer_sizes[i]))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_layer_sizes[-1], output_dim))
        # Sigmoid activation for binary classification output
        layers.append(nn.Sigmoid())

        # Register all layers
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


   
# Load Filtered_ShadowMasks.npz
filtered_data = np.load('Filtered_ShadowMasks_dpinit.npz')

# Load efits_all.npz
efits_data = np.load('efits_all.npz')


# Remove non-diverted plasmas using Zlowest as parameter
Zlowest = efits_data['Zlowests']
mask = Zlowest < -0.9


# Original indices
original_index = np.arange(len(Zlowest))

# Filter the features and target data according to the mask
X = np.column_stack(( efits_data['Ips'][mask], efits_data['q95s'][mask], efits_data['alpha1s'][mask], efits_data['alpha2s'][mask]))
y = filtered_data['ShadowMasks']
filtered_index = original_index[mask]

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)


# Find the indices where the values change in the ShadowMasks array
indices_cvxhull = filtered_data['indices'] #indices into the full ~500k array
changed_indices = np.where(np.any(filtered_data['ShadowMasks'] != filtered_data['ShadowMasks'][0], axis=0))[0]
indices = indices_cvxhull[changed_indices]
# Create new  output using the indices of the elements that change

y_changed = filtered_data['ShadowMasks'][:, changed_indices]
y_changed= y_changed[mask][:]

# Print the shapes of the new variables ouputs

print("Shape of y_changed:", y_changed.shape)

# Split the data into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X, y_changed, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test, train_index, test_index = train_test_split(X, y_changed, filtered_index, test_size=0.2, random_state=42)

# Normalize the input features
input_norm = StandardScaler()
input_norm.fit(X_train)
X_train = input_norm.transform(X_train)
X_test = input_norm.transform(X_test)

# When converting from NumPy to PyTorch Tensor, specify the dtype as torch.float32
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


# Define the Skorch NeuralNetClassifier
net = NeuralNetClassifier(
    module=BinaryClassifier,
    module__input_dim=X_train.shape[1],
    module__output_dim=y_train.shape[1],
    criterion=nn.BCELoss,
    optimizer=optim.Adam,
    max_epochs=600,
    verbose=1,
    train_split=None,
    #device='cpu'
    device='cuda' if torch.cuda.is_available() else 'cpu'
   
)

# Train the model
start_time = timeit.default_timer()
net.fit(X_train, y_train)
elapsed_time = timeit.default_timer() - start_time
print(f"Training completed in {elapsed_time:.2f} seconds.")

# Evaluate the model
y_pred = net.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared:", r2)
print("Accuracy:", accuracy)

individual_r2 = []

for i in range(len(y_test)):
    individual_r2.append(r2_score(y_test[i], y_pred[i]))

# Convert the list to a numpy array
individual_r2_array = np.array(individual_r2)


print("Saving important data at ml_pred.npz")
np.savez('ml_pred_diverted_dpinit.npz',y=y,indices=indices, y_changed=y_changed,changed_indices=changed_indices,y_test=y_test, y_pred=y_pred,all_r2=individual_r2_array,train_index=train_index,test_index=test_index)


scaler_pkl = pickle.dumps(input_norm)
data_to_save = {'state_dict': net.module_.state_dict(),
                'input_dim':net.module__input_dim,
                'output_dim':net.module__output_dim,
                'indices': indices,
                'scaler_pkl': scaler_pkl}


torch.save(data_to_save, 'model_state_dict_NN_skorch.pth')
