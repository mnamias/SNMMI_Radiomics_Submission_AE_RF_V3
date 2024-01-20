# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 15:11:28 2023

@author: Mauro Namías - Fundación Centro Diagnóstico Nuclear - mnamias@fcdn.org.ar
SNMMI AI Taskforce - FDG PET/CT Radiomics Machine Learning Challenge 2023
"""
# %% 
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
from sklearn.impute import KNNImputer
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.datasets import make_swiss_roll
from scipy.special import inv_boxcox

from Preprocess import remove_constant_cols, fix_NaNs, yeo

# %%
# Load Training dataset
data = pd.read_excel('./dataset.xlsx')

# Select only data with events for PFS regression
data["Event"] = data["Event"].astype(bool)
data_events = data[data["Event"]==True]

Eventos = np.array(data["Event"])
Tiempos = np.array(data["Outcome_PFS"])

X = data_events.iloc[:,3:507]
y = data_events.Outcome_PFS
y = np.array(y)

X = X.astype(float)
X = fix_NaNs(X)
X = remove_constant_cols(X)

# %% Scale data with the Yeo Johnson transform
scaler = StandardScaler()
scaler.fit(X)
dump(scaler, 'scaler_yeo.joblib')

X_t = scaler.transform(X)
X_t, lambdas_X = yeo(X_t)

y_yeo, lambda_y = yeo(y)



    
device = ('cuda' if torch.cuda.is_available() else 'cpu')
    
x = torch.from_numpy(X_t).to(device)

# %% Define the Autoencoder for dimensionality reduction

# Set random the seed
seed = 42
torch.manual_seed(seed)

class Autoencoder(nn.Module):
    """Makes the main denoising auto

    Parameters
    ----------
    in_shape [int] : input shape
    enc_shape [int] : desired encoded shape
    """

    def __init__(self, in_shape, enc_shape):
        super(Autoencoder, self).__init__()
        
        self.encode = nn.Sequential(
            nn.Linear(in_shape, 128),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(64, enc_shape),
        )
        
        self.decode = nn.Sequential(
            nn.BatchNorm1d(enc_shape),
            nn.Linear(enc_shape, 64),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(128, in_shape)
        )
        
    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x
    
encoder = Autoencoder(in_shape=X_t.shape[1], enc_shape=10).double().to(device)

    
# %% Train the AE   
def train(model, error, optimizer, n_epochs, x):
    model.train()
    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()
        output = model(x)
        loss = error(output, x)
        loss.backward()
        optimizer.step()
        
        if epoch % int(0.1*n_epochs) == 0:
            print(f'epoch {epoch} \t Loss: {loss.item():.4g}')
            
error = nn.MSELoss()
optimizer = optim.Adam(encoder.parameters())

train(encoder, error, optimizer,5000, x)


def saveModel(net,optimizer,lambdas_X, lambda_y, PATH):
    torch.save({            
            'model_state_dict': net.cpu().state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lambdas_X': lambdas_X,
            'lambdas_y': lambda_y,
            }, PATH)
    return

saveModel(encoder.cpu(),optimizer,lambdas_X,lambda_y, './AE10f_V3.dat')

encoder.to(device)

encoder.eval()



with torch.no_grad():
    encoded = encoder.encode(x)

enc = encoded.cpu().detach().numpy()  # encoded features (N = 10)

  
# %% Train a Random Forest Regressor with 10 inputs 

random_state=20
from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor(n_estimators=100, bootstrap = True, min_samples_leaf = 1, max_features=10, n_jobs=-1, random_state=random_state)


# %% Leave One Out cross-validation
kf = KFold(n_splits=90,shuffle=True,random_state=1)
kf.get_n_splits(X)

RMSEs = []
predictions = []
gt = []

print(kf)
for i, (train_index, test_index) in enumerate(kf.split(X)):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}")
    print(f"  Test:  index={test_index}")
    X_train = enc[train_index]
    X_test = enc[test_index]
    
    y_train = y_yeo[train_index]
    y_test = y[test_index]
       
    regr.fit(X_train, y_train)

    y_pred = regr.predict(X_test)
    y_pred = np.array(inv_boxcox(y_pred,lambda_y))-1
    predictions.append(y_pred)
    gt.append(y_test)
    
    T_true = y_test
    
    media = (np.array(T_true) + np.array(y_pred))/2
    diff = np.array(y_pred) - T_true
    
    RMSE = np.sqrt(np.mean(diff**2))  # 6.93
    RMSEs.append(RMSE)

                 
print(np.mean(RMSEs))  # average RMSE
   


# %% Final training with complete dataset
regr.fit(enc, y_yeo)
dump(regr, 'AE10f_RF_V3.joblib')
