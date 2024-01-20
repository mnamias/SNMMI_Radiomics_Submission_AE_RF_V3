# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 15:11:28 2023

@author: Mauro Namías - Fundación Centro Diagnóstico Nuclear - mnamias@fcdn.org.ar
SNMMI AI Taskforce - FDG PET/CT Radiomics Machine Learning Challenge 2023
"""
# %% 
import numpy as np
import pandas as pd
from joblib import load
from sklearn.impute import KNNImputer
import torch
import torch.nn as nn
from scipy.special import inv_boxcox


from Preprocess import fix_NaNs, yeo, yeo_lmbda, remove_constant_cols
#%% Inference on blind test dataset

# %%
# Load Training dataset
data = pd.read_excel('./dataset.xlsx')

# Select only data with events for PFS regression
data["Event"] = data["Event"].astype(bool)
data_events = data[data["Event"]==True]
X_train = data_events.iloc[:,3:507]

data = pd.read_excel('./SNMMI_CHALLENGE_TESTING_V01112023.xlsx')
X_test = data.iloc[:,1:]
X_test.rename(columns = {'ExactVolume':'ExactVolume (uL)'}, inplace = True) 

frames = [X_train, X_test]
frames = pd.concat(frames)

frames = fix_NaNs(frames)
frames = remove_constant_cols(frames)
X_test = frames.iloc[90:,]

checkpoint = torch.load('./AE10f_V3.dat')

lambdas_X = checkpoint['lambdas_X']
lambda_y = checkpoint['lambdas_y']


scaler = load('scaler_yeo.joblib')
X_test = scaler.transform(X_test)

X_test = yeo_lmbda(X_test, lambdas_X)



# Load trained models
 ## Load trained model for inference
torch.cuda.empty_cache()


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


device = ('cuda' if torch.cuda.is_available() else 'cpu')
encoder = Autoencoder(in_shape=X_test.shape[1], enc_shape=10).double().to(device)


encoder.load_state_dict(checkpoint['model_state_dict'])
del checkpoint
encoder.eval()



regr = load('AE10f_RF_V3.joblib')


x = torch.from_numpy(X_test).to(device)


with torch.no_grad():
    encoded = encoder.encode(x)

enc = encoded.cpu().detach().numpy()

y_pred = regr.predict(enc)

for i in range(y_pred.shape[0]):
    y_pred[i] = inv_boxcox(y_pred[i],lambda_y) - 1
    


# dump results to csv
np.savetxt("AE10f_RF_V3_results.csv", y_pred, delimiter=",")
