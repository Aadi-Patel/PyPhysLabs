# Package imports
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import pandas as pd
import os

# Setting variables
DataFile = 'CurveFitTest/TestData.csv'
x_units = 's'
x_label = 'Time'
y_units = 'm'
y_label = 'Signal'
title = 'Curve Fit Test for HMO'

# Fitting function
def test_func(t, A, b, w, phi, Xeq):
    return A*np.exp(-b*t)*np.cos(w*t-phi) + Xeq

p0 = [3, 0.5, 1.5, 4.5, 0] # initial guess for fit parameters (does not include dependent variable)

yerror = 0.193 # first guess this then input the rms value to get chi squared to 1
N_datapoints = 706 # number of data points

# ------ Script starts here ------

# Load data from CSV file
data = pd.read_csv(DataFile)  # read_csv uses working directory (PyPhysLabs) so specifcy locaiton
x_data = data[x_label].values  # Replace 'x' with your actual column name for x values
y_data = data[y_label].values  # Replace 'y' with your actual column name for y values

params, params_covariance = optimize.curve_fit(test_func, x_data, y_data,
                                               p0=p0)

print("Parameter output:")
print(params,'\n')

 # Make covariance matrix
print("Covariance Matrix: (rounded to 3 decimal places)")
print(np.round(params_covariance, decimals=3), '\n')

 # Make correlation matrix from the covariance matrix
Diag = np.diag(np.sqrt(np.diag(params_covariance))) #calculate the diagnol of covariance
Diag = np.linalg.inv(Diag)  #Determine the inverse of the diagonal
cor = np.matmul(Diag,params_covariance) #matrix multiplication of Diagnol and covariance
cor = np.matmul(cor,Diag)
print("Correlation Matrix: (rounded to 3 decimal places)")                           
print(np.round(cor, decimals=3), '\n')

Ytheo = test_func(x_data, *params)

res = y_data - Ytheo
rms_residual = np.sqrt(np.mean(res**2))

 #determination of chi squared
chitwo = np.sum((res**2/yerror**2)) 

print("Chi-Squared: ", chitwo)
print("Reduced Chi-Squared: ", chitwo/N_datapoints)


# Plotting
fig = plt.figure(figsize=(8, 6)) # Set figure size
ax = fig.add_subplot(211) # Top subplot (2 rows, 1 column, first plot)
ay = fig.add_subplot(313) # Bottom subplot (2 rows, 1 column, third plot)
fig.subplots_adjust(top = 0.85) # Adjust top margin

# Top subplot: Data and fit
ax.set_ylabel(y_label + ' (' + y_units + ')')

ax.plot(x_data, y_data, label='Data', markersize=1)
ax.plot(x_data,Ytheo , label='Fit', color='red')

ax.set_title(title, fontsize=14, pad=10)
ax.legend(fontsize=7, loc='upper right')
ax.grid(True)

# Bottom subplot: Residuals
ay.set_xlabel(x_label + ' (' + x_units + ')')
ay.set_ylabel('Residual')

ay.text(0.98, 0.95, f'RMS Residual = {rms_residual:.3f} m', # Format the RMS residual to 3 decimal places
        transform=ay.transAxes,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        fontsize=8)

ay.plot(x_data,res)
ay.grid(True)
ay.axhline(y=0, color='r', linestyle='--', alpha=0.5)  # Add a reference line at y=0

# Saving plots
plt.savefig('dho_fit.pdf')
#plt.show()