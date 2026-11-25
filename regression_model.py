"""
Regression-based data-driven modeling module.
"""
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def train_regression_model(t_train, x_train, degree=6):
    """
    Trains a polynomial regression model to approximate x(t).
    
    Parameters:
    t_train (numpy.ndarray): Training time points
    x_train (numpy.ndarray): Training target displacements
    degree (int): Polynomial degree to use for regression
    
    Returns:
    model: standard scikit-learn LinearRegression model trained on polynomial features
    poly: PolynomialFeatures instances used for transformations
    """
    # Polynomial transformation of t
    poly = PolynomialFeatures(degree=degree)
    t_train_poly = poly.fit_transform(t_train.reshape(-1, 1))
    
    # Linear Regression over the polynomial features
    model = LinearRegression()
    model.fit(t_train_poly, x_train)
    
    return model, poly

def predict_regression(model, poly, t_test):
    """
    Predicts displacement using the trained regression model.
    """
    t_test_poly = poly.transform(t_test.reshape(-1, 1))
    x_pred = model.predict(t_test_poly)
    return x_pred
