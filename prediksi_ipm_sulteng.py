# PREDIKSI INDEKS PEMBANGUNAN MANUSIA (IPM) SULAWESI TENGAH
# Menggunakan Regresi Linear dan Support Vector Classification
# Dataset: BPS Sulawesi Tengah 2019-2023
 
# 1. Import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
 
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC, SVR
 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, classification_report