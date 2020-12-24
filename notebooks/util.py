import numpy as np
import pandas as pd
import os,sys,time

# visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# machine learning
import sklearn.metrics as skmetrics
import scikitplot.metrics as skpmetrics

def custom_loss(y_true, y_pred):
    tn, fp, fn, tp = skmetrics.confusion_matrix(y_true,y_pred).ravel()
    loss = 400*tp - 200*fn - 100*fp
    return loss