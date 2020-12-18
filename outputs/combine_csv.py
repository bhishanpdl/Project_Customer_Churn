#!python
# -*- coding: utf-8 -*-#
"""
* File Name : combine_csv.py

* Purpose :

* Creation Date : Dec 18, 2020 Fri

* Last Modified : Fri Dec 18 17:08:02 2020

* Created By :  Bhishan Poudel

"""
# Imports 
import numpy as np
import pandas as pd
import os,sys,glob

files = glob.glob('*.csv')
files = [i for i in files if i!='combined.csv']
dfs = [pd.read_csv(f,index_col=0) for f in files]
df = pd.concat(dfs,axis=0)

print(df)
df.to_csv('combined.csv')
