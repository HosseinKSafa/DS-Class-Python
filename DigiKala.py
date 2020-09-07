# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 16:05:49 2020

@author: h.safa
"""

import pandas as pd
import numpy as np

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


dkProduct=pd.read_excel("C:/Users/h.safa/Downloads/Personal Files/digikala_dataset[www.camelcase.ir]/product.xlsx")
dkOrders=pd.read_csv("C:/Users/h.safa/Downloads/Personal Files/digikala_dataset[www.camelcase.ir]/orders.csv")

dkProduct.head()
dkProduct.shape
dkProduct.describe(include='all')
dkProduct.columns

dkOrders.describe(include='all')
