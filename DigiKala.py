

import pandas as pd
import numpy as np

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


dkProduct=pd.read_excel("C:/Users/h.safa/Downloads/Personal Files/DataScience/DataScience/Data Sets/Digikala Dataset/product.xlsx")
dkOrders=pd.read_csv("C:/Users/h.safa/Downloads/Personal Files/DataScience/DataScience/Data Sets/Digikala Dataset/orders.csv")

dkProduct.head()
dkProduct.shape
dkProduct.describe(include='all')
dkProduct.columns

dkOrders.head()
dkOrders.columns
dkOrders.describe(include='all')
