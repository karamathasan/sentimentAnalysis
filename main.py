import pandas as pd
import numpy as np
# from sklearn.preprocessing 
import gensim
from torch import nn

from env import SECRET
import re

data = pd.read_csv(SECRET,names=["target", "ids", "date", "flag", "user", "text"], nrows=20)
print(data)