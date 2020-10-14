#Natural Language Processing (NLP)

 #Bag of words (BOW) 
 #TF IDF : PCA
 #Wordzvec


import pandas as pd
import numpy as np

train = pd.read_csv("train.csv",
                    sep=",",names=["id", "premise", "hypothesis","lang_abv","language","label"])      #Import csv training file

train.head()
