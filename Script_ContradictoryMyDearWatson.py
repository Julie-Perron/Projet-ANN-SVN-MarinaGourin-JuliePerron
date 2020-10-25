#Natural Language Processing (NLP)

 #Bag of words (BOW) 
 #TF IDF : PCA
 #Wordzvec

#Library
#Library
import pandas as pd
import numpy as np

train = pd.read_csv("train.csv",
                    sep=",",header=0)      #Import csv training file

test = pd.read_csv("test.csv",
                    sep=",",header=0)      #Import csv test file

train.head()

test.head()
