#Natural Language Processing (NLP)
 #Bag of words (BOW) 
 #TF IDF : PCA
 #Wordzvec

#Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv("train.csv",
                    sep=",",header=0)      #Import csv training filed()

train.head()    #Take a look at the DataFrame

train.iloc[2]['premise']
train.iloc[2]['hypothesis']
train.iloc[2]['label']


train.language.unique()      #The list of languages in the training data


#Plotting the distribution of languages in the training data
fig, ax = plt.subplots(figsize=(12, 9), subplot_kw=dict(aspect="equal"))

languages, frequencies = np.unique(train.language.values, return_counts = True)

wedges, texts = ax.pie(frequencies, wedgeprops=dict(width=0.5), startangle=-40)

bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
kw = dict(arrowprops=dict(arrowstyle="-"),
          bbox=bbox_props, zorder=0, va="center")

for i, p in enumerate(wedges):
    ang = (p.theta2 - p.theta1)/2. + p.theta1
    y = np.sin(np.deg2rad(ang))
    x = np.cos(np.deg2rad(ang))
    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
    connectionstyle = "angle,angleA=0,angleB={}".format(ang)
    kw["arrowprops"].update({"connectionstyle": connectionstyle})
    ax.annotate(languages[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                horizontalalignment=horizontalalignment, **kw)

ax.set_title("Distribution of languages in the training data")

plt.show()
