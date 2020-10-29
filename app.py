import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df=pd.read_csv("bayes.csv",index_col=None)
titles=df.columns
X=df[titles[:-1]]
y=df[titles[-1]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#joining label with features of Test and Train

train=X_train.join(y_train)
test=X_test.join(y_test)
#X_test_list=X_test.values.tolist()


accuracy=0
for tests in test.iterrows():
    prob=1
    not_prob=1
    tests=tests[1]  #removing index values
    probability=[0]*len(test)
    not_probability=[0]*len(test)
    #NULL INITIALIZATIONS
    #where probability list is count of conditional probability of given class
    #      not_probability is count of conditional probability of opposite calss
    
    
    for trains in train.iterrows():
        trains=trains[1] #removing index values (i.e, the first column)
        for index,value in enumerate(trains):
            if value==tests[index] and trains[-1]==tests[-1]:
                probability[index]+=1  #addition of conditional sample space
            if value==tests[index] and trains[-1]!=tests[-1]:
                not_probability[index]+=1 #addition of opposite sample space
        not_probability[-1]=abs(len(train)-probability[-1])   #correcting last value of opposite case (for generalized Algo)
    
    for values in probability[:-1]:
        prob=prob*(values/probability[-1])   #applying Naive Bayes on given condition
    for values in not_probability[:-1]:
        not_prob=not_prob*(values/not_probability[-1])  #applying Naive Bayes on opposite Condition
    prob*=(probability[-1]/len(train))
    not_prob*=(not_probability[-1]/len(train))      #multiplying of given class' probability for given and opposite both cases
#    print(probability)
#    print(not_probability)
#    print(prob)
#    print(not_prob)
    if prob>=not_prob and tests[-1]=='yes':
        accuracy+=1                             #checking which probility is giving correct label
    if prob<not_prob and tests[-1]=='no':
        accuracy+=1        
accuracy/=len(test)
accuracy*=100
print(accuracy)
    
    
    
    
