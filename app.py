#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
# import required models
from sklearn.linear_model import LinearRegression
import flask
from flask import Flask, render_template, request
import pickle

# In[2]:
filename = 'first-innings-score-lr-model.pkl'
regressor = pickle.load(open(filename, 'rb'))

app = flask.Flask(__name__, template_folder='Templates')


# In[3]:


ipl_matches = pd.read_csv("ipl_matches.csv")


# In[4]:


X = ipl_matches.iloc[:,[1,5,6,8,9,10,11,12,13,14,15,16,17,18,19,20,
                      21,22,23,24,25,26,27]].values #Input features
y = ipl_matches.iloc[:, 7].values #output


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.25,
                                                    random_state = 0)


# In[6]:


# train and test the model
lin = LinearRegression()
lin.fit(X_train,y_train)


# In[ ]:


@app.route('/')
def home():
    return render_template('main.html')


# In[7]:


@app.route('/predict', methods=['POST'])

def predict_final_score():
    if request.method == "POST":
        temp = list()
        batting_team = request.form["batting-team"]
    
        if batting_team == 'Chennai Super Kings':
            temp = temp + [1,0,0,0,0,0,0,0]
        elif batting_team == 'Delhi Capitals':
            temp = temp + [0,1,0,0,0,0,0,0]
        elif batting_team == 'Punjab Kings':
            temp = temp + [0,0,1,0,0,0,0,0]
        elif batting_team == 'Kolkata Knight Riders':
            temp = temp + [0,0,0,1,0,0,0,0]
        elif batting_team == 'Mumbai Indians':
            temp = temp + [0,0,0,0,1,0,0,0]
        elif batting_team == 'Rajasthan Royals':
            temp = temp + [0,0,0,0,0,1,0,0]
        elif batting_team == 'Royal Challengers Bangalore':
            temp = temp + [0,0,0,0,0,0,1,0]
        elif batting_team == 'Sunrisers Hyderabad':
            temp = temp + [0,0,0,0,0,0,0,1]
    
        bowling_team = request.form["bowling-team"]
        
        if bowling_team == 'Chennai Super Kings':
            temp = temp + [1,0,0,0,0,0,0,0]
        elif bowling_team == 'Delhi Capitals':
            temp = temp + [0,1,0,0,0,0,0,0]
        elif bowling_team == 'Punjab Kings':
            temp = temp + [0,0,1,0,0,0,0,0]
        elif bowling_team == 'Kolkata Knight Riders':
            temp = temp + [0,0,0,1,0,0,0,0]
        elif bowling_team == 'Mumbai Indians':
            temp = temp + [0,0,0,0,1,0,0,0]
        elif bowling_team == 'Rajasthan Royals':
            temp = temp + [0,0,0,0,0,1,0,0]
        elif bowling_team == 'Royal Challengers Bangalore':
            temp = temp + [0,0,0,0,0,0,1,0]
        elif bowling_team == 'Sunrisers Hyderabad':
            temp = temp + [0,0,0,0,0,0,0,1]
        
        overs = float(overs.form["overs"])
        striker_score = int(striker_score.form["striker_score"])
        nonstriker_score = int(nonstriker_score.form["non_striker_score"])
        runs = int(runs.form["runs"])
        instant_wicket = int(instant_wicket.form["wickets"])
        runs_last_5 = int(runs_last_5.form["runs_last_5"])
        wickets_last_5 = int(wickets_last_5.form["wickets_last_5"])
        sample_input = [overs,runs,instant_wicket,striker_score,nonstriker_score,
                        runs_last_5,wickets_last_5] + temp
        
        score = lin.predict(np.array([sample_input]))
        
        return render_template('result.html' ,lower_limit = round(score[0]-10),
                               upper_limit = round(score[0]+10))



if __name__ == '__main__':
    app.run(debug=True)
