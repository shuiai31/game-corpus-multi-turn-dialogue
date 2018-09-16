#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# Load training data set from CSV file
training_data_df = pd.read_csv("game_question_answer_training.csv", dtype=float)

X_training = training_data_df.drop('question', axis=1).values
Y_training = training_data_df[['question']].values

# Load testing data set from CSV file
test_data_df = pd.read_csv("game_question_answer_test.csv", dtype=float)

X_testing = training_data_df.drop('question', axis=1).values
Y_testing = training_data_df[['question']].values

X_scaler = MinMaxScaler(feature_range=(0, 1))
Y_scaler = MinMaxScaler(feature_range=(0, 1))

X_scaled_training = X_scaler.fit_transform(X_training)
Y_scaled_training = Y_scaler.fit_transform(Y_training)

X_scaled_testing = X_scaler.transform(X_testing)
Y_scaled_testing = Y_scaler.transform(Y_testing)

print(X_scaled_testing.shape)
print(Y_scaled_testing.shape)

print("Note: Y values were scaled by multiplying by {:.10f} and adding {:.4f}".format(Y_scaler.scale_[0], Y_scaler.min_[0]))
