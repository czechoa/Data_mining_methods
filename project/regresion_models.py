import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plotly.graph_objects as go
import seaborn as sns

# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)


def wind(row):
    if row['windspeed'] > 0:
        return 1
    elif row['windspeed'] == 0:
        return 0


def prepared_data():
    data_hour = pd.read_csv('project/data/day.csv')
    data_day = pd.read_csv('project/data/day.csv')
    data = data_hour
    data['dteday'] = pd.to_datetime(data['dteday'])
    data = data.drop('instant', axis=1)
    data = data.drop(columns=['temp'])
    # data["procent_casual"] = data.apply(lambda row: row.casual / row.cnt, axis=1)
    # data["procent_registered"] = data.apply(lambda row: row.registered / row.cnt, axis=1)
    data["is_wind"] = data.apply(lambda row: wind(row), axis=1)
    data = data.drop('casual',axis=1)
    data= data.drop('registered',axis=1)

    cnt = data.pop('cnt')
    data.insert(data.shape[1], 'cnt', cnt)
    return data

def get_dummies_columns(data):
    data = pd.get_dummies(data, columns=['weekday'], prefix='weekday ', prefix_sep='')
    data = pd.get_dummies(data, columns=['mnth'], prefix='mnth ', prefix_sep='')
    data = pd.get_dummies(data, columns=['season'], prefix='season ', prefix_sep='')

    cnt = data.pop('cnt')
    data.insert(data.shape[1], 'cnt', cnt)
    return data

def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    # plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [cnt]')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_prediction(test_predictions, test_labels, date):
    # data1 = data[['dteday', 'cnt']]
    fig, ax = plt.subplots(figsize=(20, 5))
    ax.plot(date, test_predictions,label='predict')
    ax.plot(date, test_labels,label = 'true data')
    plt.setp(ax.get_xticklabels(), rotation=45)
    plt.title('Wypożyczenia w danych dniach testowych')
    plt.legend()
    plt.show()

def plot_prediction_error():
    error = test_predictions - test_labels
    plt.hist(error, bins=25)
    plt.xlabel('Prediction Error [cnt]')
    _ = plt.ylabel('Count')
    plt.show()



data = prepared_data()
data = get_dummies_columns(data)
date = data['dteday']
data = data.drop('dteday', axis=1)
# train_dataset = data.sample(frac=0.8, random_state=0)
# train_dataset = data.sample(frac=0.8, random_state=0)
train_size = int(data.shape[0] / 10 * 8) + 1  # 80 procent

train_dataset = data.iloc[:train_size]
train_date = date[:train_size]

test_dataset = data.iloc[train_size:]
date_test = date[train_size:]

train_features = train_dataset.copy()
test_features = test_dataset.copy()
train_labels = train_features.pop('cnt')
test_labels = test_features.pop('cnt')


normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))


def build_and_compile_model_linear_model(norm):
    linear_model = tf.keras.Sequential([
        norm,
        layers.Dense(units=1)
    ])

    linear_model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=0.1),
        loss='mean_absolute_error')
    return linear_model


history = linear_model.fit(
    train_features,
    train_labels,
    epochs=1000,
    # Suppress logging.
    verbose=0,
    # Calculate validation results on 20% of the training data.
    validation_split=0.2)

plot_loss(history)

test_results = {}

test_results['dnn_model'] = dnn_model.evaluate(test_features, test_labels, verbose=0)

test_predictions = linear_model.predict(test_features).flatten()
plot_prediction(test_predictions,test_labels.values,date_test)

def build_and_compile_model(norm):
    model = keras.Sequential([
        norm,
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),

        layers.Dense(1)
    ])

    model.compile(loss='mean_absolute_error',
                  optimizer=tf.keras.optimizers.Adam(0.001))
    return model


dnn_model = build_and_compile_model(normalizer)
dnn_model.summary()

history = dnn_model.fit(
    train_features,
    train_labels,
    validation_split=0.2,
    verbose=0, epochs=200,
)
plot_loss(history)

test_predictions = dnn_model.predict(test_features).flatten()
plot_prediction(test_predictions,test_labels.values,date_test)
dnn_model.save('model/dnn_model')


