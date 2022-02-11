import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    data = data.drop('casual', axis=1)
    data = data.drop('registered', axis=1)

    cnt = data.pop('cnt')
    data.insert(data.shape[1], 'cnt', cnt)
    return data


def get_dummies_columns(data):
    data = pd.get_dummies(data, columns=['weekday'], prefix='weekday ', prefix_sep='')
    data = pd.get_dummies(data, columns=['mnth'], prefix='mnth ', prefix_sep='')
    data = pd.get_dummies(data, columns=['season'], prefix='season ', prefix_sep='')
    data = pd.get_dummies(data, columns=['yr'], prefix='year ', prefix_sep='')

    data = pd.get_dummies(data, columns=['weathersit'], prefix='weathersit ', prefix_sep='')

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
    ax.plot(date, test_predictions, label='predict')
    ax.plot(date, test_labels, label='true data')
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
data['season'].unique()
# %%
data = get_dummies_columns(data)
date = data['dteday']
data = data.drop('dteday', axis=1)

# %%
train_size = int(data.shape[0] / 10 * 8) + 1  # 80 procent

train_dataset = data.iloc[:train_size]

def adding_Gaussian_Noise_to_data(data, seed=2):
    np.random.seed(seed)
    columns = ['atemp', 'hum', 'windspeed','cnt']
    train_features_dsc = data.drop(columns, axis=1)
    gauss_noise_parameters = (1 - np.random.normal(0, 0.03, [data.shape[0], len(columns)]))
    gauss_noise_values = np.multiply(data[columns], gauss_noise_parameters)
    train_features_gaues_noise = pd.concat([train_features_dsc, gauss_noise_values], axis=1)
    train_features_gaues_noise = train_features_gaues_noise.append(train_features_dsc, ignore_index=True)

    return train_features_gaues_noise

train_features = adding_Gaussian_Noise_to_data(train_dataset, 2)

test_dataset = data.iloc[train_size:]
date_test = date[train_size:]

train_features = train_dataset.copy()
test_features = test_dataset.copy()
train_labels = train_features.pop('cnt')
test_labels = test_features.pop('cnt')


def build_and_compile_model_linear_model(norm):
    linear_model = tf.keras.Sequential([
        norm,
        layers.Dense(units=1)
    ])

    linear_model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=0.1),
        loss='mean_absolute_error')
    return linear_model


test_results = {}


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


# dnn_model.save('model/dnn_model')



normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))

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
plot_prediction(test_predictions, test_labels.values, date_test)
test_results['dnn_model'] = dnn_model.evaluate(
    test_features, test_labels, verbose=0)
# dnn_model.save('model/dnn_model_with_gauesse_noise')
# %%
def build_and_compile_model_cnn(norm):
    model = keras.Sequential([
        norm,
        layers.Conv1D(filters=64, kernel_size=7, activation='relu', name="Conv1D_1"),
        layers.Conv1D(filters=32, kernel_size=3, activation='relu', name="Conv1D_2"),
        layers.Dropout(0.2),
        layers.MaxPooling1D(pool_size=2, name="MaxPooling1D"),
        keras.layers.Flatten(),
        layers.Dense(32, activation='relu', name="Dense_1"),
        layers.Dense(1)
    ])

    model.compile(loss='mean_absolute_error',
                  optimizer=tf.keras.optimizers.Adam(0.001))
    return model


cnn_model = build_and_compile_model(normalizer)
cnn_model.summary()

history = cnn_model.fit(
    train_features,
    train_labels,
    validation_split=0.2,
    verbose=0, epochs=100,
)
plot_loss(history)

test_predictions = cnn_model.predict(test_features).flatten()
plot_prediction(test_predictions, test_labels.values, date_test)
test_results['cnn_model'] = cnn_model.evaluate(
    test_features, test_labels, verbose=0)

cnn_model.save('model/cnn_model')
# %%
test_results
from sklearn.model_selection import cross_val_score
errors = []
# the_best_feuteures = []
the_best_score = 0
features = list(train_features.columns.values)

the_best_feuteures = features.copy()
lr = LinearRegression()

for L in range(0, 1 + len(features)):  
    errors = [] 
    for value in features:
        # print(the_best_feuteures)
        # if value in the_best_feuteures:
        #     continue
        if value not in the_best_feuteures:
            continue

        tmp_the_best_feuteures = the_best_feuteures.copy()
        

        # data = pd.DataFrame(data= {'wyraz wolny' : np.ones(len(train_labels))},index=train_features.index)
        # tmp_the_best_feuteures.append(value)
        tmp_the_best_feuteures.remove(value)
        
        # data[tmp_the_best_feuteures] = train_features[tmp_the_best_feuteures]
        error = np.mean(cross_val_score(lr,train_features[tmp_the_best_feuteures], train_labels))
        errors.append((tmp_the_best_feuteures, error))
            
    errors = [(k,v) for k, v in sorted(errors, key=lambda item: item[1])]
 
    if  errors[-1][1] <  the_best_score and the_best_score != 0: # Jesli  dodanie cechy zwieksza bład, to zakoncz algorytm  
        break

    the_best_feuteures = errors[-1][0]
    the_best_score = errors[-1][1]


print('the_best_feuteures\n',the_best_feuteures)
print('the_best_cross_value score\n',the_best_score)

# %%
def build_and_compile_model_cnn(norm):
    model = keras.Sequential([
        norm,
        layers.Conv1D(filters=64, kernel_size=7, activation='relu', name="Conv1D_1"),
        layers.Dropout(0.2),
        layers.Conv1D(filters=32, kernel_size=3, activation='relu', name="Conv1D_2"),
        layers.MaxPooling1D(pool_size=2, name="MaxPooling1D"),
        keras.layers.Flatten(),
        layers.Dense(32, activation='relu', name="Dense_1"),
        layers.Dense(1)
    ])

    model.compile(loss='mean_absolute_error',
                  optimizer=tf.keras.optimizers.Adam(0.001))
    return model
    
cnn_model = build_and_compile_model(normalizer)
cnn_model.summary()

history = cnn_model.fit(
    train_features,
    train_labels,
    validation_split=0.2,
    verbose=0, epochs=200,
)

name_model = 'model_cnn'

predictions = cnn_model.predict(test_features).flatten()
scores[name_model]  = R_squared(test_labels, predictions)

list_of_preditions.append(predictions)
lablels.append(name_model)

plot_prediction(predictions,test_labels.values,date_test)
plot_loss(history)