from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from matplotlib import pyplot as plt
from numpy import arange
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold

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
    data = data.drop('registered',axis=1)

    # data = pd.get_dummies(data, columns=['weekday'], prefix='weekday ', prefix_sep='')
    # data = pd.get_dummies(data, columns=['mnth'], prefix='mnth ', prefix_sep='')
    # data = pd.get_dummies(data, columns=['season'], prefix='season ', prefix_sep='')

    cnt = data.pop('cnt')
    data.insert(data.shape[1], 'cnt', cnt)
    return data

def plot_prediction(test_predictions, test_labels, date):
    # data1 = data[['dteday', 'cnt']]
    fig, ax = plt.subplots(figsize=(20, 5))
    ax.plot(date, test_predictions,label='predict')
    ax.plot(date, test_labels,label = 'true data')
    plt.setp(ax.get_xticklabels(), rotation=45)
    plt.title('Wypożyczenia w danych dniach testowych')
    plt.legend()
    plt.show()
data = prepared_data()
# data = get_dummies_columns(data)
date = data['dteday']
data = data.drop('dteday', axis=1)

# train_dataset = data.sample(frac=0.8, random_state=1)
# test_dataset = data.drop(train_dataset.index)
train_size = int(data.shape[0] / 10 * 8) + 1  # 80 procent

train_dataset = data.iloc[:train_size]
train_date = date[:train_size]

test_dataset = data.iloc[train_size:]
date_test = date[train_size:]

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('cnt')
test_labels = test_features.pop('cnt')

# Wybór cechy

model = RandomForestClassifier(n_estimators=100,random_state=1)
embeded_rf_selector = SelectFromModel(model, max_features=10)
embeded_rf_selector.fit(train_features, train_labels)

embeded_rf_support = embeded_rf_selector.get_support()
embeded_rf_feature = train_features.loc[:,embeded_rf_support].columns.tolist()
print(str(len(embeded_rf_feature)), 'selected features')
print(embeded_rf_feature)

select_features = embeded_rf_feature
model.fit(train_features[embeded_rf_feature],train_labels)
preditions = model.predict(test_features[embeded_rf_feature])
plot_prediction(preditions,test_labels,date_test)
# RandomForestClassifier_score = model.score(test_features[embeded_rf_feature],test_labels)
# plot_prediction(preditions,test_labels,date_test)


# %%
# define model evaluation method

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# define model
model = RidgeCV(alphas=arange(0, 1, 0.1), cv=cv, scoring='neg_mean_absolute_error')
# fit model
model.fit(train_features[embeded_rf_feature], train_labels)

# summarize chosen configuration
print(f'alpha: {model.alpha_}')
# %%
model = LinearRegression()
model.fit(train_features[embeded_rf_feature], train_labels)

model.predict(train_features[embeded_rf_feature])
preditions = model.predict(test_features[embeded_rf_feature])
# plot_prediction(preditions,test_labels,date_test)
score = model.score(test_features[embeded_rf_feature],test_labels)
print(score)
print(model.coef_)

plot_prediction(preditions,test_labels,date_test)
