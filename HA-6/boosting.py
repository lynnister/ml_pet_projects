from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])

class Boosting:
    def __init__(
            self,
            base_model_params: dict = None,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            subsample: float = 0.3,
            early_stopping_rounds: int = None,
            plot: bool = False,
    ):
        self.base_model_class = DecisionTreeRegressor
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate
        self.subsample: float = subsample

        self.early_stopping_rounds: int = early_stopping_rounds
        if early_stopping_rounds is not None:
            self.validation_loss = np.full(self.early_stopping_rounds, np.inf)

        self.plot: bool = plot

        self.history = defaultdict(list)

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)

    def fit_new_base_model(self, x, y, predictions):
        #Создаем бутстрап:
        bootstrap_size = int(y.shape[0] * self.subsample)
        inds = np.random.randint(low = 0, high = y.shape[0]-1, size = bootstrap_size)
        x_bootstrap = x[inds]
        y_bootstrap = y[inds]
        predictions_bootstrap = predictions[inds]
        s_bootstrap = -self.loss_derivative(y_bootstrap, predictions_bootstrap) #сдвиг на данном этапе

        #Обучаем новую модель на полученной подвыборке
        model = self.base_model_class(**self.base_model_params)
        model = model.fit(x_bootstrap, s_bootstrap)
        gamma = self.find_optimal_gamma(y, predictions, model.predict(x))
        self.gammas.append(gamma * self.learning_rate)
        self.models.append(model)

    def fit(self, x_train, y_train, x_valid, y_valid):
        """
        :param x_train: features array (train set)
        :param y_train: targets array (train set)
        :param x_valid: features array (validation set)
        :param y_valid: targets array (validation set)
        """
        train_predictions = np.zeros(y_train.shape[0])
        valid_predictions = np.zeros(y_valid.shape[0])
        train_loss = [self.loss_fn(valid_predictions, y_valid).sum()]
        validation_loss = [self.loss_fn(valid_predictions, y_valid).sum()]
            
        for _ in range(self.n_estimators):
            self.fit_new_base_model(x_train, y_train, train_predictions)
            train_predictions += self.models[-1].predict(x_train) * self.gammas[-1]  #добавляем новую модель
            train_loss.append(self.loss_fn(train_predictions, y_train).sum())
            valid_predictions += self.models[-1].predict(x_valid) * self.gammas[-1]
            validation_loss.append(self.loss_fn(valid_predictions, y_valid).sum())
    
            if self.early_stopping_rounds is not None:
                if (_ + 1 - self.early_stopping_rounds >= 0) and (validation_loss[_ + 1] >= validation_loss[_ + 1 - self.early_stopping_rounds]): #критерий останова для ESR
                    break

        if self.plot:
            plt.figure(figsize=(16, 7))
            plt.plot(train_loss)
            plt.xlabel("Число моделей в ансамбле")
            plt.ylabel("Значение функции потерь")
            plt.title("График потерь градиентного бустинга в зависимости от числа моделей в ансамбле")

    def predict_proba(self, x):
        result = np.zeros(x.shape[0])
        for gamma, model in zip(self.gammas, self.models):
            result += gamma * model.predict(x)
        return np.vstack((1 - self.sigmoid(result), self.sigmoid(result))).T 

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]
        return gammas[np.argmin(losses)]

    def score(self, x, y):
        return score(self, x, y)

    @property
    def feature_importances_(self):
        feature_importances = None
        for model in self.models:
            importances = np.array(model.feature_importances_)
            if feature_importances is None:
                feature_importances = importances
            else:
                feature_importances += importances
        feature_importances = feature_importances/len(self.models)
        zeroes = np.zeros(feature_importances.shape[0])
        feature_importances = np.multiply(feature_importances, np.zeros(feature_importances.shape[0]), where = (feature_importances < 0))
        return feature_importances/ np.sum(feature_importances)
