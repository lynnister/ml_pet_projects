from dataclasses import dataclass
from enum import auto
from enum import Enum
from typing import Dict
from typing import Type

import numpy as np


@dataclass
class LearningRate:
    lambda_: float = 1e-3
    s0: float = 1
    p: float = 0.5

    iteration: int = 0

    def __call__(self):
        """
        Calculate learning rate according to lambda (s0/(s0 + t))^p formula
        """
        self.iteration += 1
        return self.lambda_ * (self.s0 / (self.s0 + self.iteration)) ** self.p


class LossFunction(Enum):
    MSE = auto()
    MAE = auto()
    LogCosh = auto()
    Huber = auto()


class BaseDescent:
    """
    A base class and templates for all functions
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: str = 'MSE', delta: float = 0):
        """
        :param dimension: feature space dimension
        :param lambda_: learning rate parameter
        :param loss_function: optimized loss function
        """
        loss_function_mapping = {
            'MSE': LossFunction.MSE,
            'LogCosh': LossFunction.LogCosh,
            'MAE': LossFunction.MAE,
            'Huber': LossFunction.Huber,
        }
        self.delta = delta
        self.w: np.ndarray = np.random.rand(dimension)
        self.lr: LearningRate = LearningRate(lambda_=lambda_)
        self.loss_function: LossFunction = loss_function_mapping[loss_function]

    def step(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.update_weights(self.calc_gradient(x, y))

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Template for update_weights function
        Update weights with respect to gradient
        :param gradient: gradient
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        pass

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Template for calc_gradient function
        Calculate gradient of loss function with respect to weights
        :param x: features array
        :param y: targets array
        :return: gradient: np.ndarray
        """
        pass

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate loss for x and y with our weights
        :param x: features array
        :param y: targets array
        :return: loss: float
        """
        l = y.shape[0] #размер выборки
        return ((y - x @ self.w).T) @ (y - x @ self.w) / l

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate predictions for x
        :param x: features array
        :return: prediction: np.ndarray
        """
        #возвращаем <w,x> - наш предикт
        return x @ self.w


class VanillaGradientDescent(BaseDescent):
    """
    Full gradient descent class
    """

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        #считаем новые веса по формуле шага
        new_w = self.w - self.lr() * gradient 
        weight_difference = new_w - self.w
        self.w = new_w #обновляем веса
        return weight_difference #возвращаем разность

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        l = y.shape[0]
        if self.loss_function is LossFunction.MSE:
            return 2 * x.T @ (x @ self.w - y) / l
        elif self.loss_function is LossFunction.LogCosh:
            return x.T @ np.tanh(x @ self.w - y) / l
        elif self.loss_function is LossFunction.MAE:
            return -x.T @ np.sign(y - x @ self.w) / l
        elif self.loss_function is LossFunction.Huber:
            if np.linalg.norm(y - x @ self.w) <= self.delta:
                return 2 * x.T @ (x @ self.w - y) / l
            else:
                return -x.T @ np.sign(y - x @ self.w) / l


class StochasticDescent(VanillaGradientDescent):
    """
    Stochastic gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, batch_size: int = 50,
                 loss_function: str = 'MSE', delta: float = 0):
        """
        :param batch_size: batch size (int)
        """
        super().__init__(dimension, lambda_, loss_function)
        self.batch_size = batch_size

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        #выбираем индексы для нашей корзины
        batch_index = np.random.randint(low = 0, high = y.shape[0], size = self.batch_size) 
        x_batch = x[batch_index,:]
        y_batch = y[batch_index]
        l = self.batch_size
        if self.loss_function is LossFunction.MSE:
            return 2 * x_batch.T @ (x_batch @ self.w - y_batch) / l
        elif self.loss_function is LossFunction.LogCosh:
            return x_batch.T @ np.tanh(x_batch @ self.w - y_batch) / l
        elif self.loss_function is LossFunction.MAE:
            return -x_batch.T @ np.sign(y_batch - x_batch @ self.w) / l
        elif self.loss_function is LossFunction.Huber:
            if np.linalg.norm(y_batch- x_batch @ self.w) <= self.delta:
                return 2 * x_batch.T @ (x_batch @ self.w - y_batch) / l
            else:
                return -x_batch.T @ np.sign(y_batch - x_batch @ self.w) / l

            
class MomentumDescent(VanillaGradientDescent):
    """
    Momentum gradient descent class
    """
    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: str = 'MSE', delta: float = 0):
        super().__init__(dimension, lambda_, loss_function)
        self.alpha: float = 0.9

        self.h: np.ndarray = np.zeros(dimension)

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        #аналогично все записываем по формулам из условия
        self.h = self.h * self.alpha + self.lr() * gradient
        new_w = self.w - self.h
        weight_difference = new_w - self.w
        self.w = new_w
        return weight_difference

class Adam(VanillaGradientDescent):
    """
    Adaptive Moment Estimation gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: str = 'MSE', delta: float = 0):
        super().__init__(dimension, lambda_, loss_function)
        self.eps: float = 1e-8

        self.m: np.ndarray = np.zeros(dimension)
        self.v: np.ndarray = np.zeros(dimension)

        self.beta_1: float = 0.9
        self.beta_2: float = 0.999

        self.iteration: int = 0

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        self.iteration += 1
        #Аналогично по формулам из задания:
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * gradient
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * (gradient) ** 2
        m_cap = self.m / (1 - self.beta_1 ** self.iteration)
        v_cap = self.v / (1 - self.beta_2 ** self.iteration)
        new_w = self.w - m_cap * (self.lr() / (self.eps + (v_cap) ** 0.5))
        weight_difference = new_w - self.w
        self.w = new_w
        return weight_difference

class AMSGrad(VanillaGradientDescent):
    #Моя реализация AMSGrad для бонуса
    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: str = 'MSE'):
        super().__init__(dimension, lambda_, loss_function)
        self.eps: float = 1e-8

        self.m: np.ndarray = np.zeros(dimension)
        self.v: np.ndarray = np.zeros(dimension)
        self.v_cap: np.ndarray = np.zeros(dimension)

        self.beta_1: float = 0.9
        self.beta_2: float = 0.999

        self.iteration: int = 0

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        self.iteration += 1
        #Аналогично по формулам из задания:
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * gradient
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * (gradient) ** 2
        self.v_cap = max(self.v_cap.all(), self.v.all())
        new_w = self.w - self.m * (self.lr() / (self.eps + (self.v_cap) ** 0.5))
        weight_difference = new_w - self.w
        self.w = new_w
        return weight_difference





class BaseDescentReg(BaseDescent):
    """
    A base class with regularization
    """
    def __init__(self, *args, mu: float = 0, **kwargs):
        """
        :param mu: regularization coefficient (float)
        """
        super().__init__(*args, **kwargs)

        self.mu = mu

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculate gradient of loss function and L2 regularization with respect to weights
        """
        l2_gradient: np.ndarray = self.w #d<w,w^T> = <w^T, dw> => grad = w, 2 перед mu сократится
        l2_gradient[-1] = 0 #Чтобы не было регуляризации интерсепта
        return super().calc_gradient(x, y) + l2_gradient * self.mu #gradQ_new = gradQ + mu*w


class VanillaGradientDescentReg(BaseDescentReg, VanillaGradientDescent):
    """
    Full gradient descent with regularization class
    """


class StochasticDescentReg(BaseDescentReg, StochasticDescent):
    """
    Stochastic gradient descent with regularization class
    """


class MomentumDescentReg(BaseDescentReg, MomentumDescent):
    """
    Momentum gradient descent with regularization class
    """


class AdamReg(BaseDescentReg, Adam):
    """
    Adaptive gradient algorithm with regularization class
    """

class AMSGradReg(BaseDescentReg, AMSGrad):
    """
    Adaptive gradient algorithm with regularization class
    """


def get_descent(descent_config: dict) -> BaseDescent:
    descent_name = descent_config.get('descent_name', 'full')
    regularized = descent_config.get('regularized', False)
    
    descent_mapping: Dict[str, Type[BaseDescent]] = {
        'full': VanillaGradientDescent if not regularized else VanillaGradientDescentReg,
        'stochastic': StochasticDescent if not regularized else StochasticDescentReg,
        'momentum': MomentumDescent if not regularized else MomentumDescentReg,
        'adam': Adam if not regularized else AdamReg,
        'AMSGrad': AMSGrad if not regularized else AMSGradReg
    }

    if descent_name not in descent_mapping:
        raise ValueError(f'Incorrect descent name, use one of these: {descent_mapping.keys()}')

    descent_class = descent_mapping[descent_name]

    return descent_class(**descent_config.get('kwargs', {}))
