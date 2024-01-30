import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)
    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)я
    """
    indices_order = np.argsort(feature_vector) #сортируем наши признаки и таргет и получаем соответствующие индексы
    feature_vector = feature_vector[indices_order]
    target_vector = target_vector[indices_order]

    unique_values, unique_indices = (np.unique(feature_vector, return_index=True))
    # is_unique = np.zeros(len(target_vector) + 1)
    # is_unique[unique_indices] = 1

    def impurity(prop_vector):
        #$H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.
        return (1 - prop_vector**2 - (1 - prop_vector)**2) 

    def gini_criterion(feature_vector, target_vector):
        total = np.arange(1, feature_vector.size)
        target_cumsum = np.cumsum(target_vector)[:-1] #после сортировки признака мы можем подсчитать количество единиц путем куммулятивной суммы таргета при
        #заданном пороге. Порогом будет индекс отсечения, поэтому нам достаточно построить массив куммулятивной суммы для дальнейшей работы.
        avg_count1 = target_cumsum / total #среднее число единиц для порога threshold = index
        avg_count0 = (np.sum(target_vector) - target_cumsum) / (total[::-1]) #среднее число нулей для порога threshold = index 
        mask = (np.array(feature_vector[1:]) - np.array(feature_vector[:-1])) != 0 #проверка того, что последующие элементы не совпадают с прошлым
        return (-(total / len(feature_vector)) * impurity(avg_count1) - (1 - total / len(feature_vector)) * impurity(avg_count0))[mask]

    ginis = np.array(gini_criterion(feature_vector, target_vector))
    thresholds = (unique_values[1:] + unique_values[0:-1]) / 2
    
    if ginis.shape[0] == 0:
        return (None, None, 1, None)
    
    index_max = np.nanargmax(ginis)
    return thresholds, ginis, thresholds[index_max], ginis[index_max]

class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
            else:
                raise ValueError

            if np.all(feature_vector == feature_vector[0]):
                continue
            
            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if (gini_best is None) or (gini > gini_best):
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"])
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"])
    
    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]
        feature_split = node['feature_split']
        feature_type = self._feature_types[feature_split]
        feature_value = x[feature_split]
        if feature_type == "categorical":
            if feature_value in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            return self._predict_node(x, node["right_child"])

        feature_crit_value = node["threshold"]
        if feature_value < feature_crit_value:
            return self._predict_node(x, node["left_child"])
        return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
    
    def get_params(self, deep = True):
       return {'feature_types': self._feature_types, 'max_depth': self._max_depth, 
               'min_samples_split':self._min_samples_split, 'min_samples_leaf': self._min_samples_leaf
               }