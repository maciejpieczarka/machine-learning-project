import numpy as np

class NaiveBayes:
    def fit(self, X, y, flat_prior=False):
        self.classes = np.unique(y)
        # wyznaczenie prawdopodobieństwa wystąpienia każdej klasy
        if flat_prior:
            self.classes_probabilities = {c: 1/len(self.classes) for c in self.classes}
        else:
            self.classes_probabilities = {c: np.mean(y == c) for c in self.classes}
        self.feature_probabilities = {}
        # prawdopodobieństwa wystąpienia cechy pod warunkiem wystąpienia danej klasy, czyli P(feature|class)
        for c in self.classes:
            class_indices = np.where(y == c)[0]
            class_samples = X[class_indices]
            for feature in range(X.shape[1]):
                feature_values = class_samples[:, feature]
                self.feature_probabilities[(c, feature)] = np.mean(feature_values)
    
    def predict(self, X):
        predictions = []
        for x in X:
            best_probability = 0
            best_class = None
            # znalezienie klasy, która maksymalizuje iloczyn prawdopodobieństwa wystąpienia tej klasy i prawdopodobieństwa wystąpienia cech pod warunkiem tej klasy
            for c in self.classes:
                class_prbability = self.classes_probabilities[c]
                for feature in range(X.shape[1]):
                    feature_value = x[feature]
                    p = self.feature_probabilities.get((c, feature), 0.5)
                    feature_probability = max(0.01, min(0.99, p))
                    # jeśli cecha występuje to bierzemy prawdopobieństwo tak jak je zapisano (prawdopodobieństwo wystąpienia tej cechy)
                    if feature_value == 1:
                        class_prbability *= feature_probability
                    # w przeciwnym wypadku bierzemy dopełnienie tego prawdopodobieństwa (prawdopodobieństwo niewystąpienia tej cechy)
                    else:
                        class_prbability *= (1 - feature_probability)
                if class_prbability > best_probability:
                    best_probability = class_prbability
                    best_class = c
            predictions.append(best_class)
        return np.array(predictions)