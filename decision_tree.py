import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
class DecisionTree:
    def fit(self, X, y, max_depth, min_elements):
        self.max_depth = max_depth
        self.min_elements = min_elements
        self.tree = self.build_tree(X, y)
    
    def gini_score(self, y):
        classes = np.unique(y)
        gini = 1.0
        for cls in classes:
            proportion = np.sum(y == cls) / len(y)
            gini -= proportion ** 2
        return gini
    
    def best_split(self, X, y):
        best_gini = float('inf')
        best_feature = None
        best_threshold = None
        
        # dla każdej cechy i każdego unikalnego progu dzielimy dane na dwie części i obliczamy gini dla tego podziału
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = np.where(X[:, feature] <= threshold)[0]
                right_indices = np.where(X[:, feature] > threshold)[0]
                
                if len(left_indices) <= self.min_elements or len(right_indices) <= self.min_elements:
                    continue
                
                gini_left = self.gini_score(y[left_indices])
                gini_right = self.gini_score(y[right_indices])
                gini_split = (len(left_indices) * gini_left + len(right_indices) * gini_right) / len(y)
                
                if gini_split < best_gini:
                    best_gini = gini_split
                    best_feature = feature
                    best_threshold = threshold
        
        # zwracamy najlepszą cechę i próg, które minimalizują gini            
        return best_feature, best_threshold
        
    def build_tree(self, X, y, current_depth=0):
        # jeżeli zbiór zawiera obiekty tylko jednej klasy to jest liściem tej klasy
        if len(np.unique(y)) == 1:
            return Node(value=np.unique(y)[0])
        
        # jeżeli osiągnięto maksymalną głębokość wybierz dominującą klasę i zwróć liść tej klasy
        if current_depth == self.max_depth:
            parameter, count = np.unique(y, return_counts=True)
            idx_max = np.argmax(count)
            return Node(value=parameter[idx_max])
        
        feature, threshold = self.best_split(X, y)
        #jeśli nie znaleziono podziału spełniającego kryteria (min_elements) wybierz dominującą klasę i zwróć liść tej klasy
        if feature is None:
            parameter, count = np.unique(y, return_counts=True)
            idx_max = np.argmax(count)
            return Node(value=parameter[idx_max])
        
        left_indices = np.where(X[:, feature] <= threshold)[0]
        right_indices = np.where(X[:, feature] > threshold)[0]
        #zapamiętaj cechę i próg podziału, rekurencyjnie stwórz odgałęzienia
        return Node(feature=feature, threshold=threshold, left=self.build_tree(X[left_indices], y[left_indices], current_depth=current_depth+1), right=self.build_tree(X[right_indices], y[right_indices], current_depth=current_depth+1))
        
    #funkcja "przechodząca" po drzewie dla pojedynczego obiektu, zwracająca wartość liścia, do którego dotarliśmy
    def traverse_tree(self, X):
        current_node = self.tree
        while current_node.value is None:
            if X[current_node.feature] <= current_node.threshold:
                current_node = current_node.left
            else:
                current_node = current_node.right
        return current_node.value
    
    def predict(self, X):
        return np.array([self.traverse_tree(x) for x in X])
    
class TreeCrossValidation:
    def perform(self, X, y):
        parts = []
        # podział danych na 5 części, każda część będzie raz użyta jako zbiór testowy, a pozostałe 4 części będą użyte jako zbiór treningowy
        for i in range(5):
            bottom_idx = int (len(X) * i / 5)
            top_idx = int (len(X) * (i + 1) / 5)
            parts.append((X[bottom_idx:top_idx], y[bottom_idx:top_idx]))
        params_accuracy = []
        # szukanie najlepszych parametrów drzewa (max_depth i min_elements) poprzez sprawdzenie różnych kombinacji tych parametrów i obliczenie średniej dokładności dla każdej kombinacji na 5 iteracjach walidacji krzyżowej
        for max_depth in range(3, X.shape[1]):
            for min_elements in range(int(X.shape[0] * 0.001), int(X.shape[0] * 0.01), int((int(X.shape[0] * 0.01) - int(X.shape[0] * 0.001)) / 5)):
                accuracies = []
                for i in range(len(parts)):
                    tree = DecisionTree()
                    test_x, test_y = parts[i]
                    train_x_parts = [parts[j][0] for j in range(len(parts)) if j != i]
                    training_x = np.vstack(train_x_parts)
                    train_y_parts = [parts[j][1] for j in range(len(parts)) if j != i]
                    training_y = np.hstack(train_y_parts)
                    tree.fit(training_x, training_y, max_depth, min_elements)
                    predictions = tree.predict(test_x)
                    accuracy = np.mean(predictions == test_y)
                    print(f"max_depth: {max_depth}; min_elements: {min_elements}; iteration: {i+1}; accuraccy: {accuracy:.4f}")
                    accuracies.append(accuracy)
                params_accuracy.append((max_depth, min_elements, np.average(accuracies)))
                print(f"Average accuracy for max_depth: {max_depth}; min_elements: {min_elements}; is {np.average(accuracies):.4f}")
        best_params = max(params_accuracy, key=lambda item: item[2])
        print(f"Najlepsze znalezione parametry dla drzewa: max_depth: {best_params[0]}; min_elements: {best_params[1]}; with average accuracy {best_params[2]} ")
        return best_params