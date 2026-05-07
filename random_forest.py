import numpy as np
import math

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class RandomForestDecisionTree:
    def fit(self, X, y):
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
        
        # wybór losowej podprzestrzeni cech (feature bagging) - wybieramy losowo sqrt(n_features) cech do rozważenia przy każdym podziale
        ammount_of_features = max(2, int(math.sqrt(X.shape[1])))
        features_indices = np.random.choice(X.shape[1], size=ammount_of_features, replace=False)
        # dla każdej cechy i każdego unikalnego progu dzielimy dane na dwie części i obliczamy gini dla tego podziału
        for feature in features_indices:
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = np.where(X[:, feature] <= threshold)[0]
                right_indices = np.where(X[:, feature] > threshold)[0]
                
                if len(left_indices) == 0 or len(right_indices) == 0:
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
        
    def build_tree(self, X, y):
        # jeżeli zbiór zawiera obiekty tylko jednej klasy to jest liściem tej klasy
        if len(np.unique(y)) == 1:
            return Node(value=np.unique(y)[0])
        
        feature, threshold = self.best_split(X, y)
        #jeśli nie znaleziono podziału spełniającego kryteria (min_elements) wybierz dominującą klasę i zwróć liść tej klasy
        if feature is None:
            parameter, count = np.unique(y, return_counts=True)
            idx_max = np.argmax(count)
            return Node(value=parameter[idx_max])
        
        left_indices = np.where(X[:, feature] <= threshold)[0]
        right_indices = np.where(X[:, feature] > threshold)[0]
        #zapamiętaj cechę i próg podziału, rekurencyjnie stwórz odgałęzienia
        return Node(feature=feature, threshold=threshold, left=self.build_tree(X[left_indices], y[left_indices]), right=self.build_tree(X[right_indices], y[right_indices]))
        
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
    
class RandomForest:
    def fit(self, X, y, n_trees, oversampling=False):
        self.n_trees = n_trees
        self.trees = []
        if oversampling:
            classes, counts = np.unique(y, return_counts=True)
            max_class_size = np.max(counts)
            
            for i in range(n_trees):
                print(f"Training tree {i+1}/{n_trees} with oversampling...")
                X_sample = []
                y_sample = []
                
                for cls in classes:
                    cls_indices = np.where(y == cls)[0]
                    
                    sampled_indices = np.random.choice(cls_indices, size=max_class_size, replace=True)
                    
                    X_sample.append(X[sampled_indices])
                    y_sample.append(y[sampled_indices])
                
                X_sample = np.vstack(X_sample)
                y_sample = np.hstack(y_sample)
                
                shuffle_idx = np.random.permutation(len(X_sample))
                X_sample = X_sample[shuffle_idx]
                y_sample = y_sample[shuffle_idx]
                
                tree = RandomForestDecisionTree()
                tree.fit(X_sample, y_sample)
                self.trees.append(tree)
        else:
            for i in range(n_trees):
                print(f"Training tree {i+1}/{n_trees}...")
                indices = np.random.choice(len(X), size=len(X), replace=True)
                X_sample = X[indices]
                y_sample = y[indices]
                tree = RandomForestDecisionTree()
                tree.fit(X_sample, y_sample)
                self.trees.append(tree)
    
    def predict(self, X):
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_predictions = np.swapaxes(tree_predictions, 0, 1)
        
        final_predictions = []
        for vote in tree_predictions:
            diagnosis, votes = np.unique(vote, return_counts=True)
            most_common = diagnosis[np.argmax(votes)]
            final_predictions.append(most_common)
            
        return np.array(final_predictions)
 