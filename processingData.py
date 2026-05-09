import numpy as np
# Klasa do przetwarzania danych przed trenowaniem modelu
class ProcessingData:
    def __init__(self, random_state=None):
        self.scaler_mean = None
        self.scaler_std = None
        self.selected_features = None
        self.random_state = random_state

    # Metoda do ustawiania ziarna losowosci dla powtarzalnosci wynikow
    def set_random_seed(self, seed):
        self.random_state = seed
        np.random.seed(seed)

    # Metoda obslugujaca wybieranie tylko numerycznych cech do modelu
    def select_features(self, data, feature_list):
        self.selected_features = feature_list
        return data[feature_list]

    def train_test_split(self, X, y, test_size=0.2, stratify=False):
        """
        Dzieli dane na zbiór treningowy i testowy.

        Parametry:
        - X: macierz cech
        - y: wektor etykiet
        - test_size: proporcja zbioru testowego (domyślnie 0.2 = 20%)
        - stratify: czy zachować proporcje klas (domyślnie False)

        Zwraca:
        - X_train, X_test, y_train, y_test
        """
        n_samples = len(X)
        n_test = int(n_samples * test_size)

        # Ustawienie ziarna losowości
        if self.random_state is not None:
            np.random.seed(self.random_state)

        if stratify:
            # Stratyfikowany podział - zachowuje proporcje klas
            return self._stratified_split(X, y, test_size)
        else:
            # Losowy podział
            indices = np.random.permutation(n_samples)
            test_indices = indices[:n_test]
            train_indices = indices[n_test:]

            return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

    def _stratified_split(self, X, y, test_size):
        unique_classes = np.unique(y)
        X_train_list, X_test_list = [], []
        y_train_list, y_test_list = [], []

        # Ustaw ziarno RAZ przed pętlą
        if self.random_state is not None:
            np.random.seed(self.random_state)

        for cls in unique_classes:
            cls_indices = np.where(y == cls)[0]
            cls_X = X[cls_indices]
            cls_y = y[cls_indices]

            n_cls_test = max(1, int(len(cls_indices) * test_size))
            shuffled = np.random.permutation(len(cls_indices))

            test_idx = shuffled[:n_cls_test]
            train_idx = shuffled[n_cls_test:]

            X_test_list.append(cls_X[test_idx])
            X_train_list.append(cls_X[train_idx])
            y_test_list.append(cls_y[test_idx])
            y_train_list.append(cls_y[train_idx])

        # Połączenie wyników
        X_train = np.vstack(X_train_list) if X_train_list else np.array([])
        X_test = np.vstack(X_test_list) if X_test_list else np.array([])
        y_train = np.hstack(y_train_list) if y_train_list else np.array([])
        y_test = np.hstack(y_test_list) if y_test_list else np.array([])

        # Ponowne przetasowanie (bo klasy są teraz posklejane)
        train_indices = np.random.permutation(len(X_train))
        test_indices = np.random.permutation(len(X_test))

        return X_train[train_indices], X_test[test_indices], y_train[train_indices], y_test[test_indices]

    # Normalizacja min-max, aby wartosci byly w zakresie [0,1]
    def normalize_minmax(self, X_train, X_test):
        min_vals = np.min(X_train, axis=0)
        max_vals = np.max(X_train, axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1

        X_train_norm = (X_train - min_vals) / range_vals
        X_test_norm = (X_test - min_vals) / range_vals

        return X_train_norm, X_test_norm

    def remove_outliers_iqr(self, X, y):
        # Usuwanie znaczaco rozniacych sie punktow danych (outlierow) stosujac rozstep miedzykwartylowy
        Q1 = np.percentile(X, 25, axis=0) # kwartyl 1
        Q3 = np.percentile(X, 75, axis=0) # kwartyl 3
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Sprawdzamy które wiersze nie mają outlierów
        mask = np.all((X >= lower_bound) & (X <= upper_bound), axis=1)

        return X[mask], y[mask]

    # Metoda do standaryzacji danych
    def standardize(self, X_train, X_test):

        # Obliczamy średnie i odchylenia na danych treningowych
        self.scaler_mean = np.mean(X_train, axis=0)
        self.scaler_std = np.std(X_train, axis=0)

        # Zabezpieczenie przed dzieleniem przez zero
        self.scaler_std[self.scaler_std == 0] = 1

        # Standaryzujemy dane treningowe i testowe
        X_train_scaled = (X_train - self.scaler_mean) / self.scaler_std
        X_test_scaled = (X_test - self.scaler_mean) / self.scaler_std

        return X_train_scaled, X_test_scaled
    
    def result_analysis(self, y_true, y_pred):
        # Analiza wyników - obliczanie dokładności, macierzy pomyłek itp.
        class_metrics = {}
        for c in np.unique(y_true):
            tp = np.sum((y_true == c) & (y_pred == c))
            tn = np.sum((y_true != c) & (y_pred != c))
            fp = np.sum((y_true != c) & (y_pred == c))
            fn = np.sum((y_true == c) & (y_pred != c))
            accuracy = (tp + tn) / len(y_true)
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
            class_metrics[c] = {
                'accuracy': accuracy,
                'sensitivity': sensitivity,
                'precision': precision,
                'specificity': specificity,
                'f1_score': f1_score
            }
        return class_metrics