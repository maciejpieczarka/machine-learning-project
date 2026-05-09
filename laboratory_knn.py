import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from processingData import ProcessingData


data = pd.read_csv("vitamin_deficiency_disease_dataset_20260123.csv")
process = ProcessingData(42)

# wybranie tylko cech, ktore sa danymi laboratoryjnymi
lab_features = [
    'serum_vitamin_d_ng_ml',
    'serum_vitamin_b12_pg_ml',
    'serum_folate_ng_ml',
    'hemoglobin_g_dl',
    'vitamin_a_percent_rda',
    'vitamin_c_percent_rda',
    'calcium_percent_rda',
    'iron_percent_rda'
    ]

X = process.select_features(data,lab_features).values
y = data['disease_diagnosis'].values

# Usuwnaie outlierow
X, y = process.remove_outliers_iqr(X, y)

# Podzial na zbiory testowy i treningowy
X_train, X_test, y_train, y_test = process.train_test_split(
    X, y, test_size=0.2, stratify=True  # stratify=True zachowuje proporcje chorób
)

print(f"Train: {X_train.shape[0]} próbek, {X_train.shape[1]} cech")
print(f"Test:  {X_test.shape[0]} próbek")
print(f"Klasy w train: {np.unique(y_train, return_counts=True)}")
print(f"Klasy w test:  {np.unique(y_test, return_counts=True)}")

# Standaryzacja danych
X_train_scaled, X_test_scaled = process.standardize(X_train, X_test)

# Podzielenie train na train i validation dla grid search
X_train_sub, X_val, y_train_sub, y_val = process.train_test_split(
    X_train_scaled, y_train, test_size=0.2, stratify=True
)


def euclidean_distance(a, b):
    return np.sqrt(np.sum((b - a) ** 2))


class KNN:
    def __init__(self, k, metric='euclidean', weights='uniform'):
        self.k = k
        self.metric = metric
        self.weights = weights

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, new_points):
        predictions = [self.predict_class(point) for point in new_points]
        return np.array(predictions)

    def predict_class(self, new_point):
        # Obliczenie odległości
        if self.metric == 'euclidean':
            distances = np.sqrt(np.sum((self.X_train - new_point) ** 2, axis=1))
        elif self.metric == 'manhattan':
            distances = np.sum(np.abs(self.X_train - new_point), axis=1)
        else:
            distances = np.sqrt(np.sum((self.X_train - new_point) ** 2, axis=1))

        # Znajdź k najbliższych sąsiadów
        k_nearest_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_nearest_indices]
        k_nearest_distances = distances[k_nearest_indices]

        if self.weights == 'distance':
            # Głosowanie ważone odległością (bliżsi = większa waga)
            weights = 1.0 / (k_nearest_distances + 1e-8)
            unique_labels = np.unique(k_nearest_labels)
            weighted_votes = {}
            for label in unique_labels:
                mask = k_nearest_labels == label
                weighted_votes[label] = np.sum(weights[mask])
            return max(weighted_votes, key=weighted_votes.get)
        else:
            # Głosowanie zwykłe
            return Counter(k_nearest_labels).most_common(1)[0][0]


# Metoda do wyszukiwania najlepszych parametrów dla knn
def grid_search(X_train, y_train, X_val, y_val):
    # Szukamy najlepszych parametrów:
    # - k: liczba sąsiadów (testujemy 1, 3, 5, 7, 9, 11, 15, 21)
    # - metric: euclidean lub manhattan
    # - weights: uniform lub distance

    best_score = 0
    best_params = {}

    for k in [1, 3, 5, 7, 9, 11, 15, 21]:
        for metric in ['euclidean', 'manhattan']:
            for weights in ['uniform', 'distance']:
                knn = KNN(k=k, metric=metric, weights=weights)
                knn.fit(X_train, y_train)
                predictions = knn.predict(X_val)
                accuracy = np.mean(predictions == y_val)

                print(f"k={k:2d}, {metric:10s}, {weights:10s} -> accuracy: {accuracy:.4f}")

                if accuracy > best_score:
                    best_score = accuracy
                    best_params = {'k': k, 'metric': metric, 'weights': weights}

    print(f"\nNAJLEPSZE PARAMETRY: {best_params} (accuracy: {best_score:.4f})")
    return best_params

# Szukanie najlepszego parametru k za pomocą grid search
print("\n=== GRID SEARCH ===")
best_params = grid_search(X_train_sub, y_train_sub, X_val, y_val)

# Wywolanie modelu z najlepszymi parametrami
final_knn = KNN(k=best_params['k'], metric=best_params['metric'], weights=best_params['weights'])
final_knn.fit(X_train_scaled, y_train)

# Predykcja na zbiorze testowym
y_pred = final_knn.predict(X_test_scaled)


def calculate_metrics(y_true, y_pred):
    """
    Oblicza metryki dla każdej klasy

    Macierz pomyłek (Confusion Matrix):
                 Przewidziane
                Chory   Zdrowy
    Rzeczywiste:
    Chory        TP       FN
    Zdrowy       FP       TN

    Gdzie:
    TP (True Positive) - poprawnie wykryci chorzy
    TN (True Negative) - poprawnie wykryci zdrowi
    FP (False Positive) - fałszywy alarm
    FN (False Negative) - chorzy, których nie wykryliśmy
    """
    classes = np.unique(y_true)
    metrics = {}

    for cls in classes:
        tp = np.sum((y_true == cls) & (y_pred == cls))  # True Positives
        tn = np.sum((y_true != cls) & (y_pred != cls))  # True Negatives
        fp = np.sum((y_true != cls) & (y_pred == cls))  # False Positives
        fn = np.sum((y_true == cls) & (y_pred != cls))  # False Negatives

        # Accuracy = (TP + TN) / (TP + TN + FP + FN) - ogólna skuteczność
        accuracy = (tp + tn) / len(y_true)

        # Sensitivity/Recall = TP / (TP + FN) - wykrywalność choroby
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

        # Precision = TP / (TP + FP) - dokładność przewidywań
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        # Specificity = TN / (TN + FP) - wykrywalność zdrowych
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # F1-score = średnia harmoniczna precision i recall
        f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

        metrics[cls] = {
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'precision': precision,
            'specificity': specificity,
            'f1_score': f1
        }

    return metrics

# Obliczenie metryk
metrics = calculate_metrics(y_test, y_pred)

print("\n=== WYNIKI KOŃCOWE ===")
overall_accuracy = np.mean(y_pred == y_test)
print(f"Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")

print("\n=== METRYKI DLA KAŻDEJ KLASY ===")
for cls, m in metrics.items():
    print(f"\n{cls}:")
    print(f"  Accuracy:   {m['accuracy']:.4f}")
    print(f"  Precision:  {m['precision']:.4f}")
    print(f"  Recall:     {m['sensitivity']:.4f}")
    print(f"  F1-score:   {m['f1_score']:.4f}")
    print(f"  Specificity:{m['specificity']:.4f}")


def plot_confusion_matrix(y_true, y_pred):
    """Rysuje macierz pomyłek"""
    classes = np.unique(y_true)
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    for true, pred in zip(y_true, y_pred):
        cm[class_to_idx[true]][class_to_idx[pred]] += 1

    # Wizualizacja
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(len(classes)),
           yticks=np.arange(len(classes)),
           xticklabels=classes, yticklabels=classes,
           title='Macierz pomyłek (Confusion Matrix)',
           ylabel='Rzeczywista klasa',
           xlabel='Przewidziana klasa')

    # Wpisz wartości
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.tight_layout()
    plt.show()


def plot_metrics_comparison(metrics):
    """Rysuje porównanie metryk dla różnych klas"""
    classes = list(metrics.keys())
    precision = [metrics[c]['precision'] for c in classes]
    recall = [metrics[c]['sensitivity'] for c in classes]
    f1 = [metrics[c]['f1_score'] for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precision, width, label='Precision')
    ax.bar(x, recall, width, label='Recall')
    ax.bar(x + width, f1, width, label='F1-score')

    ax.set_xlabel('Klasa')
    ax.set_ylabel('Wynik')
    ax.set_title('Porównanie metryk dla poszczególnych klas')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.show()

# Wizualizacja
# 11. Wizualizacje
plot_confusion_matrix(y_test, y_pred)
plot_metrics_comparison(metrics)