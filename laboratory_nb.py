import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from processingData import ProcessingData

class NaiveBayesClassifier():
    def fit(self, X_train, y_train):
        self.classes = np.unique(y_train)

        self.priors = [len(y_train[y_train == c]) / len(y_train) for c in self.classes]

        self.means = [X_train[y_train == c].mean() for c in self.classes]
        self.stds = [X_train[y_train == c].std() for c in self.classes]

    def compute_likelihood(self, row, class_idx):
        likelihood = 1
        for feature in row.index:
            mean = self.means[class_idx][feature]
            std = self.stds[class_idx][feature]
            likelihood *= (1 / (np.sqrt(2 * np.pi) * std)) * np.exp((-(row[feature] - mean) ** 2) / (2 * std ** 2))

        return likelihood

    def predict(self, X):
        y_pred = []
        for _, row in X.iterrows():
            posteriors = []
            for i in range(len(self.classes)):
                likelihood = self.compute_likelihood(row, i)
                posteriors.append(likelihood * self.priors[i])

            y_pred.append(self.classes[np.argmax(posteriors)])

        return np.array(y_pred)

if __name__ == '__main__':
    data = pd.read_csv('vitamin_deficiency_disease_dataset_20260123.csv')

    process = ProcessingData(42)

    numeric_features = [
        'age',
        'bmi',
        'hemoglobin_g_dl',
        'serum_vitamin_d_ng_ml',
        'serum_vitamin_b12_pg_ml',
        'serum_folate_ng_ml',
        'vitamin_a_percent_rda',
        'vitamin_c_percent_rda',
        'vitamin_d_percent_rda',
        'folate_percent_rda',
        'vitamin_b12_percent_rda'
    ]

    X = process.select_features(data, numeric_features)
    y = data['disease_diagnosis']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    nb = NaiveBayesClassifier()
    nb.fit(X_train, y_train)
    predictions = nb.predict(X_test)
    accuracy = np.mean(predictions == y_test) * 100
    print('Accuracy:', accuracy)


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
    metrics = calculate_metrics(y_test, predictions)
    print("\n=== WYNIKI KOŃCOWE ===")
    overall_accuracy = np.mean(predictions == y_test)
    print(f"Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy * 100:.2f}%)")

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
    plot_confusion_matrix(y_test, predictions)
    plot_metrics_comparison(metrics)