import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from processingData import ProcessingData

data = pd.read_csv("vitamin_deficiency_disease_dataset_20260123.csv")
process = ProcessingData()

# wybranie tylko cech, ktore sa danymi laboratoryjnymi
lab_features = ['age', 'bmi', 'vitamin_a_percent_rda', 'vitamin_c_percent_rda',
                'vitamin_d_percent_rda', 'vitamin_e_percent_rda',
                'vitamin_b12_percent_rda', 'folate_percent_rda',
                'calcium_percent_rda', 'iron_percent_rda', 'hemoglobin_g_dl',
                'serum_vitamin_d_ng_ml', 'serum_vitamin_b12_pg_ml',
                'serum_folate_ng_ml']

X = process.select_features(data,lab_features)
y = data['disease_diagnosis'].values

X_train, X_test, y_train, y_test = process.train_test_split(
    X, y, test_size=0.2, stratify=True  # stratify=True zachowuje proporcje chorób
)

print(f"Train: {X_train.shape[0]} próbek")
print(f"Test:  {X_test.shape[0]} próbek")

X_train_scaled, X_test_scaled = process.standardize(X_train, X_test)

#
# plt.scatter(X_train[y_train == 'B', 0], X_train[y_train == 'B', 1], color='tab:green', label='Benign')
# plt.scatter(X_train[y_train == 'M', 0], X_train[y_train == 'M', 1], color='tab:red', label='Malignant')
# # plt.show()
#
# def euclidean_distance(a, b):
#     return np.sqrt(np.sum((b - a) ** 2))
#
#
# class KNN:
#     def __init__(self, k):
#         self.k = k
#
#     def fit(self, X, y):
#         self.X_train = X
#         self.y_train = y
#
#     def predict(self, new_points):
#         predictions = [self.predict_class(new_point) for new_point in new_points]
#         return np.array(predictions)
#
#     def predict_class(self, new_point):
#         distances = [euclidean_distance(point, new_point) for point in self.X_train]
#
#         k_nearest_indices = np.argsort(distances)[:self.k]
#         k_nearest_labels = [self.y_train[i] for i in k_nearest_indices]
#
#         most_common = Counter(k_nearest_labels).most_common(1)[0][0]
#
#         return most_common
#
# knn = KNN(5)
# knn.fit(X_train_scaled, y_train)
# predictions = knn.predict(X_test_scaled)
# accuracy = np.mean(predictions == y_test) * 100
# print(f"Accuracy: {accuracy:.2f}%")
#
# print(np.unique(predictions))
# plt.scatter(X_test_scaled[predictions == 'Anemia', 0], X_test_scaled[predictions == 'Anemia', 1], color='tab:green', label='Predicted Benign', marker='x')
# plt.scatter(X_test_scaled[predictions == 'Healthy', 0], X_test_scaled[predictions == 'Healthy', 1], color='tab:red', label='Predicted Malignant', marker='x')
# plt.xlabel('Radius Mean')
# plt.ylabel('Texture Mean')
# plt.legend()
# plt.show()
