from data_analysis import main as data_analysis_main
from processingData import ProcessingData
from bernoulli_naive_bayes import NaiveBayes as BernoulliNaiveBayes
from decision_tree import DecisionTree, TreeCrossValidation
from random_forest import RandomForest
import pandas as pd

def main():
    # Uruchomienie analizy danych
    data_analysis_main()

    # Przetwarzanie danych
    process = ProcessingData(random_state=42)
    df = pd.read_csv('vitamin_deficiency_disease_dataset_20260123.csv')
    
    symptoms_features = ['has_night_blindness', 'has_fatigue', 'has_bleeding_gums', 'has_bone_pain',
                         'has_muscle_weakness', 'has_numbness_tingling', 'has_memory_problems', 'has_pale_skin']
    decision_tree_features = ['age', 'bmi', 'hemoglobin_g_dl', 'serum_vitamin_d_ng_ml', 'serum_vitamin_b12_pg_ml',
        'serum_folate_ng_ml', 'vitamin_a_percent_rda', 'vitamin_c_percent_rda',
        'vitamin_d_percent_rda', 'vitamin_e_percent_rda', 'vitamin_b12_percent_rda',
        'folate_percent_rda', 'calcium_percent_rda', 'iron_percent_rda'] + symptoms_features
    X_symptoms = process.select_features(df, symptoms_features)
    y = df['disease_diagnosis'].values
    X_train_symptoms, X_test_symptoms, y_train_symptoms, y_test_symptoms = process.train_test_split(
        X_symptoms, y, test_size=0.2, stratify=True  # stratify=True zachowuje proporcje chorób
    )
    
    # Naive bayes dla samych objawów
    nb = BernoulliNaiveBayes()
    nb.fit(X_train_symptoms, y_train_symptoms)
    nb_predictions = nb.predict(X_test_symptoms)
    nb_scores = process.result_analysis(y_test_symptoms, nb_predictions)
    print("\nNaive Bayes oparty tylko na objawach:")
    for metric in nb_scores:
        print(f"Metric: {metric}, accuracy: {nb_scores[metric]['accuracy']:.4f}, sensitivity: {nb_scores[metric]['sensitivity']:.4f}, precision: {nb_scores[metric]['precision']:.4f}, specificity: {nb_scores[metric]['specificity']:.4f}, f1_score: {nb_scores[metric]['f1_score']:.4f}")
    
    # Naive bayes dla samych objawów z płaskim priorem (bez uwzględniania częstości chorób)
    nb.fit(X_train_symptoms, y_train_symptoms, flat_prior=True)
    nb_flat_prior_predictions = nb.predict(X_test_symptoms)
    nb_flat_prior_scores = process.result_analysis(y_test_symptoms, nb_flat_prior_predictions)
    print("\nNaive Bayes oparty tylko na objawach bez uwzględnienia częstości chorób:")
    for metric in nb_flat_prior_scores:
        print(f"Metric: {metric}, accuracy: {nb_flat_prior_scores[metric]['accuracy']:.4f}, sensitivity: {nb_flat_prior_scores[metric]['sensitivity']:.4f}, precision: {nb_flat_prior_scores[metric]['precision']:.4f}, specificity: {nb_flat_prior_scores[metric]['specificity']:.4f}, f1_score: {nb_flat_prior_scores[metric]['f1_score']:.4f}")
    
    
    X = process.select_features(df, decision_tree_features)
    X_train, X_test, y_train, y_test = process.train_test_split(
        X, y, test_size=0.2, stratify=True  # stratify=True zachowuje proporcje chorób
    )
    # Drzewo decyzyjne dla wszystkich cech z walidacją krzyżową
    print("\nTrening drzewa decyzyjnego z walidacją krzyżową...")
    cv = TreeCrossValidation()
    tree_best_params = cv.perform(X, y)
    tree = DecisionTree()
    tree.fit(X_train, y_train, tree_best_params[0], tree_best_params[1])
    tree_predictions = tree.predict(X_test)
    tree_scores = process.result_analysis(y_test, tree_predictions)
    print("\nDrzewo decyzyjne z walidacją krzyżową:")
    for metric in tree_scores:
        print(f"Metric: {metric}, accuracy: {tree_scores[metric]['accuracy']:.4f}, sensitivity: {tree_scores[metric]['sensitivity']:.4f}, precision: {tree_scores[metric]['precision']:.4f}, specificity: {tree_scores[metric]['specificity']:.4f}, f1_score: {tree_scores[metric]['f1_score']:.4f}")
    
    # Las losowy dla wszystkich cech
    print("\nTrening lasu losowego...")
    rf = RandomForest()
    rf.fit(X_train, y_train, n_trees=100)
    rf_predictions = rf.predict(X_test)
    rf_scores = process.result_analysis(y_test, rf_predictions)
    print("\nRandom Forest:")
    for metric in rf_scores:
        print(f"Metric: {metric}, accuracy: {rf_scores[metric]['accuracy']:.4f}, sensitivity: {rf_scores[metric]['sensitivity']:.4f}, precision: {rf_scores[metric]['precision']:.4f}, specificity: {rf_scores[metric]['specificity']:.4f}, f1_score: {rf_scores[metric]['f1_score']:.4f}")
    
    # Las losowy z oversamplingiem
    rf.fit(X_train, y_train, n_trees=100, oversampling=True)
    rf_oversampling_predictions = rf.predict(X_test)
    rf_oversampling_scores = process.result_analysis(y_test, rf_oversampling_predictions)
    print("\nRandom Forest z oversamplingiem:")
    for metric in rf_oversampling_scores:
        print(f"Metric: {metric}, accuracy: {rf_oversampling_scores[metric]['accuracy']:.4f}, sensitivity: {rf_oversampling_scores[metric]['sensitivity']:.4f}, precision: {rf_oversampling_scores[metric]['precision']:.4f}, specificity: {rf_oversampling_scores[metric]['specificity']:.4f}, f1_score: {rf_oversampling_scores[metric]['f1_score']:.4f}")
    
if __name__ == "__main__":
    main()