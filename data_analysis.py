import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    #wczytanie danych
    df = pd.read_csv('vitamin_deficiency_disease_dataset_20260123.csv', encoding='utf-8')

    print("Base number of columns:", len(df.columns))

    columns_to_drop = [
        "smoking_status", "alcohol_consumption", "exercise_level",
        "diet_type", "sun_exposure", "income_level", "latitude_region",
        "vitamin", 'vitamin_a_percent_rda', 'vitamin_c_percent_rda',
        'vitamin_d_percent_rda', 'vitamin_e_percent_rda', 'vitamin_b12_percent_rda',
        'folate_percent_rda', 'calcium_percent_rda', 'iron_percent_rda',
        'symptoms_list', 'has_multiple_deficiencies', 'bmi',
        'age',
        'gender'
    ]

    #usuniecie niepotrzebnych kolumn
    df.drop(columns=columns_to_drop, errors='ignore', inplace=True)

    print("Number of columns after dropping irrelevant ones:", len(df.columns))

    lab_columns = [
        'hemoglobin_g_dl',
        'serum_vitamin_d_ng_ml',
        'serum_vitamin_b12_pg_ml',
        'serum_folate_ng_ml'
    ]

    symptom_columns = [
        'has_night_blindness','has_fatigue', 'has_bleeding_gums', 'has_bone_pain',
        'has_muscle_weakness', 'has_numbness_tingling', 'has_memory_problems',
        'has_pale_skin'
    ]

    #konwersja na liczby i wypisanie wartosci minimalnych i maksymalnych
    for col in lab_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        print(f"{col}: min - {df[col].min()}, max - {df[col].max()}")
    df[symptom_columns] = df[symptom_columns].astype(int)

    #wykresy dla objawów:
    #rysowanie wykresu dla tego jak często dany objaw pojawia się przy chorobach
    sns.set_theme(style="whitegrid")
    frequency_df = df.groupby('disease_diagnosis')[symptom_columns].mean().T
    frequency_df  *= 100
    frequency_df.plot(
        kind='bar',
        figsize=(14, 7),
        colormap='Set2',
        edgecolor='black'
    )

    plt.title("Symptom Frequency by Diagnosis", fontsize=16, pad=20)
    plt.ylabel("Percentage of Patients (%)", fontsize=12)
    plt.xlabel("Symptom", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Diagnosis", bbox_to_anchor=(1, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    #rysowanie wykresu dla ilości pacjentów z chorobami według ilości objawów
    plt.figure(figsize=(12, 6))
    sns.histplot(
        data=df,
        x='symptoms_count',
        hue='disease_diagnosis',
        multiple='dodge',
        discrete=True,
        shrink=0.8,
        palette='Set2'
    )

    plt.title("Symptom Distribution", fontsize=16, pad=20)
    plt.xlabel("Total number of symptoms per patient", fontsize=12)
    plt.ylabel("Number of patients", fontsize=12)
    plt.xticks(range(0, int(df['symptoms_count'].max()) + 1))
    sns.move_legend(plt.gca(), "upper left", bbox_to_anchor=(0.8, 1), title='Diagnosis')

    plt.show()

    #wykresy dla danych laboratoryjnych:
    #wyrysowanie macierzy korelacji pearsona

    cols_for_corr = lab_columns
    pearson_corr = df[cols_for_corr].corr(method='pearson')

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        pearson_corr,
        annot=True,
        cmap='coolwarm',
        vmin=-1, vmax=1,
        linewidths=0.5,
    )

    plt.title("Pearson Correlation Matrix", fontsize=16, pad=20)
    plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    #rysowanie wykresów gęstości występowania chorób dla konkretnych danych laboratoryjnych
    for col in lab_columns:
        plt.figure(figsize=(12, 6))

        sns.kdeplot(
            data=df,
            x=col,
            hue='disease_diagnosis',
            fill=True,
            common_norm=False,
            palette='tab10',
            alpha=0.1,
            linewidth=2
        )

        plt.title(f"Density Distribution: {col}", fontsize=16, pad=20)
        plt.xlabel(col.replace('_', ' ').title(), fontsize=12)
        plt.ylabel("Density", fontsize=12)
        sns.move_legend(plt.gca(), "upper left", bbox_to_anchor=(1, 1), title='Diagnosis')

        plt.tight_layout()
        plt.show()

    #wyrysowanie uśrednionych oraz znormalizowanych sylwetek dla każdej z diagnoz

    mean_profiles = df.groupby('disease_diagnosis')[lab_columns].mean()
    normalized_profiles = (mean_profiles - df[lab_columns].min()) / (df[lab_columns].max() - df[lab_columns].min())

    categories = lab_columns
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    plt.figure(figsize=(14, 10))
    ax = plt.subplot(111, polar=True)

    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for idx, (disease, row) in enumerate(normalized_profiles.iterrows()):
        values = row.tolist()
        values += values[:1]
        ax.plot(angles, values, color=colors[idx], linewidth=2, label=disease)
        ax.fill(angles, values, color=colors[idx], alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    plt.title("Average Normalized Profile per Diagnosis", fontsize=16, pad=30)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.show()

    #usuniecie kolumny z iloscia objawow
    df = df.drop('symptoms_count', axis=1)



if __name__ == "__main__":
    main()