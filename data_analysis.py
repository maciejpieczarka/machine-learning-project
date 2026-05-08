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
        'symptoms_list','has_multiple_deficiencies'
        #  'vitamin_a_percent_rda', 'vitamin_c_percent_rda',
        # 'vitamin_d_percent_rda', 'vitamin_e_percent_rda', 'vitamin_b12_percent_rda',
        # 'folate_percent_rda', 'calcium_percent_rda', 'iron_percent_rda',
        #  , 'bmi',
        # 'age',
        # 'gender'
    ]

    #usuniecie niepotrzebnych kolumn
    df.drop(columns=columns_to_drop, errors='ignore', inplace=True)

    #zmiana płci na wartości numeryczne
    def change_value(value):
        if value == "Male":
            return 1
        return 0
    df["gender"] = df["gender"].apply(change_value)
    df = df.rename(columns={"gender": "is_male"})

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

    intake_columns = [
        'vitamin_a_percent_rda', 'vitamin_c_percent_rda',
        'vitamin_d_percent_rda', 'vitamin_e_percent_rda', 'vitamin_b12_percent_rda',
        'folate_percent_rda', 'calcium_percent_rda', 'iron_percent_rda'
    ]

    #konwersja na liczby i wypisanie wartosci minimalnych i maksymalnych
    for col in lab_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        print(f"{col}: min - {df[col].min()}, max - {df[col].max()}")
    df[symptom_columns] = df[symptom_columns].astype(int)

    #macierz korelacji pearsona

    pearson_df = df.loc[:, df.columns != "disease_diagnosis"]
    pearson_corr = pearson_df.corr(method='pearson')

    plt.figure(figsize=(20, 20))
    sns.heatmap(
        pearson_corr,
        annot=True,
        fmt=".2f",
        vmin=-1, vmax=1,
        linewidths=0.5,

    )
    plt.tick_params(axis='both', labelsize=13)
    plt.title("Pearson Correlation Matrix", fontsize=16, pad=20)
    plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    plots_per_figure = 3
    #wykresy gestosci dla danych laboratoryjnych oraz dla przyjmowanych dawek witamin i minerałów
    columns_to_plot = lab_columns + intake_columns
    for i in range(0, len(columns_to_plot), plots_per_figure):
        column_chunk = columns_to_plot[i:i + plots_per_figure]
        fig, axes = plt.subplots(1, plots_per_figure, figsize=(18, 5))

        for j, col in enumerate(column_chunk):
            ax = axes[j]

            sns.kdeplot(
                data=df,
                x=col,
                hue='disease_diagnosis',
                fill=True,
                common_norm=False,
                palette='tab10',
                alpha=0.1,
                linewidth=2,
                ax=ax
            )

            ax.set_title(f"Density: {col}", fontsize=14, pad=10)
            ax.set_xlabel(col.replace('_', ' ').title(), fontsize=12)
            ax.set_ylabel("Density", fontsize=12)

            if j == len(column_chunk) - 1:
                sns.move_legend(ax, "upper right", title='Diagnosis')
            else:
                if ax.get_legend() is not None:
                    ax.get_legend().remove()
        for k in range(len(column_chunk), plots_per_figure):
            fig.delaxes(axes[k])

        plt.tight_layout()
        plt.show()

    #wykres tego jak często dany objaw pojawia się przy chorobach
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

    #wykres ilości pacjentów chorujących na daną chorobę
    disease_counts = df['disease_diagnosis'].value_counts()

    disease_counts.plot(
        kind='bar',
        colormap='flare',
        edgecolor='black'
    )

    plt.title("Number of Patients by Diagnosis", fontsize=16, pad=20)
    plt.ylabel("Number of Patients", fontsize=12)
    plt.xlabel("Diagnosis", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


    #wykres dla ilości pacjentów z chorobami według ilości objawów
    plt.figure(figsize=(12, 6))
    sns.histplot(
        data=df,
        x='symptoms_count',
        hue='disease_diagnosis',
        multiple='dodge',
        discrete=True,
        shrink=0.8,
        palette='Set2',
        edgecolor='black'
    )

    plt.title("Symptom Distribution", fontsize=16, pad=20)
    plt.xlabel("Total number of symptoms per patient", fontsize=12)
    plt.ylabel("Number of patients", fontsize=12)
    plt.xticks(range(0, int(df['symptoms_count'].max()) + 1))
    sns.move_legend(plt.gca(), "upper left", bbox_to_anchor=(0.8, 1), title='Diagnosis')

    plt.show()

    #wykres uśrednionych oraz znormalizowanych sylwetek dla każdej z diagnoz

    profile_columns = lab_columns + intake_columns + ['age', 'bmi', 'is_male', 'symptoms_count']

    mean_profiles = df.groupby('disease_diagnosis')[profile_columns].mean()
    normalized_profiles = (mean_profiles - df[profile_columns].min()) / (df[profile_columns].max() - df[profile_columns].min())

    num_vars = len(profile_columns)
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
    ax.set_xticklabels(profile_columns, fontsize=12)
    plt.title("Average Normalized Profile per Diagnosis", fontsize=16, pad=30)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.show()





if __name__ == "__main__":
    main()