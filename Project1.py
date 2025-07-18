import pandas as pd
import numpy as np
df = pd.read_csv("C:/Users/Saad/Desktop/ProMach1/appartements-data-db-6872f0ba853ec096170787.csv")
# print(df)
df.shape
df.info()
print(df.isnull().sum())
print(df.duplicated().sum())
df = df.drop_duplicates()
print(df)
print(df.duplicated().sum())

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Sélection des colonnes numériques
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
print(numeric_cols)

# Afficher les statistiques globales
print(df[numeric_cols].describe())

# Visualiser la distribution pour chaque variable
for col in numeric_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution de la variable {col}')
    plt.xlabel(col)
    plt.ylabel('Nombre d\'observations')
    plt.show()

# Calcul de la matrice de corrélation
corr_matrix = df[numeric_cols].corr()

# Visualisation : heatmap de corrélation
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

# Scatterplot entre deux variables importantes (exemple : surface vs prix)
sns.scatterplot(x=df['surface_area'], y=df['price'])
plt.title('Relation entre Surface et Prix')
plt.xlabel('Surface (m²)')
plt.ylabel('Prix')
plt.show()

df['equipment'].str.get_dummies("/")

df['price'] = (
    df['price'].astype(str)
    .str.replace('\u202f', '').str.replace(' ', '').str.replace('DH', '').str.replace(',', '')
    .astype(float)
)

df['city_name'].unique()

city_mapping = { 'الدار البيضاء': 'Casablanca',
    'الرباط': 'Rabat',
    'فاس': 'Fès',
    'مراكش': 'Marrakech',
    'طنجة': 'Tanger',
    'أكادير': 'Agadir',
    'المحمدية': 'Mohammedia',
    'القنيطرة':'Kénitra'
    }

    
    # تحويل الأسماء باستعمال map() وتعويض القيم المفقودة بـ "Unknown"
df['city_name'] = df['city_name'].replace(city_mapping)
df['city_name'] = df['city_name'].fillna('unkown')

df['city_name'].unique()

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

# Imputer les valeurs manquantes par la médiane (pour chaque colonne numérique)
for col in numeric_cols:
    median_value = df[col].median()
    df[col].fillna(median_value, inplace=True)

print(df)

print(df.isnull().sum())

unko_columns = df.select_dtypes(include='object').columns
df[unko_columns] = df[unko_columns].fillna('Unknown')
print(df)

# Supprimer avant le traitement des outliers
df = df.drop(columns=['equipment', 'link'], errors='ignore')

print(df.isnull().sum())

import pandas as pd

# Suppose que df est déjà chargé et que les colonnes sont numériques

# Liste des colonnes cibles
cols_to_clean = ['price', 'surface_area']

# Détection et suppression des outliers avec IQR uniquement
outlier_indices = set()

for col in cols_to_clean:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
    outlier_indices.update(outliers)

    print(f"{col} : {len(outliers)} outliers détectés.")

# Supprimer toutes les lignes contenant des outliers
df_clean = df.drop(index=list(outlier_indices))

print(f"\n✅ Total lignes supprimées : {len(outlier_indices)}")

sns.boxplot(df_clean)
