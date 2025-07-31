# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carrega o conjunto de dados
# Certifique-se de que o arquivo 'medical_insurance.csv' esteja no diretório './dataset/'
try:
    df = pd.read_csv('./dataset/medical_insurance.csv')
    print("Conjunto de dados 'medical_insurance.csv' carregado com sucesso.")
except FileNotFoundError:
    print("Erro: O arquivo 'medical_insurance.csv' não foi encontrado. Certifique-se de que está no diretório correto.")
    exit()  # Sai do script se o arquivo não for encontrado

# Análise inicial do DataFrame
print("\n--- Análise Inicial ---")
print("Formato do DataFrame (linhas, colunas):", df.shape)
print("\nInformações do DataFrame:")
df.info()

# Exibe as primeiras 5 linhas do DataFrame
print("\n--- Primeiras 5 linhas do DataFrame ---")
print(df.head())

# Exibe estatísticas descritivas para colunas numéricas
print("\n--- Estatísticas Descritivas ---")
print(df.describe())

# Verificando se os dados estão corretamente padronizados para colunas categóricas
print("\n--- Valores Únicos para Colunas Categóricas ---")
print("Gênero:", df['gender'].unique())
print("Elegibilidade para Desconto:", df['discount_eligibility'].unique())
print("Região:", df['region'].unique())

# Correlação dos dados numéricos
print("\n--- Matriz de Correlação ---")
# Agrupa as colunas numéricas
numericas = df.select_dtypes(include=['int64', 'float64'])
# Calcula a matriz de correlação
correlacao = numericas.corr()
print(correlacao)

# Gera e exibe o mapa de calor da matriz de correlação
plt.figure(figsize=(8, 6))
sns.heatmap(correlacao, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Matriz de Correlação')
plt.show()


# Gera correlação completa
df_dummies = pd.get_dummies(df, drop_first=True)
correlacao_completa = df_dummies.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlacao_completa, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Matriz de Correlação')
plt.show()

sns.lmplot(x='age', y='insurance_cost', data=df, line_kws={"color": "red"})
plt.title('Relação entre Idade e Custo do Seguro')
plt.show()

sns.lmplot(x='bmi', y='insurance_cost', data=df, line_kws={"color": "red"})
plt.title('Relação entre IMC e Custo do Seguro')
plt.show()
