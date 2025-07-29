# Detecção de Fraudes em E-commerce: Análise de Padrões Comportamentais
# Importar bibliotecas necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo dos gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("=== ANÁLISE DE FRAUDES EM E-COMMERCE ===")
print("Carregando e analisando os datasets...")

# =============================================================================
# FASE 1: CARREGAMENTO E INSPEÇÃO INICIAL DOS DADOS
# =============================================================================


def carregar_dados():
    """Carrega os datasets de produtos e transações"""
    try:
        produtos_df = pd.read_csv('./dataset/counterfeit_products.csv')
        transacoes_df = pd.read_csv('./dataset/_counterfeit_transactions.csv')

        print(
            f"\n📊 PRODUTOS: {produtos_df.shape[0]} registros, {produtos_df.shape[1]} colunas")
        print(
            f"📊 TRANSAÇÕES: {transacoes_df.shape[0]} registros, {transacoes_df.shape[1]} colunas")

        return produtos_df, transacoes_df
    except FileNotFoundError as e:
        print(f"❌ Erro ao carregar arquivos: {e}")
        print("Certifique-se de que os arquivos 'produtos.csv' e 'transacoes.csv' estão no diretório correto")
        return None, None


def inspecionar_dados(df, nome):
    """Realiza inspeção inicial dos dados"""
    print(f"\n=== INSPEÇÃO: {nome.upper()} ===")
    print(f"Shape: {df.shape}")
    print(f"\nTipos de dados:")
    print(df.dtypes)
    print(f"\nValores ausentes:")
    print(df.isnull().sum())
    print(f"\nDuplicatas: {df.duplicated().sum()}")
    print(f"\nPrimeiras 5 linhas:")
    print(df.head())

# =============================================================================
# FASE 2: DATA WRANGLING E LIMPEZA
# =============================================================================


def limpar_dados(df):
    """Aplica limpeza básica nos dados"""
    df_limpo = df.copy()

    # Tratar valores ausentes numéricas com mediana
    colunas_numericas = df_limpo.select_dtypes(include=[np.number]).columns
    for col in colunas_numericas:
        if df_limpo[col].isnull().sum() > 0:
            df_limpo[col].fillna(df_limpo[col].median(), inplace=True)

    # Tratar valores ausentes categóricas com moda
    colunas_categoricas = df_limpo.select_dtypes(include=['object']).columns
    for col in colunas_categoricas:
        if df_limpo[col].isnull().sum() > 0:
            df_limpo[col].fillna(df_limpo[col].mode()[0], inplace=True)

    return df_limpo


def integrar_datasets(transacoes_df, produtos_df):
    """Integra os datasets de transações e produtos"""
    print("\n🔗 INTEGRANDO DATASETS...")

    # Verificar chaves de junção
    print(
        f"Produtos únicos em transações: {transacoes_df['product_id'].nunique()}")
    print(
        f"Produtos únicos em produtos: {produtos_df['product_id'].nunique()}")

    # Realizar merge
    dados_completos = pd.merge(transacoes_df, produtos_df,
                               on='product_id', how='left')

    print(f"Registros após merge: {dados_completos.shape[0]}")
    print(
        f"Produtos não encontrados: {dados_completos['product_id'].isnull().sum()}")

    return dados_completos

# =============================================================================
# FASE 3: FEATURE ENGINEERING PARA DETECÇÃO DE FRAUDES
# =============================================================================


def criar_indicadores_fraude(df):
    """Cria indicadores comportamentais de fraude"""
    print("\n🔍 CRIANDO INDICADORES DE FRAUDE...")

    df_features = df.copy()

    # Converter data para datetime se necessário
    if 'transaction_date' in df.columns:
        df_features['transaction_date'] = pd.to_datetime(
            df_features['transaction_date'])
        df_features = df_features.sort_values(
            ['customer_id', 'transaction_date'])

        # Calcular tempo entre transações por cliente
        df_features['time_diff'] = df_features.groupby(
            'customer_id')['transaction_date'].diff()
        df_features['time_diff_hours'] = df_features['time_diff'].dt.total_seconds() / \
            3600

        # Velocidade suspeita (transações muito rápidas)
        df_features['velocidade_suspeita'] = (
            df_features['time_diff_hours'] < 1).astype(int)
        df_features['velocidade_suspeita'].fillna(0, inplace=True)

    # Incompatibilidades geográficas
    if all(col in df.columns for col in ['customer_country', 'seller_country']):
        df_features['incompatibilidade_geo'] = (
            df_features['customer_country'] != df_features['seller_country']
        ).astype(int)

    # Outros indicadores de risco
    if 'discount_rate' in df.columns:
        df_features['desconto_alto'] = (
            df_features['discount_rate'] > 0.5).astype(int)

    if 'seller_rating' in df.columns:
        df_features['avaliacao_baixa'] = (
            df_features['seller_rating'] < 3.0).astype(int)

    if 'product_images' in df.columns:
        df_features['poucas_imagens'] = (
            df_features['product_images'] < 3).astype(int)

    if 'spelling_errors' in df.columns:
        df_features['muitos_erros'] = (
            df_features['spelling_errors'] > 5).astype(int)

    # Score composto de fraude
    indicadores = ['velocidade_suspeita', 'incompatibilidade_geo', 'desconto_alto',
                   'avaliacao_baixa', 'poucas_imagens', 'muitos_erros']

    # Verificar quais indicadores existem
    indicadores_existentes = [
        ind for ind in indicadores if ind in df_features.columns]

    if indicadores_existentes:
        df_features['score_fraude'] = df_features[indicadores_existentes].mean(
            axis=1)
        df_features['is_fraud'] = (
            df_features['score_fraude'] > 0.4).astype(int)

        print(f"✅ Criados {len(indicadores_existentes)} indicadores de fraude")
        print(
            f"Score médio de fraude: {df_features['score_fraude'].mean():.3f}")
        print(
            f"Transações classificadas como fraude: {df_features['is_fraud'].sum()}")

    return df_features

# =============================================================================
# FASE 4: ANÁLISE DE CORRELAÇÕES
# =============================================================================


def analisar_correlacoes(df):
    """Analisa correlações entre variáveis"""
    print("\n📈 ANÁLISE DE CORRELAÇÕES...")

    # Selecionar apenas colunas numéricas
    colunas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(colunas_numericas) < 2:
        print("❌ Poucas variáveis numéricas para análise de correlação")
        return None

    # Calcular matriz de correlação
    correlacao_matrix = df[colunas_numericas].corr()

    # Visualizar matriz de correlação
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(correlacao_matrix, dtype=bool))
    sns.heatmap(correlacao_matrix, mask=mask, annot=True, cmap='coolwarm',
                center=0, square=True, linewidths=0.5, fmt='.2f')
    plt.title('Matriz de Correlação - Indicadores de Fraude')
    plt.tight_layout()
    plt.show()

    # Identificar correlações mais fortes com score de fraude
    if 'score_fraude' in correlacao_matrix.columns:
        correlacoes_fraude = correlacao_matrix['score_fraude'].abs(
        ).sort_values(ascending=False)
        print("\n🎯 CORRELAÇÕES MAIS FORTES COM SCORE DE FRAUDE:")
        print(correlacoes_fraude.head(10))

    return correlacao_matrix

# =============================================================================
# FASE 5: MODELAGEM COM REGRESSÃO
# =============================================================================


def preparar_dados_modelagem(df):
    """Prepara dados para modelagem"""
    print("\n🤖 PREPARANDO DADOS PARA MODELAGEM...")

    # Selecionar features relevantes (ajustar conforme seu dataset)
    features_candidatas = ['seller_rating', 'seller_reviews', 'product_price',
                           'shipping_days', 'product_images', 'description_length',
                           'spelling_errors', 'discount_rate', 'time_diff_hours']

    # Verificar quais features existem
    features = [f for f in features_candidatas if f in df.columns]

    if not features:
        print("❌ Nenhuma feature encontrada para modelagem")
        return None, None, None

    print(f"Features selecionadas: {features}")

    # Preparar X e y
    X = df[features].fillna(0)

    if 'is_fraud' in df.columns:
        y = df['is_fraud']
    else:
        print("❌ Variável target 'is_fraud' não encontrada")
        return None, None, None

    # Remover outliers extremos (opcional)
    Q1 = X.quantile(0.25)
    Q3 = X.quantile(0.75)
    IQR = Q3 - Q1
    outlier_condition = ~((X < (Q1 - 1.5 * IQR)) |
                          (X > (Q3 + 1.5 * IQR))).any(axis=1)

    X_clean = X[outlier_condition]
    y_clean = y[outlier_condition]

    print(f"Registros após limpeza de outliers: {len(X_clean)}")

    return X_clean, y_clean, features


def treinar_modelo_logistico(X, y, features):
    """Treina modelo de regressão logística"""
    print("\n📊 TREINANDO MODELO DE REGRESSÃO LOGÍSTICA...")

    # Normalizar features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Treinar modelo
    modelo = LogisticRegression(random_state=42, max_iter=1000)
    modelo.fit(X_train, y_train)

    # Fazer predições
    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:, 1]

    # Avaliar modelo
    print("=== PERFORMANCE DO MODELO LOGÍSTICO ===")
    print(classification_report(y_test, y_pred))

    # Importância das features
    coeficientes = pd.DataFrame({
        'Feature': features,
        'Coeficiente': modelo.coef_[0],
        'Importancia_Abs': np.abs(modelo.coef_[0])
    }).sort_values('Importancia_Abs', ascending=False)

    print("\n🎯 IMPORTÂNCIA DAS FEATURES:")
    print(coeficientes)

    return modelo, scaler, coeficientes

# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================


def main():
    """Função principal que executa toda a análise"""

    # Carregar dados
    produtos_df, transacoes_df = carregar_dados()

    if produtos_df is None or transacoes_df is None:
        return

    # Inspeção inicial
    inspecionar_dados(produtos_df, "PRODUTOS")
    inspecionar_dados(transacoes_df, "TRANSAÇÕES")

    # Limpeza
    produtos_limpo = limpar_dados(produtos_df)
    transacoes_limpo = limpar_dados(transacoes_df)

    # Integração
    dados_completos = integrar_datasets(transacoes_limpo, produtos_limpo)

    # Feature engineering
    dados_com_features = criar_indicadores_fraude(dados_completos)

    # Análise de correlações
    matriz_corr = analisar_correlacoes(dados_com_features)

    # Modelagem
    X, y, features = preparar_dados_modelagem(dados_com_features)

    if X is not None:
        modelo, scaler, importancias = treinar_modelo_logistico(X, y, features)

    print("\n✅ ANÁLISE CONCLUÍDA!")
    print("Próximos passos:")
    print("1. Analisar os resultados das correlações")
    print("2. Interpretar a importância das features")
    print("3. Criar visualizações para o artigo")
    print("4. Documentar os achados científicos")


# Executar análise
if __name__ == "__main__":
    main()
