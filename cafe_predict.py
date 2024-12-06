import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Carregar dados corretamente do Excel
dataset = pd.read_excel('D:/Downloads/vendas_cafe.xlsx')

# Converter coluna de date para datetime
dataset['date'] = pd.to_datetime(dataset['date'])

# Extrair componentes da data (ano, mês)
dataset['year'] = dataset['date'].dt.year
dataset['month'] = dataset['date'].dt.month

# Verificar o tipo da coluna 'money' e garantir que seja float
if dataset['money'].dtype == 'object':  # 'object' indica string em pandas
    dataset['money'] = dataset['money'].str.replace(',', '.').astype(float)
else:
    dataset['money'] = dataset['money'].astype(float)

# Agrupar dados por ano, mês, somando os valores da coluna 'money'
dados_agrupados = dataset.groupby(['year', 'month']).agg({'money': 'sum'}).reset_index()

# Separar recursos (X) e alvo (y)
X = dados_agrupados[['year', 'month']]
y = dados_agrupados['money']

# Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Treinar modelo de RandomForestRegressor
modelo = RandomForestRegressor(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Função para adicionar previsões ao dataset e prever o próximo mês
def prever_proximos_meses(dados_agrupados, scaler, modelo, meses_previsao):
    previsoes_totais = []

    for i in range(meses_previsao):
        # Atualizar os dados para incluir as previsões mais recentes
        X = dados_agrupados[['year', 'month']]
        y = dados_agrupados['money']

        # Normalizar os dados
        X_scaled = scaler.fit_transform(X)

        # Treinar o modelo novamente
        modelo.fit(X_scaled, y)

        # Gerar previsões para o próximo mês
        ultimo_ano = dados_agrupados['year'].max()
        ultimo_mes = dados_agrupados[dados_agrupados['year'] == ultimo_ano]['month'].max()

        # Criar o novo mês para previsão
        novo_mes = pd.DataFrame({
            'year': [ultimo_ano] * 1,
            'month': [ultimo_mes + 1]
        })

        # Normalizar os novos dados
        novo_mes_scaled = scaler.transform(novo_mes)

        # Fazer a previsão
        previsao = modelo.predict(novo_mes_scaled)[0]
        previsoes_totais.append({
            'year': ultimo_ano,
            'month': ultimo_mes + 1,
            'Previsoes': previsao
        })

        # Adicionar a previsão ao dataset para a próxima iteração
        nova_linha = pd.DataFrame({'year': [ultimo_ano], 'month': [ultimo_mes + 1], 'money': [previsao]})
        dados_agrupados = pd.concat([dados_agrupados, nova_linha], ignore_index=True)

    return previsoes_totais

# Gerar previsões para os próximos 3 meses
previsoes_futuras = prever_proximos_meses(dados_agrupados, scaler, modelo, 2)

# Converter previsões para DataFrame e exibir
previsoes_df_novas = pd.DataFrame(previsoes_futuras)
print(previsoes_df_novas)
