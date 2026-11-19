REGRAS TÉCNICAS OBRIGATÓRIAS
- Divisão cronológica treino/teste, sem shuffle antes de dividir
- Scalers ajustados só com dados de treino (sem data leakage)
- Comparar sempre com um baseline ingénuo (preço de amanhã = preço de hoje)
- Métricas: RMSE, MAE, MAPE e precisão direcional, guardadas em reports/results.csv
- Seeds fixas e versões das bibliotecas em requirements.txt
- Comentários no código em inglês; explicações em português de Portugal
- Não instales nada fora do venv nem alteres ficheiros fora de ~/stock-prediction

