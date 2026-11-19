# Stock Price Prediction com LSTM e RNN

Projeto educativo de previsão do preço de fecho da AAPL com dados do Yahoo Finance e TensorFlow/Keras. O objetivo é seguir o guia em `docs/guia.md` e comparar três modelos — SimpleRNN, LSTM univariado e LSTM multivariado — **sempre contra um baseline ingénuo** ("o preço de amanhã é igual ao de hoje").

> **Aviso:** isto é um projeto educativo e **não é aconselhamento financeiro**. Preços de ações aproximam-se de um passeio aleatório; um gráfico convincente não é uma estratégia de negociação.

## Instalação

Requer Python 3.12 (o guia pede 3.10+, mas as versões fixadas foram testadas em 3.12.16).

```bash
python3.12 -m venv venv
venv/bin/pip install -r requirements.txt
```

As versões estão todas fixadas em `requirements.txt`. O `pandas-ta` é uma versão beta (`0.4.71b0`) que obriga a `numba==0.61.2` e `numpy==2.2.6` — não atualizar o NumPy isoladamente.

## Como correr

Todos os comandos partem da raiz do projeto:

```bash
venv/bin/python -m src.data                 # descarrega e coloca em cache data/aapl.csv
venv/bin/python -m src.data --force         # força novo download
venv/bin/python -m src.features             # verifica splits, scalers, janelas e indicadores
venv/bin/python -m src.train --model rnn    # SimpleRNN
venv/bin/python -m src.train --model lstm   # LSTM univariado
venv/bin/python -m src.train --model multi  # LSTM multivariado (Close, Volume, RSI, EMA20, EMA50)
venv/bin/python -m src.evaluate             # tabela e gráfico comparativos
```

Não há ecrã: o matplotlib usa `Agg` e todos os gráficos são guardados em `reports/`. Não há notebooks nem testes automatizados; a verificação é feita pelos próprios scripts.

## Resultados

Os três modelos foram avaliados no **mesmo período de teste**, 2023-11-09 a 2026-10-30 (746 dias), com divisão cronológica e scalers ajustados apenas nos dados de treino.

| Model | RMSE | MAE | MAPE | Directional accuracy |
| --- | ---: | ---: | ---: | ---: |
| SimpleRNN | 19.26 | 14.20 | 5.26% | 48.72% |
| LSTM | 20.16 | 15.72 | 5.95% | 52.08% |
| Multivariate LSTM | 31.98 | 24.82 | 9.32% | 50.74% |
| Naive baseline | 3.94 | 2.70 | 1.14% | 49.87% |

O baseline ingénuo vence em RMSE, MAE e MAPE: os modelos erram o preço entre 5 e 8 vezes mais do que "copiar o dia anterior". O LSTM univariado é o único que ultrapassa o baseline na precisão direcional (52,08% contra 49,87%), ainda assim de forma marginal.

## Estrutura

```
data/      CSV em cache (aapl.csv)
models/    modelos .keras guardados
reports/   gráficos, results.csv, comparison.md e comparison.png
src/
  data.py       download, cache e sanity checks
  features.py   scaling, janelas, RSI/EMA e splits cronológicos
  models.py     build_rnn, build_lstm e seeds
  evaluate.py   métricas, baseline, gráficos e comparação
  train.py      pipeline com --model rnn|lstm|multi
docs/guia.md     especificação convertida do DOCX original
```

## Limitações

- Previsão de **níveis de preço** sofre de "ilusão de atraso": as curvas parecem seguir o real, mas os erros são grandes.
- O baseline é deliberadamente mantido em todas as comparações para evitar conclusões enganosas.
- Não há backtesting com custos de transação, slippage nem validação walk-forward.
- A normalização usa `MinMaxScaler` ajustado no treino; em trends fortes os preços de teste podem ficar acima do máximo visto no treino.
- Seeds fixas (42) e versões fixadas reduzem a variação, mas resultados podem diferir ligeiramente entre máquinas.
- Para trabalho real seria necessário prever retornos em vez de preços, acrescentar dados de mercado e usar validação rolante.
