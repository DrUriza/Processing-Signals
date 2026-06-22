# Processing-Signals Main Pipeline Patch

Este patch agrega una arquitectura ejecutable desde `main.py`:

```text
main.py
  -> MainPipeline
      -> input
      -> processing
      -> processing_math
      -> patterns
      -> classification
      -> output
```

## Carpetas agregadas

```text
src/signal_analysis/input/
src/signal_analysis/processing/
src/signal_analysis/processing_math/
src/signal_analysis/patterns/
src/signal_analysis/classification/
src/signal_analysis/output/
```

## Ejecución

Desde la raíz del repo:

```bash
python main.py --input btc_processing_jsons_v2.zip --output output/main_pipeline_output.json --max-rows 20
```

También acepta:

```bash
python main.py --input carpeta_con_jsons/ --output output/main_pipeline_output.json
python main.py --input archivo.json --output output/main_pipeline_output.json
```

## Tipos detectados

Primera versión del `DataTypeDetector`:

```text
candlestick
orderbook_conventional
orderbook_large_trades
orderbook_whale_orders
manifest
unknown
```

## Métricas iniciales

Candlesticks:
- RSI 14
- MACD
- ATR 14
- Bollinger Bands 20
- EMA 20 / EMA 50
- VWAP
- OBV
- rolling mean/std/var/skewness/kurtosis/zscore/range/iqr/autocorr/drawdown
- rolling beta de retorno vs volumen, como placeholder inicial

Order book:
- best bid / best ask
- spread
- mid price
- weighted mid price
- total bid/ask notional
- imbalance total
- imbalance top 5/10/20
- bid wall score / ask wall score

Large trades:
- buy/sell count
- buy/sell notional
- flow imbalance
- event age

Whale orders:
- buy/sell count
- buy/sell notional
- flow imbalance
- active duration / TTL
- whale order age statistics

## Próximos pasos

1. Conectar este pipeline con los indicadores ya existentes del repo.
2. Agregar `rolling_beta` contra benchmarks reales: ETH, NASDAQ, SPX, DXY, CVD, OI, ETF netflow.
3. Expandir `patterns/` con candlestick patterns completos.
4. Agregar microestructura avanzada: microprice, OFI, absorption, sweep detector, spoof/wall fade score.
