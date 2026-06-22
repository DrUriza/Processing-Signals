# Resultado de ejecución del MainPipeline

Entrada usada: `btc_processing_jsons_v2.zip`

Archivos procesados: 8

Tipos detectados:

```json
{
  "candlestick": 4,
  "orderbook_conventional": 1,
  "orderbook_large_trades": 1,
  "orderbook_whale_orders": 1,
  "manifest": 1
}
```

Bloques principales creados:

- `input`: carga ZIP/JSON/directorio.
- `processing`: `DataTypeDetector`, `Normalizer`, `IndicatorDecisionEngine`.
- `processing_math`: indicadores técnicos, estadísticas puras, microestructura.
- `patterns`: candlestick, regímenes estadísticos, liquidez/eventos.
- `classification`: rutas HMI / ML / algoritmia avanzada.
- `output`: JSON final unificado + preview de matriz ML.

Output demo:

`demo_output/main_pipeline_output.json`
