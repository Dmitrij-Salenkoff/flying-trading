Сгенерировать 20 самых ликвидных монет
uv run freqtrade test-pairlist -c config_pairlist.json --print-json > pairs.json

Скачать данные 
uv run freqtrade download-data   --exchange bybit   --pairs-file pairs.json   --timerange 20250101-20260101 -t 5m --trading-mode futures

Запустить стратегию
uv run freqtrade trade --config user_data/config.json --strategy SampleStrategy

Бэктест
freqtrade backtesting -c config_backtest.json -s SampleStrategy --timerange 20240104-20260104 -i 5m

Оптимизация
freqtrade hyperopt -c config_backtest.json -s SampleStrategy --timerange 20250101-20250701 -i 5m --spaces exit buy sell -e 200 --random-state 42 --hyperopt-loss OnlyProfitHyperOptLoss