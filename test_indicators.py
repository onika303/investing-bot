from src.database import DatabaseManager
from src.moex_loader import MOEXLoader
from src.indicators import IndicatorCalculator
from datetime import datetime, timedelta

db = DatabaseManager('finance.db')
loader = MOEXLoader(db)

# Загружаем данные по SBER за последние 2 года
today = datetime.now().strftime('%Y-%m-%d')
two_years_ago = (datetime.now() - timedelta(days=2*365)).strftime('%Y-%m-%d')

df = loader.load_asset('SBER', 'stock', two_years_ago, today, interval=24)  # дневные свечи
if not df.empty:
    calc = IndicatorCalculator(df)
    df_with_indicators = calc.compute_all()
    print("Готово! Индикаторы добавлены. Пример:")
    print(df_with_indicators[['ts', 'close', 'ema_12', 'rsi_14', 'macd_line']].tail(10))
else:
    print("Не удалось загрузить данные")