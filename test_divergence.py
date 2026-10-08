from src.database import DatabaseManager
from src.moex_loader import MOEXLoader
from src.indicators import IndicatorCalculator
from src.divergence import DivergenceDetector
from src.utils import ts_to_datetime, ts_to_date
from datetime import datetime, timedelta

db = DatabaseManager('finance.db')
loader = MOEXLoader(db)

today = datetime.now().strftime('%Y-%m-%d')
year_ago = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

df = loader.load_asset('SBER', 'stock', year_ago, today, interval=24)
if not df.empty:
    calc = IndicatorCalculator(df)
    df_with_indicators = calc.compute_all()
    
    # Выводим последние строки с датами
    print("Последние 5 свечей с индикаторами:")
    sample = df_with_indicators.tail(5)
    for _, row in sample.iterrows():
        print(f"{ts_to_datetime(row['ts'])} | Close: {row['close']:.2f} | RSI: {row['rsi_14']:.2f} | MACD: {row['macd_line']:.2f}")
    
    detector = DivergenceDetector(df_with_indicators, window=5, min_distance=3)
    divergences = detector.scan_all_indicators(lookback=200)
    
    if divergences:
        for indicator, divs in divergences.items():
            print(f"\nИндикатор {indicator}:")
            for d in divs:
                date = ts_to_date(df_with_indicators.iloc[d['price_index']]['ts'])
                print(f"  {d['description']} на дате {date}")
    else:
        print("\nДивергенций не найдено.")
else:
    print("Данные не загружены")
    