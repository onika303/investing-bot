import yaml
from src.database import DatabaseManager
from src.moex_loader import MOEXLoader
from datetime import datetime, timedelta

# Загружаем конфиг
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Инициализируем БД и загрузчик
db = DatabaseManager('finance.db')
loader = MOEXLoader(db)

# Рассчитываем даты (последние 3 года до сегодня)
today = datetime.now().strftime('%Y-%m-%d')
three_years_ago = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')

# Перебираем все типы активов и загружаем
for asset_type, secids in config['assets'].items():
    for secid in secids:
        print(f"Загружаем {secid} ({asset_type})...")
        loader.load_asset(secid, asset_type, three_years_ago, today, interval=24)