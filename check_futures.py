from src.database import DatabaseManager
from src.moex_loader import MOEXLoader

db = DatabaseManager('finance.db')
loader = MOEXLoader(db)

# Получаем список контрактов
contracts = loader.get_futures_list('Si')
print(f"Найдено контрактов: {len(contracts)}")
print(contracts[['secid', 'shortname', 'matdate']].head(10))