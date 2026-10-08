import asyncio
import aiohttp
import pandas as pd

async def fetch_all_secids():
    url = "https://iss.moex.com/iss/engines/futures/markets/forts/securities.json"
    params = {
        'iss.meta': 'off',
        'iss.only': 'securities'
    }
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params, timeout=30) as resp:
            if resp.status != 200:
                print(f"Ошибка: {resp.status}")
                return
            data = await resp.json()
            securities = data.get('securities', {}).get('data', [])
            columns = data.get('securities', {}).get('columns', [])
            df = pd.DataFrame(securities, columns=columns)
            # Выводим все SECID и SHORTNAME
            print("Все доступные фьючерсы:")
            for idx, row in df.iterrows():
                print(f"{row['SECID']} - {row['SHORTNAME']}")
            # Также ищем содержащие "Si" (регистронезависимо)
            print("\nПоиск контрактов с 'Si' в SECID или SHORTNAME:")
            mask = df['SECID'].str.contains('Si', case=False, na=False) | df['SHORTNAME'].str.contains('Si', case=False, na=False)
            df_si = df[mask]
            if df_si.empty:
                print("Ничего не найдено")
            else:
                print(df_si[['SECID', 'SHORTNAME']].to_string(index=False))

if __name__ == "__main__":
    asyncio.run(fetch_all_secids())