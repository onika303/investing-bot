"""Генерация config.yaml: все акции MOEX (сектор 'Акции') + топ-15 фьючерсов по дневному обороту."""
import datetime
import requests

ISS = "https://iss.moex.com/iss"

def get(url, **params):
    params['iss.meta'] = 'off'
    r = requests.get(url, params=params, timeout=30, headers={'User-Agent': 'Mozilla/5.0'})
    r.raise_for_status()
    return r.json()

# 1. Все акции Мосбиржи: рынок shares, досрочный торги TQBR, сектор "Акции" (SECTYPE='1')
d = get(f"{ISS}/engines/stock/markets/shares/securities.json",
        **{'iss.only': 'securities', 'securities.columns': 'SECID,SECTYPE,BOARDIDS'})
stocks = sorted({r[0] for r in d['securities']['data']
                 if r[0] and r[1] == '1' and r[2] and 'TQBR' in r[2]})
print(f"Акции (сектор 'Акции', TQBR): {len(stocks)}")

# 2. Топ-15 фьючерсов по дневному обороту (VALTODAY)
d = get(f"{ISS}/engines/futures/markets/forts/securities.json",
        **{'iss.only': 'marketdata', 'marketdata.columns': 'SECID,VALTODAY'})
fut = [(r[1] or 0, r[0]) for r in d['marketdata']['data'] if r[0]]
fut.sort(reverse=True)
top_futures = [s for _, s in fut[:15]]
print("Топ-15 фьючерсов:", top_futures)

today = datetime.date.today().isoformat()
with open('config.yaml', 'w', encoding='utf-8') as f:
    f.write(f"# Сгенерировано {today}: все акции MOEX (сектор Акции, досрочные торги TQBR) + топ-15 фьючерсов по дневной выручке\n")
    f.write("# ВНИМАНИЕ: фьючерсы — конкретные серийные контракты (обновлять при ролльовере!)\n")
    f.write("assets:\n  stocks:\n")
    for s in stocks:
        f.write(f"    - {s}\n")
    f.write("\n  futures:\n")
    for s in top_futures:
        f.write(f"    - {s}\n")
    f.write("\n  indices:\n    - IMOEX\n    - RGBI\n")
print("config.yaml записан")
