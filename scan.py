#!/usr/bin/env python3
"""Точка входа инвестиционного сканера.

Загрузка данных MOEX ISS в SQLite и формирование отчёта по сигналам.

Примеры:
    python scan.py                       # загрузить всё из config.yaml + отчёт
    python scan.py --no-load             # только отчёт по тому, что уже в БД
    python scan.py --asset SBER          # загрузить и просканировать один актив
    python scan.py --top 30              # показать топ-30 сигналов
    python scan.py --db finance.db --config config.yaml
"""
import argparse
import asyncio
import sys
from datetime import datetime, timedelta

import yaml
from loguru import logger
from tabulate import tabulate

from src.database import DatabaseManager
from src.moex_loader import MOEXLoader
from src.scanner import InvestmentScanner


def parse_args():
    p = argparse.ArgumentParser(description='Инвестиционный сканер MOEX')
    p.add_argument('--config', default='config.yaml')
    p.add_argument('--db', default='finance.db')
    p.add_argument('--asset', help='просканировать только этот secid')
    p.add_argument('--days', type=int, default=3 * 365, help='история загрузки в днях (по умолчанию 3 года)')
    p.add_argument('--no-load', action='store_true', help='не загружать данные, только скан по БД')
    p.add_argument('--top', type=int, default=25, help='сколько строк показать в отчёте')
    return p.parse_args()


def load_all_assets(loader: MOEXLoader, assets: dict, days: int):
    """load_asset у MOEXLoader синхронный (внутри свой asyncio loop).

    ВАЖНО: ISS отдаёт максимум 500 свечей за запрос и обрезает окно по началу
    (from), поэтому при start в прошлом всегда возвращаются СТАРЫе данные.
    Запрос строится от (сегодня - days) до сегодня с end_on_load=True —
    loader сам перевернёт пагинацию и дойдёт до самых свежих свечей.
    """
    today = datetime.now().strftime('%Y-%m-%d')
    from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    for asset_type, secids in assets.items():
        for secid in secids:
            try:
                loader.load_asset(secid, asset_type, from_date, today,
                                  interval=24, end_on_load=True)
            except Exception as e:
                logger.error(f"Загрузка {secid} не удалась: {e}")


def print_report(results: list, top: int):
    active = [r for r in results if r['verdict'] in ('buy', 'sell')]
    watch = [r for r in results if r['verdict'] == 'watch']

    rows = []
    for r in (active + watch)[:top]:
        rules = '; '.join(s['rule'] for s in r['signals'][:4]) or '-'
        sl_tp = ''
        if 'sl_tp' in r and r['verdict'] in ('buy', 'sell'):
            sl_tp = f"{r['sl_tp']['stop_loss']} / {r['sl_tp']['take_profit']}"
        icon = {'buy': '🟢 ПОКУПКА', 'sell': '🔴 ПРОДАЖА', 'watch': '👁 наблюдение'}.get(r['verdict'], r['verdict'])
        if r.get('sell_blocked_reasons'):
            icon += ' (шорт запрещён)'
        rows.append([r['secid'], r['asset_type'], r['date'], r['price'], icon,
                     f"B{r['score_buy']}/S{r['score_sell']}", rules, sl_tp])

    print(f"\nОтчёт сканера на {datetime.now():%Y-%m-%d %H:%M} | "
          f"просканировано: {len(results)}, сигналов buy/sell: {len(active)}, watch: {len(watch)}")
    if not rows:
        print("Активных сигналов нет. Данные в БД могут быть неполными — прогоните загрузку без --no-load.")
        return
    print(tabulate(rows, headers=['TICKER', 'ТИП', 'ДАТА', 'ЦЕНА', 'СИГНАЛ', 'SCORE', 'ОСНОВАНИЕ', 'SL/TP'],
                   tablefmt='outline', disable_numparse=True))


def main():
    args = parse_args()
    with open(args.config, encoding='utf-8') as f:
        assets = yaml.safe_load(f)['assets']

    if args.asset:
        assets = {t: [args.asset] for t, ss in assets.items() if args.asset in ss}
        if not assets:
            print(f"Актив {args.asset} не найден в config.yaml", file=sys.stderr)
            sys.exit(1)

    db = DatabaseManager(args.db)

    if not args.no_load:
        loader = MOEXLoader(db)
        load_all_assets(loader, assets, args.days)

    scanner = InvestmentScanner(db)
    results = scanner.scan_all(assets)
    print_report(results, args.top)


if __name__ == '__main__':
    main()
