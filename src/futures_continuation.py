import pandas as pd
from datetime import datetime, timedelta
from loguru import logger
from typing import Optional

from .database import DatabaseManager
from .moex_loader import MOEXLoader


def build_continuous_series(
    base_secid: str,
    from_date: str,
    till_date: str,
    loader: MOEXLoader,
    db: DatabaseManager,
    rollover_days_before: int = 0
) -> pd.DataFrame:
    """
    Строит непрерывный ряд для фьючерсного актива (например, 'GOLD').
    Возвращает DataFrame с колонками: secid, ts, open, high, low, close, volume, oi.
    Сохраняет результат в continuous_ohlc через db.
    """
    # 1. Получаем список контрактов
    contracts_df = loader.get_futures_list(base_secid)
    if contracts_df.empty:
        logger.error(f"Нет контрактов для {base_secid}")
        return pd.DataFrame()

    # Убедимся, что колонка 'secid' существует (она должна быть после переименования в moex_loader)
    if 'secid' not in contracts_df.columns:
        logger.error(f"В contracts_df нет колонки 'secid'. Доступны: {contracts_df.columns}")
        return pd.DataFrame()

    logger.info(f"Найдено контрактов для {base_secid}: {len(contracts_df)}")

    # 2. Загружаем данные по каждому контракту
    all_dfs = {}
    for _, row in contracts_df.iterrows():
        secid = row['secid']
        logger.info(f"Загружаем контракт {secid}...")
        df = loader.fetch_candles(secid, 'futures', from_date, till_date, interval=1)
        if df.empty:
            logger.warning(f"Для {secid} данные не найдены.")
            continue
        df['matdate'] = row['matdate']
        all_dfs[secid] = df

    if not all_dfs:
        logger.error(f"Не удалось загрузить данные ни по одному контракту для {base_secid}")
        return pd.DataFrame()

    # 3. Сортируем контракты по дате экспирации
    sorted_contracts = contracts_df.sort_values('matdate').to_dict('records')

    # 4. Построение непрерывного ряда
    continuous_parts = []
    prev_df = None

    for i, contract in enumerate(sorted_contracts):
        secid = contract['secid']
        if secid not in all_dfs:
            continue
        df = all_dfs[secid].copy()

        if i == 0:
            # Первый контракт, коэффициент = 1
            coef = 1.0
        else:
            # Определяем дату перехода
            rollover_date = contract['matdate'] - timedelta(days=rollover_days_before)
            rollover_ts = int(rollover_date.timestamp())
            # Находим последнюю свечу предыдущего контракта до rollover_date
            prev_last = prev_df[prev_df['ts'] <= rollover_ts].iloc[-1] if not prev_df[prev_df['ts'] <= rollover_ts].empty else None
            # Находим первую свечу нового контракта после rollover_date
            new_first = df[df['ts'] >= rollover_ts].iloc[0] if not df[df['ts'] >= rollover_ts].empty else None

            if prev_last is None or new_first is None:
                logger.warning(f"Нет пересечения для перехода {secid} на {rollover_date.date()}, используем коэффициент 1")
                coef = 1.0
            else:
                coef = new_first['open'] / prev_last['close']
                logger.info(f"Коэффициент для {secid} на {rollover_date.date()}: {coef:.4f}")

        # Применяем коэффициент ко всем ценам текущего контракта
        if coef != 1.0:
            df['open'] = df['open'] * coef
            df['high'] = df['high'] * coef
            df['low'] = df['low'] * coef
            df['close'] = df['close'] * coef

        # Обрезаем данные до даты перехода к следующему контракту (кроме последнего)
        if i < len(sorted_contracts) - 1:
            next_contract = sorted_contracts[i+1]
            next_rollover = next_contract['matdate'] - timedelta(days=rollover_days_before)
            next_ts = int(next_rollover.timestamp())
            df = df[df['ts'] < next_ts]

        continuous_parts.append(df)
        prev_df = df

    if not continuous_parts:
        return pd.DataFrame()

    # 5. Объединяем все части
    continuous_df = pd.concat(continuous_parts, ignore_index=True)
    continuous_df = continuous_df.drop_duplicates(subset=['ts']).sort_values('ts')
    continuous_df['secid'] = base_secid
    continuous_df = continuous_df.drop(columns=['matdate'], errors='ignore')

    # 6. Сохраняем в БД
    db.insert_continuous_candles(continuous_df)
    if not continuous_df.empty:
        last_ts = continuous_df['ts'].max()
        db.upsert_meta(base_secid, last_ts, 'futures_continuous', len(continuous_df))
        logger.info(f"Построен непрерывный ряд для {base_secid}, записей: {len(continuous_df)}")

    return continuous_df