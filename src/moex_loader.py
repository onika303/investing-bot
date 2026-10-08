import aiohttp
import asyncio
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from loguru import logger
import time

from .database import DatabaseManager


class MOEXLoader:
    BASE_URL = "https://iss.moex.com/iss"

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    # ---------- Вспомогательные методы ----------

    def _build_url(self, secid: str, asset_type: str, from_date: str, till_date: str, interval: int = 1) -> str:
        """
        Формирует URL для запроса к MOEX в зависимости от типа актива.
        """
        if asset_type == 'stock':
            path = f"/engines/stock/markets/shares/boards/TQBR/securities/{secid}/candles.json"
        elif asset_type == 'futures':
            path = f"/engines/futures/markets/forts/securities/{secid}/candles.json"
        elif asset_type == 'index':
            path = f"/engines/stock/markets/indexes/securities/{secid}/candles.json"
        else:
            raise ValueError(f"Неизвестный тип актива: {asset_type}")

        params = {
            'from': from_date,
            'till': till_date,
            'interval': interval,
            'iss.meta': 'off'
        }
        query = '&'.join([f"{k}={v}" for k, v in params.items()])
        return f"{self.BASE_URL}{path}?{query}"

    async def _fetch_page(self, session: aiohttp.ClientSession, url: str, start: int, secid: str) -> Dict[str, Any]:
        """
        Загружает одну страницу свечей с указанным смещением (start).
        Возвращает словарь с полями 'candles' (list of dict) и 'total' (int).
        """
        full_url = f"{url}&start={start}"
        logger.debug(f"Запрос: {full_url}")
        try:
            async with session.get(full_url, timeout=30) as resp:
                if resp.status != 200:
                    logger.error(f"Ошибка HTTP {resp.status} для {secid} при start={start}")
                    return {'candles': [], 'total': 0}
                data = await resp.json()
                candles_block = data.get('candles', {})
                columns = candles_block.get('columns', [])
                rows = candles_block.get('data', [])
                candles = [dict(zip(columns, row)) for row in rows]
                total = candles_block.get('total', len(candles))
                return {'candles': candles, 'total': total}
        except Exception as e:
            logger.error(f"Ошибка загрузки {secid}: {e}")
            return {'candles': [], 'total': 0}

    # ---------- Основной метод загрузки свечей ----------

    async def fetch_candles_async(self, secid: str, asset_type: str, from_date: str, till_date: str, interval: int = 1) -> pd.DataFrame:
        """
        Асинхронно загружает свечи за период с пагинацией.
        Возвращает DataFrame с колонками: secid, ts, open, high, low, close, volume, oi.
        """
        url = self._build_url(secid, asset_type, from_date, till_date, interval)
        all_candles = []
        start = 0
        page_size = 100

        async with aiohttp.ClientSession() as session:
            first_page = await self._fetch_page(session, url, start, secid)
            if not first_page['candles']:
                logger.warning(f"Нет данных для {secid} за период {from_date} - {till_date}")
                return pd.DataFrame()
            all_candles.extend(first_page['candles'])
            total = first_page['total']

            while len(all_candles) < total:
                start += page_size
                next_page = await self._fetch_page(session, url, start, secid)
                if not next_page['candles']:
                    break
                all_candles.extend(next_page['candles'])

        if not all_candles:
            return pd.DataFrame()

        df = pd.DataFrame(all_candles)
        df.rename(columns={
            'begin': 'ts_iso',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        }, inplace=True)
        print("DEBUG: первые 3 значения ts_iso:", df['ts_iso'].head(3).tolist())
        df['ts'] = pd.to_datetime(df['ts_iso']).astype('int64') // 10**9
        df['secid'] = secid
        if 'oi' in df.columns:
            df['oi'] = df['oi'].fillna(0).astype(int)
        else:
            df['oi'] = 0

        df = df[['secid', 'ts', 'open', 'high', 'low', 'close', 'volume', 'oi']]
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(int)
        df['oi'] = df['oi'].astype(int)

        return df

    def fetch_candles(self, secid: str, asset_type: str, from_date: str, till_date: str, interval: int = 1) -> pd.DataFrame:
        """Синхронная обёртка для fetch_candles_async."""
        return asyncio.run(self.fetch_candles_async(secid, asset_type, from_date, till_date, interval))

    # ---------- Получение списка фьючерсов ----------
    
    async def fetch_futures_list_async(self, base_secid: str) -> pd.DataFrame:
        # Пробуем несколько эндпоинтов, пока не найдём контракты
        endpoints = [
            f"{self.BASE_URL}/engines/futures/markets/forts/securities.json",
            f"{self.BASE_URL}/engines/futures/markets/forts/boards/RFUD/securities.json",
            f"{self.BASE_URL}/engines/futures/markets/forts/boards/CFI/securities.json",
        ]
        for url in endpoints:
            params = {'iss.meta': 'off', 'iss.only': 'securities'}
            # Добавим trading_status=all для исторических
            params['trading_status'] = 'all'
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=30) as resp:
                    if resp.status != 200:
                        continue
                    data = await resp.json()
                    securities = data.get('securities', {}).get('data', [])
                    columns = data.get('securities', {}).get('columns', [])
                    df = pd.DataFrame(securities, columns=columns)
                    # Фильтруем по SECID, содержащему base_secid (без учёта регистра)
                    df = df[df['SECID'].str.upper().str.contains(base_secid.upper())]
                    if not df.empty:
                        df = df[['SECID', 'SHORTNAME', 'LASTDELDATE']]
                        df.rename(columns={'SECID': 'secid', 'SHORTNAME': 'shortname', 'LASTDELDATE': 'matdate'}, inplace=True)
                        df['matdate'] = pd.to_datetime(df['matdate'])
                        df = df.sort_values('matdate')
                        return df
        return pd.DataFrame()
            
    def get_futures_list(self, base_secid: str) -> pd.DataFrame:
        """Синхронная обёртка для fetch_futures_list_async."""
        return asyncio.run(self.fetch_futures_list_async(base_secid))

    # ---------- Универсальный метод загрузки актива ----------

    def load_asset(self, secid: str, asset_type: str, from_date: str, till_date: str, interval: int = 1) -> pd.DataFrame:
        # Приводим тип к каноническому виду
        if asset_type in ('stock', 'stocks'):
            asset_type = 'stock'
        elif asset_type in ('index', 'indices'):
            asset_type = 'index'
        elif asset_type in ('futures',):
            asset_type = 'futures'
        else:
            raise ValueError(f"Неизвестный тип актива: {asset_type}")

        # Загружаем данные через fetch_candles для всех типов
        df = self.fetch_candles(secid, asset_type, from_date, till_date, interval)

        if df.empty:
            logger.warning(f"Нет данных для {secid} ({asset_type})")
            return df

        # Сохраняем в БД
        if asset_type == 'futures':
            # Для фьючерсов сохраняем в raw_candles (как и для всех)
            self.db.insert_raw_candles(df)
            self.db.upsert_meta(secid, df['ts'].max(), 'futures', len(df))
        else:
            self.db.insert_raw_candles(df)
            self.db.upsert_meta(secid, df['ts'].max(), asset_type, len(df))

        logger.info(f"Загружено {len(df)} свечей для {secid} ({asset_type})")
        return df