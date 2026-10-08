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
                # ISS отдаёт 'total' только с iss.meta=on; при meta=off его нет —
                # используем len(rows) как нижнюю границу (логика пагинации это учитывает)
                total = candles_block.get('total') or len(candles)
                return {'candles': candles, 'total': total}
        except Exception as e:
            logger.error(f"Ошибка загрузки {secid}: {e}")
            return {'candles': [], 'total': 0}

    # ---------- Основной метод загрузки свечей ----------

    async def fetch_candles_async(self, secid: str, asset_type: str, from_date: str, till_date: str, interval: int = 1, end_on_load: bool = False) -> pd.DataFrame:
        """
        Асинхронно загружает свечи за период с пагинацией.
        Возвращает DataFrame с колонками: secid, ts, open, high, low, close, volume, oi.

        end_on_load=True — идти от конца периода (свежие данные). ISS отдаёт не
        более 500 записей и отсекает окно по 'from', поэтому без этого флага
        запрос за 3 года возвращает только старые свечи (~2 года назад).
        """
        url = self._build_url(secid, asset_type, from_date, till_date, interval)
        all_candles = []
        page_size = 500  # жёсткий лимит ISS на размер страницы

        async with aiohttp.ClientSession() as session:
            first_page = await self._fetch_page(session, url, 0, secid)
            if not first_page['candles']:
                logger.warning(f"Нет данных для {secid} за период {from_date} - {till_date}")
                return pd.DataFrame()
            total = first_page['total']

            if end_on_load:
                # идём от конца окна к началу. 'total' может отсутствовать (iss.meta=off)
                # или быть заниженным (= размер страницы), поэтому определяем границу
                # по факту: первая неполная страница = конец данных в окне.
                st = 0
                while True:
                    page = await self._fetch_page(session, url, st, secid)
                    rows = page['candles']
                    if not rows:
                        break
                    all_candles.extend(rows)
                    if len(rows) < page_size:
                        break  # дошли до конца доступного окна
                    st += page_size
            else:
                all_candles.extend(first_page['candles'])
                start = 0
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
        # MOEX отдаёт begin в формате 'YYYY-MM-DDTHH:MM:SS+03:00' (MSK).
        # Дневная свеча = торговый день по Москве: нормализуем ts к полуночи MSK,
        # иначе PK (secid, ts) даёт двойные записи на границе UTC/MSK.
        ts_dt = pd.to_datetime(df['ts_iso'], format='ISO8601', utc=True)
        if interval == 24:
            ts_dt = ts_dt.dt.tz_convert('Europe/Moscow').dt.normalize().dt.tz_convert('UTC')
        df['ts'] = ts_dt.astype('int64') // 10**9
        df['secid'] = secid
        if 'oi' in df.columns:
            df['oi'] = df['oi'].fillna(0).astype(int)
        else:
            df['oi'] = 0

        df = df[['secid', 'ts', 'open', 'high', 'low', 'close', 'volume', 'oi']]
        df['interval'] = int(interval)
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(int)
        df['oi'] = df['oi'].astype(int)

        return df

    def fetch_candles(self, secid: str, asset_type: str, from_date: str, till_date: str, interval: int = 1, end_on_load: bool = False) -> pd.DataFrame:
        """Синхронная обёртка для fetch_candles_async."""
        return asyncio.run(self.fetch_candles_async(secid, asset_type, from_date, till_date, interval, end_on_load=end_on_load))

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

    def load_asset(self, secid: str, asset_type: str, from_date: str, till_date: str, interval: int = 1, end_on_load: bool = False) -> pd.DataFrame:
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
        df = self.fetch_candles(secid, asset_type, from_date, till_date, interval, end_on_load=end_on_load)

        if df.empty:
            logger.warning(f"Нет данных для {secid} ({asset_type})")
            return df

        # Сохраняем в БД (interval уже в df — попадёт в PK (secid, ts, interval))
        self.db.insert_raw_candles(df)
        self.db.upsert_meta(secid, int(df['ts'].max()), asset_type, len(df))

        logger.info(f"Загружено {len(df)} свечей для {secid} ({asset_type})")
        return df