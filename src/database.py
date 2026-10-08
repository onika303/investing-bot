import sqlite3
import pandas as pd
from typing import Optional, List, Tuple
from loguru import logger


class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Создаёт таблицы, если их нет."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Таблица для сырых свечей
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS raw_candles (
                    secid TEXT NOT NULL,
                    ts INTEGER NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    oi INTEGER DEFAULT 0,
                    PRIMARY KEY (secid, ts)
                )
            ''')
            # Таблица для непрерывных рядов
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS continuous_ohlc (
                    secid TEXT NOT NULL,
                    ts INTEGER NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    oi INTEGER DEFAULT 0,
                    PRIMARY KEY (secid, ts)
                )
            ''')
            # Таблица метаданных
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS meta (
                    secid TEXT PRIMARY KEY,
                    last_loaded_ts INTEGER,
                    last_updated TEXT,
                    total_rows INTEGER,
                    source TEXT
                )
            ''')
            # Индексы для ускорения
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_raw_secid_ts ON raw_candles(secid, ts)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_cont_secid_ts ON continuous_ohlc(secid, ts)')
            conn.commit()
            logger.info("База данных инициализирована (таблицы созданы)")

    def _get_connection(self):
        """Возвращает соединение с БД (как контекстный менеджер)."""
        return sqlite3.connect(self.db_path)

    def execute(self, query: str, params: tuple = ()):
        """Выполняет запрос без возврата данных."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()

    def executemany(self, query: str, params_list: List[tuple]):
        """Массовая вставка."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, params_list)
            conn.commit()

    def fetch_all(self, query: str, params: tuple = ()) -> List[tuple]:
        """Возвращает все строки."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchall()

    def fetch_one(self, query: str, params: tuple = ()) -> Optional[tuple]:
        """Возвращает одну строку."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchone()

    def insert_raw_candles(self, df: pd.DataFrame):
        """Вставляет DataFrame с колонками: secid, ts, open, high, low, close, volume, oi."""
        if df.empty:
            return
        # Проверяем наличие колонок
        required = {'secid', 'ts', 'open', 'high', 'low', 'close', 'volume'}
        if not required.issubset(df.columns):
            raise ValueError(f"DataFrame должен содержать колонки: {required}")
        # Если oi нет, добавляем со значением 0
        if 'oi' not in df.columns:
            df['oi'] = 0
        # Преобразуем в список кортежей
        rows = list(df[['secid', 'ts', 'open', 'high', 'low', 'close', 'volume', 'oi']].itertuples(index=False, name=None))
        query = '''
            INSERT OR REPLACE INTO raw_candles (secid, ts, open, high, low, close, volume, oi)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        '''
        self.executemany(query, rows)
        logger.info(f"Вставлено {len(rows)} записей в raw_candles")

    def insert_continuous_candles(self, df: pd.DataFrame):
        """Аналогично для continuous_ohlc."""
        if df.empty:
            return
        required = {'secid', 'ts', 'open', 'high', 'low', 'close', 'volume'}
        if not required.issubset(df.columns):
            raise ValueError(f"DataFrame должен содержать колонки: {required}")
        if 'oi' not in df.columns:
            df['oi'] = 0
        rows = list(df[['secid', 'ts', 'open', 'high', 'low', 'close', 'volume', 'oi']].itertuples(index=False, name=None))
        query = '''
            INSERT OR REPLACE INTO continuous_ohlc (secid, ts, open, high, low, close, volume, oi)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        '''
        self.executemany(query, rows)
        logger.info(f"Вставлено {len(rows)} записей в continuous_ohlc")

    def get_last_ts(self, secid: str, table='raw_candles') -> Optional[int]:
        """Возвращает максимальный timestamp для указанного secid в таблице."""
        query = f"SELECT MAX(ts) FROM {table} WHERE secid = ?"
        result = self.fetch_one(query, (secid,))
        return result[0] if result and result[0] is not None else None

    def upsert_meta(self, secid: str, last_ts: int, source: str, total_rows: int = 0):
        """Обновляет или вставляет запись в таблицу meta."""
        from datetime import datetime
        now = datetime.utcnow().isoformat()
        query = '''
            INSERT INTO meta (secid, last_loaded_ts, last_updated, total_rows, source)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(secid) DO UPDATE SET
                last_loaded_ts = excluded.last_loaded_ts,
                last_updated = excluded.last_updated,
                total_rows = excluded.total_rows,
                source = excluded.source
        '''
        self.execute(query, (secid, last_ts, now, total_rows, source))
        logger.info(f"Мета-информация обновлена для {secid}")