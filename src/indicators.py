import pandas as pd
import numpy as np
from loguru import logger


class IndicatorCalculator:
    def __init__(self, df: pd.DataFrame):
        """
        df должен содержать колонки: open, high, low, close, volume (и oi, если есть)
        Все расчёты производятся на копии DataFrame, исходный не меняется.
        """
        self.df = df.copy()
        # Убедимся, что данные отсортированы по времени
        if 'ts' in self.df.columns:
            self.df = self.df.sort_values('ts').reset_index(drop=True)

    # ---------- Экспоненциальная скользящая средняя ----------
    def ema(self, period: int, column: str = 'close') -> pd.Series:
        return self.df[column].ewm(span=period, adjust=False).mean()

    # ---------- RSI ----------
    def rsi(self, period: int = 14, column: str = 'close') -> pd.Series:
        delta = self.df[column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    # ---------- MACD ----------
    def macd(self, fast: int = 12, slow: int = 26, signal: int = 9, column: str = 'close') -> pd.DataFrame:
        ema_fast = self.ema(fast, column)
        ema_slow = self.ema(slow, column)
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return pd.DataFrame({
            'macd_line': macd_line,
            'signal_line': signal_line,
            'histogram': histogram
        })

    # ---------- OBV (On-Balance Volume) ----------
    def obv(self, column: str = 'close', volume_col: str = 'volume') -> pd.Series:
        # Вычисляем направление: +1 если close > previous close, -1 если <, 0 если =
        direction = np.sign(self.df[column].diff())
        # OBV = накопленный объём с учётом знака
        obv = (direction * self.df[volume_col]).cumsum()
        return obv

    # ---------- Полосы Боллинджера ----------
    def bollinger_bands(self, period: int = 20, std: float = 2.0, column: str = 'close') -> pd.DataFrame:
        middle = self.df[column].rolling(window=period).mean()
        std_dev = self.df[column].rolling(window=period).std()
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        return pd.DataFrame({
            'bb_middle': middle,
            'bb_upper': upper,
            'bb_lower': lower,
            'bb_width': upper - lower,
            'bb_position': (self.df[column] - lower) / (upper - lower)  # где находится цена внутри полос (0..1)
        })

    # ---------- ATR (Average True Range) ----------
    def atr(self, period: int = 14) -> pd.Series:
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr

    # ---------- ADX (Average Directional Index) - упрощённая версия ----------
    def adx(self, period: int = 14) -> pd.Series:
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']

        # True Range
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        # Directional Movement
        up_move = high - high.shift()
        down_move = low.shift() - low
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # Smooth DM and TR
        plus_di = 100 * (pd.Series(plus_dm).rolling(window=period).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm).rolling(window=period).mean() / atr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        return adx

    # ---------- Главный метод: рассчитывает все индикаторы и возвращает DataFrame ----------
    def compute_all(self) -> pd.DataFrame:
        """
        Добавляет все индикаторы к DataFrame и возвращает его.
        """
        result = self.df.copy()

        # EMA
        for period in [12, 24, 50, 200]:
            result[f'ema_{period}'] = self.ema(period)

        # RSI
        result['rsi_14'] = self.rsi(14)

        # MACD
        macd_df = self.macd()
        result['macd_line'] = macd_df['macd_line']
        result['macd_signal'] = macd_df['signal_line']
        result['macd_hist'] = macd_df['histogram']

        # OBV
        result['obv'] = self.obv()

        # Bollinger Bands
        bb_df = self.bollinger_bands()
        for col in ['bb_middle', 'bb_upper', 'bb_lower', 'bb_width', 'bb_position']:
            result[col] = bb_df[col]

        # ATR
        result['atr_14'] = self.atr(14)

        # ADX
        result['adx_14'] = self.adx(14)

        return result