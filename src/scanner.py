"""Инвестиционный сканер: индикаторы + дивергенции + правила -> сигналы со скорингом."""
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from src.database import DatabaseManager
from src.indicators import IndicatorCalculator
from src.divergence import DivergenceDetector


class InvestmentScanner:
    """Сканирует один актив по дневным свечам из БД и выдаёт список сигналов."""

    MIN_ROWS = 60  # минимум свечей для осмысленного расчёта индикаторов

    def __init__(self, db: DatabaseManager):
        self.db = db

    # ---------- данные ----------
    def load_candles(self, secid: str, limit_rows: int = 500) -> Optional[pd.DataFrame]:
        rows = self.db.fetch_all(
            """SELECT ts, open, high, low, close, volume, oi
               FROM raw_candles WHERE secid = ?
               ORDER BY ts DESC LIMIT ?""",
            (secid, limit_rows),
        )
        if len(rows) < self.MIN_ROWS:
            return None
        df = pd.DataFrame(rows, columns=['ts', 'open', 'high', 'low', 'close', 'volume', 'oi'])
        df = df.sort_values('ts').reset_index(drop=True)
        df['date'] = pd.to_datetime(df['ts'], unit='s')
        return df

    # ---------- правила поверх индикаторов ----------
    @staticmethod
    def _rule_signals(df: pd.DataFrame) -> List[Dict]:
        sig: List[Dict] = []
        last = df.iloc[-1]
        prev = df.iloc[-2]

        # Пересечение EMA12/EMA26
        if {'ema_12', 'ema_26'} <= set(df.columns):
            if prev['ema_12'] <= prev['ema_26'] and last['ema_12'] > last['ema_26']:
                sig.append({'rule': 'EMA golden cross', 'type': 'buy', 'weight': 25})
            elif prev['ema_12'] >= prev['ema_26'] and last['ema_12'] < last['ema_26']:
                sig.append({'rule': 'EMA death cross', 'type': 'sell', 'weight': 25})

        # RSI-зоны
        rsi_col = 'rsi_14' if 'rsi_14' in df.columns else None
        if rsi_col:
            if last[rsi_col] < 30:
                sig.append({'rule': f"RSI oversold ({last[rsi_col]:.1f})", 'type': 'buy', 'weight': 15})
            elif last[rsi_col] > 70:
                sig.append({'rule': f"RSI overbought ({last[rsi_col]:.1f})", 'type': 'sell', 'weight': 15})

        # Bollinger breakout
        if {'bb_lower', 'bb_upper'} <= set(df.columns):
            if last['close'] < last['bb_lower']:
                sig.append({'rule': 'Close below lower BB', 'type': 'buy', 'weight': 10})
            elif last['close'] > last['bb_upper']:
                sig.append({'rule': 'Close above upper BB', 'type': 'sell', 'weight': 10})

        # ADX — сила тренда (не направление, усиливает существующий сигнал)
        if 'adx_14' in df.columns and last['adx_14'] > 25:
            sig.append({'rule': f"Strong trend ADX={last['adx_14']:.0f}", 'type': 'boost', 'weight': 10})

        return sig

    # ---------- основной метод ----------
    def scan_asset(self, secid: str, asset_type: str) -> Optional[Dict]:
        df = self.load_candles(secid)
        if df is None:
            return None

        df = IndicatorCalculator(df).compute_all()
        detector = DivergenceDetector(df)
        divergences = detector.scan_all_indicators(lookback=50)

        signals = self._rule_signals(df)
        for ind_name, divs in divergences.items():
            for d in divs:
                signals.append({
                    'rule': f"{ind_name.upper()} {d.get('kind', 'divergence')}",
                    'type': d.get('type', 'watch'),
                    'weight': 30,
                })

        score_buy = sum(s['weight'] for s in signals if s['type'] == 'buy')
        score_sell = sum(s['weight'] for s in signals if s['type'] == 'sell')
        boost = sum(s['weight'] for s in signals if s['type'] == 'boost')
        score_buy += boost
        score_sell += boost

        atr = float(df['atr_14'].iloc[-1]) if 'atr_14' in df.columns else None
        price = float(df['close'].iloc[-1])
        result = {
            'secid': secid,
            'asset_type': asset_type,
            'date': str(df['date'].iloc[-1].date()),
            'price': round(price, 4),
            'signals': signals,
            'score_buy': score_buy,
            'score_sell': score_sell,
            'verdict': ('buy' if score_buy > score_sell and score_buy >= 30
                        else 'sell' if score_sell > score_buy and score_sell >= 30
                        else 'watch' if signals else 'neutral'),
        }
        if atr:
            result['sl_tp'] = {
                'stop_loss': round(price - 2 * atr, 4) if result['verdict'] == 'buy' else round(price + 2 * atr, 4),
                'take_profit': round(price + 3 * atr, 4) if result['verdict'] == 'buy' else round(price - 3 * atr, 4),
            }
        return result

    def scan_all(self, config_assets: Dict[str, List[str]]) -> List[Dict]:
        results = []
        no_data = []
        for asset_type, secids in config_assets.items():
            for secid in secids:
                try:
                    r = self.scan_asset(secid, asset_type)
                    if r is None:
                        no_data.append(secid)
                    else:
                        results.append(r)
                except Exception as e:
                    logger.error(f"Ошибка скана {secid}: {e}")
                    no_data.append(secid)
        if no_data:
            logger.warning(f"Без данных (нет свечей в БД): {len(no_data)} активов, напр. {no_data[:5]}")
        results.sort(key=lambda x: max(x['score_buy'], x['score_sell']), reverse=True)
        return results
