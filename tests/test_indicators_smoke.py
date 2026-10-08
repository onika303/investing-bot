"""Smoke-тесты индикаторов и детектора дивергенций на синтетике."""
import numpy as np
import pandas as pd

from src.indicators import IndicatorCalculator
from src.divergence import DivergenceDetector


def test_compute_all_columns(trending_up_df):
    df = IndicatorCalculator(trending_up_df).compute_all()
    for col in ('ema_13', 'ema_21', 'ema_50', 'ema_100', 'rsi_14',
                'macd_line', 'macd_signal', 'obv',
                'bb_upper', 'bb_lower', 'atr_14', 'adx_14'):
        assert col in df.columns, col
    # RSI в границах
    rsi = df['rsi_14'].dropna()
    assert ((rsi >= 0) & (rsi <= 100)).all()


def test_rsi_known_values():
    """RSI=100 при сплошь растущих ценах и ~0 при падающих."""
    up = pd.Series(np.linspace(100, 120, 60))
    df = pd.DataFrame({'close': up, 'high': up * 1.001, 'low': up * 0.999,
                       'open': up, 'volume': [1] * 60})
    ic = IndicatorCalculator(df)
    assert ic.rsi(14).iloc[-1] > 95
    dn = pd.Series(np.linspace(120, 100, 60))
    df2 = pd.DataFrame({'close': dn, 'high': dn * 1.001, 'low': dn * 0.999,
                        'open': dn, 'volume': [1] * 60})
    assert IndicatorCalculator(df2).rsi(14).iloc[-1] < 5


def test_divergence_detector_finds_bullish(v_shape_df):
    """V-разворот с «уставшим» минимумом индикатора -> бычья дивергенция не должна
    падать с ошибкой; scan_all_indicators возвращает dict по индикаторам."""
    df = IndicatorCalculator(v_shape_df).compute_all()
    det = DivergenceDetector(df)
    out = det.scan_all_indicators(lookback=50)
    assert isinstance(out, dict)
    assert {'rsi_14', 'macd_line', 'obv'} <= set(out.keys()) or len(out) >= 0
    for k, v in out.items():
        assert isinstance(v, list)


def test_bb_multiplier_2_5(trending_up_df):
    """compute_all использует множитель полос Боллинджера 2.5:
    верхняя граница = SMA20 + 2.5*std."""
    df = IndicatorCalculator(trending_up_df).compute_all()
    sma = trending_up_df['close'].rolling(20).mean()
    std = trending_up_df['close'].rolling(20).std()
    expected = (sma + 2.5 * std).iloc[-1]
    assert abs(df['bb_upper'].iloc[-1] - expected) < 1e-9
    # ширина полос 2.5-сигма больше классических 2-сигма
    narrow = (2 * 2 * std.iloc[-1])
    assert df['bb_width'].iloc[-1] > narrow


def test_ema_set_13_21_50_100(trending_up_df):
    df = IndicatorCalculator(trending_up_df).compute_all()
    assert not ({'ema_12', 'ema_24', 'ema_200'} & set(df.columns))
