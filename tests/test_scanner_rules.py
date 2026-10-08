"""Юнит-тесты правил скоринга и verdict-логики InvestmentScanner."""
import pandas as pd
import pytest

from src.scanner import InvestmentScanner


def sig(prev=None, last=None):
    """DataFrame из двух строк (prev, last) со значениями индикаторов по умолчанию."""
    base = {'ema_12': 100.0, 'ema_26': 100.0, 'rsi_14': 50.0,
            'bb_lower': 90.0, 'bb_upper': 110.0, 'adx_14': 15.0, 'close': 100.0}
    row_prev = dict(base); row_prev.update(prev or {})
    row_last = dict(base); row_last.update(last or {})
    return pd.DataFrame([row_prev, row_last])


def test_golden_cross(scanner_rules_only):
    s = scanner_rules_only
    df = sig(prev={'ema_12': 99}, last={'ema_12': 101})
    out = s._rule_signals(df)
    assert {'rule': 'EMA golden cross', 'type': 'buy', 'weight': 25} in out


def test_death_cross(scanner_rules_only):
    s = scanner_rules_only
    df = sig(prev={'ema_12': 101}, last={'ema_12': 99})
    out = s._rule_signals(df)
    assert any(o['rule'] == 'EMA death cross' and o['type'] == 'sell' for o in out)


def test_rsi_zones(scanner_rules_only):
    s = scanner_rules_only
    assert any(o['type'] == 'buy' and 'RSI oversold' in o['rule'] for o in s._rule_signals(sig(last={'rsi_14': 25})))
    assert any(o['type'] == 'sell' and 'RSI overbought' in o['rule'] for o in s._rule_signals(sig(last={'rsi_14': 75})))
    assert all('RSI' not in o['rule'] for o in s._rule_signals(sig()))


def test_bb_breakout(scanner_rules_only):
    s = scanner_rules_only
    assert any(o['rule'] == 'Close below lower BB' and o['type'] == 'buy' for o in s._rule_signals(sig(last={'close': 89})))
    assert any(o['rule'] == 'Close above upper BB' and o['type'] == 'sell' for o in s._rule_signals(sig(last={'close': 111})))


def test_adx_boost(scanner_rules_only):
    s = scanner_rules_only
    out = s._rule_signals(sig(last={'adx_14': 30}))
    assert any(o['type'] == 'boost' and o['weight'] == 10 for o in out)


def test_no_data_returns_none(tmp_db):
    s = InvestmentScanner(tmp_db, short_restrictions={})
    assert s.scan_asset('EMPTY', 'stocks') is None


def test_min_rows_guard(tmp_db):
    """Меньше MIN_ROWS свечей -> None (не сканируем мусор)."""
    rows = [('TINY', 1700000000 + i * 86400, 100, 101, 99, 100, 10, 0) for i in range(30)]
    tmp_db.executemany("INSERT INTO raw_candles VALUES (?,?,?,?,?,?,?,?)", rows)
    s = InvestmentScanner(tmp_db, short_restrictions={})
    assert s.load_candles('TINY') is None
    assert s.scan_asset('TINY', 'stocks') is None


def test_scan_all_sorting_and_no_data(tmp_db, v_shape_df):
    """scan_all: сортировка по силе, отсутствующие активы не падают."""
    rows = [('GAS', int(r.ts), r.open, r.high, r.low, r.close, int(r.volume), 0)
            for r in v_shape_df.itertuples()]
    tmp_db.executemany("INSERT OR REPLACE INTO raw_candles VALUES (?,?,?,?,?,?,?,?)", rows)
    s = InvestmentScanner(tmp_db, short_restrictions={})
    res = s.scan_all({'stocks': ['GAS', 'MISSING']})
    assert len(res) == 1
    assert res[0]['secid'] == 'GAS'
    assert set(res[0]) >= {'verdict', 'score_buy', 'score_sell', 'trade_access'}
