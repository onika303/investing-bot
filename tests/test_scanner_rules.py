"""Юнит-тесты правил скоринга и verdict-логики InvestmentScanner."""
import pandas as pd
import pytest

from src.scanner import InvestmentScanner


def sig(prev=None, last=None):
    """DataFrame из двух строк (prev, last) со значениями индикаторов по умолчанию.

    Набор EMA — 13/21/50/100 (требование сканера); по умолчанию цена стоит ровно
    на всех EMA и нейтральна по остальным правилам.
    """
    base = {'ema_13': 100.0, 'ema_21': 100.0, 'ema_50': 100.0, 'ema_100': 100.0,
            'rsi_14': 50.0,
            'bb_lower': 90.0, 'bb_upper': 110.0, 'adx_14': 15.0, 'close': 100.0}
    row_prev = dict(base); row_prev.update(prev or {})
    row_last = dict(base); row_last.update(last or {})
    return pd.DataFrame([row_prev, row_last])


def test_golden_cross_13x21(scanner_rules_only):
    s = scanner_rules_only
    df = sig(prev={'ema_13': 99}, last={'ema_13': 101})
    out = s._rule_signals(df)
    assert {'rule': 'EMA 13x21 golden cross', 'type': 'buy', 'weight': 25} in out


def test_death_cross_21x50(scanner_rules_only):
    s = scanner_rules_only
    df = sig(prev={'ema_21': 101}, last={'ema_21': 99})
    out = s._rule_signals(df)
    assert any(o['rule'] == 'EMA 21x50 death cross' and o['type'] == 'sell'
               and o['weight'] == 20 for o in out)


def test_trend_confluence(scanner_rules_only):
    """Цена выше всех EMA 13/21/50/100 -> конфлюэнция buy (+15)."""
    s = scanner_rules_only
    out = s._rule_signals(sig(last={'close': 130}))
    assert any(o['rule'] == 'Close above EMA 13/21/50/100' and o['type'] == 'buy'
               and o['weight'] == 15 for o in out)
    out = s._rule_signals(sig(last={'close': 70}))
    assert any(o['rule'] == 'Close below EMA 13/21/50/100' and o['type'] == 'sell'
               for o in out)


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
    rows = [('TINY', 1700000000 + i * 86400, 24, 100, 101, 99, 100, 10, 0) for i in range(30)]
    tmp_db.executemany("INSERT INTO raw_candles VALUES (?,?,?,?,?,?,?,?,?)", rows)
    s = InvestmentScanner(tmp_db, short_restrictions={})
    assert s.load_candles('TINY') is None
    assert s.scan_asset('TINY', 'stocks') is None


def test_scan_all_sorting_and_no_data(tmp_db, v_shape_df):
    """scan_all: сортировка по силе, отсутствующие активы не падают."""
    rows = [('GAS', int(r.ts), 24, r.open, r.high, r.low, r.close, int(r.volume), 0)
            for r in v_shape_df.itertuples()]
    tmp_db.executemany("INSERT OR REPLACE INTO raw_candles VALUES (?,?,?,?,?,?,?,?,?)", rows)
    s = InvestmentScanner(tmp_db, short_restrictions={})
    res = s.scan_all({'stocks': ['GAS', 'MISSING']})
    assert len(res) == 1
    assert res[0]['secid'] == 'GAS'
    assert set(res[0]) >= {'verdict', 'score_buy', 'score_sell', 'trade_access'}


def test_result_has_strength_and_tf_scores(tmp_db=None):
    """scan_asset отдаёт поле 'horizons' с отдельным verdict/strength на каждый ТФ;
    плоские поля = основной горизонт (1Д). Смешивания очков между ТФ нет."""
    import sqlite3
    from src.database import DatabaseManager
    from src.scanner import InvestmentScanner

    con = sqlite3.connect(':memory:')
    db = DatabaseManager.__new__(DatabaseManager)
    db.db_path = ':memory:'
    db._conn = con
    # схема совпадает с src/database.py: (secid, ts, interval, open, high, low, close, volume, oi)
    con.execute("CREATE TABLE raw_candles (secid TEXT, ts INTEGER, interval INT DEFAULT 24,"
                " open REAL, high REAL, low REAL, close REAL, volume REAL, oi INT)")
    # ровный растущий ряд 1Д + флэт 1Ч — индикаторы считаются стабильно
    rows = []
    for i in range(250):
        c = 100 + i * 0.5
        rows.append(("TEST", 1_600_000_000 + i * 86400, 24, c * 0.99, c * 1.01, c, c, 1_000_000, 0))
    for i in range(200):
        c = 100 + 249 * 0.5
        rows.append(("TEST", 1_740_000_000 + i * 3600, 60, c * 0.995, c * 1.005, c, c, 100_000, 0))
    con.executemany("INSERT INTO raw_candles VALUES (?,?,?,?,?,?,?,?,?)", rows)
    db.fetch_all = lambda q, p=(): list(con.execute(q, p))

    sc = InvestmentScanner(db, short_restrictions={})
    sc._fetch_issue_info = lambda secid: {}  # без сети
    r = sc.scan_asset("TEST", "stocks")
    assert r is not None
    assert set(r['horizons']) == {'1d', '1h'}
    for tf, h in r['horizons'].items():
        # сила считается строго внутри горизонта
        assert h['strength'] == round(max(h['score_buy'], h['score_sell']), 1)
        assert h['horizon'] == sc.HORIZON[tf]
        assert all(s['tf'] == sc.TF_LABEL[tf] for s in h['signals'])
    # сводные поля = дневной горизонт, без примеси часовика
    hd = r['horizons']['1d']
    assert r['verdict'] == hd['verdict']
    assert r['score_buy'] == hd['score_buy'] and r['score_sell'] == hd['score_sell']
    assert r['strength'] == hd['strength']
