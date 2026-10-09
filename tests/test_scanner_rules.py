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
            'bb_lower': 90.0, 'bb_middle': 100.0, 'bb_upper': 110.0,
            'adx_14': 15.0, 'close': 100.0}
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
    """Цена выше всех EMA 13/21/50/100 -> направленность buy (+W_EMA_CONFLUENCE)."""
    s = scanner_rules_only
    # bb_middle=105, чтобы close=130 не попадал в зону BB-касания
    out = s._rule_signals(sig(last={'close': 130, 'bb_middle': 105}))
    assert any(o['rule'] == 'Close above EMA 13/21/50/100' and o['type'] == 'buy'
               and o['weight'] == InvestmentScanner.W_EMA_CONFLUENCE for o in out)
    # bb_middle=95, чтобы close=70 не попадал в зону BB-касания
    out = s._rule_signals(sig(last={'close': 70, 'bb_middle': 95}))
    assert any(o['rule'] == 'Close below EMA 13/21/50/100' and o['type'] == 'sell'
               for o in out)


def test_rsi_zones(scanner_rules_only):
    s = scanner_rules_only
    assert any(o['type'] == 'buy' and 'RSI oversold' in o['rule'] for o in s._rule_signals(sig(last={'rsi_14': 25})))
    assert any(o['type'] == 'sell' and 'RSI overbought' in o['rule'] for o in s._rule_signals(sig(last={'rsi_14': 75})))
    assert all('RSI' not in o['rule'] for o in s._rule_signals(sig()))


def test_bb_breakout(scanner_rules_only):
    """Закрытие у границ Bollinger Bands (σ=2.5): нижняя -> buy, верхняя -> sell."""
    s = scanner_rules_only
    out = s._rule_signals(sig(last={'close': 89}))   # ниже mid-0.95*(mid-lower)
    assert any(o['rule'] == 'Close at lower BB (2.5σ)' and o['type'] == 'buy'
               and o['weight'] == InvestmentScanner.W_BB_EDGE for o in out)
    out = s._rule_signals(sig(last={'close': 111}))
    assert any(o['rule'] == 'Close at upper BB (2.5σ)' and o['type'] == 'sell'
               for o in out)


# ---------- приоритет критериев (решение пользователя от 10.10.2026) ----------
def _mk(rule, typ, weight):
    return {'rule': rule, 'type': typ, 'weight': weight}


def test_group_signals_steps(scanner_rules_only):
    """Каждый тип сигнала попадает в свой шаг приоритета; прочее — aux."""
    s = scanner_rules_only
    sigs = [_mk('RSI bullish divergence', 'buy', 40),
            _mk('Close above EMA 13/21/50/100', 'buy', 10),
            _mk('Close at lower BB (2.5σ)', 'buy', 30),
            _mk('EMA 13x21 golden cross', 'buy', 25),
            _mk('RSI < 20 (15.0)', 'buy', 35)]
    g = s._group_signals(sigs)
    assert g['divergence'] == [sigs[0]]
    assert g['ema_trend'] == [sigs[1]]
    assert g['bb'] == [sigs[2]]
    assert g['ema_cross'] == [sigs[3]]
    assert sigs[4].get('aux') is True


def test_first_step_wins_over_opposite_later_step(scanner_rules_only):
    """Шаг 1 дал sell (дивергенция), шаг 2 — противоположный buy:
    вердикт остаётся sell (приоритет первого сигнала), оба сигнала видны."""
    s = scanner_rules_only
    sigs = [_mk('MACD histogram bearish divergence', 'sell', 40),
            _mk('Close above EMA 13/21/50/100', 'buy', 10)]
    g = s._group_signals(sigs)
    pri = s._prioritized(g, [])
    assert pri['dir'] == 'sell' and pri['step'] == 'divergence'
    assert pri['score_dir'] == 40          # сила продажного сигнала
    assert pri['score_opp'] == 10           # контра-сигнал не отменяет, а учтён
    assert pri['conflicting'] == [sigs[1]]  # оба сигнала попадут в отчёт


def test_same_direction_signals_accumulate(scanner_rules_only):
    """Сигналы одного направления на разных шагах складываются в силу."""
    s = scanner_rules_only
    sigs = [_mk('RSI bearish divergence', 'sell', 40),
            _mk('Close below EMA 13/21/50/100', 'sell', 10),
            _mk('Close at upper BB (2.5σ)', 'sell', 30),
            _mk('EMA 13x21 death cross', 'sell', 25)]
    g = s._group_signals(sigs)
    pri = s._prioritized(g, [])
    assert pri['dir'] == 'sell' and pri['score_dir'] == 105
    assert pri['conflicting'] == []


def test_aux_extends_primary_direction(scanner_rules_only):
    """Вспомогательный сигнал (RSI-экстремум) усиливает направление первого шага,
    но не может задать его сам при отсутствии приоритетных сигналов — тоже задаёт
    (шаги пустые -> aux становится первичным)."""
    s = scanner_rules_only
    sigs = [_mk('RSI > 80 (85.0)', 'sell', 35)]
    g = s._group_signals(sigs)
    aux = [x for x in sigs if x.get('aux')]
    pri = s._prioritized(g, aux)
    assert pri['dir'] == 'sell' and pri['step'] == 'aux'

    sigs2 = [_mk('RSI bullish divergence', 'buy', 40),
             _mk('RSI < 20 (15.0)', 'buy', 35)]
    g2 = s._group_signals(sigs2)
    aux2 = [x for x in sigs2 if x.get('aux')]
    pri2 = s._prioritized(g2, aux2)
    assert pri2['dir'] == 'buy' and pri2['step'] == 'divergence'
    assert pri2['score_dir'] == 75


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
