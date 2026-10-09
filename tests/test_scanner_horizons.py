"""Интеграционные тесты сканера по ГОРИЗОНТАМ (1Ч = 1–2 дня, 1Д = внутри недели).

Подход (решение от 10.10.2026): вместо хрупкой подгонки синтетики под граничные
значения индикаторов тесты ПОДМЕНЫВАЮТ расчёт индикаторов (scanner.compute_indicators)
детерминированным DataFrame с известными значениями EMA/RSI/MACD/BB/ADX.
Проверяется именно ЛОГИКА ПРАВИЛ и раздельность горизонтов, а не численные значения индикаторов.
"""
import numpy as np
import pandas as pd
import pytest

from src.scanner import InvestmentScanner


# ---------- конструктор «сырых» свечей в БД (без сети) ----------

def raw_df(n=120, price=100.0):
    """Плоский ряд OHLCV — содержимое не важно: индикаторы подменяются."""
    dates = pd.bdate_range('2025-01-01', periods=n)
    c = np.full(n, price)
    return pd.DataFrame({
        'ts': dates.astype('int64') // 10**9,
        'open': c, 'high': c * 1.005, 'low': c * 0.995, 'close': c,
        'volume': 1000, 'oi': 0, 'date': dates})


def ind_df(last: dict, prev: dict | None = None, n: int = 120):
    """DataFrame «как после compute_all»: две последние строки задают правила."""
    base = {'ema_13': 100.0, 'ema_21': 100.0, 'ema_50': 100.0, 'ema_100': 100.0,
            'rsi_14': 50.0, 'macd_line': 0.0, 'macd_signal': 0.0, 'macd_hist': 0.0,
            'bb_middle': 100.0, 'bb_upper': 110.0, 'bb_lower': 90.0,
            'adx_14': 15.0, 'atr_14': 2.0, 'close': 100.0}
    rows = [dict(base) for _ in range(n)]
    rows[-2] = {**base, **(prev or {})}
    rows[-1] = {**base, **last}
    df = pd.DataFrame(rows)
    df['date'] = pd.bdate_range('2025-01-01', periods=n)
    df['ts'] = df['date'].astype('int64') // 10**9
    return df


@pytest.fixture
def scanner(tmp_db):
    s = InvestmentScanner(tmp_db, short_restrictions={})
    s._fetch_issue_info = lambda secid: {'issuedate': '2015-01-01'}  # без сети
    return s


def inject(scanner, frames_by_step):
    """compute_indicators возвращает готовый DataFrame в зависимости от интервала
    (различаем ТФ по шагу ts: >=86400 -> 1Д, иначе 1Ч)."""
    def _ci(df):
        step = int(df['ts'].iloc[-1] - df['ts'].iloc[-2]) if len(df) > 1 else 86400
        return frames_by_step.get(86400 if step >= 86400 else 3600, df)
    scanner.compute_indicators = _ci


def flat_ind(n=120, step=86400, t0=1_700_000_000):
    """Нейтральный ряд с индикаторными колонками: дивергенции на нём молчат."""
    ts = t0 + np.arange(n) * step
    return pd.DataFrame({
        'ts': ts,
        'ema_13': 100.0, 'ema_21': 100.0, 'ema_50': 100.0, 'ema_100': 100.0,
        'rsi_14': 50.0, 'macd_line': 0.0, 'macd_signal': 0.0, 'macd_hist': 0.0,
        'bb_middle': 100.0, 'bb_upper': 110.0, 'bb_lower': 90.0,
        'adx_14': 15.0, 'atr_14': 2.0, 'close': 100.0,
        'date': pd.to_datetime(ts, unit='s')})


def put_candles(db, secid, df, interval):
    rows = [(secid, int(r.ts), interval, r.open, r.high, r.low, r.close,
             int(r.volume), 0) for r in df.itertuples()]
    db.executemany("INSERT OR REPLACE INTO raw_candles VALUES (?,?,?,?,?,?,?,?,?)", rows)


# ---------- вердикты внутри одного горизонта ----------

def test_buy_needs_dominant_score_and_threshold(scanner):
    """RSI < 20 (buy 35) + закрытие у нижней BB (buy 30) -> verdict buy, SL/TP есть."""
    f = ind_df({'rsi_14': 15.0, 'close': 90.0}, prev={'macd_line': 0.0, 'macd_signal': 0.0})
    inject(scanner, {86400: f})
    put_candles(scanner.db, 'BUY1', raw_df(), 24)
    r = scanner.scan_asset('BUY1', 'stocks')
    h = r['horizons']['1d']
    assert h['verdict'] == 'buy'
    assert h['score_buy'] >= scanner.SIGNAL_THRESHOLD and h['score_buy'] > h['score_sell']
    assert h['sl_tp']['stop_loss'] < h['price'] < h['sl_tp']['take_profit']


def test_sell_on_rsi_extreme_and_macd_cross(scanner):
    """RSI > 80 (sell 35) + пересечение сигнальной MACD сверху вниз (sell 30) -> sell."""
    f = ind_df({'rsi_14': 85.0, 'macd_line': -0.5, 'macd_signal': 0.0},
               prev={'macd_line': 0.5, 'macd_signal': 0.0})
    inject(scanner, {86400: f})
    put_candles(scanner.db, 'SELL1', raw_df(), 24)
    r = scanner.scan_asset('SELL1', 'stocks')
    h = r['horizons']['1d']
    assert h['verdict'] == 'sell'
    rules = [s['rule'] for s in h['signals']]
    assert any('RSI > 80' in x for x in rules)
    assert any('MACD cross signal ↓' in x for x in rules)


def test_watch_below_threshold(scanner):
    """Слабое правило (конфлюэнция EMA, вес 10 < порога 30) -> watch, не сигнал.
    Порог SIGNAL_THRESHOLD — порог СИЛЫ доминирующей стороны, а не разницы."""
    f = ind_df({'close': 95.0})  # ниже всех EMA, других правил нет
    inject(scanner, {86400: f})
    put_candles(scanner.db, 'WEAK', raw_df(), 24)
    r = scanner.scan_asset('WEAK', 'stocks')
    assert r['horizons']['1d']['verdict'] == 'watch'
    assert r['horizons']['1d']['strength'] < scanner.SIGNAL_THRESHOLD


def test_neutral_without_rules(scanner):
    inject(scanner, {86400: ind_df({})})
    put_candles(scanner.db, 'FLAT', raw_df(), 24)
    r = scanner.scan_asset('FLAT', 'stocks')
    assert r['horizons']['1d']['verdict'] == 'neutral'
    assert r['horizons']['1d']['signals'] == []


# ---------- веса новых критериев выше EMA ----------

def test_new_criteria_weights_exceed_ema(scanner):
    assert scanner.W_RSI_EXTREME > scanner.W_EMA_FAST
    assert scanner.W_BB_EDGE > scanner.W_EMA_TREND
    assert scanner.W_MACD_CROSS > scanner.W_EMA_TREND
    assert scanner.W_DIVERGENCE > scanner.W_EMA_FAST


def _bearish_div_frame():
    """Медвежья дивергенция гистограммы MACD: новый max цены при lower-high гистограммы."""
    df = flat_ind()
    p = df['close'].to_numpy().copy(); hst = df['macd_hist'].to_numpy().copy()
    lo, hi = len(df) - 40, len(df) - 2          # окно последних экстремумов
    mid = len(df) - 22
    p[lo:mid] = np.linspace(100, 110, mid - lo)
    p[mid:hi + 1] = np.linspace(110, 99, hi + 1 - mid)
    p[hi:] = 112.0                               # новый максимум цены (выше 110)
    hst[lo:mid] = np.linspace(-0.5, 1.0, mid - lo)      # первый пик гистограммы выше
    hst[mid:hi + 1] = np.linspace(1.0, -1.0, hi + 1 - mid)
    hst[hi:] = -0.8                              # второй «пик» ниже первого -> расхождение
    df['close'] = p; df['macd_hist'] = hst
    return df


def test_divergence_alone_is_active_signal(scanner):
    """Дивергенция (вес 40 >= порога 30) сама по себе — активный сигнал."""
    inject(scanner, {86400: _bearish_div_frame()})
    put_candles(scanner.db, 'DIV1', raw_df(), 24)
    r = scanner.scan_asset('DIV1', 'stocks')
    h = r['horizons']['1d']
    assert any(s['rule'] == 'MACD histogram bearish divergence' for s in h['signals']), \
        f"expected hist divergence, got {[s['rule'] for s in h['signals']]}"
    assert h['score_sell'] >= scanner.SIGNAL_THRESHOLD
    assert h['verdict'] == 'sell'


def _bullish_rsi_div_frame():
    """Бычья дивергенция RSI: новый min цены при higher-low RSI."""
    df = flat_ind()
    p = df['close'].to_numpy().copy(); rsi = df['rsi_14'].to_numpy().copy()
    lo, hi = len(df) - 40, len(df) - 2
    mid = len(df) - 22
    p[lo:mid] = np.linspace(100, 90, mid - lo)
    p[mid:hi + 1] = np.linspace(90, 101, hi + 1 - mid)
    p[hi:] = 88.0                                # новый минимум цены (ниже 90)
    rsi[lo:mid] = np.linspace(50, 25, mid - lo)  # первый trough RSI ниже
    rsi[mid:hi + 1] = np.linspace(25, 55, hi + 1 - mid)
    rsi[hi:] = 30.0                              # второй trough выше первого -> бычья
    df['close'] = p; df['rsi_14'] = rsi
    return df


def test_rsi_divergence_detected(scanner):
    inject(scanner, {86400: _bullish_rsi_div_frame()})
    put_candles(scanner.db, 'DIV2', raw_df(), 24)
    r = scanner.scan_asset('DIV2', 'stocks')
    h = r['horizons']['1d']
    assert any(s['rule'] == 'RSI bullish divergence' for s in h['signals']), \
        f"got {[s['rule'] for s in h['signals']]}"
    assert h['verdict'] == 'buy'                 # вес 40 >= порога 30


# ---------- горизонты не смешиваются ----------

def test_horizons_do_not_mix_scores(scanner):
    """Buy на 1Д и sell на 1Ч: очки каждого горизонта считаются только по своим свечам."""
    fd = ind_df({'rsi_14': 15.0, 'close': 90.0})          # strong buy на дневке
    # часовик: только RSI>80 (sell 35); close=100 — BB-касаний и EMA-конфлюэнции нет
    fh = ind_df({'rsi_14': 85.0})
    inject(scanner, {86400: fd, 3600: fh})
    put_candles(scanner.db, 'MIX', raw_df(), 24)
    d_h = raw_df()
    d_h['ts'] = 1_740_000_000 + np.arange(len(d_h)) * 3600
    put_candles(scanner.db, 'MIX', d_h, 60)
    r = scanner.scan_asset('MIX', 'stocks')
    hd, hh = r['horizons']['1d'], r['horizons']['1h']
    assert hd['verdict'] == 'buy' and hd['score_sell'] == 0
    assert hh['verdict'] == 'sell' and hh['score_buy'] == 0
    # сводные плоские поля = основной горизонт 1Д, без примеси часовика
    assert r['verdict'] == 'buy'
    assert r['score_buy'] == hd['score_buy'] and r['score_sell'] == hd['score_sell']
    assert all(s['tf'] == '1Д' for s in hd['signals'])
    assert all(s['tf'] == '1Ч' for s in hh['signals'])


def test_short_restriction_applies_per_horizon_only_when_sell_dominates(scanner):
    """Sell на дневке при запрете шорта -> watch; buy на часовике не блокируется."""
    fd = ind_df({'rsi_14': 85.0})                        # чистый sell на дневке
    fh = ind_df({'rsi_14': 15.0})                        # чистый buy на часовике
    inject(scanner, {86400: fd, 3600: fh})
    scanner.short_restrictions = {'BLOCK1': 'санкция'}
    put_candles(scanner.db, 'BLOCK1', raw_df(), 24)
    h = raw_df(); h['ts'] = 1_740_000_000 + np.arange(len(h)) * 3600
    put_candles(scanner.db, 'BLOCK1', h, 60)
    r = scanner.scan_asset('BLOCK1', 'stocks')
    assert r['horizons']['1d']['verdict'] == 'watch'
    assert 'санкция' in ' '.join(r['horizons']['1d']['sell_blocked_reasons'])
    assert r['horizons']['1h']['verdict'] == 'buy'   # buy ограничениями не трогается
    assert 'sell_blocked_reasons' not in r['horizons']['1h']
