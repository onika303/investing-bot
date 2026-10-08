"""Тесты проверки доступности актива к сделке (short_allowed)."""
import json
from datetime import timedelta
from unittest.mock import patch

import pandas as pd
import pytest

from src.scanner import InvestmentScanner, load_short_restrictions


class FakeResp:
    """Заглушка ответа requests.get."""
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status = status

    def raise_for_status(self):
        if self.status != 200:
            raise RuntimeError(f"HTTP {self.status}")

    def json(self):
        return self._payload


def iss_payload(issuedate):
    """Формат ISS /securities/{secid}.json как в реальном ответе."""
    return {'securities': {
        'columns': ['SECID', 'ISSUEDATE', 'LISTEDTILL', 'CURRENTSTATUS'],
        'data': [['TEST', issuedate, None, 'A']],
    }}


@pytest.fixture
def scanner(tmp_db, tmp_path):
    """Сканер с временным restrictions-файлом и изолированным дисковым кэшем."""
    restr_file = tmp_path / 'short_restrictions.json'
    with patch('src.scanner.SHORT_RESTRICTIONS_FILE', restr_file):
        s = InvestmentScanner(tmp_db, short_restrictions={})
    # переносим дисковый кэш issue_info во временную директорию
    s._disk_issue_cache = tmp_path / '.cache' / 'issue_info.json'
    yield s


# ---------- load_short_restrictions ----------
def test_load_restrictions_missing_file(tmp_path):
    assert load_short_restrictions(tmp_path / 'nope.json') == {}


def test_load_restrictions_ok(tmp_path):
    p = tmp_path / 'r.json'
    p.write_text(json.dumps({"LVHK": "sanctioned"}), encoding='utf-8')
    assert load_short_restrictions(p) == {"LVHK": "sanctioned"}


def test_load_restrictions_broken_json(tmp_path):
    p = tmp_path / 'r.json'
    p.write_text('{not json', encoding='utf-8')
    assert load_short_restrictions(p) == {}


# ---------- статический список ограничений ----------
def test_static_restriction_blocks_short(scanner):
    scanner.short_restrictions = {'BAZA': 'new emission IPO'}
    res = scanner.check_trade_access('BAZA', 'stocks', pd.Timestamp('2020-01-01'))
    assert res['short_allowed'] is False
    assert any('new emission IPO' in r for r in res['reasons'])


def test_no_restriction_allows_short(scanner):
    scanner.short_restrictions = {}
    with patch('src.scanner.requests.get', return_value=FakeResp(iss_payload('2010-05-18'))):
        res = scanner.check_trade_access('SBER', 'stocks', pd.Timestamp('2020-01-01'))
    assert res['short_allowed'] is True
    assert res['reasons'] == []


# ---------- эвристика новой эмиссии через ISS ----------
def test_new_emission_via_issuedate(scanner):
    recent = (pd.Timestamp.now(tz='Europe/Moscow') - timedelta(days=300)).strftime('%Y-%m-%d')
    with patch('src.scanner.requests.get', return_value=FakeResp(iss_payload(recent))):
        res = scanner.check_trade_access('MOCK', 'stocks', pd.Timestamp('2020-01-01'))
    assert res['short_allowed'] is False
    assert any('новая эмиссия' in r for r in res['reasons'])


def test_old_emission_via_issuedate(scanner):
    old = (pd.Timestamp.now(tz='Europe/Moscow') - timedelta(days=400)).strftime('%Y-%m-%d')
    with patch('src.scanner.requests.get', return_value=FakeResp(iss_payload(old))):
        res = scanner.check_trade_access('MOCK', 'stocks', pd.Timestamp('2020-01-01'))
    assert res['short_allowed'] is True


def test_futures_not_checked_by_emission_heuristic(scanner):
    """Для фьючерсов эвристика 'новой эмиссии' не применяется (сезонные контракты
    всегда короче года); сетевой запрос к ISS не нужен."""
    with patch('src.scanner.requests.get') as mock_get:
        res = scanner.check_trade_access('GDZ6', 'futures',
                                         pd.Timestamp.now() - timedelta(days=100))
        mock_get.assert_not_called()
    assert res['short_allowed'] is True


def test_iss_failure_falls_back_to_first_date(scanner):
    """Если ISS недоступна — фолбэк по дате первой свечи в БД."""
    with patch('src.scanner.requests.get', side_effect=ConnectionError('network down')):
        res = scanner.check_trade_access('NEWX', 'stocks',
                                         pd.Timestamp.now() - timedelta(days=100))
    assert res['short_allowed'] is False  # 100 дней < 365 -> новая эмиссия
    with patch('src.scanner.requests.get', side_effect=ConnectionError('network down')):
        res = scanner.check_trade_access('OLDX', 'stocks',
                                         pd.Timestamp.now() - timedelta(days=700))
    assert res['short_allowed'] is True


def test_iss_empty_response_falls_back(scanner):
    """ISS вернула пустой securities — фолбэк по first_date."""
    with patch('src.scanner.requests.get', return_value=FakeResp({'securities': {'columns': [], 'data': []}})):
        res = scanner.check_trade_access('NEWX', 'stocks',
                                         pd.Timestamp.now() - timedelta(days=50))
    assert res['short_allowed'] is False


# ---------- интеграция с scan_asset: sell блокируется ----------

def _apply_closes(base_df, closes):
    df = base_df.copy()[:len(closes)].reset_index(drop=True)
    df['close'] = closes
    df['open'] = closes
    df['high'] = closes * 1.005
    df['low'] = closes * 0.995
    return df


def downtrend_df(base_df):
    """Боковик 200 -> обвальный drop до 100 -> резкий V-отскок до 150 за 6 баров:
    RSI overbought (sell 15) + Close above upper BB (sell 10) + ADX boost (10)
    -> S=35/B=0 (уверенная sell-доминанта >= порога 30)."""
    import numpy as np
    drop = np.array([200., 170., 140., 110., 100.])
    closes = np.concatenate([np.full(60, 200.), drop, np.linspace(100, 150, 6)])
    return _apply_closes(base_df, closes)


def uptrend_df(base_df):
    """Зеркальный бычий паттерн: боковик 100 -> рывок до 200 -> резкий откат
    до 150 за 6 баров -> B=35/S=0 (buy-доминанта >= порога 30)."""
    import numpy as np
    rally = np.array([100., 130., 160., 190., 200.])
    closes = np.concatenate([np.full(60, 100.), rally, np.linspace(200, 150, 6)])
    return _apply_closes(base_df, closes)



def _insert_candles(db, secid, df, interval=24):
    rows = [(secid, int(r.ts), interval, r.open, r.high, r.low, r.close, int(r.volume), 0)
            for r in df.itertuples()]
    db.executemany("INSERT OR REPLACE INTO raw_candles VALUES (?,?,?,?,?,?,?,?,?)", rows)


def test_scan_asset_sell_blocked_to_watch(scanner, v_shape_df):
    """Sell-доминанта при запрещённом шорте -> verdict watch + sell_blocked_reasons."""
    df = downtrend_df(v_shape_df)
    _insert_candles(scanner.db, 'BLOCKED', df)

    scanner.short_restrictions = {'BLOCKED': 'sanctioned'}
    with patch.object(InvestmentScanner, '_fetch_issue_info', return_value={'issuedate': '2015-01-01'}):
        r = scanner.scan_asset('BLOCKED', 'stocks')
    assert r is not None
    h = r['horizons']['1d']
    assert h['score_sell'] >= 30 and h['score_sell'] > h['score_buy'], \
        f"expected sell-dominant, got B{h['score_buy']}/S{h['score_sell']} {[s['rule'] for s in h['signals']]}"
    assert h['verdict'] == 'watch'
    assert r['verdict'] == 'watch'  # основной горизонт — дневной
    assert 'sanctioned' in ' '.join(h['sell_blocked_reasons'])
    assert r['trade_access']['short_allowed'] is False


def test_scan_asset_sell_allowed_when_accessible(scanner, v_shape_df):
    """Sell-паттерн на 1д и отдельно на 1ч -> активный сигнал в КАЖДОМ горизонте
    (смешивания ТФ больше нет: вердикт считается внутри своего горизонта)."""
    df = downtrend_df(v_shape_df)
    _insert_candles(scanner.db, 'ALLOWED', df, interval=24)
    _insert_candles(scanner.db, 'ALLOWED', df, interval=60)  # тот же паттерн на часовике
    with patch.object(InvestmentScanner, '_fetch_issue_info', return_value={'issuedate': '2015-01-01'}):
        r = scanner.scan_asset('ALLOWED', 'stocks')
    assert set(r['horizons']) == {'1d', '1h'}
    assert r['horizons']['1d']['verdict'] == 'sell'
    assert r['horizons']['1h']['verdict'] == 'sell'
    assert r['verdict'] == 'sell'          # основной горизонт — 1Д
    assert r['trade_access']['short_allowed'] is True
    assert r['can_short'] is True
    assert 'sl_tp' in r and r['sl_tp']['stop_loss'] > r['price']
    # SL/TP каждого горизонта считаются по ATR этого горизонта
    h1 = r['horizons']['1h']
    assert 'sl_tp' in h1 and h1['sl_tp']['stop_loss'] > h1['price']


def test_horizons_analyzed_separately(scanner, v_shape_df):
    """Бычий паттерн на дневке + медвежий на часовике -> горизонты НЕ перемешиваются:
    1Д даёт buy, 1Ч даёт sell, вердикт основного горизонта = buy."""
    import numpy as np
    up = uptrend_df(v_shape_df)
    down = downtrend_df(v_shape_df)
    _insert_candles(scanner.db, 'MIXTF', up, interval=24)
    _insert_candles(scanner.db, 'MIXTF', down, interval=60)
    with patch.object(InvestmentScanner, '_fetch_issue_info', return_value={'issuedate': '2015-01-01'}):
        r = scanner.scan_asset('MIXTF', 'stocks')
    hd, hh = r['horizons']['1d'], r['horizons']['1h']
    assert hd['verdict'] == 'buy' and hd['score_buy'] >= 30
    assert hh['verdict'] == 'sell' and hh['score_sell'] >= 30
    # сигналы горизонтов не сливаются в один список
    assert all(s['tf'] == '1Д' for s in hd['signals'])
    assert all(s['tf'] == '1Ч' for s in hh['signals'])
    assert r['verdict'] == 'buy'  # основной = дневной


def test_sell_on_hourly_only_does_not_affect_daily_verdict(scanner, v_shape_df):
    """Sell только на часовике: горизонт 1Ч -> sell, горизонт 1Д -> watch;
    сводный вердикт (основной горизонт) остаётся watch, без смешивания очков."""
    df = downtrend_df(v_shape_df)
    _insert_candles(scanner.db, 'HONLY', df, interval=60)  # только 1ч свечи
    with patch.object(InvestmentScanner, '_fetch_issue_info', return_value={'issuedate': '2015-01-01'}):
        r = scanner.scan_asset('HONLY', 'stocks')
    assert r['timeframes'] == ['1h']
    assert r['horizons']['1h']['verdict'] == 'sell'
    # плоские поля = часовой горизонт (дневного нет)
    assert r['verdict'] == 'sell'
    assert r['strength'] == r['horizons']['1h']['strength']


def test_buy_signal_not_affected_by_short_restriction(scanner, v_shape_df):
    """Ограничение на шорт НЕ блокирует buy-сигналы."""
    df = uptrend_df(v_shape_df)
    _insert_candles(scanner.db, 'BUYREST', df)
    scanner.short_restrictions = {'BUYREST': 'sanctioned'}
    with patch.object(InvestmentScanner, '_fetch_issue_info', return_value={'issuedate': '2015-01-01'}):
        r = scanner.scan_asset('BUYREST', 'stocks')
    h = r['horizons']['1d']
    assert h['score_buy'] > h['score_sell']
    assert h['verdict'] == 'buy'             # buy не блокируется ограничениями шорта
    assert 'sell_blocked_reasons' not in h   # sell не доминировал — блокировка не применялась
