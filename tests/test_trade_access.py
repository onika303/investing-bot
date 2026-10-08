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
    """Обвал затем резкий отскок: RSI overbought + close above upper BB + ADX-boost
    -> S=35/B=10 (sell-доминанта >= 30)."""
    import numpy as np
    closes = np.concatenate([np.linspace(150, 100, 100), np.linspace(100, 135, 12)])
    return _apply_closes(base_df, closes)


def uptrend_df(base_df):
    """Обвал с разворотом вверх: RSI oversold + ADX-boost -> B=25/S=10
    (buy ниже порога 30 => verdict watch — используется в тесте buy+ограничение)."""
    import numpy as np
    closes = np.linspace(150, 60, 120)
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
    assert r['score_sell'] >= 30 and r['score_sell'] > r['score_buy'], \
        f"expected sell-dominant, got B{r['score_buy']}/S{r['score_sell']} {[s['rule'] for s in r['signals']]}"
    assert r['verdict'] == 'watch'
    assert 'sanctioned' in ' '.join(r['sell_blocked_reasons'])
    assert r['trade_access']['short_allowed'] is False


def test_scan_asset_sell_allowed_when_accessible(scanner, v_shape_df):
    """Sell-паттерн на 1д + подтверждение на 1ч без ограничений -> verdict sell."""
    df = downtrend_df(v_shape_df)
    _insert_candles(scanner.db, 'ALLOWED', df, interval=24)
    _insert_candles(scanner.db, 'ALLOWED', df, interval=60)  # тот же паттерн на часовике
    with patch.object(InvestmentScanner, '_fetch_issue_info', return_value={'issuedate': '2015-01-01'}):
        r = scanner.scan_asset('ALLOWED', 'stocks')
    assert r['verdict'] == 'sell', f"got {r['verdict']} B{r['score_buy']}/S{r['score_sell']}"
    assert r['tf_confirmed'] is True
    assert r['trade_access']['short_allowed'] is True
    assert r['can_short'] is True
    assert 'sl_tp' in r and r['sl_tp']['stop_loss'] > r['price']


def test_sell_without_hourly_confirm_is_watch(scanner, v_shape_df):
    """Дневная sell-доминанта БЕЗ подтверждения 1ч -> максимум watch."""
    df = downtrend_df(v_shape_df)
    _insert_candles(scanner.db, 'NOCONF', df, interval=24)  # 1ч свечей нет
    with patch.object(InvestmentScanner, '_fetch_issue_info', return_value={'issuedate': '2015-01-01'}):
        r = scanner.scan_asset('NOCONF', 'stocks')
    assert r['score_sell'] >= 30 and r['score_sell'] > r['score_buy']
    assert r['verdict'] == 'watch'
    assert r['tf_confirmed'] is False
    assert r['timeframes'] == ['1d']

def test_buy_signal_not_affected_by_short_restriction(scanner, v_shape_df):
    """Ограничение на шорт НЕ блокирует buy-сигналы."""
    df = uptrend_df(v_shape_df)
    _insert_candles(scanner.db, 'BUYREST', df)
    scanner.short_restrictions = {'BUYREST': 'sanctioned'}
    with patch.object(InvestmentScanner, '_fetch_issue_info', return_value={'issuedate': '2015-01-01'}):
        r = scanner.scan_asset('BUYREST', 'stocks')
    assert r['score_buy'] > r['score_sell']
    assert r['verdict'] == 'watch'          # B=25 < порога 30
    assert 'sell_blocked_reasons' not in r  # sell не доминировал — блокировка не применялась
