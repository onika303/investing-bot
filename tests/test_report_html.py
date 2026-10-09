"""Регрессия: отчёт сканера должен быть валидным Telegram HTML.

Баг 09.10.2026: сигналы 'RSI < 20 (...)', 'RSI > 80 (...)' и причины
блокировки шорта ('история торгов < 1 года') вставлялись в сообщение с
parse_mode=HTML без экранирования -> Telegram Bad Request
'Unsupported start tag ""' и потеря части сообщений.
"""
import json
import re
from pathlib import Path

import pytest

from src.pipeline import format_report_text, esc_html

TAGS = re.compile(r'</?(b|code|i)\b[^>]*>')


def _assert_valid_html(msg: str):
    stripped = TAGS.sub('', msg)
    assert '<' not in stripped and '>' not in stripped, "голые угловые скобки"
    stack = []
    for m in re.finditer(r'<(/?)(b|code|i)\b[^>]*>', msg):
        closing, tag = m.group(1), m.group(2)
        if closing:
            assert stack and stack[-1] == tag
            stack.pop()
        else:
            stack.append(tag)
    assert not stack, "незакрытые теги"
    assert len(msg) <= 4096


@pytest.fixture
def fake_report():
    return {"generated_at": "2026-10-09T10:00:00", "results": [{
        "secid": "TEST", "asset_type": "stocks", "price": 100.0,
        "horizons": {
            "1d": {"tf": "1d", "verdict": "watch", "strength": 35,
                   "score_buy": 35, "score_sell": 0, "price": 100.0,
                   "signals": [{"rule": "RSI < 20 (18.3)", "side": "buy",
                                "weight": 35}]},
            "1h": {"tf": "1h", "verdict": "sell", "strength": 40,
                   "score_buy": 0, "score_sell": 40, "price": 100.0,
                   "sl_tp": {"stop_loss": 105, "take_profit": 92},
                   "signals": [{"rule": "RSI > 80 (83.1)", "side": "sell",
                                "weight": 35}]}},
        "sell_blocked_reasons": [
            "новая эмиссия: история торгов < 1 года, шорт недоступен"]}] }


def test_esc_html_basic():
    assert esc_html("a < b & c > d") == "a &lt; b &amp; c &gt; d"
    assert esc_html("&lt;") == "&amp;lt;"


def test_signals_with_angle_brackets_are_escaped(fake_report):
    text = "\n".join(format_report_text(fake_report))
    assert "RSI &lt; 20" in text
    assert "RSI &gt; 80" in text
    assert "&lt; 1 года" in text


def test_horizon_messages_valid_html(fake_report):
    msgs = format_report_text(fake_report)
    assert len(msgs) >= 2  # 1Д и 1Ч раздельно
    for msg in msgs:
        _assert_valid_html(msg)


def test_real_last_report_is_valid_html():
    p = Path(__file__).resolve().parents[1] / "last_report.json"
    if not p.exists():
        pytest.skip("нет кэша отчёта")
    report = json.loads(p.read_text(encoding="utf-8"))
    for msg in format_report_text(report):
        _assert_valid_html(msg)
