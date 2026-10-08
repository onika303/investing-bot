"""Unit-тесты Этапа 4: конфигурация бота, pipeline (формат отчёта, экспирации),
планировщик и хендлеры команд aiogram (с мок-ботом, без сети).

Сетевые вызовы (MOEX ISS) не выполняются — используются фикстуры и моки.
"""
import asyncio
import json
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
import yaml

from src.config import BotConfig, ConfigError, load_config
from src.pipeline import (
    TG_MESSAGE_LIMIT,
    check_futures_expiration,
    format_report_text,
    futures_exp_date,
    load_cached_report,
    report_age_hours,
    save_report,
)


# ---------------------------------------------------------------- helpers

def mk_result(secid, verdict, buy=0, sell=0, rules=("RSI перепроданность",),
              price=100.0, blocked=None, sl_tp=True):
    h = {
        "tf": "1d", "tf_label": "1Д", "horizon": "внутри недели",
        "verdict": verdict, "score_buy": buy, "score_sell": sell,
        "strength": max(buy, sell), "price": price,
        "signals": [{"rule": x, "tf": "1Д",
                     "type": verdict if verdict in ("buy", "sell") else "sell",
                     "weight": 15} for x in rules],
    }
    if sl_tp and verdict in ("buy", "sell"):
        h["sl_tp"] = {"stop_loss": round(price * 0.97, 2),
                      "take_profit": round(price * 1.05, 2)}
    if blocked:
        h["sell_blocked_reasons"] = blocked
    r = {
        "secid": secid, "asset_type": "stocks", "date": "2026-10-08",
        "price": price, "verdict": verdict, "score_buy": buy, "score_sell": sell,
        "strength": max(buy, sell),
        "signals": h["signals"], "horizons": {"1d": dict(h)},
        "trade_access": {"short_allowed": not blocked, "reasons": blocked or []},
    }
    if sl_tp and verdict in ("buy", "sell"):
        r["sl_tp"] = h["sl_tp"]
    if blocked:
        r["sell_blocked_reasons"] = blocked
    return r


def mk_report(results):
    return {"generated_at": "2026-10-08T23:59:00", "scanned": len(results),
            "results": results}


# ---------------------------------------------------------------- config

def test_load_config_ok(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("BOT_TOKEN=123:ABC\nADMIN_CHAT_ID=-100123\n"
                   "ALLOWED_USER_IDS=111, 222\nTIMEZONE=Europe/Moscow\n",
                   encoding="utf-8")
    monkeypatch.delenv("BOT_TOKEN", raising=False)
    monkeypatch.delenv("ADMIN_CHAT_ID", raising=False)
    cfg = load_config(str(env))
    assert cfg.bot_token == "123:ABC"
    assert cfg.admin_chat_id == -100123
    assert cfg.allowed_user_ids == [111, 222]
    assert cfg.timezone == "Europe/Moscow"


def test_load_config_missing_token(tmp_path, monkeypatch):
    monkeypatch.delenv("BOT_TOKEN", raising=False)
    monkeypatch.delenv("ADMIN_CHAT_ID", raising=False)
    env = tmp_path / ".env"
    env.write_text("ADMIN_CHAT_ID=1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="BOT_TOKEN"):
        load_config(str(env))


def test_load_config_bad_chatid(tmp_path, monkeypatch):
    monkeypatch.delenv("BOT_TOKEN", raising=False)
    monkeypatch.delenv("ADMIN_CHAT_ID", raising=False)
    env = tmp_path / ".env"
    env.write_text("BOT_TOKEN=1:AA\nADMIN_CHAT_ID=abc\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="ADMIN_CHAT_ID"):
        load_config(str(env))


def test_is_authorized():
    cfg = BotConfig(bot_token="1:a", admin_chat_id=100, allowed_user_ids=[200, 300])
    assert cfg.is_authorized(100)
    assert cfg.is_authorized(200)
    assert not cfg.is_authorized(999)


# ---------------------------------------------------------------- report cache

def test_save_and_load_report(tmp_path):
    p = tmp_path / "last_report.json"
    rep = save_report([mk_result("SBER", "watch", 10, 10)], path=p)
    loaded = load_cached_report(p)
    assert loaded["generated_at"] == rep["generated_at"]
    assert loaded["results"][0]["secid"] == "SBER"
    assert load_cached_report(tmp_path / "missing.json") is None


def test_report_age_hours():
    rep = {"generated_at": datetime.now().isoformat(timespec="seconds")}
    assert report_age_hours(rep) < 0.01


# ---------------------------------------------------------------- formatting

def test_format_report_basic():
    chunks = format_report_text(mk_report([
        mk_result("FLOT", "buy", 35, 0),
        mk_result("BAZA", "watch", 10, 35, blocked=["новая эмиссия"]),
    ]))
    text = "\n".join(chunks)
    assert "🟢 ПОКУПКА" in text and "FLOT" in text
    assert "шорт запрещён" in text and "BAZA" in text
    assert "SL" in text and "TP" in text


def test_format_report_no_signals():
    chunks = format_report_text(mk_report([]))
    assert any("Просканировано активов: 0" in c for c in chunks)


def test_format_report_params_in_header():
    """В шапке отчёта указаны параметры сигналосва: EMA 13/21/50/100, BB σ=2.5."""
    text = "\n".join(format_report_text(mk_report([mk_result("FLOT", "buy", 35, 0)])))
    assert "EMA 13/21/50/100" in text
    assert "BB σ=2.5" in text


def test_format_report_sorted_by_strength_not_alphabet():
    """Списки ранжируются по очкам (strength внутри горизонта), а не по алфавиту."""
    a = mk_result("AAA", "watch", 40, 0)   # слабее по силе
    b = mk_result("BBB", "watch", 90, 0)   # сильнее, но буквенно позже
    text = "\n".join(format_report_text(mk_report([a, b])))
    assert text.index("BBB") < text.index("AAA"), "наблюдение должно быть по очкам"
    # формат строки: 💪 сила, 📊 очки горизонта
    assert "💪 90" in text and "📊 B90/S0" in text


def test_format_report_horizons_not_mixed():
    """Отчёт разделён по горизонтам планирования: 1Д (внутри недели) и 1Ч (1-2 дня);
    сигналы горизонтов не смешиваются, вердикт считается в каждом отдельно."""
    r = mk_result("MTSX", "buy", 60, 10)
    r["horizons"]["1h"] = {
        "tf": "1h", "tf_label": "1Ч", "horizon": "1–2 дня",
        "verdict": "sell", "score_buy": 5, "score_sell": 40, "strength": 40,
        "price": 99.0,
        "signals": [{"rule": "RSI overbought (75.0)", "tf": "1Ч", "type": "sell", "weight": 15}],
    }
    text = "\n".join(format_report_text(mk_report([r])))
    assert "ГОРИЗОНТ 1Д — внутри недели" in text
    assert "ГОРИЗОНТ 1Ч — 1–2 дня" in text
    # дневной buy НЕ «подтверждается» часовиком — каждый горизонт сам по себе:
    i_d = text.index("ГОРИЗОНТ 1Д")
    i_h = text.index("ГОРИЗОНТ 1Ч")
    seg_d, seg_h = text[i_d:i_h], text[i_h:]
    assert "🟢 ПОКУПКА" in seg_d and "🔴 ПРОДАЖА" not in seg_d
    assert "🔴 ПРОДАЖА" in seg_h and "🟢 ПОКУПКА" not in seg_h


def test_format_report_chunking_under_limit():
    """Много длинных блоков -> разбивка на сообщения ≤ лимита Telegram."""
    results = [mk_result(f"TIC{i}", "watch", 35, 0, rules=("правило " * 30,))
               for i in range(60)]
    chunks = format_report_text(mk_report(results), top=60)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= TG_MESSAGE_LIMIT


# ---------------------------------------------------------------- expiration hygiene

NOW = datetime(2026, 10, 8)


def test_futures_exp_date_seasonal():
    assert futures_exp_date("SuV6", NOW) == datetime(2026, 10, 15)
    assert futures_exp_date("BRX6", NOW) == datetime(2026, 11, 15)
    assert futures_exp_date("SiZ6", NOW) == datetime(2026, 12, 15)
    assert futures_exp_date("W4Z6", NOW) == datetime(2026, 12, 15)  # строчная буква в корне
    assert futures_exp_date("S3Z6", NOW) == datetime(2026, 12, 15)


def test_futures_exp_date_perpetual_none():
    for s in ["USDRUBF", "EURRUBF", "BTCUSDF", "RGBIF", "IMOEXF", "GLDRUBF"]:
        assert futures_exp_date(s, NOW) is None, s


def test_check_expiration_horizon():
    assets = {"futures": ["SuV6", "BRX6", "SiZ6", "USDRUBF", "GDZ6"]}
    w4 = check_futures_expiration(assets, weeks_ahead=4, now=NOW)
    assert any("SuV6" in x for x in w4)          # октябрь — до horizon
    assert not any("SiZ6" in x for x in w4)      # декабрь — за пределами 4 нед.
    w12 = check_futures_expiration(assets, weeks_ahead=12, now=NOW)
    assert any("SiZ6" in x for x in w12)


def test_check_expiration_on_real_config():
    from src.pipeline import load_assets
    assets = load_assets("config.yaml")
    w = check_futures_expiration(assets, weeks_ahead=4, now=NOW)
    # хотя бы октябрьский сахар должен требовать замены
    assert any("SuV6" in x for x in w)


# ---------------------------------------------------------------- scheduler triggers

def test_scheduler_jobs_next_run():
    """Триггеры дают правильные ближайшие даты/дни недели (наивное время теста)."""
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger

    sched = AsyncIOScheduler()
    noop = lambda: None
    sched.add_job(noop, CronTrigger(hour=9, minute=45, day_of_week="mon-fri"), id="u")
    sched.add_job(noop, CronTrigger(hour=10, minute=0, day_of_week="mon-fri"), id="s")
    sched.add_job(noop, CronTrigger(hour=0, minute=0), id="e")
    sched.add_job(noop, CronTrigger(day_of_week="sat", hour=12), id="h")
    # фиксируем «сейчас» как четверг 08.10.2026 12:00
    start = datetime(2026, 10, 8, 12, 0)
    nxt = {j.id: j.trigger.get_next_fire_time(None, start) for j in sched.get_jobs()}
    assert nxt["u"].strftime("%H:%M") == "09:45" and nxt["u"].weekday() < 5
    assert nxt["s"].strftime("%H:%M") == "10:00" and nxt["s"].weekday() < 5
    assert nxt["e"].strftime("%H:%M") == "00:00"
    assert nxt["h"].weekday() == 5 and nxt["h"].strftime("%H:%M") == "12:00"


# ---------------------------------------------------------------- bot handlers (mock)

@pytest.fixture()
def fake_message():
    class FakeMessage:
        def __init__(self, text="/scan", chat_id=100):
            self.text = text
            self.chat = SimpleNamespace(id=chat_id)
            self.bot = SimpleNamespace(
                send_message=_AsyncRecorder(), chat_id=chat_id)
            self.answers = []
            self.edits = []

        async def answer(self, text, **kw):
            self.answers.append(text)
            return SimpleNamespace(
                edit_text=lambda t: self._edit(t))

        async def _edit(self, t):
            self.edits.append(t)

    return FakeMessage


class _AsyncRecorder:
    def __init__(self):
        self.sent = []

    async def __call__(self, chat_id, text, **kw):
        self.sent.append((chat_id, text))


def make_state(cfg=None):
    from src.bot import AppState
    cfg = cfg or BotConfig(bot_token="1:a", admin_chat_id=100)
    st = AppState.__new__(AppState)  # без чтения config.yaml/last_report в тесте
    st.cfg = cfg
    st.report = mk_report([mk_result("FLOT", "buy", 35, 0),
                           mk_result("BAZA", "watch", 10, 35,
                                     blocked=["новая эмиссия"])])
    st.assets = {"stocks": ["SBER", "GAZP", "FLOT", "BAZA"], "futures": ["SiZ6"]}
    st.scan_lock = asyncio.Lock()
    st.update_lock = asyncio.Lock()
    st.last_update_stats = None
    st.bot_started_at = datetime(2026, 10, 8, 9, 0)
    return st


def run(coro):
    return asyncio.run(coro)


def test_cmd_start_help(fake_message):
    from src.bot import cmd_start
    msg = fake_message()
    run(cmd_start(msg))
    assert any("Инвестиционный сканер" in a for a in msg.answers)


def test_status_shows_counts(fake_message):
    from src.bot import cmd_status
    msg = fake_message("/status")
    run(cmd_status(msg, make_state()))
    text = "".join(msg.answers)
    assert "Статус" in text and "Уникальных secid" in text


def test_status_unauthorized(fake_message):
    """Авторизация вынесена в фильтр IsAuthorized (шаг 4.1/4.3)."""
    from src.bot import IsAuthorized
    state = make_state()
    allowed = fake_message("/status", chat_id=100)
    denied = fake_message("/status", chat_id=777)
    assert run(IsAuthorized()(allowed, app=state)) is True
    assert run(IsAuthorized()(denied, app=state)) is False
    assert "chat_id" in denied.answers[0] and "777" in denied.answers[0]


def test_scan_returns_cached_report(fake_message):
    from src.bot import cmd_scan
    msg = fake_message("/scan")
    state = make_state(BotConfig(bot_token="1:a", admin_chat_id=100,
                                 report_cache_hours=1000))
    run(cmd_scan(msg, state))
    sent = "".join(t for _, t in msg.bot.send_message.sent)
    assert "Сканер MOEX" in sent and "FLOT" in sent


def test_signals_filter(fake_message):
    from src.bot import cmd_signals
    from aiogram.filters.command import CommandObject
    msg = fake_message("/signals buy")
    run(cmd_signals(msg, CommandObject(prefix="/", command="signals", args="buy"),
                    make_state()))
    text = "".join(msg.answers)
    assert "BUY" in text and "FLOT" in text and "BAZA" not in text


def test_signals_bad_arg(fake_message):
    from src.bot import cmd_signals
    from aiogram.filters.command import CommandObject
    msg = fake_message("/signals meow")
    run(cmd_signals(msg, CommandObject(prefix="/", command="signals", args="meow"),
                    make_state()))
    assert "Использование" in msg.answers[0]


def test_asset_details_with_sell_block(fake_message, monkeypatch):
    import src.bot as botmod
    res = mk_result("BAZA", "watch", 10, 35, blocked=["новая эмиссия"])
    scanner_stub = SimpleNamespace(scan_asset=lambda s, t: res)
    monkeypatch.setattr(botmod, "InvestmentScanner", lambda db: scanner_stub)
    monkeypatch.setattr(botmod, "DatabaseManager", lambda p: None)
    msg = fake_message("/asset BAZA")
    run(botmod.cmd_asset(msg, SimpleNamespace(args="BAZA"), make_state()))
    text = "".join(msg.answers)
    assert "BAZA" in text and "Шорт запрещён" in text


def test_reload_config(tmp_path, fake_message, monkeypatch):
    import src.bot as botmod
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("assets:\n  stocks: [SBER]\n  futures: [SiZ6]\n",
                        encoding="utf-8")
    monkeypatch.setattr(botmod, "CONFIG_PATH", str(cfg_file))
    msg = fake_message("/reload_config")
    state = make_state()
    run(botmod.cmd_reload(msg, state))
    assert "перечитан" in msg.answers[0] and state.assets["stocks"] == ["SBER"]


def test_do_full_scan_guard_blocks_parallel(monkeypatch):
    """Guard: второй скан не стартует, пока первый держит лок."""
    import src.bot as botmod
    state = make_state()

    async def slow_scan(*a, **k):
        return {"results": [], "generated_at": "x"}

    monkeypatch.setattr(botmod, "scan_and_report", slow_scan)
    await_holder = []

    async def scenario():
        await state.scan_lock.acquire()          # имитируем идущий скан
        try:
            with pytest.raises(RuntimeError):
                await botmod.do_full_scan(state, None, None, do_update=False)
        finally:
            state.scan_lock.release()
    run(scenario())


# ---------------------------------------------------------------- cache/config consistency

class TestReportMatchesConfig:
    def _write_cfg(self, tmp_path, tickers):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(yaml.safe_dump({"assets": {"stocks": tickers}}),
                       encoding="utf-8")
        return str(cfg)

    def test_consistent_report_matches(self, tmp_path):
        from src.pipeline import report_matches_config
        path = self._write_cfg(tmp_path, ["SBER", "GAZP"])
        rep = {"generated_at": datetime.now().isoformat(),
               "results": [{"secid": "SBER"}, {"secid": "GAZP"}]}
        assert report_matches_config(rep, None, path) is True

    def test_reduced_ticker_list_invalidates_cache(self, tmp_path):
        """Кэш со старым (большим) списком не должен считаться актуальным."""
        from src.pipeline import report_matches_config
        path = self._write_cfg(tmp_path, ["SBER"])
        rep = {"generated_at": datetime.now().isoformat(),
               "results": [{"secid": "SBER"}, {"secid": "GAZP"}]}
        assert report_matches_config(rep, None, path) is False

    def test_missing_asset_in_cache_invalidates(self, tmp_path):
        from src.pipeline import report_matches_config
        path = self._write_cfg(tmp_path, ["SBER", "LKOH"])
        rep = {"generated_at": datetime.now().isoformat(),
               "results": [{"secid": "SBER"}]}
        assert report_matches_config(rep, None, path) is False

    def test_edited_config_newer_than_report_invalidates(self, tmp_path):
        from src.pipeline import report_matches_config
        path = self._write_cfg(tmp_path, ["SBER"])
        old = (datetime.now() - timedelta(hours=5)).isoformat()
        rep = {"generated_at": old, "results": [{"secid": "SBER"}]}
        # config.yaml только что перезаписан → mtime новее generated_at
        assert report_matches_config(rep, None, path) is False

    def test_broken_report_is_not_matching(self, tmp_path):
        from src.pipeline import report_matches_config
        path = self._write_cfg(tmp_path, ["SBER"])
        assert report_matches_config({}, None, path) is False
