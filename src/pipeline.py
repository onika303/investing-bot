"""Фоновый конвейер: инкрементальная загрузка данных MOEX + полный скан + отчёт.

Логика вынесена из CLI (scan.py) в переиспользуемый модуль, который одинаково
вызывается и планировщиком бота (APScheduler), и вручную (scripts/run_report_once.py).

Все функции синхронные (MOEXLoader внутри поднимает свой asyncio loop) —
в асинхронном коде бота запускать через asyncio.to_thread.
"""
import json
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import yaml
from loguru import logger

from src.database import DatabaseManager
from src.moex_loader import MOEXLoader
from src.scanner import InvestmentScanner

LAST_REPORT_PATH = Path("last_report.json")

TG_MESSAGE_LIMIT = 4096

# Сезонный код фьючерса MOEX (COB): корень серии + буква месяца + последняя цифра года.
# Напр. SiZ6, BRX6, SuV6, W4Z6, KCX6, S3Z6. Год в коде — одна цифра (цикл 10 лет),
# поэтому экспирация НЕЧЁТНОГО года не определяется однозначно (в config сейчас Z6=2026).
_SEASON_RE = re.compile(r"^(?P<root>[A-Z0-9]{1,5})(?P<month>[FGHJKMNQUVXZ])(?P<year>\d)$")
_MONTHS = {'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
           'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12}


def load_assets(config_path: str = "config.yaml") -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)["assets"]


def incremental_update(assets: dict, db_path: str = "finance.db",
                       default_days: int = 45) -> dict:
    """Дозагрузка только хвоста истории каждого актива.

    Точка начала — get_last_ts(secid) из БД (+1 день); если актива в БД нет —
    грузим полную историю (default_days). Запись meta обновляет loader.
    Возвращает сводку {updated, new, empty, errors}.
    """
    db = DatabaseManager(db_path)
    loader = MOEXLoader(db)
    today = datetime.now().strftime("%Y-%m-%d")
    stats = {"updated": 0, "new": 0, "empty": 0, "errors": 0}

    for asset_type, secids in assets.items():
        for secid in secids:
            try:
                last_ts = db.get_last_ts(secid)
                if last_ts:
                    from_date = (datetime.fromtimestamp(last_ts)
                                 + timedelta(days=1)).strftime("%Y-%m-%d")
                    stats["updated"] += 1
                else:
                    from_date = (datetime.now() - timedelta(days=default_days)
                                 ).strftime("%Y-%m-%d")
                    stats["new"] += 1
                df = loader.load_asset(secid, asset_type, from_date, today,
                                       interval=24, end_on_load=True)
                if df is None or len(df) == 0:
                    stats["empty"] += 1  # выходные/праздники/нет торгов — штатно
            except Exception as e:
                stats["errors"] += 1
                logger.error(f"Инкрементальная загрузка {secid}: {e}")
    logger.info(f"Incremental update: {stats}")
    return stats


def run_full_scan(assets: dict, db_path: str = "finance.db") -> list[dict]:
    db = DatabaseManager(db_path)
    scanner = InvestmentScanner(db)
    return scanner.scan_all(assets)


def save_report(results: list[dict], path: Path = LAST_REPORT_PATH) -> dict:
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scanned": len(results),
        "results": results,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, default=str),
                    encoding="utf-8")
    return payload


def load_cached_report(path: Path = LAST_REPORT_PATH) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def report_age_hours(report: dict) -> float:
    gen = datetime.fromisoformat(report["generated_at"])
    return (datetime.now() - gen).total_seconds() / 3600


def fmt_price(v) -> str:
    try:
        return f"{float(v):,.2f}".replace(",", " ")
    except (TypeError, ValueError):
        return "-"


def format_report_text(report: dict, top: int = 25) -> list[str]:
    """Отчёт по результатам скана в сообщения Telegram (с учётом лимита 4096)."""
    results = report["results"]
    active = [r for r in results if r["verdict"] in ("buy", "sell")]
    watch = [r for r in results if r["verdict"] == "watch"]

    header = (f"📊 <b>Отчёт сканера MOEX</b>\n"
              f"Дата: {report['generated_at'][:16].replace('T', ' ')} MSK\n"
              f"Просканировано: {len(results)} | сигналы: 🟢{sum(1 for r in active if r['verdict']=='buy')} "
              f"🔴{sum(1 for r in active if r['verdict']=='sell')} 👁{len(watch)}\n")

    lines = []
    for r in active + watch[:max(0, top - len(active))]:
        icon = {"buy": "🟢 ПОКУПКА", "sell": "🔴 ПРОДАЖА",
                "watch": "👁 наблюдение"}.get(r["verdict"], r["verdict"])
        if r.get("sell_blocked_reasons"):
            icon += " (шорт запрещён)"
        rules = "; ".join(s["rule"] for s in r["signals"][:4]) or "-"
        sl_tp = ""
        if r["verdict"] in ("buy", "sell") and r.get("sl_tp"):
            sl_tp = f" | SL {fmt_price(r['sl_tp']['stop_loss'])} → TP {fmt_price(r['sl_tp']['take_profit'])}"
        lines.append(f"<code>{r['secid']:<7}</code> {icon} "
                     f"B{r['score_buy']}/S{r['score_sell']} @ {fmt_price(r['price'])}{sl_tp}\n"
                     f"   └ {rules}")
    if not lines:
        lines = ["Активных сигналов нет."]

    body = "\n".join(lines)
    chunks = []
    cur = header
    for ln in lines:
        if len(cur) + len(ln) + 1 > TG_MESSAGE_LIMIT - 200:
            chunks.append(cur)
            cur = ""
        cur += ln + "\n"
    if cur.strip():
        chunks.append(cur)
    return chunks


def futures_exp_date(secid: str, now: datetime | None = None) -> datetime | None:
    """Ориентировочная дата экспирации сезонного контракта (середина месяца, 15-е число).

    Год в тикере — одна цифра (цикл COB 10 лет): выбираем ближайшее будущее десятилетие.
    Возвращает None для бессрочных/нераспознанных кодов.
    """
    now = now or datetime.now()
    # тикеры MOEX чувствительны к регистру (SuV6, W4Z6, S3Z6) — матчим оригинал и upper
    m = _SEASON_RE.match(secid) or _SEASON_RE.match(secid.upper())
    if not m:
        return None
    month = _MONTHS.get(m["month"].upper())
    if not month:
        return None
    year_digit = int(m["year"])
    candidates = [2020 + year_digit + 10 * k for k in range(3)]
    for y in candidates:
        d = datetime(y, month, 15)
        if d >= now - timedelta(days=7):  # неделя запаса на «уже почти экспирировал»
            return d
    return None


def check_futures_expiration(assets: dict, weeks_ahead: int = 4,
                             now: datetime | None = None) -> list[str]:
    """Субботняя гигиена: сезонные тикеры с экспирацией в пределах weeks_ahead."""
    warnings = []
    now = now or datetime.now()
    horizon = now + timedelta(weeks=weeks_ahead)
    for secid in assets.get("futures", []):
        exp = futures_exp_date(secid, now)
        if exp is None:
            continue  # бессрочные (USDRUBF, BTCUSDF...) и нераспознанные — пропускаем
        if exp <= horizon:
            warnings.append(
                f"⚠️ {secid}: экспирация ~{exp:%d.%m.%Y} "
                f"(менее {weeks_ahead} нед.) — заменить контракт в config.yaml")
    return sorted(warnings)


def scan_and_report(db_path: str = "finance.db", config_path: str = "config.yaml",
                    do_update: bool = True) -> tuple[dict, dict]:
    """Полный рабочий цикл планировщика: update → scan → кэш-отчёт. Возвращает (report, stats)."""
    t0 = time.monotonic()
    assets = load_assets(config_path)
    stats = incremental_update(assets, db_path) if do_update else {}
    results = run_full_scan(assets, db_path)
    report = save_report(results)
    report["duration_sec"] = round(time.monotonic() - t0, 1)
    save_report(results)  # пересохранить с длительностью
    return report, stats
