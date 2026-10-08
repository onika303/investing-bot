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
                       default_days: int = 45, hours_days: int = 60) -> dict:
    """Дозагрузка только хвоста истории каждого актива на двух ТФ (1д и 1ч).

    1д: точка начала — get_last_ts(secid, interval=24) из БД (+1 день); если актива
        в БД нет — грузим полную историю (default_days).
    1ч: MOEX отдаёт часовые свечи историей ~70 дней; при первой загрузке берём окно
        hours_days, далее — хвост от get_last_ts(secid, interval=60).
    Пустые ответы (выходные/праздники/нет торгов) обрабатываются штатно.
    Возвращает сводку {updated, new, empty, errors}.
    """
    db = DatabaseManager(db_path)
    loader = MOEXLoader(db)
    today = datetime.now().strftime("%Y-%m-%d")
    stats = {"updated": 0, "new": 0, "empty": 0, "errors": 0}

    for asset_type, secids in assets.items():
        for secid in secids:
            # --- дневной таймфрейм ---
            try:
                last_ts = db.get_last_ts(secid, interval=24)
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
                logger.error(f"Инкрементальная загрузка {secid} (1д): {e}")

            # --- часовой таймфрейм ---
            try:
                last_h = db.get_last_ts(secid, interval=60)
                if last_h:
                    from_h = datetime.fromtimestamp(last_h) + timedelta(hours=1)
                    if from_h.strftime("%Y-%m-%d") >= today:
                        pass  # свежее сегодняшнего — ничего не тянем
                    else:
                        dfh = loader.load_asset(secid, asset_type,
                                                from_h.strftime("%Y-%m-%d"), today,
                                                interval=60, end_on_load=True)
                        if dfh is None or len(dfh) == 0:
                            stats["empty"] += 1
                else:
                    dfh = loader.load_asset(secid, asset_type,
                                            (datetime.now() - timedelta(days=hours_days)
                                             ).strftime("%Y-%m-%d"), today,
                                            interval=60, end_on_load=True)
                    if dfh is None or len(dfh) == 0:
                        stats["empty"] += 1
            except Exception as e:
                stats["errors"] += 1
                logger.error(f"Инкрементальная загрузка {secid} (1ч): {e}")
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
    """Отчёт сканера для Telegram: сверху важные сигналы, ниже — список наблюдения,
    отсортированный по сумме очков (не по алфавиту). Сообщения разбиваются под
    лимит 4096 символов."""
    results = report["results"]

    def strength(r):
        # Итоговая сила: 1Д (вес 1.0) + 1Ч (вес 0.5); для старых кэшей без поля — max(B,S)
        return r.get("strength") or max(r.get("score_buy", 0), r.get("score_sell", 0))

    active = [r for r in results if r["verdict"] in ("buy", "sell")]
    watch = [r for r in results if r["verdict"] == "watch"]
    active.sort(key=strength, reverse=True)
    watch.sort(key=strength, reverse=True)

    n_buy = sum(1 for r in active if r["verdict"] == "buy")
    n_sell = sum(1 for r in active if r["verdict"] == "sell")
    header = (f"📡 <b>Сканер MOEX · {report['generated_at'][:16].replace('T', ' ')} МСК</b>\n"
              f"<code>{'─' * 38}</code>\n"
              f"Таймфреймы: <b>1Д + 1Ч</b> | EMA 13/21/50/100 | BB σ=2.5\n"
              f"Просканировано: {len(results)}\n"
              f"🟢 покупок: <b>{n_buy}</b>   🔴 продаж: <b>{n_sell}</b>   👁 наблюдение: <b>{len(watch)}</b>\n")

    def signal_line(r: dict, rank: int | None = None) -> str:
        icon = {"buy": "🟢 ПОКУПКА", "sell": "🔴 ПРОДАЖА",
                "watch": "👁 НАБЛЮДЕНИЕ"}.get(r["verdict"], r["verdict"])
        head = ""
        if rank is not None:
            head = f"{rank:>2}. "
        line = (f"{head}<b>{r['secid']}</b>  {icon}"
                f"  {'✅' if r.get('tf_confirmed') else '⏳'}"
                f"  💪 {strength(r):g}")
        scores = (f"     📊 1Д: B{r['score_buy']}/S{r['score_sell']}"
                  + (f"  ·  1Ч: B{r.get('score_buy_1h', 0)}/S{r.get('score_sell_1h', 0)}"
                     if r.get("score_buy_1h") or r.get("score_sell_1h") else ""))
        line += "\n" + scores + f"\n     💵 {fmt_price(r['price'])}"
        if r["verdict"] in ("buy", "sell") and r.get("sl_tp"):
            line += (f"  ·  🛑 SL {fmt_price(r['sl_tp']['stop_loss'])}"
                     f"  →  🎯 TP {fmt_price(r['sl_tp']['take_profit'])}")
        rules = "; ".join(f"[{s.get('tf', '1Д')}] {s['rule']}"
                          for s in r["signals"][:4]) or "-"
        line += f"\n     💡 {rules}"
        if r.get("sell_blocked_reasons"):
            line += f"\n     ⛔ шорт запрещён: {r['sell_blocked_reasons'][0]}"
        return line

    lines: list[str] = []
    if active:
        lines.append("\n🚨 <b>ВАЖНЫЕ СИГНАЛЫ</b>")
        lines.append("<code>" + "═" * 38 + "</code>")
        for i, r in enumerate(active, 1):
            lines.append(signal_line(r, rank=i))
        lines.append("")
    else:
        lines.append("\n🚨 Активных сигналов нет.\n")

    show_watch = watch[:max(0, top - len(active))]
    if show_watch:
        lines.append("👁 <b>НАБЛЮДЕНИЕ</b> (по очкам)")
        lines.append("<code>" + "═" * 38 + "</code>")
        for i, r in enumerate(show_watch, 1):
            lines.append(signal_line(r, rank=i))
    body = "\n".join(lines)

    # разбивка на сообщения ≤ 4096 по границам блоков
    chunks: list[str] = []
    cur = header
    for block in lines:
        piece = block + "\n"
        if len(cur) + len(piece) > TG_MESSAGE_LIMIT - 100 and cur.strip():
            chunks.append(cur)
            cur = ""
        cur += piece
    if cur.strip():
        chunks.append(cur)
    # нумерация частей, если их несколько
    if len(chunks) > 1:
        chunks = [f"{c}\n<i>({i}/{len(chunks)})</i>" for i, c in enumerate(chunks, 1)]
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
