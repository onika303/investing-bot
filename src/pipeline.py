"""Фоновый конвейер: инкрементальная загрузка данных MOEX + полный скан + отчёт.

Логика вынесена из CLI (scan.py) в переиспользуемый модуль, который одинаково
вызывается и планировщиком бота (APScheduler), и вручную (scripts/run_report_once.py).

Все функции синхронные (MOEXLoader внутри поднимает свой asyncio loop) —
в асинхронном коде бота запускать через asyncio.to_thread.
"""
import json
import os
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
    Если хвост в БД уже свежее сегодняшнего дня (загрузка после 23:00 MSK или
    повторный прогон в тот же день) — актив пропускается (skipped), чтобы не
    слать заведомо пустой запрос с from > till.
    Пустые ответы (выходные/праздники/нет торгов) обрабатываются штатно.
    Возвращает сводку {updated, new, empty, errors, skipped}.
    """
    db = DatabaseManager(db_path)
    loader = MOEXLoader(db)
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    stats = {"updated": 0, "new": 0, "empty": 0, "errors": 0, "skipped": 0}

    for asset_type, secids in assets.items():
        for secid in secids:
            # --- дневной таймфрейм ---
            try:
                last_ts = db.get_last_ts(secid, interval=24)
                if last_ts:
                    from_dt = datetime.fromtimestamp(last_ts) + timedelta(days=1)
                    if from_dt.date() > now.date():
                        # данные уже актуальны (загрузка позже 23:00 MSK или повторный
                        # прогон в тот же день) — запрос был бы пустым (from > till)
                        stats["skipped"] += 1
                    else:
                        from_date = from_dt.strftime("%Y-%m-%d")
                        stats["updated"] += 1
                        df = loader.load_asset(secid, asset_type, from_date, today,
                                               interval=24, end_on_load=True)
                        if df is None or len(df) == 0:
                            stats["empty"] += 1  # праздники/нет торгов — штатно
                else:
                    from_date = (now - timedelta(days=default_days)
                                 ).strftime("%Y-%m-%d")
                    stats["new"] += 1
                    loader.load_asset(secid, asset_type, from_date, today,
                                      interval=24, end_on_load=True)
            except Exception as e:
                stats["errors"] += 1
                logger.error(f"Инкрементальная загрузка {secid} (1д): {e}")

            # --- часовой таймфрейм ---
            try:
                last_h = db.get_last_ts(secid, interval=60)
                if last_h:
                    from_h = datetime.fromtimestamp(last_h) + timedelta(hours=1)
                    if from_h >= now:
                        stats["skipped"] += 1  # часовки свежее текущего часа — не тянем
                    else:
                        stats["updated"] += 1
                        dfh = loader.load_asset(secid, asset_type,
                                                from_h.strftime("%Y-%m-%d"), today,
                                                interval=60, end_on_load=True)
                        if dfh is None or len(dfh) == 0:
                            stats["empty"] += 1
                else:
                    stats["new"] += 1
                    loader.load_asset(secid, asset_type,
                                      (now - timedelta(days=hours_days)
                                       ).strftime("%Y-%m-%d"), today,
                                      interval=60, end_on_load=True)
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


def report_matches_config(report: dict, assets: dict, config_path: str = "config.yaml") -> bool:
    """Актуален ли кэш отчёта относительно ТЕКУЩЕГО config.yaml.

    Отчёт считается несогласованным, если mtime config.yaml новее времени
    генерации отчёта либо состав активов в отчёте расходится со списком
    из файла конфигурации (например, список сократили, а бот отдаёт старый
    кэш со «старым количеством активов»)."""
    import yaml  # локальный импорт: модуль используется и в лёгких тестах
    try:
        gen = datetime.fromisoformat(report["generated_at"])
        if datetime.fromtimestamp(os.path.getmtime(config_path)) > gen:
            return False
        with open(config_path, encoding="utf-8") as f:
            cur = yaml.safe_load(f)["assets"]
        cached = {r["secid"] for r in report["results"]}
        wanted = {s for lst in cur.values() for s in lst}
        return cached == wanted
    except Exception:
        return False


def fmt_price(v) -> str:
    try:
        return f"{float(v):,.2f}".replace(",", " ")
    except (TypeError, ValueError):
        return "-"


def _horizons_of(r: dict) -> dict:
    """Горизонты результата. Старые кэши без поля 'horizons' — считаем 1Д."""
    h = r.get("horizons")
    if h:
        return h
    return {"1d": {
        "tf": "1d", "tf_label": "1Д", "horizon": "внутри недели",
        "verdict": r.get("verdict", "neutral"),
        "score_buy": r.get("score_buy", 0), "score_sell": r.get("score_sell", 0),
        "strength": max(r.get("score_buy", 0), r.get("score_sell", 0)),
        "price": r.get("price"), "signals": r.get("signals", []),
        "sl_tp": r.get("sl_tp"),
        "sell_blocked_reasons": r.get("sell_blocked_reasons"),
    }}


def format_report_text(report: dict, top: int = 25) -> list[str]:
    """Отчёт сканера для Telegram. Аналитика в разрезе ГОРИЗОНТОВ ПЛАНИРОВАНИЯ:
    1Ч = сделки на 1-2 дня, 1Д = сделка внутри недели; сигналы горизонтов НЕ
    смешиваются. Внутри каждого горизонта: сверху важные сигналы (buy/sell),
    ниже — наблюдение; оба списка отсортированы по очкам (не по алфавиту)."""
    results = report["results"]

    # раскладываем активы по горизонтам
    per_tf: dict[str, list[tuple[dict, dict]]] = {}
    for r in results:
        for tf, h in _horizons_of(r).items():
            per_tf.setdefault(tf, []).append((r, h))

    header = (f"📡 <b>Сканер MOEX · {report['generated_at'][:16].replace('T', ' ')} МСК</b>\n"
              f"<code>{'─' * 38}</code>\n"
              f"Индикаторы: EMA 13/21/50/100 | BB σ=2.5\n"
              f"Просканировано активов: {len(results)}\n")

    def active_count(items) -> tuple[int, int, int]:
        nb = sum(1 for _, h in items if h["verdict"] == "buy")
        ns = sum(1 for _, h in items if h["verdict"] == "sell")
        nw = sum(1 for _, h in items if h["verdict"] == "watch")
        return nb, ns, nw

    tf_meta = {"1h": ("⚡", "1–2 дня"), "1d": ("📅", "внутри недели")}
    lines: list[str] = [header]
    for tf in ("1d", "1h"):  # дневной — основной, показываем первым
        items = per_tf.get(tf)
        if not items:
            continue
        icon, horizon = tf_meta.get(tf, ("•", tf))
        label = {"1d": "1Д", "1h": "1Ч"}.get(tf, tf)
        nb, ns, nw = active_count(items)
        lines.append(f"\n{icon} <b>ГОРИЗОНТ {label} — {horizon}</b>")
        lines.append(f"🟢 покупка: <b>{nb}</b>   🔴 продажа: <b>{ns}</b>   👁 наблюдение: <b>{nw}</b>")
        lines.append("<code>" + "═" * 38 + "</code>")

        active = sorted([p for p in items if p[1]["verdict"] in ("buy", "sell")],
                        key=lambda p: p[1]["strength"], reverse=True)
        watch = sorted([p for p in items if p[1]["verdict"] == "watch"],
                       key=lambda p: p[1]["strength"], reverse=True)

        def signal_line(r: dict, h: dict, rank: int) -> str:
            v = h["verdict"]
            sym = {"buy": "🟢 ПОКУПКА", "sell": "🔴 ПРОДАЖА",
                   "watch": "👁 наблюдение"}.get(v, v)
            line = (f"{rank:>2}. <b>{r['secid']}</b>  {sym}  💪 {h['strength']:g}\n"
                    f"     📊 B{h['score_buy']}/S{h['score_sell']}"
                    f"  ·  💵 {fmt_price(h.get('price') or r['price'])}")
            if v in ("buy", "sell") and h.get("sl_tp"):
                line += (f"\n     🛑 SL {fmt_price(h['sl_tp']['stop_loss'])}"
                         f"  →  🎯 TP {fmt_price(h['sl_tp']['take_profit'])}")
            rules = "; ".join(s["rule"] for s in h.get("signals", [])[:4]) or "-"
            line += f"\n     💡 {rules}"
            if h.get("sell_blocked_reasons"):
                line += f"\n     ⛔ шорт запрещён: {h['sell_blocked_reasons'][0]}"
            return line

        if active:
            lines.append("\n🚨 <b>ВАЖНЫЕ СИГНАЛЫ</b>")
            for i, (r, h) in enumerate(active, 1):
                lines.append(signal_line(r, h, i))
        else:
            lines.append("\n🚨 Активных сигналов нет.")

        show_watch = watch[:max(0, top - len(active))]
        if show_watch:
            lines.append("\n👁 <b>НАБЛЮДЕНИЕ</b> (по очкам)")
            for i, (r, h) in enumerate(show_watch, 1):
                lines.append(signal_line(r, h, i))
        lines.append("")

    body_lines = lines

    # разбивка на сообщения ≤ 4096 по границам блоков
    chunks: list[str] = []
    cur = ""
    for block in body_lines:
        piece = block if block.endswith("\n") else block + "\n"
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
