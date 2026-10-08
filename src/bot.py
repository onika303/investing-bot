"""Инвестиционный Telegram-бот сканера MOEX (aiogram 3.x + APScheduler).

Команды:
    /start /help   — справка
    /status        — состояние данных, последнего скана, планировщика
    /scan [force]  — отчёт по сигналам (кэш last_report.json или перезапуск)
    /signals buy|sell|watch — фильтр кэша
    /asset TICKER  — детали по активу (свечи, индикаторы, правила, SL/TP, доступность сделок)
    /reload_config — перечитать config.yaml без рестарта

Расписание (Europe/Moscow):
    09:45 пн-пт — инкрементальная дозагрузка данных перед открытием
    10:00 пн-пт — полный скан → отчёт в чат
    00:00 ежедневно — вечерний скан+отчёт (после закрытия всех сессий MOEX)
    суббота 12:00 — проверка приближающейся экспирации сезонных фьючерсов

Запуск: python src/bot.py   (нужны BOT_TOKEN и ADMIN_CHAT_ID в .env)
"""
import asyncio
import sys
from datetime import datetime
from pathlib import Path

# allow `python src/bot.py` from repo root
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import BaseFilter, Command, CommandObject, CommandStart
from aiogram.types import Message
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger

from src.config import ConfigError, load_config
from src.database import DatabaseManager
from src.pipeline import (
    check_futures_expiration,
    format_report_text,
    incremental_update,
    load_assets,
    load_cached_report,
    report_age_hours,
    scan_and_report,
)
from src.scanner import InvestmentScanner

DB_PATH = "finance.db"
CONFIG_PATH = "config.yaml"

router = Router()


class AppState:
    """Общее состояние: кэш результатов, блокировка одновременных задач."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.report: dict | None = load_cached_report()
        self.assets = load_assets(CONFIG_PATH)
        self.scan_lock = asyncio.Lock()
        self.update_lock = asyncio.Lock()
        self.last_update_stats: dict | None = None
        self.bot_started_at = datetime.now()

    @property
    def busy(self) -> bool:
        return self.scan_lock.locked() or self.update_lock.locked()


class IsAuthorized(BaseFilter):
    """Приватный режим: доступ только разрешённым чатам (ADMIN_CHAT_ID / ALLOWED_USER_IDS).

    AppState лежит в workflow_data под ключом "app" (имя "state" занято FSMContext).
    Фильтр получает его как kwarg `app`. Неавторизованным — отказ с их chat_id.
    """

    async def __call__(self, message: Message, app: AppState | None = None) -> bool:
        if app is None or app.cfg.is_authorized(message.chat.id):
            return True
        await message.answer(
            f"⛔ Доступ закрыт. Ваш chat_id: <code>{message.chat.id}</code>",
            parse_mode="HTML")
        return False


async def send_chunks(bot: Bot, chat_id: int, chunks: list[str]):
    for ch in chunks:
        try:
            await bot.send_message(chat_id, ch, parse_mode="HTML",
                                   link_preview=False)
        except Exception as e:
            logger.error(f"Не удалось отправить кусок сообщения: {e}")


# ---------------------------------------------------------------- handlers

def _persist_admin_chat_id(chat_id: int) -> bool:
    """Записывает ADMIN_CHAT_ID в .env (PENDING-режим приёмки)."""
    try:
        env_path = Path(".env")
        lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
        replaced = False
        for i, line in enumerate(lines):
            if line.strip().startswith("ADMIN_CHAT_ID"):
                lines[i] = f"ADMIN_CHAT_ID={chat_id}"
                replaced = True
                break
        if not replaced:
            lines.append(f"ADMIN_CHAT_ID={chat_id}")
        env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info(f"ADMIN_CHAT_ID={chat_id} сохранён в .env")
        return True
    except OSError as e:
        logger.error(f"Не удалось сохранить chat_id в .env: {e}")
        return False


@router.message(CommandStart(), IsAuthorized())
async def cmd_start(message: Message, app: AppState | None = None):
    # Режим приёмки (ADMIN_CHAT_ID=PENDING): первый /start назначает админский чат.
    if app is not None and str(app.cfg.admin_chat_id) == "":
        app.cfg.admin_chat_id = message.chat.id
        saved = _persist_admin_chat_id(message.chat.id)
        await message.answer(
            f"✅ Чат <code>{message.chat.id}</code> назначен админским."
            + ("\nЗначение сохранено в .env — после перезапуска бота режим PENDING больше не нужен."
               if saved else
               "\nНе удалось записать в .env — добавьте ADMIN_CHAT_ID вручную."),
            parse_mode="HTML")
    await message.answer(
        "🤖 <b>Инвестиционный сканер MOEX</b>\n\n"
        "Каждый будний день: 09:45 дозагрузка данных, 10:00 утренний отчёт, "
        "00:00 вечерний отчёт.\n\n"
        "Команды:\n"
        "/scan — отчёт по сигналам (/scan force — перезапустить скан)\n"
        "/signals buy|sell|watch — фильтр сигналов\n"
        "/asset TICKER — детали по активу (например, /asset SBER)\n"
        "/status — состояние данных и планировщика\n"
        "/reload_config — перечитать config.yaml",
        parse_mode="HTML")


@router.message(Command("help"), IsAuthorized())
async def cmd_help(message: Message, app: AppState | None = None):
    await cmd_start(message, app)


@router.message(Command("status"), IsAuthorized())
async def cmd_status(message: Message, app: AppState):
    db = DatabaseManager(DB_PATH)
    n_secids = db.fetch_one("SELECT COUNT(DISTINCT secid) FROM raw_candles")[0]
    last_scan = (app.report["generated_at"].replace("T", " ")
                 if app.report else "нет")
    lines = [
        "📡 <b>Статус</b>",
        f"Активов в config: {sum(len(v) for v in app.assets.values())} "
        f"(акции {len(app.assets.get('stocks', []))}, "
        f"фьючерсы {len(app.assets.get('futures', []))})",
        f"Уникальных secid в БД: {n_secids}",
        f"Последний скан: {last_scan}",
        f"Идёт задача: {'да ⏳' if app.busy else 'нет'}",
        f"Бот запущен: {app.bot_started_at:%d.%m %H:%M}",
    ]
    if app.last_update_stats:
        lines.append(f"Последняя дозагрузка: {app.last_update_stats}")
    await message.answer("\n".join(lines), parse_mode="HTML")


async def do_full_scan(app: AppState, bot: Bot | None, chat_id: int | None,
                       do_update: bool) -> dict:
    """Единая точка запуска скана с guard-локом. Возвращает report."""
    if app.scan_lock.locked():
        raise RuntimeError("Скан уже выполняется, подождите завершения.")
    async with app.scan_lock:
        report, stats = await asyncio.to_thread(
            scan_and_report, DB_PATH, CONFIG_PATH, do_update)
        app.report = report
        app.last_update_stats = stats
    if bot and chat_id:
        await send_chunks(bot, chat_id, format_report_text(report))
    return report


@router.message(Command("scan"), IsAuthorized())
async def cmd_scan(message: Message, app: AppState):
    force = bool(message.text and "force" in message.text.lower())

    if not force and app.report:
        age = report_age_hours(app.report)
        if age <= app.cfg.report_cache_hours:
            return await send_chunks(message.bot, message.chat.id,
                                     format_report_text(app.report))

    if app.busy:
        return await message.answer("⏳ Уже выполняется задача, попробуйте позже.")

    total = sum(len(v) for v in app.assets.values())
    progress = await message.answer(f"⏳ Запускаю полный скан (~{total} активов)...")
    try:
        report = await do_full_scan(app, message.bot, message.chat.id,
                                    do_update=True)
        dur = report.get("duration_sec")
        await progress.edit_text(f"✅ Скан завершён за {dur} с.")
    except Exception as e:
        logger.exception("Скан упал")
        await progress.edit_text(f"❌ Ошибка скана: {e}")
        await notify_admin(message.bot, app, f"❌ Ошибка ручного /scan: {e}")


@router.message(Command("signals"), IsAuthorized())
async def cmd_signals(message: Message, command: CommandObject, app: AppState):
    wanted = (command.args or "").strip().lower()
    if wanted not in ("buy", "sell", "watch"):
        return await message.answer("Использование: /signals buy|sell|watch")
    if not app.report:
        return await message.answer("Нет кэша отчёта — сначала /scan")
    rows = [r for r in app.report["results"] if r["verdict"] == wanted]
    icon = {"buy": "🟢", "sell": "🔴", "watch": "👁"}[wanted]
    if not rows:
        return await message.answer(f"{icon} Сигналов '{wanted}' нет.")
    text = "\n".join(
        f"<code>{r['secid']:<7}</code> B{r['score_buy']}/S{r['score_sell']} "
        f"@ {r['price']} — " + "; ".join(s["rule"] for s in r["signals"][:3])
        for r in sorted(rows, key=lambda x: -max(x["score_buy"], x["score_sell"])))
    await message.answer(f"{icon} <b>{wanted.upper()}</b> ({len(rows)}):\n{text}",
                         parse_mode="HTML")


@router.message(Command("asset"), IsAuthorized())
async def cmd_asset(message: Message, command: CommandObject, app: AppState):
    secid = (command.args or "").strip().upper()
    if not secid:
        return await message.answer("Использование: /asset SBER")
    asset_type = next((t for t, lst in app.assets.items() if secid in lst), None)
    if not asset_type:
        asset_type = "stocks"  # попробуем как акцию; вердикт «нет данных» если не так

    res = await asyncio.to_thread(
        InvestmentScanner(DatabaseManager(DB_PATH)).scan_asset, secid, asset_type)
    if not res:
        return await message.answer(f"Нет данных по {secid} в БД.")

    rules = "\n".join(f"• {s['rule']} ({s['type']}, вес {s['weight']})"
                      for s in res["signals"]) or "правил не сработало"
    sl_tp = ""
    if res.get("sl_tp"):
        sl_tp = (f"\n🎯 SL: {res['sl_tp']['stop_loss']} | "
                 f"TP: {res['sl_tp']['take_profit']} (ATR-based)")
    blocked = ("\n⛔ Шорт запрещён: " + "; ".join(res["sell_blocked_reasons"])
               if res.get("sell_blocked_reasons") else "")
    access = res.get("trade_access", {})
    allowed_txt = "?" if not access else ("да" if access.get("short_allowed") else "нет")
    await message.answer(
        f"<b>{secid}</b> ({asset_type}) на {res['date']}\n"
        f"Цена: {res['price']} | Вердикт: {res['verdict'].upper()} "
        f"B{res['score_buy']}/S{res['score_sell']}{sl_tp}{blocked}\n"
        f"Шорт доступен: {allowed_txt}\n\n{rules}",
        parse_mode="HTML")


@router.message(Command("reload_config"), IsAuthorized())
async def cmd_reload(message: Message, app: AppState):
    try:
        app.assets = await asyncio.to_thread(load_assets, CONFIG_PATH)
        return await message.answer(
            f"♻️ config.yaml перечитан: "
            f"{sum(len(v) for v in app.assets.values())} активов.")
    except Exception as e:
        return await message.answer(f"❌ Ошибка чтения config.yaml: {e}")


async def notify_admin(bot: Bot, app: AppState, text: str):
    if str(app.cfg.admin_chat_id) == "":  # PENDING-режим приёмки — слать некуда
        logger.info(f"Уведомление (чат не назначен): {text[:120]}")
        return
    try:
        await bot.send_message(app.cfg.admin_chat_id, text)
    except Exception as e:
        logger.error(f"Не удалось уведомить админа: {e}")


# ---------------------------------------------------------------- schedule

def build_scheduler(app: AppState, bot: Bot) -> AsyncIOScheduler:
    sched = AsyncIOScheduler(timezone=app.cfg.timezone)
    chat = lambda: app.cfg.admin_chat_id  # читаем динамически (PENDING → /start назначит чат)

    async def job_morning_update():
        if app.update_lock.locked():
            logger.warning("Пропуск 09:45: уже идёт загрузка")
            return
        async with app.update_lock:
            try:
                stats = await asyncio.to_thread(
                    incremental_update, app.assets, DB_PATH)
                app.last_update_stats = stats
                logger.info(f"09:45 update done: {stats}")
            except Exception as e:
                logger.exception("Утренняя загрузка упала")
                await notify_admin(bot, app, f"❌ Ошибка загрузки 09:45: {e}")

    async def job_morning_scan():
        try:
            await do_full_scan(app, bot, chat, do_update=False)
        except RuntimeError as e:
            await notify_admin(bot, app, f"ℹ️ Утренний скан пропущен: {e}")
        except Exception as e:
            logger.exception("Утренний скан упал")
            await notify_admin(bot, app, f"❌ Ошибка утреннего скана 10:00: {e}")

    async def job_evening_scan():
        # 00:00 MSK — «вечерний» отчёт после закрытия всех сессий MOEX
        # (основная до ~23:50); do_update=True подтянет всё, что появилось за сутки.
        try:
            await do_full_scan(app, bot, chat, do_update=True)
        except RuntimeError as e:
            await notify_admin(bot, app, f"ℹ️ Вечерний скан пропущен: {e}")
        except Exception as e:
            logger.exception("Вечерний скан упал")
            await notify_admin(bot, app, f"❌ Ошибка вечернего скана 00:00: {e}")

    async def job_saturday_hygiene():
        try:
            assets = await asyncio.to_thread(load_assets, CONFIG_PATH)
            warnings = check_futures_expiration(assets)
            if warnings:
                await bot.send_message(
                    chat, "🧽 Субботняя гигиена config.yaml:\n"
                    + "\n".join(warnings))
            else:
                logger.info("Экспираций в ближайшие 4 недели не найдено")
        except Exception as e:
            logger.exception("Субботняя проверка упала")
            await notify_admin(bot, app, f"❌ Ошибка субботней проверки: {e}")

    tz = app.cfg.timezone
    sched.add_job(job_morning_update,
                  CronTrigger(hour=9, minute=45, day_of_week="mon-fri", timezone=tz),
                  id="update_0945", misfire_grace_time=3600, coalesce=True)
    sched.add_job(job_morning_scan,
                  CronTrigger(hour=10, minute=0, day_of_week="mon-fri", timezone=tz),
                  id="scan_1000", misfire_grace_time=3600, coalesce=True)
    sched.add_job(job_evening_scan,
                  CronTrigger(hour=0, minute=0, timezone=tz),
                  id="scan_0000", misfire_grace_time=3600, coalesce=True)
    sched.add_job(job_saturday_hygiene,
                  CronTrigger(day_of_week="sat", hour=12, minute=0, timezone=tz),
                  id="hygiene_sat", misfire_grace_time=86400, coalesce=True)
    return sched


async def main():
    try:
        cfg = load_config()
    except ConfigError as e:
        print(f"Конфигурация: {e}", file=sys.stderr)
        sys.exit(1)

    bot = Bot(token=cfg.bot_token)
    dp = Dispatcher()
    app = AppState(cfg)
    # Ключ "app": в aiogram 3.x имя "state" зарезервировано под FSMContext,
    # поэтому AppState инжектится в хендлеры/фильтры как отдельный kwarg.
    dp.workflow_data.update({"app": app})
    dp.include_router(router)

    scheduler = build_scheduler(app, bot)
    scheduler.start()
    for job in scheduler.get_jobs():
        logger.info(f"Планировщик: {job.id} → следующий запуск {job.next_run_time}")

    logger.info("Бот запущен, включаю long polling…")
    await bot.delete_webhook(drop_pending_updates=False)  # не теряем команды, присланные пока бот лежал
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Бот остановлен")
