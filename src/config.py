"""Загрузка конфигурации бота из .env (python-dotenv).

Секреты (BOT_TOKEN, ADMIN_CHAT_ID) никогда не коммитятся: см. .env.example.
"""
import os
from pathlib import Path
from dataclasses import dataclass, field

from dotenv import load_dotenv
from loguru import logger


class ConfigError(RuntimeError):
    """Ошибка конфигурации окружения."""


@dataclass
class BotConfig:
    bot_token: str
    admin_chat_id: int | str  # числовой id либо @username (публичные каналы/чаты с ботом-админом)
    allowed_user_ids: list[int] = field(default_factory=list)
    timezone: str = "Europe/Moscow"
    report_cache_hours: float = 6.0  # насколько «свежим» считается last_report для /scan без force

    def is_authorized(self, chat_id: int | str) -> bool:
        """Приватный режим: доступ только админу и явно разрешённым чатам.

        chat_id может быть числом (личные чаты) или "@username" (публичный
        канал/группа) — сравниваем как строки, чтобы поддерживать оба варианта.

        Пустой admin_chat_id = режим приёмки (ADMIN_CHAT_ID=PENDING): авторизуем
        всех, логгируя chat_id, чтобы потом вписать его в .env.
        """
        if str(self.admin_chat_id) == "":
            logger.info(f"PENDING-режим: входящий chat_id = {chat_id}")
            return True
        if str(chat_id).lower() == str(self.admin_chat_id).lower():
            return True
        return chat_id in self.allowed_user_ids


def _parse_int_list(raw: str) -> list[int]:
    out = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            raise ConfigError(f"Не удалось разобрать ID в ALLOWED_USER_IDS: {part!r}")
    return out


def load_config(env_file: str | None = None) -> BotConfig:
    """Читает .env и возвращает BotConfig; при отсутствии обязательных переменных — ConfigError."""
    path = Path(env_file or os.getenv("ENV_FILE", ".env"))
    # Явно читаем файл (не полагаясь на уже выставленные переменные окружения),
    # чтобы load_config в тестах и при смене .env был детерминирован.
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

    token = os.getenv("BOT_TOKEN", "").strip()
    if not token or ":" not in token:
        raise ConfigError(
            "BOT_TOKEN не задан или выглядит некорректно. "
            "Получите токен у @BotFather (/newbot) и пропишите его в файл .env "
            "(см. образец .env.example)."
        )

    chat_raw = os.getenv("ADMIN_CHAT_ID", "").strip()
    if not chat_raw:
        raise ConfigError(
            "ADMIN_CHAT_ID не задан. Узнайте свой chat_id (напишите @userinfobot "
            "или команду /id боту @getmyid_bot) и добавьте в .env."
        )
    if chat_raw.upper() == "PENDING":
        # Режим первичной приёмки: чат ещё неизвестен — бот подхватит его
        # из первого полученного апдейта (см. src/bot.py).
        admin_chat_id: int | str = ""
    elif chat_raw.startswith("@"):
        # Публичный канал/группа по username (бот должен быть там админом,
        # а пользователь — сначала отправить боту /start в этом чате).
        admin_chat_id: int | str = chat_raw
    else:
        try:
            admin_chat_id = int(chat_raw)
        except ValueError:
            raise ConfigError(
                f"ADMIN_CHAT_ID должен быть числом или '@username', получено: {chat_raw!r}"
            )

    try:
        allowed = _parse_int_list(os.getenv("ALLOWED_USER_IDS", ""))
    except ConfigError as e:
        raise ConfigError(str(e))

    tz = os.getenv("TIMEZONE", "Europe/Moscow").strip() or "Europe/Moscow"

    cache_h = 6.0
    if os.getenv("REPORT_CACHE_HOURS"):
        try:
            cache_h = float(os.getenv("REPORT_CACHE_HOURS"))
        except ValueError:
            raise ConfigError("REPORT_CACHE_HOURS должен быть числом")

    return BotConfig(
        bot_token=token,
        admin_chat_id=admin_chat_id,
        allowed_user_ids=allowed,
        timezone=tz,
        report_cache_hours=cache_h,
    )
