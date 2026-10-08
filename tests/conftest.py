"""Общие фикстуры pytest для тестов сканера."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.scanner import InvestmentScanner

# чтобы `import src.*` работал из каталога tests/
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def make_ohlcv(closes, start='2025-01-01', volumes=None):
    """Синтетический OHLCV-датафрейм по массиву цен закрытия."""
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    dates = pd.bdate_range(start=start, periods=n)
    vols = volumes if volumes is not None else [1000] * n
    return pd.DataFrame({
        'ts': dates.astype('int64') // 10**9,
        'open': closes,
        'high': closes * 1.005,
        'low': closes * 0.995,
        'close': closes,
        'volume': list(vols),
        'oi': [0] * n,
        'date': dates,
    })


@pytest.fixture
def tmp_db(tmp_path):
    """Инициализированная временная SQLite-база."""
    from src.database import DatabaseManager
    return DatabaseManager(str(tmp_path / 'test.db'))


@pytest.fixture
def trending_up_df():
    """Восходящий тренд с лёгким шумом (без дивергенций)."""
    rng = np.random.default_rng(42)
    closes = np.linspace(100, 150, 120) + rng.normal(0, 0.3, 120)
    return make_ohlcv(closes)


@pytest.fixture
def v_shape_df():
    """V-образный разворот: падение -> рост (RSI oversold в хвосте)."""
    down = np.linspace(150, 100, 80)
    up = np.linspace(100, 118, 40)
    return make_ohlcv(np.concatenate([down, up]))


@pytest.fixture
def scanner_rules_only(tmp_db):
    """Сканер только для проверки статических правил (без сети)."""
    from unittest.mock import patch
    with patch('src.scanner.SHORT_RESTRICTIONS_FILE', Path('/nonexistent.json')):
        return InvestmentScanner(tmp_db, short_restrictions={})
