"""Инвестиционный сканер: индикаторы + дивергенции + правила -> сигналы со скорингом."""
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from loguru import logger

from src.database import DatabaseManager
from src.indicators import IndicatorCalculator
from src.divergence import DivergenceDetector

# --- Ограничения доступа к сделкам (short selling restrictions) ---
# На MOEX продажа (шорт) некоторых бумаг запрещена или ограничена:
#   * new_emission — бумаги новых размещений/ИПО: короткая история, узкий free-float,
#     сопряжение с рисками нового эмитента, шорт часто недоступен;
#   * sanctioned — бумаги под санкциями (например LVHK): иностранцам продажи запрещены,
#     у российских брокеров ограничения на маржинальные (шорт) позиции.
SHORT_RESTRICTIONS_FILE = Path(__file__).resolve().parent.parent / 'short_restrictions.json'


def load_short_restrictions(path: Optional[Path] = None) -> Dict[str, str]:
    """Читает файл ограничений; если файла нет — возвращаем пустой словарь."""
    p = path or SHORT_RESTRICTIONS_FILE
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception as e:
        logger.error(f"Не удалось прочитать {p}: {e}")
        return {}


class InvestmentScanner:
    """Сканирует актив по дневным (1д) и часовым (1ч) свечам из БД -> сигналы со скорингом."""

    MIN_ROWS = 60  # минимум свечей для осмысленного расчёта индикаторов
    NEW_EMISSION_DAYS = 365  # бумага считается "новой эмиссией", если торгуется < года
    ISS_TIMEOUT = 15
    TIMEFRAMES = ('1d', '1h')  # таймфреймы скана
    TF_INTERVAL = {'1d': 24, '1h': 60}
    TF_LABEL = {'1d': '1Д', '1h': '1Ч'}
    # Горизонты планирования: каждый ТФ = отдельный горизонт, СМЕШИВАТЬ
    # сигналы между ними для подтверждения сетапа НЕЛЬЗЯ (решение от 10.10.2026):
    #   1Ч -> сделки на 1-2 дня (скальп/свиг), 1Д -> сделка внутри недели (свинг).
    HORIZON = {'1h': '1–2 дня', '1d': 'внутри недели'}
    # Пороги вердикта — теперь ОТДЕЛЬНЫЕ НА ГОРИЗОНТ (без смешивания ТФ):
    SIGNAL_THRESHOLD = 30  # порог активного сигнала внутри одного горизонта
    SOLO_THRESHOLD = 30    # совместимость со старыми тестами/кодом
    # Сила сигнала для ранжирования — считается ОТДЕЛЬНО на каждый горизонт
    # (см. _horizon_result); WEIGHT_* сохранены только для совместимости.

    def __init__(self, db: DatabaseManager,
                 short_restrictions: Optional[Dict[str, str]] = None):
        self.db = db
        self.short_restrictions = (load_short_restrictions()
                                   if short_restrictions is None else short_restrictions)
        self._issue_cache: Dict[str, dict] = {}

    # ---------- доступность актива к сделке ----------
    def _fetch_issue_info(self, secid: str) -> dict:
        """ISS /securities/{secid}.json: дата начала торгов, статус листинга.

        Только сетевой GET (без JS-парсинга), результат кэшируется в памяти и
        на диске (src/.cache/issue_info.json), чтобы не дёргать ISS при каждом прогоне.
        """
        if secid in self._issue_cache:
            return self._issue_cache[secid]
        # путь кэша можно переопределить (тесты); по умолчанию src/.cache/issue_info.json
        disk_cache = Path(getattr(self, '_disk_issue_cache',
                                  Path(__file__).resolve().parent / '.cache' / 'issue_info.json'))
        if disk_cache.exists():
            try:
                cached = json.loads(disk_cache.read_text(encoding='utf-8'))
                if secid in cached:
                    self._issue_cache[secid] = cached[secid]
                    return cached[secid]
            except Exception:
                pass
        info = {}
        url = f"https://iss.moex.com/iss/securities/{secid}.json?iss.meta=off"
        try:
            resp = requests.get(url, timeout=self.ISS_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            rows = data.get('securities', {}).get('data', [])
            cols = data.get('securities', {}).get('columns', [])
            if rows and cols:
                d = dict(zip(cols, rows[0]))
                info = {'issuedate': d.get('ISSUEDATE'),
                        'listeduntil': d.get('LISTEDTILL'),
                        'status': d.get('CURRENTSTATUS')}
            cached = (json.loads(disk_cache.read_text(encoding='utf-8'))
                      if disk_cache.exists() else {})
            cached[secid] = info
            disk_cache.parent.mkdir(parents=True, exist_ok=True)
            disk_cache.write_text(json.dumps(cached, ensure_ascii=False), encoding='utf-8')
        except Exception as e:
            logger.warning(f"ISS issue info для {secid} недоступна: {e}")
        self._issue_cache[secid] = info
        return info

    def check_trade_access(self, secid: str, asset_type: str,
                           first_date: pd.Timestamp) -> Dict:
        """Возвращает {'short_allowed': bool, 'reasons': [str]}.

        Источники ограничений:
          1) статический файл short_restrictions.json (известные санкционные/рисковые бумаги);
          2) автоматическая эвристика: новая эмиссия (< NEW_EMISSION_DAYS дней торгов).
        """
        reasons: List[str] = []
        allowed = True

        restr = self.short_restrictions.get(secid)
        if restr:
            allowed = False
            reasons.append(f"ограничение доступа к шорту ({restr})")

        if asset_type == 'stocks':
            issuedate = self._fetch_issue_info(secid).get('issuedate')
            recent = None
            if issuedate:
                try:
                    # ISS отдаёт tz-naive дату -> сравниваем с naive "сегодня" МСК
                    today_msk = pd.Timestamp.now(tz='Europe/Moscow').tz_localize(None).normalize()
                    days = (today_msk - pd.Timestamp(issuedate)).days
                    recent = days < self.NEW_EMISSION_DAYS
                except Exception:
                    recent = None
            if recent is None and first_date is not None:
                recent = (pd.Timestamp.now() - first_date).days < self.NEW_EMISSION_DAYS
            if recent:
                allowed = False
                reasons.append("новая эмиссия: история торгов < 1 года, "
                               "шорт недоступен/сопряжён с повышенными рисками")

        return {'short_allowed': allowed, 'reasons': reasons}

    # ---------- данные ----------
    def load_candles(self, secid: str, limit_rows: int = 500,
                     interval: int = 24) -> Optional[pd.DataFrame]:
        rows = self.db.fetch_all(
            """SELECT ts, open, high, low, close, volume, oi
               FROM raw_candles WHERE secid = ? AND interval = ?
               ORDER BY ts DESC LIMIT ?""",
            (secid, interval, limit_rows),
        )
        if len(rows) < self.MIN_ROWS:
            return None
        df = pd.DataFrame(rows, columns=['ts', 'open', 'high', 'low', 'close', 'volume', 'oi'])
        df = df.sort_values('ts').reset_index(drop=True)
        df['date'] = pd.to_datetime(df['ts'], unit='s')
        return df

    # ---------- правила поверх индикаторов ----------
    @staticmethod
    def _rule_signals(df: pd.DataFrame) -> List[Dict]:
        sig: List[Dict] = []
        last = df.iloc[-1]
        prev = df.iloc[-2]

        # Пересечения EMA по набору 13/21/50/100:
        #   13x21 — быстрый кросс (вход), 21x50 и 50x100 — трендовые (медленнее, весомее)
        for fast, slow, w in ((13, 21, 25), (21, 50, 20), (50, 100, 15)):
            fc, sc = f'ema_{fast}', f'ema_{slow}'
            if {fc, sc} <= set(df.columns):
                if prev[fc] <= prev[sc] and last[fc] > last[sc]:
                    sig.append({'rule': f'EMA {fast}x{slow} golden cross',
                                'type': 'buy', 'weight': w})
                elif prev[fc] >= prev[sc] and last[fc] < last[sc]:
                    sig.append({'rule': f'EMA {fast}x{slow} death cross',
                                'type': 'sell', 'weight': w})

        # Конфлюэнция тренда: цена над/под всеми EMA 13/21/50/100
        ema_cols = [f'ema_{p}' for p in (13, 21, 50, 100) if f'ema_{p}' in df.columns]
        if len(ema_cols) == 4:
            if all(last['close'] > last[c] for c in ema_cols):
                sig.append({'rule': 'Close above EMA 13/21/50/100',
                            'type': 'buy', 'weight': 15})
            elif all(last['close'] < last[c] for c in ema_cols):
                sig.append({'rule': 'Close below EMA 13/21/50/100',
                            'type': 'sell', 'weight': 15})

        # RSI-зоны
        rsi_col = 'rsi_14' if 'rsi_14' in df.columns else None
        if rsi_col:
            if last[rsi_col] < 30:
                sig.append({'rule': f"RSI oversold ({last[rsi_col]:.1f})", 'type': 'buy', 'weight': 15})
            elif last[rsi_col] > 70:
                sig.append({'rule': f"RSI overbought ({last[rsi_col]:.1f})", 'type': 'sell', 'weight': 15})

        # Bollinger breakout
        if {'bb_lower', 'bb_upper'} <= set(df.columns):
            if last['close'] < last['bb_lower']:
                sig.append({'rule': 'Close below lower BB', 'type': 'buy', 'weight': 10})
            elif last['close'] > last['bb_upper']:
                sig.append({'rule': 'Close above upper BB', 'type': 'sell', 'weight': 10})

        # ADX — сила тренда (не направление, усиливает существующий сигнал)
        if 'adx_14' in df.columns and last['adx_14'] > 25:
            sig.append({'rule': f"Strong trend ADX={last['adx_14']:.0f}", 'type': 'boost', 'weight': 10})

        return sig

    # ---------- основной метод: скан по двум ТФ (1д + 1ч) ----------
    def _scan_timeframe(self, df: pd.DataFrame) -> Dict:
        """Индикаторы + дивергенции + правила для одного набора свечей."""
        df = IndicatorCalculator(df).compute_all()
        detector = DivergenceDetector(df)
        divergences = detector.scan_all_indicators(lookback=50)

        signals = self._rule_signals(df)
        for ind_name, divs in divergences.items():
            for d in divs:
                signals.append({
                    'rule': f"{ind_name.upper()} {d.get('kind', 'divergence')}",
                    'type': d.get('type', 'watch'),
                    'weight': 30,
                })

        score_buy = sum(s['weight'] for s in signals if s['type'] == 'buy')
        score_sell = sum(s['weight'] for s in signals if s['type'] == 'sell')
        boost = sum(s['weight'] for s in signals if s['type'] == 'boost')
        return {'df': df, 'signals': signals,
                'score_buy': score_buy + boost, 'score_sell': score_sell + boost}

    # ---------- вердикт в разрезе ОДНОГО горизонта ----------
    def _horizon_result(self, secid: str, asset_type: str, tf: str,
                        scan: Dict, access: Dict) -> Optional[Dict]:
        """Сигналы одного таймфрейма = отдельный горизонт планирования.
        Ничего не смешивается с другими ТФ: вердикт, сила и SL/TP считаются
        только по свечам этого горизонта."""
        df = scan['df']
        if len(df) < self.MIN_ROWS:
            return None
        price = float(df['close'].iloc[-1])
        atr = float(df['atr_14'].iloc[-1]) if 'atr_14' in df.columns else None
        score_buy, score_sell = scan['score_buy'], scan['score_sell']
        signals = scan['signals']

        verdict = 'watch' if signals else 'neutral'
        sell_blocked = (not access['short_allowed']
                        and score_sell > score_buy
                        and score_sell >= self.SIGNAL_THRESHOLD)
        if sell_blocked:
            verdict = 'watch'
        elif score_buy >= self.SIGNAL_THRESHOLD and score_buy > score_sell:
            verdict = 'buy'
        elif score_sell >= self.SIGNAL_THRESHOLD and score_sell > score_buy:
            verdict = 'sell'

        res = {
            'tf': tf,
            'tf_label': self.TF_LABEL[tf],
            'horizon': self.HORIZON[tf],
            'date': str(df['date'].iloc[-1]),
            'price': round(price, 4),
            'signals': [{**s, 'tf': self.TF_LABEL[tf]} for s in signals],
            'score_buy': score_buy,
            'score_sell': score_sell,
            'verdict': verdict,
            # Сила — строго внутри горизонта (для ранжирования списков)
            'strength': round(max(score_buy, score_sell), 1),
        }
        if sell_blocked:
            res['sell_blocked_reasons'] = access['reasons']
        if verdict in ('buy', 'sell') and atr:
            res['sl_tp'] = {
                'stop_loss': round(price - 2 * atr, 4) if verdict == 'buy'
                             else round(price + 2 * atr, 4),
                'take_profit': round(price + 3 * atr, 4) if verdict == 'buy'
                               else round(price - 3 * atr, 4),
            }
        return res

    def scan_asset(self, secid: str, asset_type: str) -> Optional[Dict]:
        tf_results: Dict[str, Dict] = {}
        first_dates = []
        for tf in self.TIMEFRAMES:
            df = self.load_candles(secid, interval=self.TF_INTERVAL[tf])
            if df is None:
                continue  # часовиков может не быть в БД — тогда только 1д
            try:
                tf_results[tf] = self._scan_timeframe(df)
                first_dates.append(str(df['date'].iloc[0]))
            except Exception as e:
                logger.error(f"Скан {secid} ({tf}): {e}")
        if not tf_results:
            return None

        # Доступность шорта проверяем один раз на актив (по самой длинной истории)
        daily = tf_results.get('1d') or tf_results.get('1h')
        access = self.check_trade_access(
            secid, asset_type, pd.Series(daily['df']['date']).iloc[0])

        horizons = {}
        for tf, scan in tf_results.items():
            h = self._horizon_result(secid, asset_type, tf, scan, access)
            if h is not None:
                horizons[tf] = h
        if not horizons:
            return None

        # Плоские поля для обратной совместимости (CLI-отчёт, /asset):
        # основной горизонт — дневной; если его нет — часовой.
        main = horizons.get('1d') or horizons.get('1h')
        result = {
            'secid': secid,
            'asset_type': asset_type,
            'timeframes': sorted(horizons.keys()),
            'horizons': horizons,          # аналитика по отдельным горизонтам
            'date': main['date'],
            'price': main['price'],
            'signals': main['signals'],
            'score_buy': main['score_buy'],
            'score_sell': main['score_sell'],
            'verdict': main['verdict'],
            'strength': main['strength'],
            'trade_access': access,
        }
        if 'sell_blocked_reasons' in main:
            result['sell_blocked_reasons'] = main['sell_blocked_reasons']
        if main['verdict'] == 'sell':
            result['can_short'] = access['short_allowed']
        if main.get('sl_tp'):
            result['sl_tp'] = main['sl_tp']
        return result

    def scan_all(self, config_assets: Dict[str, List[str]]) -> List[Dict]:
        results = []
        no_data = []
        for asset_type, secids in config_assets.items():
            for secid in secids:
                try:
                    r = self.scan_asset(secid, asset_type)
                    if r is None:
                        no_data.append(secid)
                    else:
                        results.append(r)
                except Exception as e:
                    logger.error(f"Ошибка скана {secid}: {e}")
                    no_data.append(secid)
        if no_data:
            logger.warning(f"Без данных (нет свечей в БД): {len(no_data)} активов, напр. {no_data[:5]}")
        results.sort(key=lambda x: max(x['score_buy'], x['score_sell']), reverse=True)
        return results
