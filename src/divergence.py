import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from loguru import logger


class DivergenceDetector:
    def __init__(self, df: pd.DataFrame, window: int = 5, min_distance: int = 3):
        """
        df: DataFrame с колонками ['ts', 'close', 'rsi_14', 'macd_line', 'obv'] (как минимум)
        window: число свечей слева и справа для поиска локальных экстремумов
        min_distance: минимальное количество свечей между двумя экстремумами (чтобы не было слишком частых сигналов)
        """
        self.df = df.copy()
        self.window = window
        self.min_distance = min_distance
        self._prepare_data()

    def _prepare_data(self):
        """Убедимся, что данные отсортированы и сброшены индексы."""
        self.df = self.df.sort_values('ts').reset_index(drop=True)

    def _find_peaks(self, series: pd.Series) -> pd.DataFrame:
        """
        Находит локальные максимумы и минимумы в ряду.
        Возвращает DataFrame с колонками: index, value, type ('max' или 'min')
        """
        peaks = []
        for i in range(self.window, len(series) - self.window):
            left = i - self.window
            right = i + self.window
            if series.iloc[i] == series.iloc[left:right+1].max():
                peaks.append((i, series.iloc[i], 'max'))
            elif series.iloc[i] == series.iloc[left:right+1].min():
                peaks.append((i, series.iloc[i], 'min'))
        # Фильтруем по минимальному расстоянию
        filtered = []
        for idx, val, typ in peaks:
            if not filtered or (idx - filtered[-1][0]) >= self.min_distance:
                filtered.append((idx, val, typ))
            else:
                # Если расстояние меньше, оставляем более выраженный экстремум
                last_idx, last_val, last_typ = filtered[-1]
                if abs(val) > abs(last_val):
                    filtered[-1] = (idx, val, typ)
        return pd.DataFrame(filtered, columns=['index', 'value', 'type'])

    def detect_divergences(self, price_col: str = 'close', indicator_col: str = 'rsi_14', lookback: int = 50) -> List[Dict]:
        """
        Ищет дивергенции между ценой и индикатором за последние lookback свечей.
        Возвращает список словарей с описанием каждой дивергенции.
        """
        if len(self.df) < lookback:
            lookback = len(self.df)

        df_subset = self.df.iloc[-lookback:].reset_index(drop=True)

        # Находим экстремумы цены
        price_series = df_subset[price_col]
        price_peaks = self._find_peaks(price_series)

        # Находим экстремумы индикатора
        ind_series = df_subset[indicator_col]
        ind_peaks = self._find_peaks(ind_series)

        divergences = []

        # Для каждого экстремума цены ищем соответствующий экстремум индикатора
        # и проверяем условия расхождения
        for i in range(len(price_peaks)):
            p_idx, p_val, p_type = price_peaks.iloc[i]
            # Ищем ближайший экстремум индикатора того же типа (max->max, min->min)
            matching_ind = ind_peaks[(ind_peaks['type'] == p_type)]
            if matching_ind.empty:
                continue
            # Ближайший по индексу
            closest = matching_ind.iloc[(matching_ind['index'] - p_idx).abs().argsort()[:1]]
            i_idx, i_val, i_type = closest.iloc[0]

            # Проверяем, что они находятся примерно в одной зоне (разница не более 10 свечей)
            if abs(i_idx - p_idx) > 10:
                continue

            # Условия дивергенции
            if p_type == 'min':
                # Бычья дивергенция: цена обновила минимум, а индикатор — нет
                if p_val < price_series.iloc[p_idx - 1] and i_val > ind_series.iloc[i_idx - 1]:
                    divergences.append({
                        'type': 'bullish',
                        'price_index': p_idx,
                        'price_value': p_val,
                        'indicator_index': i_idx,
                        'indicator_value': i_val,
                        'indicator_name': indicator_col,
                        'description': f'Бычья дивергенция {indicator_col}: цена обновила минимум, индикатор нет'
                    })
            elif p_type == 'max':
                # Медвежья дивергенция: цена обновила максимум, а индикатор — нет
                if p_val > price_series.iloc[p_idx - 1] and i_val < ind_series.iloc[i_idx - 1]:
                    divergences.append({
                        'type': 'bearish',
                        'price_index': p_idx,
                        'price_value': p_val,
                        'indicator_index': i_idx,
                        'indicator_value': i_val,
                        'indicator_name': indicator_col,
                        'description': f'Медвежья дивергенция {indicator_col}: цена обновила максимум, индикатор нет'
                    })

        return divergences

    def scan_all_indicators(self, lookback: int = 50) -> Dict[str, List[Dict]]:
        """
        Сканирует дивергенции по всем доступным индикаторам (RSI, MACD, OBV).
        Возвращает словарь {indicator_name: list_of_divergences}
        """
        result = {}
        indicator_cols = [col for col in self.df.columns if col in ('rsi_14', 'macd_line', 'obv')]
        for col in indicator_cols:
            divergences = self.detect_divergences(indicator_col=col, lookback=lookback)
            if divergences:
                result[col] = divergences
        return result

    # ---------- классические дивергенции по парам экстремумов ----------
    @staticmethod
    def _local_extremes(series: pd.Series, kind: str, window: int = 2) -> List[int]:
        """Индексы строгих локальных максимумов/минимумов (fractal): экстремум
        должен быть больше/меньше всех соседей в окне (без плато-артефактов)."""
        idx = []
        for i in range(window, len(series) - window):
            seg = series.iloc[i - window:i + window + 1].dropna()
            if len(seg) < window * 2 + 1:
                continue
            v = series.iloc[i]
            if pd.isna(v):
                continue
            if kind == 'max' and v > seg.drop(index=i).max():
                idx.append(i)
            elif kind == 'min' and v < seg.drop(index=i).min():
                idx.append(i)
        return idx

    def detect_classic_divergences(self, price_col: str = 'close',
                                   indicator_col: str = 'rsi_14',
                                   lookback: int = 50) -> List[Dict]:
        """Классический поиск дивергенций по последним двум экстремумам.

        Правила (решение пользователя от 10.10.2026):
          * новый МАКСИМУМ цены (> предыдущего max) при СНИЖАЮЩЕМСЯ максимуме
            индикатора -> bearish (продажа);
          * новый МИНИМУМ цены (< предыдущего min) при ПОВЫШАЮЩЕМСЯ минимуме
            индикатора -> bullish (покупка).
        Работает для rsi_14, macd_hist, macd_line и т.п.
        """
        if len(self.df) < lookback:
            lookback = len(self.df)
        sub = self.df.iloc[-lookback:].reset_index(drop=True)
        if price_col not in sub.columns or indicator_col not in sub.columns:
            return []

        out: List[Dict] = []
        for kind in ('max', 'min'):
            p_idx = self._local_extremes(sub[price_col], kind)
            i_idx = self._local_extremes(sub[indicator_col], kind)
            if len(p_idx) < 2 or not i_idx:
                continue
            p1, p2 = p_idx[-2], p_idx[-1]
            # индикаторный экстремум того же типа рядом с каждым пиком цены
            m1 = [i for i in i_idx if abs(i - p1) <= 10]
            m2 = [i for i in i_idx if abs(i - p2) <= 10]
            if not m1 or not m2 or m1[-1] == m2[-1]:
                continue
            pv1, pv2 = sub[price_col].iloc[p1], sub[price_col].iloc[p2]
            iv1 = sub[indicator_col].iloc[m1[-1]]
            iv2 = sub[indicator_col].iloc[m2[-1]]
            if any(pd.isna(x) for x in (pv1, pv2, iv1, iv2)):
                continue
            if kind == 'max' and pv2 > pv1 and iv2 < iv1:
                out.append({'type': 'bearish',
                            'description': f"Bearish divergence: new price high, lower {indicator_col} peak"})
            elif kind == 'min' and pv2 < pv1 and iv2 > iv1:
                out.append({'type': 'bullish',
                            'description': f"Bullish divergence: new price low, higher {indicator_col} trough"})
        return out

    def detect_histogram_divergences(self, lookback: int = 50) -> List[Dict]:
        """Дивергенции цены с ГИСТОГРАММОЙ MACD (требование пользователя):
        новый max цены + снижающийся max гистограммы -> sell;
        новый min цены + повышающийся min гистограммы -> buy."""
        if 'macd_hist' not in self.df.columns:
            return []
        return self.detect_classic_divergences(price_col='close',
                                               indicator_col='macd_hist',
                                               lookback=lookback)