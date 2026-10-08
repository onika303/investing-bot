from datetime import datetime
import pandas as pd

def ts_to_datetime(ts: int) -> str:
    """
    Преобразует Unix timestamp (секунды) в строку формата 'YYYY-MM-DD HH:MM:SS'.
    """
    return datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

def ts_to_date(ts: int) -> str:
    """
    Преобразует Unix timestamp в строку формата 'YYYY-MM-DD'.
    """
    return datetime.fromtimestamp(ts).strftime('%Y-%m-%d')