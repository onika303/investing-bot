#!/usr/bin/env python3
"""Разовый прогон цикла (загрузка+скан) и печать Telegram-отчёта в консоль.

Для приёмки расписания без ожидания cron: python scripts/run_report_once.py [--no-update]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import format_report_text, scan_and_report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no-update", action="store_true", help="не дозагружать данные")
    p.add_argument("--db", default="finance.db")
    a = p.parse_args()
    report, stats = scan_and_report(a.db, do_update=not a.no_update)
    print(f"stats: {stats}, длительность: {report.get('duration_sec')} с\n")
    for chunk in format_report_text(report):
        print(chunk)
        print("-" * 60)


if __name__ == "__main__":
    main()
