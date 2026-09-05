"""APScheduler job that refreshes the RAG index every N minutes.

Two ways to use it:
  * `python scheduler/feed_scheduler.py`  - standalone blocking process
  * `start_background_scheduler()`         - from inside the Streamlit app

`max_instances=1` + `coalesce=True` guarantee we never run two refreshes at
once even if Ollama is slow and a job overruns its slot.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.blocking import BlockingScheduler
from loguru import logger

from config import settings, configure_logging
from rag.indexer import refresh

JOB_ID = "rag_refresh"
last_run: dict = {"at": None, "result": None, "error": None}


def _job() -> None:
    logger.info("Scheduled RAG refresh starting")
    try:
        last_run["result"] = refresh()
        last_run["error"] = None
    except Exception as exc:
        logger.exception("Scheduled refresh failed")
        last_run["error"] = str(exc)
    last_run["at"] = datetime.now()


def _configure(scheduler) -> None:
    scheduler.add_job(_job, "interval", minutes=settings.rag_refresh_interval_minutes, id=JOB_ID,
                      max_instances=1, coalesce=True, next_run_time=datetime.now())


def start_background_scheduler() -> BackgroundScheduler:
    scheduler = BackgroundScheduler(daemon=True)
    _configure(scheduler)
    scheduler.start()
    logger.info(f"Background RAG refresh every {settings.rag_refresh_interval_minutes} min")
    return scheduler


if __name__ == "__main__":
    configure_logging("scheduler")
    scheduler = BlockingScheduler()
    _configure(scheduler)
    logger.info(f"Feed scheduler running every {settings.rag_refresh_interval_minutes} min (Ctrl+C to stop)")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        pass
