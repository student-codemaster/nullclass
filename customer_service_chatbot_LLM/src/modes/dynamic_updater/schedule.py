import time
import threading
import yaml
import logging

try:
    import schedule
except Exception:
    schedule = None

from .updater import update_vector_db

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _load_config(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.error("Scheduler config not found: %s", path)
        return {}

def run_scheduler(config_path: str = None):
    base_dir = __file__
    if config_path is None:
        # default config.yaml next to this file
        config_path = __file__.replace("schedule.py", "config.yaml")

    cfg = _load_config(config_path)
    interval = cfg.get("interval_minutes", 60)

    logger.info("Scheduling updater every %s minutes", interval)

    if schedule is None:
        logger.warning("`schedule` package not available; scheduler will not run.")
        return

    schedule.every(interval).minutes.do(update_vector_db, config_path=config_path)

    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user.")


def start_background_scheduler(config_path: str = None) -> threading.Thread:
    if schedule is None:
        logger.warning("Cannot start background scheduler because `schedule` package is missing.")
        return None

    t = threading.Thread(target=run_scheduler, args=(config_path,), daemon=True)
    t.start()
    logger.info("Background scheduler started.")
    return t


if __name__ == "__main__":
    run_scheduler()
