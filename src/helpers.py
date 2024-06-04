"""Helper functions"""

import logging

## Define annualized inflation, per 12 months
INF_1012 = [0.45, 60.79, 0.45, 60.79, 0.45, 60.79, 0.45, 60.79, 0.45, 60.79]
INF_430 = [0.38, 0.47, 26.85, 55.49, 64.18, 0.38, 0.47, 26.85, 55.49, 64.18]


def disable_module_debug_log(level: str) -> None:
    """Disable logger ouputs for other modules up to defined `level`"""
    for log_name in logging.Logger.manager.loggerDict:
        if log_name != "__name__":
            log_level = getattr(logging, level.upper())
            logging.getLogger(log_name).setLevel(log_level)
