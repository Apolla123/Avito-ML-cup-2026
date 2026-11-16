"""loguru, если установлен; иначе стандартный logging с тем же API ({}-форматирование)."""

try:
    from loguru import logger  # noqa: F401
except ImportError:  # pragma: no cover
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _l = logging.getLogger("avito_rank")

    class _Shim:
        def info(self, msg, *a):
            _l.info(msg.format(*a))

        warning = info

    logger = _Shim()
