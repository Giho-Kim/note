__all__ = ["TDJEPA"]


def __getattr__(name):
    if name == "TDJEPA":
        from .agent import TDJEPA

        return TDJEPA
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
