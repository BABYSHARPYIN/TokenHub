class PoolError(Exception):
    """Base error for TokenHub."""


class SessionFailed(PoolError):
    """A session failed closed. Dual sessions do not fall back mid-flight."""

    def __init__(self, message: str, *, code: str):
        super().__init__(message)
        self.code = code


class LayerRangeError(PoolError):
    """A node was asked to run a layer range it does not own."""


class PlaintextLeak(PoolError):
    """An activation payload carried text or a forbidden field."""


class EngineUnavailable(PoolError):
    """The requested real engine cannot run in this process."""
