"""KV cache service. v1 default is in-process memory; Windows exposes it over cleartext HTTP."""

from pool.kv.memory import InMemoryKVStore

__all__ = ["InMemoryKVStore"]
