"""
Bantz v3 — Data Access Layer

Unified entry point for all persistence. Import the singleton and go:

    from bantz.data import data_layer

    data_layer.init(config)
    data_layer.conversations.add("user", "hello")
    profile = data_layer.profile.load()
"""
from bantz.data.layer import data_layer

__all__ = ["data_layer"]
