"""Errors for the tib/fib strength analysis. These are failures, not fallback numbers."""

from __future__ import annotations


class StrengthAnalysisError(RuntimeError):
    def __init__(self, message: str, modality: str = "unknown"):
        super().__init__(message)
        self.modality = modality
