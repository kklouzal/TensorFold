"""Gemma 4's native packed-affine CUDA target and separately loaded DFlash companion."""

from .weights import Config, Gemma4Weights

__all__ = ["Config", "Gemma4Weights"]
