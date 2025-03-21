"""Patches for external libraries."""

from .langchain_google_genai_patch import apply_patches

__all__ = ["apply_patches"] 