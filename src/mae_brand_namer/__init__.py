"""
Mae Brand Namer - A LangGraph-powered brand name generation and evaluation system.
"""

# Apply patches to external libraries
from .patches import apply_patches
apply_patches()

# Import models that don't have circular dependencies
from .models.state import BrandNameGenerationState, AppState
from .cli import cli

# Avoid immediate circular imports
__version__ = "0.1.0"

# Expose key functionality but without circular imports
__all__ = ["BrandNameGenerationState", "AppState", "cli"]

# This enables importing run_brand_naming_workflow from the package
# but avoids the circular import during module initialization
def __getattr__(name):
    if name == "run_brand_naming_workflow":
        from .workflows.brand_naming import run_brand_naming_workflow
        return run_brand_naming_workflow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
