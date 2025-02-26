# Import Style Guide for Mae Brand Namer

This document outlines the import conventions to be followed in the Mae Brand Namer project.

## Import Organization

All imports should be organized into three distinct groups with a blank line between each group:

1. **Standard library imports**
2. **Third-party imports**
3. **Local application imports**

Within each group, imports should be alphabetically sorted.

```python
# Standard library imports
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any

# Third-party imports
from langchain.callbacks import tracing_enabled
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph
from langsmith import Client
from postgrest import APIError as PostgrestError

# Local application imports
from mae_brand_namer.agents import BrandContextExpert, BrandNameGenerator
from mae_brand_namer.config.settings import settings
from mae_brand_namer.utils.logging import get_logger
```

## Import Conventions

### Aliasing

- Use import aliasing only when necessary to prevent naming conflicts or when following widely accepted conventions
- Document the reason for aliasing in a comment when it's not immediately obvious
- Always use consistent alias names across the codebase

```python
# Good: Consistent alias for APIError
from postgrest import APIError as PostgrestError

# Avoid inconsistent aliases across files
# from postgrest import APIError as PostgrestAPIError  # Don't do this
```

### Absolute vs. Relative Imports

- Use relative imports for imports within the same package
- Use absolute imports for imports from other packages

```python
# Relative import (within the same package)
from ..utils.logging import get_logger

# Absolute import (from another package)
from mae_brand_namer.config.settings import settings
```

### Wildcard Imports

- Avoid wildcard imports (`from module import *`) as they make it unclear which names are present in the namespace

```python
# Bad
from mae_brand_namer.agents import *

# Good
from mae_brand_namer.agents import BrandContextExpert, BrandNameGenerator
```

### Unused Imports

- Remove unused imports to keep the codebase clean and prevent confusion

### Circular Imports

- Avoid circular imports by restructuring code or using lazy imports
- Import inside functions when necessary to prevent circular dependencies

```python
# To avoid circular imports
def some_function():
    # Import inside function to avoid circular dependency
    from mae_brand_namer.some_module import SomeClass
    
    # Use SomeClass here
```

## LangGraph Specific Imports

For LangGraph-related code, follow these specific conventions:

1. Import from `langgraph.graph` for `StateGraph` and graph-related functionality
2. Import from `langgraph.constants` for constants like `Send`
3. Import from `langsmith` for `Client` and LangSmith-related functionality

```python
from langgraph.graph import StateGraph
from langgraph.constants import Send
from langsmith import Client
```

## Pydantic Imports

For Pydantic-related code:

1. Import from `pydantic` for base functionality
2. Import from `pydantic_settings` for settings functionality

```python
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
```

## Type Hints

- Import type hints from `typing` at the top of the file
- Group related type hints in a single import statement

```python
from typing import Dict, List, Optional, Any, Tuple, TypedDict
``` 