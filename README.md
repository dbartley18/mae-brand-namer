# Mae Brand Namer

A LangGraph-powered application for automated brand name generation and evaluation.

## Overview

This application uses LangGraph to orchestrate a complex workflow of AI agents that collaborate to:
1. Generate unique and meaningful brand names
2. Evaluate them across multiple dimensions (linguistic, cultural, market fit, etc.)
3. Produce comprehensive brand naming reports

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

## Installation

You can install Mae Brand Namer directly from the repository:

```bash
# Install in development mode
pip install -e .

# Or install from the repository
pip install git+https://github.com/your-username/mae-brand-namer.git
```

This will install the `mae-brand-namer` command-line tool and make the package available for import in your Python projects.

## Project Structure

```
src/mae_brand_namer/
├── agents/         # Individual agent implementations
├── config/         # Configuration and settings
├── models/         # Data models and schemas
├── workflows/      # LangGraph workflow definitions
└── utils/          # Utility functions and helpers
```

## Usage

### Command Line Interface

Mae Brand Namer provides a convenient CLI for running the brand naming workflow:

```bash
# Basic usage
mae-brand-namer run "Create a name for a tech startup that specializes in AI-powered environmental monitoring solutions"

# Save results to a JSON file
mae-brand-namer run "Create a name for a sustainable fashion brand" --output results.json

# View configuration
mae-brand-namer config
```

### Programmatic Usage

You can also use Mae Brand Namer programmatically in your Python applications:

```python
import asyncio
from mae_brand_namer import run_brand_naming_workflow

async def generate_brand_names():
    # Run the workflow with your prompt
    prompt = "Create a name for a luxury skincare brand focusing on organic ingredients"
    result = await run_brand_naming_workflow(prompt)
    
    # Access the results
    print(f"Run ID: {result['run_id']}")
    print(f"Generated {len(result['generated_names'])} brand names")
    print(f"Shortlisted names: {result['shortlisted_names']}")
    
    # The full report URL is available in the results
    print(f"Report URL: {result['report_url']}")
    
    return result

# Run the async function
if __name__ == "__main__":
    asyncio.run(generate_brand_names())
```

### Workflow Process

The brand naming workflow follows these steps:

1. **Generate UID**: Creates a unique identifier for the workflow run
2. **Understand Brand Context**: Analyzes your prompt to extract key brand elements
3. **Generate Brand Names**: Creates unique name candidates based on the brand context
4. **Process Analyses**: Runs multiple analyses on each name:
   - Linguistic analysis
   - Cultural sensitivity analysis
   - Market research
   - Domain availability
   - SEO potential
5. **Evaluate Brand Names**: Scores each name based on all analyses
6. **Compile Report**: Creates a comprehensive brand naming report
7. **Store Report**: Saves the report and returns the results

### Error Handling

The workflow includes robust error handling with automatic retries for transient errors. The `ProcessSupervisor` monitors task execution and logs details to Supabase for tracking and analysis.

### Advanced Configuration

Advanced settings can be configured in `.env`:

- Model configuration (Gemini/OpenAI)
- LangSmith integration for tracing
- Supabase settings for data persistence
- Retry configuration for error handling
- Report format and output settings

See `.env.example` for all available options.

## LangGraph Integration

Mae Brand Namer is built with LangGraph, enabling powerful orchestration of the brand naming workflow.

### LangSmith Tracing

The application integrates with LangSmith for workflow monitoring and debugging. To enable tracing:

1. Set up a LangSmith account at [smith.langchain.com](https://smith.langchain.com)
2. Configure your environment variables:
   ```
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
   LANGCHAIN_API_KEY=your_api_key
   LANGCHAIN_PROJECT=mae-brand-namer
   ```

3. Once enabled, you can view detailed traces of each workflow run, including:
   - Complete agent execution paths
   - Input/output for each step
   - Timing and performance metrics
   - Error information

### Process Monitoring

The `ProcessSupervisor` logs detailed execution information to Supabase:

- Task start/completion times
- Execution duration
- Retry attempts
- Error details

This data can be queried from the `process_logs` table in Supabase for analytics and monitoring.

### Workflow Visualization

You can visualize the workflow structure using LangGraph Studio:

1. Install LangGraph Studio Desktop
2. Connect to your project
3. Visualize the workflow graph and execution paths

This provides powerful insights into the brand naming process and helps identify optimization opportunities.
