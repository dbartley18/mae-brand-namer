# LangGraph Callback Architecture

This document explains how the callback mechanism works in the Mae Brand Namer application, specifically how we track and monitor execution of workflow nodes using LangGraph callbacks.

## Overview

The application uses a custom callback handler called `ProcessSupervisorCallbackHandler` that integrates with:

1. The `ProcessSupervisor` class for tracking task execution in Supabase
2. The LangSmith client for enhanced tracing and run analysis
3. The LangGraph workflow for real-time monitoring

This callback system replaces the deprecated pre-processor, post-processor, and interrupt handler methods that were previously used in LangGraph.

## Callback Flow

Here's how callbacks flow through the system:

1. When a node starts execution:
   - `on_chain_start` is called
   - The node name and run ID are extracted
   - The execution is logged in Supabase via `ProcessSupervisor`
   - Metadata is added to LangSmith

2. When a node completes successfully:
   - `on_chain_end` is called
   - The execution is marked as complete in Supabase
   - Duration metrics are calculated
   - Success is recorded in LangSmith

3. When a node encounters an error:
   - `on_chain_error` is called
   - The error is logged in Supabase
   - The retry count is updated
   - A decision is made whether to retry
   - Error details are recorded in LangSmith

## ProcessSupervisorCallbackHandler

The `ProcessSupervisorCallbackHandler` extends LangChain's `BaseCallbackHandler` class with these key features:

### Tracking State

- Maintains current run ID and node name for context
- Tracks start times for calculating durations
- Maintains a set of nodes that have encountered errors
- Maps node names to agent types and task names

### Robust Metadata Extraction

The handler implements robust methods for extracting metadata:

- `_extract_node_name`: Gets the node name using multiple fallback strategies
- `_extract_run_id`: Gets the run ID using multiple fallback strategies, with auto-generation as a last resort

### LangSmith Integration

For enhanced monitoring and analysis, the handler:

- Updates LangSmith runs with node execution metadata
- Records status changes (in_progress, completed, error)
- Logs errors and retry information
- Maintains execution context across the workflow

## Usage

To use the callback handler:

```python
# Create a LangSmith client
client = Client()

# Create the workflow
workflow = create_workflow(langsmith_client=client)

# Create a LangSmith client
client = Client()

# Create the workflow with proper config
workflow_config = {
    "configurable": {
        "langsmith_client": client,
        "default_step_delay": 2.0,
        "step_delays": None  # Use default step delays
    }
}
workflow = create_workflow(workflow_config)

# Create the callback handler
supervisor_handler = ProcessSupervisorCallbackHandler(langsmith_client=client)

# Invoke the workflow with callbacks
result = await workflow.ainvoke(
    {"user_prompt": "Create a brand name for a tech startup"}, 
    config={"callbacks": [client, supervisor_handler]}
)
```

## Benefits

This callback architecture provides several benefits:

1. **Robust Tracking**: Handles edge cases and ensures we never lose context
2. **Enhanced Debugging**: Provides rich logging information for troubleshooting
3. **Performance Monitoring**: Tracks execution duration for optimization
4. **Retry Management**: Implements exponential backoff for transient failures
5. **LangSmith Integration**: Enhances tracing and run analysis
6. **Database Persistence**: Stores execution history in Supabase for long-term analysis

## Future Improvements

Potential enhancements to the callback system:

- Add support for streaming responses
- Implement webhook notifications for long-running workflows
- Create a dashboard for visualizing workflow execution
- Add custom metrics collection for performance analysis
- Implement intelligent retry policies based on error type 