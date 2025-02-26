#!/usr/bin/env python3
"""
Example script demonstrating how to use the ProcessSupervisorCallbackHandler
with a LangGraph workflow.

This script shows how to:
1. Create and configure the callback handler
2. Integrate it with a LangGraph workflow
3. Run the workflow and track execution
4. Handle and analyze errors

Run this script directly to see the callback handler in action.
"""

import asyncio
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional

from langsmith import Client
from dotenv import load_dotenv

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.mae_brand_namer.workflows.brand_naming import (
    run_brand_naming_workflow,
    ProcessSupervisorCallbackHandler,
    ProcessSupervisor,
    create_workflow
)
from src.mae_brand_namer.utils.logging import setup_logging, get_logger
from src.mae_brand_namer.utils.supabase_utils import SupabaseManager

# Set up logging
setup_logging(level="DEBUG", json_format=False)
logger = get_logger(__name__)

async def run_with_custom_callback(
    prompt: str, 
    langsmith_project: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run the brand naming workflow with a custom callback handler.
    
    This function demonstrates how to create and use the ProcessSupervisorCallbackHandler
    with detailed step-by-step execution tracking.
    
    Args:
        prompt: The user prompt for brand naming
        langsmith_project: Optional custom LangSmith project name
        
    Returns:
        The result of the workflow execution
    """
    # Load environment variables
    load_dotenv()
    
    # 1. Create a LangSmith client with optional custom project
    client = Client()
    if langsmith_project:
        os.environ["LANGCHAIN_PROJECT"] = langsmith_project
    
    # 2. Create a Supabase manager for database operations
    supabase = SupabaseManager()
    
    # 3. Create the process supervisor with the Supabase manager
    #    The supervisor handles database operations and retry logic
    supervisor = ProcessSupervisor(supabase=supabase)
    
    # 4. Create the custom callback handler with the supervisor and LangSmith client
    #    This handler will track execution of each node in the workflow
    callback_handler = ProcessSupervisorCallbackHandler(
        supervisor=supervisor,
        langsmith_client=client
    )
    
    # 5. Create the workflow with the LangSmith client
    workflow = create_workflow(langsmith_client=client)
    
    # 6. Log the start of the workflow
    run_start_time = datetime.now()
    logger.info(
        f"Starting brand naming workflow",
        extra={
            "prompt": prompt,
            "start_time": run_start_time.isoformat()
        }
    )
    
    try:
        # 7. Invoke the workflow with the callback handler
        #    This is where the magic happens - each node's execution will be tracked
        result = await workflow.ainvoke(
            {"user_prompt": prompt, "client": client},
            config={"callbacks": [client, callback_handler]}
        )
        
        # 8. Log the successful completion of the workflow
        run_end_time = datetime.now()
        duration_sec = int((run_end_time - run_start_time).total_seconds())
        
        logger.info(
            f"Workflow completed successfully",
            extra={
                "run_id": result.get("run_id"),
                "duration_sec": duration_sec,
                "shortlisted_names_count": len(result.get("shortlisted_names", [])),
                "report_url": result.get("report_url")
            }
        )
        
        # 9. Analyze which nodes had errors (if any)
        if callback_handler.error_nodes:
            logger.warning(
                f"Workflow completed with some node errors",
                extra={
                    "error_nodes": list(callback_handler.error_nodes),
                    "run_id": result.get("run_id")
                }
            )
        
        return result
        
    except Exception as e:
        # 10. Handle workflow-level exceptions
        run_end_time = datetime.now()
        duration_sec = int((run_end_time - run_start_time).total_seconds())
        
        logger.error(
            f"Workflow failed with exception",
            extra={
                "error_type": type(e).__name__,
                "error_message": str(e),
                "duration_sec": duration_sec,
                "error_nodes": list(callback_handler.error_nodes) if hasattr(callback_handler, "error_nodes") else []
            }
        )
        raise

async def main():
    """Run the example."""
    # Example prompt for brand naming
    prompt = "Create a brand name for a tech startup focused on AI-powered healthcare solutions"
    
    try:
        # Run the workflow with our custom callback handler
        result = await run_with_custom_callback(
            prompt=prompt,
            langsmith_project="mae-brand-namer-example"
        )
        
        # Print some information about the result
        print("\n=== Workflow Result ===")
        print(f"Run ID: {result.get('run_id')}")
        print(f"Generated names: {len(result.get('generated_names', []))}")
        print(f"Shortlisted names: {len(result.get('shortlisted_names', []))}")
        
        if result.get("shortlisted_names"):
            print("\nTop brand name suggestions:")
            for i, name in enumerate(result.get("shortlisted_names", [])[:3], 1):
                print(f"{i}. {name}")
                
        if result.get("report_url"):
            print(f"\nDetailed report available at: {result.get('report_url')}")
            
    except Exception as e:
        print(f"Error running example: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run the async example
    asyncio.run(main()) 