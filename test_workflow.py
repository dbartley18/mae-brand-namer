#!/usr/bin/env python3
import asyncio
from src.mae_brand_namer.workflows.brand_naming import graph_factory

async def test():
    print("Creating workflow...")
    workflow = graph_factory({
        'configurable': {
            'langsmith_client': None, 
            'default_step_delay': 0.1, 
            'step_delays': None
        }
    })
    
    # Compile the graph first
    print("Compiling workflow...")
    compiled_graph = workflow.compile()
    
    print("Invoking workflow...")
    try:
        # Use the compiled graph's invoke method
        result = await compiled_graph.ainvoke({'user_prompt': 'Test brand for pet food company'})
        print('Workflow completed successfully!')
        print(f'Run ID: {result.run_id if hasattr(result, "run_id") else "unknown"}')
        print(f'Status: {result.status if hasattr(result, "status") else "unknown"}')
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test()) 