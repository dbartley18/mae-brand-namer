import asyncio
import json
import sys
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Let's check the prompts first
def check_prompts():
    print("Checking prompt files...")
    prompt_dir = Path("src/mae_brand_namer/agents/prompts/semantic_analysis")
    
    if not prompt_dir.exists():
        print(f"ERROR: Prompt directory not found: {prompt_dir}")
        return False
    
    system_file = prompt_dir / "system.yaml"
    human_file = prompt_dir / "human.yaml"
    
    if not system_file.exists():
        print(f"ERROR: System prompt file not found: {system_file}")
        return False
    
    if not human_file.exists():
        print(f"ERROR: Human prompt file not found: {human_file}")
        return False
    
    # Print the content of the system prompt
    print(f"\nSystem prompt content:")
    print("=" * 50)
    with open(system_file, 'r') as f:
        system_content = f.read()
        print(system_content)
    print("=" * 50)
    
    # Print the content of the human prompt
    print(f"\nHuman prompt content:")
    print("=" * 50)
    with open(human_file, 'r') as f:
        human_content = f.read()
        print(human_content)
    print("=" * 50)
    
    return True

def create_semantic_analyzer_with_mocked_db():
    from src.mae_brand_namer.agents.semantic_analysis_expert import SemanticAnalysisExpert
    
    # Create a mock supabase manager
    mock_supabase = MagicMock()
    mock_supabase.table = MagicMock()
    mock_supabase.table.return_value.insert.return_value.execute = AsyncMock()
    
    # Create the expert with our mock
    expert = SemanticAnalysisExpert(supabase=mock_supabase)
    
    # Patch the _store_analysis method to do nothing
    async def noop_store(*args, **kwargs):
        print("Skipping database insertion for testing...")
        return None
    
    expert._store_analysis = noop_store
    
    return expert

def main():
    # Check the prompt files first
    if not check_prompts():
        print("Cannot proceed with test due to missing prompt files.")
        return

    # Proceed with regular test only if prompts check out
    print("\nProceeding with SemanticAnalysisExpert test...")
    asyncio.run(test_semantic_analysis())

async def test_semantic_analysis():
    # Set up test data
    run_id = "test_run_123"
    brand_name = "Acme"
    brand_context = {
        "brand_promise": "Quality products for all",
        "brand_personality": ["Reliable", "Innovative"],
        "industry_focus": "Technology"
    }
    
    try:
        print("Initializing SemanticAnalysisExpert with mocked database...")
        expert = create_semantic_analyzer_with_mocked_db()
        
        print("\nExpert initialized successfully!")
        print("Running analysis...")
        
        # Run the analysis
        result = await expert.analyze_brand_name(
            run_id=run_id,
            brand_name=brand_name,
            brand_context=brand_context
        )
        
        print("\nSUCCESS: Analysis completed without errors")
        print(f"Result keys: {list(result.keys())}")
        print(f"Has 'denotative_meaning': {'denotative_meaning' in result}")
        
        # Print the 'denotative_meaning' value
        if 'denotative_meaning' in result:
            print(f"\nDenotative meaning: {result['denotative_meaning']}")
        
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 