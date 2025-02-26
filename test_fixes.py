"""Test script to verify the fixes we've implemented."""

import sys
sys.path.insert(0, 'src')  # Add src to the module search path

from mae_brand_namer.config.settings import settings
import asyncio

async def test_settings():
    """Test the settings object to verify the google_api_key property."""
    print("Testing settings.google_api_key...")
    print(f"settings.gemini_api_key: {settings.gemini_api_key}")
    print(f"settings.google_api_key: {settings.google_api_key}")
    assert settings.google_api_key == settings.gemini_api_key
    print("✅ settings.google_api_key is correctly pointing to settings.gemini_api_key")

async def test_tracing():
    """Test the tracing_v2_enabled context manager."""
    print("Testing tracing_v2_enabled...")
    try:
        from langchain_core.tracers.context import tracing_v2_enabled
        with tracing_v2_enabled():
            print("✅ tracing_v2_enabled is working correctly")
    except Exception as e:
        print(f"❌ Error with tracing_v2_enabled: {e}")
        raise

async def main():
    """Run all tests."""
    print("Running tests...\n")
    await test_settings()
    print()
    await test_tracing()
    print("\nAll tests passed!")
    
if __name__ == "__main__":
    asyncio.run(main()) 