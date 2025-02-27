from typing import Dict, Optional, Any
from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    """Application configuration settings for the brand naming workflow.
    
    This class defines all configurable parameters for the brand naming application,
    including workflow settings, agent parameters, and feature flags.
    """
    
    # General app settings
    debug_mode: bool = Field(False, description="Enable debug mode for detailed logging")
    verbose_logging: bool = Field(False, description="Enable verbose logging for all operations")
    
    # Workflow configuration
    parallel_analyses: bool = Field(True, description="Run analysis agents in parallel")
    max_retry_attempts: int = Field(3, description="Maximum number of retry attempts for failed steps")
    timeout_seconds: int = Field(600, description="Timeout in seconds for workflow operations")
    
    # Agent configuration
    temperature: float = Field(0.7, description="Default temperature setting for LLM")
    max_tokens: int = Field(4000, description="Maximum tokens for LLM responses")
    
    # Agent-specific temperature settings
    agent_temperatures: Dict[str, float] = Field(
        default_factory=lambda: {
            # Analysis agents
            "semantic_analysis_expert": 1.0,
            "linguistic_analysis_expert": 1.0,
            "cultural_sensitivity_expert": 1.0,
            "translation_analysis_expert": 1.0,
            "domain_analysis_expert": 1.0,
            "seo_online_discovery_expert": 1.0,
            "competitor_analysis_expert": 1.0,
            "survey_simulation_expert": 1.0,
            "market_research_expert": 1.0,
            
            # Core workflow agents
            "uid_generator": 0.1,  # Deterministic, low temperature
            "brand_context_expert": 1.5,
            "brand_name_creation_expert": 1.5, # Higher for more creativity
            "brand_name_evaluator": 1.0,  # Lower for more consistent evaluation
            
            # Output agents
            "report_compiler": 1.0,
            "report_storer": 0.1,  # Deterministic, low temperature
            "process_supervisor": 0.5,
        },
        description="Temperature settings for individual agents"
    )
    
    # Method to get temperature for a specific agent
    def get_temperature_for_agent(self, agent_name: str) -> float:
        """Get the temperature setting for a specific agent.
        
        Args:
            agent_name (str): The name of the agent to get the temperature for
            
        Returns:
            float: The temperature setting for the specified agent, or the default temperature if not found
        """
        return self.agent_temperatures.get(agent_name, self.temperature)
    
    # Feature flags
    enable_market_research: bool = Field(True, description="Enable market research analysis")
    enable_seo_analysis: bool = Field(True, description="Enable SEO analysis")
    enable_domain_analysis: bool = Field(True, description="Enable domain availability analysis")
    enable_translation_analysis: bool = Field(True, description="Enable translation analysis")
    enable_cultural_analysis: bool = Field(True, description="Enable cultural sensitivity analysis")
    enable_semantic_analysis: bool = Field(True, description="Enable semantic analysis")
    enable_linguistic_analysis: bool = Field(True, description="Enable linguistic analysis")
    enable_competitor_analysis: bool = Field(True, description="Enable competitor analysis")
    enable_survey_simulation: bool = Field(True, description="Enable survey simulation")
    
    # Performance settings
    cache_results: bool = Field(True, description="Cache analysis results")
    step_delays: Optional[Dict[str, float]] = Field(
        None, 
        description="Optional delays in seconds for workflow steps"
    )
    
    # Integration settings
    integration_keys: Optional[Dict[str, Any]] = Field(
        None,
        description="External integration API keys and configuration"
    )
    
    # Quality and reporting settings
    report_generation: bool = Field(True, description="Generate final report")
    quality_threshold: float = Field(0.7, description="Minimum quality threshold for brand names") 