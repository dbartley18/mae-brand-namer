"""Agent module exports."""

from .uid_generator import UIDGeneratorAgent
from .brand_context_expert import BrandContextExpert
from .brand_name_creation_expert import BrandNameCreationExpert
from .linguistic_analysis_expert import LinguisticsExpert
from .semantic_analysis_expert import SemanticAnalysisExpert
from .cultural_sensitivity_expert import CulturalSensitivityExpert
# from .translation_analysis_expert import TranslationAnalysisExpert  # No longer used - replaced by language-specific experts
from .domain_analysis_expert import DomainAnalysisExpert
from .seo_online_discovery_expert import SEOOnlineDiscoveryExpert
from .competitor_analysis_expert import CompetitorAnalysisExpert
from .survey_simulation_expert import SurveySimulationExpert
from .market_research_expert import MarketResearchExpert
from .report_compiler import ReportCompiler
from .process_supervisor import ProcessSupervisor
from .brand_name_evaluator import BrandNameEvaluator

# Language-specific experts
from .base_language_expert import BaseLanguageTranslationExpert
from .spanish_translation_expert import SpanishTranslationExpert
from .french_translation_expert import FrenchTranslationExpert
from .german_translation_expert import GermanTranslationExpert
from .chinese_translation_expert import ChineseTranslationExpert
from .japanese_translation_expert import JapaneseTranslationExpert
from .arabic_translation_expert import ArabicTranslationExpert
from .language_expert_factory import (
    get_language_expert,
    get_all_language_experts,
    get_available_languages,
    get_language_display_name
)

__all__ = [
    "UIDGeneratorAgent",
    "BrandContextExpert",
    "BrandNameCreationExpert",
    "LinguisticsExpert",
    "SemanticAnalysisExpert",
    "CulturalSensitivityExpert",
    # "TranslationAnalysisExpert",  # No longer used - replaced by language-specific experts
    "DomainAnalysisExpert",
    "SEOOnlineDiscoveryExpert",
    "CompetitorAnalysisExpert",
    "SurveySimulationExpert",
    "MarketResearchExpert",
    "ReportCompiler",
    "ProcessSupervisor",
    "BrandNameEvaluator",
    
    # Language-specific experts
    "BaseLanguageTranslationExpert",
    "SpanishTranslationExpert",
    "FrenchTranslationExpert",
    "GermanTranslationExpert",
    "ChineseTranslationExpert",
    "JapaneseTranslationExpert",
    "ArabicTranslationExpert",
    
    # Factory methods
    "get_language_expert",
    "get_all_language_experts",
    "get_available_languages",
    "get_language_display_name"
]
