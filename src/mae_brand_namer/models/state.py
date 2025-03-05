from typing import List, Dict, Optional, Sequence, Any
from typing_extensions import TypedDict, Annotated
from pydantic import BaseModel, Field, ConfigDict
from langchain_core.messages import BaseMessage
from langchain.load.serializable import Serializable
from datetime import datetime

class SemanticAnalysisResult(TypedDict, total=False):
    """Type definition for semantic analysis results."""
    denotative_meaning: str
    etymology: str
    descriptiveness: float
    concreteness: float
    emotional_valence: str
    brand_personality: str
    sensory_associations: str
    figurative_language: str
    ambiguity: bool
    irony_or_paradox: bool
    humor_playfulness: bool
    phoneme_combinations: str
    sound_symbolism: str
    rhyme_rhythm: bool
    alliteration_assonance: bool
    word_length_syllables: int
    compounding_derivation: str
    brand_name_type: str
    memorability_score: float
    original_pronunciation_ease: float
    clarity_understandability: float
    uniqueness_differentiation: float
    brand_fit_relevance: float
    semantic_trademark_risk: str
    rank: float

class BrandNameData(TypedDict, total=False):
    """Type definition for brand name data."""
    brand_name: str
    naming_categories: List[str]
    brand_personality_alignments: List[str]
    brand_promise_alignments: List[str]
    target_audience_relevance: float
    market_differentiation: float
    memorability_scores: float
    pronounceability_scores: float
    visual_branding_potential: float
    name_generation_methodology: str
    name_rankings: float
    semantic_analysis: Optional[SemanticAnalysisResult]

class ErrorInfo(TypedDict, total=True):
    """Type definition for error information."""
    step: str
    error: str

class AppState(TypedDict, total=False):
    """Application state for the brand name generation workflow.
    
    This state object uses TypedDict for type hints without using Pydantic Field.
    """
    # Core fields
    run_id: Optional[str]
    user_prompt: str
    errors: List[ErrorInfo]
    client: Optional[Any]  # Client for LangGraph/LangSmith operations
    
    # Brand context fields
    brand_identity_brief: Optional[str]
    brand_promise: Optional[str]
    brand_values: List[str]
    brand_personality: List[str]
    brand_tone_of_voice: Optional[str]
    brand_purpose: Optional[str]
    brand_mission: Optional[str]
    target_audience: Optional[str]
    customer_needs: List[str]
    market_positioning: Optional[str]
    competitive_landscape: Optional[str]
    industry_focus: Optional[str]
    industry_trends: List[str]
    
    # Brand name generation fields
    generated_names: List[BrandNameData]
    naming_categories: List[str]
    brand_personality_alignments: List[str]
    brand_promise_alignments: List[str]
    target_audience_relevance: List[str]
    market_differentiation: List[str]
    memorability_scores: List[float]
    pronounceability_scores: List[float]
    visual_branding_potential: List[str]
    name_generation_methodology: Optional[str]
    name_rankings: List[float]
    timestamp: Optional[str]
    
    # Process monitoring
    task_statuses: Dict[str, Dict[str, Any]]
    current_task: Optional[str]
    
    # Analysis results
    linguistic_analysis_results: Dict[str, Dict[str, Any]]
    cultural_analysis_results: Dict[str, Dict[str, Any]]
    evaluation_results: Dict[str, Dict[str, Any]]
    shortlisted_names: List[str]
    
    # SEO Analysis Fields
    keyword_alignment: Optional[str]
    search_volume: Optional[float]
    keyword_competition: Optional[str]
    branded_keyword_potential: Optional[str]
    non_branded_keyword_potential: Optional[str]
    exact_match_search_results: Optional[int]
    competitor_domain_strength: Optional[str]
    name_length_searchability: Optional[str]
    unusual_spelling_impact: Optional[bool]
    content_marketing_opportunities: Optional[str]
    social_media_availability: Optional[bool]
    social_media_discoverability: Optional[str]
    negative_keyword_associations: Optional[str]
    negative_search_results: Optional[bool]
    seo_viability_score: Optional[float]
    seo_recommendations: Optional[str]
    seo_analysis_results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="SEO analysis results for shortlisted brand names"
    )
    
    # Market Research Analysis Fields
    market_research_results: List[Dict[str, Any]]
    
    # Domain Analysis Fields
    domain_analysis_results: List[Dict[str, Any]]
    
    # Competitor analysis
    competitor_analysis_results: Dict[str, "CompetitorAnalysisResult"]

class TaskStatus(BaseModel):
    """Status information for a workflow task."""
    
    task_name: str = Field(..., description="Name of the task")
    agent_type: str = Field(..., description="Type of agent executing the task")
    status: str = Field(..., description="Current status of the task")
    start_time: datetime = Field(..., description="When the task started")
    end_time: Optional[datetime] = Field(None, description="When the task completed")
    duration_sec: Optional[int] = Field(None, description="Duration in seconds")
    error_message: Optional[str] = Field(None, description="Error message if task failed")
    retry_count: int = Field(0, description="Number of retry attempts")
    last_retry_at: Optional[datetime] = Field(None, description="Timestamp of last retry")
    retry_status: Optional[str] = Field(None, description="Current retry status")

class LinguisticAnalysisResult(BaseModel):
    """Results of linguistic analysis for a brand name."""
    
    pronunciation_ease: str = Field(..., description="How easily the name can be pronounced")
    euphony_vs_cacophony: str = Field(..., description="Analysis of pleasant vs. harsh sound combinations")
    rhythm_and_meter: str = Field(..., description="Analysis of rhythmic patterns and meter")
    phoneme_frequency_distribution: str = Field(..., description="Distribution of sound units")
    sound_symbolism: str = Field(..., description="How sounds contribute to meaning")
    word_class: str = Field(..., description="Grammatical category (noun, verb, etc.)")
    morphological_transparency: str = Field(..., description="Clarity of word structure")
    grammatical_gender: str = Field(..., description="Gender implications in relevant languages")
    inflectional_properties: str = Field(..., description="How the word changes in different contexts")
    ease_of_marketing_integration: str = Field(..., description="Usability in marketing contexts")
    naturalness_in_collocations: str = Field(..., description="How naturally it combines with other words")
    homophones_homographs: bool = Field(..., description="Similar sounding or looking words")
    semantic_distance_from_competitors: str = Field(..., description="Linguistic uniqueness")
    neologism_appropriateness: str = Field(..., description="Effectiveness as a new word")
    overall_readability_score: str = Field(..., description="Overall ease of reading and understanding")
    notes: str = Field(..., description="Additional linguistic observations")
    rank: float = Field(..., description="Overall linguistic effectiveness score (1-10)")

class CulturalAnalysisResult(BaseModel):
    """Results of cultural sensitivity analysis for a brand name."""
    cultural_connotations: str = Field(
        ..., 
        description="Cultural associations across target markets"
    )
    symbolic_meanings: str = Field(
        ..., 
        description="Symbolic meanings in different cultures"
    )
    alignment_with_cultural_values: str = Field(
        ..., 
        description="How well the name aligns with societal norms"
    )
    religious_sensitivities: str = Field(
        ..., 
        description="Any unintended religious implications"
    )
    social_political_taboos: str = Field(
        ..., 
        description="Potential sociopolitical sensitivities"
    )
    body_part_bodily_function_connotations: bool = Field(
        ..., 
        description="Unintended anatomical/physiological meanings"
    )
    age_related_connotations: str = Field(
        ..., 
        description="Age-related perception of the name"
    )
    gender_connotations: str = Field(
        ..., 
        description="Any unintentional gender bias"
    )
    regional_variations: str = Field(
        ..., 
        description="Perception in different dialects and subcultures"
    )
    historical_meaning: str = Field(
        ..., 
        description="Historical or traditional significance"
    )
    current_event_relevance: str = Field(
        ..., 
        description="Connection to current events or trends"
    )
    overall_risk_rating: str = Field(
        ..., 
        description="Overall cultural risk assessment"
    )
    notes: str = Field(
        ..., 
        description="Additional cultural observations"
    )
    rank: float = Field(
        ..., 
        description="Overall cultural sensitivity score (1-10)"
    )

class BrandNameEvaluationResult(BaseModel):
    """Results of brand name evaluation and scoring."""
    
    strategic_alignment_score: float = Field(..., description="How well the name aligns with the Brand Identity Brief (1-10)")
    distinctiveness_score: float = Field(..., description="How unique the name is compared to competitors (1-10)")
    competitive_advantage: str = Field(..., description="Analysis of competitive differentiation")
    brand_fit_score: float = Field(..., description="How well the name aligns with brand strategy (1-10)")
    positioning_strength: str = Field(..., description="Effectiveness in market positioning")
    memorability_score: float = Field(..., description="How easy the name is to recall (1-10)")
    pronounceability_score: float = Field(..., description="How easily the name is spoken (1-10)")
    meaningfulness_score: float = Field(..., description="Clarity and positive connotation (1-10)")
    phonetic_harmony: str = Field(..., description="Analysis of sound patterns and flow")
    visual_branding_potential: str = Field(..., description="Potential for visual identity development")
    storytelling_potential: str = Field(..., description="Capacity for brand narrative development")
    domain_viability_score: float = Field(..., description="Initial domain name availability assessment (1-10)")
    overall_score: float = Field(..., description="Total weighted evaluation score (1-10)")
    shortlist_status: str = Field(..., description="Whether selected for final round (Yes/No)")
    evaluation_comments: str = Field(..., description="Detailed rationale for evaluation")
    rank: float = Field(..., description="Final ranking among all candidates (1-N)")

class CompetitorAnalysisResult(BaseModel):
    """Results of competitor analysis for a brand name."""
    
    competitor_name: str = Field(..., description="Name of the competitor being analyzed")
    competitor_naming_style: str = Field(..., description="Whether competitors use descriptive, abstract, or other naming styles")
    competitor_keywords: str = Field(..., description="Common words or themes in competitor brand names")
    competitor_positioning: str = Field(..., description="How competitors position their brands")
    competitor_strengths: str = Field(..., description="Strengths of competitor names")
    competitor_weaknesses: str = Field(..., description="Weaknesses of competitor names")
    competitor_differentiation_opportunity: str = Field(..., description="How to create differentiation")
    differentiation_score: float = Field(..., description="Quantified differentiation from competitors (1-10)")
    risk_of_confusion: str = Field(..., description="Likelihood of brand confusion")
    target_audience_perception: str = Field(..., description="How consumers may compare this name to competitors")
    competitive_advantage_notes: str = Field(..., description="Any competitive advantages of the brand name")
    trademark_conflict_risk: str = Field(..., description="Potential conflicts with existing trademarks")

# Keeping the BrandNameGenerationState with proper v2 config for backward compatibility
class BrandNameGenerationState(Serializable, BaseModel):
    """State model for the brand name generation workflow."""
    
    # Core fields
    run_id: Optional[str] = Field(default=None, description="Unique identifier for this workflow run")
    user_prompt: str = Field(description="Original user prompt describing the brand")
    errors: List[ErrorInfo] = Field(default_factory=list, description="List of errors encountered during workflow")
    start_time: Optional[str] = Field(default=None, description="Timestamp when the workflow started")
    status: Optional[str] = Field(default=None, description="Current status of the workflow")
    
    # Message tracking for LangGraph Studio
    messages: List[BaseMessage] = Field(default_factory=list, description="List of messages in the conversation")
    
    # Brand context fields
    brand_identity_brief: Optional[str] = Field(default=None, description="Comprehensive brand identity brief")
    brand_promise: Optional[str] = Field(default=None, description="Core brand promise")
    brand_values: List[str] = Field(default_factory=list, description="List of brand values")
    brand_personality: List[str] = Field(default_factory=list, description="List of brand personality traits")
    brand_tone_of_voice: Optional[str] = Field(default=None, description="Brand's tone of voice")
    brand_purpose: Optional[str] = Field(default=None, description="Brand's purpose statement")
    brand_mission: Optional[str] = Field(default=None, description="Brand's mission statement")
    target_audience: Optional[str] = Field(default=None, description="Description of target audience")
    customer_needs: List[str] = Field(default_factory=list, description="List of customer needs")
    market_positioning: Optional[str] = Field(default=None, description="Brand's market positioning")
    competitive_landscape: Optional[str] = Field(default=None, description="Overview of competitive landscape")
    industry_focus: Optional[str] = Field(default=None, description="Primary industry focus")
    industry_trends: List[str] = Field(default_factory=list, description="List of relevant industry trends")
    
    # Brand name generation fields (from tasks.yaml)
    generated_names: List[BrandNameData] = Field(default_factory=list, description="List of generated brand names with metadata")
    brand_name: Optional[str] = Field(default=None, description="The selected brand name")
    naming_category: Optional[str] = Field(
        default=None, 
        description="Category of the selected brand name (Descriptive, Evocative/Suggestive, Invented/Coined/Abstract, Experiential, Founder/Personal, Geographic, Symbolic)"
    )
    brand_personality_alignment: Optional[str] = Field(default=None, description="Alignment with brand personality")
    brand_promise_alignment: Optional[str] = Field(default=None, description="Alignment with brand promise")
    target_audience_relevance: Optional[float] = Field(default=None, description="Relevance to target audience (1-10)")
    market_differentiation: Optional[float] = Field(default=None, description="Market differentiation score (1-10)")
    visual_branding_potential: Optional[Any] = Field(default=None, description="Visual branding potential (string or score)")
    memorability_score: Optional[float] = Field(default=None, description="Memorability score (1-10)")
    pronounceability_score: Optional[float] = Field(default=None, description="Pronounceability score (1-10)")
    
    # Details fields for brand name generation
    target_audience_relevance_details: Optional[str] = Field(default=None, description="Details about target audience relevance")
    market_differentiation_details: Optional[str] = Field(default=None, description="Details about market differentiation")
    visual_branding_potential_details: Optional[str] = Field(default=None, description="Details about visual branding potential")
    memorability_score_details: Optional[str] = Field(default=None, description="Details about memorability score")
    pronounceability_score_details: Optional[str] = Field(default=None, description="Details about pronounceability score")
    
    name_generation_methodology: Optional[str] = Field(default=None, description="Methodology used for name generation")
    timestamp: Optional[str] = Field(default=None, description="Timestamp of name generation")
    rank: Optional[float] = Field(default=None, description="Ranking score based on strategic fit")
    
    # Lists for multiple brand names
    naming_categories: List[str] = Field(
        default_factory=list, 
        description="Categories for each generated name (Descriptive, Evocative/Suggestive, Invented/Coined/Abstract, Experiential, Founder/Personal, Geographic, Symbolic)"
    )
    brand_personality_alignments: List[str] = Field(default_factory=list, description="Alignment with brand personality for each name")
    brand_promise_alignments: List[str] = Field(default_factory=list, description="Alignment with brand promise for each name")
    target_audience_relevance_list: List[str] = Field(default_factory=list, description="Relevance to target audience for each name")
    market_differentiation_list: List[str] = Field(default_factory=list, description="Market differentiation for each name")
    memorability_scores: List[float] = Field(default_factory=list, description="Memorability scores for each name")
    pronounceability_scores: List[float] = Field(default_factory=list, description="Pronounceability scores for each name")
    visual_branding_potential_list: List[str] = Field(default_factory=list, description="Visual branding potential for each name")
    name_rankings: List[float] = Field(default_factory=list, description="Ranking scores for each name")
    
    # Evaluation fields from process_evaluation function
    strategic_alignment_score: Optional[int] = Field(default=None, description="How well the name aligns with the Brand Identity Brief (1-10)")
    distinctiveness_score: Optional[int] = Field(default=None, description="How unique the name is compared to competitors (1-10)")
    competitive_advantage: Optional[str] = Field(default=None, description="Analysis of competitive differentiation")
    brand_fit_score: Optional[int] = Field(default=None, description="How well the name aligns with brand strategy (1-10)")
    positioning_strength: Optional[str] = Field(default=None, description="Effectiveness in market positioning")
    meaningfulness_score: Optional[int] = Field(default=None, description="Clarity and positive connotation (1-10)")
    phonetic_harmony: Optional[str] = Field(default=None, description="Analysis of sound patterns and flow")
    storytelling_potential: Optional[str] = Field(default=None, description="Capacity for brand narrative development")
    domain_viability_score: Optional[int] = Field(default=None, description="Initial domain name availability assessment (1-10)")
    overall_score: Optional[int] = Field(default=None, description="Total weighted evaluation score (1-10)")
    shortlist_status: Optional[bool] = Field(default=None, description="Whether selected for final round")
    evaluation_comments: Optional[str] = Field(default=None, description="Detailed rationale for evaluation")
    
    # Process monitoring
    task_statuses: Dict[str, TaskStatus] = Field(
        default_factory=dict,
        description="Status information for each task"
    )
    current_task: Optional[str] = Field(None, description="Currently executing task")
    
    # Linguistic analysis
    linguistic_analysis_results: Dict[str, LinguisticAnalysisResult] = Field(
        default_factory=dict,
        description="Linguistic analysis results for each brand name"
    )
    
    # Semantic analysis
    semantic_analysis_results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Semantic analysis results for each brand name"
    )
    
    # Cultural sensitivity analysis
    cultural_analysis_results: Dict[str, CulturalAnalysisResult] = Field(
        default_factory=dict,
        description="Cultural sensitivity analysis results for each brand name"
    )
    
    # Translation analysis
    translation_analysis_results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Translation analysis results for each brand name"
    )
    
    # Consolidated analysis results
    analysis_results: Dict[str, List[Dict[str, Any]]] = Field(
        default_factory=dict,
        description="Combined results from all analyses"
    )
    
    # Brand name evaluation
    evaluation_results: Dict[str, BrandNameEvaluationResult] = Field(
        default_factory=dict,
        description="Evaluation and scoring results for each brand name"
    )
    shortlisted_names: List[str] = Field(
        default_factory=list,
        description="List of brand names selected for the final shortlist"
    )
    
    # SEO Analysis Fields
    keyword_alignment: Optional[str] = None
    search_volume: Optional[float] = None
    keyword_competition: Optional[str] = None
    branded_keyword_potential: Optional[str] = None
    non_branded_keyword_potential: Optional[str] = None
    exact_match_search_results: Optional[int] = None
    competitor_domain_strength: Optional[str] = None
    name_length_searchability: Optional[str] = None
    unusual_spelling_impact: Optional[bool] = None
    content_marketing_opportunities: Optional[str] = None
    social_media_availability: Optional[bool] = None
    social_media_discoverability: Optional[str] = None
    negative_keyword_associations: Optional[str] = None
    negative_search_results: Optional[bool] = None
    seo_viability_score: Optional[float] = None
    seo_recommendations: Optional[str] = None
    seo_analysis_results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="SEO analysis results for shortlisted brand names"
    )
    
    # Market Research Analysis Fields
    market_research_results: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Market research analysis results for shortlisted brand names"
    )
    
    # Domain Analysis Fields
    domain_analysis_results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Domain analysis results for shortlisted brand names"
    )
    
    # Competitor analysis
    competitor_analysis_results: Dict[str, "CompetitorAnalysisResult"] = Field(
        default_factory=dict,
        description="Competitor analysis results for each brand name"
    )
    
    # Brand evaluation fields
    brand_name_data: Optional[Dict[str, Any]] = Field(default=None, description="Combined brand name evaluation data")
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def dict(self, *args, **kwargs):
        """Convert state to dictionary, handling BaseMessage serialization."""
        d = super().model_dump(*args, **kwargs)
        if "messages" in d:
            d["messages"] = [
                {
                    "type": msg.__class__.__name__,
                    "content": msg.content,
                    "additional_kwargs": msg.additional_kwargs
                }
                for msg in self.messages
            ]
        return d
    
    def update_task_status(
        self,
        task_name: str,
        agent_type: str,
        status: str,
        error: Optional[Exception] = None
    ) -> None:
        """Update the status of a task in the workflow."""
        now = datetime.now()
        
        if task_name not in self.task_statuses:
            # Create new task status
            self.task_statuses[task_name] = TaskStatus(
                task_name=task_name,
                agent_type=agent_type,
                status=status,
                start_time=now,
                end_time=None,
                duration_sec=None
            )
        else:
            # Update existing task status
            task_status = self.task_statuses[task_name]
            task_status.status = status
            
            if status == "completed":
                task_status.end_time = now
                if task_status.start_time:
                    duration = now - task_status.start_time
                    task_status.duration_sec = int(duration.total_seconds())
            
            if status == "error" and error:
                task_status.error_message = str(error)
                
        # Update current task
        if status == "in_progress":
            self.current_task = task_name
        elif self.current_task == task_name:
            self.current_task = None 