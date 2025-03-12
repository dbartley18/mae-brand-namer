"""
Report Formatter

This module handles the second step of the two-step report generation process.
It pulls raw data from the report_raw_data table and formats it into a polished report.
"""

import os
import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.enum.table import WD_TABLE_ALIGNMENT

from src.mae_brand_namer.utils.supabase_utils import SupabaseManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReportFormatter:
    """
    Handles formatting and generation of reports using raw data from the report_raw_data table.
    This is the second step in the two-step report generation process.
    """

    def __init__(self, supabase_client=None):
        """Initialize the ReportFormatter with a Supabase connection."""
        if supabase_client:
            self.supabase = supabase_client
        else:
            # Create a new connection if none was provided
            logger.info("Initializing ReportFormatter with a new Supabase connection")
            self.supabase = SupabaseManager()

    async def fetch_raw_data(self, run_id: str) -> Dict[str, Any]:
        """Fetch all raw data for a specific run_id from the report_raw_data table."""
        logger.info(f"Fetching all raw data for run_id: {run_id}")
        
        query = f"""
        SELECT section_name, raw_data
        FROM report_raw_data
        WHERE run_id = '{run_id}'
        ORDER BY section_name
        """
        
        result = await self.supabase.execute_with_retry(query, {})
        
        if not result:
            logger.warning(f"No data found for run_id: {run_id}")
            return {}
        
        # Transform results into a dictionary with section_name as keys
        sections_data = {}
        for row in result:
            section_name = row['section_name']
            raw_data = row['raw_data']
            sections_data[section_name] = raw_data
            
        logger.info(f"Found {len(sections_data)} sections for run_id: {run_id}")
        return sections_data

    async def generate_report(self, run_id: str, output_dir: Optional[str] = None) -> str:
        """
        Generate a complete report document for the given run_id.
        
        Args:
            run_id: The run ID to generate a report for
            output_dir: Optional directory to save the report in
            
        Returns:
            Path to the generated report document
        """
        # Create output directory if needed
        if not output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"formatted_report_{timestamp}"
            
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Report generation for {run_id} - Output in {output_dir}")
        
        # Fetch all raw data for this run
        sections_data = await self.fetch_raw_data(run_id)
        
        if not sections_data:
            error_msg = f"No data found for run_id: {run_id}. Cannot generate report."
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        # Create a new document
        doc = Document()
        
        # Add title and metadata
        self._add_report_header(doc, run_id)
        
        # Define the order of sections - updated to match requirements
        section_order = [
            "brand_context",
            "name_generation",
            "semantic_analysis",
            "linguistic_analysis",
            "cultural_sensitivity",
            "brand_name_evaluation",
            "translation_analysis",
            "market_research",
            "competitor_analysis",
            "domain_analysis",
            "survey_simulation",
            "executive_summary",
            "recommendations"
        ]
        
        # Process each section in order
        for section_name in section_order:
            if section_name in sections_data:
                logger.info(f"Formatting section: {section_name}")
                
                # Add section header
                section_title = section_name.replace("_", " ").title()
                heading = doc.add_heading(section_title, level=1)
                
                # Format the section based on its type
                try:
                    self._format_section(doc, section_name, sections_data[section_name])
                except Exception as e:
                    logger.error(f"Error formatting section {section_name}: {str(e)}")
                    # Add error info to the document rather than failing completely
                    doc.add_paragraph(f"Error formatting this section: {str(e)}", style='Intense Quote')
                
                # Add space after section
                doc.add_paragraph()
            else:
                logger.warning(f"Missing section: {section_name}")
        
        # Save the document
        doc_path = os.path.join(output_dir, f"Report_{run_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx")
        doc.save(doc_path)
        logger.info(f"Report saved to {doc_path}")
        
        return doc_path

    def _add_report_header(self, doc: Document, run_id: str) -> None:
        """Add title and metadata to the report."""
        # Add title
        title = doc.add_heading("Brand Naming Report", level=0)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # Add subtitle with run ID and date
        subtitle = doc.add_paragraph()
        subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
        subtitle.add_run(f"Run ID: {run_id}").italic = True
        subtitle.add_run("\n")
        subtitle.add_run(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}").italic = True
        
        # Add a divider
        doc.add_paragraph("_" * 50)
        doc.add_paragraph()  # Add some space

    def _format_section(self, doc: Document, section_name: str, section_data: Dict[str, Any]) -> None:
        """Format a specific section based on its type."""
        # Use specific formatting methods based on section_name
        if section_name == "executive_summary":
            self._format_executive_summary(doc, section_data)
        elif section_name == "brand_context":
            self._format_brand_context(doc, section_data)
        elif section_name == "name_generation":
            self._format_name_generation(doc, section_data)
        elif section_name == "linguistic_analysis":
            self._format_linguistic_analysis(doc, section_data)
        elif section_name == "semantic_analysis":
            self._format_semantic_analysis(doc, section_data)
        elif section_name == "cultural_sensitivity":
            self._format_cultural_sensitivity(doc, section_data)
        elif section_name == "translation_analysis":
            self._format_translation_analysis(doc, section_data)
        elif section_name == "survey_simulation":
            self._format_survey_simulation(doc, section_data)
        elif section_name == "name_evaluation":
            self._format_name_evaluation(doc, section_data)
        elif section_name == "domain_analysis":
            self._format_domain_analysis(doc, section_data)
        elif section_name == "seo_analysis":
            self._format_seo_analysis(doc, section_data)
        elif section_name == "competitor_analysis":
            self._format_competitor_analysis(doc, section_data)
        elif section_name == "market_research":
            self._format_market_research(doc, section_data)
        elif section_name == "recommendations":
            self._format_recommendations(doc, section_data)
        else:
            # Generic formatting if no specific method exists
            self._format_generic_section(doc, section_name, section_data)

    def _format_executive_summary(self, doc: Document, data: Dict[str, Any]) -> None:
        """Format the executive summary section."""
        # Extract key data
        shortlisted_names = data.get("shortlisted_names", [])
        total_names = data.get("total_names_generated", 0)
        
        # Add introduction
        doc.add_paragraph(
            f"The brand naming process generated {total_names} potential names, "
            f"of which {len(shortlisted_names)} have been shortlisted for your consideration."
        )
        
        # Add shortlisted names
        if shortlisted_names:
            doc.add_heading("Shortlisted Names", level=2)
            table = doc.add_table(rows=1, cols=2)
            table.style = 'Table Grid'
            
            # Add headers
            header_cells = table.rows[0].cells
            header_cells[0].text = "Brand Name"
            header_cells[1].text = "Overall Score"
            
            # Add rows
            for name in shortlisted_names:
                row = table.add_row().cells
                row[0].text = name.get("brand_name", "")
                row[1].text = str(name.get("overall_score", ""))

    def _format_brand_context(self, doc: Document, data: Dict[str, Any]) -> None:
        """Format the brand context section according to requirements."""
        # Extract brand context data
        context = data.get("brand_context", {})
        if not context:
            doc.add_paragraph("No brand context data available.")
            return
            
        # Add key elements of brand context in the required order
        for field in [
            "brand_promise", 
            "brand_personality", 
            "brand_tone_of_voice", 
            "brand_values", 
            "brand_purpose", 
            "brand_mission", 
            "target_audience",
            "customer_needs", 
            "market_positioning", 
            "competitive_landscape",
            "industry_focus", 
            "industry_trends", 
            "brand_identity_brief"
        ]:
            if field in context:
                heading = field.replace("_", " ").title()
                doc.add_heading(heading, level=2)
                
                # Handle arrays vs text fields
                if isinstance(context[field], list):
                    for item in context[field]:
                        doc.add_paragraph(f"• {item}", style='List Bullet')
                elif isinstance(context[field], dict):
                    # Handle JSON fields
                    for key, value in context[field].items():
                        p = doc.add_paragraph(style='List Bullet')
                        p.add_run(f"{key}: ").bold = True
                        p.add_run(str(value))
                else:
                    doc.add_paragraph(context[field])

    def _format_name_generation(self, doc: Document, data: Dict[str, Any]) -> None:
        """Format the name generation section according to requirements."""
        # Extract name generation data
        names = data.get("name_generations", [])
        if not names:
            doc.add_paragraph("No name generation data available.")
            return
            
        # Group names by category as required
        categories = {}
        for name in names:
            category = name.get("naming_category", "Uncategorized")
            if category not in categories:
                categories[category] = []
            categories[category].append(name)
            
        # Add summary
        doc.add_paragraph(f"A total of {len(names)} brand names were generated across {len(categories)} different categories.")
        
        # Add each category with all required fields
        for category, names_list in categories.items():
            doc.add_heading(f"{category}", level=2)
            
            # Create a table for names in this category with all required fields
            table = doc.add_table(rows=1, cols=5)
            table.style = 'Table Grid'
            
            # Add headers
            header_cells = table.rows[0].cells
            header_cells[0].text = "Brand Name"
            header_cells[1].text = "Personality Alignment"
            header_cells[2].text = "Promise Alignment"
            header_cells[3].text = "Memorability"
            header_cells[4].text = "Market Differentiation"
            
            # Add rows with all the required data
            for name in names_list:
                row = table.add_row().cells
                row[0].text = name.get("brand_name", "")
                row[1].text = name.get("brand_personality_alignment", "")
                row[2].text = name.get("brand_promise_alignment", "")
                row[3].text = name.get("memorability_score_details", "")
                row[4].text = name.get("market_differentiation_details", "")
                
            # Add some space after the table
            doc.add_paragraph()

    def _format_linguistic_analysis(self, doc: Document, data: Dict[str, Any]) -> None:
        """Format the linguistic analysis section according to requirements."""
        # Extract linguistic analysis data
        analyses = data.get("linguistic_analyses", [])
        if not analyses:
            doc.add_paragraph("No linguistic analysis data available.")
            return
            
        # Add introduction
        doc.add_paragraph(
            f"Linguistic analysis was performed on {len(analyses)} brand names to evaluate "
            "pronunciation, readability, and other linguistic features."
        )
        
        # Group by brand name as required
        brands = {}
        for analysis in analyses:
            brand_name = analysis.get("brand_name", "Unknown")
            brands[brand_name] = analysis
        
        # Process each brand
        for brand_name, analysis in brands.items():
            doc.add_heading(brand_name, level=2)
            
            # Create a table with key linguistic features
            table = doc.add_table(rows=6, cols=2)
            table.style = 'Table Grid'
            
            # Add rows with field-value pairs
            row = table.rows[0].cells
            row[0].text = "Pronunciation Ease"
            row[0].paragraphs[0].runs[0].bold = True
            row[1].text = analysis.get("pronunciation_ease", "")
            
            row = table.rows[1].cells
            row[0].text = "Euphony vs Cacophony"
            row[0].paragraphs[0].runs[0].bold = True
            row[1].text = analysis.get("euphony_vs_cacophony", "")
            
            row = table.rows[2].cells
            row[0].text = "Word Class"
            row[0].paragraphs[0].runs[0].bold = True
            row[1].text = analysis.get("word_class", "")
            
            row = table.rows[3].cells
            row[0].text = "Semantic Distance from Competitors"
            row[0].paragraphs[0].runs[0].bold = True
            row[1].text = analysis.get("semantic_distance_from_competitors", "")
            
            row = table.rows[4].cells
            row[0].text = "Ease of Marketing Integration"
            row[0].paragraphs[0].runs[0].bold = True
            row[1].text = analysis.get("ease_of_marketing_integration", "")
            
            row = table.rows[5].cells
            row[0].text = "Overall Readability Score"
            row[0].paragraphs[0].runs[0].bold = True
            row[1].text = analysis.get("overall_readability_score", "")
            
            # Add notes if available
            if analysis.get("notes"):
                doc.add_heading("Notes", level=3)
                doc.add_paragraph(analysis.get("notes"))
            
            # Add some space between brands
            doc.add_paragraph()

    def _format_semantic_analysis(self, doc: Document, data: Dict[str, Any]) -> None:
        """Format the semantic analysis section."""
        # Extract semantic analysis data
        analyses = data.get("semantic_analyses", [])
        if not analyses:
            doc.add_paragraph("No semantic analysis data available.")
            return
            
        # Add introduction
        doc.add_paragraph(
            f"Semantic analysis was conducted for {len(analyses)} brand names to understand "
            "their meaning, emotional valence, and associations."
        )
        
        # Create a table
        table = doc.add_table(rows=1, cols=3)
        table.style = 'Table Grid'
        
        # Add headers
        header_cells = table.rows[0].cells
        header_cells[0].text = "Brand Name"
        header_cells[1].text = "Denotative Meaning"
        header_cells[2].text = "Emotional Valence"
        
        # Add data rows
        for analysis in analyses:
            row = table.add_row().cells
            row[0].text = analysis.get("brand_name", "")
            row[1].text = analysis.get("denotative_meaning", "")
            row[2].text = analysis.get("emotional_valence", "")

    def _format_cultural_sensitivity(self, doc: Document, data: Dict[str, Any]) -> None:
        """Format the cultural sensitivity section."""
        # Extract cultural sensitivity data
        analyses = data.get("cultural_analyses", [])
        if not analyses:
            doc.add_paragraph("No cultural sensitivity analysis data available.")
            return
            
        # Add introduction
        doc.add_paragraph(
            f"Cultural sensitivity analysis was performed on {len(analyses)} brand names to identify "
            "any potential cultural issues or sensitivities."
        )
        
        # Create a table
        table = doc.add_table(rows=1, cols=3)
        table.style = 'Table Grid'
        
        # Add headers
        header_cells = table.rows[0].cells
        header_cells[0].text = "Brand Name"
        header_cells[1].text = "Overall Risk Rating"
        header_cells[2].text = "Cultural Connotations"
        
        # Add data rows
        for analysis in analyses:
            row = table.add_row().cells
            row[0].text = analysis.get("brand_name", "")
            row[1].text = analysis.get("overall_risk_rating", "")
            row[2].text = analysis.get("cultural_connotations", "")

    def _format_translation_analysis(self, doc: Document, data: Dict[str, Any]) -> None:
        """Format the translation analysis section according to requirements."""
        # Extract translation analysis data
        analyses = data.get("translation_analyses", [])
        if not analyses:
            doc.add_paragraph("No translation analysis data available.")
            return
            
        # Add introduction
        doc.add_paragraph(
            f"Translation analysis was conducted for {len(analyses)} brand name-language combinations "
            "to evaluate international viability."
        )
        
        # Group by brand name as required
        brands = {}
        for analysis in analyses:
            brand_name = analysis.get("brand_name", "Unknown")
            if brand_name not in brands:
                brands[brand_name] = []
            brands[brand_name].append(analysis)
        
        # Process each brand
        for brand_name, brand_analyses in brands.items():
            doc.add_heading(brand_name, level=2)
            
            # Create a table for this brand with all required fields
            table = doc.add_table(rows=1, cols=5)
            table.style = 'Table Grid'
            
            # Add headers
            header_cells = table.rows[0].cells
            header_cells[0].text = "Target Language"
            header_cells[1].text = "Direct Translation"
            header_cells[2].text = "Cultural Acceptability"
            header_cells[3].text = "Adaptation Needed"
            header_cells[4].text = "Brand Essence Preserved"
            
            # Add data rows with all required fields
            for analysis in brand_analyses:
                row = table.add_row().cells
                row[0].text = analysis.get("target_language", "")
                row[1].text = analysis.get("direct_translation", "")
                row[2].text = analysis.get("cultural_acceptability", "")
                row[3].text = "Yes" if analysis.get("adaptation_needed") else "No"
                row[4].text = analysis.get("brand_essence_preserved", "")
                
                # If adaptation needed, show the proposed adaptation
                if analysis.get("adaptation_needed") and analysis.get("proposed_adaptation"):
                    additional_row = table.add_row().cells
                    additional_row[0].text = "Proposed Adaptation:"
                    additional_row[1].text = analysis.get("proposed_adaptation", "")
                    additional_row[1].merge(additional_row[4])
            
            # Add space after table
            doc.add_paragraph()

    def _format_survey_simulation(self, doc: Document, data: Dict[str, Any]) -> None:
        """Format the survey simulation section according to requirements."""
        # Extract survey simulation data
        surveys = data.get("survey_simulations", [])
        if not surveys:
            doc.add_paragraph("No survey simulation data available.")
            return
            
        # Add introduction
        doc.add_paragraph(
            f"Market survey simulations were conducted for {len(surveys)} brand names "
            "to evaluate market reception and adoption potential."
        )
        
        # Group by brand name as required
        brands = {}
        for survey in surveys:
            brand_name = survey.get("brand_name", "Unknown")
            if brand_name not in brands:
                brands[brand_name] = []
            brands[brand_name].append(survey)
        
        # Process each brand
        for brand_name, brand_surveys in brands.items():
            doc.add_heading(brand_name, level=2)
            
            # Create a summary table with key metrics
            summary_table = doc.add_table(rows=1, cols=5)
            summary_table.style = 'Table Grid'
            
            # Add headers
            headers = summary_table.rows[0].cells
            headers[0].text = "Brand Promise Score"
            headers[1].text = "Personality Fit"
            headers[2].text = "Competitive Diff."
            headers[3].text = "Market Adoption"
            headers[4].text = "Strategic Ranking"
            
            # Calculate averages
            avg_promise = sum(float(s.get("brand_promise_perception_score", 0)) for s in brand_surveys) / len(brand_surveys)
            avg_personality = sum(float(s.get("personality_fit_score", 0)) for s in brand_surveys) / len(brand_surveys)
            avg_diff = sum(float(s.get("competitive_differentiation_score", 0)) for s in brand_surveys) / len(brand_surveys)
            avg_adoption = sum(float(s.get("simulated_market_adoption_score", 0)) for s in brand_surveys) / len(brand_surveys)
            
            # Get strategic ranking (use the first one as representative)
            strategic_ranking = brand_surveys[0].get("strategic_ranking", "N/A")
            
            # Add summary row
            summary_row = summary_table.add_row().cells
            summary_row[0].text = f"{avg_promise:.1f}"
            summary_row[1].text = f"{avg_personality:.1f}"
            summary_row[2].text = f"{avg_diff:.1f}"
            summary_row[3].text = f"{avg_adoption:.1f}"
            summary_row[4].text = str(strategic_ranking)
            
            # Qualitative feedback
            doc.add_heading("Qualitative Feedback", level=3)
            for i, survey in enumerate(brand_surveys):
                if survey.get("qualitative_feedback_summary"):
                    p = doc.add_paragraph(style='List Bullet')
                    p.add_run(f"Feedback {i+1}: ").bold = True
                    p.add_run(survey.get("qualitative_feedback_summary", ""))
            
            # Final recommendation if available
            if brand_surveys[0].get("final_survey_recommendation"):
                doc.add_heading("Final Recommendation", level=3)
                doc.add_paragraph(brand_surveys[0].get("final_survey_recommendation"))
            
            # Add demographic information if available
            doc.add_heading("Survey Demographics", level=3)
            demo_table = doc.add_table(rows=5, cols=2)
            demo_table.style = 'Table Grid'
            
            row = demo_table.rows[0].cells
            row[0].text = "Industry"
            row[1].text = brand_surveys[0].get("industry", "N/A")
            
            row = demo_table.rows[1].cells
            row[0].text = "Company Size"
            row[1].text = brand_surveys[0].get("company_size_employees", "N/A")
            
            row = demo_table.rows[2].cells
            row[0].text = "Job Titles"
            row[1].text = ", ".join(set(s.get("job_title", "N/A") for s in brand_surveys if s.get("job_title")))
            
            row = demo_table.rows[3].cells
            row[0].text = "Departments"
            row[1].text = ", ".join(set(s.get("department", "N/A") for s in brand_surveys if s.get("department")))
            
            row = demo_table.rows[4].cells
            row[0].text = "Seniority Levels"
            row[1].text = ", ".join(set(s.get("seniority", "N/A") for s in brand_surveys if s.get("seniority")))
            
            # Add separation between brands
            doc.add_paragraph()
            doc.add_paragraph("---")
            doc.add_paragraph()

    def _format_name_evaluation(self, doc: Document, data: Dict[str, Any]) -> None:
        """Format the name evaluation section."""
        # Extract name evaluation data
        evaluations = data.get("evaluations", [])
        if not evaluations:
            doc.add_paragraph("No name evaluation data available.")
            return
            
        # Add introduction
        doc.add_paragraph(
            f"Comprehensive evaluations were performed for {len(evaluations)} brand names "
            "to assess their overall performance and suitability."
        )
        
        # Create a table
        table = doc.add_table(rows=1, cols=3)
        table.style = 'Table Grid'
        
        # Add headers
        header_cells = table.rows[0].cells
        header_cells[0].text = "Brand Name"
        header_cells[1].text = "Overall Score"
        header_cells[2].text = "Shortlisted"
        
        # Add data rows
        for eval in evaluations:
            row = table.add_row().cells
            row[0].text = eval.get("brand_name", "")
            row[1].text = str(eval.get("overall_score", ""))
            row[2].text = "Yes" if eval.get("shortlist_status") else "No"
            
        # Highlight top performers
        shortlisted = [e for e in evaluations if e.get("shortlist_status")]
        if shortlisted:
            doc.add_heading("Shortlisted Names", level=2)
            for name in shortlisted:
                p = doc.add_paragraph(style='List Bullet')
                p.add_run(name.get("brand_name", "")).bold = True
                p.add_run(f" - {name.get('evaluation_comments', '')}")

    def _format_domain_analysis(self, doc: Document, data: Dict[str, Any]) -> None:
        """Format the domain analysis section."""
        # Extract domain analysis data
        analyses = data.get("domain_analyses", [])
        if not analyses:
            doc.add_paragraph("No domain analysis data available.")
            return
            
        # Add introduction
        doc.add_paragraph(
            f"Domain availability analysis was conducted for {len(analyses)} brand names "
            "to evaluate online presence opportunities."
        )
        
        # Create a table
        table = doc.add_table(rows=1, cols=3)
        table.style = 'Table Grid'
        
        # Add headers
        header_cells = table.rows[0].cells
        header_cells[0].text = "Brand Name"
        header_cells[1].text = "Domain Exact Match"
        header_cells[2].text = "Acquisition Cost"
        
        # Add data rows
        for analysis in analyses:
            row = table.add_row().cells
            row[0].text = analysis.get("brand_name", "")
            row[1].text = analysis.get("domain_exact_match", "")
            row[2].text = analysis.get("acquisition_cost", "")

    def _format_seo_analysis(self, doc: Document, data: Dict[str, Any]) -> None:
        """Format the SEO analysis section."""
        # Extract SEO analysis data
        analyses = data.get("seo_analyses", [])
        if not analyses:
            doc.add_paragraph("No SEO analysis data available.")
            return
            
        # Add introduction
        doc.add_paragraph(
            f"SEO and online discoverability analysis was performed for {len(analyses)} brand names "
            "to evaluate search engine performance potential."
        )
        
        # Create a table
        table = doc.add_table(rows=1, cols=3)
        table.style = 'Table Grid'
        
        # Add headers
        header_cells = table.rows[0].cells
        header_cells[0].text = "Brand Name"
        header_cells[1].text = "SEO Viability Score"
        header_cells[2].text = "Search Volume"
        
        # Add data rows
        for analysis in analyses:
            row = table.add_row().cells
            row[0].text = analysis.get("brand_name", "")
            row[1].text = analysis.get("seo_viability_score", "")
            row[2].text = analysis.get("search_volume", "")
            
        # Add SEO recommendations
        doc.add_heading("SEO Recommendations", level=2)
        for analysis in analyses:
            if analysis.get("seo_recommendations"):
                p = doc.add_paragraph(style='List Bullet')
                p.add_run(f"{analysis.get('brand_name', '')}: ").bold = True
                p.add_run(analysis.get("seo_recommendations", ""))

    def _format_competitor_analysis(self, doc: Document, data: Dict[str, Any]) -> None:
        """Format the competitor analysis section."""
        # Extract competitor analysis data
        analyses = data.get("competitor_analyses", [])
        if not analyses:
            doc.add_paragraph("No competitor analysis data available.")
            return
            
        # Add introduction
        doc.add_paragraph(
            "Competitor analysis was performed to evaluate the proposed brand names "
            "against existing competitors in the marketplace."
        )
        
        # Group by brand name
        brands = {}
        for analysis in analyses:
            brand_name = analysis.get("brand_name", "Unknown")
            if brand_name not in brands:
                brands[brand_name] = []
            brands[brand_name].append(analysis)
        
        # Process each brand
        for brand_name, competitors in brands.items():
            doc.add_heading(brand_name, level=2)
            
            # Create a table for this brand
            table = doc.add_table(rows=1, cols=3)
            table.style = 'Table Grid'
            
            # Add headers
            header_cells = table.rows[0].cells
            header_cells[0].text = "Competitor"
            header_cells[1].text = "Risk of Confusion"
            header_cells[2].text = "Trademark Conflict Risk"
            
            # Add data rows
            for comp in competitors:
                row = table.add_row().cells
                row[0].text = comp.get("competitor_name", "")
                row[1].text = comp.get("risk_of_confusion", "")
                row[2].text = comp.get("trademark_conflict_risk", "")
            
            # Add space after table
            doc.add_paragraph()

    def _format_market_research(self, doc: Document, data: Dict[str, Any]) -> None:
        """Format the market research section."""
        # Extract market research data
        research = data.get("market_researches", [])
        if not research:
            doc.add_paragraph("No market research data available.")
            return
            
        # Add introduction
        doc.add_paragraph(
            f"Market research was conducted for {len(research)} shortlisted brand names "
            "to evaluate their market viability and potential."
        )
        
        # Process each brand
        for res in research:
            brand_name = res.get("brand_name", "Unknown")
            doc.add_heading(brand_name, level=2)
            
            # Market opportunity
            if res.get("market_opportunity"):
                doc.add_heading("Market Opportunity", level=3)
                doc.add_paragraph(res.get("market_opportunity"))
                
            # Target audience fit
            if res.get("target_audience_fit"):
                doc.add_heading("Target Audience Fit", level=3)
                doc.add_paragraph(res.get("target_audience_fit"))
                
            # Potential risks
            if res.get("potential_risks"):
                doc.add_heading("Potential Risks", level=3)
                doc.add_paragraph(res.get("potential_risks"))
                
            # Add space between brands
            doc.add_paragraph()

    def _format_recommendations(self, doc: Document, data: Dict[str, Any]) -> None:
        """Format the recommendations section."""
        # Extract recommendations data
        shortlisted = data.get("shortlisted_names", [])
        if not shortlisted:
            doc.add_paragraph("No recommendations data available.")
            return
            
        # Add introduction
        doc.add_paragraph(
            "Based on comprehensive analysis of all brand name candidates, "
            "the following recommendations are provided:"
        )
        
        # Create a summary table of shortlisted names
        table = doc.add_table(rows=1, cols=3)
        table.style = 'Table Grid'
        
        # Add headers
        header_cells = table.rows[0].cells
        header_cells[0].text = "Brand Name"
        header_cells[1].text = "Overall Score"
        header_cells[2].text = "Key Strengths"
        
        # Add data rows
        for name in shortlisted:
            row = table.add_row().cells
            row[0].text = name.get("brand_name", "")
            row[1].text = str(name.get("overall_score", ""))
            row[2].text = name.get("evaluation_comments", "")
            
        # Final recommendation
        doc.add_heading("Final Recommendation", level=2)
        if shortlisted:
            top_name = max(shortlisted, key=lambda x: x.get("overall_score", 0))
            p = doc.add_paragraph()
            p.add_run(f"The recommended brand name is ").bold = False
            p.add_run(f"{top_name.get('brand_name', '')}").bold = True
            p.add_run(f" with an overall score of {top_name.get('overall_score', '')}.")
            
            # Add some final notes
            doc.add_paragraph(
                "This recommendation is based on a holistic assessment of linguistic qualities, "
                "market potential, domain availability, and audience reception. "
                "The name aligns well with the brand's positioning and offers strong opportunities "
                "for building a distinctive brand identity."
            )

    def _format_generic_section(self, doc: Document, section_name: str, data: Dict[str, Any]) -> None:
        """Format a section with no specific formatting method."""
        # Add a simple dump of the data
        doc.add_paragraph(f"Data for {section_name}:")
        
        # Dump the keys
        keys = ", ".join(data.keys())
        doc.add_paragraph(f"Available data fields: {keys}")
        
        # Try to detect lists and display them
        for key, value in data.items():
            if isinstance(value, list) and value:
                doc.add_heading(f"{key.replace('_', ' ').title()}", level=2)
                doc.add_paragraph(f"Total items: {len(value)}")
                
                # Display a sample of items
                for i, item in enumerate(value[:5]):
                    if isinstance(item, dict):
                        # For dictionaries, show key-value pairs
                        p = doc.add_paragraph(style='List Bullet')
                        p.add_run(f"Item {i+1}: ")
                        for k, v in item.items():
                            if k in ["brand_name", "name", "title"]:
                                p.add_run(f"{v} ").bold = True
                            elif isinstance(v, str) and len(v) < 100:
                                p.add_run(f"{k}: {v}; ")
                    else:
                        # For non-dictionaries, show the item directly
                        doc.add_paragraph(f"• {item}", style='List Bullet')
                
                if len(value) > 5:
                    doc.add_paragraph(f"... and {len(value) - 5} more items.")


async def main(run_id: str = None):
    """Main function to run the formatter."""
    if not run_id:
        # Use a default run ID for testing
        run_id = "mae_20250312_141302_d45cccde"  # Replace with an actual run ID
        
    formatter = ReportFormatter()
    output_path = await formatter.generate_report(run_id)
    print(f"Report generated at: {output_path}")


if __name__ == "__main__":
    # Command-line interface
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate a formatted report from raw data")
    parser.add_argument("run_id", help="The run ID to generate a report for")
    args = parser.parse_args()
    
    asyncio.run(main(args.run_id)) 