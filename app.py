import streamlit as st
import asyncio
from mae_brand_namer.workflows.brand_naming import BrandNamingWorkflow
from mae_brand_namer.config import Config
from mae_brand_namer.utils.supabase_utils import SupabaseManager
import os
from dotenv import load_dotenv
import tempfile
import io

# Load environment variables
load_dotenv()

# Initialize Supabase
supabase = SupabaseManager()

# Set page config
st.set_page_config(
    page_title="MAE Brand Namer",
    page_icon="🏢",
    layout="wide"
)

# Add title and description
st.title("MAE Brand Namer")
st.markdown("""
This tool helps you generate and analyze brand names for your business using AI.
Simply fill out the form below with your brand requirements, and we'll generate
a comprehensive brand naming report for you.
""")

# Create the input form
with st.form("brand_naming_form"):
    # Basic Information
    st.subheader("Basic Information")
    company_name = st.text_input("Company Name")
    industry = st.text_input("Industry")
    
    # Brand Values
    st.subheader("Brand Values")
    brand_purpose = st.text_area("Brand Purpose")
    mission_statement = st.text_area("Mission Statement")
    vision_statement = st.text_area("Vision Statement")
    
    # Target Audience
    st.subheader("Target Audience")
    target_audience = st.text_area("Describe your target audience")
    
    # Brand Personality
    st.subheader("Brand Personality")
    brand_personality = st.text_area("Describe your desired brand personality")
    
    # Name Preferences
    st.subheader("Name Preferences")
    name_style = st.multiselect(
        "Preferred name styles",
        ["Modern", "Classic", "Fun", "Professional", "Technical", "Abstract"],
        default=["Modern", "Professional"]
    )
    avoid_words = st.text_area("Words or themes to avoid (optional)")
    
    # Submit button
    submitted = st.form_submit_button("Generate Brand Names")

# Handle form submission
if submitted:
    try:
        with st.spinner("Generating brand names and analysis..."):
            # Prepare the input data
            input_data = {
                "company_name": company_name,
                "industry": industry,
                "brand_purpose": brand_purpose,
                "mission_statement": mission_statement,
                "vision_statement": vision_statement,
                "target_audience": target_audience,
                "brand_personality": brand_personality,
                "name_preferences": {
                    "styles": name_style,
                    "avoid_words": avoid_words.split("\n") if avoid_words else []
                }
            }
            
            # Initialize the workflow
            config = Config()
            workflow = BrandNamingWorkflow(config)
            
            # Create a progress bar
            progress_text = "Operation in progress. Please wait."
            progress_bar = st.progress(0, text=progress_text)
            
            # Run the workflow
            @st.cache_data(ttl=3600)  # Cache results for 1 hour
            async def run_workflow(input_data_str):
                # Convert input_data_str back to dict (needed for caching)
                import json
                input_data = json.loads(input_data_str)
                return await workflow.run(input_data)
            
            # Convert input_data to string for caching
            import json
            input_data_str = json.dumps(input_data)
            
            # Run the workflow asynchronously
            result = asyncio.run(run_workflow(input_data_str))
            
            # Update progress
            progress_bar.progress(100, text="Analysis complete!")
            
            # Display results
            st.success("Brand naming analysis complete!")
            
            # Handle the report file
            if result.get("report_path"):
                try:
                    # Read the report file into memory
                    with open(result["report_path"], "rb") as file:
                        report_data = file.read()
                    
                    # Create a download button with the in-memory file
                    st.download_button(
                        label="Download Brand Naming Report",
                        data=report_data,
                        file_name="brand_naming_report.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )
                    
                    # Clean up the temporary file if it exists
                    if os.path.exists(result["report_path"]):
                        try:
                            os.remove(result["report_path"])
                        except Exception:
                            pass  # Ignore cleanup errors
                            
                except Exception as e:
                    st.error("Error preparing the report for download. Please try again.")
                    st.error(f"Error details: {str(e)}")
            
            # Display generated names
            if result.get("names"):
                st.subheader("Generated Brand Names")
                for name in result["names"]:
                    st.write(f"- {name}")
                    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please try again or contact support if the problem persists.")

# Add footer
st.markdown("---")
st.markdown("Powered by MAE Brand Namer") 