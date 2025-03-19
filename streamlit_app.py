import streamlit as st
import requests
import json
import os
import time
import pandas as pd
import altair as alt
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="MAE Brand Namer",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# MAE Brand Namer\nAI-powered brand name generator"
    }
)

# API configuration
API_URL = os.getenv("LANGGRAPH_STUDIO_URL", "https://maestro-b43940c7842f5cde81f5de39d8bc85e4.us.langgraph.app")
ASSISTANT_ID = os.getenv("LANGGRAPH_ASSISTANT_ID", "1136708e-f643-5539-865c-8c28e4c90fbe")
API_KEY = os.getenv("LANGGRAPH_API_KEY")

# Check if API key is set
if not API_KEY:
    st.error("Please set the LANGGRAPH_API_KEY environment variable")
    st.stop()

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "favorite_names" not in st.session_state:
    st.session_state.favorite_names = []
if "current_thread_id" not in st.session_state:
    st.session_state.current_thread_id = None
if "generation_complete" not in st.session_state:
    st.session_state.generation_complete = False

# Example prompts
example_prompts = {
    "Agentic Software": "A B2B enterprise software company providing AI-powered software solutions for Fortune 500 companies",
    "Professional Services": "A global management consulting firm specializing in digital transformation and operational excellence",
    "Financial Services": "An institutional investment management firm focusing on sustainable infrastructure investments",
    "B2B HealthTech Company": "A healthcare technology company providing enterprise solutions for hospital resource management"
}

# Cached API functions
@st.cache_data(ttl=3600)
def fetch_assistants():
    """Fetch available assistants from the API"""
    headers = {"X-Api-Key": API_KEY, "Content-Type": "application/json"}
    try:
        response = requests.post(
            f"{API_URL}/assistants/search",
            headers=headers,
            json={"graph_id": "brand_naming"}
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching assistants: {str(e)}")
        return []

@st.cache_data(ttl=60)
def get_thread_history(thread_id: str):
    """Cached function to get thread history"""
    headers = {"X-Api-Key": API_KEY, "Content-Type": "application/json"}
    try:
        response = requests.get(
            f"{API_URL}/threads/{thread_id}/history",
            headers=headers
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching thread history: {str(e)}")
        return []

def build_complete_prompt(base_prompt, industry, target_audience, geographic_scope, name_style):
    """Build a complete prompt with additional context"""
    prompt_parts = [base_prompt.strip()]
    
    additional_details = []
    if industry and industry != "Other":
        additional_details.append(f"The company is in the {industry} industry.")
    if target_audience:
        additional_details.append(f"The target audience is {target_audience}.")
    if geographic_scope:
        additional_details.append(f"The brand will operate at a {geographic_scope.lower()} level.")
    if name_style:
        additional_details.append(f"The name should have a {', '.join(name_style).lower()} feel.")
    
    if additional_details:
        prompt_parts.append("Additional context: " + " ".join(additional_details))
    
    return " ".join(prompt_parts)

def create_radar_chart(evaluation_data):
    """Create a radar chart for name evaluation metrics"""
    if not evaluation_data or "metrics" not in evaluation_data:
        return None
    
    metrics = evaluation_data.get("metrics", {})
    if not metrics:
        return None
    
    # Extract metric values and prepare for radar chart
    chart_data = pd.DataFrame({
        'category': list(metrics.keys()),
        'value': list(metrics.values())
    })
    
    # Create the radar chart
    chart = alt.Chart(chart_data).mark_line(point=True).encode(
        x=alt.X('category:N', title=None),
        y=alt.Y('value:Q', scale=alt.Scale(domain=[0, 10])),
        order='category',
        tooltip=['category', 'value']
    ).properties(
        width=300,
        height=200
    )
    
    return chart

def add_to_favorites(name):
    """Add a name to favorites"""
    if name not in st.session_state.favorite_names:
        st.session_state.favorite_names.append(name)
        return True
    return False

def remove_from_favorites(name):
    """Remove a name from favorites"""
    if name in st.session_state.favorite_names:
        st.session_state.favorite_names.remove(name)
        return True
    return False

def process_stream_data(stream, container, status_container, progress_bar):
    """Process streaming data from the API"""
    progress_text = ""
    generated_names = []
    evaluations = {}
    
    for i, line in enumerate(stream):
        if not line:
            continue
            
        try:
            # Update progress
            progress_value = min(i % 100 / 100, 0.95)
            progress_bar.progress(progress_value)
            
            # Decode bytes to string
            line_str = line.decode('utf-8')
            # Remove "data: " prefix if it exists
            if line_str.startswith("data: "):
                line_str = line_str[6:]
            data = json.loads(line_str)
            
            event_type = data.get("type")
            
            if event_type == "status":
                progress_text = data.get("message", "")
                status_container.info(f"**{progress_text}**")
            
            elif event_type == "output":
                result = data.get("output", {})
                if isinstance(result, dict):
                    if "generated_names" in result:
                        generated_names = result["generated_names"]
                    if "evaluations" in result:
                        evaluations = result["evaluations"]
                    
                    # Update results in real-time
                    with container:
                        st.empty().markdown("## Generated Names")
                        
                        if generated_names:
                            for name in generated_names:
                                col1, col2 = st.columns([5, 1])
                                with col1:
                                    st.markdown(f"### {name}")
                                with col2:
                                    if name in st.session_state.favorite_names:
                                        if st.button("❤️", key=f"unfav_{name}"):
                                            remove_from_favorites(name)
                                            st.rerun()
                                    else:
                                        if st.button("🤍", key=f"fav_{name}"):
                                            add_to_favorites(name)
                                            st.rerun()
                                
                                if name in evaluations:
                                    with st.expander("View analysis"):
                                        col1, col2 = st.columns([3, 2])
                                        with col1:
                                            st.markdown("#### Analysis")
                                            st.write(evaluations[name].get("analysis", "No analysis available"))
                                        with col2:
                                            chart = create_radar_chart(evaluations[name])
                                            if chart:
                                                st.altair_chart(chart)
                                st.markdown("---")
        
        except json.JSONDecodeError:
            continue
        except Exception as e:
            st.error(f"Error processing stream: {str(e)}")
    
    # Complete progress
    progress_bar.progress(1.0)
    status_container.success("**Generation complete!**")
    
    return generated_names, evaluations

# Main application layout
st.title("MAE Brand Namer")

# Sidebar for inputs
with st.sidebar:
    st.subheader("Brand Requirements")
    
    # Example templates
    st.caption("Quick Templates (click to use)")
    template_cols = st.columns(2)
    
    for i, (name, prompt) in enumerate(example_prompts.items()):
        with template_cols[i % 2]:
            if st.button(name, help=prompt):
                st.session_state.user_input = prompt
                st.rerun()
    
    st.markdown("---")
    
    # Main input
    user_input = st.text_area(
        "Brand Description",
        key="user_input",
        placeholder="Example: A global enterprise software company specializing in supply chain optimization",
        height=120
    )
    
    # Advanced parameters in expander
    with st.expander("Additional Parameters", expanded=False):
        industry = st.selectbox(
            "Industry",
            ["", "Enterprise Technology", "Professional Services", "Financial Services", 
             "Healthcare", "Industrial", "Other"]
        )
        
        target_audience = st.text_input(
            "Target Market",
            placeholder="e.g., Enterprise manufacturing companies"
        )
        
        geographic_scope = st.selectbox(
            "Market Scope",
            ["", "Global Enterprise", "Regional", "National", "Local"]
        )
        
        name_style = st.multiselect(
            "Brand Positioning",
            ["Enterprise", "Technical", "Professional", "Innovative", "Traditional"]
        )
    
    # Generate button
    generate_button = st.button("Generate Brand Names", type="primary", use_container_width=True)
    
    # Display favorites
    if st.session_state.favorite_names:
        st.markdown("---")
        st.subheader("Favorite Names")
        for name in st.session_state.favorite_names:
            cols = st.columns([4, 1])
            with cols[0]:
                st.markdown(f"**{name}**")
            with cols[1]:
                if st.button("✖️", key=f"remove_{name}"):
                    remove_from_favorites(name)
                    st.rerun()

# Main content area with tabs
tab1, tab2 = st.tabs(["Generator", "History"])

with tab1:
    # Message area
    if not user_input.strip():
        st.info("👈 Enter your brand requirements in the sidebar to get started.")
    
    # Results area
    results_container = st.container()
    status_container = st.empty()
    progress_container = st.empty()
    
    # Process generation
    if generate_button:
        if not user_input.strip():
            st.error("Please provide a description of your brand requirements.")
            st.stop()

        # Build complete prompt with additional requirements
        complete_prompt = build_complete_prompt(
            user_input,
            industry,
            target_audience,
            geographic_scope,
            name_style
        )
        
        # Store the current request in session state
        current_run = {
            "prompt": complete_prompt,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "running",
            "results": None,
            "thread_id": None
        }
        st.session_state.history.append(current_run)
        current_index = len(st.session_state.history) - 1
        
        # Clear previous results
        with results_container:
            st.empty()
        
        # Initialize progress bar
        progress_bar = progress_container.progress(0)
        status_container.info("Starting generation...")
        
        try:
            # API headers
            headers = {
                "X-Api-Key": API_KEY,
                "Content-Type": "application/json"
            }
            
            # Create a new thread
            thread_response = requests.post(
                f"{API_URL}/threads",
                headers=headers,
                json={}
            )
            thread_response.raise_for_status()
            thread_id = thread_response.json()["thread_id"]
            
            # Save current thread ID to session state
            st.session_state.current_thread_id = thread_id
            current_run["thread_id"] = thread_id
            
            # Start a run with the user input
            run_response = requests.post(
                f"{API_URL}/threads/{thread_id}/runs/stream",
                headers=headers,
                json={
                    "assistant_id": ASSISTANT_ID,
                    "input": {
                        "user_prompt": complete_prompt
                    }
                },
                stream=True
            )
            run_response.raise_for_status()
            
            # Process the stream
            generated_names, evaluations = process_stream_data(
                run_response.iter_lines(),
                results_container,
                status_container,
                progress_bar
            )
            
            # Update session state
            current_run["status"] = "completed"
            current_run["generated_names"] = generated_names
            current_run["evaluations"] = evaluations
            st.session_state.history[current_index] = current_run
            st.session_state.generation_complete = True

        except requests.RequestException as e:
            st.error(f"Error connecting to the API: {str(e)}")
            current_run["status"] = "failed"
            current_run["error"] = str(e)
            st.session_state.history[current_index] = current_run
            if st.checkbox("Show detailed error"):
                st.code(str(e))

# History tab
with tab2:
    if not st.session_state.history:
        st.info("No generation history yet. Generate some brand names first!")
    else:
        st.subheader("Previous Generations")
        
        for i, run in enumerate(reversed(st.session_state.history)):
            with st.expander(f"Generation {len(st.session_state.history) - i} - {run['timestamp']}"):
                st.write(f"**Status:** {run['status'].title()}")
                st.write(f"**Prompt:** {run['prompt']}")
                
                if run['status'] == "completed" and run.get("generated_names"):
                    st.write("**Generated Names:**")
                    for name in run.get("generated_names", []):
                        cols = st.columns([4, 1])
                        with cols[0]:
                            st.markdown(f"- **{name}**")
                        with cols[1]:
                            if name in st.session_state.favorite_names:
                                if st.button("❤️", key=f"h_unfav_{i}_{name}"):
                                    remove_from_favorites(name)
                                    st.rerun()
                            else:
                                if st.button("🤍", key=f"h_fav_{i}_{name}"):
                                    add_to_favorites(name)
                                    st.rerun()
                
                if run.get("thread_id"):
                    if st.button("Load Full Results", key=f"load_{i}"):
                        thread_data = get_thread_history(run["thread_id"])
                        st.json(thread_data)

# Footer
st.markdown("---")
st.caption("MAE Brand Namer | Powered by LangGraph AI") 