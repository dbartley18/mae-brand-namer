import streamlit as st
import requests
import json
import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="MAE Brand Namer",
    layout="wide"
)

# Define API endpoint and credentials from environment variables
API_URL = os.getenv("LANGGRAPH_STUDIO_URL", "https://maestro-b43940c7842f5cde81f5de39d8bc85e4.us.langgraph.app")
ASSISTANT_ID = os.getenv("LANGGRAPH_ASSISTANT_ID", "1136708e-f643-5539-865c-8c28e4c90fbe")
API_KEY = os.getenv("LANGGRAPH_API_KEY")

# Check if API key is set
if not API_KEY:
    st.error("Please set the LANGGRAPH_API_KEY environment variable")
    st.stop()

# Example prompts
example_prompts = {
    "Enterprise Software": "A B2B enterprise software company providing AI-powered data analytics solutions for Fortune 500 companies",
    "Professional Services": "A global management consulting firm specializing in digital transformation and operational excellence",
    "Financial Services": "An institutional investment management firm focusing on sustainable infrastructure investments",
    "B2B Healthcare": "A healthcare technology company providing enterprise solutions for hospital resource management"
}

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []

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

# Simple Header
st.title("MAE Brand Namer")
st.subheader("Generate brand names using AI")

# Main input
st.write("### Brand Description")
user_input = st.text_area(
    "Enter your brand requirements",
    placeholder="Example: A global enterprise software company specializing in supply chain optimization",
    height=100
)

# Simple Examples
st.write("### Quick Examples (click to use)")
for name, prompt in example_prompts.items():
    if st.button(name):
        st.session_state.user_input = prompt
        st.experimental_rerun()

# Optional parameters
st.write("### Additional Parameters (optional)")
col1, col2 = st.columns(2)

with col1:
    industry = st.selectbox(
        "Industry",
        ["", "Enterprise Technology", "Professional Services", "Financial Services", "Healthcare", "Industrial", "Other"]
    )
    
    target_audience = st.text_input(
        "Target Market",
        placeholder="e.g., Enterprise manufacturing companies"
    )

with col2:
    geographic_scope = st.selectbox(
        "Market Scope",
        ["", "Global Enterprise", "Regional", "National", "Local"]
    )
    
    name_style = st.multiselect(
        "Brand Positioning",
        ["Enterprise", "Technical", "Professional", "Innovative", "Traditional"]
    )

# Generate button
if st.button("Generate Brand Names", type="primary"):
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
        "results": None
    }
    st.session_state.history.append(current_run)
    
    # Show progress and results
    st.write("### Processing...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_area = st.container()
    
    try:
        # Create headers
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
        progress_text = ""
        generated_names = []
        evaluations = {}
        
        for i, line in enumerate(run_response.iter_lines()):
            if line:
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
                        status_text.write(f"**Current Step:** {progress_text}")
                    
                    elif event_type == "output":
                        result = data.get("output", {})
                        if isinstance(result, dict):
                            if "generated_names" in result:
                                generated_names = result["generated_names"]
                            if "evaluations" in result:
                                evaluations = result["evaluations"]
                            
                            # Update results in real-time
                            with results_area:
                                if generated_names:
                                    st.write("### Generated Names")
                                    for name in generated_names:
                                        st.write(f"- **{name}**")
                                
                                if evaluations:
                                    st.write("### Evaluations")
                                    tabs = st.tabs(["Insights", "Details"])
                                    
                                    with tabs[0]:
                                        for name, eval_data in evaluations.items():
                                            with st.expander(name):
                                                st.write(f"**{name}**")
                                                st.write(eval_data.get('analysis', 'No analysis available'))
                                    
                                    with tabs[1]:
                                        st.json(evaluations)
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    st.error(f"Error processing stream: {str(e)}")
        
        # Complete progress
        progress_bar.progress(1.0)
        status_text.write("**Completed!**")
        
        # Get final results
        history_response = requests.get(
            f"{API_URL}/threads/{thread_id}/history",
            headers=headers
        )
        history_response.raise_for_status()
        history = history_response.json()
        
        # Update session state
        current_run["status"] = "completed"
        current_run["results"] = history[-1] if history else None

    except requests.RequestException as e:
        st.error(f"Error connecting to the API: {str(e)}")
        if st.checkbox("Show detailed error"):
            st.code(str(e))

# Show generation history
if st.session_state.history:
    st.write("### Previous Generations")
    for i, run in enumerate(reversed(st.session_state.history)):
        with st.expander(f"Generation {len(st.session_state.history) - i} - {run['timestamp']}"):
            st.write(f"**Status:** {run['status'].title()}")
            st.write(f"**Prompt:** {run['prompt']}")
            if run['results']:
                if st.button("View Full Results", key=f"history_{i}"):
                    st.json(run['results']) 