import streamlit as st
import requests
import json
import os
import time
import pandas as pd
import altair as alt
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional
import logging
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

# Load environment variables
load_dotenv()

# Add file-based debug logging
logging.basicConfig(
    filename="debug_output.txt",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w"  # Overwrite the file on each run
)

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

# Initialize session state for industry selection
if "industry_selection" not in st.session_state:
    st.session_state.industry_selection = {
        "industry": "",
        "sector": "",
        "subsector": ""
    }

# Example prompts
example_prompts = {
    "Agentic Software": "A B2B enterprise software company providing AI-powered software solutions for Fortune 500 companies",
    "Professional Services": "A global management consulting firm specializing in digital transformation and operational excellence",
    "Financial Services": "An institutional investment management firm focusing on sustainable infrastructure investments",
    "B2B HealthTech Company": "A healthcare technology company providing enterprise solutions for hospital resource management"
}

# Define industry hierarchy data structure
INDUSTRY_HIERARCHY = {
    "Consumer": {
        "Automotive": ["Automotive Manufacturing", "Auto Parts & Suppliers", "Dealerships", "Mobility Services"],
        "Consumer Products": ["Food and Beverage", "Apparel and Footwear", "Personal Care", "Household Products"],
        "Retail": ["Grocery", "Department Stores", "E-commerce", "Specialty Retail"],
        "Transportation, Hospitality & Services": ["Aviation", "Gaming", "Hotels", "Restaurants", "Logistics"]
    },
    "Energy, Resources & Industrials": {
        "Energy & Chemicals": ["Oil & Gas", "Chemicals"],
        "Power, Utilities & Renewables": ["Power Generation", "Utilities", "Renewable Energy"],
        "Industrial Products & Construction": ["Industrial Products Manufacturing", "Construction"],
        "Mining & Metals": ["Mining", "Metals Processing", "Materials"]
    },
    "Financial Services": {
        "Banking & Capital Markets": ["Retail Banking", "Commercial Banking", "Investment Banking", "Capital Markets"],
        "Insurance": ["Life Insurance", "Property & Casualty", "Reinsurance", "InsurTech"],
        "Investment Management & Private Equity": ["Asset Management", "Private Equity", "Hedge Funds", "Wealth Management"],
        "Real Estate": ["Commercial Real Estate", "Residential Real Estate", "REITs"]
    },
    "Government & Public Services": {
        "Central Government": ["Federal Agencies", "Defense", "Public Administration"],
        "Regional and Local Government": ["State Government", "Municipal Services", "Local Administration"],
        "Defense, Security & Justice": ["Defense", "Security", "Justice"],
        "Health & Human Services": ["Public Health", "Social Services", "Welfare"],
        "Infrastructure & Transport": ["Transportation Infrastructure", "Public Transportation"],
        "International Donor Organizations": ["NGOs", "Foundations", "Aid Organizations"]
    },
    "Life Sciences & Health Care": {
        "Health Care": ["Providers", "Payers", "Health Services"],
        "Life Sciences": ["Pharmaceutical", "Biotechnology", "Medical Devices", "Diagnostics"]
    },
    "Technology, Media & Telecommunications": {
        "Technology": ["Software", "Hardware", "Cloud Computing", "Cybersecurity", "Data Analytics"],
        "Media & Entertainment": ["Media", "Entertainment", "Sports", "Gaming"],
        "Telecommunications": ["Wireless Carriers", "Internet Service Providers", "Telecom Infrastructure"]
    },
    "Other": {
        "Other": ["Other"]
    }
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

@st.cache_data(ttl=60)
def get_thread_details(thread_id: str):
    """Get detailed information about a thread"""
    headers = {"X-Api-Key": API_KEY, "Content-Type": "application/json"}
    try:
        response = requests.get(
            f"{API_URL}/threads/{thread_id}",
            headers=headers
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching thread details: {str(e)}")
        return None

@st.cache_data(ttl=60)
def get_thread_runs(thread_id: str):
    """Get all runs for a thread"""
    headers = {"X-Api-Key": API_KEY, "Content-Type": "application/json"}
    try:
        response = requests.get(
            f"{API_URL}/threads/{thread_id}/runs",
            headers=headers
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching thread runs: {str(e)}")
        return []

@st.cache_data(ttl=60)
def get_run_details(thread_id: str, run_id: str):
    """Get detailed information about a specific run"""
    headers = {"X-Api-Key": API_KEY, "Content-Type": "application/json"}
    try:
        response = requests.get(
            f"{API_URL}/threads/{thread_id}/runs/{run_id}",
            headers=headers
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching run details: {str(e)}")
        return None

@st.cache_data(ttl=300)
def fetch_all_threads():
    """Fetch all threads from the LangGraph API"""
    headers = {"X-Api-Key": API_KEY, "Content-Type": "application/json"}
    try:
        # Use the threads/search endpoint to get all threads
        response = requests.post(
            f"{API_URL}/threads/search",
            headers=headers,
            json={
                "limit": 50,  # Fetch up to 50 threads
                "order": "desc",  # Most recent first
                "order_by": "created_at"
            }
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching threads: {str(e)}")
        return []

def build_complete_prompt(base_prompt, industry_info, target_audience, geographic_scope, name_style):
    """Build a complete prompt with additional context"""
    prompt_parts = [base_prompt.strip()]
    
    additional_details = []
    
    # Extract industry information
    industry = industry_info.get("industry", "")
    sector = industry_info.get("sector", "")
    subsector = industry_info.get("subsector", "")
    
    # Add industry details if available and explicitly selected
    if industry and industry != "Other" and industry != "":
        industry_text = f"The company is in the {industry} industry"
        if sector and sector != "Other" and sector != "":
            industry_text += f", specifically in the {sector} sector"
            if subsector and subsector != "Other" and subsector != "":
                industry_text += f", focusing on {subsector}"
        industry_text += "."
        additional_details.append(industry_text)
    
    # Only include target audience if explicitly provided
    if target_audience and target_audience.strip():
        additional_details.append(f"The target audience is {target_audience}.")
        
    # Only include geographic scope if explicitly selected
    if geographic_scope and geographic_scope.strip():
        additional_details.append(f"The brand will operate at a {geographic_scope.lower()} level.")
        
    # Only include name style if explicitly selected
    if name_style and len(name_style) > 0:
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
    generated_names = []
    evaluations = {}
    
    # Track run metrics
    token_counts = {"total": 0, "prompt": 0, "completion": 0}
    run_metadata = {"start_time": time.time(), "steps_completed": 0}
    models_used = {}
    
    # Create containers for metrics and progress
    metrics_container = status_container.container()
    with metrics_container:
        st.subheader("Generation Progress")
        
        # Create multi-column layout for metrics
        metrics_cols = st.columns(4)
        with metrics_cols[0]:
            agent_display = st.empty()
        with metrics_cols[1]:
            steps_display = st.empty()
        with metrics_cols[2]:
            tokens_display = st.empty()
        with metrics_cols[3]:
            time_display = st.empty()
            
        # Progress indicators
        current_step_display = st.empty()
        status_message = st.empty()
    
    # Create separate containers for different types of information
    steps_container = status_container.container()
    debug_container = st.container()  # Move outside status_container to make it more visible
    
    # Initialize debug data list in session state if not already there
    if "debug_data" not in st.session_state:
        st.session_state.debug_data = []
    
    if "raw_debug_data" not in st.session_state:
        st.session_state.raw_debug_data = []
    
    # Track steps and state
    steps_completed = []
    current_agent = ""
    current_step = ""
    current_node = ""
    last_update_time = time.time()
    
    # LangGraph specific tracking
    langgraph_data = {
        "nodes_visited": set(),
        "triggers": set(),
        "run_id": "",
        "thread_id": "",
        "model_calls": []
    }
    
    # Start tracking
    run_metadata["start_time"] = time.time()
    
    # Process each line from the stream
    for i, line in enumerate(stream):
        if not line:
            continue
            
        try:
            # Update progress based on number of events received
            progress_value = min((i + 1) / 20, 0.95)  # More conservative progress estimate
            progress_bar.progress(progress_value)
            
            # Update elapsed time
            elapsed = time.time() - run_metadata["start_time"]
            time_display.metric("Time", f"{elapsed:.1f}s")
            
            # Decode bytes to string
            line_str = line.decode('utf-8') if isinstance(line, bytes) else line
            # Remove "data: " prefix if it exists
            if line_str.startswith("data: "):
                line_str = line_str[6:]
                
            # Parse JSON data
            try:
                data = json.loads(line_str)
                
                # Add raw data to session state for debugging
                st.session_state.raw_debug_data.append(data)
                
                # Extract event type and metadata
                event_type = data.get("type", "")
                metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
                
                # Process different event types
                if event_type == "status":
                    # Handle status message
                    message = data.get("message", "")
                    if message:
                        status_message.info(message)
                    
                    # Extract LangGraph node information
                    if "langgraph_node" in metadata:
                        current_node = metadata["langgraph_node"]
                        langgraph_data["nodes_visited"].add(current_node)
                    
                    # Track triggers
                    if "langgraph_triggers" in metadata and isinstance(metadata["langgraph_triggers"], list):
                        for trigger in metadata["langgraph_triggers"]:
                            langgraph_data["triggers"].add(trigger)
                    
                    # Track run and thread IDs
                    if "run_id" in metadata and not langgraph_data["run_id"]:
                        langgraph_data["run_id"] = metadata["run_id"]
                    
                    if "thread_id" in metadata and not langgraph_data["thread_id"]:
                        langgraph_data["thread_id"] = metadata["thread_id"]
                    
                    # Extract model information
                    if "ls_model_name" in metadata:
                        model_info = {
                            "name": metadata.get("ls_model_name", "Unknown"),
                            "type": metadata.get("ls_model_type", "Unknown"),
                            "provider": metadata.get("ls_provider", "Unknown"),
                            "temperature": metadata.get("ls_temperature", "N/A")
                        }
                        
                        # Store for later display
                        model_key = f"{model_info['provider']}:{model_info['name']}"
                        if model_key not in models_used:
                            models_used[model_key] = model_info
                            langgraph_data["model_calls"].append(model_info)
                    
                    # Update agent information
                    agent_name = data.get("agent", metadata.get("langgraph_node", ""))
                    if agent_name and agent_name != current_agent:
                        current_agent = agent_name
                        agent_display.metric("Current Node", current_agent)
                    
                    # Extract step information
                    step_name = data.get("step", "")
                    step_num = metadata.get("langgraph_step", "")
                    
                    if step_name and (step_name != current_step or step_num):
                        current_step = step_name
                        run_metadata["steps_completed"] += 1
                        
                        # Create step record
                        step_record = {
                            "name": step_name,
                            "time": time.time() - last_update_time,
                            "node": current_node,
                            "step_number": step_num
                        }
                        
                        steps_completed.append(step_record)
                        last_update_time = time.time()
                        
                        # Update step displays
                        steps_display.metric("Steps", len(steps_completed))
                        
                        # Step info with node path if available
                        if current_node:
                            current_step_display.markdown(f"**Current Step:** {current_step} (Node: {current_node})")
                        else:
                            current_step_display.markdown(f"**Current Step:** {current_step}")
                
                # Handle token information
                if "token_count" in metadata:
                    token_counts["total"] = metadata["token_count"]
                    tokens_display.metric("Tokens", token_counts["total"])
                elif "prompt_tokens" in metadata:
                    token_counts["prompt"] = metadata["prompt_tokens"]
                    token_counts["total"] = token_counts["prompt"] + token_counts.get("completion", 0)
                    tokens_display.metric("Tokens", f"{token_counts['total']} ({token_counts['prompt']} prompt)")
                elif "completion_tokens" in metadata:
                    token_counts["completion"] = metadata["completion_tokens"]
                    token_counts["total"] = token_counts.get("prompt", 0) + token_counts["completion"]
                    tokens_display.metric("Tokens", f"{token_counts['total']} ({token_counts['completion']} completion)")
                
                # Handle output/result data
                elif event_type == "output" or event_type == "result":
                    result = data.get("output", data.get("result", {}))
                    if not isinstance(result, dict):
                        # Try to parse if it's a string
                        if isinstance(result, str):
                            try:
                                result = json.loads(result)
                            except:
                                # If it's not JSON, treat as plain text
                                pass
                    
                    # Extract generated names and evaluations
                    if isinstance(result, dict):
                        # Extract names
                        names = result.get("generated_names", [])
                        if names:
                            generated_names = names
                            logging.debug(f"Extracted {len(names)} generated names: {names}")
                        
                            # Extract evaluations
                            evals = result.get("evaluations", {})
                            if evals:
                                evaluations = evals
                                logging.debug(f"Extracted evaluations for {len(evals)} names")
                        
                            # Update results in real-time
                            if generated_names:
                                logging.debug(f"Displaying results with {len(generated_names)} names")
                                display_results(generated_names, evaluations, container)
            except json.JSONDecodeError:
                # Skip lines that aren't valid JSON
                continue
        except Exception as e:
            # Handle any errors
            st.error(f"Error processing stream line: {str(e)}")
    
    # Complete progress
    progress_bar.progress(1.0)
    
    # Final summary of run
    total_time = time.time() - run_metadata["start_time"]
    
    with metrics_container:
        st.success(f"✅ Generation complete in {total_time:.2f} seconds")
        
        # Display final metrics
        final_metrics = st.columns(4)
        with final_metrics[0]:
            st.metric("Total Steps", len(steps_completed))
        with final_metrics[1]:
            st.metric("Total Tokens", token_counts["total"])
        with final_metrics[2]:
            st.metric("Total Time", f"{total_time:.2f}s")
        with final_metrics[3]:
            if token_counts["total"] > 0 and total_time > 0:
                st.metric("Tokens/Second", f"{token_counts['total']/total_time:.1f}")
    
    return generated_names, evaluations

def display_results(generated_names, evaluations, container):
    """Helper function to display generated names and evaluations"""
    logging.debug(f"In display_results with {len(generated_names) if generated_names else 0} names")
    
    with container:
        st.empty().markdown("## Generated Names")
        
        if generated_names:
            logging.debug(f"Displaying names: {generated_names}")
            for name in generated_names:
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.markdown(f"### {name}")
                with col2:
                    if name in st.session_state.favorite_names:
                        if st.button("❤️", key=f"unfav_{name}"):
                            remove_from_favorites(name)
                    else:
                        if st.button("🤍", key=f"fav_{name}"):
                            add_to_favorites(name)
                
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
        else:
            logging.debug("No names to display")

def display_run_details(thread_id, run_id):
    """Display detailed information about a run in a structured way"""
    run_data = get_run_details(thread_id, run_id)
    
    if not run_data:
        st.warning("Could not fetch run details")
        return
    
    # Display basic run info
    st.subheader(f"Run Details: {run_id[:8]}...")
    
    # Split info into columns
    info_cols = st.columns(3)
    with info_cols[0]:
        st.metric("Status", run_data.get("status", "Unknown"))
    with info_cols[1]:
        created_at = run_data.get("created_at", "")
        if created_at:
            st.metric("Created", created_at.split("T")[0])
    with info_cols[2]:
        start_time = run_data.get("start_time")
        end_time = run_data.get("end_time")
        if start_time and end_time:
            try:
                # Convert to datetime and calculate duration
                from datetime import datetime
                start = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                end = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                duration = (end - start).total_seconds()
                st.metric("Duration", f"{duration:.2f}s")
            except:
                st.metric("Duration", "Unknown")
    
    # Display run input/output
    with st.expander("Run Input/Output", expanded=False):
        # Input
        if "input" in run_data:
            st.markdown("##### Input")
            st.json(run_data["input"])
        
        # Output
        if "output" in run_data:
            st.markdown("##### Output")
            st.json(run_data["output"])
    
    # Display any errors
    if "error" in run_data and run_data["error"]:
        st.error(f"Run Error: {run_data['error']}")

def display_thread_history(thread_id):
    """Display comprehensive thread history with visualizations"""
    history_data = get_thread_history(thread_id)
    thread_details = get_thread_details(thread_id)
    thread_runs = get_thread_runs(thread_id)
    
    if not history_data:
        st.warning("No history data available")
        return
        
    # Display thread details
    st.markdown("#### Thread Information")
    
    # Show thread metadata
    if thread_details:
        meta_cols = st.columns(3)
        with meta_cols[0]:
            st.metric("Thread ID", thread_id[:8] + "...")
        with meta_cols[1]:
            created_at = thread_details.get("created_at", "").split("T")[0]
            st.metric("Created", created_at)
        with meta_cols[2]:
            st.metric("Run Count", len(thread_runs) if thread_runs else 0)
    
    # Display runs
    if thread_runs:
        st.markdown("#### Thread Runs")
        for i, run in enumerate(thread_runs):
            run_id = run.get("run_id")
            status = run.get("status", "Unknown")
            
            status_emoji = "🟢" if status == "completed" else "🔴" if status == "failed" else "🟡"
            
            with st.expander(f"{status_emoji} Run {i+1}: {run_id[:8]}... ({status})", expanded=i==0):
                display_run_details(thread_id, run_id)
    
    # Display message history
    if history_data:
        st.markdown("#### Message History")
        
        # Create a more structured view of messages
        for i, message in enumerate(history_data):
            # Determine message role
            role = message.get("role", "Unknown")
            role_emoji = "👤" if role == "user" else "🤖" if role == "assistant" else "🔄"
            
            # Format the message
            with st.container():
                st.markdown(f"##### {role_emoji} {role.title()} Message")
                
                # Content
                if "content" in message and message["content"]:
                    st.markdown(message["content"])
                
                # Handle structured data
                if "data" in message and message["data"]:
                    with st.expander("Message Data", expanded=False):
                        st.json(message["data"])

# Main application layout
st.title("MAE Brand Namer")
st.caption("Multi-Agent Brand Generation & Strategic Analysis")

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
    
    # Create callback functions to maintain hierarchy
    def on_industry_change():
        """Reset sector and subsector when industry changes"""
        st.session_state.industry_selection["sector"] = ""
        st.session_state.industry_selection["subsector"] = ""

    def on_sector_change():
        """Reset subsector when sector changes"""
        st.session_state.industry_selection["subsector"] = ""

    # Advanced parameters in expander
    with st.expander("Additional Parameters", expanded=False):
        # Industry selection with 3-level hierarchy
        st.markdown("#### Industry Classification")
        
        # Industry dropdown (top level)
        industry = st.selectbox(
            "Industry",
            options=[""] + list(INDUSTRY_HIERARCHY.keys()),
            key="industry_dropdown",
            index=0,  # Start with empty selection
            on_change=on_industry_change,
            format_func=lambda x: x if x else "Select Industry (Optional)"
        )
        
        # Store in session state
        st.session_state.industry_selection["industry"] = industry
        
        # Sector dropdown (dependent on industry)
        if industry:
            sector_options = [""] + list(INDUSTRY_HIERARCHY.get(industry, {}).keys())
            sector = st.selectbox(
                "Sector",
                options=sector_options,
                key="sector_dropdown",
                index=0,  # Start with empty selection
                on_change=on_sector_change,
                format_func=lambda x: x if x else "Select Sector (Optional)"
            )
            # Store in session state
            st.session_state.industry_selection["sector"] = sector
            
            # Subsector dropdown (dependent on industry and sector)
            if sector:
                subsector_options = [""] + INDUSTRY_HIERARCHY.get(industry, {}).get(sector, [])
                subsector = st.selectbox(
                    "Subsector",
                    options=subsector_options,
                    key="subsector_dropdown",
                    index=0,  # Start with empty selection
                    format_func=lambda x: x if x else "Select Subsector (Optional)"
                )
                # Store in session state
                st.session_state.industry_selection["subsector"] = subsector
        
        # Create an industry info dictionary to pass to the prompt builder
        industry_info = {
            "industry": st.session_state.industry_selection["industry"],
            "sector": st.session_state.industry_selection["sector"],
            "subsector": st.session_state.industry_selection["subsector"]
        }
        
        st.markdown("#### Additional Brand Context")
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
        
        # Build complete prompt with additional requirements
        complete_prompt = build_complete_prompt(
            user_input,
            industry_info,
            target_audience,
            geographic_scope,
            name_style
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
    
    # Results area - modify the order and structure
    main_content = st.container()
    with main_content:
        results_container = st.container()
    
    # Debug section needs to be created BEFORE generation starts
    debug_header = st.container()
    with debug_header:
        st.markdown("---")
        st.subheader("LangGraph Execution Flow")
        st.caption("This section shows detailed information about each step in the graph execution pipeline.")
    
    # Create a container for Streamlit callback
    st_callback_container = st.container()
    
    # Progress indicators
    progress_bar = st.progress(0)
    status_container = st.container()
    
    # Process generation
    if generate_button:
        if not user_input.strip():
            st.error("Please provide a description of your brand requirements.")
            st.stop()
            
        # Clear debug data from previous runs
        if "debug_data" not in st.session_state:
            st.session_state.debug_data = []
        else:
            st.session_state.debug_data = []
        
        if "raw_debug_data" not in st.session_state:
            st.session_state.raw_debug_data = []
        else:
            st.session_state.raw_debug_data = []
        
        # Display initial status
        status_container.info("Initializing generation process...")
        
        # Build complete prompt with additional requirements
        complete_prompt = build_complete_prompt(
            user_input,
            industry_info,
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
        
        # Clear previous results - but only if we have new data
        with results_container:
            if not st.session_state.generation_complete:
                st.empty()
        
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
            
            # Create Streamlit callback handler for real-time visualization
            with st_callback_container:
                st_callback = StreamlitCallbackHandler(st.container())

            # Process the stream
            generated_names, evaluations = process_stream_data(
                run_response.iter_lines(),
                results_container,
                status_container,
                progress_bar  # Pass the Streamlit callback handler
            )
            
            # Update session state
            current_run["status"] = "completed"
            current_run["generated_names"] = generated_names
            current_run["evaluations"] = evaluations
            st.session_state.history[current_index] = current_run
            st.session_state.generation_complete = True

            # Log the final results
            logging.debug(f"Final generation results: {len(generated_names)} names")
            for name in generated_names:
                logging.debug(f"Generated name: {name}")

            # Ensure results are displayed clearly
            with results_container:
                st.markdown("## Final Results")
                if generated_names:
                    st.success(f"Successfully generated {len(generated_names)} brand names")
                    
                    # Display each name with its evaluation
                    for name in generated_names:
                        st.markdown(f"### {name}")
                        
                        # Add favorite button
                        if name in st.session_state.favorite_names:
                            if st.button("❤️ Remove from Favorites", key=f"final_unfav_{name}"):
                                remove_from_favorites(name)
                        else:
                            if st.button("🤍 Add to Favorites", key=f"final_fav_{name}"):
                                add_to_favorites(name)
                        
                        # Display evaluation if available
                        if name in evaluations:
                            with st.expander("View Analysis", expanded=False):
                                st.write(evaluations[name].get("analysis", "No analysis available"))
                        
                        st.markdown("---")
                else:
                    st.warning("No names were generated. Please check the debug information below.")

            # Display debug data in case it wasn't shown
            debug_container = st.container()
            with debug_container:
                st.write(f"Debug data count: {len(st.session_state.raw_debug_data)}")
                if len(st.session_state.raw_debug_data) > 0:
                    st.json(st.session_state.raw_debug_data[:10])  # Show first 10 entries to avoid overwhelming the UI

            # Force refresh the history display
            st.rerun()

        except requests.RequestException as e:
            st.error(f"Error connecting to the API: {str(e)}")
            current_run["status"] = "failed"
            current_run["error"] = str(e)
            st.session_state.history[current_index] = current_run
            if st.checkbox("Show detailed error"):
                st.code(str(e))

# History tab
with tab2:
    st.subheader("Generation History")
    
    # Add refresh button
    if st.button("Refresh History"):
        st.cache_data.clear()
    
    # Create tabs for local and API history
    history_tabs = st.tabs(["Current Session", "All API Generations"])
    
    # Current session history
    with history_tabs[0]:
        if not st.session_state.history:
            st.info("No generations in current session. Generate some brand names first!")
        else:
            for i, run in enumerate(reversed(st.session_state.history)):
                with st.expander(f"Generation {len(st.session_state.history) - i} - {run['timestamp']}", expanded=i==0):
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
                                else:
                                    if st.button("🤍", key=f"h_fav_{i}_{name}"):
                                        add_to_favorites(name)
                    
                    if run.get("thread_id"):
                        if st.button("Load Full Results", key=f"load_{i}"):
                            thread_data = get_thread_history(run["thread_id"])
                            st.json(thread_data)
    
    # All API history
    with history_tabs[1]:
        # Fetch all threads from API
        with st.spinner("Loading past generations..."):
            all_threads = fetch_all_threads()
        
        if not all_threads:
            st.info("No generation history found in the API")
        else:
            st.success(f"Found {len(all_threads)} past generations")
            
            # First, show a summary table
            thread_data = []
            for thread in all_threads:
                # Extract thread info
                thread_id = thread.get("thread_id", "N/A")
                created_at = thread.get("created_at", "Unknown")
                if isinstance(created_at, str) and "T" in created_at:
                    created_at = created_at.split("T")[0]
                
                # Add to table data
                thread_data.append({
                    "Thread ID": thread_id[:8] + "..." if len(thread_id) > 8 else thread_id,
                    "Created": created_at,
                    "Full Thread ID": thread_id  # For reference
                })
            
            # Display as dataframe
            df = pd.DataFrame(thread_data)
            
            # Add selection functionality
            selected_thread = st.selectbox(
                "Select a thread to view details:",
                options=df["Full Thread ID"].tolist(),
                format_func=lambda x: f"Thread {x[:8]}... - {df[df['Full Thread ID']==x]['Created'].iloc[0]}"
            )
            
            # Show thread details when selected
            if selected_thread:
                st.markdown("### Thread Details")
                
                # Get thread history
                thread_history = get_thread_history(selected_thread)
                
                # Extract generated names if available
                generated_names = []
                for message in thread_history:
                    if message.get("role") == "assistant" and "data" in message:
                        data = message.get("data", {})
                        if isinstance(data, dict) and "output" in data:
                            output = data["output"]
                            if isinstance(output, dict) and "generated_names" in output:
                                generated_names = output["generated_names"]
                                break
                
                # Display generated names if found
                if generated_names:
                    st.markdown("#### Generated Names")
                    name_cols = st.columns(2)
                    for i, name in enumerate(generated_names):
                        with name_cols[i % 2]:
                            col1, col2 = st.columns([4, 1])
                            with col1:
                                st.markdown(f"**{name}**")
                            with col2:
                                if name in st.session_state.favorite_names:
                                    if st.button("❤️", key=f"api_unfav_{name}"):
                                        remove_from_favorites(name)
                                        st.rerun()
                                else:
                                    if st.button("🤍", key=f"api_fav_{name}"):
                                        add_to_favorites(name)
                
                # Display thread details
                with st.expander("Conversation History", expanded=not generated_names):
                    # Show user inputs and assistant responses
                    for i, message in enumerate(thread_history):
                        role = message.get("role", "Unknown")
                        content = message.get("content", "")
                        
                        # Style based on role
                        if role == "user":
                            st.markdown(f"**👤 User:**")
                            st.info(content)
                        elif role == "assistant":
                            st.markdown(f"**🤖 Assistant:**")
                            st.success(content if content else "Generated names")
                
                # Option to view raw data
                if st.button("View Raw Thread Data"):
                    # Thread details
                    thread_details = get_thread_details(selected_thread)
                    st.json(thread_details)
                    
                    # Thread runs
                    thread_runs = get_thread_runs(selected_thread)
                    if thread_runs:
                        st.markdown("#### Thread Runs")
                        st.json(thread_runs)

# Footer
st.markdown("---")
st.caption("MAE Brand Namer | Powered by LangGraph AI") 