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
from langchain.callbacks.streamlit import StreamlitCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler

# Utility function to validate LangSmith traces
def validate_langsmith_trace(trace_id):
    """Check if a LangSmith trace exists"""
    langsmith_url = f"https://smith.langchain.com/api/traces/{trace_id}"
    try:
        # Just a HEAD request to see if it exists, we don't need the full data
        response = requests.head(langsmith_url, timeout=2)
        return response.status_code == 200
    except Exception:
        return False

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
    debug_container = st.container()  # Container for debug information
    
    # Initialize debug data list in session state if not already there
    if "raw_debug_data" not in st.session_state:
        st.session_state.raw_debug_data = []
    else:
        # Clear existing debug data for new run
        st.session_state.raw_debug_data = []
    
    # Also track raw stream data before JSON processing
    if "raw_stream_lines" not in st.session_state:
        st.session_state.raw_stream_lines = []
    else:
        st.session_state.raw_stream_lines = []
    
    # Create a container for raw data display
    raw_json_container = debug_container.container()
    raw_json_display = raw_json_container.empty()
    
    # Set up counters and trackers
    line_count = 0
    langgraph_data = {
        "nodes_visited": set(),
        "node_data": {},
        "triggers": set(),
        "run_id": "",
        "thread_id": ""
    }
    
    # Update display function for raw JSON
    def update_raw_json_display():
        if not st.session_state.raw_debug_data:
            raw_json_display.info("Waiting for data...")
            return
            
        # Show the count of events received
        raw_json_display.markdown(f"### Raw Stream Data ({len(st.session_state.raw_debug_data)} events)")
        
        # Display the last 10 events as pretty JSON
        with raw_json_display.expander("Latest Events", expanded=True):
            for i, event in enumerate(st.session_state.raw_debug_data[-10:]):
                st.markdown(f"**Event {len(st.session_state.raw_debug_data) - 10 + i + 1}:**")
                st.json(event)
    
    # Process stream data
    for line in stream:
        if not line:
            continue
            
        line_count += 1
        line_str = line.decode("utf-8")
        
        # Store the raw line before any processing
        st.session_state.raw_stream_lines.append(line_str)
        
        # Skip empty lines
        if not line_str.strip():
            continue
        
        # Update progress information
        progress_bar.progress((line_count % 100) / 100)
        elapsed_time = time.time() - run_metadata["start_time"]
        time_display.metric("Time", f"{elapsed_time:.1f}s")
            
        # Handle Server-Sent Events (SSE) format
        if line_str.startswith("event:") or line_str.startswith(":"):
            # This is an SSE event marker or comment, not JSON data
            if line_str.startswith(":"):
                # This is a comment/heartbeat
                status_message.info("Server heartbeat")
                continue
                
            # Extract event type for debugging
            event_type = line_str.replace("event:", "").strip()
            status_message.info(f"Event stream: {event_type}")
            continue
            
        # Process JSON data
        data = None
        json_str = None
            
        # Look for data payload in SSE format
        if line_str.startswith("data:"):
            # Extract the JSON data after "data:"
            json_str = line_str[5:].strip()
            
            # Skip empty data
            if not json_str:
                continue
        else:
            # Try to parse as raw JSON (fallback for non-SSE format)
            json_str = line_str
        
        # Try to parse the JSON data
        try:
            data = json.loads(json_str)
            
            # Store raw data for debugging
            st.session_state.raw_debug_data.append(data)
            print(f"DEBUG: Received data: {data.get('type', 'unknown')}")
            
            # Update the raw JSON display
            update_raw_json_display()
        except json.JSONDecodeError as json_err:
            # Log the error and the problematic data
            print(f"Error parsing JSON: {str(json_err)}")
            print(f"Problematic data: '{json_str}'")
            status_message.warning(f"Received non-JSON data (length: {len(json_str)})")
            
            # Store as raw text for debugging
            st.session_state.raw_debug_data.append({"type": "raw_text", "content": json_str})
            continue  # Skip to next line
            
        # If we have valid data, process it
        if data:
            try:
                # Extract event type and metadata
                event_type = data.get("type", "unknown")
                metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
                
                # Handle status message (keep this simple)
                if event_type == "status" and "message" in data:
                    status_message.info(data["message"])
                    
                    # Extract step info if available
                    if "langgraph_step" in metadata:
                        steps_display.metric("Steps", metadata["langgraph_step"])
                    
                    # Extract node name if available
                    if "langgraph_node" in metadata:
                        current_node = metadata["langgraph_node"]
                        current_step_display.info(f"Processing node: {current_node}")
                
                # Check for result data - multiple possible locations
                result = None
                for key in ["data", "output", "result"]:
                    if key in data:
                        result = data[key]
                        break
                
                # If result contains generated names, display them
                if isinstance(result, dict) and "generated_names" in result:
                    names = result["generated_names"]
                    if names:
                        generated_names = names
                        evaluations = result.get("evaluations", {})
                        display_results(generated_names, evaluations, container)
                
                # Also check if names are in the top-level data
                elif isinstance(data, dict) and "generated_names" in data:
                    names = data["generated_names"]
                    if names:
                        generated_names = names
                        evaluations = data.get("evaluations", {})
                        display_results(generated_names, evaluations, container)
                
                # Try to parse if it's a string that might contain the results
                elif isinstance(result, str):
                    try:
                        parsed = json.loads(result)
                        if isinstance(parsed, dict) and "generated_names" in parsed:
                            names = parsed["generated_names"]
                            if names:
                                generated_names = names
                                evaluations = parsed.get("evaluations", {})
                                display_results(generated_names, evaluations, container)
                    except:
                        # Not JSON or doesn't contain what we're looking for
                        pass
            except Exception as e:
                # Log any errors in processing
                print(f"Error processing data: {str(e)}")
                status_message.error(f"Error: {str(e)}")
        else:
            print(f"No valid data after parsing: {json_str}")
    
    # Final update to progress indicators
    progress_bar.progress(100)
    run_metadata["end_time"] = time.time()
    elapsed_time = run_metadata["end_time"] - run_metadata["start_time"]
    time_display.metric("Time", f"{elapsed_time:.1f}s (Completed)")
    current_step_display.success("Generation completed")
    
    # Mark completion in session state
    st.session_state.generation_complete = True
    
    # Display final raw JSON state
    with debug_container:
        st.markdown("---")
        st.subheader("LangGraph Execution Flow")
        st.caption("This section shows the raw data received from the stream.")
        
        # First, show the raw stream output
        with st.expander("Raw Stream Data (before JSON parsing)", expanded=True):
            st.info(f"Received {len(st.session_state.raw_stream_lines)} lines from stream")
            if st.session_state.raw_stream_lines:
                for i, line in enumerate(st.session_state.raw_stream_lines[:20]):  # Limit to first 20 lines
                    st.text(f"Line {i+1}: {line}")
                if len(st.session_state.raw_stream_lines) > 20:
                    st.text(f"... and {len(st.session_state.raw_stream_lines) - 20} more lines")
            else:
                st.warning("No raw stream data captured")
        
        if st.session_state.raw_debug_data:
            # Display event count
            st.info(f"Received {len(st.session_state.raw_debug_data)} events from the stream")
            
            # Create tabs for different views
            debug_tabs = st.tabs(["All Events", "Status Events", "Result Data"])
            
            # All events tab
            with debug_tabs[0]:
                st.markdown("### All Stream Events")
                for i, event in enumerate(st.session_state.raw_debug_data):
                    event_type = event.get("type", "unknown")
                    with st.expander(f"Event {i+1}: {event_type}", expanded=i==0):
                        st.json(event)
            
            # Status events tab
            with debug_tabs[1]:
                st.markdown("### Status Events")
                status_events = [e for e in st.session_state.raw_debug_data if e.get("type") == "status"]
                if status_events:
                    for i, event in enumerate(status_events):
                        message = event.get("message", "No message")
                        metadata = event.get("metadata", {})
                        node = metadata.get("langgraph_node", "Unknown")
                        step = metadata.get("langgraph_step", "?")
                        with st.expander(f"Step {step}: {node} - {message}", expanded=i==0):
                            st.json(event)
                else:
                    st.warning("No status events found")
            
            # Result data tab
            with debug_tabs[2]:
                st.markdown("### Result Data")
                result_events = [
                    e for e in st.session_state.raw_debug_data 
                    if e.get("type") in ["output", "result"] or "generated_names" in str(e)
                ]
                if result_events:
                    for i, event in enumerate(result_events):
                        with st.expander(f"Result {i+1}", expanded=i==0):
                            st.json(event)
                else:
                    st.warning("No result events found")
        else:
            st.warning("No debug data captured from the stream")
    
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
    
    # Create a container for Streamlit callback and place it before the progress indicators
    st_callback_container = st.container()
    # Initialize LangChain callback handler for Streamlit
    st_callback = StreamlitCallbackHandler(st_callback_container, expand_new_thoughts=False, max_thought_containers=10)

    # Progress indicators
    progress_bar = st.progress(0)
    status_container = st.container()
    
    # Show persisted debug data if we have it (from previous runs/tab switches)
    debug_container = st.container()
    with debug_container:
        if "generation_complete" in st.session_state and st.session_state.generation_complete:
            if "langsmith_trace_ids" in st.session_state and st.session_state.langsmith_trace_ids:
                st.subheader("LangSmith Traces")
                for trace_id in st.session_state.langsmith_trace_ids:
                    # Create LangSmith trace URL
                    langsmith_url = f"https://smith.langchain.com/traces/{trace_id}"
                    st.markdown(f"[View detailed trace on LangSmith]({langsmith_url})")
                
                st.info("LangSmith traces provide the most detailed view of your flow's execution. Click the links above to view in the LangSmith UI.")
            
            if "raw_debug_data" in st.session_state and len(st.session_state.raw_debug_data) > 0:
                st.write(f"Debug data available: {len(st.session_state.raw_debug_data)} events")
                
                # Display LangSmith trace IDs if available
                if "langsmith_trace_ids" in st.session_state and st.session_state.langsmith_trace_ids:
                    st.subheader("LangSmith Traces")
                    valid_traces = []
                    
                    for trace_id in st.session_state.langsmith_trace_ids:
                        # Create LangSmith trace URL
                        langsmith_url = f"https://smith.langchain.com/traces/{trace_id}"
                        
                        # Add the trace link
                        with st.spinner(f"Validating trace {trace_id[:8]}..."):
                            is_valid = validate_langsmith_trace(trace_id)
                        
                        if is_valid:
                            st.markdown(f"✅ [View detailed trace on LangSmith]({langsmith_url})")
                            valid_traces.append(trace_id)
                        else:
                            st.markdown(f"❌ Trace {trace_id[:8]}... may not be available")
                    
                    if valid_traces:
                        st.info(f"LangSmith traces provide the most detailed view of your flow's execution. {len(valid_traces)} valid trace(s) found.")
                    else:
                        st.warning("No valid LangSmith traces were found. This might be due to API limitations or LangSmith configuration.")
                else:
                    st.info("No LangSmith traces were captured during execution. This may be due to the LangSmith tracing being disabled in your LangGraph flow.")
                    
                    # Offer a manual lookup option
                    run_id_manual = st.text_input("Enter a run ID manually to check LangSmith:")
                    if run_id_manual and st.button("Check Trace"):
                        with st.spinner("Validating trace ID..."):
                            is_valid = validate_langsmith_trace(run_id_manual)
                        
                        if is_valid:
                            langsmith_url = f"https://smith.langchain.com/traces/{run_id_manual}"
                            st.success(f"✅ Valid trace found! [View on LangSmith]({langsmith_url})")
                        else:
                            st.error("❌ No valid trace found with that ID")
                
                # Continue with the rest of the debug section
                if len(st.session_state.raw_debug_data) > 0:
                    # Extract LangGraph-specific events
                    langgraph_events = [
                        event for event in st.session_state.raw_debug_data 
                        if (event.get("type") == "status" and 
                            "metadata" in event and 
                            "langgraph_node" in event.get("metadata", {}))
                    ]
                    
                    # Extract streaming deltas and unknown events
                    delta_events = [
                        event for event in st.session_state.raw_debug_data
                        if "delta" in event and isinstance(event["delta"], dict)
                    ]
                    
                    unknown_events = [
                        event for event in st.session_state.raw_debug_data
                        if event.get("type", "unknown") == "unknown"
                    ]
                    
                    # Display LangGraph execution events
                    if langgraph_events:
                        st.subheader("LangGraph Execution Path")
                        for i, event in enumerate(langgraph_events):
                            metadata = event.get("metadata", {})
                            node_name = metadata.get("langgraph_node", "Unknown")
                            step = metadata.get("langgraph_step", "?")
                            
                            with st.expander(f"Step {step}: {node_name}", expanded=i==0):
                                # Show additional metadata if available
                                col1, col2 = st.columns(2)
                                with col1:
                                    if "ls_model_name" in metadata:
                                        st.markdown(f"**Model:** {metadata.get('ls_model_name')}")
                                    if "prompt_tokens" in metadata:
                                        st.markdown(f"**Tokens:** {metadata.get('prompt_tokens')}")
                                with col2:
                                    if "ls_provider" in metadata:
                                        st.markdown(f"**Provider:** {metadata.get('ls_provider')}")
                                    if "ls_run_id" in metadata:
                                        run_id = metadata.get('ls_run_id')
                                        langsmith_url = f"https://smith.langchain.com/runs/{run_id}"
                                        st.markdown(f"**Run ID:** [{run_id[:8]}...]({langsmith_url})")
                    
                    # Display streaming completion events
                    if delta_events:
                        st.subheader("Streaming Completion Events")
                        for i, event in enumerate(delta_events[:10]):  # Limit to 10 for performance
                            delta = event.get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                with st.expander(f"Delta {i+1}: {content[:30]}...", expanded=False):
                                    st.text(content)
                    
                    # Display unknown events
                    if unknown_events:
                        st.subheader("Unrecognized Event Types")
                        st.caption("These events don't have a standard type field and may contain important metadata")
                        
                        for i, event in enumerate(unknown_events[:5]):  # Limit to 5 for UI clarity
                            event_keys = list(event.keys())
                            if "run_id" in event:
                                title = f"Run Metadata: {event.get('run_id')[:8]}..."
                            elif "content" in event:
                                title = f"Content Chunk: {event.get('content')[:20]}..."
                            else:
                                title = f"Unknown Event {i+1}: Keys={', '.join(event_keys[:3])}..."
                            
                            with st.expander(title, expanded=i==0):
                                # Attempt to extract useful information
                                if "run_id" in event:
                                    st.markdown(f"**Run ID:** {event.get('run_id')}")
                                    
                                    # Add LangSmith link if it appears to be a trace ID
                                    langsmith_url = f"https://smith.langchain.com/traces/{event.get('run_id')}"
                                    st.markdown(f"[View on LangSmith]({langsmith_url})")
                                
                                # Show a formatted version of the event
                                st.json(event)
                    
                    # Still show raw data for complete visibility
                    with st.expander("View Raw Event Data", expanded=False):
                        st.json(st.session_state.raw_debug_data[:10])

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
            
            # Process the stream
            generated_names, evaluations = process_stream_data(
                run_response.iter_lines(),
                results_container,
                status_container,
                progress_bar
            )
            
            # If we didn't get LangGraph data, try to get it directly from LangSmith
            if not st.session_state.langsmith_trace_ids and thread_id:
                try:
                    # Get run details to extract LangSmith trace IDs
                    thread_runs = get_thread_runs(thread_id)
                    logging.debug(f"Retrieved {len(thread_runs) if thread_runs else 0} runs for thread {thread_id}")
                    
                    if thread_runs:
                        for run in thread_runs:
                            run_id = run.get("run_id")
                            if run_id:
                                # Add run ID to the trace IDs
                                st.session_state.langsmith_trace_ids.add(run_id)
                                logging.debug(f"Added run_id {run_id} from thread runs")
                                
                                # Get the detailed run info
                                run_details = get_run_details(thread_id, run_id)
                                if run_details and "metadata" in run_details:
                                    metadata = run_details.get("metadata", {})
                                    # Look for trace IDs in metadata
                                    if "ls_run_id" in metadata:
                                        st.session_state.langsmith_trace_ids.add(metadata["ls_run_id"])
                                        logging.debug(f"Added ls_run_id {metadata['ls_run_id']} from run metadata")
                                    if "ls_parent_run_id" in metadata:
                                        st.session_state.langsmith_trace_ids.add(metadata["ls_parent_run_id"])
                                        logging.debug(f"Added ls_parent_run_id {metadata['ls_parent_run_id']} from run metadata")
                except Exception as e:
                    logging.error(f"Error fetching additional trace info: {str(e)}")
            
            # Manual debug log if we didn't capture anything
            if len(st.session_state.raw_debug_data) == 0:
                logging.warning("No debug data was captured during processing. Creating synthetic debug data.")
                
                # Create synthetic debug data
                debug_entry = {
                    "type": "status",
                    "message": "Generation completed",
                    "metadata": {
                        "langgraph_node": "brand_generator",
                        "langgraph_step": "1",
                        "run_id": thread_id,
                        "thread_id": thread_id
                    }
                }
                
                # Add to our debug data
                st.session_state.raw_debug_data.append(debug_entry)
                
                # If we have at least one name, add it as result data
                if generated_names:
                    result_entry = {
                        "type": "result",
                        "data": {
                            "generated_names": generated_names,
                            "evaluations": evaluations
                        }
                    }
                    st.session_state.raw_debug_data.append(result_entry)
            
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
            with debug_container:
                st.write(f"Debug data count: {len(st.session_state.raw_debug_data)}")
                
                # Display LangSmith trace IDs if available
                if "langsmith_trace_ids" in st.session_state and st.session_state.langsmith_trace_ids:
                    st.subheader("LangSmith Traces")
                    valid_traces = []
                    
                    for trace_id in st.session_state.langsmith_trace_ids:
                        # Create LangSmith trace URL
                        langsmith_url = f"https://smith.langchain.com/traces/{trace_id}"
                        
                        # Add the trace link
                        with st.spinner(f"Validating trace {trace_id[:8]}..."):
                            is_valid = validate_langsmith_trace(trace_id)
                        
                        if is_valid:
                            st.markdown(f"✅ [View detailed trace on LangSmith]({langsmith_url})")
                            valid_traces.append(trace_id)
                        else:
                            st.markdown(f"❌ Trace {trace_id[:8]}... may not be available")
                    
                    if valid_traces:
                        st.info(f"LangSmith traces provide the most detailed view of your flow's execution. {len(valid_traces)} valid trace(s) found.")
                    else:
                        st.warning("No valid LangSmith traces were found. This might be due to API limitations or LangSmith configuration.")
                else:
                    st.info("No LangSmith traces were captured during execution. This may be due to the LangSmith tracing being disabled in your LangGraph flow.")
                    
                    # Offer a manual lookup option
                    run_id_manual = st.text_input("Enter a run ID manually to check LangSmith:")
                    if run_id_manual and st.button("Check Trace"):
                        with st.spinner("Validating trace ID..."):
                            is_valid = validate_langsmith_trace(run_id_manual)
                        
                        if is_valid:
                            langsmith_url = f"https://smith.langchain.com/traces/{run_id_manual}"
                            st.success(f"✅ Valid trace found! [View on LangSmith]({langsmith_url})")
                        else:
                            st.error("❌ No valid trace found with that ID")

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
        # Also check all runs for completion and update statuses
        for i, run in enumerate(st.session_state.history):
            if run["status"] == "running":
                # Check if there are results
                if run.get("generated_names"):
                    st.session_state.history[i]["status"] = "completed"
    
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
                    
                    # Display generated names, even for runs in "running" state that have results
                    if (run['status'] == "completed" or run.get("generated_names")) and run.get("generated_names"):
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
                    
                    # For runs that are still "running" but have no results, show a spinner
                    elif run['status'] == "running":
                        st.info("Generation in progress... Refresh to check for updates.")
                    
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