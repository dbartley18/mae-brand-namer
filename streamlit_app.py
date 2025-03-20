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
import re

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
if "langsmith_trace_ids" not in st.session_state:
    st.session_state.langsmith_trace_ids = set()

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
    """Get the history of a thread"""
    if not thread_id:
        print("DEBUG: No thread_id provided to get_thread_history")
        return []
        
    headers = {"X-Api-Key": API_KEY, "Content-Type": "application/json"}
    try:
        print(f"DEBUG: Fetching thread history for {thread_id}")
        response = requests.post(
            f"{API_URL}/threads/{thread_id}/history",
            headers=headers,
            json={}  # Send empty JSON payload for POST request
        )
        
        # Check if the response was successful
        if response.status_code == 200:
            history_data = response.json()
            print(f"DEBUG: Successfully fetched thread history. Data type: {type(history_data)}")
            return history_data
        else:
            print(f"DEBUG: Error fetching thread history: HTTP {response.status_code} - {response.text}")
            st.error(f"Error fetching thread history: HTTP {response.status_code}")
            return []
            
    except Exception as e:
        print(f"DEBUG: Exception in get_thread_history: {str(e)}")
        st.error(f"Error fetching thread history: {str(e)}")
        return []

@st.cache_data(ttl=60)
def get_thread_details(thread_id: str):
    """Get detailed information about a thread"""
    if not thread_id:
        print("DEBUG: No thread_id provided to get_thread_details")
        return None
        
    headers = {"X-Api-Key": API_KEY, "Content-Type": "application/json"}
    try:
        print(f"DEBUG: Fetching thread details for {thread_id}")
        response = requests.get(
            f"{API_URL}/threads/{thread_id}",
            headers=headers
        )
        
        # Check if the response was successful
        if response.status_code == 200:
            thread_data = response.json()
            print(f"DEBUG: Successfully fetched thread details. Data keys: {list(thread_data.keys()) if isinstance(thread_data, dict) else 'Not a dict'}")
            return thread_data
        else:
            print(f"DEBUG: Error fetching thread details: HTTP {response.status_code} - {response.text}")
            st.error(f"Error fetching thread details: HTTP {response.status_code}")
            return None
            
    except Exception as e:
        print(f"DEBUG: Exception in get_thread_details: {str(e)}")
        st.error(f"Error fetching thread details: {str(e)}")
        return None

@st.cache_data(ttl=60)
def get_thread_runs(thread_id: str):
    """Get all runs for a thread"""
    if not thread_id:
        print("DEBUG: No thread_id provided to get_thread_runs")
        return None
        
    headers = {"X-Api-Key": API_KEY, "Content-Type": "application/json"}
    try:
        print(f"DEBUG: Fetching thread runs for {thread_id}")
        response = requests.get(
            f"{API_URL}/threads/{thread_id}/runs",
            headers=headers
        )
        
        # Check if the response was successful
        if response.status_code == 200:
            runs_data = response.json()
            print(f"DEBUG: Successfully fetched thread runs. Found {len(runs_data) if isinstance(runs_data, list) else '0'} runs.")
            return runs_data
        else:
            print(f"DEBUG: Error fetching thread runs: HTTP {response.status_code} - {response.text}")
            st.error(f"Error fetching thread runs: HTTP {response.status_code}")
            return None
            
    except Exception as e:
        print(f"DEBUG: Exception in get_thread_runs: {str(e)}")
        st.error(f"Error fetching thread runs: {str(e)}")
        return None

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
                
                # Log more details for unknown data types
                if event_type == "unknown":
                    print(f"DEBUG: Unknown data type received. Keys: {list(data.keys())}")
                    
                    # Try to identify if this is thread data
                    if "thread_id" in data or "id" in data:
                        print(f"DEBUG: This appears to be thread data")
                        # Store it so it can be processed
                        thread_data = data
                        # Don't return unknown, continue processing
                    else:
                        # Continue trying to process it like other data types
                        pass
                
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
            
            # Create 2 columns per row for better space utilization
            for i in range(0, len(generated_names), 2):
                cols = st.columns(2)
                
                # Process names for this row
                for j in range(2):
                    idx = i + j
                    if idx < len(generated_names):
                        name_data = generated_names[idx]
                        
                        # Extract the name string if it's a dictionary object
                        if isinstance(name_data, dict):
                            name = name_data.get("brand_name", "")
                            analysis_data = name_data
                        else:
                            name = str(name_data)
                            analysis_data = None
                        
                        if not name:  # Skip empty names
                            continue
                            
                        with cols[j]:
                            # Display name heading
                            st.markdown(f"### {name}")
                            
                            # Add category as caption if available
                            if isinstance(name_data, dict) and "naming_category" in name_data:
                                st.caption(f"Category: {name_data['naming_category']}")
                            elif isinstance(name_data, dict) and "rank" in name_data:
                                st.caption(f"Rank: {name_data['rank']}/10")
                            
                            # Display metrics if available
                            metrics = {}
                            if analysis_data:
                                metric_keys = [
                                    "pronounceability_score", "memorability_score", 
                                    "market_differentiation", "target_audience_relevance",
                                    "rank", "overall_readability_score"
                                ]
                                
                                for key in metric_keys:
                                    if key in analysis_data and isinstance(analysis_data[key], (int, float)):
                                        display_key = key.replace("_score", "").replace("_", " ").title()
                                        metrics[display_key] = analysis_data[key]
                            
                            if metrics:
                                # Display up to 3 metrics per row
                                metric_cols = st.columns(min(3, len(metrics)))
                                for k, (metric, value) in enumerate(metrics.items()):
                                    with metric_cols[k % 3]:
                                        if isinstance(value, float):
                                            display_value = f"{value:.1f}/10"
                                        else:
                                            display_value = f"{value}/10"
                                        st.metric(metric, display_value)
                            
                            # Display notes if available
                            if isinstance(name_data, dict) and "notes" in name_data and name_data["notes"]:
                                with st.expander("Analysis Notes", expanded=False):
                                    st.info(name_data["notes"])
                            
                            # Display additional data if available in evaluations
                            if name in evaluations:
                                with st.expander("View Detailed Analysis"):
                                    col1, col2 = st.columns([3, 2])
                                    with col1:
                                        st.markdown("#### Analysis")
                                        st.write(evaluations[name].get("analysis", "No analysis available"))
                                    with col2:
                                        chart = create_radar_chart(evaluations[name])
                                        if chart:
                                            st.altair_chart(chart)
                            
                            # Display linguistic analysis if available
                            if isinstance(name_data, dict):
                                linguistic_data = {}
                                for key in ["word_class", "sound_symbolism", "rhythm_and_meter", 
                                           "pronunciation_ease", "euphony_vs_cacophony"]:
                                    if key in name_data and name_data[key]:
                                        linguistic_data[key] = name_data[key]
                                
                                if linguistic_data:
                                    with st.expander("Linguistic Details", expanded=False):
                                        for key, value in linguistic_data.items():
                                            st.markdown(f"**{key.replace('_', ' ').title()}**: {value}")
                            
                            st.markdown("---")
        else:
            logging.debug("No names to display")
            st.info("No brand names generated yet. Run a brand naming workflow to see results.")

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

def render_thread_data(thread_data):
    """
    Renders thread data in a structured format with tabs for different report sections.
    
    Args:
        thread_data: Can be either:
            - A dictionary containing thread details with 'values' field that has report sections and metadata
            - A list of thread messages (from get_thread_history)
    
    Returns:
        None (displays content via Streamlit)
    """
    if not thread_data:
        return st.error("No thread data available")
    
    # Determine the type of thread_data
    if isinstance(thread_data, dict) and 'values' in thread_data:
        # This is thread details from get_thread_details()
        has_report_data = True
        values = thread_data.get('values', {})
        thread_id = thread_data.get('thread_id')
    elif isinstance(thread_data, list):
        # This is thread history from get_thread_history()
        has_report_data = False
        values = {}
        thread_id = None
        if thread_data and isinstance(thread_data[0], dict) and 'thread_id' in thread_data[0]:
            thread_id = thread_data[0]['thread_id']
    else:
        return st.error(f"Unrecognized thread data format: {type(thread_data).__name__}")
    
    # Create tabs for different groups of data
    display_tabs = ["Name Generation", "Analysis", "Market & Competition", "Metadata"]
    if thread_id:
        display_tabs.append("Thread Messages")
    display_tabs.append("Raw JSON")
    
    tabs = st.tabs(display_tabs)
    
    # Name generation tab
    with tabs[0]:
        if has_report_data:
            st.header("Generated Names")
            
            # Using the actual key "generated_names" that exists in the data
            name_generation_data = values.get("generated_names", {})
            
            if name_generation_data:
                # Process name categories if categorized
                if isinstance(name_generation_data, dict):
                    for category, brand_names in name_generation_data.items():
                        st.subheader(f"Category: {category}")
                        
                        # Create a table for names in this category
                        table_data = []
                        for brand in brand_names:
                            if isinstance(brand, dict):
                                name_data = {
                                    "Brand Name": brand.get("brand_name", ""),
                                    "Category": brand.get("naming_category", ""),
                                    "Rationale": brand.get("rationale", ""),
                                    "Brand Promise Alignment": brand.get("brand_promise_alignment", ""),
                                    "Brand Personality Alignment": brand.get("brand_personality_alignment", "")
                                }
                                table_data.append(name_data)
                            elif isinstance(brand, str):
                                # Simple string names
                                table_data.append({"Brand Name": brand})
                        
                        if table_data:
                            df = pd.DataFrame(table_data)
                            st.dataframe(df, use_container_width=True)
                        else:
                            st.info(f"No names found in category: {category}")
                # Handle list of names
                elif isinstance(name_generation_data, list):
                    st.subheader("Generated Names")
                    table_data = []
                    for brand in name_generation_data:
                        if isinstance(brand, dict):
                            name_data = {
                                "Brand Name": brand.get("brand_name", ""),
                                "Category": brand.get("naming_category", ""),
                                "Rationale": brand.get("rationale", ""),
                                "Notes": brand.get("notes", "")
                            }
                            table_data.append(name_data)
                        elif isinstance(brand, str):
                            # Simple string names
                            table_data.append({"Brand Name": brand})
                    
                    if table_data:
                        df = pd.DataFrame(table_data)
                        st.dataframe(df, use_container_width=True)
                
                # Name evaluation data
                evaluation_data = values.get("evaluation_results", {})
                if evaluation_data:
                    st.header("Name Evaluation")
                    
                    # Check if there are shortlisted names
                    shortlisted = values.get("shortlisted_names", [])
                    if shortlisted:
                        st.subheader("Shortlisted Names")
                        shortlist_data = []
                        for name in shortlisted:
                            if isinstance(name, dict):
                                shortlist_data.append({
                                    "Brand Name": name.get("brand_name", ""),
                                    "Overall Score": name.get("overall_score", 0),
                                    "Comments": name.get("evaluation_comments", "")
                                })
                            elif isinstance(name, str):
                                shortlist_data.append({"Brand Name": name})
                        st.dataframe(pd.DataFrame(shortlist_data), use_container_width=True)
                    
                    # Present evaluation results in a structured format
                    if isinstance(evaluation_data, dict):
                        st.subheader("Evaluation Details")
                        for name, evaluation in evaluation_data.items():
                            with st.expander(f"{name}"):
                                if isinstance(evaluation, dict):
                                    st.markdown(f"**Overall Score:** {evaluation.get('overall_score', 'N/A')}")
                                    st.markdown(f"**Comments:** {evaluation.get('evaluation_comments', 'N/A')}")
                                else:
                                    st.write(evaluation)
                    elif isinstance(evaluation_data, list):
                        eval_data = []
                        for item in evaluation_data:
                            if isinstance(item, dict):
                                eval_data.append({
                                    "Brand Name": item.get("brand_name", ""),
                                    "Score": item.get("overall_score", "N/A"),
                                    "Comments": item.get("evaluation_comments", "")
                                })
                        
                        if eval_data:
                            st.dataframe(pd.DataFrame(eval_data), use_container_width=True)
            else:
                st.info("No name generation data available.")
        elif not has_report_data:
            st.info("Name generation data is not available in thread message history. Please view the thread details for full report data.")
    
    # Analysis tab
    with tabs[1]:
        if has_report_data:
            analysis_tabs = st.tabs([
                "Linguistic", "Semantic", "Cultural Sensitivity", 
                "Translation"
            ])
            
            # Linguistic analysis - using actual key "linguistic_analysis_results"
            with analysis_tabs[0]:
                ling_data = values.get("linguistic_analysis_results", {})
                if ling_data:
                    st.header("Linguistic Analysis")
                    
                    # Display each brand's linguistic analysis
                    for brand_name, analysis in ling_data.items() if isinstance(ling_data, dict) else []:
                        with st.expander(f"Linguistic Analysis: {brand_name}"):
                            st.markdown(f"**Notes:** {analysis.get('notes', 'N/A')}")
                            
                            cols = st.columns(2)
                            with cols[0]:
                                st.markdown(f"**Word Class:** {analysis.get('word_class', 'N/A')}")
                                st.markdown(f"**Sound Symbolism:** {analysis.get('sound_symbolism', 'N/A')}")
                                st.markdown(f"**Rhythm and Meter:** {analysis.get('rhythm_and_meter', 'N/A')}")
                                st.markdown(f"**Pronunciation Ease:** {analysis.get('pronunciation_ease', 'N/A')}")
                                st.markdown(f"**Euphony vs Cacophony:** {analysis.get('euphony_vs_cacophony', 'N/A')}")
                                st.markdown(f"**Inflectional Properties:** {analysis.get('inflectional_properties', 'N/A')}")
                                
                            with cols[1]:
                                st.markdown(f"**Neologism Appropriateness:** {analysis.get('neologism_appropriateness', 'N/A')}")
                                st.markdown(f"**Overall Readability Score:** {analysis.get('overall_readability_score', 'N/A')}")
                                st.markdown(f"**Morphological Transparency:** {analysis.get('morphological_transparency', 'N/A')}")
                                st.markdown(f"**Naturalness in Collocations:** {analysis.get('naturalness_in_collocations', 'N/A')}")
                                st.markdown(f"**Ease of Marketing Integration:** {analysis.get('ease_of_marketing_integration', 'N/A')}")
                                st.markdown(f"**Phoneme Frequency Distribution:** {analysis.get('phoneme_frequency_distribution', 'N/A')}")
                                st.markdown(f"**Semantic Distance from Competitors:** {analysis.get('semantic_distance_from_competitors', 'N/A')}")
                    
                    # If ling_data is a list or another format
                    if not isinstance(ling_data, dict):
                        st.json(ling_data)
                else:
                    st.info("No linguistic analysis data available.")
            
            # Semantic analysis - using actual key "semantic_analysis_results"
            with analysis_tabs[1]:
                semantic_data = values.get("semantic_analysis_results", {})
                if semantic_data:
                    st.header("Semantic Analysis")
                    
                    for brand_name, analysis in semantic_data.items() if isinstance(semantic_data, dict) else []:
                        with st.expander(f"Semantic Analysis: {brand_name}"):
                            cols = st.columns(2)
                            with cols[0]:
                                st.markdown(f"**Etymology:** {analysis.get('etymology', 'N/A')}")
                                st.markdown(f"**Sound Symbolism:** {analysis.get('sound_symbolism', 'N/A')}")
                                st.markdown(f"**Brand Personality:** {analysis.get('brand_personality', 'N/A')}")
                                st.markdown(f"**Emotional Valence:** {analysis.get('emotional_valence', 'N/A')}")
                                st.markdown(f"**Denotative Meaning:** {analysis.get('denotative_meaning', 'N/A')}")
                                st.markdown(f"**Figurative Language:** {analysis.get('figurative_language', 'N/A')}")
                            
                            with cols[1]:
                                st.markdown(f"**Phoneme Combinations:** {analysis.get('phoneme_combinations', 'N/A')}")
                                st.markdown(f"**Sensory Associations:** {analysis.get('sensory_associations', 'N/A')}")
                                st.markdown(f"**Word Length/Syllables:** {analysis.get('word_length_syllables', 'N/A')}")
                                st.markdown(f"**Alliteration/Assonance:** {'Yes' if analysis.get('alliteration_assonance', False) else 'No'}")
                                st.markdown(f"**Compounding/Derivation:** {analysis.get('compounding_derivation', 'N/A')}")
                                st.markdown(f"**Semantic Trademark Risk:** {analysis.get('semantic_trademark_risk', 'N/A')}")
                    
                    # If semantic_data is a list or another format
                    if not isinstance(semantic_data, dict):
                        st.json(semantic_data)
                else:
                    st.info("No semantic analysis data available.")
            
            # Cultural sensitivity analysis - using actual key "cultural_analysis_results"
            with analysis_tabs[2]:
                cultural_data = values.get("cultural_analysis_results", {})
                if cultural_data:
                    st.header("Cultural Sensitivity Analysis")
                    
                    for brand_name, analysis in cultural_data.items() if isinstance(cultural_data, dict) else []:
                        with st.expander(f"Cultural Sensitivity Analysis: {brand_name}"):
                            st.markdown(f"**Notes:** {analysis.get('notes', 'N/A')}")
                            st.markdown(f"**Overall Risk Rating:** {analysis.get('overall_risk_rating', 'N/A')}")
                            
                            cols = st.columns(2)
                            with cols[0]:
                                st.markdown(f"**Symbolic Meanings:** {analysis.get('symbolic_meanings', 'N/A')}")
                                st.markdown(f"**Historical Meaning:** {analysis.get('historical_meaning', 'N/A')}")
                                st.markdown(f"**Regional Variations:** {analysis.get('regional_variations', 'N/A')}")
                                st.markdown(f"**Cultural Connotations:** {analysis.get('cultural_connotations', 'N/A')}")
                            
                            with cols[1]:
                                st.markdown(f"**Current Event Relevance:** {analysis.get('current_event_relevance', 'N/A')}")
                                st.markdown(f"**Religious Sensitivities:** {analysis.get('religious_sensitivities', 'N/A')}")
                                st.markdown(f"**Social/Political Taboos:** {analysis.get('social_political_taboos', 'N/A')}")
                                st.markdown(f"**Age-related Connotations:** {analysis.get('age_related_connotations', 'N/A')}")
                                st.markdown(f"**Alignment with Cultural Values:** {analysis.get('alignment_with_cultural_values', 'N/A')}")
                    
                    # If cultural_data is a list or another format
                    if not isinstance(cultural_data, dict):
                        st.json(cultural_data)
                else:
                    st.info("No cultural sensitivity analysis data available.")
            
            # Translation analysis - using actual key "translation_analysis_results"
            with analysis_tabs[3]:
                translation_data = values.get("translation_analysis_results", {})
                if translation_data:
                    st.header("Translation Analysis")
                    
                    if isinstance(translation_data, dict):
                        for brand_name, languages in translation_data.items():
                            st.subheader(f"Translation Analysis for {brand_name}")
                            
                            if isinstance(languages, dict):
                                for language, analysis in languages.items():
                                    with st.expander(f"Language: {language}"):
                                        st.markdown(f"**Target Language:** {analysis.get('target_language', 'N/A')}")
                                        st.markdown(f"**Direct Translation:** {analysis.get('direct_translation', 'N/A')}")
                                        st.markdown(f"**Notes:** {analysis.get('notes', 'N/A')}")
                                        
                                        cols = st.columns(2)
                                        with cols[0]:
                                            st.markdown(f"**Semantic Shift:** {analysis.get('semantic_shift', 'N/A')}")
                                            st.markdown(f"**Adaptation Needed:** {'Yes' if analysis.get('adaptation_needed', False) else 'No'}")
                                            st.markdown(f"**Phonetic Retention:** {analysis.get('phonetic_retention', 'N/A')}")
                                            st.markdown(f"**Proposed Adaptation:** {analysis.get('proposed_adaptation', 'N/A')}")
                                        
                                        with cols[1]:
                                            st.markdown(f"**Cultural Acceptability:** {analysis.get('cultural_acceptability', 'N/A')}")
                                            st.markdown(f"**Brand Essence Preserved:** {analysis.get('brand_essence_preserved', 'N/A')}")
                                            st.markdown(f"**Pronunciation Difficulty:** {analysis.get('pronunciation_difficulty', 'N/A')}")
                                            st.markdown(f"**Global Consistency vs Localization:** {analysis.get('global_consistency_vs_localization', 'N/A')}")
                    else:
                        # Display raw data for non-dict format
                        st.json(translation_data)
                else:
                    st.info("No translation analysis data available.")
        else:
            st.info("Analysis data is not available in thread message history.")
    
    # Market & Competition tab
    with tabs[2]:
        if has_report_data:
            market_tabs = st.tabs([
                "Domain Analysis", "SEO & Discoverability", 
                "Market Research", "Competitor Analysis", "Survey Simulation"
            ])
            
            # Domain analysis - using actual key "domain_analysis_results"
            with market_tabs[0]:
                domain_data = values.get("domain_analysis_results", {})
                if domain_data:
                    st.header("Domain Analysis")
                    
                    if isinstance(domain_data, dict):
                        for brand_name, analysis in domain_data.items():
                            with st.expander(f"Domain Analysis: {brand_name}"):
                                st.markdown(f"**Notes:** {analysis.get('notes', 'N/A')}")
                                st.markdown(f"**Acquisition Cost:** {analysis.get('acquisition_cost', 'N/A')}")
                                
                                cols = st.columns(2)
                                with cols[0]:
                                    st.markdown(f"**Domain Exact Match:** {'Yes' if analysis.get('domain_exact_match', False) else 'No'}")
                                    st.markdown(f"**Hyphens/Numbers Present:** {'Yes' if analysis.get('hyphens_numbers_present', False) else 'No'}")
                                    st.markdown(f"**Brand Name Clarity in URL:** {analysis.get('brand_name_clarity_in_url', 'N/A')}")
                                    st.markdown(f"**Domain Length/Readability:** {analysis.get('domain_length_readability', 'N/A')}")
                                
                                with cols[1]:
                                    st.markdown(f"**Scalability/Future-proofing:** {analysis.get('scalability_future_proofing', 'N/A')}")
                                    st.markdown(f"**Misspellings/Variations Available:** {'Yes' if analysis.get('misspellings_variations_available', False) else 'No'}")
                                
                                # Alternative TLDs
                                alt_tlds = analysis.get('alternative_tlds', [])
                                if alt_tlds:
                                    st.markdown("**Alternative TLDs:**")
                                    st.markdown(", ".join(alt_tlds))
                                
                                # Social media availability
                                social_handles = analysis.get('social_media_availability', [])
                                if social_handles:
                                    st.markdown("**Available Social Media Handles:**")
                                    st.markdown(", ".join(social_handles))
                    else:
                        # Display raw data for non-dict format
                        st.json(domain_data)
                else:
                    st.info("No domain analysis data available.")
            
            # SEO analysis - using actual key "seo_analysis_results"
            with market_tabs[1]:
                seo_data = values.get("seo_analysis_results", [])
                if seo_data:
                    st.header("SEO & Online Discoverability")
                    
                    if isinstance(seo_data, list):
                        for analysis in seo_data:
                            brand_name = analysis.get("brand_name", "Unnamed")
                            with st.expander(f"SEO Analysis: {brand_name}"):
                                st.markdown(f"**Search Volume:** {analysis.get('search_volume', 'N/A')}")
                                st.markdown(f"**SEO Viability Score:** {analysis.get('seo_viability_score', 'N/A')}")
                                
                                cols = st.columns(2)
                                with cols[0]:
                                    st.markdown(f"**Keyword Alignment:** {analysis.get('keyword_alignment', 'N/A')}")
                                    st.markdown(f"**Keyword Competition:** {analysis.get('keyword_competition', 'N/A')}")
                                    st.markdown(f"**Negative Search Results:** {'Yes' if analysis.get('negative_search_results', False) else 'No'}")
                                    st.markdown(f"**Unusual Spelling Impact:** {'Yes' if analysis.get('unusual_spelling_impact', False) else 'No'}")
                                    st.markdown(f"**Branded Keyword Potential:** {analysis.get('branded_keyword_potential', 'N/A')}")
                                
                                with cols[1]:
                                    st.markdown(f"**Name Length/Searchability:** {analysis.get('name_length_searchability', 'N/A')}")
                                    st.markdown(f"**Social Media Availability:** {'Yes' if analysis.get('social_media_availability', False) else 'No'}")
                                    st.markdown(f"**Competitor Domain Strength:** {analysis.get('competitor_domain_strength', 'N/A')}")
                                    st.markdown(f"**Exact Match Search Results:** {analysis.get('exact_match_search_results', 'N/A')}")
                                    st.markdown(f"**Social Media Discoverability:** {analysis.get('social_media_discoverability', 'N/A')}")
                                    st.markdown(f"**Non-branded Keyword Potential:** {analysis.get('non_branded_keyword_potential', 'N/A')}")
                                
                                # SEO Recommendations
                                recs = analysis.get('seo_recommendations', {}).get('recommendations', [])
                                if recs:
                                    st.markdown("**SEO Recommendations:**")
                                    for rec in recs:
                                        st.markdown(f"- {rec}")
                    elif isinstance(seo_data, dict):
                        for brand_name, analysis in seo_data.items():
                            with st.expander(f"SEO Analysis: {brand_name}"):
                                for key, value in analysis.items():
                                    st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")
                    else:
                        # Display raw data for other formats
                        st.json(seo_data)
                else:
                    st.info("No SEO analysis data available.")
            
            # Market research - using actual key "market_research_results"
            with market_tabs[2]:
                market_data = values.get("market_research_results", {})
                if market_data:
                    st.header("Market Research")
                    
                    if isinstance(market_data, dict):
                        for brand_name, analysis in market_data.items():
                            with st.expander(f"Market Research: {brand_name}"):
                                cols = st.columns(2)
                                with cols[0]:
                                    st.markdown(f"**Industry Name:** {analysis.get('industry_name', 'N/A')}")
                                    st.markdown(f"**Market Size:** {analysis.get('market_size', 'N/A')}")
                                    st.markdown(f"**Market Growth Rate:** {analysis.get('market_growth_rate', 'N/A')}")
                                    st.markdown(f"**Market Viability:** {analysis.get('market_viability', 'N/A')}")
                                    st.markdown(f"**Market Opportunity:** {analysis.get('market_opportunity', 'N/A')}")
                                    st.markdown(f"**Target Audience Fit:** {analysis.get('target_audience_fit', 'N/A')}")
                                
                                with cols[1]:
                                    st.markdown(f"**Emerging Trends:** {analysis.get('emerging_trends', 'N/A')}")
                                    st.markdown(f"**Potential Risks:** {analysis.get('potential_risks', 'N/A')}")
                                    st.markdown(f"**Competitive Analysis:** {analysis.get('competitive_analysis', 'N/A')}")
                                    st.markdown(f"**Market Entry Barriers:** {analysis.get('market_entry_barriers', 'N/A')}")
                                    st.markdown(f"**Recommendations:** {analysis.get('recommendations', 'N/A')}")
                                
                                # Key competitors
                                competitors = analysis.get('key_competitors', [])
                                if competitors:
                                    st.markdown("**Key Competitors:**")
                                    for comp in competitors:
                                        st.markdown(f"- {comp}")
                                
                                # Customer pain points
                                pain_points = analysis.get('customer_pain_points', [])
                                if pain_points:
                                    st.markdown("**Customer Pain Points:**")
                                    for point in pain_points:
                                        st.markdown(f"- {point}")
                    else:
                        # Display raw data for non-dict format
                        st.json(market_data)
                else:
                    st.info("No market research data available.")
            
            # Competitor analysis - using actual key "competitor_analysis_results"
            with market_tabs[3]:
                competitor_data = values.get("competitor_analysis_results", [])
                if competitor_data:
                    st.header("Competitor Analysis")
                    
                    if isinstance(competitor_data, list):
                        for brand_comp in competitor_data:
                            brand_name = brand_comp.get("brand_name", "Unnamed")
                            st.subheader(f"Competitors for {brand_name}")
                            
                            competitors = brand_comp.get("competitors", [])
                            for comp in competitors:
                                with st.expander(f"Competitor: {comp.get('competitor_name', 'Unnamed')}"):
                                    st.markdown(f"**Risk of Confusion:** {comp.get('risk_of_confusion', 'N/A')}/10")
                                    st.markdown(f"**Trademark Conflict Risk:** {comp.get('trademark_conflict_risk', 'N/A')}")
                                    
                                    cols = st.columns(2)
                                    with cols[0]:
                                        st.markdown(f"**Strengths:** {comp.get('competitor_strengths', 'N/A')}")
                                        st.markdown(f"**Weaknesses:** {comp.get('competitor_weaknesses', 'N/A')}")
                                    
                                    with cols[1]:
                                        st.markdown(f"**Positioning:** {comp.get('competitor_positioning', 'N/A')}")
                                        st.markdown(f"**Target Audience Perception:** {comp.get('target_audience_perception', 'N/A')}")
                                        st.markdown(f"**Differentiation Opportunity:** {comp.get('competitor_differentiation_opportunity', 'N/A')}")
                    elif isinstance(competitor_data, dict):
                        for brand_name, competitors in competitor_data.items():
                            st.subheader(f"Competitors for {brand_name}")
                            for comp_name, comp_details in competitors.items():
                                with st.expander(f"Competitor: {comp_name}"):
                                    for key, value in comp_details.items():
                                        st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")
                    else:
                        # Display raw data for other formats
                        st.json(competitor_data)
                else:
                    st.info("No competitor analysis data available.")
            
            # Survey simulation - using actual key "survey_simulation_results"
            with market_tabs[4]:
                survey_data = values.get("survey_simulation_results", [])
                if survey_data:
                    st.header("Survey Simulation")
                    
                    if isinstance(survey_data, list):
                        for survey in survey_data:
                            brand_name = survey.get("brand_name", "Unnamed")
                            with st.expander(f"Survey Results: {brand_name}"):
                                company = survey.get("company_name", "Unknown Company")
                                st.markdown(f"**Company:** {company}")
                                
                                cols = st.columns(2)
                                with cols[0]:
                                    st.markdown(f"**Emotional Association:** {survey.get('emotional_association', 'N/A')}")
                                    st.markdown(f"**Personality Fit Score:** {survey.get('personality_fit_score', 'N/A')}/10")
                                    st.markdown(f"**Competitor Benchmarking Score:** {survey.get('competitor_benchmarking_score', 'N/A')}/10")
                                    st.markdown(f"**Brand Promise Perception:** {survey.get('brand_promise_perception_score', 'N/A')}/10")
                                
                                with cols[1]:
                                    st.markdown(f"**Market Adoption Score:** {survey.get('simulated_market_adoption_score', 'N/A')}/10")
                                    st.markdown(f"**Competitive Differentiation:** {survey.get('competitive_differentiation_score', 'N/A')}/10")
                                    st.markdown(f"**Industry:** {survey.get('industry', 'N/A')}")
                                    st.markdown(f"**Company Size:** {survey.get('company_size_employees', 'N/A')}")
                                
                                st.markdown("**Qualitative Feedback Summary:**")
                                st.write(survey.get('qualitative_feedback_summary', 'No feedback summary available.'))
                                
                                st.markdown("**Final Recommendation:**")
                                st.write(survey.get('final_survey_recommendation', 'No recommendation available.'))
                                
                                # Raw qualitative feedback
                                raw_feedback = survey.get('raw_qualitative_feedback', {})
                                if raw_feedback:
                                    with st.expander("Raw Qualitative Feedback"):
                                        st.json(raw_feedback)
                    elif isinstance(survey_data, dict):
                        for brand_name, survey_results in survey_data.items():
                            with st.expander(f"Survey Results: {brand_name}"):
                                for key, value in survey_results.items():
                                    if isinstance(value, (dict, list)):
                                        with st.expander(f"**{key.replace('_', ' ').title()}**"):
                                            st.json(value)
                                    else:
                                        st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")
                    else:
                        # Display raw data for other formats
                        st.json(survey_data)
                else:
                    st.info("No survey simulation data available.")
        else:
            st.info("Market and competition data is not available in thread message history.")
    
    # Metadata tab
    with tabs[3]:
        if has_report_data:
            # Brand context
            brand_context_data = {}
            
            # Extract brand context fields from values
            brand_fields = [
                "brand_values", "brand_mission", "brand_promise", "brand_purpose",
                "customer_needs", "industry_focus", "industry_trends", "target_audience",
                "brand_personality", "market_positioning", "brand_tone_of_voice",
                "brand_identity_brief", "competitive_landscape"
            ]
            
            for field in brand_fields:
                if field in values:
                    brand_context_data[field] = values[field]
            
            if brand_context_data:
                st.header("Brand Context")
                cols = st.columns(2)
                with cols[0]:
                    if "brand_mission" in brand_context_data:
                        st.markdown(f"**Brand Mission:** {brand_context_data.get('brand_mission', 'N/A')}")
                    if "brand_promise" in brand_context_data:
                        st.markdown(f"**Brand Promise:** {brand_context_data.get('brand_promise', 'N/A')}")
                    if "brand_purpose" in brand_context_data:
                        st.markdown(f"**Brand Purpose:** {brand_context_data.get('brand_purpose', 'N/A')}")
                    if "industry_focus" in brand_context_data:
                        st.markdown(f"**Industry Focus:** {brand_context_data.get('industry_focus', 'N/A')}")
                    if "target_audience" in brand_context_data:
                        st.markdown(f"**Target Audience:** {brand_context_data.get('target_audience', 'N/A')}")
                    if "market_positioning" in brand_context_data:
                        st.markdown(f"**Market Positioning:** {brand_context_data.get('market_positioning', 'N/A')}")
                
                with cols[1]:
                    if "brand_tone_of_voice" in brand_context_data:
                        st.markdown(f"**Brand Tone of Voice:** {brand_context_data.get('brand_tone_of_voice', 'N/A')}")
                    if "brand_identity_brief" in brand_context_data:
                        st.markdown(f"**Brand Identity Brief:** {brand_context_data.get('brand_identity_brief', 'N/A')}")
                    if "competitive_landscape" in brand_context_data:
                        st.markdown(f"**Competitive Landscape:** {brand_context_data.get('competitive_landscape', 'N/A')}")
                
                # Brand values
                values_list = brand_context_data.get('brand_values', [])
                if values_list:
                    st.markdown("**Brand Values:**")
                    if isinstance(values_list, list):
                        st.write(", ".join(values_list))
                    else:
                        st.write(values_list)
                
                # Brand personality
                personality = brand_context_data.get('brand_personality', [])
                if personality:
                    st.markdown("**Brand Personality:**")
                    if isinstance(personality, list):
                        st.write(", ".join(personality))
                    else:
                        st.write(personality)
                
                # Industry trends
                trends = brand_context_data.get('industry_trends', [])
                if trends:
                    st.markdown("**Industry Trends:**")
                    if isinstance(trends, list):
                        for trend in trends:
                            st.markdown(f"- {trend}")
                    else:
                        st.write(trends)
                
                # Customer needs
                needs = brand_context_data.get('customer_needs', [])
                if needs:
                    st.markdown("**Customer Needs:**")
                    if isinstance(needs, list):
                        for need in needs:
                            st.markdown(f"- {need}")
                    else:
                        st.write(needs)
            else:
                st.info("No brand context data available.")
            
            # Thread metadata
            st.header("Thread Metadata")
            metadata_cols = st.columns(2)
            with metadata_cols[0]:
                st.markdown(f"**Thread ID:** {thread_id or 'N/A'}")
                st.markdown(f"**Created At:** {thread_data.get('created_at', 'N/A')}")
                st.markdown(f"**Status:** {thread_data.get('status', 'N/A')}")
                
                # User prompt if available
                if 'user_prompt' in values:
                    st.markdown("**User Prompt:**")
                    st.markdown(values['user_prompt'])
            
            with metadata_cols[1]:
                st.markdown(f"**Run ID:** {thread_data.get('run_id', 'N/A')}")
                st.markdown(f"**Updated At:** {thread_data.get('updated_at', 'N/A')}")
                st.markdown(f"**Last Updated:** {values.get('last_updated', 'N/A')}")
                
                # Timestamp
                if 'timestamp' in values:
                    st.markdown(f"**Timestamp:** {values['timestamp']}")
                
                # Configuration
                config = thread_data.get('config', {})
                if config:
                    with st.expander("Configuration"):
                        st.json(config)
                
                # Task statuses
                task_statuses = values.get('task_statuses', {})
                if task_statuses:
                    with st.expander("Task Statuses"):
                        st.json(task_statuses)
        else:
            st.info("Metadata is not available in thread message history.")
    
    # Thread messages tab (if applicable)
    if thread_id and "Thread Messages" in display_tabs:
        with tabs[display_tabs.index("Thread Messages")]:
            st.header("Thread Messages")
            
            # Try to get messages from thread_data directly or fetch them
            messages = values.get("messages", [])
            if not messages and isinstance(thread_data, list):
                messages = thread_data
            
            if messages:
                for msg in messages:
                    role = msg.get("role", "Unknown")
                    role_emoji = "👤" if role == "user" else "🤖" if role == "assistant" else "🔄"
                    
                    with st.container():
                        st.markdown(f"##### {role_emoji} {role.title()}")
                        
                        if "content" in msg and msg["content"]:
                            st.markdown(msg.get("content", ""))
                            
                        if "data" in msg and msg["data"]:
                            with st.expander("Message Data"):
                                st.json(msg.get("data", {}))
            else:
                st.info(f"No messages found for thread {thread_id}. You may need to fetch them separately.")
    
    # Raw JSON tab (always last)
    with tabs[-1]:
        st.header("Raw Thread Data")
        with st.expander("Expand to view raw JSON data", expanded=False):
            st.json(thread_data)

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
                    
                    # Check for report URL in debug data
                    report_url = None
                    for event in st.session_state.raw_debug_data:
                        # Look for events with report_url
                        if isinstance(event, dict):
                            # Check in different possible locations
                            if "report_url" in event:
                                report_url = event["report_url"]
                                break
                            elif "data" in event and isinstance(event["data"], dict) and "report_url" in event["data"]:
                                report_url = event["data"]["report_url"]
                                break
                            elif "output" in event and isinstance(event["output"], dict) and "report_url" in event["output"]:
                                report_url = event["output"]["report_url"]
                                break
                            elif "result" in event and isinstance(event["result"], dict) and "report_url" in event["result"]:
                                report_url = event["result"]["report_url"]
                                break
                    
                    # Display report URL if found
                    if report_url:
                        st.info("📄 Report generated!")
                        st.markdown(f"[Download the full brand analysis report]({report_url})")
                    
                    # Display each name with its evaluation
                    for name_data in generated_names:
                        # Extract the name string if it's a dictionary object
                        if isinstance(name_data, dict):
                            name = name_data.get("brand_name", "")
                        else:
                            name = str(name_data)
                            
                        if not name:  # Skip empty names
                            continue
                            
                        # Use more appropriate heading level
                        st.markdown(f"### {name}")
                        
                        # Add category as caption if available
                        if isinstance(name_data, dict) and "naming_category" in name_data:
                            st.caption(f"Category: {name_data['naming_category']}")
                        
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
    if st.button("🔄 Refresh History"):
        # Clear the cache to force fresh data fetch
        st.cache_data.clear()
        st.toast("Refreshing data...", icon="🔄")
        
        # Refresh the page to ensure all data is updated
        st.rerun()
        
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
                            render_thread_data(thread_data)
    
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
                
                # Render thread data
                render_thread_data(thread_history)

                # Option to view raw data
                if st.button("View Raw Thread Data"):
                    # Thread details
                    thread_details = get_thread_details(selected_thread)
                    render_thread_data(thread_details)
                    
                    # Thread runs
                    thread_runs = get_thread_runs(selected_thread)
                    if thread_runs:
                        st.markdown("#### Thread Runs")
                        render_thread_data(thread_runs)

# Footer
st.markdown("---")
st.caption("MAE Brand Namer | Powered by LangGraph AI") 

def find_report_url_in_data(data, max_depth=10, current_depth=0):
    """
    Recursively search through a data structure to find URLs that resemble report URLs.
    
    Args:
        data: The data structure to search (can be dict, list, or scalar)
        max_depth: Maximum recursion depth to prevent stack overflow
        current_depth: Current recursion depth (used internally)
        
    Returns:
        The first valid report URL found, or None if no URLs are found
    """
    # Prevent excessive recursion
    if current_depth > max_depth:
        return None
        
    # Base case: if data is None or a primitive type (except string)
    if data is None or (not isinstance(data, (dict, list, str))):
        return None
        
    # Check if data is a string that looks like a URL
    if isinstance(data, str):
        # Look for strings that resemble report URLs
        lower_data = data.lower()
        if ('http' in lower_data and 
            ('report' in lower_data or 
             'pdf' in lower_data or 
             'doc' in lower_data or 
             'analysis' in lower_data or 
             'result' in lower_data)):
            return data
        return None
        
    # Recursively check dictionaries
    if isinstance(data, dict):
        # Check keys that are likely to contain URLs first
        url_likely_keys = ['report_url', 'pdf_url', 'report_link', 'document_url', 
                          'url', 'link', 'href', 'download_url', 'analysis_url',
                          'report_download', 'results_url']
        
        # First check keys that are likely to contain URLs
        for key in url_likely_keys:
            if key in data:
                result = find_report_url_in_data(data[key], max_depth, current_depth + 1)
                if result:
                    return result
        
        # Then check all other keys
        for key, value in data.items():
            # Skip keys that are unlikely to contain URLs to improve performance
            if key in ['created_at', 'updated_at', 'timestamp', 'id', 'thread_id', 
                      'run_id', 'rank', 'score', 'count', 'index']:
                continue
                
            result = find_report_url_in_data(value, max_depth, current_depth + 1)
            if result:
                return result
                
    # Recursively check lists
    elif isinstance(data, list):
        for item in data:
            result = find_report_url_in_data(item, max_depth, current_depth + 1)
            if result:
                return result
                
    return None

def find_value_in_data(data, possible_keys, max_depth=10, current_depth=0):
    """
    Recursively search through a data structure to find values associated with any of the specified keys.
    Optimized for the thread data structure returned by LangSmith API and aligned with
    the BrandNameGenerationState Pydantic model from state.py.
    
    Args:
        data: The data structure to search (can be dict, list, or scalar)
        possible_keys: A list of possible key names to look for
        max_depth: Maximum recursion depth to prevent stack overflow
        current_depth: Current recursion depth (used internally)
        
    Returns:
        The first value found associated with any of the keys, or None if not found
    """
    # Prevent excessive recursion
    if current_depth > max_depth:
        return None
        
    # If data is None, return None
    if data is None:
        return None
    
    # Special case: Check top-level thread structure first
    if current_depth == 0 and isinstance(data, dict):
        # Direct check at thread root level
        for key in possible_keys:
            if key in data:
                return data[key]
                
        # Check in thread metadata
        if "metadata" in data and isinstance(data["metadata"], dict):
            for key in possible_keys:
                if key in data["metadata"]:
                    return data["metadata"][key]
                    
        # Check in thread values section
        if "values" in data and isinstance(data["values"], dict):
            for key in possible_keys:
                if key in data["values"]:
                    return data["values"][key]
    
    # Define mappings from possible keys to actual JSON keys based on BrandNameGenerationState from state.py
    key_mappings = {
        # Core fields from BrandNameGenerationState
        "run_id": ["run_id"],
        "user_prompt": ["user_prompt", "prompt"],
        "errors": ["errors"],
        "start_time": ["start_time", "timestamp", "created_at"],
        "status": ["status"],
        "messages": ["messages"],
        
        # Brand context fields from BrandNameGenerationState
        "brand_identity_brief": ["brand_identity_brief"],
        "brand_promise": ["brand_promise"],
        "brand_values": ["brand_values"],
        "brand_personality": ["brand_personality"],
        "brand_tone_of_voice": ["brand_tone_of_voice"],
        "brand_purpose": ["brand_purpose"],
        "brand_mission": ["brand_mission"],
        "target_audience": ["target_audience"],
        "customer_needs": ["customer_needs"],
        "market_positioning": ["market_positioning"],
        "competitive_landscape": ["competitive_landscape"],
        "industry_focus": ["industry_focus"],
        "industry_trends": ["industry_trends"],
        
        # Brand name generation fields from BrandNameGenerationState
        "generated_names": ["generated_names"],
        "brand_name": ["brand_name"],
        "naming_category": ["naming_category"],
        "brand_personality_alignment": ["brand_personality_alignment"],
        "brand_promise_alignment": ["brand_promise_alignment"],
        "target_audience_relevance": ["target_audience_relevance"],
        "market_differentiation": ["market_differentiation"],
        "visual_branding_potential": ["visual_branding_potential"],
        "memorability_score": ["memorability_score"],
        "pronounceability_score": ["pronounceability_score"],
        
        # Details fields for brand name generation
        "target_audience_relevance_details": ["target_audience_relevance_details"],
        "market_differentiation_details": ["market_differentiation_details"],
        "visual_branding_potential_details": ["visual_branding_potential_details"],
        "memorability_score_details": ["memorability_score_details"],
        "pronounceability_score_details": ["pronounceability_score_details"],
        
        # Name generation fields
        "name_generation_methodology": ["name_generation_methodology"],
        "timestamp": ["timestamp"],
        "rank": ["rank"],
        
        # Lists for multiple brand names
        "naming_categories": ["naming_categories"],
        "brand_personality_alignments": ["brand_personality_alignments"],
        "brand_promise_alignments": ["brand_promise_alignments"],
        "target_audience_relevance_list": ["target_audience_relevance_list"],
        "market_differentiation_list": ["market_differentiation_list"],
        "memorability_scores": ["memorability_scores"],
        "pronounceability_scores": ["pronounceability_scores"],
        "visual_branding_potential_list": ["visual_branding_potential_list"],
        "name_rankings": ["name_rankings"],
        
        # Evaluation fields
        "strategic_alignment_score": ["strategic_alignment_score"],
        "distinctiveness_score": ["distinctiveness_score"],
        "competitive_advantage": ["competitive_advantage"],
        "brand_fit_score": ["brand_fit_score"],
        "positioning_strength": ["positioning_strength"],
        "meaningfulness_score": ["meaningfulness_score"],
        "phonetic_harmony": ["phonetic_harmony"],
        "storytelling_potential": ["storytelling_potential"],
        "domain_viability_score": ["domain_viability_score"],
        "overall_score": ["overall_score"],
        "shortlist_status": ["shortlist_status"],
        "evaluation_comments": ["evaluation_comments"],
        
        # Process monitoring
        "task_statuses": ["task_statuses"],
        "current_task": ["current_task"],
        
        # Analysis results fields
        "linguistic_analysis_results": ["linguistic_analysis_results"],
        "semantic_analysis_results": ["semantic_analysis_results"],
        "cultural_analysis_results": ["cultural_analysis_results"],
        "translation_analysis_results": ["translation_analysis_results"],
        "analysis_results": ["analysis_results"],
        "evaluation_results": ["evaluation_results"],
        "shortlisted_names": ["shortlisted_names"],
        
        # SEO Analysis Fields - exact from state.py
        "keyword_alignment": ["keyword_alignment"],
        "search_volume": ["search_volume"],
        "keyword_competition": ["keyword_competition"],
        "branded_keyword_potential": ["branded_keyword_potential"],
        "non_branded_keyword_potential": ["non_branded_keyword_potential"],
        "exact_match_search_results": ["exact_match_search_results"],
        "competitor_domain_strength": ["competitor_domain_strength"],
        "name_length_searchability": ["name_length_searchability"],
        "unusual_spelling_impact": ["unusual_spelling_impact"],
        "content_marketing_opportunities": ["content_marketing_opportunities"],
        "social_media_availability": ["social_media_availability"],
        "social_media_discoverability": ["social_media_discoverability"],
        "negative_keyword_associations": ["negative_keyword_associations"],
        "negative_search_results": ["negative_search_results"],
        "seo_viability_score": ["seo_viability_score"],
        "seo_recommendations": ["seo_recommendations"],
        "seo_analysis_results": ["seo_analysis_results"],
        
        # Competitor Analysis Fields
        "competitor_analysis_results": ["competitor_analysis_results"],
        
        # Market Research Analysis Fields - exact from state.py
        "market_research": ["market_research_results"],
        "market_research_results": ["market_research_results"],
        
        # Domain Analysis Fields - exact from state.py
        "domain_analysis": ["domain_analysis_results"],
        "domain_analysis_results": ["domain_analysis_results"],
        
        # Brand evaluation fields
        "brand_name_data": ["brand_name_data"],
        
        # Survey simulation fields
        "survey_simulation_results": ["survey_simulation_results"],
        
        # Report fields
        "report": ["report"],
        "compiled_report": ["compiled_report"],
        "report_url": ["report_url"],
        "formatted_report_path": ["formatted_report_path"],
        "version": ["version"],
        "created_at": ["created_at"],
        "last_updated": ["last_updated"],
        "format": ["format"],
        "file_size_kb": ["file_size_kb"],
        "notes": ["notes"],
        "token_count": ["token_count"],
    }
    
    # Try exact key match first
    if isinstance(data, dict):
        for key in possible_keys:
            if key in data:
                return data[key]
    
    # Expand possible keys with their mappings
    expanded_keys = []
    for key in possible_keys:
        if key in key_mappings:
            expanded_keys.extend(key_mappings[key])
        else:
            expanded_keys.append(key)
            
    # Make the expanded keys unique
    expanded_keys = list(set(expanded_keys))
        
    # If data is a dictionary, check if any of the possible keys exist
    if isinstance(data, dict):
        # First, check if any of the provided keys exist at this level
        for key in expanded_keys:
            if key in data:
                return data[key]
        
        # If not found at this level, check nested dictionaries
        # Skip metadata and config keys that are unlikely to contain target values
        skip_keys = ["created_at", "updated_at", "timestamp", "id", "thread_id", 
                     "run_id", "status", "metadata", "config", "messages", "interrupts"]
                     
        for key, value in data.items():
            if key in skip_keys:
                continue
                
            result = find_value_in_data(value, possible_keys, max_depth, current_depth + 1)
            if result is not None:
                return result
                
    # If data is a list, check each item
    elif isinstance(data, list):
        for item in data:
            result = find_value_in_data(item, possible_keys, max_depth, current_depth + 1)
            if result is not None:
                return result
                
    # If we're looking for complex nested keys like 'linguistics.phonetic'
    for key in possible_keys:
        if '.' in key:
            parts = key.split('.', 1)  # Split only on the first dot
            if isinstance(data, dict) and parts[0] in data:
                result = find_value_in_data(data[parts[0]], [parts[1]], max_depth, current_depth + 1)
                if result is not None:
                    return result
    
    # If nothing is found, return None
    return None

def find_names_in_data(data, max_depth=10, current_depth=0):
    """
    Recursively search through a data structure to find brand names and their associated data.
    
    Args:
        data: The data structure to search (can be dict, list, or scalar)
        max_depth: Maximum recursion depth to prevent stack overflow
        current_depth: Current recursion depth (used internally)
        
    Returns:
        A list of dictionaries containing brand names and their associated data
    """
    # Prevent excessive recursion
    if current_depth > max_depth:
        return []
        
    # If data is None, return empty list
    if data is None:
        return []
        
    result = []
    
    # Check if this is a brand name entry directly
    if isinstance(data, dict):
        # Try to identify if this is a dictionary containing a brand name
        # Check for key pattern that would indicate a brand name entry
        if 'brand_name' in data or 'name' in data:
            # Normalize the data structure
            name_key = 'brand_name' if 'brand_name' in data else 'name'
            name = data.get(name_key)
            
            # Only include entries that have a valid name
            if name and isinstance(name, str):
                # Create a normalized entry
                entry = {
                    'name': name,
                    # Include other relevant fields if they exist
                    'rank': data.get('rank', None),
                    'notes': data.get('notes', None),
                    'errors': data.get('errors', [])
                }
                
                # Copy over any other fields that might be relevant
                for key, value in data.items():
                    if key not in ['name', 'brand_name', 'rank', 'notes', 'errors']:
                        entry[key] = value
                        
                result.append(entry)
                return result
                
        # Check if this is a dictionary of brand names
        # First check keys that might contain brand names
        brand_keys = ['brand_names', 'generated_names', 'names', 'options', 'proposals']
        for key in brand_keys:
            if key in data and isinstance(data[key], (list, dict)):
                brands = find_names_in_data(data[key], max_depth, current_depth + 1)
                if brands:
                    result.extend(brands)
                    
        # If still no results, recursively check all values
        if not result:
            for key, value in data.items():
                # Skip metadata keys that are unlikely to contain brand names
                if key in ['created_at', 'updated_at', 'timestamp', 'id', 'thread_id',
                          'run_id', 'status', 'metadata']:
                    continue
                    
                brands = find_names_in_data(value, max_depth, current_depth + 1)
                if brands:
                    result.extend(brands)
    
    # If data is a list, check each item
    elif isinstance(data, list):
        for item in data:
            brands = find_names_in_data(item, max_depth, current_depth + 1)
            if brands:
                result.extend(brands)
                
    return result