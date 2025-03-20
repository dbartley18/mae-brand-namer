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
    Renders thread data in a structured format with tabs for different sections.
    
    Args:
        thread_data: A dictionary containing thread data from the LangSmith API
    """
    if not thread_data:
        st.error("No thread data available. Please check the thread ID and try again.")
        return
        
    # Create main tabs for different sections in the specified order
    tabs = st.tabs([
        "Brand Context",
        "Name Generation",
        "Pre Analyses",
        "Name Evaluation",
        "Post Processing",
        "Translation Analysis",
        "Domain Analysis",
        "Research",
        "Metadata",
        "Raw JSON"
    ])

    def _render_translation_analysis(analysis):
        """Helper function to render translation analysis consistently"""
        cols = st.columns(2)
        with cols[0]:
            st.write("**Direct Translation:**", analysis.get("direct_translation", ""))
            st.write("**Semantic Shift:**", analysis.get("semantic_shift", ""))
            st.write("**Pronunciation Difficulty:**", analysis.get("pronunciation_difficulty", ""))
            st.write("**Phonetic Retention:**", analysis.get("phonetic_retention", ""))
            st.write("**Cultural Acceptability:**", analysis.get("cultural_acceptability", ""))
            st.write("**Brand Essence Preserved:**", analysis.get("brand_essence_preserved", ""))
        with cols[1]:
            st.write("**Global vs Local Balance:**", analysis.get("global_consistency_vs_localization", ""))
            st.write("**Adaptation Needed:**", "Yes" if analysis.get("adaptation_needed") else "No")
            if analysis.get("adaptation_needed"):
                st.write("**Proposed Adaptation:**", analysis.get("proposed_adaptation", ""))
            if analysis.get("phonetic_similarity_undesirable"):
                st.warning("⚠️ Potential undesirable phonetic similarity detected")
            if analysis.get("rank"):
                st.metric("Translation Viability Score", analysis.get("rank"))
        
        # Additional Notes
        if analysis.get("notes"):
            st.write("**Additional Notes:**", analysis.get("notes"))
        st.divider()

    # 1. Brand Context
    with tabs[0]:
        st.header("Brand Context")
        brand_context = {}
        
        # Extract brand context fields
        context_fields = [
            ("Brand Identity Brief", "brand_identity_brief"),
            ("Brand Promise", "brand_promise"),
            ("Brand Values", "brand_values"),
            ("Brand Personality", "brand_personality"),
            ("Brand Tone of Voice", "brand_tone_of_voice"),
            ("Brand Purpose", "brand_purpose"),
            ("Brand Mission", "brand_mission"),
            ("Target Audience", "target_audience"),
            ("Customer Needs", "customer_needs"),
            ("Market Positioning", "market_positioning"),
            ("Competitive Landscape", "competitive_landscape"),
            ("Industry Focus", "industry_focus"),
            ("Industry Trends", "industry_trends")
        ]
        
        # Populate brand_context dictionary
        for display_name, field_name in context_fields:
            value = find_value_in_data(thread_data, [field_name])
            if value:
                with st.expander(display_name, expanded=True):
                    if isinstance(value, str):
                        st.write(value)
                    elif isinstance(value, list):
                        for item in value:
                            st.write(f"- {item}")
                    elif isinstance(value, dict):
                        _render_analysis_section(value, display_name.lower())
        
        if not any(brand_context.values()):
            st.info("No brand context data found.")
    
    # 2. Name Generation
    with tabs[1]:
        st.header("Generated Brand Names")
        generated_names = find_value_in_data(thread_data, ["generated_names"])
        
        if generated_names:
            # Convert to list if it's not already
            if not isinstance(generated_names, list):
                if isinstance(generated_names, dict) and "names" in generated_names:
                    generated_names = generated_names.get("names", [])
                else:
                    generated_names = [generated_names]
            
            # Sort names by rank if available
            sorted_names = []
            for name_data in generated_names:
                if isinstance(name_data, dict):
                    rank = name_data.get("rank", 999)
                    sorted_names.append((rank, name_data))
            
            sorted_names.sort(key=lambda x: x[0])
            
            # Display each name with its details
            for rank, name_data in sorted_names:
                brand_name = name_data.get("brand_name", "") or name_data.get("name", "")
                if brand_name:
                    with st.expander(f"{brand_name}", expanded=True):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Name Details")
                            metrics = {
                                "Rank": rank if rank != 999 else None,
                                "Category": name_data.get("naming_category", ""),
                                "Brand Personality Alignment": name_data.get("brand_personality_alignment", ""),
                                "Brand Promise Alignment": name_data.get("brand_promise_alignment", ""),
                                "Target Audience Relevance": name_data.get("target_audience_relevance", ""),
                                "Market Differentiation": name_data.get("market_differentiation", "")
                            }
                            
                            for metric, value in metrics.items():
                                if value:
                                    st.metric(metric, value)
                        
                        with col2:
                            st.subheader("Rationale")
                            rationale = name_data.get("rationale", "") or name_data.get("name_generation_methodology", "")
                            if rationale:
                                st.write(rationale)
        else:
            st.info("No generated names found in the thread data.")
    
    # 3. Pre Analyses with child tabs
    with tabs[2]:
        st.header("Pre-Generation Analyses")
        pre_analysis_tabs = st.tabs(["Linguistic Analysis", "Semantic Analysis", "Cultural Sensitivity"])
        
        # Linguistic Analysis
        with pre_analysis_tabs[0]:
            st.subheader("Linguistic Analysis")
            linguistic_analysis = find_value_in_data(thread_data, ["linguistic_analysis_results"])
            if linguistic_analysis:
                if isinstance(linguistic_analysis, dict):
                    for name, analysis in linguistic_analysis.items():
                        with st.expander(f"Analysis for: {name}", expanded=True):
                            cols = st.columns(2)
                            with cols[0]:
                                st.metric("Pronunciation Ease", analysis.get("pronunciation_ease"))
                                st.metric("Sound Symbolism", analysis.get("sound_symbolism"))
                                st.metric("Overall Readability", analysis.get("overall_readability_score"))
                                st.metric("Rank", analysis.get("rank"))
                            with cols[1]:
                                st.write("**Euphony vs Cacophony:**", analysis.get("euphony_vs_cacophony"))
                                st.write("**Rhythm and Meter:**", analysis.get("rhythm_and_meter"))
                                st.write("**Word Class:**", analysis.get("word_class"))
                            
                            st.write("**Notes:**", analysis.get("notes"))
                            
                            if analysis.get("homophones_homographs"):
                                st.warning("⚠️ This name has similar sounding or looking words")
            else:
                st.info("No linguistic analysis data found.")
        
        # Semantic Analysis
        with pre_analysis_tabs[1]:
            st.subheader("Semantic Analysis")
            semantic_analysis = find_value_in_data(thread_data, ["semantic_analysis_results"])
            if semantic_analysis and isinstance(semantic_analysis, list):
                for analysis in semantic_analysis:
                    if isinstance(analysis, dict):
                        name = analysis.get("brand_name", "")
                        with st.expander(f"Analysis for: {name}", expanded=True):
                            cols = st.columns(2)
                            with cols[0]:
                                st.write("**Denotative Meaning:**", analysis.get("denotative_meaning"))
                                st.write("**Etymology:**", analysis.get("etymology"))
                                st.metric("Descriptiveness", analysis.get("descriptiveness"))
                                st.metric("Concreteness", analysis.get("concreteness"))
                            with cols[1]:
                                st.write("**Brand Name Type:**", analysis.get("brand_name_type"))
                                st.write("**Emotional Valence:**", analysis.get("emotional_valence"))
                                st.write("**Sensory Associations:**", analysis.get("sensory_associations"))
                                st.metric("Brand Fit/Relevance", analysis.get("brand_fit_relevance"))
                            
                            # Additional semantic details
                            st.write("**Figurative Language:**", analysis.get("figurative_language"))
                            if analysis.get("irony_or_paradox"):
                                st.info("Contains irony or paradox")
                            if analysis.get("humor_playfulness"):
                                st.info("Contains humor/playfulness")
                            st.metric("Memorability Score", analysis.get("memorability_score"))
            else:
                st.info("No semantic analysis data found.")
        
        # Cultural Sensitivity Analysis
        with pre_analysis_tabs[2]:
            st.subheader("Cultural Sensitivity Analysis")
            cultural_analysis = find_value_in_data(thread_data, ["cultural_analysis_results"])
            if cultural_analysis:
                if isinstance(cultural_analysis, dict):
                    for name, analysis in cultural_analysis.items():
                        with st.expander(f"Analysis for: {name}", expanded=True):
                            cols = st.columns(2)
                            with cols[0]:
                                st.write("**Cultural Connotations:**", analysis.get("cultural_connotations"))
                                st.write("**Symbolic Meanings:**", analysis.get("symbolic_meanings"))
                                st.write("**Overall Risk Rating:**", analysis.get("overall_risk_rating"))
                                st.metric("Rank", analysis.get("rank"))
                            with cols[1]:
                                st.write("**Religious Sensitivities:**", analysis.get("religious_sensitivities"))
                                st.write("**Social/Political Taboos:**", analysis.get("social_political_taboos"))
                                if analysis.get("body_part_bodily_function_connotations"):
                                    st.warning("⚠️ Contains potentially sensitive anatomical/physiological references")
                            
                            st.write("**Notes:**", analysis.get("notes"))
            else:
                st.info("No cultural sensitivity analysis data found.")
    
    # 4. Name Evaluation
    with tabs[3]:
        st.header("Name Evaluation")
        evaluation_results = find_value_in_data(thread_data, ["evaluation_results"])
        if evaluation_results:
            if isinstance(evaluation_results, dict):
                for name, eval_data in evaluation_results.items():
                    with st.expander(f"Evaluation for: {name}", expanded=True):
                        cols = st.columns(3)
                        with cols[0]:
                            st.metric("Overall Score", eval_data.get("overall_score"))
                            st.metric("Brand Fit", eval_data.get("brand_fit_score"))
                            st.metric("Strategic Alignment", eval_data.get("strategic_alignment_score"))
                        with cols[1]:
                            st.metric("Memorability", eval_data.get("memorability_score"))
                            st.metric("Pronounceability", eval_data.get("pronounceability_score"))
                            st.metric("Domain Viability", eval_data.get("domain_viability_score"))
                        with cols[2]:
                            st.write("**Positioning Strength:**", eval_data.get("positioning_strength"))
                            st.write("**Visual Branding Potential:**", eval_data.get("visual_branding_potential"))
                            st.write("**Storytelling Potential:**", eval_data.get("storytelling_potential"))
                        
                        st.write("**Evaluation Comments:**", eval_data.get("evaluation_comments"))
                        if eval_data.get("shortlist_status") == "Yes":
                            st.success("✅ Selected for shortlist")
        else:
            st.info("No evaluation results found.")
    
    # 5. Post Processing with child tabs
    with tabs[4]:
        st.header("Post-Processing Analyses")
        post_processing_tabs = st.tabs(["SEO Analysis", "Survey Results", "Competitor Analysis"])
        
        # SEO Analysis
        with post_processing_tabs[0]:
            st.subheader("SEO Analysis")
            seo_analysis = find_value_in_data(thread_data, ["seo_analysis_results"])
            if seo_analysis:
                if isinstance(seo_analysis, dict):
                    for name, analysis in seo_analysis.items():
                        st.markdown(f"### SEO Analysis for: {name}")
                        cols = st.columns(2)
                        with cols[0]:
                            st.write("**Keyword Alignment:**", analysis.get("keyword_alignment"))
                            st.write("**Search Volume:**", analysis.get("search_volume"))
                            st.write("**Keyword Competition:**", analysis.get("keyword_competition"))
                            st.write("**Branded Keyword Potential:**", analysis.get("branded_keyword_potential"))
                            st.write("**Non-Branded Keyword Potential:**", analysis.get("non_branded_keyword_potential"))
                        with cols[1]:
                            st.write("**Social Media Availability:**", analysis.get("social_media_availability"))
                            st.write("**Social Media Discoverability:**", analysis.get("social_media_discoverability"))
                            st.metric("SEO Viability Score", analysis.get("seo_viability_score"))
                            
                        # Display recommendations
                        seo_recs = analysis.get("seo_recommendations", [])
                        if seo_recs:
                            st.subheader("SEO Recommendations")
                            if isinstance(seo_recs, list):
                                for rec in seo_recs:
                                    st.write(f"- {rec}")
                            elif isinstance(seo_recs, dict):
                                for key, rec in seo_recs.items():
                                    st.write(f"- **{key}:** {rec}")
                            else:
                                st.write(seo_recs)
                        st.divider()
                elif isinstance(seo_analysis, list):
                    for analysis in seo_analysis:
                        if isinstance(analysis, dict):
                            name = analysis.get("brand_name", "")
                            st.markdown(f"### SEO Analysis for: {name}")
                            cols = st.columns(2)
                            with cols[0]:
                                st.write("**Keyword Alignment:**", analysis.get("keyword_alignment"))
                                st.write("**Search Volume:**", analysis.get("search_volume"))
                                st.write("**Keyword Competition:**", analysis.get("keyword_competition"))
                                st.write("**Branded Keyword Potential:**", analysis.get("branded_keyword_potential"))
                                st.write("**Non-Branded Keyword Potential:**", analysis.get("non_branded_keyword_potential"))
                            with cols[1]:
                                st.write("**Social Media Availability:**", analysis.get("social_media_availability"))
                                st.write("**Social Media Discoverability:**", analysis.get("social_media_discoverability"))
                                st.metric("SEO Viability Score", analysis.get("seo_viability_score"))
                                
                            # Display recommendations
                            seo_recs = analysis.get("seo_recommendations", [])
                            if seo_recs:
                                st.subheader("SEO Recommendations")
                                if isinstance(seo_recs, list):
                                    for rec in seo_recs:
                                        st.write(f"- {rec}")
                                elif isinstance(seo_recs, dict):
                                    for key, rec in seo_recs.items():
                                        st.write(f"- **{key}:** {rec}")
                                else:
                                    st.write(seo_recs)
                            st.divider()
            else:
                st.info("No SEO analysis data found.")

        # Survey Results
        with post_processing_tabs[1]:
            st.subheader("Survey Simulation Results")
            survey_results = find_value_in_data(thread_data, ["survey_simulation_results"])
            if survey_results:
                # Handle both list and dictionary formats
                if isinstance(survey_results, list):
                    # Process each brand's survey results
                    for brand_survey in survey_results:
                        brand_name = brand_survey.get("brand_name", "Unknown Brand")
                        st.markdown(f"### {brand_name}")
                        
                        individual_personas = brand_survey.get("individual_personas", [])
                        if individual_personas:
                            st.markdown("#### Individual Persona Responses")
                            for persona in individual_personas:
                                company_name = persona.get('company_name', 'Unknown Company')
                                job_title = persona.get('job_title', 'Unknown Role')
                                persona_title = f"Persona: {job_title} at {company_name}"
                                with st.expander(persona_title, expanded=False):
                                    _render_survey_persona(persona)
                        else:
                            st.info(f"No survey responses found for {brand_name}")
                        st.markdown("---")
                else:
                    # Handle single brand survey results
                    individual_personas = survey_results.get("individual_personas", [])
                    if individual_personas:
                        st.markdown("#### Individual Persona Responses")
                        for persona in individual_personas:
                            company_name = persona.get('company_name', 'Unknown Company')
                            job_title = persona.get('job_title', 'Unknown Role')
                            persona_title = f"Persona: {job_title} at {company_name}"
                            with st.expander(persona_title, expanded=False):
                                _render_survey_persona(persona)
                    else:
                        st.info("No survey responses found.")
            else:
                st.info("No survey simulation results found.")

        # Competitor Analysis
        with post_processing_tabs[2]:
            st.subheader("Competitor Analysis")
            competitor_analysis = find_value_in_data(thread_data, ["competitor_analysis_results"])
            if competitor_analysis:
                if isinstance(competitor_analysis, list):
                    for brand_analysis in competitor_analysis:
                        brand_name = brand_analysis.get("brand_name", "Unknown Brand")
                        competitors = brand_analysis.get("competitors", [])
                        
                        with st.expander(f"Competitor Analysis for: {brand_name}", expanded=True):
                            for competitor in competitors:
                                st.markdown(f"### {competitor.get('competitor_name', 'Unknown Competitor')}")
                                
                                cols = st.columns(2)
                                with cols[0]:
                                    st.metric("Risk of Confusion", competitor.get("risk_of_confusion", 0))
                                    st.write("**Positioning:**", competitor.get("competitor_positioning", ""))
                                    st.write("**Strengths:**", competitor.get("competitor_strengths", ""))
                                    st.write("**Weaknesses:**", competitor.get("competitor_weaknesses", ""))
                                
                                with cols[1]:
                                    st.write("**Target Audience Perception:**", competitor.get("target_audience_perception", ""))
                                    st.write("**Differentiation Opportunity:**", competitor.get("competitor_differentiation_opportunity", ""))
                                    st.write("**Trademark Conflict Risk:**", competitor.get("trademark_conflict_risk", ""))
                                
                                st.divider()
            else:
                st.info("No competitor analysis data found.")
    
    # 6. Translation Analysis (now a parent tab)
    with tabs[5]:
        st.header("Translation Analysis")
        translation_analysis = find_value_in_data(thread_data, ["translation_analysis_results"])
        if translation_analysis:
            # Handle both list and dictionary formats
            if isinstance(translation_analysis, dict):
                # If it's already organized by brand name and language
                for brand_name, languages in translation_analysis.items():
                    st.markdown(f"### {brand_name}")
                    for lang, analysis in languages.items():
                        with st.expander(f"{lang} Analysis", expanded=True):
                            _render_translation_analysis(analysis)
            elif isinstance(translation_analysis, list):
                # Organize list data by brand name and language
                organized_data = {}
                for analysis in translation_analysis:
                    if isinstance(analysis, dict):
                        brand_name = analysis.get("brand_name", "Unknown Brand")
                        target_lang = analysis.get("target_language", "Unknown Language")
                        
                        if brand_name not in organized_data:
                            organized_data[brand_name] = {}
                        organized_data[brand_name][target_lang] = analysis
                
                # Display organized data
                for brand_name, languages in organized_data.items():
                    st.markdown(f"### {brand_name}")
                    for lang, analysis in languages.items():
                        with st.expander(f"{lang} Analysis", expanded=True):
                            _render_translation_analysis(analysis)
        else:
            st.info("No translation analysis data found.")

    # 7. Domain Analysis (now a parent tab)
    with tabs[6]:
        st.header("Domain Analysis")
        domain_analysis = find_value_in_data(thread_data, ["domain_analysis_results"])
        if domain_analysis:
            if isinstance(domain_analysis, dict):
                # Handle dictionary format
                for name, analysis in domain_analysis.items():
                    st.markdown(f"### Domain Analysis for: {name}")
                    _render_domain_analysis(analysis)
            elif isinstance(domain_analysis, list):
                # Handle list format
                for analysis in domain_analysis:
                    if isinstance(analysis, dict):
                        name = analysis.get("brand_name", "") or analysis.get("name", "Unknown")
                        st.markdown(f"### Domain Analysis for: {name}")
                        _render_domain_analysis(analysis)
        else:
            st.info("No domain analysis data found.")
    
    # 8. Research with child tabs
    with tabs[7]:
        st.header("Market & Competitor Research")
        research_tabs = st.tabs(["Market Research", "Competitor Research"])
        
        # Market Research
        with research_tabs[0]:
            st.subheader("Market Research")
            market_research = find_value_in_data(thread_data, ["market_research_results"])
            if market_research and isinstance(market_research, list):
                for analysis in market_research:
                    if isinstance(analysis, dict):
                        name = analysis.get("brand_name", "")
                        with st.expander(f"Market Analysis for: {name}", expanded=True):
                            cols = st.columns(2)
                            with cols[0]:
                                st.write("**Industry Name:**", analysis.get("industry_name"))
                                st.write("**Market Size:**", analysis.get("market_size"))
                                st.write("**Market Growth Rate:**", analysis.get("market_growth_rate"))
                                st.write("**Market Opportunity:**", analysis.get("market_opportunity"))
                            with cols[1]:
                                st.write("**Target Audience Fit:**", analysis.get("target_audience_fit"))
                                st.write("**Customer Pain Points:**", analysis.get("customer_pain_points"))
                                st.write("**Potential Risks:**", analysis.get("potential_risks"))
                                st.write("**Market Entry Barriers:**", analysis.get("market_entry_barriers"))
            else:
                st.info("No market research data found.")
        
        # Competitor Research
        with research_tabs[1]:
            st.subheader("Competitor Research")
            competitor_analysis = find_value_in_data(thread_data, ["competitor_analysis_results"])
            if competitor_analysis and isinstance(competitor_analysis, list):
                for analysis in competitor_analysis:
                    if isinstance(analysis, dict):
                        name = analysis.get("competitor_name", "")
                        with st.expander(f"Analysis for: {name}", expanded=True):
                            cols = st.columns(2)
                            with cols[0]:
                                st.write("**Competitor Name:**", name)
                                st.write("**Positioning:**", analysis.get("competitor_positioning"))
                                st.write("**Naming Style:**", analysis.get("competitor_naming_style"))
                                st.write("**Keywords:**", analysis.get("competitor_keywords"))
                            with cols[1]:
                                st.write("**Strengths:**", analysis.get("competitor_strengths"))
                                st.write("**Weaknesses:**", analysis.get("competitor_weaknesses"))
                                st.write("**Risk of Confusion:**", analysis.get("risk_of_confusion"))
                                st.write("**Trademark Risk:**", analysis.get("trademark_conflict_risk"))
                            
                            if analysis.get("differentiation_score"):
                                st.metric("Differentiation Score", analysis.get("differentiation_score"))
            else:
                st.info("No competitor analysis data found.")
    
    # 9. Metadata
    with tabs[8]:
        st.header("Thread Metadata")
        if isinstance(thread_data, dict):
            metadata = thread_data.get("metadata", {})
            if metadata:
                st.json(metadata)
            else:
                st.info("No metadata available")
    
    # 10. Raw JSON
    with tabs[9]:
        st.header("Raw JSON Data")
        st.json(thread_data)

def _render_analysis_section(data, section_type):
    """Helper function to render an analysis section based on its type"""
    if not data:
        return
        
    # Skip if data is just a string
    if isinstance(data, str):
        st.write(data)
        return
        
    # Skip if data is a list but not of dictionaries
    if isinstance(data, list):
        # If it's a list of strings, render as bullet points
        if all(isinstance(item, str) for item in data):
            for item in data:
                st.write(f"- {item}")
            return
            
        # If it's a list of dictionaries, render each dictionary
        for item in data:
            if isinstance(item, dict):
                _render_analysis_section(item, section_type)
        return
        
    # For dictionary data, create a two-column layout
    col1, col2 = st.columns(2)
    
    # Define fields to display in each column based on section type
    fields = {}
    
    if section_type == "linguistic":
        fields = {
            col1: [
                ("Word Class", "word_class"),
                ("Sound Symbolism", "sound_symbolism"),
                ("Rhythm and Meter", "rhythm_and_meter"),
                ("Pronunciation Ease", "pronunciation_ease"),
                ("Euphony vs Cacophony", "euphony_vs_cacophony"),
                ("Overall Readability Score", "overall_readability_score")
            ],
            col2: [
                ("Inflectional Properties", "inflectional_properties"),
                ("Neologism Appropriateness", "neologism_appropriateness"),
                ("Morphological Transparency", "morphological_transparency"),
                ("Naturalness in Collocations", "naturalness_in_collocations"),
                ("Ease of Marketing Integration", "ease_of_marketing_integration"),
                ("Semantic Distance from Competitors", "semantic_distance_from_competitors")
            ]
        }
    elif section_type == "semantic":
        fields = {
            col1: [
                ("Denotative Meaning", "denotative_meaning"),
                ("Etymology", "etymology"),
                ("Descriptiveness", "descriptiveness"),
                ("Concreteness", "concreteness"),
                ("Emotional Valence", "emotional_valence"),
                ("Sensory Associations", "sensory_associations")
            ],
            col2: [
                ("Figurative Language", "figurative_language"),
                ("Ambiguity", "ambiguity"),
                ("Irony or Paradox", "irony_or_paradox"),
                ("Humor/Playfulness", "humor_playfulness"),
                ("Brand Name Type", "brand_name_type"),
                ("Brand Fit/Relevance", "brand_fit_relevance")
            ]
        }
    elif section_type == "cultural":
        fields = {
            col1: [
                ("Cultures Examined", "cultures_examined"),
                ("Overall Sensitivity Rating", "overall_sensitivity_rating"),
                ("Potentially Problematic Meanings", "potentially_problematic_meanings"),
                ("Religious Significance", "religious_significance")
            ],
            col2: [
                ("Historical Associations", "historical_associations"),
                ("Offensive Homonyms", "offensive_homonyms"),
                ("Demographic Specific Reactions", "demographic_specific_reactions"),
                ("Recommendations", "recommendations")
            ]
        }
    elif section_type == "domain":
        fields = {
            col1: [
                ("Domain Name", "domain_name"),
                ("Availability", "available"),
                ("Registrar", "registrar"),
                ("Expiration Date", "expiration_date"),
                ("Domain Quality Score", "domain_quality_score")
            ],
            col2: [
                ("Memorability Rating", "memorability_rating"),
                ("SEO Potential", "seo_potential"),
                ("Brand Match", "brand_match"),
                ("Acquisition Recommendations", "acquisition_recommendations"),
                ("Estimated Acquisition Cost", "estimated_acquisition_cost")
            ]
        }
    elif section_type == "seo":
        fields = {
            col1: [
                ("Keyword Alignment", "keyword_alignment"),
                ("Search Volume", "search_volume"),
                ("Keyword Competition", "keyword_competition"),
                ("Branded Keyword Potential", "branded_keyword_potential"),
                ("Non-Branded Keyword Potential", "non_branded_keyword_potential"),
                ("Exact Match Search Results", "exact_match_search_results"),
                ("Name Length Searchability", "name_length_searchability")
            ],
            col2: [
                ("Unusual Spelling Impact", "unusual_spelling_impact"),
                ("Content Marketing Opportunities", "content_marketing_opportunities"),
                ("Social Media Availability", "social_media_availability"),
                ("Social Media Discoverability", "social_media_discoverability"),
                ("Negative Keyword Associations", "negative_keyword_associations"),
                ("Negative Search Results", "negative_search_results"),
                ("SEO Viability Score", "seo_viability_score")
            ]
        }
    elif section_type == "market":
        fields = {
            col1: [
                ("Industry Name", "industry_name"),
                ("Market Size", "market_size"),
                ("Market Growth Rate", "market_growth_rate"),
                ("Market Opportunity", "market_opportunity"),
                ("Market Viability", "market_viability")
            ],
            col2: [
                ("Target Audience Fit", "target_audience_fit"),
                ("Customer Pain Points", "customer_pain_points"),
                ("Potential Risks", "potential_risks"),
                ("Market Entry Barriers", "market_entry_barriers")
            ]
        }
    elif section_type == "competitor":
        fields = {
            col1: [
                ("Competitor Name", "competitor_name"),
                ("Positioning", "positioning"),
                ("Strengths", "strengths"),
                ("Weaknesses", "weaknesses")
            ],
            col2: [
                ("Market Share", "market_share"),
                ("Brand Voice", "brand_voice"),
                ("Target Demographics", "target_demographics"),
                ("Key Differentiators", "key_differentiators")
            ]
        }
    elif section_type == "survey":
        fields = {
            col1: [
                ("Sample Size", "sample_size"),
                ("Methodology", "methodology"),
                ("Key Findings", "key_findings"),
                ("Statistical Significance", "statistical_significance")
            ],
            col2: [
                ("Survey Demographics", "survey_demographics"),
                ("Name Preference Ranking", "name_preference_ranking"),
                ("Qualitative Feedback", "qualitative_feedback"),
                ("Demographic Variations", "demographic_variations")
            ]
        }
    else:
        # Generic field display for other section types
        i = 0
        for key, value in data.items():
            if key.startswith("_") or key in ["id", "type", "run_id", "thread_id", "assistant_id"]:
                continue
                
            col = col1 if i % 2 == 0 else col2
            with col:
                if isinstance(value, (str, int, float, bool)):
                    st.metric(key.replace("_", " ").title(), value)
                elif isinstance(value, list) and all(isinstance(item, str) for item in value):
                    st.write(f"**{key.replace('_', ' ').title()}**")
                    for item in value:
                        st.write(f"- {item}")
                elif value is not None:
                    st.write(f"**{key.replace('_', ' ').title()}**")
                    st.write(str(value))
            i += 1
        return
    
    # Display fields for the section
    for col, field_list in fields.items():
        with col:
            for display_name, field_name in field_list:
                value = data.get(field_name)
                if value is not None:
                    if isinstance(value, (str, int, float, bool)):
                        # For numeric values, use metric
                        if isinstance(value, (int, float)) or (isinstance(value, str) and value.replace(".", "", 1).isdigit()):
                            st.metric(display_name, value)
                        else:
                            st.write(f"**{display_name}**")
                            st.write(value)
                    elif isinstance(value, list):
                        st.write(f"**{display_name}**")
                        if all(isinstance(item, str) for item in value):
                            for item in value:
                                st.write(f"- {item}")
                        else:
                            st.write(str(value))

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
                # Special handling for survey simulation results
                if key == "survey_simulation_results" and isinstance(data[key], list):
                    # If it's a list with one item that has individual_personas, return that
                    if len(data[key]) > 0 and isinstance(data[key][0], dict) and "individual_personas" in data[key][0]:
                        return data[key][0]
                return data[key]
                
        # Check in thread metadata
        if "metadata" in data and isinstance(data["metadata"], dict):
            for key in possible_keys:
                if key in data["metadata"]:
                    return data["metadata"][key]
                    
        # Check in state data
        if "state" in data and isinstance(data["state"], dict):
            for key in possible_keys:
                if key in data["state"]:
                    return data["state"][key]
                    
    # If data is a dictionary, check all values recursively
    if isinstance(data, dict):
        for key, value in data.items():
            # Direct key match
            if key in possible_keys:
                return value
            # Recurse into value
            result = find_value_in_data(value, possible_keys, max_depth, current_depth + 1)
            if result is not None:
                return result
                
    # If data is a list, check all elements recursively
    if isinstance(data, list):
        for item in data:
            result = find_value_in_data(item, possible_keys, max_depth, current_depth + 1)
            if result is not None:
                return result
                
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

def _render_domain_analysis(analysis):
    """Helper function to render domain analysis consistently"""
    cols = st.columns(2)
    with cols[0]:
        st.write("**Domain Availability:**", "Available" if analysis.get("domain_exact_match") else "Not Available")
        st.write("**Acquisition Cost:**", analysis.get("acquisition_cost", "Unknown"))
        st.write("**Alternative TLDs:**", ", ".join(analysis.get("alternative_tlds", [])))
        st.write("**Domain Length & Readability:**", analysis.get("domain_length_readability", ""))
        st.write("**Brand Name Clarity in URL:**", analysis.get("brand_name_clarity_in_url", ""))
    with cols[1]:
        st.write("**Contains Hyphens/Numbers:**", "Yes" if analysis.get("hyphens_numbers_present") else "No")
        st.write("**Misspellings Available:**", "Yes" if analysis.get("misspellings_variations_available") else "No")
        st.write("**Social Media Handles:**", ", ".join(analysis.get("social_media_availability", [])))
        st.write("**Scalability:**", analysis.get("scalability_future_proofing", ""))
    
    # Additional Notes
    if analysis.get("notes"):
        st.write("**Additional Notes:**", analysis.get("notes"))
    st.divider()

def _render_survey_persona(persona):
    """Helper function to render a detailed survey persona"""
    # Profile Overview
    st.subheader("Profile Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Job Title:**", persona.get("job_title", "N/A"))
        st.write("**Department:**", persona.get("department", "N/A"))
        st.write("**Seniority:**", persona.get("seniority", "N/A"))
        st.write("**Industry:**", persona.get("industry", "N/A"))
        st.write("**Sub-vertical:**", persona.get("industry_sub_vertical", "N/A"))
        st.write("**Experience:**", f"{persona.get('years_of_experience', 'N/A')} years")
    with col2:
        st.write("**Company Size:**", persona.get("company_size_employees", "N/A"))
        st.write("**Company Revenue:**", persona.get("company_revenue", "N/A"))
        st.write("**Location:**", persona.get("geographic_location", "N/A"))
        st.write("**Reports To:**", persona.get("reports_to", "N/A"))
        st.write("**Decision Maker:**", "Yes" if str(persona.get("decision_maker")).lower() == "true" else "No")
        st.write("**Budget Authority:**", persona.get("budget_authority", "N/A"))

    # Scores and Metrics
    st.subheader("Brand Perception Scores")
    score_cols = st.columns(3)
    scores = {
        "Strategic Ranking": persona.get("strategic_ranking"),
        "Brand Promise": persona.get("brand_promise_perception_score"),
        "Personality Fit": persona.get("personality_fit_score"),
        "Competitive Differentiation": persona.get("competitive_differentiation_score"),
        "Competitor Benchmarking": persona.get("competitor_benchmarking_score"),
        "Market Adoption": persona.get("simulated_market_adoption_score")
    }
    
    for i, (metric, score) in enumerate(scores.items()):
        with score_cols[i % 3]:
            if score:
                try:
                    # Handle string values that might be quoted
                    if isinstance(score, str):
                        score = score.strip('"')
                    score_val = float(score)
                    st.metric(metric, f"{score_val:.1f}/10")
                except (ValueError, TypeError):
                    st.metric(metric, "N/A")

    # Create tabs for detailed sections
    detail_tabs = st.tabs([
        "Company Context",
        "Professional Profile",
        "Brand Relationships",
        "Success Metrics"
    ])

    # Company Context tab
    with detail_tabs[0]:
        st.write("**Market Share:**", persona.get("market_share", "N/A"))
        st.write("**Company Structure:**", persona.get("company_structure", "N/A"))
        st.write("**Growth Stage:**", persona.get("company_growth_stage", "N/A"))
        st.write("**Technology Stack:**", persona.get("technology_stack", "N/A"))
        
        # Parse and display company focus
        company_focus = persona.get("company_focus")
        if company_focus:
            try:
                focus_list = eval(company_focus)
                st.write("**Company Focus Areas:**")
                for focus in focus_list:
                    st.write(f"- {focus}")
            except:
                st.write("**Company Focus:**", company_focus)

    # Professional Profile tab
    with detail_tabs[1]:
        # Parse and display goals and challenges
        goals_challenges = persona.get("goals_and_challenges")
        if goals_challenges:
            try:
                gc_dict = eval(goals_challenges)
                if "Goals" in gc_dict:
                    st.write("**Goals:**")
                    for goal in gc_dict["Goals"]:
                        st.write(f"- {goal}")
                if "Challenges" in gc_dict:
                    st.write("**Challenges:**")
                    for challenge in gc_dict["Challenges"]:
                        st.write(f"- {challenge}")
            except:
                st.write("**Goals and Challenges:**", goals_challenges)

        # Parse and display pain points
        pain_points = persona.get("pain_points")
        if pain_points:
            try:
                points = eval(pain_points)
                st.write("**Pain Points:**")
                for point in points:
                    st.write(f"- {point}")
            except:
                st.write("**Pain Points:**", pain_points)

    # Brand Relationships tab
    with detail_tabs[2]:
        # Parse and display current brand relationships
        relationships = persona.get("current_brand_relationships")
        if relationships:
            try:
                rel_dict = eval(relationships)
                st.write("**Current Brand Relationships:**")
                for brand, relationship in rel_dict.items():
                    st.write(f"- **{brand}:** {relationship}")
            except:
                st.write("**Brand Relationships:**", relationships)

        st.write("**Event Attendance:**", persona.get("event_attendance", "N/A"))
        st.write("**Networking Habits:**", persona.get("networking_habits", "N/A"))
        st.write("**Social Media Usage:**", persona.get("social_media_usage", "N/A"))

    # Success Metrics tab
    with detail_tabs[3]:
        # Parse and display success metrics
        metrics = persona.get("success_metrics_product_service")
        if metrics:
            try:
                metric_list = eval(metrics)
                st.write("**Success Metrics:**")
                for metric in metric_list:
                    st.write(f"- {metric}")
            except:
                st.write("**Success Metrics:**", metrics)

        # Parse and display channel preferences
        channels = persona.get("channel_preferences_brand_interaction")
        if channels:
            try:
                channel_list = eval(channels)
                st.write("**Preferred Communication Channels:**")
                for channel in channel_list:
                    st.write(f"- {channel}")
            except:
                st.write("**Channel Preferences:**", channels)

        # Parse and display barriers to adoption
        barriers = persona.get("barriers_to_adoption")
        if barriers:
            try:
                barrier_list = eval(barriers)
                st.write("**Barriers to Adoption:**")
                for barrier in barrier_list:
                    st.write(f"- {barrier}")
            except:
                st.write("**Barriers to Adoption:**", barriers)

    st.divider()

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
                    langsmith_url = f"https://smith.langchain.com/api/traces/{trace_id}"
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