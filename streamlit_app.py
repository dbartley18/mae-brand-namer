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
    debug_container = status_container.container()
    
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
            except json.JSONDecodeError:
                continue
                
            # Extract event type
            event_type = data.get("type", "")
            
            # Store debug data for later (not in an expander)
            with debug_container:
                if i == 0:  # Only add the header once
                    st.subheader("Debug Information", help="Raw API responses")
                
                # Get node name from metadata if available
                node_name = "Unknown"
                if event_type == "status" and "metadata" in data:
                    node_name = data["metadata"].get("langgraph_node", f"Event {i}")
                elif event_type == "output":
                    node_name = f"Output {i}"
                else:
                    node_name = f"Event {i}"
                
                # Add step number if available
                step_num = ""
                if event_type == "status" and "metadata" in data and "langgraph_step" in data["metadata"]:
                    step_num = f" (Step {data['metadata']['langgraph_step']})"
                
                # Create expander with node name
                with st.expander(f"{node_name}{step_num}: {event_type}", expanded=False):
                    # Show streamlined metadata first if it's a status event
                    if event_type == "status" and "metadata" in data:
                        metadata = data.get("metadata", {})
                        if "langgraph_node" in metadata:
                            st.info(f"**Node:** {metadata['langgraph_node']}")
                        
                        if "langgraph_path" in metadata and isinstance(metadata["langgraph_path"], list):
                            st.info(f"**Path:** {' → '.join(metadata['langgraph_path'])}")
                            
                        if "langgraph_triggers" in metadata and isinstance(metadata["langgraph_triggers"], list) and metadata["langgraph_triggers"]:
                            st.info(f"**Triggers:** {', '.join(metadata['langgraph_triggers'])}")
                    
                    # Show raw data
                    st.caption("Raw JSON Data")
                    st.text(f"Raw data: {line_str[:200]}..." if len(line_str) > 200 else line_str)
                    
                    # Show full data as JSON
                    st.json(data)
            
            # Handle status updates
            if event_type == "status":
                # Extract status message
                message = data.get("message", "")
                if message:
                    status_message.info(message)
                
                # Extract metadata - this is where LangGraph puts its rich information
                metadata = data.get("metadata", {})
                
                # Extract LangGraph-specific data
                if "langgraph_node" in metadata:
                    current_node = metadata["langgraph_node"]
                    langgraph_data["nodes_visited"].add(current_node)
                
                if "langgraph_triggers" in metadata and isinstance(metadata["langgraph_triggers"], list):
                    for trigger in metadata["langgraph_triggers"]:
                        langgraph_data["triggers"].add(trigger)
                
                if "run_id" in metadata and not langgraph_data["run_id"]:
                    langgraph_data["run_id"] = metadata["run_id"]
                
                if "thread_id" in metadata and not langgraph_data["thread_id"]:
                    langgraph_data["thread_id"] = metadata["thread_id"]
                
                # Extract model information if available
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
                    
                    # Create a step record with all valuable metadata
                    step_record = {
                        "name": step_name,
                        "time": time.time() - last_update_time,
                        "node": current_node,
                        "step_number": step_num,
                        "metadata": {k: v for k, v in metadata.items() if k not in 
                                    ['LANGCHAIN_CALLBACKS_BACKGROUND', 'LANGSMITH_AUTH_ENDPOINT', 
                                     'x-forwarded-for', 'x-forwarded-host', 'x-forwarded-port',
                                     'x-forwarded-proto', 'x-forwarded-scheme', 'x-real-ip',
                                     'x-request-id', 'x-scheme']}
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
                    
                    # Update steps container (not using expanders here)
                    with steps_container:
                        if len(steps_completed) == 1:  # Only add the header once
                            st.subheader("Steps Details")
                            
                        st.markdown(f"##### Step {len(steps_completed)}: {step_name}")
                        
                        # Show LangGraph path info if available
                        if "langgraph_path" in metadata and isinstance(metadata["langgraph_path"], list):
                            st.caption(f"Path: {' → '.join(metadata['langgraph_path'])}")
                        
                        # Show key metrics in columns
                        cols = st.columns(3)
                        
                        # Show step number if available
                        if step_num:
                            cols[0].metric("Step Number", step_num)
                        
                        # Show duration
                        cols[1].metric("Duration", f"{step_record['time']:.2f}s")
                        
                        # Show model if available
                        if "ls_model_name" in metadata:
                            cols[2].metric("Model", metadata.get("ls_model_name", ""))
                        
                        # Create expandable section for detailed metadata
                        with st.expander("Step Metadata"):
                            # Display important metadata in a more organized way
                            st.markdown("##### Graph Information")
                            
                            # Graph details
                            graph_cols = st.columns(2)
                            if "langgraph_node" in metadata:
                                graph_cols[0].info(f"Node: {metadata['langgraph_node']}")
                            
                            if "langgraph_triggers" in metadata:
                                graph_cols[1].info(f"Triggers: {', '.join(metadata['langgraph_triggers'])}" 
                                                 if isinstance(metadata["langgraph_triggers"], list) 
                                                 else metadata["langgraph_triggers"])
                            
                            # Model information
                            if "ls_model_name" in metadata:
                                st.markdown("##### Model Information")
                                model_cols = st.columns(4)
                                model_cols[0].info(f"Name: {metadata.get('ls_model_name', 'N/A')}")
                                model_cols[1].info(f"Type: {metadata.get('ls_model_type', 'N/A')}")
                                model_cols[2].info(f"Provider: {metadata.get('ls_provider', 'N/A')}")
                                
                                # Format temperature nicely
                                if "ls_temperature" in metadata:
                                    temp = metadata["ls_temperature"]
                                    model_cols[3].info(f"Temperature: {temp}")
                            
                            # Invocation parameters if available
                            if "invocation_params" in metadata:
                                st.markdown("##### Invocation Parameters")
                                try:
                                    if isinstance(metadata["invocation_params"], str):
                                        params = json.loads(metadata["invocation_params"])
                                    else:
                                        params = metadata["invocation_params"]
                                    
                                    st.json(params)
                                except:
                                    st.code(str(metadata["invocation_params"]))
                
                # Extract token information
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
            
            # Handle output data
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
                    
                    # Extract evaluations
                    evals = result.get("evaluations", {})
                    if evals:
                        evaluations = evals
                    
                    # Update results in real-time
                    if generated_names:
                        display_results(generated_names, evaluations, container)
                
                # Extract any completion tokens from output event
                if "metadata" in data and "completion_tokens" in data["metadata"]:
                    token_counts["completion"] = data["metadata"]["completion_tokens"]
                    token_counts["total"] = token_counts.get("prompt", 0) + token_counts["completion"]
                    tokens_display.metric("Tokens", f"{token_counts['total']} ({token_counts['completion']} completion)")
        
        except Exception as e:
            with debug_container:
                st.error(f"Error processing stream: {str(e)}")
    
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
        
        # Display LangGraph summary
        if langgraph_data["nodes_visited"]:
            st.markdown("### LangGraph Flow Summary")
            summary_cols = st.columns(2)
            
            with summary_cols[0]:
                st.markdown("**Nodes Visited:**")
                for node in langgraph_data["nodes_visited"]:
                    st.markdown(f"- {node}")
            
            with summary_cols[1]:
                st.markdown("**Triggers:**")
                for trigger in langgraph_data["triggers"]:
                    st.markdown(f"- {trigger}")
            
            if langgraph_data["model_calls"]:
                st.markdown("**Models Used:**")
                model_table_data = []
                for model in langgraph_data["model_calls"]:
                    model_table_data.append({
                        "Model": model["name"],
                        "Type": model["type"],
                        "Provider": model["provider"],
                        "Temperature": model["temperature"]
                    })
                
                if model_table_data:
                    st.dataframe(pd.DataFrame(model_table_data))
        
        # Display step timing breakdown
        if steps_completed:
            st.subheader("Step Execution Breakdown")
            
            # Create a DataFrame for visualization
            step_data = []
            for idx, step in enumerate(steps_completed):
                step_name = step["name"]
                if step.get("node"):
                    step_name = f"{step_name} ({step['node']})"
                
                step_data.append({
                    "Step": f"{idx+1}. {step_name}",
                    "Time (seconds)": step["time"],
                    "Percentage": (step["time"] / total_time) * 100 if total_time > 0 else 0
                })
            
            step_df = pd.DataFrame(step_data)
            
            # Display as a bar chart
            chart = alt.Chart(step_df).mark_bar().encode(
                x=alt.X('Time (seconds):Q', title='Time (seconds)'),
                y=alt.Y('Step:N', sort=None, title=None),
                tooltip=['Step', 'Time (seconds)', 'Percentage']
            ).properties(
                height=len(steps_completed) * 40 + 50
            )
            
            st.altair_chart(chart, use_container_width=True)
    
    return generated_names, evaluations

def display_results(generated_names, evaluations, container):
    """Helper function to display generated names and evaluations"""
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
    
    # Results area - modify the order and structure
    main_content = st.container()
    with main_content:
        results_container = st.container()
    
    # Progress indicators
    progress_bar = st.progress(0)
    status_container = st.container()
    
    # Process generation
    if generate_button:
        if not user_input.strip():
            st.error("Please provide a description of your brand requirements.")
            st.stop()
            
        # Display initial status
        status_container.info("Initializing generation process...")
        
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
                                        st.rerun()
                                else:
                                    if st.button("🤍", key=f"h_fav_{i}_{name}"):
                                        add_to_favorites(name)
                                        st.rerun()
                    
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
                                        st.rerun()
                
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