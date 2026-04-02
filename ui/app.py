"""Streamlit UI for the A2A Multi-Agent Orchestrator.

Provides a chat interface that routes queries through the LangGraph
orchestrator to remote Arithmetic and Logic agents via A2A protocol.
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from agents.orchestrator.orchestrator import Orchestrator
from agents.orchestrator.a2a_tools import discover_agent

# ─── Page Config ─────────────────────────────────────────────────────

st.set_page_config(
    page_title="A2A Multi-Agent Orchestrator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }

    .main-header h1 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    .main-header p {
        margin: 0.3rem 0 0 0;
        font-size: 0.95rem;
        opacity: 0.9;
    }

    .agent-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        border: 1px solid rgba(255,255,255,0.08);
        transition: transform 0.2s, box-shadow 0.2s;
    }

    .agent-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }

    .agent-name {
        font-weight: 600;
        font-size: 0.95rem;
        color: #e2e8f0;
        margin-bottom: 0.25rem;
    }

    .agent-framework {
        font-size: 0.8rem;
        color: #94a3b8;
        margin-bottom: 0.5rem;
    }

    .status-connected {
        color: #4ade80;
        font-weight: 500;
        font-size: 0.85rem;
    }

    .status-disconnected {
        color: #f87171;
        font-weight: 500;
        font-size: 0.85rem;
    }

    .routing-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        letter-spacing: 0.03em;
    }

    .badge-arithmetic {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: #1a1a2e;
    }

    .badge-logical {
        background: linear-gradient(135deg, #8b5cf6, #7c3aed);
        color: white;
    }

    .badge-both {
        background: linear-gradient(135deg, #f59e0b, #7c3aed);
        color: white;
    }

    .badge-general {
        background: linear-gradient(135deg, #6b7280, #4b5563);
        color: white;
    }

    .chat-message-agent {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin: 0.5rem 0;
    }

    .examples-section {
        background: rgba(102, 126, 234, 0.08);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 12px;
        padding: 1rem;
        margin-top: 1rem;
    }

    .example-chip {
        display: inline-block;
        background: rgba(102, 126, 234, 0.15);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 20px;
        padding: 0.3rem 0.8rem;
        margin: 0.2rem;
        font-size: 0.8rem;
        color: #cbd5e1;
        cursor: pointer;
        transition: all 0.2s;
    }

    .sidebar-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(102, 126, 234, 0.3);
    }

    .protocol-info {
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.2);
        border-radius: 10px;
        padding: 0.8rem;
        margin-top: 1rem;
        font-size: 0.82rem;
        color: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)


# ─── Helper Functions ────────────────────────────────────────────────


def get_orchestrator() -> Orchestrator:
    """Get or create orchestrator instance."""
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = Orchestrator()
    return st.session_state.orchestrator


def get_badge_class(classification: str) -> str:
    """Get CSS class for routing badge."""
    if classification == "arithmetic":
        return "badge-arithmetic"
    elif classification == "logical":
        return "badge-logical"
    elif classification == "both":
        return "badge-both"
    return "badge-general"


def get_badge_label(classification: str) -> str:
    """Get label for routing badge."""
    labels = {
        "arithmetic": "🧮 Arithmetic Agent",
        "logical": "🧠 Logic Agent",
        "both": "🧮🧠 Both Agents",
        "general": "💬 General",
    }
    return labels.get(classification, "💬 General")


async def check_agents():
    """Check agent connectivity."""
    orchestrator = get_orchestrator()
    return await orchestrator.check_agent_status()


async def run_query(query: str) -> dict:
    """Run a query through the orchestrator."""
    orchestrator = get_orchestrator()
    return await orchestrator.run(query)


# ─── Sidebar ─────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="sidebar-title">🔗 Agent Network Status</div>', unsafe_allow_html=True)

    if st.button("🔄 Refresh Status", use_container_width=True):
        try:
            status = asyncio.run(check_agents())
            st.session_state.agent_status = status
        except Exception as e:
            st.error(f"Error checking agents: {e}")

    # Display agent status
    if "agent_status" not in st.session_state:
        try:
            st.session_state.agent_status = asyncio.run(check_agents())
        except Exception:
            st.session_state.agent_status = {
                "arithmetic": {"connected": False, "name": None, "url": "http://localhost:10001"},
                "logical": {"connected": False, "name": None, "url": "http://localhost:10002"},
            }

    status = st.session_state.agent_status

    # Arithmetic Agent Card
    # This code is done RKG
    arith = status["arithmetic"]
    arith_status = "connected" if arith["connected"] else "disconnected"
    arith_icon = "🟢" if arith["connected"] else "🔴"
    st.markdown(f"""
    <div class="agent-card">
        <div class="agent-name">🧮 Arithmetic Agent</div>
        <div class="agent-framework">Framework: CrewAI · Port: 10001</div>
        <div class="status-{arith_status}">{arith_icon} {arith_status.capitalize()}</div>
    </div>
    """, unsafe_allow_html=True)

    # Logic Agent Card
    logic = status["logical"]
    logic_status = "connected" if logic["connected"] else "disconnected"
    logic_icon = "🟢" if logic["connected"] else "🔴"
    st.markdown(f"""
    <div class="agent-card">
        <div class="agent-name">🧠 Logical Reasoning Agent</div>
        <div class="agent-framework">Framework: Google ADK · Port: 10002</div>
        <div class="status-{logic_status}">{logic_icon} {logic_status.capitalize()}</div>
    </div>
    """, unsafe_allow_html=True)

    # Protocol info
    st.markdown("""
    <div class="protocol-info">
        <strong>🔌 A2A Protocol</strong><br/>
        Agents communicate via the Agent-to-Agent protocol using JSON-RPC over HTTP.
        Each agent publishes an Agent Card at <code>/.well-known/agent.json</code>.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Architecture**")
    st.markdown("""
    ```
    Orchestrator (LangGraph)
    ├→ Arithmetic (CrewAI)
    ├→ Logic (Google ADK)
    └→ Both (parallel)
    ```
    """)

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ─── Main Content ────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>🤖 A2A Multi-Agent Orchestrator</h1>
    <p>Powered by LangGraph · CrewAI · Google ADK · Agent-to-Agent Protocol</p>
</div>
""", unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and "metadata" in msg:
            meta = msg["metadata"]
            badge_class = get_badge_class(meta.get("classification", "general"))
            badge_label = get_badge_label(meta.get("classification", "general"))
            st.markdown(
                f'<span class="routing-badge {badge_class}">{badge_label}</span>',
                unsafe_allow_html=True,
            )
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask an arithmetic or logical reasoning question..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response
    with st.chat_message("assistant"):
        with st.spinner("🔄 Routing and processing..."):
            try:
                result = asyncio.run(run_query(prompt))

                badge_class = get_badge_class(result["classification"])
                badge_label = get_badge_label(result["classification"])

                st.markdown(
                    f'<span class="routing-badge {badge_class}">{badge_label}</span>',
                    unsafe_allow_html=True,
                )
                st.markdown(result["response"])

                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["response"],
                    "metadata": {
                        "classification": result["classification"],
                        "agent_used": result["agent_used"],
                    },
                })
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                })

# Show examples if no messages
if not st.session_state.messages:
    st.markdown("""
    <div class="examples-section">
        <strong>💡 Try these examples:</strong><br/><br/>
        <span class="example-chip">🧮 What is 25 * 4 + 10?</span>
        <span class="example-chip">🧮 Calculate 2^10</span>
        <span class="example-chip">🧮 What is 17 modulo 5?</span>
        <br/>
        <span class="example-chip">🧠 What is True AND False?</span>
        <span class="example-chip">🧠 If P→Q and P is True, what is Q?</span>
        <span class="example-chip">🧠 Evaluate: NOT (True OR False)</span>
        <br/>
        <span class="example-chip">🧮🧠 What is 5*3 and is the result greater than 10? Verify with AND logic</span>
    </div>
    """, unsafe_allow_html=True)
