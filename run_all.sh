#!/bin/bash
# ─── A2A Multi-Agent System Launcher ─────────────────────────────────
# Starts all three agents and the Streamlit UI

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Clear Python bytecode cache to ensure latest code runs
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔═══════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   🤖 A2A Multi-Agent System Launcher             ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════╝${NC}"
echo ""

# Check .env file
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠️  No .env file found. Copying from .env.example...${NC}"
    cp .env.example .env
    echo -e "${YELLOW}   Please edit .env with your Azure OpenAI credentials.${NC}"
    exit 1
fi

# Cleanup function
cleanup() {
    echo ""
    echo -e "${YELLOW}🛑 Shutting down all services...${NC}"
    kill $ARITH_PID $LOGIC_PID $UI_PID 2>/dev/null || true
    wait $ARITH_PID $LOGIC_PID $UI_PID 2>/dev/null || true
    echo -e "${GREEN}✅ All services stopped.${NC}"
}
trap cleanup EXIT INT TERM

# Start Arithmetic Agent
echo -e "${GREEN}🧮 Starting Arithmetic Agent (CrewAI) on port 10001...${NC}"
python -m agents.arithmetic --host localhost --port 10001 &
ARITH_PID=$!
sleep 2

# Start Logical Reasoning Agent
echo -e "${PURPLE}🧠 Starting Logical Reasoning Agent (ADK) on port 10002...${NC}"
python -m agents.logical_reasoning --host localhost --port 10002 &
LOGIC_PID=$!
sleep 2

# Start Streamlit UI
echo -e "${BLUE}🌐 Starting Streamlit UI on port 8501...${NC}"
streamlit run ui/app.py --server.port 8501 --server.headless true &
UI_PID=$!
sleep 2

echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   ✅ All services are running!                    ║${NC}"
echo -e "${GREEN}╠═══════════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║   🧮 Arithmetic Agent:  http://localhost:10001    ║${NC}"
echo -e "${GREEN}║   🧠 Logic Agent:       http://localhost:10002    ║${NC}"
echo -e "${GREEN}║   🌐 Streamlit UI:      http://localhost:8501     ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all services.${NC}"

# Wait for background processes
wait
