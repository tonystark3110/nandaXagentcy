"""
Exchange Agent - Version 5.1 with Manual Protocol Override
Added: force_protocol parameter for UI control
"""

import sys
import os

# Load environment variables FIRST (before any other imports)
from dotenv import load_dotenv
load_dotenv()

# Initialize OpenTelemetry BEFORE other imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from src.observability.otel_config import setup_otel
    from src.observability.clickhouse_logger import get_clickhouse_logger
    setup_otel("exchange-agent")
    print("✅ OpenTelemetry configured for exchange-agent")
except Exception as e:
    print(f"⚠️  Could not setup observability: {e}")
    import traceback
    traceback.print_exc()
    print("Continuing without telemetry...")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional, Dict, Any, List, Literal
import logging
import time
import uuid
import json
import asyncio
import random
import re

# Add parent directory to Python path for imports
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Try relative imports first, fall back to absolute
try:
    from .mcp_client import MCPClient
    from .stategraph_orchestrator import StateGraphOrchestrator  
except ImportError:
    from mcp_client import MCPClient
    from stategraph_orchestrator import StateGraphOrchestrator  

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Verify API key is loaded
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.error("=" * 60)
    logger.error("❌ OPENAI_API_KEY not found in environment!")
    logger.error("=" * 60)
    logger.error("Please ensure .env file exists in project root with:")
    logger.error("  OPENAI_API_KEY=sk-...")
    logger.error("=" * 60)
    sys.exit(1)
else:
    logger.info(f"✓ OpenAI API key loaded (ends with: ...{api_key[-4:]})")

# Initialize OpenAI client
from openai import OpenAI
openai_client = OpenAI(api_key=api_key)

# Global instances
mcp_client: Optional[MCPClient] = None
stategraph_orchestrator: Optional[StateGraphOrchestrator] = None
clickhouse_logger = None

# Tracer for OpenTelemetry
try:
    from opentelemetry import trace
    tracer = trace.get_tracer(__name__)
    logger.info("✅ OpenTelemetry tracer initialized")
except ImportError:
    # Fallback no-op tracer
    class NoOpTracer:
        def start_as_current_span(self, name):
            from contextlib import contextmanager
            @contextmanager
            def _span():
                yield type('obj', (object,), {'set_attribute': lambda *args: None, 'set_status': lambda *args: None, 'record_exception': lambda *args: None})()
            return _span()
    tracer = NoOpTracer()
    logger.warning("⚠️  OpenTelemetry not available, using no-op tracer")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle
    Startup: Initialize MCP client, StateGraph orchestrator, and ClickHouse logger
    Shutdown: Cleanup resources
    """
    global mcp_client, stategraph_orchestrator, clickhouse_logger
    
    # Startup
    logger.info("=" * 60)
    logger.info("Starting Exchange Agent v5.1 - Manual Protocol Override")
    logger.info("=" * 60)
    
    # Initialize ClickHouse Logger
    try:
        clickhouse_logger = get_clickhouse_logger()
        logger.info("✅ ClickHouse logger initialized")
    except Exception as e:
        logger.warning(f"⚠️  ClickHouse logger initialization failed: {e}")
        clickhouse_logger = None
    
    # Initialize StateGraph Orchestrator (for A2A path)
    try:
        stategraph_orchestrator = StateGraphOrchestrator()
        logger.info("✅ StateGraph Orchestrator initialized")
        
        # Validate registry connectivity and agent discovery
        logger.info("🔍 Validating registry connectivity...")
        await stategraph_orchestrator.startup_validation()
        logger.info("✅ Registry validation passed - A2A path ready")
        
    except RuntimeError as e:
        logger.error(f"❌ Registry validation failed: {e}")
        logger.error("A2A path unavailable - agents not discoverable")
        stategraph_orchestrator = None
    except Exception as e:
        logger.error(f"❌ StateGraph Orchestrator initialization failed: {e}")
        logger.exception(e)
        stategraph_orchestrator = None
    
    # Initialize MCP Client (for fast path)
    try:
        mcp_client = MCPClient()
        await mcp_client.initialize()
        logger.info("✅ MCP Client initialized - Fast path available")
    except Exception as e:
        logger.warning(f"⚠️  MCP Client initialization failed: {e}")
        logger.warning("Falling back to A2A agents only")
        mcp_client = None
    
    logger.info("=" * 60)
    
    yield
    
    # Shutdown
    logger.info("Shutting down Exchange Agent...")
    if mcp_client:
        await mcp_client.cleanup()
    logger.info("✓ Shutdown complete")


# Create FastAPI app with lifespan
app = FastAPI(
    title="MBTA Exchange Agent",
    description="Hybrid A2A + MCP with Manual Protocol Override",
    version="5.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# AUTO-INSTRUMENTATION - Automatically trace HTTP requests/responses
# ============================================================================
try:
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    
    # Auto-instrument FastAPI (all endpoints)
    FastAPIInstrumentor.instrument_app(app)
    logger.info("✅ FastAPI auto-instrumentation enabled")
    
    # Auto-instrument HTTPX (HTTP client for A2A calls)
    HTTPXClientInstrumentor().instrument()
    logger.info("✅ HTTPX auto-instrumentation enabled")
except Exception as e:
    logger.warning(f"⚠️  Auto-instrumentation failed: {e}")


# Request/Response models
class ChatRequest(BaseModel):
    query: str
    user_id: Optional[str] = "default_user"
    conversation_id: Optional[str] = None
    force_protocol: Optional[Literal["auto", "mcp", "a2a"]] = "auto"  # NEW


class ChatResponse(BaseModel):
    response: str
    path: str  # "mcp", "a2a", "shortcut", or "a2a_fallback"
    latency_ms: int
    intent: str
    confidence: float
    metadata: Optional[Dict[str, Any]] = None


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "MBTA Exchange Agent",
        "version": "5.1.0",
        "architecture": "Hybrid A2A + MCP with Manual Protocol Override",
        "routing_logic": "GPT-4o-mini semantic classification with manual override",
        "features": ["llm_routing", "domain_analysis", "multi_agent_orchestration", "manual_override"],
        "optimization": "Semantic understanding of query intent",
        "mcp_available": mcp_client is not None and mcp_client._initialized,
        "stategraph_available": stategraph_orchestrator is not None,
        "clickhouse_available": clickhouse_logger is not None,
        "status": "healthy"
    }


# ============================================================
# STEP 0: SHORTCUT PATH DETECTION (NO LLM CALL)
# ============================================================

def is_greeting_or_simple_query(query: str) -> bool:
    """Fast pattern matching to detect greetings and simple queries"""
    query_lower = query.lower().strip()
    
    # Very short queries only (less than 10 words)
    word_count = len(query_lower.split())
    if word_count > 10:
        return False
    
    # Only match pure greetings
    greeting_patterns = [
        'hi', 'hello', 'hey', 'greetings', 'good morning',
        'good afternoon', 'good evening', 'howdy', 'sup', 'yo'
    ]
    
    if any(query_lower == greeting or query_lower.startswith(greeting + " ") 
           for greeting in greeting_patterns):
        return True
    
    return False


def get_shortcut_response(query: str) -> str:
    """Generate response for shortcut path queries (NO LLM NEEDED)"""
    query_lower = query.lower().strip()
    
    greeting_keywords = ['hi', 'hello', 'hey', 'greetings']
    if any(keyword in query_lower for keyword in greeting_keywords):
        responses = [
            "Hello! I'm MBTA Agentcy. Ask about service alerts, routes, or stations!",
            "Hi! I can help with Boston MBTA transit info.",
        ]
        return random.choice(responses)
    
    return "I'm specialized in Boston MBTA transit..."


# ============================================================
# INTELLIGENT EXPERTISE-BASED ROUTING
# ============================================================

def needs_domain_expertise(query: str) -> tuple[bool, str, List[str]]:
    """
    Detect if query needs domain expertise beyond API data.
    """
    
    query_lower = query.lower()
    detected_patterns = []
    
    # CROWDING keywords
    CROWDING = [
        "crowded", "crowd", "busy", "full", "packed", "space",
        "capacity", "occupancy", "how full", "standing room",
        "seats available", "room on"
    ]
    if any(kw in query_lower for kw in CROWDING):
        detected_patterns.append("crowding_analysis")
        return True, "Query requires crowding analysis with domain expertise", detected_patterns
    
    # PREDICTIVE keywords
    PREDICTIVE = ["should i wait", "worth waiting", "how long will", "when will"]
    if any(kw in query_lower for kw in PREDICTIVE):
        detected_patterns.append("predictive")
        return True, "Query requires predictive analysis", detected_patterns
    
    # DECISION SUPPORT keywords
    DECISION = ["should i", "recommend", "suggest", "better to", "what should i do"]
    if any(kw in query_lower for kw in DECISION):
        detected_patterns.append("decision_support")
        return True, "Query needs decision support", detected_patterns
    
    # CONDITIONAL keywords
    CONDITIONAL = ["if there are", "considering", "depending on"]
    if any(kw in query_lower for kw in CONDITIONAL):
        detected_patterns.append("conditional")
        return True, "Query has conditional logic", detected_patterns
    
    # ANALYTICAL keywords
    ANALYTICAL = ["why", "explain", "what caused", "how serious"]
    if any(kw in query_lower for kw in ANALYTICAL):
        detected_patterns.append("analytical")
        return True, "Query needs analytical interpretation", detected_patterns
    
    # ROUTING pattern
    if re.search(r"from .+ to .+", query_lower):
        detected_patterns.append("routing")
        return True, "Query requires multi-agent coordination", detected_patterns
    
    # DEFAULT: Simple fact lookup
    return False, "Simple fact lookup - MCP can handle", detected_patterns


# ============================================================================
# UNIFIED CLASSIFICATION + ROUTING + TOOL SELECTION
# ============================================================================

async def classify_route_and_select_tool(query: str, available_tools: List[Dict], force_protocol: str = "auto") -> Dict:
    """
    OPTIMIZED ROUTING with manual protocol override support
    
    NEW in v5.1: force_protocol parameter
    - "auto" - intelligent routing (default)
    - "mcp" - force MCP path if available
    - "a2a" - force A2A path
    """
    
    with tracer.start_as_current_span("classify_route_and_select_tool") as span:
        span.set_attribute("query", query)
        span.set_attribute("force_protocol", force_protocol)
        span.set_attribute("query_length", len(query))
        span.set_attribute("available_tools_count", len(available_tools))
        
        # ================================================================
        # MANUAL OVERRIDE CHECK
        # ================================================================
        if force_protocol == "mcp":
            logger.info("🔧 MANUAL OVERRIDE: Forcing MCP path")
            
            # Still need to classify intent for MCP tool selection
            system_prompt = f"""You are an MBTA query classifier.

Classify the intent and select the appropriate MCP tool.

Available MCP Tools:
{chr(10).join([f"  • {tool['name']}: {tool['description']}" for tool in available_tools])}

Return JSON:
{{
  "intent": "alerts|stops|trip_planning|general",
  "confidence": 0.95,
  "mcp_tool": "mbta_get_alerts",
  "mcp_parameters": {{"route_id": "Red"}}
}}"""
            
            try:
                response = await asyncio.to_thread(
                    openai_client.chat.completions.create,
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Query: {query}"}
                    ],
                    temperature=0.3,
                    max_tokens=200
                )
                
                result_text = response.choices[0].message.content.strip()
                if result_text.startswith("```"):
                    result_text = result_text.replace("```json", "").replace("```", "").strip()
                
                decision = json.loads(result_text)
                decision["path"] = "mcp"
                decision["reasoning"] = "Manual override: User selected MCP protocol"
                decision["llm_calls"] = 1
                decision["manual_override"] = True
                
                return decision
                
            except Exception as e:
                logger.error(f"MCP override classification failed: {e}")
                return {
                    "path": "mcp",
                    "intent": "general",
                    "confidence": 0.5,
                    "reasoning": "Manual override with classification error",
                    "manual_override": True,
                    "llm_calls": 1
                }
        
        elif force_protocol == "a2a":
            logger.info("🔧 MANUAL OVERRIDE: Forcing A2A path")
            
            # Simple classification for A2A
            return {
                "path": "a2a",
                "intent": "general",
                "confidence": 1.0,
                "reasoning": "Manual override: User selected A2A protocol",
                "complexity": 0.5,
                "llm_calls": 0,
                "manual_override": True
            }
        
        # ================================================================
        # AUTO MODE - Original intelligent routing
        # ================================================================
        
        # STEP 0: SHORTCUT PATH DETECTION (NO LLM CALL)
        if is_greeting_or_simple_query(query):
            with tracer.start_as_current_span("shortcut_path_detection") as shortcut_span:
                shortcut_span.set_attribute("matched", True)
                
                shortcut_response = get_shortcut_response(query)
                
                decision = {
                    "path": "shortcut",
                    "intent": "greeting",
                    "confidence": 1.0,
                    "reasoning": "Simple greeting detected via pattern matching",
                    "complexity": 0.0,
                    "shortcut_response": shortcut_response,
                    "llm_calls": 0,
                    "manual_override": False
                }
                
                span.set_attribute("routing.path", "shortcut")
                span.set_attribute("llm.calls", 0)
                
                logger.info(f"⚡ SHORTCUT PATH: {decision['reasoning']}")
                
                return decision
        
        # Continue with normal LLM routing...
        # (Rest of the original classify_route_and_select_tool function remains the same)
        
        # Format available tools for the LLM
        tools_list = "\n".join([
            f"  • {tool['name']}: {tool['description']}"
            for tool in available_tools
        ]) if available_tools else "  (No MCP tools available - must use A2A)"
        
        system_prompt = f"""You are an intelligent MBTA query routing system.

**YOUR TASK:** Analyze the query and make ALL routing decisions in one response.

═══════════════════════════════════════════════════════════
STEP 1: CLASSIFY INTENT
═══════════════════════════════════════════════════════════

CRITICAL: Understand what counts as MBTA-related!

**"alerts"** - Anything about MBTA service, delays, or disruptions (CURRENT OR HISTORICAL):
  ✅ Current status: "Red Line delays?", "Any issues now?", "Current service disruptions?"
  ✅ Historical patterns: "How long do medical delays take?", "Typical delay duration?", "Usually how long?"
  ✅ Pattern questions: "Based on past data...", "On average...", "Generally how long...", "Typically..."
  ✅ Crowding: "How crowded?", "Is it busy?", "Room on trains?", "Packed?", "Full trains?"
  ✅ Predictions: "Should I wait?", "Worth waiting?", "How long will this last?", "When will it clear?"
  ✅ Analysis: "How serious?", "Why delays?", "What's causing this?"
  
  PRINCIPLE: If asking about MBTA delays, duration, crowding, patterns, or service status → "alerts"
  
**"stops"** - Station/stop information, finding stations:
  ✅ "Where is Copley?", "Find Harvard station", "Stops on Green Line"
  ✅ "What station is nearest to X?", "List all stations", "Show me Orange Line stops"
  
**"trip_planning"** - Route planning, directions, how to get somewhere:
  ✅ "Route from X to Y", "How do I get to X?", "Best route to Y?"
  ✅ "Park St to Harvard?", "Get me from X to Y", "Directions to MIT?"
  
**"general"** - NOT about MBTA/transit at all (completely off-topic):
  ❌ "What's the weather in Boston?" - Not transit
  ❌ "Who won the Red Sox game?" - Not transit (even though it says "Red")
  ❌ "Boston history facts?" - Not transit
  ❌ "What's 2+2?" - Not transit
  ❌ "Tell me a joke" - Not transit

PRINCIPLE: If query mentions MBTA, trains, T, subway, delays, crowding, stations, routes, or ANY transit topic → NOT "general"!

Only classify as "general" if the query has NOTHING to do with Boston public transit.

═══════════════════════════════════════════════════════════
STEP 2: CHOOSE PATH & SELECT TOOL
═══════════════════════════════════════════════════════════

**Decision Tree:**

Is it MBTA-related?
  ├─ NO → path="a2a", intent="general"
  └─ YES → Does it need analysis/prediction/historical data/expertise?
            ├─ YES → path="a2a" (Domain experts needed)
            │         Examples: "How long do delays take?", "Should I wait?", "How crowded?", "Route from X to Y"
            └─ NO → Can MCP tool provide the answer?
                      ├─ YES → path="mcp" + select tool
                      │         Examples: "Red Line delays RIGHT NOW?", "Next train at Park?"
                      └─ NO → path="a2a"

**MCP Path (Fast, ~400ms):**
- Best for: Current real-time data lookup with single API call
- Examples: "Red Line delays right now?", "Next train at Park St?", "Where are Orange Line trains?"
- NO analysis, NO historical, NO predictions - just current facts

**A2A Path (Domain Experts, ~1500ms):**
- Best for: Requires analysis, expertise, historical data, predictions, or multi-agent coordination
- Examples:
  * "How long do delays usually take?" → Needs historical data from domain expert
  * "Should I wait?" → Needs decision support analysis
  * "How crowded is it?" → Needs crowding analysis
  * "Route from X to Y" → Needs multi-agent coordination
  * "Best route considering delays?" → Needs expert reasoning

PRINCIPLE: 
- Current fact → MCP
- Analysis/Prediction/Historical/Expertise → A2A

═══════════════════════════════════════════════════════════
STEP 3: SELECT MCP TOOL (ONLY IF path="mcp")
═══════════════════════════════════════════════════════════

Available MCP Tools:
{tools_list}

**PARAMETER NAMING (CRITICAL):**
- Use "route_id" NOT "route" (e.g., route_id="Red")
- Use "stop_id" NOT "stop"
- Red Line = "Red", Orange = "Orange", Blue = "Blue", Green = "Green-B"

═══════════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════════

Return ONLY valid JSON (no markdown, no code blocks):

**For A2A path:**
{{
  "intent": "alerts",
  "confidence": 0.95,
  "path": "a2a",
  "reasoning": "Historical MBTA delay pattern question requires domain expertise with historical data",
  "complexity": 0.6
}}

**For MCP path:**
{{
  "intent": "alerts",
  "confidence": 0.95,
  "path": "mcp",
  "reasoning": "Current alert lookup can be answered with direct API call",
  "complexity": 0.2,
  "mcp_tool": "mbta_get_alerts",
  "mcp_parameters": {{"route_id": "Red"}}
}}"""

        user_message = f"""Query: "{query}"

Analyze and provide routing decision."""

        try:
            with tracer.start_as_current_span("llm_unified_routing") as llm_span:
                llm_span.set_attribute("model", "gpt-4o-mini")
                
                response = await asyncio.to_thread(
                    openai_client.chat.completions.create,
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=0.3,
                    max_tokens=300
                )
                
                decision_text = response.choices[0].message.content.strip()
                
                # Remove markdown formatting
                if decision_text.startswith("```json"):
                    decision_text = decision_text.replace("```json", "").replace("```", "").strip()
                elif decision_text.startswith("```"):
                    decision_text = decision_text.replace("```", "").strip()
                
                decision = json.loads(decision_text)
                
                # Validate and set defaults
                decision.setdefault("complexity", 0.5)
                decision.setdefault("confidence", 0.5)
                decision.setdefault("reasoning", "No reasoning provided")
                decision["llm_calls"] = 1
                decision["manual_override"] = False
                
                # Validate MCP path
                if decision["path"] == "mcp":
                    if "mcp_tool" not in decision:
                        logger.warning("MCP selected but no tool - fallback to A2A")
                        decision["path"] = "a2a"
                    elif "mcp_parameters" not in decision:
                        decision["mcp_parameters"] = {}
                
                span.set_attribute("intent", decision['intent'])
                span.set_attribute("confidence", decision['confidence'])
                span.set_attribute("path", decision['path'])
                
                logger.info(f"🧠 LLM Decision:")
                logger.info(f"   Intent: {decision['intent']} ({decision['confidence']:.2f})")
                logger.info(f"   Path: {decision['path']} (complexity: {decision['complexity']:.2f})")
                logger.info(f"   Reasoning: {decision['reasoning']}")
                
                return decision
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            return {
                "intent": "general",
                "confidence": 0.3,
                "path": "a2a",
                "reasoning": f"JSON error: {str(e)}",
                "complexity": 0.5,
                "llm_calls": 1,
                "manual_override": False
            }
        except Exception as e:
            logger.error(f"Routing failed: {e}", exc_info=True)
            return {
                "intent": "general",
                "confidence": 0.3,
                "path": "a2a",
                "reasoning": f"Error: {str(e)}",
                "complexity": 0.5,
                "llm_calls": 1,
                "manual_override": False
            }


# ============================================================================
# MAIN CHAT ENDPOINT (with manual override support)
# ============================================================================

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint with manual protocol override
    
    NEW in v5.1: force_protocol parameter
    - "auto" - intelligent routing (default)
    - "mcp" - force MCP if available, fallback to A2A
    - "a2a" - force A2A path
    """
    
    with tracer.start_as_current_span("chat_endpoint") as root_span:
        start_time = time.time()
        query = request.query
        conversation_id = request.conversation_id or str(uuid4())
        force_protocol = request.force_protocol or "auto"
        
        root_span.set_attribute("query", query)
        root_span.set_attribute("conversation_id", conversation_id)
        root_span.set_attribute("user_id", request.user_id)
        root_span.set_attribute("force_protocol", force_protocol)
        
        if not query or not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        logger.info("=" * 80)
        logger.info(f"📨 Received query: {query}")
        logger.info(f"   Conversation ID: {conversation_id}")
        logger.info(f"   Force Protocol: {force_protocol}")
        
        # Get available MCP tools
        available_tools = []
        if mcp_client and mcp_client._initialized:
            if hasattr(mcp_client, '_available_tools') and mcp_client._available_tools:
                for tool in mcp_client._available_tools:
                    available_tools.append({
                        "name": tool.name,
                        "description": tool.description or ""
                    })
                logger.info(f"📋 {len(available_tools)} MCP tools available")
        
        # ====================================================================
        # ROUTING WITH MANUAL OVERRIDE SUPPORT
        # ====================================================================
        
        with tracer.start_as_current_span("routing_with_override") as routing_span:
            # Check expertise needs
            needs_expertise, expertise_reasoning, detected_patterns = needs_domain_expertise(query)
            
            routing_span.set_attribute("needs_expertise", needs_expertise)
            routing_span.set_attribute("reasoning", expertise_reasoning)
            routing_span.set_attribute("detected_patterns", str(detected_patterns))
            
            # Classify with override support
            decision = await classify_route_and_select_tool(query, available_tools, force_protocol)
            
            # Override path based on expertise (unless manually forced)
            if needs_expertise and not decision.get("manual_override"):
                original_path = decision["path"]
                decision["path"] = "a2a"
                decision["reasoning"] = f"EXPERTISE REQUIRED: {expertise_reasoning}"
                
                if original_path != "a2a":
                    logger.info(f"   ✓ OVERRIDE: {original_path} → a2a (expertise needed)")
            
            # Handle manual override when protocol not available
            if decision.get("manual_override"):
                if force_protocol == "mcp" and (not mcp_client or not mcp_client._initialized):
                    logger.warning("MCP forced but unavailable - fallback to A2A")
                    decision["path"] = "a2a"
                    decision["reasoning"] = "Manual override requested MCP but unavailable - using A2A"
            
            intent = decision["intent"]
            confidence = decision["confidence"]
            chosen_path = decision["path"]
        
        # Log to ClickHouse
        if clickhouse_logger:
            try:
                clickhouse_logger.log_conversation(
                    conversation_id=conversation_id,
                    user_id=request.user_id,
                    role="user",
                    content=query,
                    intent=intent,
                    routed_to_orchestrator=(chosen_path == "a2a"),
                    metadata={
                        "confidence": confidence,
                        "complexity": decision.get('complexity', 0.5),
                        "reasoning": decision['reasoning'],
                        "path": chosen_path,
                        "force_protocol": force_protocol,
                        "manual_override": decision.get("manual_override", False)
                    }
                )
            except Exception as e:
                logger.warning(f"ClickHouse logging failed: {e}")
        
        # ====================================================================
        # EXECUTE CHOSEN PATH
        # ====================================================================
        
        response_text = ""
        path_taken = ""
        metadata = {
            "unified_decision": {
                "intent": intent,
                "confidence": confidence,
                "path": chosen_path,
                "reasoning": decision["reasoning"],
                "complexity": decision.get("complexity", 0.5),
                "llm_calls": decision.get("llm_calls", 0),
                "manual_override": decision.get("manual_override", False),
                "force_protocol": force_protocol
            }
        }
        
        if chosen_path == "shortcut":
            # SHORTCUT PATH
            with tracer.start_as_current_span("handle_shortcut_path"):
                response_text = decision["shortcut_response"]
                path_taken = "shortcut"
                
                metadata["shortcut_execution"] = {
                    "method": "pattern_matching",
                    "llm_calls": 0,
                    "cost_usd": 0.0
                }
                
                logger.info(f"⚡ SHORTCUT PATH executed")
        
        elif chosen_path == "mcp" and mcp_client and mcp_client._initialized:
            # MCP FAST PATH
            tool_name = decision.get('mcp_tool')
            tool_params = decision.get('mcp_parameters', {})
            
            if not tool_name:
                logger.warning("MCP path but no tool - fallback to A2A")
                response_text, a2a_metadata = await handle_a2a_path(query, conversation_id)
                path_taken = "a2a_fallback"
                metadata.update(a2a_metadata)
                metadata["fallback_reason"] = "MCP tool not specified"
            else:
                logger.info(f"🚀 MCP Fast Path:")
                logger.info(f"   Tool: {tool_name}")
                logger.info(f"   Parameters: {tool_params}")
                
                try:
                    tool_result = await call_mcp_tool_dynamic(tool_name, tool_params)
                    
                    metadata["mcp_execution"] = {
                        "tool": tool_name,
                        "parameters": tool_params,
                        "success": True
                    }
                    
                    response_text = await synthesize_mcp_response_with_llm(query, tool_name, tool_result)
                    
                    path_taken = "mcp"
                    logger.info(f"✅ MCP execution successful")
                    
                except Exception as e:
                    logger.error(f"❌ MCP execution failed: {e}")
                    root_span.record_exception(e)
                    
                    logger.info("↪️  Falling back to A2A path")
                    response_text, a2a_metadata = await handle_a2a_path(query, conversation_id)
                    path_taken = "a2a_fallback"
                    metadata.update(a2a_metadata)
                    metadata["mcp_error"] = str(e)
        
        elif chosen_path == "a2a":
            # A2A MULTI AGENT PATH
            logger.info(f"🔄 A2A Path: {decision['reasoning']}")
            
            response_text, a2a_metadata = await handle_a2a_path(query, conversation_id)
            path_taken = "a2a"
            metadata.update(a2a_metadata)
        
        else:
            # MCP selected but not available - fallback
            logger.warning("MCP selected but unavailable - fallback to A2A")
            response_text, a2a_metadata = await handle_a2a_path(query, conversation_id)
            path_taken = "a2a_fallback"
            metadata.update(a2a_metadata)
            metadata["fallback_reason"] = "MCP unavailable"
        
        # Calculate latency
        latency_ms = int((time.time() - start_time) * 1000)
        
        root_span.set_attribute("path_taken", path_taken)
        root_span.set_attribute("latency_ms", latency_ms)
        
        logger.info(f"✅ Response via {path_taken} in {latency_ms}ms")
        logger.info("=" * 80)
        
        # Log response to ClickHouse
        if clickhouse_logger:
            try:
                clickhouse_logger.log_conversation(
                    conversation_id=conversation_id,
                    user_id=request.user_id,
                    role="assistant",
                    content=response_text[:1000],
                    intent=intent,
                    routed_to_orchestrator=(path_taken in ["a2a", "a2a_fallback"]),
                    metadata={
                        "path": path_taken,
                        "latency_ms": latency_ms,
                        "confidence": confidence,
                        "force_protocol": force_protocol,
                        "manual_override": decision.get("manual_override", False)
                    }
                )
            except Exception as e:
                logger.warning(f"ClickHouse logging failed: {e}")
        
        return ChatResponse(
            response=response_text,
            path=path_taken,
            latency_ms=latency_ms,
            intent=intent,
            confidence=confidence,
            metadata=metadata
        )


# ============================================================================
# MCP TOOL EXECUTION (DYNAMIC DISPATCH)
# ============================================================================

async def call_mcp_tool_dynamic(tool_name: str, parameters: Dict) -> Dict[str, Any]:
    """Dynamically call any MCP tool"""
    
    with tracer.start_as_current_span("call_mcp_tool_dynamic") as span:
        span.set_attribute("tool_name", tool_name)
        span.set_attribute("parameters", json.dumps(parameters))
        
        tool_method_map = {
            "mbta_get_alerts": mcp_client.get_alerts,
            "mbta_get_routes": mcp_client.get_routes,
            "mbta_get_stops": mcp_client.get_stops,
            "mbta_search_stops": mcp_client.search_stops,
            "mbta_get_predictions": mcp_client.get_predictions,
            "mbta_get_predictions_for_stop": mcp_client.get_predictions_for_stop,
            "mbta_get_schedules": mcp_client.get_schedules,
            "mbta_get_trips": mcp_client.get_trips,
            "mbta_get_vehicles": mcp_client.get_vehicles,
            "mbta_get_nearby_stops": mcp_client.get_nearby_stops,
            "mbta_plan_trip": mcp_client.plan_trip,
            "mbta_list_all_routes": mcp_client.list_all_routes,
            "mbta_list_all_stops": mcp_client.list_all_stops,
            "mbta_list_all_alerts": mcp_client.list_all_alerts,
        }
        
        if tool_name not in tool_method_map:
            raise ValueError(f"Unknown MCP tool: {tool_name}")
        
        method = tool_method_map[tool_name]
        
        logger.info(f"🔧 Calling {tool_name} with params: {parameters}")
        result = await method(**parameters)
        span.set_attribute("success", True)
        logger.info(f"✓ Tool execution successful")
        
        return result


# ============================================================================
# RESPONSE SYNTHESIS
# ============================================================================

async def synthesize_mcp_response_with_llm(query: str, tool_name: str, tool_result: Dict) -> str:
    """Convert MCP JSON response into natural language"""
    
    system_prompt = """You are a helpful MBTA transit assistant.

Convert the technical API response into a natural, conversational answer.

Be concise but informative. Use natural language, not technical jargon."""

    tool_result_str = json.dumps(tool_result, indent=2)
    if len(tool_result_str) > 4000:
        tool_result_str = tool_result_str[:4000] + "\n... (truncated)"
    
    user_message = f"""User Query: "{query}"

Tool Used: {tool_name}

API Response:
{tool_result_str}

Convert to natural answer."""

    try:
        with tracer.start_as_current_span("synthesize_response"):
            response = await asyncio.to_thread(
                openai_client.chat.completions.create,
                model="gpt-4o-mini",
                temperature=0.7,
                max_tokens=500,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ]
            )
            
            return response.choices[0].message.content.strip()
    
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        return f"I found information but had trouble formatting it: {str(tool_result)[:200]}..."


# ============================================================================
# A2A PATH HANDLER
# ============================================================================

async def handle_a2a_path(query: str, conversation_id: str) -> tuple[str, Dict[str, Any]]:
    """Handle query using A2A agents with domain expertise"""
    
    with tracer.start_as_current_span("handle_a2a_path") as span:
        span.set_attribute("query", query)
        span.set_attribute("conversation_id", conversation_id)
        
        if not stategraph_orchestrator:
            logger.error("StateGraph unavailable")
            return (
                "I'm having trouble processing your request. Please try again.",
                {"error": "StateGraph unavailable"}
            )
        
        try:
            logger.info(f"🔄 Running StateGraph orchestration")
            
            result = await stategraph_orchestrator.process_message(query, conversation_id)
            
            response_text = result.get("response", "")
            
            metadata = {
                "stategraph_intent": result.get("intent"),
                "stategraph_confidence": result.get("confidence"),
                "agents_called": result.get("agents_called", []),
                "graph_execution": result.get("metadata", {}).get("graph_execution", "completed")
            }
            
            span.set_attribute("agents_called", json.dumps(metadata['agents_called']))
            span.set_attribute("agents_count", len(metadata['agents_called']))
            
            logger.info(f"✓ StateGraph completed")
            logger.info(f"   Agents: {', '.join(metadata['agents_called'])}")
            
            return response_text, metadata
        
        except Exception as e:
            logger.error(f"A2A error: {e}", exc_info=True)
            span.record_exception(e)
            return (f"Error: {str(e)}", {"error": str(e)})


# ============================================================================
# HEALTH & METRICS
# ============================================================================

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "components": {
            "mcp_client": {
                "available": mcp_client is not None,
                "initialized": mcp_client._initialized if mcp_client else False,
                "tools_count": len(mcp_client._available_tools) if mcp_client and hasattr(mcp_client, '_available_tools') else 0
            },
            "stategraph": {
                "available": stategraph_orchestrator is not None
            },
            "clickhouse": {
                "available": clickhouse_logger is not None
            },
            "routing": {
                "method": "intelligent_with_manual_override",
                "version": "5.1"
            }
        }
    }


@app.get("/metrics")
async def get_metrics():
    """Metrics endpoint"""
    tools_available = []
    if mcp_client and hasattr(mcp_client, '_available_tools'):
        tools_available = [tool.name for tool in mcp_client._available_tools]
    
    return {
        "mcp_tools_available": len(tools_available),
        "mcp_tools": tools_available,
        "stategraph_available": stategraph_orchestrator is not None,
        "version": "5.1.0",
        "routing_method": "intelligent_with_manual_override",
        "features": {
            "auto_routing": True,
            "manual_mcp_override": True,
            "manual_a2a_override": True,
            "shortcut_path": True,
            "domain_expertise_detection": True
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    logger.info("=" * 80)
    logger.info("🚀 Starting MBTA Exchange Agent Server")
    logger.info("   Version: 5.1.0")
    logger.info("   Routing: Intelligent with Manual Protocol Override")
    logger.info("   Features: Auto routing + UI control buttons")
    logger.info("=" * 80)
    
    uvicorn.run(app, host="0.0.0.0", port=8100)
