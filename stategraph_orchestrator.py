"""
StateGraph Orchestrator - OPTIMIZED v4.3
Skips StopFinder for known station names, only resolves actual landmarks
Version: 4.3 OPTIMIZED
"""

import os
from typing import TypedDict, Annotated, Sequence, Literal, Dict, List, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import operator
from dataclasses import dataclass
import asyncio
import httpx
from opentelemetry import trace
import logging
from urllib.parse import urlparse
from datetime import datetime, timedelta
import json
import re

# Import SLIM client
try:
    from .slim_client import SlimAgentClient
    SLIM_AVAILABLE = True
except ImportError:
    SLIM_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("SLIM not available")

tracer = trace.get_tracer(__name__)
logger = logging.getLogger(__name__)

# Configuration
REGISTRY_URL = os.getenv("REGISTRY_URL", "http://97.107.132.213:6900")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Caching
_agent_catalog_cache = None
_catalog_cache_time = None
_catalog_cache_ttl = timedelta(minutes=5)
_current_orchestrator = None


# ============================================================================
# STATE
# ============================================================================

class AgentState(TypedDict):
    user_message: str
    conversation_id: str
    intent: str
    confidence: float
    matched_agents: List[str]
    agent_queries: Dict[str, str]
    llm_matching_decision: Dict[str, Any]
    messages: Annotated[Sequence[BaseMessage], operator.add]
    agents_called: List[str]
    agent_responses: List[Dict[str, Any]]
    
    # Parsed values
    origin_text: str
    destination_text: str
    
    # Resolved station names
    resolved_origin: str
    resolved_destination: str
    
    # Disruption tracking
    has_disruptions: bool
    affected_routes: List[str]
    severity_level: str
    
    final_response: str
    should_end: bool
    routing_decision: str


@dataclass
class AgentConfig:
    name: str
    url: str
    port: int
    description: str
    capabilities: List[str]
    discovered_from_registry: bool = True


# ============================================================================
# QUERY PARSING
# ============================================================================

def extract_origin_destination(query: str) -> Dict[str, str]:
    """Extract origin and destination from natural language"""
    q = query.lower()
    result = {"origin": "", "destination": ""}
    
    # Pattern 1: "I am at X and ... to Y"
    match = re.search(r"i(?:'m| am) at (.+?) and (?:need to |want to )?get to (.+?)(?:\s+in time|\s+for|\s+\.|$)", q)
    if match:
        result["origin"] = match.group(1).strip()
        result["destination"] = match.group(2).strip()
        logger.info(f"✓ PARSED (pattern 1): origin='{result['origin']}', dest='{result['destination']}'")
        return result
    
    # Pattern 2: "route from X to Y" or "from X to Y"
    match = re.search(r"(?:route |get )?from (.+?) to (.+?)(?:\s+in time|\s+for|\s+\.|$)", q)
    if match:
        result["origin"] = match.group(1).strip()
        result["destination"] = match.group(2).strip()
        logger.info(f"✓ PARSED (pattern 2): origin='{result['origin']}', dest='{result['destination']}'")
        return result
    
    # Pattern 3: "get to Y" or "to Y"
    match = re.search(r"(?:get |go )?to (.+?)(?:\s+in time|\s+for|\s+\.|$)", q)
    if match:
        result["destination"] = match.group(1).strip()
        logger.info(f"✓ PARSED (pattern 3): dest='{result['destination']}'")
        return result
    
    logger.warning(f"⚠️ No parse match for: '{query}'")
    return result


def is_likely_station_name(text: str) -> bool:
    """
    Check if text is likely already an MBTA station name.
    Returns True if it looks like a station, False if it needs resolution.
    """
    if not text:
        return False
    
    text_lower = text.strip().lower()
    
    # Common MBTA station name patterns
    station_indicators = [
        "station", "square", "/mit", "kendall",
        # Red Line
        "alewife", "davis", "porter", "harvard", "central",
        "park street", "downtown crossing", "south station",
        "broadway", "andrew", "jfk", "quincy",
        # Orange Line
        "oak grove", "malden", "wellington", "sullivan",
        "community college", "north station", "haymarket",
        "state", "chinatown", "tufts", "back bay",
        "forest hills", "ruggles",
        # Blue Line
        "wonderland", "revere", "beachmont", "suffolk downs",
        "airport", "wood island", "aquarium", "government center",
        # Green Line
        "lechmere", "science park", "north station", "haymarket",
        "government center", "park street", "boylston",
        "arlington", "copley", "hynes", "kenmore",
        "blandford", "boston university", "pleasant street",
        "saint paul", "kent street", "st paul", "hawes street",
        "longwood", "brookline village", "brookline hills",
        "beaconsfield", "reservoir", "chestnut hill",
        "newton", "riverside", "heath street",
        # Common words
        "street", "square", "center", "place", "ave"
    ]
    
    # Check if it contains station indicators
    if any(indicator in text_lower for indicator in station_indicators):
        logger.info(f"✓ '{text}' looks like a station name (skipping resolution)")
        return True
    
    # If very short and simple, might be station
    if len(text.split()) <= 2 and len(text) < 20:
        # Could be station, but not certain
        return False
    
    logger.info(f"✓ '{text}' looks like a landmark (needs resolution)")
    return False


def extract_station_from_stopfinder(text: str) -> str:
    """Extract station name from StopFinder response"""
    # Pattern: "Found: Kenmore in Boston"
    match = re.search(r"Found:\s*([^\n]+)", text)
    if match:
        station = match.group(1).strip()
        station = re.sub(r' in [A-Z][a-z]+', '', station)
        station = station.split('\n')[0].split('📍')[0].strip()
        return station
    
    # Pattern: "1. Station Name (City)"
    match = re.search(r'^\d+\.\s+([^(♿\n]+)', text, re.MULTILINE)
    if match:
        return match.group(1).strip()
    
    return ""


# ============================================================================
# REGISTRY
# ============================================================================

async def validate_registry() -> bool:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{REGISTRY_URL}/health")
            return r.status_code == 200
    except:
        return False


async def get_agent_catalog() -> List[Dict]:
    global _agent_catalog_cache, _catalog_cache_time
    
    if _agent_catalog_cache and _catalog_cache_time:
        if datetime.now() - _catalog_cache_time < _catalog_cache_ttl:
            return _agent_catalog_cache
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{REGISTRY_URL}/list")
            r.raise_for_status()
            agent_list = r.json()
            
            agents = []
            for aid in agent_list.keys():
                if aid == 'agent_status':
                    continue
                try:
                    ar = await client.get(f"{REGISTRY_URL}/agents/{aid}")
                    if ar.status_code == 200:
                        ad = ar.json()
                        if ad.get("alive"):
                            agents.append(ad)
                except:
                    pass
            
            _agent_catalog_cache = agents
            _catalog_cache_time = datetime.now()
            logger.info(f"✓ Registry: {len(agents)} agents")
            return agents
    except:
        return []


async def semantic_discovery(query: str) -> List[AgentConfig]:
    catalog = await get_agent_catalog()
    if not catalog:
        return []
    
    descriptions = [f"• {a['agent_id']}: {a['description']}" for a in catalog]
    catalog_text = "\n".join(descriptions)
    
    prompt = f"""Match query to agents.

Query: "{query}"

Agents:
{catalog_text}

Match ONLY the agents actually needed:

- For ALERTS queries ("show alerts", "any delays"): ONLY mbta-alerts
- For STOP queries ("find station", "where is"): ONLY mbta-stopfinder  
- For TRIP PLANNING ("route from X to Y", "get me to"): ALL THREE (stopfinder, alerts, planner)

Do NOT include agents unless they are actually needed for the query.

Return JSON: {{"matched_agents": ["id1", "id2"]}}
"""
    
    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                    "max_tokens": 150,
                    "response_format": {"type": "json_object"}
                },
                timeout=10
            )
            r.raise_for_status()
        
        result = json.loads(r.json()["choices"][0]["message"]["content"])
        matched_ids = result.get("matched_agents", [])
        
        configs = []
        for aid in matched_ids:
            ainfo = next((a for a in catalog if a['agent_id'] == aid), None)
            if ainfo:
                p = urlparse(ainfo['agent_url'])
                configs.append(AgentConfig(
                    name=ainfo['agent_id'],
                    url=f"{p.scheme or 'http'}://{p.hostname}",
                    port=p.port or 80,
                    description=ainfo['description'],
                    capabilities=ainfo.get('capabilities', [])
                ))
        
        logger.info(f"✓ Discovery: {matched_ids}")
        return configs
    except:
        return []


# ============================================================================
# AGENT COMMUNICATION
# ============================================================================

async def call_agent_slim(slim_client, config: AgentConfig, msg: str) -> Dict:
    agent_map = {
        "mbta-alerts": "alerts",
        "mbta-planner": "planner",
        "mbta-route-planner": "planner",
        "mbta-stops": "stopfinder",
        "mbta-stopfinder": "stopfinder"
    }
    
    name = agent_map.get(config.name)
    if not name:
        raise ValueError(f"No SLIM mapping: {config.name}")
    
    return await slim_client.call_agent(name, msg)


async def call_agent_http(config: AgentConfig, msg: str, conv_id: str) -> Dict:
    url = f"{config.url}:{config.port}/a2a/message"
    payload = {
        "type": "request",
        "payload": {"message": msg, "conversation_id": conv_id},
        "metadata": {"source": "stategraph"}
    }
    
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        result = r.json()
        
        if result.get("type") == "response" and "payload" in result:
            return {"response": result["payload"].get("text", ""), "agent_used": config.name}
        return result


async def call_stopfinder_for_location(location: str, config: AgentConfig, conv_id: str) -> str:
    """Call StopFinder and extract station name"""
    if not location:
        return ""
    
    query = f"Find station: {location}"
    
    global _current_orchestrator
    
    try:
        if _current_orchestrator and _current_orchestrator.use_slim and _current_orchestrator.slim_client:
            try:
                result = await call_agent_slim(_current_orchestrator.slim_client, config, query)
            except:
                result = await call_agent_http(config, query, conv_id)
        else:
            result = await call_agent_http(config, query, conv_id)
        
        response_text = result.get("response", "")
        station = extract_station_from_stopfinder(response_text)
        
        logger.info(f"   '{location}' → '{station}'")
        return station
        
    except Exception as e:
        logger.error(f"StopFinder error for '{location}': {e}")
        return ""


# ============================================================================
# NODES
# ============================================================================

async def discovery_node(state: AgentState) -> AgentState:
    """Semantic agent discovery"""
    with tracer.start_as_current_span("discovery"):
        matched = await semantic_discovery(state["user_message"])
        matched_ids = [a.name for a in matched]
        
        # Infer intent
        intent = "general"
        conf = 0.5
        
        if matched_ids:
            first = matched[0]
            d = first.description.lower()
            
            if any(w in d for w in ["route", "plan", "trip"]):
                intent, conf = "trip_planning", 0.85
            elif any(w in d for w in ["alert", "delay"]):
                intent, conf = "alerts", 0.85
            elif any(w in d for w in ["stop", "station"]):
                intent, conf = "stop_info", 0.85
        
        # Parse origin/destination
        parsed = extract_origin_destination(state["user_message"])
        
        return {
            **state,
            "matched_agents": matched_ids,
            "intent": intent,
            "confidence": conf,
            "origin_text": parsed["origin"],
            "destination_text": parsed["destination"],
            "agent_queries": {},
            "agents_called": [],
            "agent_responses": [],
            "messages": [HumanMessage(content=state["user_message"])],
            "llm_matching_decision": {"matched_agents": matched_ids},
            "resolved_origin": "",
            "resolved_destination": "",
            "has_disruptions": False,
            "affected_routes": [],
            "severity_level": "none",
            "routing_decision": ""
        }


def routing_node(state: AgentState) -> AgentState:
    """Intelligent routing decision"""
    intent = state["intent"]
    matched = state.get("matched_agents", [])
    
    logger.info(f"🎯 Routing: {intent}")
    
    # Check if 3 agents matched (means trip planning)
    if intent == "trip_planning" or len(matched) >= 3:
        # Order: stopfinder → alerts → planner
        ordered = []
        
        for a in matched:
            if "stopfinder" in a or "stops" in a:
                ordered.append(a)
                break
        
        if not any("alerts" in a for a in ordered):
            ordered.append("mbta-alerts")
        
        for a in matched:
            if "planner" in a or "route" in a:
                ordered.append(a)
                break
        
        state["matched_agents"] = list(dict.fromkeys(ordered))
        state["routing_decision"] = "FULL_CHAIN"
        
        logger.info(f"✓ Chain: {' → '.join(state['matched_agents'])}")
    
    return state


async def execute_agents_node(state: AgentState) -> AgentState:
    """
    OPTIMIZED: Only calls StopFinder for actual landmarks, not known stations
    """
    with tracer.start_as_current_span("execute_agents"):
        matched = state.get("matched_agents", [])
        if not matched:
            return {**state, "agents_called": [], "agent_responses": []}
        
        global _current_orchestrator
        catalog = await get_agent_catalog()
        
        responses = []
        agents_called = []
        
        # Get parsed values
        origin = state.get("origin_text", "")
        destination = state.get("destination_text", "")
        
        logger.info(f"🔄 Execute: {' → '.join(matched)}")
        logger.info(f"   Origin: '{origin}', Dest: '{destination}'")
        
        # OPTIMIZATION: Check if they're already station names
        origin_is_station = is_likely_station_name(origin)
        dest_is_station = is_likely_station_name(destination)
        
        logger.info(f"   Origin is station: {origin_is_station}, Dest is station: {dest_is_station}")
        
        # STEP 1: Resolve with StopFinder (ONLY if needed)
        resolved_origin = ""
        resolved_destination = ""
        
        # Find StopFinder config
        stopfinder_id = None
        for aid in matched:
            if "stopfinder" in aid or "stops" in aid:
                stopfinder_id = aid
                break
        
        if stopfinder_id and (not origin_is_station or not dest_is_station):
            ainfo = next((a for a in catalog if a['agent_id'] == stopfinder_id), None)
            if ainfo:
                p = urlparse(ainfo['agent_url'])
                sf_config = AgentConfig(
                    name=ainfo['agent_id'],
                    url=f"{p.scheme or 'http'}://{p.hostname}",
                    port=p.port or 80,
                    description=ainfo['description'],
                    capabilities=ainfo.get('capabilities', [])
                )
                
                logger.info(f"📍 Resolving landmarks with StopFinder...")
                
                # Resolve origin (only if it's NOT a station)
                if origin and not origin_is_station:
                    logger.info(f"   Resolving origin landmark: '{origin}'")
                    resolved_origin = await call_stopfinder_for_location(origin, sf_config, state["conversation_id"])
                elif origin:
                    logger.info(f"   Skipping origin (already station): '{origin}'")
                    resolved_origin = origin
                
                # Resolve destination (only if it's NOT a station)
                if destination and not dest_is_station:
                    logger.info(f"   Resolving destination landmark: '{destination}'")
                    resolved_destination = await call_stopfinder_for_location(destination, sf_config, state["conversation_id"])
                elif destination:
                    logger.info(f"   Skipping destination (already station): '{destination}'")
                    resolved_destination = destination
                
                # Use resolved names or fall back to original text
                final_origin = resolved_origin if resolved_origin else origin
                final_destination = resolved_destination if resolved_destination else destination
                
                logger.info(f"✓ Final stations: '{final_origin}' → '{final_destination}'")
                
                # Only add StopFinder to agents_called if we actually called it
                if (origin and not origin_is_station) or (destination and not dest_is_station):
                    agents_called.append(stopfinder_id)
        else:
            # Both are stations, use as-is
            resolved_origin = origin
            resolved_destination = destination
            logger.info(f"✓ Both are stations, using as-is: '{origin}' → '{destination}'")
        
        # STEP 2: Execute remaining agents (Alerts and Planner)
        has_disruptions = False
        affected = []
        severity = "none"
        
        for idx, agent_id in enumerate(matched):
            # Skip StopFinder (already handled)
            if agent_id == stopfinder_id:
                continue
            
            ainfo = next((a for a in catalog if a['agent_id'] == agent_id), None)
            if not ainfo:
                continue
            
            p = urlparse(ainfo['agent_url'])
            config = AgentConfig(
                name=ainfo['agent_id'],
                url=f"{p.scheme or 'http'}://{p.hostname}",
                port=p.port or 80,
                description=ainfo['description'],
                capabilities=ainfo.get('capabilities', [])
            )
            
            # Construct query
            if agent_id == "mbta-alerts":
                agent_query = "Check for major service disruptions on Red Line and Green Line"
                logger.info(f"⚠️ Alerts: '{agent_query}'")
            
            elif agent_id in ["mbta-planner", "mbta-route-planner"]:
                # Planner: Use RESOLVED station names
                from_station = resolved_origin if resolved_origin else origin
                to_station = resolved_destination if resolved_destination else destination
                
                if from_station and to_station:
                    agent_query = f"""IMPORTANT: Plan route using these EXACT station names. Do NOT extract different stations.

Origin station (use exactly): {from_station}
Destination station (use exactly): {to_station}

Plan the route between these two stations."""
                elif to_station:
                    agent_query = f"Find routes to {to_station}"
                else:
                    agent_query = state["user_message"]
                
                # Add disruption context
                if has_disruptions:
                    agent_query += f"\n\nIMPORTANT: {', '.join(affected)} have {severity} disruptions. Recommend alternatives."
                
                logger.info(f"🗺️ Planner: from='{from_station}' to='{to_station}'")
            
            else:
                agent_query = state["user_message"]
            
            logger.info(f"📞 [{idx+1}] {agent_id}")
            
            with tracer.start_as_current_span(f"agent_{agent_id}"):
                try:
                    if _current_orchestrator and _current_orchestrator.use_slim and _current_orchestrator.slim_client:
                        try:
                            result = await call_agent_slim(_current_orchestrator.slim_client, config, agent_query)
                        except:
                            result = await call_agent_http(config, agent_query, state["conversation_id"])
                    else:
                        result = await call_agent_http(config, agent_query, state["conversation_id"])
                    
                    resp_text = result.get("response", "")
                    
                    # Extract disruption context
                    if agent_id == "mbta-alerts":
                        if "🔴" in resp_text or "🟠" in resp_text:
                            has_disruptions = True
                            
                            for route in ["Red Line", "Orange Line", "Blue Line", "Green Line"]:
                                if route in resp_text:
                                    affected.append(route)
                            
                            if "🔴" in resp_text:
                                severity = "critical"
                            elif "🟠" in resp_text:
                                severity = "major"
                            
                            logger.info(f"⚠️ MAJOR disruptions: {affected}, severity={severity}")
                        else:
                            logger.info(f"✓ No major disruptions")
                    
                    responses.append(result)
                    agents_called.append(agent_id)
                    
                except Exception as e:
                    logger.error(f"❌ {agent_id}: {e}")
                    responses.append({"response": f"Error: {e}", "error": True})
                    agents_called.append(f"{agent_id}_error")
        
        return {
            **state,
            "agent_responses": responses,
            "agents_called": agents_called,
            "resolved_origin": resolved_origin,
            "resolved_destination": resolved_destination,
            "has_disruptions": has_disruptions,
            "affected_routes": affected,
            "severity_level": severity,
            "messages": [
                AIMessage(content=f"{r.get('agent_used', 'agent')}: {r.get('response', '')[:100]}", 
                         name=r.get('agent_used', 'agent'))
                for r in responses if not r.get('error')
            ]
        }


async def synthesize_node(state: AgentState) -> AgentState:
    """Intelligent synthesis"""
    with tracer.start_as_current_span("synthesize"):
        if not state.get("matched_agents"):
            msg = state["user_message"].lower()
            if any(w in msg for w in ["hi", "hello", "hey"]):
                return {**state, "final_response": "Hello! I help with MBTA routes, alerts, and stops. What do you need?", "should_end": True}
            return {**state, "final_response": "I specialize in MBTA transit. Ask about routes, alerts, or stops!", "should_end": True}
        
        responses = [r.get("response", "") for r in state.get("agent_responses", []) if not r.get('error') and r.get("response")]
        
        if not responses:
            return {**state, "final_response": "Agents unavailable.", "should_end": True}
        
        routing = state.get("routing_decision", "")
        
        # Intelligent synthesis for trip planning
        if routing == "FULL_CHAIN" and len(responses) >= 2:
            agents_called_list = state.get("agents_called", [])
            
            # Find alerts and planner responses
            alerts = ""
            planner = ""
            
            for idx, agent_id in enumerate(agents_called_list):
                if idx < len(responses):
                    resp_text = responses[idx].get("response", "") if isinstance(responses[idx], dict) else ""
                    
                    if "alert" in agent_id.lower():
                        alerts = resp_text
                    elif "planner" in agent_id.lower():
                        planner = resp_text
            
            origin_text = state.get("origin_text", "")
            dest_text = state.get("destination_text", "")
            resolved_origin = state.get("resolved_origin", "")
            resolved_dest = state.get("resolved_destination", "")
            disruptions = state.get("has_disruptions", False)
            affected = state.get("affected_routes", [])
            
            prompt = f"""Synthesize MBTA trip response (3-4 sentences max):

User query: {origin_text} → {dest_text}
Resolved stations: {resolved_origin or origin_text} → {resolved_dest or dest_text}

Alerts status:
{alerts[:250]}

Route plan:
{planner[:300]}

CRITICAL RULES:
1. Use ONLY information from the Route plan above - do NOT invent stations or routes
2. If landmarks were resolved, mention it briefly
3. State ONLY major disruptions (🔴 🟠), skip minor
4. Repeat the route directions from the planner EXACTLY
5. Max 4 sentences, be concise

Example: "Park Street to Harvard is a direct Red Line route. No major disruptions. From Park Street, take the Red Line towards Alewife to Harvard (3 stops, ~8 minutes)."
"""
            
            try:
                async with httpx.AsyncClient() as client:
                    r = await client.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {OPENAI_API_KEY}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": "gpt-4o-mini",
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": 0.2,
                            "max_tokens": 200
                        },
                        timeout=15
                    )
                    r.raise_for_status()
                
                final = r.json()["choices"][0]["message"]["content"].strip()
                logger.info("✨ Synthesized")
                return {**state, "final_response": final, "should_end": True}
                
            except Exception as e:
                logger.error(f"Synthesis error: {e}")
                return {**state, "final_response": "\n\n".join(responses), "should_end": True}
        
        else:
            return {**state, "final_response": "\n\n".join(responses), "should_end": True}


# ============================================================================
# ROUTING
# ============================================================================

def route_after_discovery(state: AgentState) -> Literal["routing", "synthesize"]:
    return "routing" if state.get("matched_agents") else "synthesize"


def route_after_routing(state: AgentState) -> Literal["execute", "synthesize"]:
    return "execute" if state.get("matched_agents") else "synthesize"


def route_after_execute(state: AgentState) -> Literal["synthesize"]:
    return "synthesize"


# ============================================================================
# GRAPH
# ============================================================================

def build_graph() -> StateGraph:
    wf = StateGraph(AgentState)
    
    wf.add_node("discovery", discovery_node)
    wf.add_node("routing", routing_node)
    wf.add_node("execute", execute_agents_node)
    wf.add_node("synthesize", synthesize_node)
    
    wf.set_entry_point("discovery")
    
    wf.add_conditional_edges("discovery", route_after_discovery,
                            {"routing": "routing", "synthesize": "synthesize"})
    wf.add_conditional_edges("routing", route_after_routing,
                            {"execute": "execute", "synthesize": "synthesize"})
    wf.add_conditional_edges("execute", route_after_execute,
                            {"synthesize": "synthesize"})
    wf.add_edge("synthesize", END)
    
    return wf.compile()


# ============================================================================
# ORCHESTRATOR
# ============================================================================

class StateGraphOrchestrator:
    """
    Optimized Production Orchestrator v4.3
    - Skips StopFinder for known station names
    - Only resolves actual landmarks
    - Faster responses for station-to-station queries
    """
    
    def __init__(self):
        global _current_orchestrator
        
        logger.info("=" * 80)
        logger.info("🚀 StateGraph Orchestrator v4.3 OPTIMIZED")
        logger.info("   ✅ Smart StopFinder skipping")
        logger.info("   ✅ Registry discovery")
        logger.info("   ✅ SLIM transport")
        logger.info("=" * 80)
        
        self.graph = build_graph()
        self.use_slim = os.getenv("USE_SLIM", "false").lower() == "true"
        self.slim_client = None
        
        if self.use_slim and SLIM_AVAILABLE:
            try:
                self.slim_client = SlimAgentClient()
                logger.info("✅ SLIM enabled")
            except Exception as e:
                logger.warning(f"SLIM failed: {e}")
                self.use_slim = False
        
        _current_orchestrator = self
        logger.info("✅ Orchestrator v4.3 ready")
    
    async def startup_validation(self):
        """Validate registry connection"""
        logger.info("🔍 Validating...")
        
        if not await validate_registry():
            raise RuntimeError("Registry not accessible")
        
        agents = await get_agent_catalog()
        logger.info(f"📚 {len(agents)} agents")
        
        if self.use_slim and self.slim_client:
            try:
                await self.slim_client.initialize()
                logger.info("✅ SLIM ready")
            except:
                self.use_slim = False
        
        logger.info("✅ Startup complete")
    
    async def process_message(self, user_message: str, conversation_id: str) -> Dict[str, Any]:
        """Process message through optimized intelligent flow"""
        with tracer.start_as_current_span("stategraph_v43"):
            initial = {
                "user_message": user_message,
                "conversation_id": conversation_id,
                "intent": "",
                "confidence": 0.0,
                "matched_agents": [],
                "agent_queries": {},
                "llm_matching_decision": {},
                "messages": [],
                "agents_called": [],
                "agent_responses": [],
                "origin_text": "",
                "destination_text": "",
                "resolved_origin": "",
                "resolved_destination": "",
                "has_disruptions": False,
                "affected_routes": [],
                "severity_level": "none",
                "final_response": "",
                "should_end": False,
                "routing_decision": ""
            }
            
            final = await self.graph.ainvoke(initial)
            
            return {
                "response": final["final_response"],
                "intent": final["intent"],
                "confidence": final["confidence"],
                "matched_agents": final.get("matched_agents", []),
                "agents_called": final["agents_called"],
                "metadata": {
                    "conversation_id": conversation_id,
                    "discovery": "semantic",
                    "transport": "slim" if self.use_slim else "http",
                    "optimization": {
                        "stopfinder_skipped": not final.get("resolved_origin") and not final.get("resolved_destination"),
                    },
                    "context": {
                        "origin": final.get("origin_text", ""),
                        "destination": final.get("destination_text", ""),
                        "resolved_origin": final.get("resolved_origin", ""),
                        "resolved_dest": final.get("resolved_destination", ""),
                        "disruptions": final.get("has_disruptions", False),
                        "affected": final.get("affected_routes", []),
                        "severity": final.get("severity_level", "none")
                    },
                    "registry_url": REGISTRY_URL
                }
            }