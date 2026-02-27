"""
MBTA Route Planner Agent - SLIM Wrapper with Full Logic
Comprehensive route planning with LLM-based location extraction
Incorporates all logic from main.py into SLIM transport
"""

import asyncio
import logging
import os
import sys
import json
from typing import Optional, Dict, Any, List, Tuple
from uuid import uuid4

sys.path.insert(0, '/opt/mbta-agents')

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.types import AgentCard, AgentSkill, AgentCapabilities, Message, TextPart

from dotenv import load_dotenv
import uvicorn
from openai import OpenAI
import httpx

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Configuration
MBTA_API_KEY = os.getenv('MBTA_API_KEY', 'c845eff5ae504179bc9cfa69914059de')
MBTA_BASE_URL = "https://api-v3.mbta.com"
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

if not MBTA_API_KEY:
    logger.warning("MBTA_API_KEY not found in environment variables!")

if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not found - LLM extraction disabled!")


class PlannerExecutor(AgentExecutor):
    """
    Full-featured SLIM planner with all logic from main.py
    """
    
    def __init__(self, openai_api_key: str, mbta_api_key: str):
        self.openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None
        self.mbta_api_key = mbta_api_key
        self.session = None
    
    async def extract_locations_with_llm(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Use LLM to extract origin and destination from natural language query.
        Handles all phrasings robustly.
        
        Examples:
            "how do I get from park street to harvard" → ("park street", "harvard")
            "i wanna go to park street from northeastern" → ("northeastern", "park street")
            "take me to harvard" → (None, "harvard")
        """
        if not self.openai_client:
            logger.warning("OpenAI client not available, falling back to basic parsing")
            return await self.extract_locations_basic(query)
        
        prompt = f"""Extract the origin and destination locations from this transit query.

Query: "{query}"

Instructions:
- Return ONLY the two location names separated by a pipe |
- Use the exact location names mentioned (preserve "northeastern university", "park street", etc.)
- If only destination is mentioned, use "none" for origin
- If locations are unclear, use "none"
- Do not include words like "station", "stop" unless part of the name
- Handle variations: "harvard" could be "Harvard Square", "mit" could be "Kendall/MIT"

Format: origin|destination

Examples:
- "how do I get from park street to harvard" → park street|harvard
- "i wanna go to park street from northeastern university" → northeastern university|park street
- "take me to harvard" → none|harvard
- "northeastern to park street" → northeastern|park street
- "go to kenmore from airport" → airport|kenmore
- "show me the way to mit" → none|mit
- "from harvard to boston university" → harvard|boston university

Response:"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=50
            )
            
            result = response.choices[0].message.content.strip()
            
            if "|" in result:
                parts = result.split("|")
                origin = parts[0].strip() if parts[0].strip().lower() != "none" else None
                destination = parts[1].strip() if len(parts) > 1 and parts[1].strip().lower() != "none" else None
                
                logger.info(f"LLM extracted: origin='{origin}', destination='{destination}'")
                return origin, destination
            
            logger.warning(f"LLM returned unexpected format: {result}")
            return await self.extract_locations_basic(query)
            
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return await self.extract_locations_basic(query)
    
    async def extract_locations_basic(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Fallback: Basic string parsing for location extraction.
        Used if LLM is not available.
        """
        query_lower = query.lower()
        
        origin = None
        destination = None
        
        # Try to extract locations
        if " from " in query_lower and " to " in query_lower:
            parts = query_lower.split(" from ")
            if len(parts) > 1:
                from_part = parts[1]
                to_parts = from_part.split(" to ")
                if len(to_parts) >= 2:
                    origin = to_parts[0].strip()
                    destination = to_parts[1].strip()
        
        elif " to " in query_lower:
            parts = query_lower.split(" to ")
            if len(parts) >= 2:
                origin_part = parts[0].strip()
                destination = parts[1].strip()
                
                # Clean origin
                for word in ["how", "do", "i", "get", "go", "wanna", "want", "travel", "the", "show", "way"]:
                    origin_part = origin_part.replace(f" {word} ", " ").strip()
                
                origin = origin_part if origin_part else None
        
        # Clean up
        if origin:
            origin = origin.strip("?.,!").strip()
        if destination:
            destination = destination.strip("?.,!").strip()
        
        logger.info(f"Basic parsing extracted: origin='{origin}', destination='{destination}'")
        return origin, destination
    
    async def find_stop_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Find a stop by name using MBTA API with client-side filtering.
        Handles fuzzy matching and multiple stop types.
        """
        if not name:
            return None
        
        try:
            params = {
                "api_key": self.mbta_api_key,
                "page[limit]": 500,
                "filter[location_type]": "1"  # Only stations
            }
            
            logger.info(f"Searching for stop: '{name}'")
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{MBTA_BASE_URL}/stops",
                    params=params,
                    timeout=10
                )
                response.raise_for_status()
            
            data = response.json()
            stops = data.get("data", [])
            
            # Filter by name client-side
            name_lower = name.lower().strip()
            matching_stops = []
            
            for stop in stops:
                stop_name = stop.get("attributes", {}).get("name", "").lower()
                # Check if query is in the stop name
                if name_lower in stop_name:
                    matching_stops.append(stop)
            
            if matching_stops:
                # Return the best match (first one)
                stop = matching_stops[0]
                attributes = stop.get("attributes", {})
                
                result = {
                    "id": stop.get("id"),
                    "name": attributes.get("name"),
                    "latitude": attributes.get("latitude"),
                    "longitude": attributes.get("longitude"),
                    "address": attributes.get("address"),
                    "wheelchair_boarding": attributes.get("wheelchair_boarding")
                }
                
                logger.info(f"Found stop: {result['name']}")
                return result
            
            logger.warning(f"No stop found matching '{name}'")
            return None
        
        except Exception as e:
            logger.error(f"Error finding stop '{name}': {e}")
            return None
    
    async def get_routes_between_stops(self, origin_id: str, destination_id: str) -> List[Dict[str, Any]]:
        """
        Find routes that serve both origin and destination stops.
        Returns comprehensive route information.
        """
        if not origin_id or not destination_id:
            return []
        
        try:
            # Get routes serving origin stop
            params = {
                "api_key": self.mbta_api_key,
                "filter[stop]": origin_id
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{MBTA_BASE_URL}/routes",
                    params=params,
                    timeout=10
                )
                response.raise_for_status()
            
            origin_routes = response.json().get("data", [])
            origin_route_ids = {route.get("id") for route in origin_routes}
            
            # Get routes serving destination stop
            params["filter[stop]"] = destination_id
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{MBTA_BASE_URL}/routes",
                    params=params,
                    timeout=10
                )
                response.raise_for_status()
            
            dest_routes = response.json().get("data", [])
            dest_route_ids = {route.get("id") for route in dest_routes}
            
            # Find common routes
            common_route_ids = origin_route_ids.intersection(dest_route_ids)
            
            # Get details of common routes
            common_routes = []
            for route in origin_routes:
                if route.get("id") in common_route_ids:
                    attributes = route.get("attributes", {})
                    common_routes.append({
                        "id": route.get("id"),
                        "short_name": attributes.get("short_name", ""),
                        "long_name": attributes.get("long_name", attributes.get("short_name", "Unknown")),
                        "type": attributes.get("type"),
                        "color": attributes.get("color"),
                        "text_color": attributes.get("text_color"),
                        "description": attributes.get("description"),
                        "direction_names": attributes.get("direction_names", [])
                    })
            
            logger.info(f"Found {len(common_routes)} routes between stops")
            return common_routes
        
        except Exception as e:
            logger.error(f"Error finding routes: {e}")
            return []
    
    async def get_predictions(self, stop_id: str, route_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get real-time predictions for a stop.
        """
        if not stop_id:
            return []
        
        try:
            params = {
                "api_key": self.mbta_api_key,
                "filter[stop]": stop_id,
                "page[limit]": 5
            }
            
            if route_id:
                params["filter[route]"] = route_id
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{MBTA_BASE_URL}/predictions",
                    params=params,
                    timeout=10
                )
                response.raise_for_status()
            
            predictions = response.json().get("data", [])
            
            formatted = []
            for pred in predictions:
                attributes = pred.get("attributes", {})
                formatted.append({
                    "arrival_time": attributes.get("arrival_time"),
                    "departure_time": attributes.get("departure_time"),
                    "direction_id": attributes.get("direction_id"),
                    "status": attributes.get("status")
                })
            
            return formatted
        
        except Exception as e:
            logger.error(f"Error getting predictions: {e}")
            return []
    
    async def plan_route(self, origin: str, destination: str) -> Dict[str, Any]:
        """
        Plan a route between two locations using real MBTA data.
        Returns comprehensive route information with predictions.
        """
        try:
            logger.info(f"Planning route from '{origin}' to '{destination}'")
            
            # Step 1: Find origin stop
            origin_stop = await self.find_stop_by_name(origin)
            if not origin_stop:
                return {
                    "ok": False,
                    "error": f"Could not find origin stop: {origin}",
                    "text": f"Sorry, I couldn't find a stop matching '{origin}'. Please check the name and try again."
                }
            
            # Step 2: Find destination stop
            dest_stop = await self.find_stop_by_name(destination)
            if not dest_stop:
                return {
                    "ok": False,
                    "error": f"Could not find destination stop: {destination}",
                    "text": f"Sorry, I couldn't find a stop matching '{destination}'. Please check the name and try again."
                }
            
            logger.info(f"Found stops - Origin: {origin_stop['name']}, Destination: {dest_stop['name']}")
            
            # Step 3: Find routes connecting the stops
            routes = await self.get_routes_between_stops(origin_stop["id"], dest_stop["id"])
            
            if not routes:
                return {
                    "ok": True,
                    "origin": origin_stop,
                    "destination": dest_stop,
                    "routes": [],
                    "text": f"No direct routes found between {origin_stop['name']} and {dest_stop['name']}. You may need to transfer between lines.",
                    "summary": "No direct routes"
                }
            
            # Step 4: Get predictions for origin stop
            predictions = []
            if routes:
                predictions = await self.get_predictions(origin_stop["id"], routes[0].get("id"))
            
            # Step 5: Format response
            if len(routes) == 1:
                route = routes[0]
                text = f"🚇 Take the {route['long_name']} from {origin_stop['name']} to {dest_stop['name']}.\n"
                
                # Add predictions if available
                if predictions:
                    text += f"\n⏰ Next arrivals:\n"
                    for i, pred in enumerate(predictions[:3], 1):
                        arrival = pred.get("arrival_time") or pred.get("departure_time")
                        if arrival:
                            text += f"  {i}. {arrival}\n"
            else:
                text = f"🚇 Multiple options available from {origin_stop['name']} to {dest_stop['name']}:\n"
                for i, route in enumerate(routes, 1):
                    text += f"\n{i}. {route['long_name']}"
                    if route.get('color'):
                        text += f" (#{route['color']})"
            
            return {
                "ok": True,
                "origin": origin_stop,
                "destination": dest_stop,
                "routes": routes,
                "predictions": predictions,
                "text": text,
                "summary": f"{len(routes)} route(s) available"
            }
            
        except Exception as e:
            logger.error(f"Error planning route: {e}", exc_info=True)
            return {
                "ok": False,
                "error": str(e),
                "text": "An error occurred while planning your route. Please try again later."
            }
    
    async def execute(self, context: RequestContext, event_queue: EventQueue):
        """
        Execute the planner agent.
        Extracts locations from query and plans route.
        """
        try:
            # Extract message text
            message_text = ""
            for part in context.message.parts:
                if hasattr(part, 'root') and hasattr(part.root, 'text'):
                    message_text = part.root.text
                    break
                elif hasattr(part, 'text'):
                    message_text = part.text
                    break
            
            logger.info(f"Processing trip planning query: '{message_text[:100]}...'")
            
            # Extract locations using LLM
            origin, destination = await self.extract_locations_with_llm(message_text)
            
            logger.info(f"Extracted locations - Origin: '{origin}', Destination: '{destination}'")
            
            # Validation
            if not destination:
                response_text = "I couldn't understand where you want to go. Please specify your destination. For example: 'How do I get to Harvard?' or 'Take me from Park Street to Kenmore.'"
                
                response_message = Message(
                    message_id=str(uuid4()),
                    parts=[TextPart(text=response_text)],
                    role="agent"
                )
                await event_queue.enqueue_event(response_message)
                return
            
            if not origin:
                response_text = f"I can help you get to {destination}! Where are you starting from? For example: 'From Park Street to {destination}'"
                
                response_message = Message(
                    message_id=str(uuid4()),
                    parts=[TextPart(text=response_text)],
                    role="agent"
                )
                await event_queue.enqueue_event(response_message)
                return
            
            # Plan route using MBTA API
            result = await self.plan_route(origin=origin, destination=destination)
            
            # Extract text or create formatted response
            response_text = result.get("text", "Could not plan route")
            
            # Add metadata if available
            if result.get("ok"):
                response_text += f"\n\n✅ Route planning successful"
            
            # Create and send response
            response_message = Message(
                message_id=str(uuid4()),
                parts=[TextPart(text=response_text)],
                role="agent"
            )
            
            await event_queue.enqueue_event(response_message)
            
        except Exception as e:
            logger.error(f"Error in execute: {e}", exc_info=True)
            error_text = f"An error occurred: {str(e)}"
            error_message = Message(
                message_id=str(uuid4()),
                parts=[TextPart(text=error_text)],
                role="agent"
            )
            await event_queue.enqueue_event(error_message)
    
    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        """Cancel execution - not implemented"""
        raise NotImplementedError()


def main():
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    mbta_api_key = os.getenv("MBTA_API_KEY", "")
    
    # Define agent skills
    skills = [
        AgentSkill(
            id="route_planning",
            name="Route Planning",
            description="Plans optimal routes between two MBTA stops",
            tags=["routing", "planning", "directions"],
            examples=["Plan a route from Park Street to Harvard", "How do I get to MIT?"]
        ),
        AgentSkill(
            id="location_extraction",
            name="Location Extraction",
            description="Extracts origin and destination from natural language",
            tags=["parsing", "nlp"],
            examples=["From Park Street to Harvard", "Take me to Kendall"]
        ),
        AgentSkill(
            id="stop_finding",
            name="Stop Finding",
            description="Finds MBTA stops by name",
            tags=["search", "discovery"],
            examples=["Find Harvard Square", "Where is MIT?"]
        ),
        AgentSkill(
            id="real_time_predictions",
            name="Real-Time Predictions",
            description="Gets real-time arrival/departure predictions",
            tags=["predictions", "real-time"],
            examples=["Next trains at Park Street", "Arrival times"]
        )
    ]
    
    # Create agent card
    agent_card = AgentCard(
        name="mbta-planner",
        description="Comprehensive MBTA route planner with LLM-based location extraction and real-time predictions",
        url="http://96.126.111.107:50052/",
        version="2.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        skills=skills,
        capabilities=AgentCapabilities(streaming=True)
    )
    
    # Create executor and server
    executor = PlannerExecutor(openai_api_key, mbta_api_key)
    handler = DefaultRequestHandler(executor=executor, task_store=InMemoryTaskStore())
    server = A2AStarletteApplication(agent_card=agent_card, http_handler=handler)
    app = server.build()
    
    logger.info("🚀 Enhanced Planner Agent v2.0 with Full Logic")
    logger.info("✅ LLM-based location extraction")
    logger.info("✅ Comprehensive route planning")
    logger.info("✅ Real-time predictions")
    logger.info("✅ Fuzzy stop matching")
    
    uvicorn.run(app, host="0.0.0.0", port=50052, log_level="info")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()