"""
MBTA Alerts Agent - SLIM Wrapper with Full Logic
Comprehensive service alerts and disruptions detection
Incorporates all logic from main.py into SLIM transport
"""

import asyncio
import logging
import os
import sys
from typing import Optional, Dict, Any, List
from uuid import uuid4
from datetime import datetime

sys.path.insert(0, '/opt/mbta-agents')

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.types import AgentCard, AgentSkill, AgentCapabilities, Message, TextPart

from dotenv import load_dotenv
import uvicorn
import httpx

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Configuration
MBTA_API_KEY = os.getenv('MBTA_API_KEY', 'c845eff5ae504179bc9cfa69914059de')
MBTA_BASE_URL = "https://api-v3.mbta.com"

if not MBTA_API_KEY:
    logger.warning("MBTA_API_KEY not found in environment variables!")


class AlertsExecutor(AgentExecutor):
    """
    Full-featured SLIM alerts agent with all logic from main.py
    """
    
    def __init__(self, mbta_api_key: str):
        self.mbta_api_key = mbta_api_key
    
    def extract_route_from_query(self, query: str) -> Optional[str]:
        """
        Extract route/line name from user query.
        
        Examples:
            "Red Line delays" → "Red"
            "are there alerts on orange line?" → "Orange"
            "blue line problems" → "Blue"
            "green line disruptions" → "Green-B"
        """
        query_lower = query.lower()
        
        # Map of keywords to MBTA route IDs
        route_mapping = {
            "red line": "Red",
            "red": "Red",
            "orange line": "Orange",
            "orange": "Orange",
            "blue line": "Blue",
            "blue": "Blue",
            "green line": "Green-B",  # Default to B branch
            "green": "Green-B",
            "green-b": "Green-B",
            "green b": "Green-B",
            "green-c": "Green-C",
            "green c": "Green-C",
            "green-d": "Green-D",
            "green d": "Green-D",
            "green-e": "Green-E",
            "green e": "Green-E",
            "mattapan": "Mattapan",
            "mattapan line": "Mattapan",
            "silver line": "741",
            "silver": "741",
            "commuter rail": "Commuter",
        }
        
        for keyword, route_id in route_mapping.items():
            if keyword in query_lower:
                logger.info(f"Detected route: {route_id} from query")
                return route_id
        
        return None
    
    def classify_alert_severity(self, alert: Dict[str, Any]) -> str:
        """
        Classify alert severity level based on MBTA data.
        
        Returns emoji indicator and severity level
        """
        attributes = alert.get("attributes", {})
        effect = attributes.get("effect", "")
        
        # Severity mapping based on effect type
        severity_map = {
            "NO_SERVICE": ("🔴", "CRITICAL"),
            "REDUCED_SERVICE": ("🟠", "MAJOR"),
            "SIGNIFICANT_DELAYS": ("🟠", "MAJOR"),
            "DELAY": ("🟡", "MINOR"),
            "SHUTTLE": ("🟡", "MINOR"),
            "STOP_CLOSURE": ("🟠", "MAJOR"),
            "STATION_CLOSURE": ("🔴", "CRITICAL"),
            "ELEVATOR_CLOSURE": ("🔵", "INFO"),
            "ESCALATOR_CLOSURE": ("🔵", "INFO"),
            "PARKING_CLOSURE": ("🔵", "INFO"),
            "PLANNED_WORK": ("🔵", "INFO"),
            "MODIFIED_SERVICE": ("🟡", "MINOR"),
            "OTHER_EFFECT": ("⚪", "UNKNOWN"),
            "UNKNOWN_EFFECT": ("⚪", "UNKNOWN"),
            "DETOUR": ("🟡", "MINOR"),
            "ADDITIONAL_SERVICE": ("🟢", "INFO"),
            "SERVICE_CHANGE": ("🟡", "MINOR"),
        }
        
        emoji, severity = severity_map.get(effect, ("⚪", "UNKNOWN"))
        return emoji, severity, effect
    
    def extract_alert_summary(self, alert: Dict[str, Any]) -> str:
        """
        Extract human-readable summary from alert.
        """
        attributes = alert.get("attributes", {})
        
        # Try to get description, header, or informed_entity
        description = attributes.get("description", "")
        header = attributes.get("header", "")
        short_header = attributes.get("short_header", "")
        
        # Use the best available summary
        if header:
            return header[:100]  # Limit to 100 chars
        elif short_header:
            return short_header[:100]
        elif description:
            return description.split('\n')[0][:100]  # First line, limit to 100 chars
        else:
            effect = attributes.get("effect", "Service Alert")
            return effect
    
    def extract_affected_routes(self, alert: Dict[str, Any]) -> List[str]:
        """
        Extract affected routes from alert's informed_entity.
        """
        attributes = alert.get("attributes", {})
        informed_entity = attributes.get("informed_entity", [])
        
        affected_routes = []
        for entity in informed_entity:
            route = entity.get("route")
            if route:
                affected_routes.append(route)
        
        return list(set(affected_routes))  # Remove duplicates
    
    def format_alert_for_display(
        self,
        alert: Dict[str, Any],
        include_full_details: bool = False
    ) -> str:
        """
        Format alert for display to user.
        """
        emoji, severity, effect = self.classify_alert_severity(alert)
        summary = self.extract_alert_summary(alert)
        affected_routes = self.extract_affected_routes(alert)
        
        attributes = alert.get("attributes", {})
        created_at = attributes.get("created_at", "")
        updated_at = attributes.get("updated_at", "")
        
        # Format datetime
        if created_at:
            try:
                dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                time_str = dt.strftime("%I:%M %p")
            except:
                time_str = ""
        else:
            time_str = ""
        
        # Build display text
        text = f"{emoji} [{severity}] {summary}"
        
        if time_str:
            text += f" (at {time_str})"
        
        if include_full_details:
            if affected_routes:
                text += f"\n   Routes: {', '.join(affected_routes)}"
            
            description = attributes.get("description", "")
            if description:
                text += f"\n   Details: {description[:150]}"
        
        return text
    
    async def get_all_alerts(self) -> List[Dict[str, Any]]:
        """
        Fetch all alerts from MBTA API.
        """
        try:
            params = {
                "api_key": self.mbta_api_key,
                "include": "routes,stops"
            }
            
            logger.info("Fetching alerts from MBTA API")
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{MBTA_BASE_URL}/alerts",
                    params=params,
                    timeout=10
                )
                response.raise_for_status()
            
            data = response.json()
            alerts = data.get("data", [])
            
            logger.info(f"Fetched {len(alerts)} alerts from MBTA API")
            return alerts
        
        except Exception as e:
            logger.error(f"Error fetching alerts: {e}")
            return []
    
    async def get_alerts_for_route(self, route: str) -> List[Dict[str, Any]]:
        """
        Get alerts for a specific route.
        """
        try:
            params = {
                "api_key": self.mbta_api_key,
                "filter[route]": route,
                "include": "routes,stops"
            }
            
            logger.info(f"Fetching alerts for route: {route}")
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{MBTA_BASE_URL}/alerts",
                    params=params,
                    timeout=10
                )
                response.raise_for_status()
            
            data = response.json()
            alerts = data.get("data", [])
            
            logger.info(f"Found {len(alerts)} alerts for route {route}")
            return alerts
        
        except Exception as e:
            logger.error(f"Error fetching alerts for route {route}: {e}")
            return []
    
    def filter_alerts_by_severity(
        self,
        alerts: List[Dict[str, Any]],
        severity_threshold: str = "MINOR"
    ) -> List[Dict[str, Any]]:
        """
        Filter alerts by severity level.
        Severity levels (highest to lowest): CRITICAL, MAJOR, MINOR, INFO
        """
        severity_levels = {"CRITICAL": 4, "MAJOR": 3, "MINOR": 2, "INFO": 1}
        threshold_level = severity_levels.get(severity_threshold, 0)
        
        filtered = []
        for alert in alerts:
            _, severity, _ = self.classify_alert_severity(alert)
            alert_level = severity_levels.get(severity, 0)
            
            if alert_level >= threshold_level:
                filtered.append(alert)
        
        return filtered
    
    def analyze_alert_impact(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze impact of an alert.
        Returns information about affected routes, stops, and impact level.
        """
        attributes = alert.get("attributes", {})
        effect = attributes.get("effect", "")
        informed_entity = attributes.get("informed_entity", [])
        
        affected_routes = set()
        affected_stops = set()
        
        for entity in informed_entity:
            if entity.get("route"):
                affected_routes.add(entity.get("route"))
            if entity.get("stop"):
                affected_stops.add(entity.get("stop"))
        
        return {
            "effect": effect,
            "affected_routes": list(affected_routes),
            "affected_stops": list(affected_stops),
            "total_affected_routes": len(affected_routes),
            "total_affected_stops": len(affected_stops)
        }
    
    async def execute(self, context: RequestContext, event_queue: EventQueue):
        """
        Execute the alerts agent.
        Fetches and analyzes MBTA service alerts.
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
            
            logger.info(f"📨 Alerts Agent received: '{message_text}'")
            
            # Check for route mention
            route = self.extract_route_from_query(message_text)
            
            # Fetch alerts
            if route:
                logger.info(f"Fetching alerts for route: {route}")
                alerts = await self.get_alerts_for_route(route)
            else:
                logger.info("Fetching all alerts")
                alerts = await self.get_all_alerts()
            
            # Build response
            if not alerts:
                if route:
                    response_text = f"✅ No alerts on the {route} Line. Everything is running smoothly!"
                else:
                    response_text = "✅ No active alerts. All MBTA services are operating normally!"
            else:
                if route:
                    response_text = f"⚠️ Alerts on the {route} Line:\n\n"
                else:
                    response_text = f"⚠️ Current MBTA Alerts ({len(alerts)} total):\n\n"
                
                # Sort by severity (critical first)
                severity_order = {"CRITICAL": 4, "MAJOR": 3, "MINOR": 2, "INFO": 1}
                
                def get_severity_value(alert):
                    _, severity, _ = self.classify_alert_severity(alert)
                    return severity_order.get(severity, 0)
                
                sorted_alerts = sorted(alerts, key=get_severity_value, reverse=True)
                
                # Display top 10 alerts
                for i, alert in enumerate(sorted_alerts[:10], 1):
                    alert_text = self.format_alert_for_display(alert, include_full_details=True)
                    response_text += f"{i}. {alert_text}\n\n"
                
                if len(alerts) > 10:
                    response_text += f"... and {len(alerts) - 10} more alerts"
            
            # Create and send response
            response_message = Message(
                message_id=str(uuid4()),
                parts=[TextPart(text=response_text)],
                role="agent"
            )
            
            await event_queue.enqueue_event(response_message)
            logger.info("✅ Alert response sent")
            
        except Exception as e:
            logger.error(f"Error in execute: {e}", exc_info=True)
            error_text = f"❌ Error fetching alerts: {str(e)}"
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
    mbta_api_key = os.getenv("MBTA_API_KEY", "")
    
    # Define agent skills
    skills = [
        AgentSkill(
            id="service_alerts",
            name="Service Alerts",
            description="Retrieves real-time service alerts and disruptions for MBTA transit lines",
            tags=["alerts", "disruptions", "service"],
            examples=["Are there any delays?", "Red Line alerts", "What's wrong with the Orange Line?"]
        ),
        AgentSkill(
            id="alert_severity",
            name="Alert Severity Classification",
            description="Classifies alert severity levels (Critical, Major, Minor, Info)",
            tags=["severity", "classification", "impact"],
            examples=["How serious is this alert?", "Major service disruptions"]
        ),
        AgentSkill(
            id="impact_analysis",
            name="Impact Analysis",
            description="Analyzes which routes and stops are affected by alerts",
            tags=["impact", "analysis", "affected"],
            examples=["Which routes are affected?", "Show me all affected stops"]
        ),
        AgentSkill(
            id="alert_tracking",
            name="Alert Tracking",
            description="Tracks when alerts were created and updated",
            tags=["tracking", "timeline", "updates"],
            examples=["When was this alert posted?", "Show alert timeline"]
        )
    ]
    
    # Create agent card
    agent_card = AgentCard(
        name="mbta-alerts",
        description="Real-time MBTA service alerts and disruptions with severity classification and impact analysis",
        url="http://96.126.111.107:50051/",
        version="2.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        skills=skills,
        capabilities=AgentCapabilities(streaming=True)
    )
    
    # Create executor and server - FIXED: use agent_executor as positional argument
    executor = AlertsExecutor(mbta_api_key)
    handler = DefaultRequestHandler(executor, task_store=InMemoryTaskStore())
    server = A2AStarletteApplication(agent_card=agent_card, http_handler=handler)
    app = server.build()
    
    logger.info("🚀 Enhanced Alerts Agent v2.0 with Full Logic")
    logger.info("✅ Route detection and filtering")
    logger.info("✅ Severity classification")
    logger.info("✅ Impact analysis")
    logger.info("✅ Alert formatting and display")
    logger.info("✅ Real-time alert tracking")
    
    uvicorn.run(app, host="0.0.0.0", port=50051, log_level="info")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()