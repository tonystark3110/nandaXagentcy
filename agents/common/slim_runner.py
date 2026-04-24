"""
SLIM Transport Runner — agents/common/slim_runner.py
=====================================================
Generic SLIM server wrapper for any MBTA A2A agent.

Each agent calls run_slim_server() alongside its existing uvicorn A2A server.
This registers the agent with the SLIM controller so the exchange can reach it
via SLIM transport (instead of plain HTTP SSE).

Transport stack:
  Exchange  →  SLIM Controller (:46357)  →  Agent SLIM listener
                                              ↓
                                         DefaultRequestHandler
                                              ↓
                                         AgentExecutor.execute()

Environment variables (set per agent in supervisor):
  SLIM_ENDPOINT     = http://96.126.111.107:46357  (SLIM controller)
  SLIM_SHARED_SECRET= <32+ char secret, same on all agents + exchange>
  SLIM_ORG          = mbta
  SLIM_NS           = transit-ci
  SLIM_LABEL        = alerts | planner | stopfinder
"""

import asyncio
import logging
import os

logger = logging.getLogger(__name__)

SLIM_ENDPOINT      = os.getenv("SLIM_ENDPOINT",      "http://96.126.111.107:46357")
SLIM_SHARED_SECRET = os.getenv("SLIM_SHARED_SECRET", "")
SLIM_ORG           = os.getenv("SLIM_ORG",           "mbta")
SLIM_NS            = os.getenv("SLIM_NS",            "transit-ci")
SLIM_LABEL         = os.getenv("SLIM_LABEL",         "")   # alerts / planner / stopfinder


async def run_slim_server(agent_card, executor, task_store):
    """
    Register this agent with the SLIM controller and serve incoming A2A requests.

    Blocks until the process is killed — call from asyncio.run() in each
    agent's run_slim.py entry point.

    Args:
        agent_card:  a2a.types.AgentCard instance (same card the A2A server uses)
        executor:    AgentExecutor subclass instance (AlertsExecutor, etc.)
        task_store:  InMemoryTaskStore() instance
    """
    if not SLIM_LABEL:
        raise RuntimeError("SLIM_LABEL env var must be set (alerts / planner / stopfinder)")
    if not SLIM_SHARED_SECRET:
        raise RuntimeError(
            "SLIM_SHARED_SECRET env var must be set (≥32 chars, same value on all servers)"
        )
    if len(SLIM_SHARED_SECRET) < 32:
        raise RuntimeError(
            f"SLIM_SHARED_SECRET too short ({len(SLIM_SHARED_SECRET)} chars) — need ≥32"
        )

    identity = f"{SLIM_ORG}/{SLIM_NS}/{SLIM_LABEL}"
    logger.info(f"🔌 Starting SLIM transport: identity={identity} endpoint={SLIM_ENDPOINT}")

    try:
        from agntcy_app_sdk.factory import AgntcyFactory
        from agntcy_app_sdk.semantic.a2a.server.srpc import (
            A2ASlimRpcServerConfig,
            SlimRpcConnectionConfig,
        )
        from a2a.server.request_handlers import DefaultRequestHandler
    except ImportError as exc:
        raise ImportError(
            "agntcy-app-sdk not installed. Run:\n"
            "  pip install agntcy-app-sdk==0.5.5 slim-bindings==1.3.0"
        ) from exc

    config = A2ASlimRpcServerConfig(
        agent_card=agent_card,
        request_handler=DefaultRequestHandler(
            agent_executor=executor,
            task_store=task_store,
        ),
        connection=SlimRpcConnectionConfig(
            identity=identity,
            shared_secret=SLIM_SHARED_SECRET,
            endpoint=SLIM_ENDPOINT,
        ),
    )

    factory = AgntcyFactory()
    session = factory.create_session(config)

    logger.info(f"✅ SLIM registered: {identity} @ {SLIM_ENDPOINT}")
    logger.info(f"   Waiting for incoming A2A requests via SLIM...")

    # Blocks until process exits — keep_alive=True means reconnect on drop
    await session.start_all_sessions(keep_alive=True)
