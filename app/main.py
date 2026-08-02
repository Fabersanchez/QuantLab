"""
QuantLab Application Entry Point.

Institutional Quantitative Research Laboratory main runner script.
"""

import sys
from pathlib import Path

# Add project root to sys.path for standalone module execution
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core import QuantEngine, BaseModule, Event


class ExampleCoreModule(BaseModule):
    """Demonstration decoupled core module for testing architecture capabilities."""

    def __init__(self, name: str = "TelemetryService") -> None:
        self._name = name
        self._active = False

    @property
    def name(self) -> str:
        return self._name

    def initialize(self) -> None:
        self._active = True
        print(f"[{self._name}] Subsystem initialized and active.")

    def shutdown(self) -> None:
        self._active = False
        print(f"[{self._name}] Subsystem safely shut down.")


def display_banner() -> None:
    """Print the institutional startup banner to stdout."""
    banner = """==================================================
QuantLab
Institutional Quantitative Research Laboratory
=================================================="""
    print(banner)


def main() -> None:
    """Main execution function for QuantLab application."""
    display_banner()

    # Initialize QuantLab Core Engine
    engine = QuantEngine(version="0.1.0")

    # Subscribe an event handler to system lifecycle events
    def on_system_event(event: Event) -> None:
        engine.logger.info(
            f"EventBus Received -> Topic: {event.event_type} | Payload: {event.payload}"
        )

    engine.event_bus.subscribe_all(on_system_event)

    # Register a sample decoupled module
    telemetry_mod = ExampleCoreModule(name="TelemetryService")
    engine.register_module(telemetry_mod)

    # Start engine lifecycle (CREATED -> INITIALIZING -> READY -> RUNNING)
    engine.start()

    # Load the registered module
    engine.load_module("TelemetryService")

    # Inspect registered core components
    registered_components = engine.registry.list_components()
    engine.logger.info(
        f"Registered Components in Registry: {registered_components}"
    )

    # Stop engine gracefully (RUNNING -> STOPPING -> STOPPED)
    engine.stop()


if __name__ == "__main__":
    main()
