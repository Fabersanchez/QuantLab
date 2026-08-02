"""
QuantLab Core Engine.

The main orchestrator uniting LifecycleManager, ComponentRegistry, EventBus,
ModuleManager, and QuantLogger into a single institutional framework.
"""

from typing import Optional
from core.logger import QuantLogger, get_logger
from core.lifecycle import LifecycleManager, SystemState
from core.registry import ComponentRegistry
from core.event_bus import EventBus
from core.module_manager import ModuleManager, BaseModule


class QuantEngine:
    """Institutional Quantitative Engine Core.

    Central coordinator of system components, lifecycle transitions, event routing,
    and module management in QuantLab.
    """

    def __init__(
        self, version: str = "0.1.0", logger: Optional[QuantLogger] = None
    ) -> None:
        """Initialize QuantEngine and core infrastructure.

        Args:
            version: Core engine semantic version string.
            logger: Custom logger instance or None to instantiate default.
        """
        self._version: str = version
        self._logger: QuantLogger = logger or get_logger("QuantEngine")
        self._lifecycle: LifecycleManager = LifecycleManager()
        self._registry: ComponentRegistry = ComponentRegistry()
        self._event_bus: EventBus = EventBus()
        self._module_manager: ModuleManager = ModuleManager()

        # Register self and core subsystems in the component registry
        self._registry.register("engine", self)
        self._registry.register("logger", self._logger)
        self._registry.register("lifecycle", self._lifecycle)
        self._registry.register("registry", self._registry)
        self._registry.register("event_bus", self._event_bus)
        self._registry.register("module_manager", self._module_manager)

    @property
    def version(self) -> str:
        """Return semantic version string of QuantEngine."""
        return self._version

    @property
    def state(self) -> SystemState:
        """Return current lifecycle state."""
        return self._lifecycle.current_state

    @property
    def logger(self) -> QuantLogger:
        """Access core logger instance."""
        return self._logger

    @property
    def registry(self) -> ComponentRegistry:
        """Access component registry instance."""
        return self._registry

    @property
    def lifecycle(self) -> LifecycleManager:
        """Access lifecycle manager instance."""
        return self._lifecycle

    @property
    def event_bus(self) -> EventBus:
        """Access event bus instance."""
        return self._event_bus

    @property
    def module_manager(self) -> ModuleManager:
        """Access module manager instance."""
        return self._module_manager

    def start(self) -> None:
        """Start the quantitative research engine lifecycle.

        Executes state transitions: CREATED -> INITIALIZING -> READY -> RUNNING.
        """
        self._logger.info(f"Starting QuantEngine v{self._version}...")

        # CREATED -> INITIALIZING
        self._lifecycle.transition_to(SystemState.INITIALIZING)
        self._logger.info("Initializing core system components...")

        # Notify via EventBus
        self._event_bus.publish("SYSTEM_INITIALIZING", {"version": self._version})

        # INITIALIZING -> READY
        self._lifecycle.transition_to(SystemState.READY)
        self._logger.info("Core components initialized. State: READY.")

        # READY -> RUNNING
        self._lifecycle.transition_to(SystemState.RUNNING)
        self._event_bus.publish("SYSTEM_RUNNING", {"version": self._version})
        self._logger.info("Engine lifecycle active. State: RUNNING.")

    def stop(self) -> None:
        """Stop the quantitative engine and safely unload active modules."""
        self._logger.info("Stopping QuantEngine...")

        if self._lifecycle.current_state in (
            SystemState.RUNNING,
            SystemState.PAUSED,
            SystemState.READY,
        ):
            self._lifecycle.transition_to(SystemState.STOPPING)
            self._event_bus.publish("SYSTEM_STOPPING")

            # Unload loaded modules in reverse order
            loaded = self._module_manager.list_loaded_modules()
            for mod_name in reversed(loaded):
                self._logger.info(f"Unloading module: {mod_name}")
                self._module_manager.unload_module(mod_name)

            self._lifecycle.transition_to(SystemState.STOPPED)
            self._event_bus.publish("SYSTEM_STOPPED")
            self._logger.info("QuantEngine stopped cleanly. State: STOPPED.")

    def pause(self) -> None:
        """Pause engine execution."""
        if self._lifecycle.current_state == SystemState.RUNNING:
            self._lifecycle.transition_to(SystemState.PAUSED)
            self._event_bus.publish("SYSTEM_PAUSED")
            self._logger.info("QuantEngine execution PAUSED.")

    def resume(self) -> None:
        """Resume paused engine execution."""
        if self._lifecycle.current_state == SystemState.PAUSED:
            self._lifecycle.transition_to(SystemState.RUNNING)
            self._event_bus.publish("SYSTEM_RESUMED")
            self._logger.info("QuantEngine execution RESUMED.")

    def register_module(self, module: BaseModule) -> None:
        """Register a module into the engine's ModuleManager."""
        self._module_manager.register_module(module)
        self._logger.info(f"Module '{module.name}' registered.")

    def load_module(self, name: str) -> BaseModule:
        """Load an already registered module."""
        mod = self._module_manager.load_module(name)
        self._logger.info(f"Module '{name}' loaded successfully.")
        return mod

    def unload_module(self, name: str) -> None:
        """Unload an active module."""
        self._module_manager.unload_module(name)
        self._logger.info(f"Module '{name}' unloaded.")
