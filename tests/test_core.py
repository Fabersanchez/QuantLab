"""
QuantLab Core Architecture Unit Tests.

Verifies functionality of Logger, LifecycleManager, ComponentRegistry,
EventBus, ModuleManager, and QuantEngine using standard library unittest.
"""

import unittest
from core import (
    QuantLogger,
    get_logger,
    LifecycleManager,
    SystemState,
    InvalidStateTransitionError,
    ComponentRegistry,
    ComponentAlreadyRegisteredError,
    ComponentNotFoundError,
    EventBus,
    Event,
    ModuleManager,
    BaseModule,
    ModuleAlreadyRegisteredError,
    ModuleNotFoundError,
    QuantEngine,
)


class DummyModule(BaseModule):
    def __init__(self, name: str = "Dummy") -> None:
        self._name = name
        self.initialized = False
        self.shutdown_called = False

    @property
    def name(self) -> str:
        return self._name

    def initialize(self) -> None:
        self.initialized = True

    def shutdown(self) -> None:
        self.shutdown_called = True


class TestQuantLabCore(unittest.TestCase):
    def test_logger(self) -> None:
        logger = get_logger("TestLogger")
        self.assertIsInstance(logger, QuantLogger)
        logger.info("Test info message")
        logger.warning("Test warning message")
        logger.error("Test error message")

    def test_lifecycle_manager(self) -> None:
        lifecycle = LifecycleManager()
        self.assertEqual(lifecycle.current_state, SystemState.CREATED)

        lifecycle.transition_to(SystemState.INITIALIZING)
        self.assertEqual(lifecycle.current_state, SystemState.INITIALIZING)

        lifecycle.transition_to(SystemState.READY)
        self.assertEqual(lifecycle.current_state, SystemState.READY)

        lifecycle.transition_to(SystemState.RUNNING)
        self.assertEqual(lifecycle.current_state, SystemState.RUNNING)

        lifecycle.transition_to(SystemState.PAUSED)
        self.assertEqual(lifecycle.current_state, SystemState.PAUSED)

        lifecycle.transition_to(SystemState.RUNNING)
        self.assertEqual(lifecycle.current_state, SystemState.RUNNING)

        lifecycle.transition_to(SystemState.STOPPING)
        self.assertEqual(lifecycle.current_state, SystemState.STOPPING)

        lifecycle.transition_to(SystemState.STOPPED)
        self.assertEqual(lifecycle.current_state, SystemState.STOPPED)

        with self.assertRaises(InvalidStateTransitionError):
            lifecycle.transition_to(SystemState.RUNNING)

    def test_component_registry(self) -> None:
        registry = ComponentRegistry()
        registry.register("service_a", "InstanceA")
        self.assertTrue(registry.has("service_a"))
        self.assertEqual(registry.get("service_a"), "InstanceA")

        with self.assertRaises(ComponentAlreadyRegisteredError):
            registry.register("service_a", "InstanceA_Duplicate")

        registry.register("service_b", "InstanceB")
        self.assertEqual(len(registry.list_components()), 2)
        self.assertIn("service_a", registry.search("service"))

        removed = registry.unregister("service_a")
        self.assertEqual(removed, "InstanceA")
        self.assertFalse(registry.has("service_a"))

        with self.assertRaises(ComponentNotFoundError):
            registry.get("service_a")

    def test_event_bus(self) -> None:
        bus = EventBus()
        received_events = []

        def handler(event: Event) -> None:
            received_events.append(event)

        bus.subscribe("MARKET_DATA", handler)
        notified = bus.publish("MARKET_DATA", {"symbol": "BTC/USD"})

        self.assertEqual(notified, 1)
        self.assertEqual(len(received_events), 1)
        self.assertEqual(received_events[0].payload, {"symbol": "BTC/USD"})

        unsub = bus.unsubscribe("MARKET_DATA", handler)
        self.assertTrue(unsub)
        notified = bus.publish("MARKET_DATA", {"symbol": "ETH/USD"})
        self.assertEqual(notified, 0)
        self.assertEqual(len(received_events), 1)

    def test_module_manager(self) -> None:
        manager = ModuleManager()
        mod = DummyModule("AlphaModule")

        manager.register_module(mod)
        self.assertIn("AlphaModule", manager.list_modules())

        with self.assertRaises(ModuleAlreadyRegisteredError):
            manager.register_module(mod)

        loaded = manager.load_module("AlphaModule")
        self.assertTrue(loaded.initialized)
        self.assertIn("AlphaModule", manager.list_loaded_modules())

        manager.unload_module("AlphaModule")
        self.assertTrue(loaded.shutdown_called)
        self.assertNotIn("AlphaModule", manager.list_loaded_modules())

    def test_quant_engine_full_lifecycle(self) -> None:
        engine = QuantEngine(version="0.1.0")
        self.assertEqual(engine.state, SystemState.CREATED)

        mod = DummyModule("BetaModule")
        engine.register_module(mod)

        engine.start()
        self.assertEqual(engine.state, SystemState.RUNNING)

        engine.load_module("BetaModule")
        self.assertTrue(mod.initialized)

        engine.pause()
        self.assertEqual(engine.state, SystemState.PAUSED)

        engine.resume()
        self.assertEqual(engine.state, SystemState.RUNNING)

        engine.stop()
        self.assertEqual(engine.state, SystemState.STOPPED)
        self.assertTrue(mod.shutdown_called)


if __name__ == "__main__":
    unittest.main()
