"""
QuantLab Reinforcement Learning Research Lab Unit Tests.

Verifies complete functionality of all 17 modules:
BaseEnvironment, ActionSpace, StateBuilder, RewardEngine, MarketEnvironment,
Replay Buffers, Exploration Strategies, CurriculumManager, RLAgentRegistry,
PolicyManager, RLCheckpointManager, RL Algorithm Agents (DQN, Double DQN,
Dueling DQN, PPO, A2C, A3C, SAC, TD3, DDPG, Rainbow), RLTrainer, RLEvaluator,
RLExperimentTracker, RLVisualizer, RLReportGenerator, ReinforcementLearningEngine.
"""

import os
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd

from reinforcement_learning import (
    # Environment
    BaseEnvironment, SpaceSpec, StepResult,
    # Action Space
    DiscreteAction, ActionSpace, N_DISCRETE_ACTIONS,
    # State Builder
    PortfolioState, StateBuilder,
    # Reward
    RewardConfig, RewardEngine,
    # Market Environment
    MarketConfig, MarketEnvironment,
    # Replay Buffers
    Transition, UniformReplayBuffer, PrioritizedReplayBuffer, NStepReplayBuffer,
    # Exploration
    EpsilonGreedyExploration, SoftmaxExploration, UCBExploration,
    EntropyExploration, NoisyNetworksExploration,
    # Curriculum
    CurriculumStage, CurriculumManager, CURRICULUM_STAGES,
    # Registry & Policy
    RLAgentRecord, RLAgentRegistry, PolicyManager,
    # Checkpoint
    RLCheckpoint, RLCheckpointManager,
    # Algorithms & Trainer
    BaseRLAgent,
    DQNAgent, DoubleDQNAgent, DuelingDQNAgent, RainbowDQNAgent,
    PPOAgent, A2CAgent, A3CAgent, SACAgent, TD3Agent, DDPGAgent,
    create_agent, TrainingConfig, TrainingResult, RLTrainer,
    # Evaluator
    RLEvaluationReport, RLEvaluator,
    # Tracker
    RLExperimentRun, RLExperimentTracker,
    # Visualization & Reports
    RLVisualizer, RLReportGenerator,
    # Engine
    RLEngineConfig, RLEngineResult, ReinforcementLearningEngine,
)


def _make_market_df(n: int = 200) -> pd.DataFrame:
    """Build synthetic OHLCV DataFrame for testing."""
    np.random.seed(42)
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.3)
    high = close + np.abs(np.random.randn(n) * 0.1) + 0.05
    low = close - np.abs(np.random.randn(n) * 0.1) - 0.05
    return pd.DataFrame({
        "open": low + (high - low) * 0.4,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.random.randint(500, 3000, n).astype(float),
    })


class TestActionSpace(unittest.TestCase):
    def test_discrete_actions(self):
        space = ActionSpace()
        self.assertEqual(space.n_actions, N_DISCRETE_ACTIONS)
        self.assertEqual(space.to_label(DiscreteAction.BUY), "BUY")
        self.assertEqual(space.to_label(DiscreteAction.SELL), "SELL")
        self.assertTrue(space.is_valid(0))
        self.assertFalse(space.is_valid(99))

    def test_sample(self):
        space = ActionSpace()
        for _ in range(20):
            a = space.sample()
            self.assertIn(a, range(N_DISCRETE_ACTIONS))


class TestStateBuilder(unittest.TestCase):
    def setUp(self):
        self.df = _make_market_df(100)
        self.builder = StateBuilder(lookback=15, feature_cols=[])

    def test_obs_dim(self):
        self.assertGreater(self.builder.obs_dim, 0)

    def test_build_returns_correct_shape(self):
        obs = self.builder.build(self.df, current_idx=50)
        self.assertEqual(obs.shape[0], self.builder.obs_dim)
        self.assertEqual(obs.dtype, np.float32)

    def test_build_with_portfolio(self):
        port = PortfolioState(equity=12000.0, position=1)
        obs = self.builder.build(self.df, current_idx=50, portfolio=port)
        self.assertIsNotNone(obs)


class TestRewardEngine(unittest.TestCase):
    def setUp(self):
        self.engine = RewardEngine()

    def test_step_reward_range(self):
        self.engine.reset()
        r = self.engine.calculate(10000, 10100, 10100, DiscreteAction.HOLD, 0, 0)
        self.assertGreater(r, -11.0)
        self.assertLess(r, 11.0)

    def test_drawdown_penalty(self):
        self.engine.reset()
        r = self.engine.calculate(10000, 8000, 10000, DiscreteAction.HOLD, 0, 0)
        self.assertLess(r, 0)

    def test_terminal_reward(self):
        equity_hist = [10000.0, 10200.0, 10500.0, 10400.0, 10800.0]
        r = self.engine.calculate_terminal(equity_hist)
        self.assertIsInstance(r, float)

    def test_sharpe_sortino(self):
        returns = np.array([0.01, -0.005, 0.02, 0.003, -0.001])
        sharpe = RewardEngine._sharpe(returns)
        sortino = RewardEngine._sortino(returns)
        self.assertIsInstance(sharpe, float)
        self.assertIsInstance(sortino, float)


class TestMarketEnvironment(unittest.TestCase):
    def setUp(self):
        self.df = _make_market_df(200)
        self.env = MarketEnvironment(self.df, config=MarketConfig(lookback=15, max_steps=100))

    def test_reset(self):
        obs, info = self.env.reset()
        self.assertIsInstance(obs, np.ndarray)
        self.assertGreater(len(obs), 0)
        self.assertIsInstance(info, dict)

    def test_step(self):
        self.env.reset()
        result = self.env.step(DiscreteAction.BUY)
        self.assertIsInstance(result.observation, np.ndarray)
        self.assertIsInstance(result.reward, float)
        self.assertIsInstance(result.done, bool)

    def test_full_episode(self):
        obs, _ = self.env.reset()
        total_reward = 0.0
        for _ in range(50):
            action = int(np.random.randint(0, N_DISCRETE_ACTIONS))
            result = self.env.step(action)
            obs = result.observation
            total_reward += result.reward
            if result.done:
                break
        self.assertIsInstance(total_reward, float)

    def test_render(self):
        self.env.reset()
        output = self.env.render(mode="ansi")
        self.assertIsInstance(output, str)
        self.assertIn("MarketEnv", output)

    def test_seed(self):
        s = self.env.seed(42)
        self.assertEqual(s, 42)

    def test_close(self):
        self.env.reset()
        self.env.close()  # Should not raise

    def test_observation_space(self):
        obs_space = self.env.observation_space
        self.assertIsInstance(obs_space, SpaceSpec)
        self.assertGreater(obs_space.shape[0], 0)


class TestReplayBuffers(unittest.TestCase):
    def _make_transition(self, obs_dim=10):
        s = np.random.randn(obs_dim).astype(np.float32)
        ns = np.random.randn(obs_dim).astype(np.float32)
        return s, np.random.randint(0, 7), float(np.random.randn()), ns, bool(np.random.rand() > 0.9)

    def test_uniform_buffer(self):
        buf = UniformReplayBuffer(capacity=100)
        for _ in range(50):
            buf.push(*self._make_transition())
        self.assertEqual(len(buf), 50)
        self.assertTrue(buf.is_ready(32))
        s, a, r, ns, d = buf.sample(32)
        self.assertEqual(s.shape[0], 32)

    def test_prioritized_buffer(self):
        buf = PrioritizedReplayBuffer(capacity=100)
        for _ in range(60):
            buf.push(*self._make_transition())
        self.assertTrue(buf.is_ready(32))
        s, a, r, ns, d, w, idx = buf.sample(32)
        self.assertEqual(len(idx), 32)
        buf.update_priorities(idx, np.random.rand(32))

    def test_nstep_buffer(self):
        buf = NStepReplayBuffer(capacity=100, n_step=3)
        for _ in range(50):
            buf.push(*self._make_transition())
        self.assertIsInstance(len(buf), int)


class TestExplorationStrategies(unittest.TestCase):
    def test_epsilon_greedy(self):
        strat = EpsilonGreedyExploration(epsilon_start=1.0, epsilon_end=0.05, decay_steps=100)
        q = np.random.randn(7)
        for _ in range(50):
            a = strat.select_action(q, 7)
            self.assertIn(a, range(7))
        self.assertLess(strat.epsilon, 1.0)

    def test_softmax(self):
        strat = SoftmaxExploration()
        q = np.random.randn(7)
        a = strat.select_action(q, 7)
        self.assertIn(a, range(7))

    def test_ucb(self):
        strat = UCBExploration(c=1.0, n_actions=7)
        q = np.random.randn(7)
        for _ in range(20):
            a = strat.select_action(q)
            self.assertIn(a, range(7))

    def test_entropy(self):
        strat = EntropyExploration(temperature=0.5)
        q = np.random.randn(7)
        a = strat.select_action(q, 7)
        self.assertIn(a, range(7))
        ent = strat.entropy(q, 7)
        self.assertGreaterEqual(ent, 0.0)

    def test_noisy_networks(self):
        strat = NoisyNetworksExploration()
        q = np.random.randn(7)
        a = strat.select_action(q, 7)
        self.assertIn(a, range(7))


class TestCurriculumLearning(unittest.TestCase):
    def test_curriculum_stages(self):
        self.assertEqual(len(CURRICULUM_STAGES), 5)

    def test_manager_initial_stage(self):
        mgr = CurriculumManager()
        self.assertEqual(mgr.stage_idx, 0)
        self.assertFalse(mgr.is_complete)

    def test_force_advance(self):
        mgr = CurriculumManager()
        advanced = mgr.force_advance()
        self.assertTrue(advanced)
        self.assertEqual(mgr.stage_idx, 1)

    def test_apply_to_dataframe(self):
        mgr = CurriculumManager()
        df = _make_market_df(100)
        df_out = mgr.apply_to_dataframe(df)
        self.assertIn("close", df_out.columns)

    def test_summary(self):
        mgr = CurriculumManager()
        summary = mgr.summary()
        self.assertIn("current_stage", summary)
        self.assertIn("stage_idx", summary)


class TestAgentRegistry(unittest.TestCase):
    def test_register_and_get(self):
        reg = RLAgentRegistry()
        rec = reg.register("dqn_agent", "DQN", metrics={"mean_reward": 12.5})
        self.assertIsNotNone(rec.agent_id)
        self.assertEqual(rec.version, 1)
        fetched = reg.get(rec.agent_id)
        self.assertIsNotNone(fetched)

    def test_version_increment(self):
        reg = RLAgentRegistry()
        r1 = reg.register("agent", "PPO")
        r2 = reg.register("agent", "PPO")
        self.assertEqual(r1.version, 1)
        self.assertEqual(r2.version, 2)

    def test_update_status(self):
        reg = RLAgentRegistry()
        rec = reg.register("agent", "SAC")
        reg.update_status(rec.agent_id, "PRODUCTION")
        self.assertEqual(reg.get(rec.agent_id).status, "PRODUCTION")

    def test_get_best_agent(self):
        reg = RLAgentRegistry()
        reg.register("a1", "DQN", metrics={"mean_reward": 5.0})
        reg.register("a2", "PPO", metrics={"mean_reward": 12.0})
        best = reg.get_best_agent(metric="mean_reward")
        self.assertEqual(best.metrics["mean_reward"], 12.0)


class TestPolicyManager(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_save_and_load(self):
        agent = DQNAgent(obs_dim=10, n_actions=7)
        path = os.path.join(self.tmp, "policy.pkl")
        saved = PolicyManager.save(agent, path, metadata={"test": True})
        self.assertTrue(os.path.exists(saved))
        loaded = PolicyManager.load(saved)
        self.assertIsNotNone(loaded)

    def test_clone(self):
        agent = PPOAgent(obs_dim=10, n_actions=7)
        clone = PolicyManager.clone(agent)
        self.assertIsNot(agent, clone)

    def test_freeze_unfreeze(self):
        agent = SACAgent(obs_dim=5, n_actions=7)
        PolicyManager.freeze(agent)
        PolicyManager.unfreeze(agent)

    def test_archive(self):
        agent = DQNAgent(obs_dim=5, n_actions=7)
        archive_path = PolicyManager.archive(agent, self.tmp, "agent-v1")
        self.assertTrue(os.path.exists(archive_path))


class TestCheckpointManager(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.ckpt_mgr = RLCheckpointManager(checkpoint_dir=self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_save_and_load(self):
        agent = DQNAgent(obs_dim=10, n_actions=7)
        path = self.ckpt_mgr.save(
            agent_id="test_agent",
            algorithm="DQN",
            episode=10,
            step=500,
            policy_weights=agent.get_weights(),
            reward_history=[1.0, 2.0, 1.5],
            metrics={"mean_reward": 1.5},
        )
        self.assertTrue(os.path.exists(path))
        ckpt = self.ckpt_mgr.load("test_agent")
        self.assertIsNotNone(ckpt)
        self.assertEqual(ckpt.episode, 10)

    def test_list_checkpoints(self):
        agent = DQNAgent(obs_dim=5, n_actions=7)
        self.ckpt_mgr.save("agent2", "PPO", 1, 100, agent.get_weights())
        tags = self.ckpt_mgr.list_checkpoints("agent2")
        self.assertIn("latest", tags)


class TestRLAlgorithmAgents(unittest.TestCase):
    def _run_agent(self, algo: str):
        obs_dim, n_actions = 15, 7
        agent = create_agent(algo, obs_dim=obs_dim, n_actions=n_actions)
        obs = np.random.randn(obs_dim).astype(np.float32)
        action = agent.select_action(obs)
        self.assertIn(action, range(n_actions))

        # Update pass
        states = np.random.randn(16, obs_dim).astype(np.float32)
        actions = np.random.randint(0, n_actions, 16).astype(np.int32)
        rewards = np.random.randn(16).astype(np.float32)
        next_states = np.random.randn(16, obs_dim).astype(np.float32)
        dones = np.zeros(16, dtype=np.float32)
        losses = agent.update(states, actions, rewards, next_states, dones)
        self.assertIsInstance(losses, dict)

    def test_dqn(self): self._run_agent("DQN")
    def test_double_dqn(self): self._run_agent("DOUBLE_DQN")
    def test_dueling_dqn(self): self._run_agent("DUELING_DQN")
    def test_rainbow(self): self._run_agent("RAINBOW")
    def test_ppo(self): self._run_agent("PPO")
    def test_a2c(self): self._run_agent("A2C")
    def test_a3c(self): self._run_agent("A3C")
    def test_sac(self): self._run_agent("SAC")
    def test_td3(self): self._run_agent("TD3")
    def test_ddpg(self): self._run_agent("DDPG")


class TestRLTrainer(unittest.TestCase):
    def setUp(self):
        self.df = _make_market_df(200)
        self.env = MarketEnvironment(self.df, config=MarketConfig(lookback=15, max_steps=50))
        obs_dim = self.env.observation_space.shape[0]
        self.agent = DQNAgent(obs_dim=obs_dim, n_actions=N_DISCRETE_ACTIONS)

    def test_training_result(self):
        cfg = TrainingConfig(n_episodes=5, max_steps_per_episode=50, batch_size=16, buffer_capacity=500)
        trainer = RLTrainer(self.agent, self.env, cfg)
        result = trainer.train()

        self.assertIsInstance(result, TrainingResult)
        self.assertEqual(result.n_episodes, 5)
        self.assertEqual(len(result.episode_rewards), 5)
        self.assertGreater(result.total_steps, 0)


class TestRLEvaluator(unittest.TestCase):
    def test_evaluation_report(self):
        df = _make_market_df(200)
        env = MarketEnvironment(df, config=MarketConfig(lookback=15, max_steps=50))
        obs_dim = env.observation_space.shape[0]
        agent = PPOAgent(obs_dim=obs_dim, n_actions=N_DISCRETE_ACTIONS)

        report = RLEvaluator.evaluate(agent, env, n_episodes=3, max_steps=50)
        self.assertIsInstance(report, RLEvaluationReport)
        self.assertIn("mean_reward", report.metrics)
        self.assertIn("sharpe_ratio", report.metrics)
        self.assertIn("max_drawdown_pct", report.metrics)
        self.assertIn("win_rate", report.metrics)
        self.assertIn("profit_factor", report.metrics)


class TestExperimentTracker(unittest.TestCase):
    def test_log_and_list(self):
        tracker = RLExperimentTracker()
        run = tracker.log_run(
            "test_exp", "DQN", {"lr": 0.001},
            episode_rewards=[1.0, 2.0, 1.5],
            metrics={"mean_reward": 1.5},
        )
        self.assertIsNotNone(run.run_id)
        runs = tracker.list_runs()
        self.assertEqual(len(runs), 1)

    def test_to_dataframe(self):
        tracker = RLExperimentTracker()
        tracker.log_run("exp1", "PPO", {"lr": 0.0003}, [1.0, 2.0], metrics={"mean_reward": 1.5})
        df = tracker.to_dataframe()
        self.assertFalse(df.empty)
        self.assertIn("algorithm", df.columns)


class TestRLVisualizer(unittest.TestCase):
    def test_reward_curve_svg(self):
        rewards = np.random.randn(50).tolist()
        svg = RLVisualizer.generate_reward_curve_svg(rewards)
        self.assertIn("<svg", svg)
        self.assertIn("polyline", svg)

    def test_action_distribution_svg(self):
        action_counts = {0: 30, 1: 20, 2: 15, 3: 10}
        svg = RLVisualizer.generate_action_distribution_svg(action_counts)
        self.assertIn("<svg", svg)

    def test_learning_progress_svg(self):
        loss = [0.5, 0.4, 0.35, 0.3, 0.28]
        svg = RLVisualizer.generate_learning_progress_svg(loss)
        self.assertIn("<svg", svg)

    def test_empty_inputs(self):
        svg = RLVisualizer.generate_reward_curve_svg([])
        self.assertIn("<svg", svg)


class TestRLReportGenerator(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.df = _make_market_df(200)

        config = RLEngineConfig(
            algorithm="DQN",
            n_episodes=5,
            max_steps_per_episode=50,
            n_eval_episodes=2,
            export_reports=False,
            checkpoint_dir=os.path.join(self.tmp, "ckpts"),
        )
        engine = ReinforcementLearningEngine(config)
        engine.load_data(self.df, "BTCUSD")
        self.result = engine.start_pipeline()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_export_all_formats(self):
        reporter = RLReportGenerator(self.result)
        out_dir = os.path.join(self.tmp, "reports")
        paths = reporter.export_all(out_dir)

        self.assertTrue(os.path.exists(paths["html"]))
        self.assertTrue(os.path.exists(paths["markdown"]))
        self.assertTrue(os.path.exists(paths["json"]))
        self.assertTrue(os.path.exists(paths["pdf"]))
        self.assertTrue(os.path.exists(paths["episode_rewards_csv"]))
        self.assertTrue(os.path.exists(paths["action_distribution_csv"]))


class TestReinforcementLearningEngine(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.df = _make_market_df(200)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _run_engine(self, algorithm: str) -> RLEngineResult:
        config = RLEngineConfig(
            algorithm=algorithm,
            n_episodes=5,
            max_steps_per_episode=50,
            n_eval_episodes=2,
            checkpoint_dir=os.path.join(self.tmp, f"ckpts_{algorithm}"),
        )
        engine = ReinforcementLearningEngine(config)
        engine.load_data(self.df, "EURUSD")
        return engine.start_pipeline()

    def test_dqn_pipeline(self):
        res = self._run_engine("DQN")
        self.assertIsInstance(res, RLEngineResult)
        self.assertGreater(res.execution_time_seconds, 0)
        self.assertEqual(len(res.episode_rewards), 5)

    def test_ppo_pipeline(self):
        res = self._run_engine("PPO")
        self.assertIsInstance(res, RLEngineResult)

    def test_sac_pipeline(self):
        res = self._run_engine("SAC")
        self.assertIsInstance(res, RLEngineResult)

    def test_ddpg_pipeline(self):
        res = self._run_engine("DDPG")
        self.assertIsInstance(res, RLEngineResult)

    def test_predict_action(self):
        config = RLEngineConfig(
            algorithm="DQN",
            n_episodes=3,
            max_steps_per_episode=30,
            n_eval_episodes=1,
            checkpoint_dir=os.path.join(self.tmp, "ckpts_pred"),
        )
        engine = ReinforcementLearningEngine(config)
        engine.load_data(self.df, "XAUUSD")
        engine.start_pipeline()

        obs_dim = engine._active_env.observation_space.shape[0]
        obs = np.random.randn(obs_dim).astype(np.float32)
        action = engine.predict_action(obs, deterministic=True)
        self.assertIn(action, range(N_DISCRETE_ACTIONS))

    def test_curriculum_learning_pipeline(self):
        config = RLEngineConfig(
            algorithm="A2C",
            n_episodes=5,
            max_steps_per_episode=30,
            n_eval_episodes=1,
            use_curriculum=True,
            checkpoint_dir=os.path.join(self.tmp, "ckpts_curr"),
        )
        engine = ReinforcementLearningEngine(config)
        engine.load_data(self.df, "GBPUSD")
        res = engine.start_pipeline()
        self.assertIsInstance(res, RLEngineResult)


if __name__ == "__main__":
    unittest.main()
