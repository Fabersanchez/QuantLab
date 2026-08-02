"""
QuantLab Master Reinforcement Learning Engine.

Orchestrates the entire RL research lifecycle: environment creation,
state building, reward calculation, curriculum management, agent training,
evaluation, MLOps registry, experiment tracking, and multi-format reporting.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from core.logger import get_logger
from reinforcement_learning.agent_registry import RLAgentRegistry
from reinforcement_learning.checkpoint_manager import RLCheckpointManager
from reinforcement_learning.curriculum_learning import CurriculumManager
from reinforcement_learning.evaluator import RLEvaluationReport, RLEvaluator
from reinforcement_learning.experiment_tracker import RLExperimentTracker
from reinforcement_learning.market_environment import MarketConfig, MarketEnvironment
from reinforcement_learning.trainer import (
    BaseRLAgent,
    TrainingConfig,
    TrainingResult,
    RLTrainer,
    create_agent,
)


logger = get_logger("ReinforcementLearningEngine")


@dataclass
class RLEngineConfig:
    """Master configuration for the ReinforcementLearningEngine."""

    algorithm: str = "DQN"           # 'DQN', 'DOUBLE_DQN', 'DUELING_DQN', 'PPO', 'A2C', 'SAC', 'TD3', 'DDPG', 'RAINBOW'
    n_episodes: int = 50
    max_steps_per_episode: int = 200
    batch_size: int = 32
    lr: float = 1e-3
    gamma: float = 0.99
    buffer_capacity: int = 10_000
    initial_equity: float = 10_000.0
    lookback: int = 20
    use_curriculum: bool = False
    n_eval_episodes: int = 5
    checkpoint_dir: str = "checkpoints/rl"
    export_reports: bool = False
    author: str = "QuantLabRL"


@dataclass
class RLEngineResult:
    """Dataclass encapsulating complete RL pipeline outputs."""

    algorithm: str
    agent_id: str
    n_episodes: int
    episode_rewards: List[float]
    loss_history: List[float]
    action_distribution: Dict[int, int]
    metrics: Dict[str, float]
    evaluation_report: RLEvaluationReport
    execution_time_seconds: float = 0.0


class ReinforcementLearningEngine:
    """Master Institutional Reinforcement Learning Research Engine.

    Coordinates environments, agents, curriculum learning, training,
    evaluation, MLOps registry, experiment tracking, and reporting.
    """

    def __init__(self, config: Optional[RLEngineConfig] = None) -> None:
        """Initialize ReinforcementLearningEngine."""
        self.config = config or RLEngineConfig()

        self.agent_registry = RLAgentRegistry()
        self.experiment_tracker = RLExperimentTracker()
        self.checkpoint_manager = RLCheckpointManager(self.config.checkpoint_dir)
        self.curriculum_manager = CurriculumManager() if self.config.use_curriculum else None

        self._df: Optional[pd.DataFrame] = None
        self._asset_symbol: str = "GENERIC"
        self._active_agent: Optional[BaseRLAgent] = None
        self._active_env: Optional[MarketEnvironment] = None

    def load_data(self, df: pd.DataFrame, asset_symbol: str = "GENERIC") -> None:
        """Load market DataFrame for environment creation.

        Args:
            df: OHLCV market DataFrame.
            asset_symbol: Asset symbol identifier.
        """
        self._df = df.reset_index(drop=True)
        self._asset_symbol = asset_symbol
        logger.info(f"Loaded market data: Asset='{asset_symbol}', Shape={df.shape}")

    def start_pipeline(self) -> RLEngineResult:
        """Execute the full RL research pipeline end-to-end.

        Pipeline stages:
          1. Create MarketEnvironment from loaded DataFrame.
          2. Apply curriculum stage transformations (if enabled).
          3. Instantiate RL agent via algorithm factory.
          4. Train agent for n_episodes via RLTrainer.
          5. Evaluate trained policy deterministically.
          6. Register agent in MLOps RLAgentRegistry.
          7. Log run in RLExperimentTracker.

        Returns:
            RLEngineResult summary dataclass.
        """
        if self._df is None:
            raise RuntimeError("No market data loaded. Call load_data() first.")

        start_time = time.time()
        logger.info(f"Starting RL Pipeline: Algorithm='{self.config.algorithm}', Asset='{self._asset_symbol}'...")

        # 1. Optionally apply curriculum stage to DataFrame
        df = self._df.copy()
        curriculum_stage = 0
        if self.curriculum_manager is not None:
            df = self.curriculum_manager.apply_to_dataframe(df)
            curriculum_stage = self.curriculum_manager.stage_idx

        # 2. Create Market Environment
        market_cfg = MarketConfig(
            initial_equity=self.config.initial_equity,
            lookback=self.config.lookback,
            max_steps=self.config.max_steps_per_episode,
        )
        self._active_env = MarketEnvironment(df=df, config=market_cfg)

        # 3. Determine observation dimension from env
        obs_dim = self._active_env.observation_space.shape[0]
        n_actions = self._active_env.action_space.n or 7

        # 4. Instantiate RL Agent
        self._active_agent = create_agent(
            algorithm=self.config.algorithm,
            obs_dim=obs_dim,
            n_actions=n_actions,
            lr=self.config.lr,
            gamma=self.config.gamma,
        )
        logger.info(f"Agent instantiated: {self._active_agent.__class__.__name__}, obs_dim={obs_dim}, n_actions={n_actions}")

        # 5. Train Agent
        training_cfg = TrainingConfig(
            n_episodes=self.config.n_episodes,
            max_steps_per_episode=self.config.max_steps_per_episode,
            batch_size=self.config.batch_size,
            buffer_capacity=self.config.buffer_capacity,
            gamma=self.config.gamma,
            lr=self.config.lr,
        )
        trainer = RLTrainer(agent=self._active_agent, env=self._active_env, config=training_cfg)
        training_result: TrainingResult = trainer.train()

        logger.info(
            f"Training completed: episodes={training_result.n_episodes}, "
            f"mean_reward={training_result.mean_reward:.4f}, "
            f"best_reward={training_result.best_reward:.4f}"
        )

        # Log curriculum episode rewards
        if self.curriculum_manager is not None:
            for ep_r in training_result.episode_rewards:
                self.curriculum_manager.log_episode_reward(ep_r)

        # 6. Evaluate Trained Policy
        eval_env = MarketEnvironment(df=df, config=market_cfg)
        eval_report: RLEvaluationReport = RLEvaluator.evaluate(
            agent=self._active_agent,
            env=eval_env,
            n_episodes=self.config.n_eval_episodes,
            max_steps=self.config.max_steps_per_episode,
        )

        # 7. Register in MLOps Agent Registry
        agent_record = self.agent_registry.register(
            name=self.config.algorithm,
            algorithm=self.config.algorithm,
            hyperparameters={
                "lr": self.config.lr,
                "gamma": self.config.gamma,
                "n_episodes": self.config.n_episodes,
                "batch_size": self.config.batch_size,
            },
            dataset_info={"asset": self._asset_symbol, "n_rows": len(df)},
            metrics=eval_report.metrics,
            author=self.config.author,
            status="EXPERIMENTAL",
        )

        exec_time = time.time() - start_time

        # 8. Log to Experiment Tracker
        self.experiment_tracker.log_run(
            experiment_name=f"RL_{self.config.algorithm}_{self._asset_symbol}",
            algorithm=self.config.algorithm,
            hyperparameters={"lr": self.config.lr, "gamma": self.config.gamma},
            episode_rewards=training_result.episode_rewards,
            loss_history=training_result.loss_history,
            action_distribution=eval_report.action_counts,
            curriculum_stages_completed=curriculum_stage,
            metrics=eval_report.metrics,
            duration_seconds=exec_time,
            author=self.config.author,
        )

        logger.info(
            f"RL Pipeline completed in {exec_time:.2f}s | "
            f"Mean Reward={eval_report.metrics.get('mean_reward', 0):.4f} | "
            f"Sharpe={eval_report.metrics.get('sharpe_ratio', 0):.4f}"
        )

        return RLEngineResult(
            algorithm=self.config.algorithm,
            agent_id=agent_record.agent_id,
            n_episodes=self.config.n_episodes,
            episode_rewards=training_result.episode_rewards,
            loss_history=training_result.loss_history,
            action_distribution=eval_report.action_counts,
            metrics=eval_report.metrics,
            evaluation_report=eval_report,
            execution_time_seconds=exec_time,
        )

    def predict_action(self, obs: np.ndarray, deterministic: bool = True) -> int:
        """Predict a trading action for the given observation.

        Args:
            obs: Normalized observation vector.
            deterministic: Whether to use greedy (True) or stochastic policy.

        Returns:
            Integer action index from DiscreteAction.
        """
        if self._active_agent is None:
            raise RuntimeError("No trained agent available. Call start_pipeline() first.")
        return self._active_agent.select_action(obs, deterministic=deterministic)
