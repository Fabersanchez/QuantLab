"""
QuantLab Institutional Reinforcement Learning Research Lab Package.

Complete MLOps and RL infrastructure for financial market decision-making:
Gymnasium-compatible market environment, state/observation builder,
discrete action space (HOLD, BUY, SELL, CLOSE, PARTIAL_CLOSE, MODIFY_SL, MODIFY_TP),
multi-objective reward engine (Profit, Sharpe, Sortino, Drawdown, Capital Preservation),
experience replay buffers (Uniform, Prioritized PER, N-step),
exploration strategies (Epsilon-greedy, Softmax, UCB, Entropy, NoisyNetworks),
progressive curriculum learning (5 market regime stages),
RL algorithm agents (DQN, Double DQN, Dueling DQN, PPO, A2C, A3C, SAC, TD3, DDPG, Rainbow),
policy manager, checkpoint manager, agent registry, experiment tracker,
SVG visual analytics (reward curves, action distributions, learning progress),
multi-format reporting (HTML, Markdown, PDF, JSON, CSV),
and master ReinforcementLearningEngine orchestrator.
"""

from reinforcement_learning.environment import BaseEnvironment, SpaceSpec, StepResult
from reinforcement_learning.action_space import DiscreteAction, ActionSpace, ACTION_LABELS, N_DISCRETE_ACTIONS
from reinforcement_learning.state_builder import PortfolioState, StateBuilder
from reinforcement_learning.reward_engine import RewardConfig, RewardEngine
from reinforcement_learning.market_environment import MarketConfig, MarketEnvironment
from reinforcement_learning.replay_buffer import (
    Transition,
    UniformReplayBuffer,
    PrioritizedReplayBuffer,
    NStepReplayBuffer,
)
from reinforcement_learning.exploration import (
    EpsilonGreedyExploration,
    SoftmaxExploration,
    UCBExploration,
    EntropyExploration,
    NoisyNetworksExploration,
)
from reinforcement_learning.curriculum_learning import CurriculumStage, CurriculumManager, CURRICULUM_STAGES
from reinforcement_learning.agent_registry import RLAgentRecord, RLAgentRegistry
from reinforcement_learning.policy_manager import PolicyManager
from reinforcement_learning.checkpoint_manager import RLCheckpoint, RLCheckpointManager
from reinforcement_learning.trainer import (
    BaseRLAgent,
    DQNAgent,
    DoubleDQNAgent,
    DuelingDQNAgent,
    RainbowDQNAgent,
    PPOAgent,
    A2CAgent,
    A3CAgent,
    SACAgent,
    TD3Agent,
    DDPGAgent,
    create_agent,
    TrainingConfig,
    TrainingResult,
    RLTrainer,
)
from reinforcement_learning.evaluator import RLEvaluationReport, RLEvaluator
from reinforcement_learning.experiment_tracker import RLExperimentRun, RLExperimentTracker
from reinforcement_learning.visualization import RLVisualizer
from reinforcement_learning.report_generator import RLReportGenerator
from reinforcement_learning.rl_engine import RLEngineConfig, RLEngineResult, ReinforcementLearningEngine

__all__ = [
    # Environment
    "BaseEnvironment", "SpaceSpec", "StepResult",
    # Action Space
    "DiscreteAction", "ActionSpace", "ACTION_LABELS", "N_DISCRETE_ACTIONS",
    # State Builder
    "PortfolioState", "StateBuilder",
    # Reward
    "RewardConfig", "RewardEngine",
    # Market Environment
    "MarketConfig", "MarketEnvironment",
    # Replay Buffers
    "Transition", "UniformReplayBuffer", "PrioritizedReplayBuffer", "NStepReplayBuffer",
    # Exploration
    "EpsilonGreedyExploration", "SoftmaxExploration", "UCBExploration",
    "EntropyExploration", "NoisyNetworksExploration",
    # Curriculum
    "CurriculumStage", "CurriculumManager", "CURRICULUM_STAGES",
    # Registry & Policy
    "RLAgentRecord", "RLAgentRegistry",
    "PolicyManager",
    # Checkpoint
    "RLCheckpoint", "RLCheckpointManager",
    # Algorithms & Trainer
    "BaseRLAgent",
    "DQNAgent", "DoubleDQNAgent", "DuelingDQNAgent", "RainbowDQNAgent",
    "PPOAgent", "A2CAgent", "A3CAgent",
    "SACAgent", "TD3Agent", "DDPGAgent",
    "create_agent",
    "TrainingConfig", "TrainingResult", "RLTrainer",
    # Evaluator
    "RLEvaluationReport", "RLEvaluator",
    # Experiment Tracker
    "RLExperimentRun", "RLExperimentTracker",
    # Visualization
    "RLVisualizer",
    # Reporting
    "RLReportGenerator",
    # Engine
    "RLEngineConfig", "RLEngineResult", "ReinforcementLearningEngine",
]
