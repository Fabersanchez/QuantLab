# QuantLab
### Institutional Quantitative Research Laboratory

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Architecture](https://img.shields.io/badge/Architecture-Modular-green.svg)](#-system-architecture)
[![Test Suite](https://img.shields.io/badge/Tests-233%20Passing-brightgreen.svg)](#-testing--quality-assurance)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-orange.svg)](#-roadmap--project-milestones)

---

## 📌 Overview

**QuantLab** is an institutional-grade, modular quantitative research and execution framework designed for developing, backtesting, optimizing, and deploying quantitative trading strategies, financial models, and Machine Learning / Deep Learning / Reinforcement Learning research pipelines.

Engineered with high software architecture standards, **QuantLab** ensures strict component decoupling, vertical and horizontal scalability, deterministic reproducibility, and mathematical rigor in processing financial time series.

---

## 🎯 Key Objectives

- **Rigorous Quantitative Research:** Provide a structured environment for hypothesis formulation, backtesting, and validation while eliminating look-ahead bias and data snooping.
- **Decoupled Modular Architecture:** Enable seamless interchangeability of data sources, indicator calculation engines, Machine Learning models, optimization algorithms, and execution rules without altering core logic.
- **Multi-Paradigm ML / DL / RL Integration:** Native support for predictive statistical models, Deep Learning architectures (LSTM, Transformer), and Reinforcement Learning agents (PPO, DDPG, SAC, DQN, Rainbow) built for market dynamics.
- **Enterprise-Grade Verification:** Integrated Walk-Forward Analysis, Monte Carlo stress testing, risk budgeting, portfolio optimization, and lineage tracking.
- **Institutional Software Standards:** Built with Python 3.10+, strict type hinting, event-driven orchestration, full coverage test suites (233+ tests), and PEP 8 compliance.

---

## 💡 Core Philosophy

1. **Simplicity First (KISS):** Clean, expressive abstractions prior to introducing systemic complexity.
2. **Strict Component Decoupling:** Core orchestration, data ingestion, indicator pipelines, strategy rules, portfolio allocation, and visualization modules maintain zero redundant cross-dependencies.
3. **Deterministic Reproducibility:** Every experiment, random seed, generated signal, walk-forward split, and risk metric is fully deterministic and auditable.
4. **Production Readiness:** Smooth transition from exploratory Jupyter notebooks to high-performance simulation engines and Studio GUI applications.

---

## 🏗 System Architecture

QuantLab adopts an **Event-Driven & Component-Based Modular Architecture**:

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               QuantLab Application Layer                               │
│                ┌───────────────────────────┐   ┌───────────────────────────┐           │
│                │       app / main.py       │   │  studio / studio_app.py   │           │
│                │         (CLI Engine)      │   │       (Studio GUI)        │           │
│                └─────────────┬─────────────┘   └─────────────┬─────────────┘           │
└──────────────────────────────┼───────────────────────────────┼─────────────────────────┘
                               │                               │
                               ▼                               ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 core / QuantEngine                                     │
│  (LifecycleManager | ComponentRegistry | EventBus | ModuleManager | QuantLogger)       │
└───────┬───────────┬───────────┬───────────┬───────────┬───────────┬───────────┬────────┘
        │           │           │           │           │           │           │
        ▼           ▼           ▼           ▼           ▼           ▼           ▼
┌──────────────┐┌──────────┐┌───────────┐┌───────────┐┌───────────┐┌───────────┐┌─────────┐
│ data_engine/ ││indicators││strategies/││backtesting││ portfolio/││walk_forward││monte_c. │
│ Ingestion &  ││Vectorized││Signals &  ││Vectorized ││Allocation ││Anchored & ││Resampling│
│ Connectors   ││Transforms││Composition││& Event-Dr.││& Sizing   ││Rolling WF ││& Stress │
└──────────────┘└──────────┘└───────────┘└───────────┘└───────────┘└───────────┘└─────────┘
        │           │           │           │           │           │           │
        ▼           ▼           ▼           ▼           ▼           ▼           ▼
┌──────────────┐┌──────────┐┌───────────┐┌───────────┐┌───────────┐┌───────────┐┌─────────┐
│optimization/ ││ machine_ ││   deep_   ││reinforce_ ││ research/ ││visualizat.││registry/│
│Genetic/Bayes ││ learning ││ learning  ││ learning  ││Experiments││Charts &   ││Lineage &│
│/Grid Search  ││& Feature ││LSTM/Transf││PPO/DDPG/RL││& Hypothes.││Heatmaps   ││Appr. WF │
└──────────────┘└──────────┘└───────────┘└───────────┘└───────────┘└───────────┘└─────────┘
```

---

## 🧰 Subsystem & Module Highlights

### ⚙️ Core Engine (`core/`)
- **`QuantEngine`**: Central orchestrator providing unified control over component initialization, execution states, and module lifecycles.
- **Event Bus (`event_bus.py`)**: Asynchronous, decoupled event publisher/subscriber bus for intra-system communication (`SYSTEM_INITIALIZING`, `SYSTEM_RUNNING`, strategy signals, trade events).
- **Component Registry (`registry.py`)**: Thread-safe dynamic dependency injection registry.
- **Lifecycle Manager (`lifecycle.py`)**: Strict state transition machine (`CREATED` ➔ `INITIALIZING` ➔ `READY` ➔ `RUNNING` ➔ `STOPPING` ➔ `STOPPED`).

### 📊 Data Engine (`data/`, `data_engine/`)
- **Connectors**: Automated data fetchers for Yahoo Finance, Binance API, local Parquet, CSV, and streaming data sockets.
- **Data Pipeline**: Automated data cleaning, outlier removal, missing value imputation, multi-timeframe alignment, and standard formatting.

### 📈 Technical Indicators Library (`indicators/`)
- **Vectorized Calculations**: High-throughput NumPy/Pandas implementations of Trend (SMA, EMA, WMA, MACD, ADX), Oscillators (RSI, Stochastic, CCI, ROC), Volatility (ATR, Bollinger Bands, Keltner), Volume (OBV, VWAP, CMF), and Custom mathematical transforms.

### 🧠 Strategy & Signal Composition (`strategies/`)
- **Strategy Framework**: Abstract base class supporting multi-asset, multi-factor strategies.
- **Strategy Implementations**: Trend-following, Mean-reversion, Statistical Arbitrage (Pairs Trading), and ML/RL signal-driven strategies.
- **Signal Composition Engine**: Dynamic signal weighting, thresholding, filtering, and strategy validation/export routines.

### ⚡ Backtesting & Execution Engine (`backtesting/`)
- **Dual Engine Architecture**: Supports fast Vectorized Backtesting for rapid exploration and granular Event-Driven Backtesting for market realistic execution simulation.
- **Order & Execution Platform**: Limit, Market, Stop-loss, Take-profit order execution with configurable slippage models (fixed, dynamic, volume-based) and commission structures (flat fee, percentage, tiered).
- **Performance Analytics**: Automated calculation of Sharpe Ratio, Sortino Ratio, Calmar Ratio, Max Drawdown, CAGR, Win Rate, Profit Factor, Expectancy, and Risk-Adjusted metrics.

### 💼 Portfolio Management & Allocation (`portfolio/`)
- **Position Sizing Engine**: Fixed Fractional, Risk Parity, Volatility Parity, Kelly Criterion, and Target Risk allocation models.
- **Rebalancing Framework**: Dynamic rebalancing (calendar-based, threshold-based) and multi-asset capital allocation.

### 🔄 Walk-Forward Optimization & Validation (`walk_forward/`)
- **Window Generators**: Anchored, Rolling, and Expanding window generators for out-of-sample parameter optimization.
- **Robustness & Efficiency Engine**: Robustness index computation, Out-of-Sample Efficiency (WFE) ratios, parameter stability testing, and automated PDF/HTML report generation.

### 🎲 Monte Carlo Simulation Engine (`monte_carlo/`)
- **Simulation Suite**: Historical return resampling (Block Bootstrap, Stationary Bootstrap), Geometric Brownian Motion path generation, equity curve distribution analysis.
- **Stress Testing & Risk Estimation**: Sensitivity analysis, stress testing against market regimes, Value at Risk (VaR), and Conditional Value at Risk (CVaR / Expected Shortfall).

### 🎯 Strategy & Portfolio Optimization (`optimization/`)
- **Multi-Algorithm Optimization**: Grid Search, Random Search, Genetic Algorithms (NSGA-II inspired), Bayesian Optimization, and Multi-Objective Pareto optimization.

### 🤖 Machine Learning & Deep Learning (`machine_learning/`, `deep_learning/`)
- **Feature Engineering**: Automated lag feature generation, rolling statistical metrics, fractional differentiation, and feature importance rankings.
- **Predictive Models**: Scikit-Learn integration for classifiers, regressors, ensemble models (Random Forest, Gradient Boosting), model validation pipelines, and hyperparameter tuning.
- **Deep Learning Architectures**: PyTorch/TensorFlow-compatible LSTM and Transformer models specialized for financial time-series forecasting.

### 🎮 Reinforcement Learning Suite (`reinforcement_learning/`)
- **Trading Environments**: OpenAI Gym / Farama Gymnasium compatible market environments with configurable state builders, discrete/continuous action spaces, and custom reward functions (Sharpe reward, drawdown penalties).
- **RL Algorithms**: PPO, DDPG, SAC, DQN, Double DQN, Dueling DQN, Rainbow, A2C/A3C, and TD3.
- **Curriculum & Policy Management**: Automated curriculum learning, policy cloning, checkpoint management, and replay buffers (Prioritized, N-step).

### 🔬 Research & Experimentation Platform (`research/`)
- **Experiment Tracker**: Structured logging of research experiments, hyperparameters, performance metrics, dataset signatures, and reproducible artifacts.
- **Anti-Overfitting Policies**: Built-in score rating models, anti-winrate bias rules, and statistical significance testing (t-test, p-value verification).

### 🎨 Advanced Financial Visualization Engine (`visualization/`)
- **Institutional Renderers**: Interactive Candlestick charts, Equity curves with benchmark overlays, Drawdown underwater charts, Correlation heatmaps, Monte Carlo simulation fans, Walk-Forward window heatmaps, and Trade distribution plots.
- **Export & Themes**: Built-in dark/light themes, chart caching, video/GIF animations, and high-resolution SVG/PNG report exporters.

### 💻 QuantLab Studio Platform (`studio/`)
- **GUI Application Shell**: Complete desktop/web interactive workbench (`studio/studio_app.py`) featuring an IoC Service Container, Workspace Manager, Task Engine, Worker Framework, Session Recovery, Perspective Manager, Theme Engine, and Monitoring Services.

### 📜 Component & Strategy Registry (`registry/`)
- **Lineage & Governance**: Model registry, strategy version control, cryptographic hash verification, metadata signatures, lineage graph tracking, and deployment approval workflows.

---

## 📁 Repository Directory Structure

```text
QuantLab/
├── app/                        # CLI application entry points
│   ├── __init__.py
│   └── main.py                 # Core CLI runner script
├── core/                       # QuantEngine framework & lifecycle orchestration
│   ├── engine.py               # Main QuantEngine orchestrator
│   ├── event_bus.py            # Decoupled event bus
│   ├── lifecycle.py            # System state machine
│   ├── logger.py               # Institutional logging wrapper
│   ├── module_manager.py       # Plugin & module management
│   └── registry.py             # Dynamic component registry
├── data_engine/                # Market data ingestion & connectors
│   ├── cleaners/               # Data cleaning & normalization
│   ├── connectors/             # API connectors (Yahoo, Binance, Parquet/CSV)
│   └── storage/                # High-performance data persistence
├── indicators/                 # Vectorized technical indicator library
│   ├── custom.py               # Mathematical & custom transforms
│   ├── momentum.py             # Momentum indicators (RSI, Stochastic, etc.)
│   ├── moving_averages.py      # Moving averages (SMA, EMA, WMA)
│   ├── oscillators.py          # Oscillators (MACD, CCI, ROC)
│   ├── volatility.py           # Volatility indicators (ATR, Bollinger Bands)
│   └── volume.py               # Volume indicators (OBV, VWAP, CMF)
├── strategies/                 # Quantitative strategy & signal engine
│   ├── base_strategy.py        # Strategy base class
│   ├── composition/            # Signal composition & combination
│   ├── mean_reversion.py       # Mean reversion strategies
│   ├── stat_arb.py             # Statistical arbitrage strategies
│   └── trend_following.py      # Trend following strategies
├── backtesting/                # Backtesting & order execution engine
│   ├── backtest_engine.py      # Vectorized & Event-Driven backtesters
│   ├── execution_engine.py     # Execution simulation platform
│   ├── metrics.py              # Performance analytics (Sharpe, Sortino, etc.)
│   ├── order_management.py     # Order types & state tracking
│   ├── risk_engine.py          # Real-time risk evaluation
│   └── slippage.py             # Slippage & commission models
├── portfolio/                  # Portfolio management & capital allocation
│   ├── allocation.py           # Capital allocation strategies
│   ├── portfolio_manager.py    # Master portfolio coordinator
│   ├── position_sizing.py      # Position sizing algorithms
│   └── rebalancing.py          # Portfolio rebalancing routines
├── walk_forward/               # Walk-Forward optimization & validation
│   ├── efficiency.py           # Walk-forward efficiency (WFE) metrics
│   ├── report_generator.py     # PDF / HTML report generator
│   ├── robustness_metrics.py   # Robustness score calculation
│   ├── validation_runner.py    # Out-of-sample validation runner
│   ├── walkforward_engine.py   # Main walk-forward engine
│   └── window_generator.py     # Anchored, rolling & expanding windows
├── monte_carlo/                # Monte Carlo simulation & stress testing
│   ├── montecarlo_engine.py    # Monte Carlo simulation engine
│   ├── path_generator.py       # Return & equity path generation
│   ├── resampling.py           # Block & stationary bootstrap
│   ├── sensitivity_analysis.py # Parameter sensitivity analyzer
│   └── stress_testing.py       # Scenario stress testing & VaR/CVaR
├── optimization/               # Strategy & portfolio optimization framework
│   ├── bayesian.py             # Bayesian optimization engine
│   ├── genetic_algorithm.py    # Genetic algorithm optimizer (NSGA-II)
│   ├── grid_search.py          # Grid & random search
│   └── multi_objective.py      # Multi-objective Pareto optimization
├── machine_learning/           # Machine learning framework & feature pipeline
│   ├── feature_engineering.py  # Lag, rolling, & fractional features
│   ├── model_validation.py     # Cross-validation & anti-overfitting
│   └── pipeline.py             # ML pipeline orchestrator
├── deep_learning/              # Deep Learning time-series models
│   ├── dl_engine.py            # Deep learning engine framework
│   ├── lstm_model.py           # Recurrent LSTM models for time series
│   └── transformer_model.py    # Temporal Transformer models
├── reinforcement_learning/     # Reinforcement Learning suite
│   ├── agents.py               # RL agents (PPO, DDPG, SAC, DQN, etc.)
│   ├── curriculum.py           # Curriculum learning manager
│   ├── rewards.py              # Custom financial reward functions
│   └── rl_environment.py       # Gymnasium market environment
├── research/                   # Research platform & experiment tracker
│   ├── experiment_tracker.py   # Experiment tracking & metadata registry
│   ├── hypothesis_testing.py   # Hypothesis testing framework
│   └── metrics_calculator.py   # Research scoring & validation rules
├── studio/                     # QuantLab Studio GUI platform
│   ├── studio_app.py           # Main Studio application startup script
│   ├── shell/                  # Application shell & window layout
│   ├── services/               # IoC service container & core services
│   ├── workspace/              # Workspace manager & project state
│   ├── widgets/                # Interactive UI widget framework
│   └── task_engine/            # Background worker & task engine
├── visualization/              # Advanced financial plotting engine
│   ├── chart_manager.py        # Master chart manager
│   ├── equity_curve.py         # Equity curve & drawdown renderers
│   ├── heatmap.py              # Optimization & correlation heatmaps
│   └── visualization_engine.py # Unified visualization interface
├── registry/                   # Component registry & lineage tracking
│   ├── lineage_graph.py        # Lineage dependency graph
│   └── master_registry.py      # Version control & approval workflow
├── tests/                      # Unit & integration test suite (233 tests)
├── assets/                     # Graphic assets & diagrams
├── docs/                       # Technical documentation & reports
├── notebooks/                  # Exploratory Jupyter research notebooks
├── .gitignore                  # Optimized Git exclusions
├── LICENSE                     # MIT License file
├── README.md                   # Repository documentation (English)
└── requirements.txt            # Base Python dependencies
```

---

## 🛠 Tech Stack & Dependencies

- **Programming Language:** [Python 3.10+](https://www.python.org/)
- **Numerical Computing:** [NumPy](https://numpy.org/)
- **Data Manipulation:** [Pandas](https://pandas.pydata.org/)
- **Testing Framework:** [Pytest](https://docs.pytest.org/)
- **Version Control:** [Git](https://git-scm.com/) / [GitHub](https://github.com/)

---

## 🚀 Quick Start Guide

### Prerequisites
Ensure Python 3.10+ is installed on your system.

### 1. Installation
Clone the repository and install core dependencies:

```bash
git clone https://github.com/Fabersanchez/QuantLab.git
cd QuantLab
pip install -r requirements.txt
```

### 2. Running QuantLab CLI Core Engine
To initialize and launch the QuantLab core framework:

```bash
python app/main.py
```

### 3. Launching QuantLab Studio Platform
To launch the desktop/web Studio application workbench:

```bash
python studio/studio_app.py
```

---

## 🧪 Testing & Quality Assurance

QuantLab features a comprehensive test suite of **233 automated unit and integration tests** covering all 20+ subsystems.

Run the test suite using `pytest`:

```bash
python -m pytest
```

To run tests with detailed output:

```bash
python -m pytest -v --tb=short
```

---

## 🚀 Roadmap & Project Milestones

- [x] **Phase 1:** Laboratory architecture design, repository initialization, and standards definition.
- [x] **Phase 2:** Core `QuantEngine` v0.1.0 lifecycle manager, component registry, and event bus implementation.
- [x] **Phase 3:** Financial Data Engine, ingestion connectors (Yahoo, Binance, Parquet, CSV), and data cleaners.
- [x] **Phase 4:** Vectorized Technical Indicators library (Moving Averages, Oscillators, Volatility, Volume, Custom).
- [x] **Phase 5:** Strategy framework, signal composition, and execution signal generators.
- [x] **Phase 6:** Backtesting engine (Vectorized & Event-Driven), order execution simulator, and risk metrics analytics.
- [x] **Phase 7:** Portfolio manager, position sizing algorithms, capital allocation, and dynamic rebalancing routines.
- [x] **Phase 8:** Walk-Forward Optimization Engine with anchored, rolling, and expanding windows + report generator.
- [x] **Phase 9:** Monte Carlo simulation engine, bootstrap resampling, sensitivity analysis, and VaR/CVaR stress testing.
- [x] **Phase 10:** Strategy & Portfolio Optimization Suite (Grid, Genetic Algorithms, Bayesian, Multi-Objective Pareto).
- [x] **Phase 11:** Machine Learning, Deep Learning (LSTM, Transformer), and Reinforcement Learning (PPO, DDPG, SAC, DQN) engines.
- [x] **Phase 12:** Quantitative Research Platform, experiment tracking, hypothesis testing, and anti-overfitting policies.
- [x] **Phase 13:** Advanced Financial Visualization engine (interactive charts, equity curves, heatmaps, animations).
- [x] **Phase 14:** QuantLab Studio GUI Workbench Platform, IoC container, widgets, worker framework, and workspace manager.
- [x] **Phase 15:** Master registry, lineage graph tracking, metadata signatures, and deployment approval workflows.
- [x] **Phase 16:** Complete unit and integration test suite verification (233 passing tests).

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---

## 👨‍💻 Author & Engineering Team

**QuantLab Engineering Team**  
*Institutional Quantitative Research & Software Architecture*
