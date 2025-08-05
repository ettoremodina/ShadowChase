# 🕵️ Shadow Chase - AI Agent Training & Evaluation Platform

> **⚖️ Legal Notice**: This is an independent educational/research implementation inspired by the Scotland Yard board game. This project is not affiliated with or endorsed by Ravensburger AG (owners of Scotland Yard trademark). For educational and research use only.

## 🔗 Quick Links

**🚀 Getting Started:** [Installation](#installation) • [Quick Start](#quick-start) • [Usage Examples](#-usage-examples)  
**📚 Documentation:** [Terminal Guide](game_controls/README.md) • [API Reference](other/FUNCTION_DOCUMENTATION.md) • [Configuration](#️-configuration)  
**🤖 Development:** [Training](#-train-new-dqn-agents) • [Contributing](#-contributing) • [Troubleshooting](#-troubleshooting--support)

---

A comprehensive implementation of the Shadow Chase pursuit-evasion game with multiple AI agents for research, learning, and experimentation with reinforcement learning and game theory techniques.

## 🎯 Project Overview

This project aims to provide a platform for:
- **Learning Deep Q-Learning (DQN)** and comparing it to other AI techniques
- **Creating a testing environment** for various AI strategies and algorithms
- **Exploring game theory** in pursuit-evasion scenarios
- **Benchmarking different agent implementations** against each other

## 📋 Table of Contents

### 🚀 Getting Started
- [How Shadow Chase Works](#-how-shadow-chase-works)
- [Quick Start](#-quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Three Main Ways to Use This Project](#three-main-ways-to-use-this-project)

### 🏗️ Project Structure & Documentation
- [Project Structure](#️-project-structure)
- [Available Agent Types](#-available-agent-types)
- [Configuration](#-configuration)

### 🎮 Usage & Examples
- [Usage Examples](#-usage-examples)
- [Examples & Tutorials](#-examples--tutorials)
- [Terminal Interface Guide](game_controls/README.md)
- [Function Documentation](other/FUNCTION_DOCUMENTATION.md)

### 🤖 Development & Training
- [Configuration](#️-configuration)
- [Training Configuration Files](#training-configuration-files)
- [Troubleshooting & Support](#-troubleshooting--support)
- [Future Development & Improvements](#-future-development--improvements)
- [Contributing](#-contributing)

### 📚 Research & Legal
- [References](#-references)
- [License & Legal Notice](#-license--legal-notice)
- [How to Cite This Work](#-how-to-cite-this-work)

---

## 🎲 How Shadow Chase Works

Shadow Chase is an asymmetric pursuit-evasion game with imperfect information, inspired by Shadow Chase:

- **MrX (1 player)**: The fugitive who moves secretly around the city using taxi, bus, underground, and black tickets. MrX becomes visible only on certain turns and has special abilities like double moves.

- **Detectives (5 players)**: Work together to capture the MrX by landing on their position. They have limited transport tickets and their movements are always visible.

- **Victory Conditions**:
  - **Detectives win**: If any agent lands on MrX's position
  - **MrX wins**: If they evade capture for 24 turns or if Detectives run out of moves

- **Transport Network**: Players move through the city using three transport types:
  - 🚕 **Taxi**: Available everywhere, limited tickets
  - 🚌 **Bus**: Faster routes, fewer connections
  - 🚇 **Underground**: Fastest but most limited network


## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- 8GB+ RAM (recommended for DQN training)
- CUDA-compatible GPU (optional, for faster DQN training)
- Git (for cloning the repository)

### Installation
```bash
git clone https://github.com/ettoremodina/ShadowChase.git
cd ShadowChase
pip install -r requirements.txt
```

### Three Main Ways to Use This Project

#### 1. 🎮 Interactive Play (Graphical Interface)
Play Shadow Chase with a visual interface - control Detectives manually or watch AI agents play:
```bash
python main.py
```
- Choose between human and AI players
- Watch agent strategies in action
- Great for understanding game dynamics

#### 2. 🧪 Agent Testing & Comparison
Systematically test different agent combinations and analyze their performance:
```bash
python test_agents.py
```
- Runs all agent combinations automatically
- Generates detailed performance analysis
- Perfect for benchmarking and research

#### 3. 🤖 Train New DQN Agents
Train your own Deep Q-Learning agents:
```bash
python train_dqn.py --role MrX --episodes 10000
python train_dqn.py --role detectives --episodes 10000
```
- Train either MrX or agent teams
- Supports CPU and GPU training
- Saves models for later use

## 🏗️ Project Structure

```
ShadowChase/
├── 📁 agents/                    # AI agent implementations
│   ├── base_agent.py            # Abstract base classes
│   ├── random_agent.py          # Random baseline agents
│   ├── heuristic_agent.py       # Distance-based strategy agents
│   ├── mcts_agent.py            # Monte Carlo Tree Search agents
│   ├── dqn_agent.py             # Deep Q-Learning agents
│   └── agent_registry.py        # Agent selection system
├── 📁 game_controls/             # Terminal-based gameplay
│   ├── simple_game.py           # Command-line interface
│   ├── game_logic.py            # Game flow control
│   └── display_utils.py         # Terminal formatting
├── 📁 training/                  # Training infrastructure
│   ├── deep_q/                  # DQN-specific components
│   ├── feature_extractor.py     # State representation
│   └── training_environment.py  # Standardized training interface
├── 📁 ShadowChase/              # Core game engine
│   ├── core/                    # Game logic and rules
│   ├── ui/                      # Graphical interface
│   └── services/                # Utilities and caching
├── 📁 data/                     # Game maps and configurations
└── 📁 papers/                   # Research literature
```

### Key Files

#### Main Entry Points
- **[main.py](main.py)**: Interactive graphical interface
- **[test_agents.py](test_agents.py)**: Agent testing and comparison
- **[train_dqn.py](train_dqn.py)**: Deep Q-Learning training
- **[game_controls/simple_game.py](game_controls/simple_game.py)**: Terminal interface

#### Configuration & Data
- **[training/configs/](training/configs/)**: Training configuration files
- **[data/](data/)**: Board data and game configurations
- **[requirements.txt](requirements.txt)**: Project dependencies

## 🤖 Available Agent Types

### Basic Agents
- **Random Agent**: Makes random valid moves (baseline)
- **Heuristic Agent**: Uses distance-based strategies
  - MrX: Maximizes distance from closest agent
  - Detectives: Minimize distance to MrX's last known position

### Advanced AI Agents
- **MCTS Agent**: Monte Carlo Tree Search with random simulations
- **Optimized MCTS**: Enhanced with caching for better performance  
- **Epsilon-Greedy MCTS**: MCTS with heuristic-guided simulations
- **DQN Agent**: Deep Q-Learning neural networks

### Performance Notes
- **Cache system works well for MCTS** agents, providing significant speedup
- **Game method caching** is less effective (retrieval cost > recalculation cost)
- **DQN training** benefits from GPU acceleration but runs fine on CPU
- **Memory requirements**: 8GB+ RAM recommended for large training sessions
- **Training time**: DQN training can take several hours for 10,000+ episodes


## 🎮 Usage Examples

### Run a Quick Test
```bash
# Play 10 games between DQN and random agents
python test_agents.py
# Results saved to debug/ folder with analysis graphs
```

### Train a New Agent
```bash
# Train MrX agent for 5000 episodes
python train_dqn.py --role MrX --episodes 5000 --map extracted

# Train detective agents for 5000 episodes
python train_dqn.py --role detectives --episodes 5000 --save-dir my_training
```

### Batch Analysis
```bash
# Terminal gameplay - 50 automated games
python game_controls/simple_game.py --batch 50 --map-size extracted --detectives 5
```

## 🚧 Examples & Tutorials

### Quick Start Examples
- **[main.py](main.py)**: Interactive graphical gameplay with manual or AI control
- **[test_agents.py](test_agents.py)**: Systematic agent comparison and benchmarking
- **[train_dqn.py](train_dqn.py)**: Deep Q-Learning agent training

### Terminal Interface Examples
- **[simple_game.py](game_controls/simple_game.py)**: Command-line gameplay
- **[Terminal Guide](game_controls/README.md)**: Complete terminal interface documentation

### Advanced Examples
- **[Example Games](ShadowChase/examples/example_games.py)**: Different game configurations
- **[Board Creation](other/createBoard.py)**: Interactive board creation tool
- **[Performance Analysis](other/profile_mcts_agent.py)**: MCTS agent profiling

## ⚙️ Configuration

### Training Configuration Files

The project includes several configuration files to customize training and game parameters:

- **[DQN Configuration](training/configs/dqn_config.json)**: Deep Q-Learning network architecture, training parameters, and hyperparameters
- **[MCTS Configuration](training/configs/mcts_config.json)**: Monte Carlo Tree Search parameters and optimization settings
- **[Board Metadata](data/board_metadata.json)**: Game board configuration and transport network settings
- **[Transport Colors](data/transport_colors.json)**: Visualization colors for different transport types

### Training Configuration
Edit `training/configs/dqn_config.json` to adjust:
- Network architecture (hidden layers, dropout)
- Training parameters (learning rate, epsilon decay)
- Experience replay settings

### Cache Settings
The caching system can be configured in your scripts:
```python
from ShadowChase.services.cache_system import *

enable_cache()
enable_namespace_cache(CacheNamespace.MCTS_NODES)      # Good for MCTS
disable_namespace_cache(CacheNamespace.GAME_METHODS)   # Not helpful
```

### Game Configuration
You can customize game parameters in several ways:
- **Map size**: Choose between test (10 nodes), full (199 nodes), or extracted board
- **Number of detectives**: Typically 2-5 detectives
- **Agent types**: Select from Random, Heuristic, MCTS, or DQN agents
- **Verbosity levels**: Control output detail in terminal mode

## � Troubleshooting & Support

### Common Issues

#### Installation Problems
- **CUDA Issues**: DQN training works fine on CPU if GPU is unavailable
- **Missing Dependencies**: Run `pip install -r requirements.txt` to ensure all packages are installed
- **Python Version**: Requires Python 3.8+

#### Performance Issues
- **Slow MCTS**: Enable caching with `enable_namespace_cache(CacheNamespace.MCTS_NODES)`
- **Memory Usage**: Large replay buffers in DQN training may require 8GB+ RAM
- **Training Speed**: Use `--episodes` parameter to control training length

#### Gameplay Issues
- **Invalid Moves**: Check transport type availability and ticket counts
- **Game Not Starting**: Ensure starting positions don't overlap
- **Save/Load Errors**: Check that `saved_games/` directory exists and is writable

## 🚧 Future Development & Improvements

### High Priority Improvements Needed
- **🎨 User Interface Overhaul**: The current GUI needs significant improvement for better usability
- **🤝 Adversarial Training**: Implement adversarial learning to train Deep Learning based agents
- **🔄 Advanced RL Techniques**: Add PPO, A3C, or other modern RL algorithms
- **📊 Better Evaluation**: More sophisticated metrics and analysis tools

### Potential Enhancements
- **🎯 Curriculum Learning**: Progressive difficulty in training scenarios
- **🔗 Multi-Agent RL**: Cooperative learning for agent teams
- **📈 Real-time Visualization**: Live training progress and agent behavior
- **🌐 Web Interface**: Browser-based gameplay and training monitoring

### Research Opportunities
- **Game Theory Analysis**: Nash equilibria and optimal strategies
- **Transfer Learning**: Adapting agents to different map sizes

## 🤝 Contributing

We welcome contributions! Areas where help is especially needed:

1. **UI/UX Improvements**: The current interface needs major enhancements
2. **Advanced RL Algorithms**: Implementing state-of-the-art techniques
3. **Performance Optimization**: Speeding up training and evaluation
4. **Documentation**: Tutorials, examples, and general documentation are still lacking
5. **Testing**: Unit tests and integration tests

## 📚 References

The `papers/` directory contains relevant research literature on:
- Pursuit-evasion games, Scotland Yard, and strategy analysis
- Monte Carlo Tree Search applications
- Deep reinforcement learning for multi-agent systems
- Game theory and optimal strategies

## 📄 License & Legal Notice

### Project License
This project's **source code** is open source under the MIT License. See LICENSE file for details.

### Important Legal Disclaimer
This is an **independent implementation** inspired by the Scotland Yard board game, created for educational and research purposes. This project:
- ✅ **Educational/Research purposes only** - Not for commercial use
- ✅ **Independent implementation** - Original code and AI research
- ✅ **Open source** - Freely available for academic use
- ✅ **Inspired by Scotland Yard** - Acknowledges the original game concept
- ❌ **NOT affiliated** with Ravensburger AG or official Scotland Yard
- ❌ **Does NOT include** any copyrighted artwork or official assets

### Usage Restrictions
- **Academic & Research Use**: ✅ Encouraged for learning RL, game theory, and AI
- **Commercial Use**: ✅ Allowed with proper attribution (see citation below)
- **Distribution**: Feel free to share and modify
- **Attribution**: Please cite this project in academic work

### For Commercial Use
This project uses original content and can be used commercially:
1. Maintain attribution to the original author
2. Follow the MIT License terms
3. Consider contributing improvements back to the community
4. Ensure compliance with any dependencies' licenses

This project exists purely for educational and research purposes in artificial intelligence and game theory.

## 📖 How to Cite This Work

If you use this project in your research or academic work, please cite it as:

### BibTeX
```bibtex
@software{modina2025MrXchase,
  author = {Modina, Ettore},
  title = {Shadow Chase: AI Agent Training \& Evaluation Platform},
  year = {2025},
  url = {https://github.com/ettoremodina/ShadowChase},
  note = {Educational implementation inspired by Shadow Chase for reinforcement learning and game theory research}
}
```

### APA Style
Modina, E. (2025). *Shadow Chase: AI Agent Training & Evaluation Platform* [Computer software]. https://github.com/ettoremodina/ShadowChase

### IEEE Style
E. Modina, "Shadow Chase: AI Agent Training & Evaluation Platform," 2025. [Online]. Available: https://github.com/ettoremodina/ShadowChase


---

*Happy hunting, detectives! 🕵️‍♂️*