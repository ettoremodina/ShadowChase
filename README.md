# 🕵️ Shadow Chase - AI Agent Training & Evaluation Platform

> **⚖️ Legal Notice**: This is an independent educational/research implementation inspired by the Shadow Chase board game. This project is not affiliated with or endorsed by Ravensburger AG (owners of Shadow Chase trademark). For educational and research use only.

A comprehensive implementation of the MrX Chase pursuit-evasion game with multiple AI agents for research, learning, and experimentation with reinforcement learning and game theory techniques.

## 🎯 Project Overview

This project aims to provide a platform for:
- **Learning Deep Q-Learning (DQN)** and comparing it to other AI techniques
- **Creating a testing environment** for various AI strategies and algorithms
- **Exploring game theory** in pursuit-evasion scenarios
- **Benchmarking different agent implementations** against each other

## 🎲 How MrX Chase Works

MrX Chase is an asymmetric pursuit-evasion game with inperfect information, inspired by Shadow Chase:

- **The MrX (1 player)**: The fugitive who moves secretly around the city using taxi, bus, underground, and black tickets. The MrX becomes visible only on certain turns (3, 8, 13, 18, 24) and has special abilities like double moves.

- **Detectives (2-5 players)**: Work together to capture the MrX by landing on their position. They have limited transport tickets and their movements are always visible.

- **Victory Conditions**:
  - **Detectives win**: If any agent lands on the MrX's position
  - **MrX wins**: If they evade capture for 24 turns or if Detectives run out of moves

- **Transport Network**: Players move through the city using three transport types:
  - 🚕 **Taxi**: Available everywhere, limited tickets
  - 🚌 **Bus**: Faster routes, fewer connections
  - 🚇 **Underground**: Fastest but most limited network


## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (optional, for faster DQN training)

### Installation
```bash
git clone https://github.com/ettoremodina/ShadowChase.git
cd ShadowChase
pip install -r requirements.txt
```

### Three Main Ways to Use This Project

#### 1. 🎮 Interactive Play (Graphical Interface)
Play MrX Chase with a visual interface - control Detectives manually or watch AI agents play:
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
python train_dqn.py --role agents --episodes 10000
```
- Train either MrX or agent teams
- Supports CPU and GPU training
- Saves models for later use

## 🏗️ Project Structure

```
MrXChase/
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
├── 📁 MrXChase/              # Core game engine
│   ├── core/                    # Game logic and rules
│   ├── ui/                      # Graphical interface
│   └── services/                # Utilities and caching
├── 📁 data/                     # Game maps and configurations
└── 📁 papers/                   # Research literature
```

## 🤖 Available Agent Types

### Basic Agents
- **Random Agent**: Makes random valid moves (baseline)
- **Heuristic Agent**: Uses distance-based strategies
  - MrX: Maximizes distance from closest agent
  - Agents: Minimize distance to MrX's last known position

### Advanced AI Agents
- **MCTS Agent**: Monte Carlo Tree Search with random simulations
- **Optimized MCTS**: Enhanced with caching for better performance  
- **Epsilon-Greedy MCTS**: MCTS with heuristic-guided simulations
- **DQN Agent**: Deep Q-Learning neural networks

### Performance Notes
- **Cache system works well for MCTS** agents, providing significant speedup
- **Game method caching** is less effective (retrieval cost > recalculation cost)
- **DQN training** benefits from GPU acceleration but runs fine on CPU

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

# Train agent team
python train_dqn.py --role detectives --episodes 5000 --save-dir my_training
```

### Batch Analysis
```bash
# Terminal gameplay - 50 automated games
python game_controls/simple_game.py --batch 50 --map-size extracted --detectives 5
```

## 🔧 Configuration

### Training Configuration
Edit `training/configs/dqn_config.json` to adjust:
- Network architecture (hidden layers, dropout)
- Training parameters (learning rate, epsilon decay)
- Experience replay settings

### Cache Settings
The caching system can be configured in your scripts:
```python
from MrXChase.services.cache_system import *

enable_cache()
enable_namespace_cache(CacheNamespace.MCTS_NODES)      # Good for MCTS
disable_namespace_cache(CacheNamespace.GAME_METHODS)   # Not helpful
```

## 🚧 Future Development & Improvements

### High Priority Improvements Needed
- **🎨 User Interface Overhaul**: The current GUI needs significant improvement for better usability
- **🤝 Adversarial Training**: Implement self-play and adversarial learning for DQN detectives
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
- **Explainable AI**: Understanding agent decision-making processes

## 🤝 Contributing

We welcome contributions! Areas where help is especially needed:

1. **UI/UX Improvements**: The current interface needs major enhancements
2. **Advanced RL Algorithms**: Implementing state-of-the-art techniques
3. **Performance Optimization**: Speeding up training and evaluation
4. **Documentation**: Tutorials, examples, and API documentation
5. **Testing**: Unit tests and integration tests

## 📚 References

The `papers/` directory contains relevant research literature on:
- Pursuit-evasion games, Shadow Chase, and strategy analysis
- Monte Carlo Tree Search applications
- Deep reinforcement learning for multi-agent systems
- Game theory and optimal strategies

## 📄 License & Legal Notice

### Project License
This project's **source code** is open source under the MIT License. See LICENSE file for details.

### Important Legal Disclaimer
This is an **independent implementation** inspired by the Shadow Chase board game, created for educational and research purposes. This project:
- ✅ **Educational/Research purposes only** - Not for commercial use
- ✅ **Independent implementation** - Original code and AI research
- ✅ **Open source** - Freely available for academic use
- ✅ **Inspired by Shadow Chase** - Acknowledges the original game concept
- ❌ **NOT affiliated** with Ravensburger AG or official Shadow Chase
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
  title = {MrX Chase: AI Agent Training \& Evaluation Platform},
  year = {2025},
  url = {https://github.com/ettoremodina/ShadowChase},
  note = {Educational implementation inspired by Shadow Chase for reinforcement learning and game theory research}
}
```

### APA Style
Modina, E. (2025). *MrX Chase: AI Agent Training & Evaluation Platform* [Computer software]. https://github.com/ettoremodina/ShadowChase

### IEEE Style
E. Modina, "MrX Chase: AI Agent Training & Evaluation Platform," 2025. [Online]. Available: https://github.com/ettoremodina/ShadowChase


---

*Happy hunting, detectives! 🕵️‍♂️*
