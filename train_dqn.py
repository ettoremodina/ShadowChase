"""
DQN training entry point with ml_logger-backed monitoring and diagnostics.

One command owns exactly one run. Training opens a ``training`` run and records
post-training evaluation inside it as ``validation`` metrics; ``--evaluate``
opens a standalone ``evaluation`` run instead.
"""

import argparse
from typing import Optional

import torch

from ml_logger import get_logger, run
from ShadowChase.integrations import TrainingRunRecorder
from training.training_environment import TrainingEnvironment
from agents.dqn_agent import DQNMrXAgent, DQNMultiDetectiveAgent
from training.deep_q.dqn_trainer import DQNTrainer
from agents import AgentType, get_agent_registry
from training.plot_utils import plot_training_metrics


logger = get_logger(__name__)

DEFAULT_MAP_SIZE = "extended"
DEFAULT_NUM_DETECTIVES = 5
DEFAULT_MAX_TURNS = 24
DEFAULT_SAVE_DIR = "training_results"
EVALUATION_MAP_SIZE = "test"
EVALUATION_NUM_DETECTIVES = 2
EVALUATION_MAX_TURNS = 24


def resolve_device(preference: Optional[str] = None) -> torch.device:
    """Select the training device, preferring CUDA exactly as before."""
    if preference:
        return torch.device(preference)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_with_monitoring(player_role="MrX", num_episodes=None,
                          plotting_every=None, device=None,
                          recorder: Optional[TrainingRunRecorder] = None,
                          map_size=DEFAULT_MAP_SIZE,
                          num_detectives=DEFAULT_NUM_DETECTIVES,
                          max_turns=DEFAULT_MAX_TURNS,
                          show_plots=True,
                          save_dir=DEFAULT_SAVE_DIR):
    """Train DQN with enhanced monitoring and diagnostics.

    ``num_episodes`` of ``None`` keeps the historical behavior of taking the
    episode count from the DQN configuration file rather than the caller.
    """
    logger.info("Starting enhanced DQN training for %s", player_role)

    # Set device if not provided
    if device is None:
        device = resolve_device()

    logger.info("Using device: %s", device)

    # Create trainer
    trainer = DQNTrainer(
        player_role=player_role,
        config_path="training/configs/dqn_config.json",
        save_dir=save_dir,
        device=device,
        run_recorder=recorder
    )

    resolved_episodes = trainer.num_episodes if num_episodes is None else num_episodes
    if plotting_every is None:
        plotting_every = max(1, resolved_episodes // 3)
    if recorder is not None:
        recorder.record_run_parameters({'training/plotting_every': plotting_every})

    # Override the training loop to add monitoring
    original_train_episode = trainer._train_episode

    def monitored_train_episode(env, opponent_agent):
        episode_reward = original_train_episode(env, opponent_agent)

        # Monitor Q-values periodically using trainer's method
        current_episode = len(trainer.episode_rewards)

        # Plot progress periodically
        if current_episode % plotting_every == 0 and current_episode > 0:
            logger.info("Generating training progress plot")
            _emit_training_plot(
                trainer,
                f"{save_dir}/training_progress_{player_role}_{current_episode}.png",
                plotting_every,
                recorder,
                show_plots,
            )

        return episode_reward

    trainer._train_episode = monitored_train_episode

    # Start training
    result = trainer.train(
        num_episodes=num_episodes,
        map_size=map_size,
        num_detectives=num_detectives,
        max_turns_per_game=max_turns,
        plotting_every=plotting_every
    )

    # Final plots
    logger.info("Generating final training plots")
    _emit_training_plot(
        trainer,
        f"{save_dir}/final_training_metrics_{player_role}.png",
        plotting_every,
        recorder,
        show_plots,
    )

    logger.info("Training complete")
    logger.info("Model: %s", result.model_path)
    logger.info("Performance: %s", result.final_performance)

    if recorder is not None:
        recorder.finalize({
            'algorithm': result.algorithm,
            'player_role': player_role,
            'total_episodes': result.total_episodes,
            'duration_seconds': result.training_duration,
            'model_path': result.model_path,
            **result.final_performance,
        })

    return result, trainer


def evaluate_trained_agent(model_path, player_role, num_games=100, device=None,
                           recorder: Optional[TrainingRunRecorder] = None,
                           phase: str = "validation"):
    """Evaluate a trained agent against random opponents."""
    logger.info("Evaluating trained %s agent", player_role)

    # Set device if not provided
    if device is None:
        device = resolve_device()

    env = TrainingEnvironment(
        EVALUATION_MAP_SIZE,
        EVALUATION_NUM_DETECTIVES,
        EVALUATION_MAX_TURNS,
    )
    registry = get_agent_registry()

    # Create trained agent
    if player_role == "MrX":
        trained_agent = DQNMrXAgent(model_path=model_path, epsilon=0.0, device=device)  # No exploration
        opponent = registry.create_multi_detective_agent(AgentType.RANDOM, EVALUATION_NUM_DETECTIVES)
    else:
        trained_agent = DQNMultiDetectiveAgent(EVALUATION_NUM_DETECTIVES, model_path=model_path, epsilon=0.0, device=device)
        opponent = registry.create_MrX_agent(AgentType.RANDOM)

    if recorder is not None:
        recorder.record_run_parameters({
            f'{phase}/model_path': str(model_path),
            f'{phase}/player_role': player_role,
            f'{phase}/games': num_games,
            f'{phase}/opponent': AgentType.RANDOM.value,
            f'{phase}/map_size': EVALUATION_MAP_SIZE,
            f'{phase}/num_detectives': EVALUATION_NUM_DETECTIVES,
        })

    # Run evaluation games
    wins = 0
    total_turns = 0

    for i in range(num_games):
        if player_role == "MrX":
            result, _ = env.run_episode(trained_agent, opponent)
            won = result.winner == "MrX"
        else:
            result, _ = env.run_episode(opponent, trained_agent)
            won = result.winner == "detectives"

        wins += won
        total_turns += result.total_turns

        if recorder is not None:
            recorder.record_metrics(
                i,
                {'win': int(won), 'turns': result.total_turns},
                phase=phase,
            )

        if i % 20 == 0 and i > 0:
            current_win_rate = wins / (i + 1)
            logger.info(
                "Game %3d/%d | current win rate: %5.1f%%",
                i + 1,
                num_games,
                100 * current_win_rate,
            )

    win_rate = wins / num_games
    logger.info("Final win rate: %5.1f%% (%d/%d)", 100 * win_rate, wins, num_games)
    logger.info(
        "%s",
        "Better than random" if win_rate > 0.5 else "Needs improvement",
    )

    if recorder is not None:
        recorder.finalize(
            {
                'games': num_games,
                'wins': wins,
                'win_rate': win_rate,
                'average_turns': total_turns / num_games if num_games else 0.0,
            },
            namespace=phase,
        )

    return win_rate


def _emit_training_plot(trainer, save_path, plotting_every, recorder, show_plots):
    """Write a metrics plot and register it as a run artifact when recording.

    Short runs produce no plot at all, so the artifact is only registered when
    a file was actually written.
    """
    written_path = plot_training_metrics(
        trainer, save_path, plotting_every, show_plot=show_plots
    )
    if written_path is not None and recorder is not None:
        recorder.record_artifact(written_path, kind="plot")


def parse_arguments() -> argparse.Namespace:
    """Parse training, evaluation, and ml_logger options."""
    parser = argparse.ArgumentParser(description='Enhanced DQN Training')
    parser.add_argument('--role', choices=['MrX', 'detectives'], default='MrX',
                        help='Player role to train')
    parser.add_argument('--episodes', type=int,
                        help='Number of training episodes (default: the DQN config value)')
    parser.add_argument('--plotting-every', type=int,
                        help='Plot and Q-value monitoring interval (default: episodes // 3)')
    parser.add_argument('--map-size', type=str, default=DEFAULT_MAP_SIZE,
                        help='Training map size')
    parser.add_argument('--detectives', type=int, default=DEFAULT_NUM_DETECTIVES,
                        help='Number of detectives during training')
    parser.add_argument('--max-turns', type=int, default=DEFAULT_MAX_TURNS,
                        help='Maximum turns per training game')
    parser.add_argument('--save-dir', type=str, default=DEFAULT_SAVE_DIR,
                        help='Directory for model checkpoints and plots')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate existing model instead of training')
    parser.add_argument('--model_path', '--model-path', dest='model_path', type=str,
                        help='Path to model for evaluation')
    parser.add_argument('--eval-games', type=int, default=50,
                        help='Games played when evaluating a model')
    parser.add_argument('--device', type=str,
                        help='Torch device override, for example cpu or cuda:0')
    parser.add_argument('--no-show-plots', action='store_true',
                        help='Write plots without opening interactive windows')
    parser.add_argument('--run-name', type=str,
                        help='Optional ml_logger run name')
    parser.add_argument('--logger-config', type=str,
                        help='Path to an ml_logger YAML configuration')
    parser.add_argument('--artifact-root', type=str,
                        help='Override the ml_logger artifact root directory')
    return parser.parse_args()


def main():
    """Main training and evaluation pipeline."""
    args = parse_arguments()
    device = resolve_device(args.device)

    if args.evaluate:
        if not args.model_path:
            logger.error("--model_path is required for evaluation")
            return None
        return _run_evaluation(args, device)

    return _run_training(args, device)


def _run_evaluation(args: argparse.Namespace, device: torch.device):
    """Own one evaluation run around a standalone model evaluation."""
    effective_config = {
        'mode': 'evaluate',
        'player_role': args.role,
        'model_path': args.model_path,
        'games': args.eval_games,
        'device': str(device),
    }
    run_name = args.run_name or f"dqn-{args.role}-evaluation"
    with run(
        "evaluation",
        name=run_name,
        config=effective_config,
        root_dir=args.artifact_root,
        metadata={"entry_point": "train_dqn"},
        logger_config_path=args.logger_config,
    ) as context:
        recorder = TrainingRunRecorder(context)
        logger.info("Using device: %s", device)
        win_rate = evaluate_trained_agent(
            args.model_path,
            args.role,
            num_games=args.eval_games,
            device=device,
            recorder=recorder,
            phase="evaluation",
        )
        logger.info("Run artifacts: %s", context.run_dir)
        return win_rate


def _run_training(args: argparse.Namespace, device: torch.device):
    """Own one training run covering training and its follow-up evaluation."""
    effective_config = {
        'mode': 'train',
        'player_role': args.role,
        'episodes': args.episodes,
        'plotting_every': args.plotting_every,
        'map_size': args.map_size,
        'num_detectives': args.detectives,
        'max_turns': args.max_turns,
        'evaluation_games': args.eval_games,
        'save_dir': args.save_dir,
        'device': str(device),
    }
    run_name = args.run_name or f"dqn-{args.role}"
    with run(
        "training",
        name=run_name,
        config=effective_config,
        root_dir=args.artifact_root,
        metadata={"entry_point": "train_dqn"},
        logger_config_path=args.logger_config,
    ) as context:
        recorder = TrainingRunRecorder(context)
        logger.info("Using device: %s", device)

        result, trainer = train_with_monitoring(
            args.role,
            args.episodes,
            args.plotting_every,
            device,
            recorder=recorder,
            map_size=args.map_size,
            num_detectives=args.detectives,
            max_turns=args.max_turns,
            show_plots=not args.no_show_plots,
            save_dir=args.save_dir,
        )

        # Quick evaluation inside the same run
        evaluate_trained_agent(
            result.model_path,
            args.role,
            num_games=args.eval_games,
            device=device,
            recorder=recorder,
            phase="validation",
        )
        logger.info("Run artifacts: %s", context.run_dir)
        return result, trainer


if __name__ == "__main__":
    main()
