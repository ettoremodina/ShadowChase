#!/usr/bin/env python3
"""
Command line tool to export Scotland Yard game replays as videos.

Usage:
    python export_video.py <game_file> [options]

Examples:
    python export_video.py saved_games/games/game_20250719_175113.pkl
    python export_video.py saved_games/games/game_20250719_175113.pkl --output my_video.mp4 --duration 2.0
    python export_video.py saved_games/games/game_20250719_175113.pkl --duration 0.5 --output fast_replay.mp4
"""

import argparse
import sys
import os
from pathlib import Path

# Add the parent directory to Python path to import modules
current_dir = Path(__file__).parent
project_root = current_dir.parent if current_dir.name == "ScotlandYardRL" else current_dir
sys.path.insert(0, str(project_root))

from ScotlandYard.ui.video_exporter import export_video_from_command_line


def main():
    parser = argparse.ArgumentParser(
        description="Export Scotland Yard game replays as MP4 videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s game.pkl
  %(prog)s game.pkl --output my_replay.mp4
  %(prog)s game.pkl --duration 2.0 --output slow_replay.mp4
  %(prog)s game.pkl --duration 0.5 --output fast_replay.mp4
        """
    )
    
    parser.add_argument('game_file', 
                       help='Path to the saved game file (.pkl)')
    
    parser.add_argument('-o', '--output',
                       help='Output video file path (default: auto-generated)')
    
    parser.add_argument('-d', '--duration', type=float, default=1.0,
                       help='Duration of each frame in seconds (default: 1.0)')
    
    parser.add_argument('--fps', type=int,
                       help='Frames per second (default: 1/duration)')
    
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.game_file):
        print(f"Error: Game file '{args.game_file}' not found.")
        sys.exit(1)
    
    if not args.game_file.endswith('.pkl'):
        print(f"Warning: Game file should have .pkl extension.")
    
    # Validate duration
    if args.duration <= 0:
        print(f"Error: Duration must be positive, got {args.duration}")
        sys.exit(1)
    
    try:
        if args.verbose:
            print(f"Loading game from: {args.game_file}")
            print(f"Frame duration: {args.duration} seconds")
            if args.output:
                print(f"Output file: {args.output}")
        
        # Export video
        output_path = export_video_from_command_line(
            args.game_file, 
            args.output, 
            args.duration
        )
        
        print(f"✅ Video exported successfully: {output_path}")
        
        # Show file size
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            print(f"📁 File size: {file_size:.1f} MB")
        
    except KeyboardInterrupt:
        print("\n❌ Export cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Export failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
