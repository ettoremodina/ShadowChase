# Video Export Feature for Scotland Yard Game

This feature allows you to export game replays as MP4 videos, showing the progression of the game with visual game state information.

## Features

- **Visual Game Board**: Shows the Scotland Yard board with player positions
- **Game State Information**: Displays current turn, player positions, and game status
- **Ticket Information**: Shows remaining tickets for all players
- **Move History**: Displays recent moves with transport types
- **Configurable Speed**: Adjust frame duration from 0.1 to 5.0 seconds per step
- **Multiple Export Options**: Supports MP4 video, GIF animation, or image sequence

## Usage

### From the GUI

1. **Open the Game Visualizer**
2. **Click "🎬 Export Video"** button in the Load Game section
3. **Select a completed game** from the list
4. **Configure video settings**:
   - Frame Duration: How long each game step is displayed (in seconds)
   - Output File: Where to save the video (optional)
5. **Click "Export Video"**

### From Command Line

```bash
# Basic usage
python export_video.py saved_games/games/game_20250719_175113.pkl

# Custom output file
python export_video.py saved_games/games/game_20250719_175113.pkl --output my_replay.mp4

# Slow motion (2 seconds per step)
python export_video.py saved_games/games/game_20250719_175113.pkl --duration 2.0

# Fast replay (0.5 seconds per step)
python export_video.py saved_games/games/game_20250719_175113.pkl --duration 0.5 --output fast_replay.mp4

# Verbose output
python export_video.py saved_games/games/game_20250719_175113.pkl --verbose
```

### Command Line Arguments

- `game_file`: Path to the saved game file (.pkl)
- `--output`, `-o`: Output video file path (default: auto-generated)
- `--duration`, `-d`: Duration of each frame in seconds (default: 1.0)
- `--fps`: Frames per second (default: calculated from duration)
- `--verbose`, `-v`: Show detailed output during export

## Video Content

Each frame shows:

### Main Graph Panel
- **Game board** with transport connections (taxi=yellow, bus=blue, underground=red, ferry=green)
- **Player positions**: 
  - Blue circles: Detectives
  - Red circle: Mr. X (when visible)
  - Yellow circle: When detective and Mr. X are at same position
- **Node labels** showing position numbers

### Info Panels (Right Side)
1. **Game State Panel**:
   - Current turn (Detectives/Mr. X)
   - Turn count
   - Player positions
   - Game over status

2. **Tickets Panel**:
   - Remaining tickets for each player
   - Table format showing taxi 🚕, bus 🚌, underground 🚇, black ⚫, double move ⚡

3. **History Panel**:
   - Recent moves (last 3-4 turns)
   - Shows player, move, and transport type used

## Requirements

### For MP4 Video Export (Recommended)
- **FFmpeg**: Install from [https://ffmpeg.org/](https://ffmpeg.org/)
  - Windows: Download and add to PATH
  - Linux: `sudo apt install ffmpeg` (Ubuntu) or `sudo yum install ffmpeg` (CentOS)
  - macOS: `brew install ffmpeg`

### Fallback Options
If FFmpeg is not available, the system will automatically try:
1. **Matplotlib Animation**: Creates GIF files
2. **Image Sequence**: Creates a ZIP file with individual frame images

### Python Dependencies
All required packages are already included in the project:
- `matplotlib` (for graphics)
- `networkx` (for graph layout)
- `tkinter` (for GUI dialogs)

## Examples

### Quick Start
```bash
# Export with default settings (1 second per step)
python export_video.py saved_games/games/game_20250719_175113.pkl
```

### Custom Settings
```bash
# Slow, detailed replay
python export_video.py saved_games/games/game_20250719_175113.pkl \
    --duration 3.0 \
    --output detailed_replay.mp4

# Fast summary
python export_video.py saved_games/games/game_20250719_175113.pkl \
    --duration 0.3 \
    --output quick_summary.mp4
```

## Output

- **Video files**: High-quality MP4 videos at 1920x1000 resolution
- **File size**: Typically 1-5 MB for a complete game
- **Duration**: Depends on game length and frame duration setting

## Troubleshooting

### "FFmpeg not found"
- Install FFmpeg and ensure it's in your system PATH
- The system will automatically fall back to creating GIF or image sequence

### "No game history available"
- Ensure the game file contains a completed game with move history
- Some games may not have been saved with full history

### Video quality issues
- Use longer frame durations (2.0+ seconds) for better readability
- Ensure the game has proper board positions for best visualization

### Performance
- Video generation may take 1-2 minutes for longer games
- Progress is shown during export

## File Locations

- **Saved games**: `saved_games/games/` directory
- **Export script**: `export_video.py` in project root
- **Video module**: `ScotlandYard/ui/video_exporter.py`

## Technical Details

### Video Format
- **Codec**: H.264 (libx264)
- **Resolution**: 1600x1000 pixels
- **Pixel Format**: yuv420p (for maximum compatibility)
- **Quality**: CRF 23 (high quality)

### Frame Layout
- **Left side**: Game board visualization (66% width)
- **Right side**: Three info panels (33% width)
- **Aspect ratio**: 16:10 for optimal viewing

This feature makes it easy to create shareable videos of your Scotland Yard games for analysis, presentation, or entertainment!
