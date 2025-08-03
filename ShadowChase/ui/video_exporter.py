"""
Video Export Module for Shadow Chase Game Replay
Creates MP4 videos from saved game states with configurable frame duration.
"""

import os
import tempfile
import subprocess
from pathlib import Path

from matplotlib.figure import Figure
import networkx as nx
from tkinter import messagebox, filedialog
from ..core.game import ShadowChaseGame
from ..services.game_loader import GameLoader
from .base_visualizer import BaseVisualizer


class GameVideoExporter(BaseVisualizer):
    """Exports game replays as MP4 videos"""
    
    def __init__(self, game: ShadowChaseGame, game_id: str, output_path: str = None, 
                 frame_duration: float = 1.0, fps: int = 1):
        """
        Initialize video exporter
        
        Args:
            game: The ShadowChaseGame instance with game history
            game_id: Game identifier for video title
            output_path: Path where video will be saved (optional)
            frame_duration: Duration of each frame in seconds (default: 1.0)
            fps: Frames per second for video (should match 1/frame_duration)
        """
        super().__init__(game)
        self.game_id = game_id
        self.output_path = output_path
        self.frame_duration = frame_duration
        self.fps = max(1, int(1 / frame_duration)) if frame_duration > 0 else 1
        
        # Video settings
        self.fig_size = (16, 10)  # Larger figure for better video quality
        self.dpi = 100
        self.video_quality = 'high'  # high, medium, low
        
        # State tracking
        self.current_frame = 0
        self.total_frames = 0
        
        # Setup figure for video frames
        self.setup_video_figure()
        
    def setup_video_figure(self):
        """Setup matplotlib figure for video rendering"""
        self.fig = Figure(figsize=self.fig_size, dpi=self.dpi, facecolor='white')
        
        # Create subplots: info panels on left, main graph on right (larger)
        gs = self.fig.add_gridspec(2, 3, width_ratios=[1, 2.5, 0.5], height_ratios=[1, 1],
                                  hspace=0.3, wspace=0.2)
        
        # Info panels on the left
        self.info_ax = self.fig.add_subplot(gs[0, 0])
        self.tickets_ax = self.fig.add_subplot(gs[1, 0])
        
        # Main graph takes up right columns (larger and stretched)
        self.ax = self.fig.add_subplot(gs[:, 1:])
        
        # Remove axes for info panels
        for ax in [self.info_ax, self.tickets_ax]:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
        
        # Setup graph positions (similar to BaseVisualizer)
        if hasattr(self.game, 'node_positions') and self.game.node_positions:
            positions = self.game.node_positions
            x_coords = [pos[0] for pos in positions.values()]
            y_coords = [pos[1] for pos in positions.values()]
            
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            x_range = x_max - x_min
            y_range = y_max - y_min
            scale = max(x_range, y_range)
            
            self.pos = {}
            for node, (x, y) in positions.items():
                # Stretch the x-axis more for better aspect ratio
                normalized_x = 2.5 * (x - x_min) / scale - 1.25
                normalized_y = 2 * (y - y_min) / scale - 1
                self.pos[node] = (normalized_x, -normalized_y)
        else:
            # Try to load board positions from CSV if available
            try:
                from ShadowChase.services.board_loader import load_board_graph_from_csv
                _, node_positions = load_board_graph_from_csv()
                
                # Check if this game uses the same nodes as the extracted board
                game_nodes = set(self.game.graph.nodes())
                extracted_nodes = set(node_positions.keys())
                
                if game_nodes == extracted_nodes:
                    # Use extracted board positions
                    x_coords = [pos[0] for pos in node_positions.values()]
                    y_coords = [pos[1] for pos in node_positions.values()]
                    
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    
                    x_range = x_max - x_min
                    y_range = y_max - y_min
                    scale = max(x_range, y_range)
                    
                    self.pos = {}
                    for node, (x, y) in node_positions.items():
                        # Stretch the x-axis more for better aspect ratio
                        normalized_x = 2.5 * (x - x_min) / scale - 1.25
                        normalized_y = 2 * (y - y_min) / scale - 1
                        self.pos[node] = (normalized_x, -normalized_y)
                else:
                    # Fall back to spring layout
                    self.pos = nx.spring_layout(self.game.graph, seed=42, k=1, iterations=50)
            except:
                # Fall back to spring layout
                self.pos = nx.spring_layout(self.game.graph, seed=42, k=1, iterations=50)
    
    def create_video_frame(self, step: int) -> None:
        """Create a single frame for the video"""
        if step >= len(self.game.game_history):
            return
            
        state = self.game.game_history[step]
        
        # Clear all axes
        self.ax.clear()
        self.info_ax.clear()
        self.tickets_ax.clear()
        
        # Draw main graph
        self._draw_game_graph(state, step)
        
        # Draw info panels
        self._draw_info_panel(state, step)
        self._draw_tickets_panel(state)
        
        # Set titles and formatting
        self._format_axes(step)
    
    def _draw_game_graph(self, state, step: int):
        """Draw the main game graph for current state"""
        # Draw edges with parallel positioning
        self.draw_edges_with_parallel_positioning(alpha=0.4)
        
        # Get node colors and sizes
        node_colors, node_sizes = self._get_video_node_colors_and_sizes(state)
        
        # Draw nodes
        nx.draw_networkx_nodes(self.game.graph, self.pos, ax=self.ax,
                              node_color=node_colors, node_size=node_sizes)
        
        # Draw labels
        nx.draw_networkx_labels(self.game.graph, self.pos, ax=self.ax, 
                               font_size=8, font_weight='bold')
        
        # Add transport legend
        self.draw_transport_legend()
        
        self.ax.set_title(f"Shadow Chase Game - Step {step + 1}/{len(self.game.game_history)}", 
                         fontsize=16, fontweight='bold', pad=20)
        self.ax.axis('off')
    
    def _get_video_node_colors_and_sizes(self, state):
        """Get node colors and sizes for video frame"""
        node_colors = []
        node_sizes = []
        node_size = 400  # Larger for video visibility
        
        for node in self.game.graph.nodes():
            if node in state.detective_positions and node == state.MrX_position:
                # Both detective and Mr. X at same position
                node_colors.append('yellow')
                node_sizes.append(node_size * 1.2)
            elif node in state.detective_positions:
                node_colors.append('blue')
                node_sizes.append(node_size)
            elif node == state.MrX_position:
                node_colors.append('red')
                node_sizes.append(node_size)
              
            else:
                node_colors.append('lightgray')
                node_sizes.append(node_size * 0.8)
        
        return node_colors, node_sizes
    
    def _draw_info_panel(self, state, step: int):
        """Draw game state information panel"""
        info_text = f"Turn: {state.turn.value.title()}\n"
        info_text += f"Turn Count: {state.turn_count}\n"
        info_text += f"Detectives: {state.detective_positions}\n"
        
        if hasattr(state, 'mr_x_visible') and not state.mr_x_visible:
            info_text += f"Mr. X: HIDDEN\n"
        else:
            info_text += f"Mr. X: {state.MrX_position}\n"
        
        if hasattr(state, 'double_move_active') and state.double_move_active:
            info_text += "Double Move: ACTIVE\n"
        
        # Check if game is over at this step
        temp_game_state = self.game.game_state
        self.game.game_state = state
        is_over = self.game.is_game_over()
        if is_over:
            winner = self.game.get_winner()
            info_text += f"\nGAME OVER!\nWinner: {winner.value.title() if winner else 'None'}"
        self.game.game_state = temp_game_state
        
        self.info_ax.text(0.05, 0.95, info_text, transform=self.info_ax.transAxes,
                         fontsize=12, verticalalignment='top', fontfamily='monospace',
                         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        self.info_ax.set_title("Game State", fontsize=14, fontweight='bold')
    
    def _draw_tickets_panel(self, state):
        """Draw tickets information panel"""
        # Use the base visualizer method to create ticket table
        tickets_text = "TICKETS:\n\n"
        tickets_text += "Player│Tax│Bus│Sub│Blk│Dbl\n"
        tickets_text += "──────┼───┼───┼───┼───┼───\n"
        
        # Detective tickets
        for i in range(self.game.num_detectives):
            if hasattr(state, 'detective_tickets') and i in state.detective_tickets:
                tickets = state.detective_tickets[i]
            else:
                tickets = {}
            
            taxi = self._get_ticket_count(tickets, 'taxi')
            bus = self._get_ticket_count(tickets, 'bus')
            underground = self._get_ticket_count(tickets, 'underground')
            
            tickets_text += f"Det {i+1:<2}│{taxi:>3}│{bus:>3}│{underground:>3}│ - │ - \n"
        
        # Mr. X tickets
        if hasattr(state, 'mr_x_tickets'):
            mr_x_tickets = state.mr_x_tickets
        else:
            mr_x_tickets = {}
        
        taxi = self._get_ticket_count(mr_x_tickets, 'taxi')
        bus = self._get_ticket_count(mr_x_tickets, 'bus')
        underground = self._get_ticket_count(mr_x_tickets, 'underground')
        black = self._get_ticket_count(mr_x_tickets, 'black')
        double = self._get_ticket_count(mr_x_tickets, 'double_move')
        
        tickets_text += f"Mr. X │{taxi:>3}│{bus:>3}│{underground:>3}│{black:>3}│{double:>3}\n"
        
        self.tickets_ax.text(0.05, 0.95, tickets_text, transform=self.tickets_ax.transAxes,
                            fontsize=10, verticalalignment='top', fontfamily='monospace',
                            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
        self.tickets_ax.set_title("Tickets", fontsize=14, fontweight='bold')
    
    def _format_axes(self, step: int):
        """Format axes and overall figure"""
        # Remove tick marks from info panels
        for ax in [self.info_ax, self.tickets_ax]:
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Add game ID and timestamp
        self.fig.suptitle(f"Shadow Chase Game Replay - {self.game_id}", 
                         fontsize=18, fontweight='bold')
    
    def export_video(self, progress_callback=None) -> str:
        """
        Export the game replay as an MP4 video
        
        Args:
            progress_callback: Optional callback function to report progress
            
        Returns:
            Path to the created video file
        """
        if not self.game.game_history:
            raise ValueError("No game history available for video export")
        
        if len(self.game.game_history) == 0:
            raise ValueError("Game history is empty")
        
        # Setup output path
        if not self.output_path:
            timestamp = self.game_id.replace('game_', '')
            self.output_path = f"shadow_chase_replay_{timestamp}.mp4"
        
        # Ensure output directory exists
        output_dir = Path(self.output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create temporary directory for frames
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Generate individual frames
            total_frames = len(self.game.game_history)
            if progress_callback:
                progress_callback(0, total_frames, f"Preparing to generate {total_frames} frames...")
            
            for i in range(total_frames):
                if progress_callback:
                    progress_callback(i, total_frames, f"Generating frame {i+1}/{total_frames}")
                
                try:
                    self.create_video_frame(i)
                    
                    # Save frame
                    frame_path = temp_path / f"frame_{i:04d}.png"
                    self.fig.savefig(frame_path, dpi=self.dpi, bbox_inches='tight', 
                                   facecolor='white', edgecolor='none')
                except Exception as e:
                    print(f"Warning: Failed to create frame {i}: {e}")
                    continue
            
            # Verify frames were created
            frame_files = list(temp_path.glob('frame_*.png'))
            if not frame_files:
                raise RuntimeError("No frames were successfully created")
            
            # Create video from frames
            if progress_callback:
                progress_callback(total_frames, total_frames, "Creating video from frames...")
            
            self._create_video_from_frames(temp_path, self.output_path)
        
        # Verify output file was created
        if not os.path.exists(self.output_path):
            raise RuntimeError(f"Video file was not created: {self.output_path}")
        
        return self.output_path
    
    def _create_video_from_frames(self, frames_dir: Path, output_path: str):
        """Create MP4 video from frame images using ffmpeg"""
        # First try ffmpeg
        if self._try_ffmpeg(frames_dir, output_path):
            return
        
        # Fallback to matplotlib if ffmpeg fails
        try:
            self._create_video_with_matplotlib_animation(frames_dir, output_path)
        except Exception as e:
            # Final fallback: create image sequence
            print(f"Warning: Video creation failed, creating image sequence instead: {e}")
            self._create_image_sequence(frames_dir, output_path)
    
    def _try_ffmpeg(self, frames_dir: Path, output_path: str) -> bool:

        """Try to create video using ffmpeg"""
        try:
            # Check if ffmpeg is available
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            
            # ffmpeg command to create video
            cmd = [
                'ffmpeg', '-y',  # -y to overwrite output file
                '-framerate', str(self.fps),  # Input framerate
                '-i', str(frames_dir / 'frame_%04d.png'),  # Input pattern
                '-c:v', 'libx264',  # Video codec
                '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
                '-crf', '23',  # Quality setting (lower = better quality)
                str(output_path)
            ]
            
            # Run ffmpeg
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                return True
            else:
                print(f"ffmpeg failed: {result.stderr}")
                return False
                
        except (FileNotFoundError, subprocess.CalledProcessError):
            print("ffmpeg not available, trying alternative methods...")
            return False
    
    def _create_video_with_matplotlib_animation(self, frames_dir: Path, output_path: str):
        """Create video using matplotlib animation"""
        try:
            import matplotlib.animation as animation
            
            # Create animation function
            def animate(frame_num):
                if frame_num < len(self.game.game_history):
                    self.create_video_frame(frame_num)
                return []
            
            # Create animation
            anim = animation.FuncAnimation(
                self.fig, animate, frames=len(self.game.game_history),
                interval=int(self.frame_duration * 1000), blit=False, repeat=False
            )
            
            # Try different writers
            if 'ffmpeg' in animation.writers.list():
                Writer = animation.writers['ffmpeg']
                writer = Writer(fps=self.fps, metadata=dict(artist='Shadow Chase Game'), bitrate=1800)
            elif 'pillow' in animation.writers.list():
                Writer = animation.writers['pillow']
                writer = Writer(fps=self.fps)
                # Change extension to gif for pillow
                if output_path.endswith('.mp4'):
                    output_path = output_path.replace('.mp4', '.gif')
            else:
                raise RuntimeError("No suitable animation writer found")
            
            anim.save(output_path, writer=writer)
            print(f"Video created using matplotlib: {output_path}")
            
        except Exception as e:
            raise RuntimeError(f"Matplotlib video creation failed: {str(e)}")
    
    def _create_image_sequence(self, frames_dir: Path, output_path: str):
        """Create a zip file with the image sequence as final fallback"""
        import zipfile
        
        # Change output to zip file
        zip_path = output_path.replace('.mp4', '_frames.zip')
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for frame_file in sorted(frames_dir.glob('frame_*.png')):
                zipf.write(frame_file, frame_file.name)
        
        print(f"Created image sequence: {zip_path}")
        return zip_path


def show_video_export_dialog(parent, loader: GameLoader):
    """Show dialog to select game and export video"""
    try:
        # Show game selection dialog
        import tkinter as tk
        from tkinter import ttk
        
        dialog = tk.Toplevel(parent)
        dialog.title("Export Game Video")
        dialog.geometry("700x500")
        dialog.transient(parent)
        dialog.grab_set()
        
        # Game selection methods
        ttk.Label(dialog, text="Select a game to export:", font=('Arial', 12, 'bold')).pack(pady=10)
        
        # Create notebook for two selection methods
        notebook = ttk.Notebook(dialog)
        notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Tab 1: Browse for file
        file_frame = ttk.Frame(notebook)
        notebook.add(file_frame, text="Browse for File")
        
        # File selection
        file_selection_frame = ttk.Frame(file_frame)
        file_selection_frame.pack(fill=tk.X, padx=20, pady=20)
        
        ttk.Label(file_selection_frame, text="Game File (.pkl):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        game_file_var = tk.StringVar()
        game_file_entry = ttk.Entry(file_selection_frame, textvariable=game_file_var, width=50)
        game_file_entry.grid(row=0, column=1, padx=5, pady=5)
        
        def browse_game_file():
            filename = filedialog.askopenfilename(
                title="Select Game File",
                filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
                initialdir="saved_games"
            )
            if filename:
                game_file_var.set(filename)
                # Auto-generate output filename
                base_name = Path(filename).stem
                output_name = f"{base_name}.mp4"
                output_var.set(output_name)
        
        ttk.Button(file_selection_frame, text="Browse...", command=browse_game_file).grid(row=0, column=2, padx=5, pady=5)
        
        # Tab 2: Saved games list
        list_frame = ttk.Frame(notebook)
        notebook.add(list_frame, text="Saved Games")
        
        # Get list of available games
        games = loader.list_games()
        
        if games:
            # Listbox with scrollbar for saved games
            games_list_frame = ttk.Frame(list_frame)
            games_list_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
            
            scrollbar = ttk.Scrollbar(games_list_frame)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            game_listbox = tk.Listbox(games_list_frame, yscrollcommand=scrollbar.set)
            game_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.config(command=game_listbox.yview)
            
            # Populate games list
            for game_info in games:
                game_id = game_info.get('game_id', 'Unknown')
                created_at = game_info.get('created_at', 'Unknown')
                status = "Complete" if game_info.get('game_completed', False) else "Incomplete"
                game_listbox.insert(tk.END, f"{game_id} - {created_at} ({status})")
        else:
            ttk.Label(list_frame, text="No saved games found.", font=('Arial', 10)).pack(pady=50)
        
        # Video settings
        settings_frame = ttk.LabelFrame(dialog, text="Video Settings")
        settings_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Frame duration
        ttk.Label(settings_frame, text="Frame Duration (seconds):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        duration_var = tk.DoubleVar(value=1.0)
        duration_spinbox = ttk.Spinbox(settings_frame, from_=0.1, to=5.0, increment=0.1, 
                                      textvariable=duration_var, width=10)
        duration_spinbox.grid(row=0, column=1, padx=5, pady=5)
        
        # Output location
        ttk.Label(settings_frame, text="Output File:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        output_var = tk.StringVar()
        output_entry = ttk.Entry(settings_frame, textvariable=output_var, width=40)
        output_entry.grid(row=1, column=1, padx=5, pady=5)
        
        def browse_output():
            filename = filedialog.asksaveasfilename(
                defaultextension=".mp4",
                filetypes=[("MP4 videos", "*.mp4"), ("All files", "*.*")]
            )
            if filename:
                # Ensure .mp4 extension
                if not filename.lower().endswith('.mp4'):
                    filename += '.mp4'
                output_var.set(filename)
        
        ttk.Button(settings_frame, text="Browse...", command=browse_output).grid(row=1, column=2, padx=5, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, padx=20, pady=10)
        
        def export_video():
            # Check which tab is selected
            current_tab = notebook.index(notebook.select())
            
            if current_tab == 0:  # Browse for file tab
                game_file_path = game_file_var.get()
                if not game_file_path:
                    messagebox.showwarning("No File Selected", "Please select a game file to export.")
                    return
                
                try:
                    # Load game directly from file
                    import pickle
                    with open(game_file_path, 'rb') as f:
                        game_data = pickle.load(f)
                    
                    if hasattr(game_data, 'game_config'):
                        # GameRecord object
                        game = loader._reconstruct_game_from_record(game_data)
                        game_id = game_data.game_id
                    else:
                        messagebox.showerror("Error", "Invalid game file format")
                        return
                        
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to load game file:\n{str(e)}")
                    return
                    
            else:  # Saved games list tab
                if not games:
                    messagebox.showwarning("No Games", "No saved games available.")
                    return
                    
                selection = game_listbox.curselection()
                if not selection:
                    messagebox.showwarning("No Selection", "Please select a game to export.")
                    return
                
                # Get selected game
                game_info = games[selection[0]]
                game_id = game_info['game_id']
                
                try:
                    # Load the game
                    game = loader.load_game(game_id)
                    if not game:
                        messagebox.showerror("Error", f"Failed to load game {game_id}")
                        return
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to load game:\n{str(e)}")
                    return
            
            # Get settings
            frame_duration = duration_var.get()
            output_path = output_var.get()
            
            if not output_path:
                # Auto-generate output filename with .mp4 extension
                output_path = f"shadow_chase_replay_{game_id}.mp4"
            else:
                # Ensure .mp4 extension
                if not output_path.lower().endswith('.mp4'):
                    output_path += '.mp4'
            
            try:
                # Create progress dialog
                progress_dialog = tk.Toplevel(dialog)
                progress_dialog.title("Exporting Video...")
                progress_dialog.geometry("400x100")
                progress_dialog.transient(dialog)
                progress_dialog.grab_set()
                
                progress_label = ttk.Label(progress_dialog, text="Initializing...")
                progress_label.pack(pady=10)
                
                progress_bar = ttk.Progressbar(progress_dialog, mode='determinate')
                progress_bar.pack(fill=tk.X, padx=20, pady=10)
                
                def update_progress(current, total, message):
                    progress_label.config(text=message)
                    progress_bar['maximum'] = total
                    progress_bar['value'] = current
                    progress_dialog.update()
                
                # Export video
                exporter = GameVideoExporter(game, game_id, output_path, frame_duration)
                result_path = exporter.export_video(update_progress)
                
                # Close progress dialog
                progress_dialog.destroy()
                
                # Show success message
                messagebox.showinfo("Success", f"Video exported successfully to:\n{result_path}")
                dialog.destroy()
                
            except Exception as e:
                if 'progress_dialog' in locals():
                    progress_dialog.destroy()
                messagebox.showerror("Error", f"Failed to export video:\n{str(e)}")
        
        ttk.Button(button_frame, text="Export Video", command=export_video).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT, padx=5)
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to show video export dialog:\n{str(e)}")


def export_video_from_command_line(game_file: str, output_file: str = None, 
                                  frame_duration: float = 1.0) -> str:
    """
    Export video from command line
    
    Args:
        game_file: Path to the saved game file
        output_file: Output video file path (optional)
        frame_duration: Duration of each frame in seconds
        
    Returns:
        Path to the created video file
    """
    import pickle
    
    # Load game from file
    try:
        with open(game_file, 'rb') as f:
            game_data = pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load game from {game_file}: {str(e)}")
    
    # Handle different game storage formats
    if hasattr(game_data, 'game_history'):
        # Direct game object
        game = game_data
        game_id = Path(game_file).stem
    elif hasattr(game_data, 'game_config'):
        # GameRecord object - need to reconstruct
        from ..services.game_loader import GameLoader
        loader = GameLoader()
        game = loader._reconstruct_game_from_record(game_data)
        game_id = game_data.game_id
    else:
        raise ValueError(f"File does not contain a valid Shadow Chase game: {type(game_data)}")
    
    # Setup output path with automatic .mp4 extension
    if not output_file:
        output_file = f"shadow_chase_replay_{game_id}.mp4"
    else:
        # Ensure .mp4 extension
        if not output_file.lower().endswith('.mp4'):
            output_file += '.mp4'
    
    # Export video
    exporter = GameVideoExporter(game, game_id, output_file, frame_duration)
    return exporter.export_video()
