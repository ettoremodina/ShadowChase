import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import networkx as nx
import os
import pickle
import matplotlib.pyplot as plt
from ..core.game import Game, Player, ShadowChaseGame, TicketType, TransportType
from ..services.game_loader import GameLoader
from ..services.game_service import GameService
from .ui_components import ScrollableFrame, StyledButton, InfoDisplay, VisualTicketDisplay
from .setup_controls import SetupControls
from .game_controls import GameControls
from .game_replay import GameReplayWindow
from .video_exporter import show_video_export_dialog
from .base_visualizer import BaseVisualizer

# Import heuristics for position analysis
try:
    from agents.heuristics import GameHeuristics
    HEURISTICS_AVAILABLE = True
except ImportError:
    HEURISTICS_AVAILABLE = False


NODE_SIZE = 300  # Default node size for visualization

class GameVisualizer(BaseVisualizer):
    """Interactive GUI for detectives and MrXs game"""
    
    def __init__(self, game: Game, loader: 'GameLoader' = None, auto_positions: list = None):
        super().__init__(game)
        self.loader = loader or GameLoader()
        self.game_service = GameService(self.loader)
        self.solver = None
        self.root = tk.Tk()
        self.root.title("Shadow Chase Game")
        # Set the window to fullscreen
        self.root.state('zoomed')  # Windows fullscreen
        self.root.configure(bg="#f8f9fa")
        
        # Game mode and AI agents
        self.game_mode = "human_vs_human"
        self.detective_agent = None
        self.MrX_agent = None
        
        # Agent types for UI selection - import locally to avoid circular import
        from agents import AgentType
        self.MrX_agent_type = AgentType.RANDOM
        self.detective_agent_type = AgentType.RANDOM
        
        # Game state
        self.selected_positions = auto_positions or []
        self.setup_mode = True
        self.current_player_moves = {}
        self.highlighted_edges = []
        self.active_player_positions = []
        self.current_detective_index = 0
        self.detective_selections = []
        self.selected_nodes = []
        
        # Board image overlay settings
        self.show_edges_with_image = False  # Hide edges when board image is shown
        
        # Mr. X special moves state
        self.use_black_ticket = tk.BooleanVar()
        self.MrX_selections = []

        # Heuristics for position analysis
        self.heuristics = None
        
        # UI Controllers
        self.setup_controls = SetupControls(self)
        self.game_controls = GameControls(self)
        
        # UI components
        self.setup_ui()
        self.setup_graph_display()
        
        # Key bindings
        self.root.bind('<Return>', self._on_enter_key)
        
        # If auto_positions were provided, enable the start button but don't auto-start
        if auto_positions and len(auto_positions) == self.game.num_detectives + 1:
            self.setup_controls.start_button.config(state=tk.NORMAL)

    def _on_enter_key(self, event):
        """Handle Enter key press - same as clicking Make Move button"""
        # Only trigger if we're not in setup mode and the move button is enabled
        if (not self.setup_mode and 
            hasattr(self.game_controls, 'move_button') and 
            self.game_controls.move_button['state'] == 'normal'):
            self.game_controls.send_move()

    def _initialize_heuristics(self):
        """Initialize heuristics calculator for position analysis"""
        if not HEURISTICS_AVAILABLE:
            return
            
        if isinstance(self.game, ShadowChaseGame) and self.game.game_state:
            try:
                self.heuristics = GameHeuristics(self.game)
            except Exception as e:
                print(f"Failed to initialize heuristics: {e}")
                import traceback
                traceback.print_exc()
                self.heuristics = None

    def setup_ui(self):
        """Setup the improved user interface"""
        # Configure ttk styles
        style = ttk.Style()
        style.theme_use('clam')
        
        # Main frame with padding
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Left panel with scrollable frame
        left_panel = ttk.Frame(main_frame, width=380)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))
        left_panel.pack_propagate(False)
        
        # Create scrollable container for controls
        self.scrollable_controls = ScrollableFrame(left_panel)
        self.scrollable_controls.pack(fill=tk.BOTH, expand=True)
        
        # Title section
        title_frame = ttk.Frame(self.scrollable_controls.scrollable_frame)
        title_frame.pack(fill=tk.X, pady=(0, 15))
        
        title_label = ttk.Label(title_frame, text="🎯 Game Control Panel", 
                               font=('Arial', 14, 'bold'))
        title_label.pack(anchor="w")
        
        # Setup controls section
        self.setup_section = self.setup_controls.create_setup_section(
            self.scrollable_controls.scrollable_frame)
        self.setup_section.pack(fill=tk.X, pady=(0, 10))
        
        # Game controls section
        self.controls_section = self.game_controls.create_controls_section(
            self.scrollable_controls.scrollable_frame)
        self.controls_section.pack(fill=tk.X, pady=(0, 10))
        
        # Mr. X controls section
        self.mrx_section = self.game_controls.create_mrx_controls_section(
            self.scrollable_controls.scrollable_frame)
        self.mrx_section.pack(fill=tk.X, pady=(0, 10))
        
        # Information displays
        self.turn_display = self.game_controls.create_turn_display(
            self.scrollable_controls.scrollable_frame)
        self.turn_display.pack(fill=tk.X, pady=(0, 10))
        
        self.moves_display = self.game_controls.create_moves_display(
            self.scrollable_controls.scrollable_frame)
        self.moves_display.pack(fill=tk.X, pady=(0, 10))
        
        # Create enhanced ticket table display for Shadow Chase games
        self.tickets_table_display = VisualTicketDisplay(self.scrollable_controls.scrollable_frame,
                                                         "🎫 Ticket Information")
        # Don't pack it here - let update_ui_visibility() control when it appears
        
        # Save/Load section
        self.create_save_load_section()
        
        # Game info section
        self.create_info_section()
        
        # Right panel for graph
        self.graph_frame = ttk.Frame(main_frame)
        self.graph_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Initial UI state
        self.update_ui_visibility()
    
    def create_save_load_section(self):
        """Create the save/load controls section"""
        self.save_load_section = ttk.LabelFrame(self.scrollable_controls.scrollable_frame, 
                                              text="💾 Load Game")
        
        style = ttk.Style()
        style.configure("SaveLoad.TLabelframe.Label", anchor="w", font=('Arial', 11, 'bold'))
        self.save_load_section.configure(style="SaveLoad.TLabelframe")
        
        button_frame = ttk.Frame(self.save_load_section)
        button_frame.pack(fill=tk.X, padx=10, pady=8)
        
        self.load_button = StyledButton(button_frame, "📂 Load Game", 
                                      command=self.show_load_dialog)
        self.load_button.pack(fill=tk.X, pady=3)
        
        self.video_export_button = StyledButton(button_frame, "🎬 Export Video", 
                                              command=self.show_video_export_dialog)
        self.video_export_button.pack(fill=tk.X, pady=3)
        
        self.toggle_board_button = StyledButton(button_frame, "🖼️ Toggle Board Image", 
                                              command=self.toggle_board_image_overlay)
        self.toggle_board_button.pack(fill=tk.X, pady=3)
        
        # self.calibrate_button = StyledButton(button_frame, "🎯 Calibrate Board Overlay", 
        #                                    command=self.open_calibrator)
        # self.calibrate_button.pack(fill=tk.X, pady=3)
        
        # self.refresh_calibration_button = StyledButton(button_frame, "🔄 Refresh Calibration", 
        #                                              command=self.refresh_calibration)
        # self.refresh_calibration_button.pack(fill=tk.X, pady=3)
        
        # self.history_button = StyledButton(button_frame, "📈 Game History", 
        #                                  command=self.show_game_history)
        # self.history_button.pack(fill=tk.X, pady=3)
        
        self.save_load_section.pack(fill=tk.X, pady=(0, 10))
    
    def create_info_section(self):
        """Create the game information section"""
        self.info_display = InfoDisplay(self.scrollable_controls.scrollable_frame, 
                                      "ℹ️ Game Information", height=8)
        self.info_display.pack(fill=tk.X, pady=(0, 10))
        

    def update_info(self):
        """Update game information display"""
        if not self.info_display:
            return
            
        if self.setup_mode:
            info_text = "🎯 Setup Mode\n"
            info_text += f"📊 Need: {self.game.num_detectives} detectives + 1 MrX\n"
            info_text += f"✅ Selected: {len(self.selected_positions)}\n"
        elif self.game.game_state:
            state_info = self.game.get_state_representation()
            info_text = f"🎮 Turn: {state_info['turn'].title()}\n"
            info_text += f"📊 Turn count: {state_info['turn_count']}\n"
            info_text += f"👮 detective positions: {state_info['detective_positions']}\n"
            
            is_shadow_chase = isinstance(self.game, ShadowChaseGame)
            if is_shadow_chase:
                if self.game.game_state.MrX_visible:
                    info_text += f"🕵️‍♂️ Mr. X position: {state_info['MrX_position']} (VISIBLE)\n"
                else:
                    info_text += f"🕵️‍♂️ Mr. X position: HIDDEN\n"
            else:
                info_text += f"🏃 MrX position: {state_info['MrX_position']}\n"
            
            if self.game.is_game_over():
                winner = self.game.get_winner()
                if winner:
                    info_text += f"🏆 Game Over - Winner: {winner.value.title()}\n"
            else:
                info_text += "⏳ Game in progress\n"
        else:
            info_text = "ℹ️ No game initialized"
        
        self.info_display.set_text(info_text)
        
        # Update enhanced ticket table for Shadow Chase games
        if isinstance(self.game, ShadowChaseGame) and self.game.game_state:
            self.update_enhanced_tickets_display(self.tickets_table_display)
        
        # Update other displays
        self.game_controls.update_turn_display()
        self.game_controls.update_moves_display()
        self.setup_controls.update_status()

    def on_graph_click(self, event):
        """Handle mouse clicks on graph"""
        if event.inaxes != self.ax:
            return
        
        click_pos = (event.xdata, event.ydata)
        if click_pos[0] is None or click_pos[1] is None:
            return
        
        closest_node = None
        min_dist = float('inf')
        threshold = 0.15
        
        for node in self.game.graph.nodes():
            if node in self.pos:
                node_pos = self.pos[node]
                dist = ((node_pos[0] - click_pos[0])**2 + (node_pos[1] - click_pos[1])**2)**0.5
                if dist < min_dist and dist < threshold:
                    min_dist = dist
                    closest_node = node
        
        if closest_node is not None:
            if self.setup_mode:
                self.handle_setup_click(closest_node)
            else:
                self.handle_game_click(closest_node)

    def handle_setup_click(self, node):
        """Handle node clicks during setup"""
        if node in self.selected_positions:
            self.selected_positions.remove(node)
        else:
            if len(self.selected_positions) < self.game.num_detectives + 1:
                self.selected_positions.append(node)
        
        # Enable start button when we have enough positions
        if len(self.selected_positions) == self.game.num_detectives + 1:
            detective_positions = self.selected_positions[:self.game.num_detectives]
            MrX_position = self.selected_positions[self.game.num_detectives]
            
            if MrX_position not in detective_positions:
                self.setup_controls.start_button.config(state=tk.NORMAL)
            else:
                self.setup_controls.start_button.config(state=tk.DISABLED)
        else:
            self.setup_controls.start_button.config(state=tk.DISABLED)
        
        self.draw_graph()
    

    def save_current_game(self):
        """Save the current game state"""
        if not self.game.game_state:
            messagebox.showerror("Error", "No game to save")
            return
        
        try:
            game_id = self.game_service.save_ui_game(
                self.game, 
                self.game_mode,
                self.detective_agent,
                self.MrX_agent,
                self.detective_agent_type.value if hasattr(self.detective_agent_type, 'value') else None,
                self.MrX_agent_type.value if hasattr(self.MrX_agent_type, 'value') else None
            )
            messagebox.showinfo("💾 Game Saved", f"Game saved as {game_id}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save game: {str(e)}")
    
    def load_game_from_file(self, file_path: str):
        """Load a game directly from a file path, bypassing the GameLoader's directory assumptions"""
        try:
            print(f"📂 Loading game from: {file_path}")
            
            with open(file_path, 'rb') as f:
                game_record = pickle.load(f)
                
            print(f"📦 Loaded object type: {type(game_record)}")
            
            # If it's already a ShadowChaseGame object, return it directly
            if isinstance(game_record, ShadowChaseGame):
                print("✅ Found ShadowChaseGame object directly")
                return game_record
                
            # If it's a GameRecord, reconstruct the game from it
            if hasattr(game_record, 'game_history') and hasattr(game_record, 'game_config'):
                print("🔧 Reconstructing game from GameRecord")
                reconstructed = self.loader._reconstruct_game_from_record(game_record)
                print(f"✅ Reconstruction result: {type(reconstructed)}")
                return reconstructed
                
            # Handle legacy format or other structures
            print(f"❌ Unrecognized game format: {type(game_record)}")
            if hasattr(game_record, '__dict__'):
                print(f"📋 Available attributes: {list(game_record.__dict__.keys())}")
            return None
            
        except Exception as e:
            print(f"❌ Error loading game from {file_path}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def show_load_dialog(self):
        """Show file browser to load a saved game"""
        try:
            # Open file browser in the root saved_games directory to allow access to all folders
            saved_games_root = "saved_games"
            if not os.path.exists(saved_games_root):
                messagebox.showinfo("No Games", "No saved games directory found")
                return
            
            # Show file dialog starting from the root saved_games directory
            game_file = filedialog.askopenfilename(
                title="Load Saved Game",
                initialdir=saved_games_root,
                filetypes=[("Game files", "*.pkl"), ("All files", "*.*")]
            )
            
            if game_file:
                # Extract game ID from filename
                game_id = os.path.basename(game_file).replace('.pkl', '')
                
                try:
                    # Load the game directly from the selected file path
                    loaded_game = self.load_game_from_file(game_file)
                    if loaded_game:
                        # Always open replay window directly
                        self.open_game_replay(game_id, loaded_game)
                            
                    else:
                        messagebox.showerror("Error", f"Failed to load game {game_id}")
                        
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to load game: {str(e)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open load dialog: {str(e)}")

    def show_video_export_dialog(self):
        """Show dialog to export game replay as video"""
        try:
            show_video_export_dialog(self.root, self.loader)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to show video export dialog: {str(e)}")

    def open_game_replay(self, game_id: str, game: ShadowChaseGame):
        """Open the game replay window"""
        try:
            replay_window = GameReplayWindow(self.root, game_id, game, self.loader)
            replay_window.show()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open replay: {str(e)}")

    def auto_save_completed_game(self):
        """Automatically save the game when it completes"""
        if not self.game.game_state or not self.game.is_game_over():
            return
        
        try:
            game_id = self.game_service.save_ui_game(
                self.game,
                self.game_mode,
                self.detective_agent, 
                self.MrX_agent,
                self.detective_agent_type.value if hasattr(self.detective_agent_type, 'value') else None,
                self.MrX_agent_type.value if hasattr(self.MrX_agent_type, 'value') else None
            )
            messagebox.showinfo("🎉 Game Completed & Saved!", 
                              f"Game automatically saved as: {game_id}\n\n"
                              f"You can replay it anytime from the Load Game menu.")
        except Exception as e:
            messagebox.showerror("Auto-Save Error", f"Failed to auto-save game: {str(e)}")

    def run(self):
        """Start the GUI application"""
        self.root.mainloop()
    
    
    
    def update_available_moves(self):
        """Update available moves for the current player by querying the game object."""
        self.current_player_moves = {}
        self.highlighted_edges = []
        self.active_player_positions = []
        self.available_moves_text = ""  # Store text for enhanced display
    
        if not self.game.game_state or self.game.is_game_over():
            self.available_moves_text = "No moves available - game over"
            return
    
        is_shadow_chase = isinstance(self.game, ShadowChaseGame)
        current_player = self.game.game_state.turn
        moves_text_lines = []

        if current_player == Player.DETECTIVES:
            self.game.game_state.double_move_active = False # Reset on detectives' turn
            self.game_controls.MrX_selections = []
            if self.current_detective_index < self.game.num_detectives:
                detective_pos = self.game.game_state.detective_positions[self.current_detective_index]
                self.active_player_positions = [detective_pos]
                
                moves_text_lines.append(f"🕵️ Detective {self.current_detective_index + 1} Moves (from {detective_pos}):")
                moves_text_lines.append("")
                
                # The get_valid_moves method in ShadowChaseGame already filters by tickets and occupied positions.
                if is_shadow_chase:
                    valid_moves = self.game.get_valid_moves(Player.DETECTIVES, detective_pos, pending_moves=self.detective_selections)
                else:
                    valid_moves = self.game.get_valid_moves(Player.DETECTIVES, detective_pos)
                
                self.current_player_moves[detective_pos] = {}
                
                # Group moves by transport type for better display
                transport_moves = {}
                
                for move in valid_moves:
                    if is_shadow_chase:
                        dest, transport = move
                        transport_name = transport.name.lower()
                        if transport_name not in transport_moves:
                            transport_moves[transport_name] = []
                        transport_moves[transport_name].append(dest)
                        
                        if dest not in self.current_player_moves[detective_pos]:
                            self.current_player_moves[detective_pos][dest] = []
                        self.current_player_moves[detective_pos][dest].append(transport.value)
                        self.highlighted_edges.append((detective_pos, dest, transport.value))
                    else: # Standard Game
                        dest = move
                        if 'taxi' not in transport_moves:
                            transport_moves['taxi'] = []
                        transport_moves['taxi'].append(dest)
                        self.current_player_moves[detective_pos][dest] = [1] # Generic transport
                        self.highlighted_edges.append((detective_pos, dest, 1))
                
                # Format moves text by transport type
                for transport_type, destinations in transport_moves.items():
                    if destinations:
                        moves_text_lines.append(f"{transport_type.upper()}: {', '.join(map(str, sorted(destinations)))}")

        else:  # MrX's turn
            # Always use the current MrX position, regardless of double move state
            MrX_pos = self.game.game_state.MrX_position
            self.active_player_positions = [MrX_pos]
            
            moves_text_lines.append(f"🕵️‍♂️ Mr. X Moves (from {MrX_pos}):")
            moves_text_lines.append("")
            
            valid_moves = self.game.get_valid_moves(Player.MRX, MrX_pos)
            
            self.current_player_moves[MrX_pos] = {}
            MrX_tickets = self.game.get_MrX_tickets()
            
            # Group moves by transport type
            transport_moves = {}
            black_destinations = set()  # Track destinations already added to black ticket list
            
            for move in valid_moves:
                if is_shadow_chase:
                    dest, transport = move
                    
                    # Initialize destination if not exists
                    if dest not in self.current_player_moves[MrX_pos]:
                        self.current_player_moves[MrX_pos][dest] = []
                    
                    # Check if Mr. X can use the specific transport type
                    required_ticket = TicketType[transport.name]
                    if MrX_tickets.get(required_ticket, 0) > 0:
                        transport_name = transport.name.lower()
                        if transport_name not in transport_moves:
                            transport_moves[transport_name] = []
                        transport_moves[transport_name].append(dest)
                        
                        self.current_player_moves[MrX_pos][dest].append(transport.value)
                        self.highlighted_edges.append((MrX_pos, dest, transport.value))
                    
                    # Check if Mr. X can use black ticket for this destination (only add once per destination)
                    if (MrX_tickets.get(TicketType.BLACK, 0) > 0 and 
                        dest not in black_destinations):
                        
                        black_destinations.add(dest)
                        if 'black' not in transport_moves:
                            transport_moves['black'] = []
                        transport_moves['black'].append(dest)
                            
                        if TransportType.BLACK.value not in self.current_player_moves[MrX_pos][dest]:
                            self.current_player_moves[MrX_pos][dest].append(TransportType.BLACK.value)
                            self.highlighted_edges.append((MrX_pos, dest, TransportType.BLACK.value))
                else: # Standard Game
                    dest = move
                    if 'taxi' not in transport_moves:
                        transport_moves['taxi'] = []
                    transport_moves['taxi'].append(dest)
                    self.current_player_moves[MrX_pos][dest] = [1]
                    self.highlighted_edges.append((MrX_pos, dest, 1))
            
            # Format moves text by transport type
            for transport_type, destinations in transport_moves.items():
                if destinations:
                    moves_text_lines.append(f"{transport_type.upper()}: {', '.join(map(str, sorted(set(destinations))))}")

        # Store the formatted text for the enhanced moves display
        if not moves_text_lines or len(moves_text_lines) <= 2:
            self.available_moves_text = "No valid moves available"
        else:
            self.available_moves_text = "\n".join(moves_text_lines)

    def handle_game_click(self, node):
        """Handle a node click during the game by checking against available moves."""
        if not self.game.game_state or self.game.is_game_over() or not self.active_player_positions:
            return
    
        source_pos = self.active_player_positions[0]
        
        # Check if the clicked node is a valid destination from the active player's position.
        if source_pos not in self.current_player_moves or node not in self.current_player_moves[source_pos]:
            messagebox.showwarning("Invalid Move", f"Position {node} is not a valid move for the current player.")
            return
    
        is_shadow_chase = isinstance(self.game, ShadowChaseGame)
        current_player = self.game.game_state.turn
    
        if current_player == Player.DETECTIVES:
            if is_shadow_chase:
                # Get available moves for this detective (already filtered by tickets)
                detective_pos = self.game.game_state.detective_positions[self.current_detective_index]
                valid_moves = self.game.get_valid_moves(Player.DETECTIVES, detective_pos, pending_moves=self.detective_selections)
                
                # Filter for only the clicked destination - these are already ticket-filtered
                available_transports = []
                for move in valid_moves:
                    if move[0] == node:
                        available_transports.append(move[1])
                
                if not available_transports:
                    messagebox.showerror("Error", "No valid transport found for this destination")
                    return
                
                # Use the new transport selection system
                if len(available_transports) > 1:
                    # Show transport selection buttons instead of dialog
                    self.game_controls.show_transport_selection(node, available_transports)
                    return  # Wait for transport selection
                else:
                    # Only one transport option - select automatically
                    transport = available_transports[0]
                    self.detective_selections.append((node, transport))
            else:
                self.detective_selections.append(node)
            
            self.selected_nodes.append(node)
            self.current_detective_index += 1
    
            if len(self.detective_selections) == self.game.num_detectives:
                self.game_controls.move_button.config(state=tk.NORMAL)
        
        else:  # MrX's turn
            if is_shadow_chase:
                # Get available moves for Mr. X (already filtered by tickets)
                valid_moves = self.game.get_valid_moves(Player.MRX, source_pos)
                
                # Filter for only the clicked destination - these are already ticket-filtered
                available_transports = []
                for move in valid_moves:
                    if move[0] == node:
                        available_transports.append(move[1])
                
                if not available_transports:
                    messagebox.showerror("Error", "No valid transport found for this destination")
                    return
                
                # Check if Mr. X can use black ticket for this destination
                MrX_tickets = self.game.get_MrX_tickets()
                can_use_black = (MrX_tickets.get(TicketType.BLACK, 0) > 0 and 
                               any(move[0] == node for move in valid_moves))
                
                # Use the new transport selection system
                if len(available_transports) > 1 or can_use_black:
                    # Show transport selection buttons instead of dialog
                    self.game_controls.show_transport_selection(node, available_transports, can_use_black)
                    return  # Wait for transport selection
                else:
                    # Only one transport option - select automatically
                    selected_transport = available_transports[0] if available_transports else None
                    if selected_transport is None:
                        return
                    
                    # Always treat as a single move - the double move logic is handled in make_move
                    self.game_controls.MrX_selections = [(node, selected_transport)]
                    self.selected_nodes = [node]
                    self.game_controls.move_button.config(state=tk.NORMAL)
                    
                    # Clear the black ticket checkbox since we're using the dialog now
                    self.game_controls.use_black_ticket.set(False)
                
            else: # Standard game
                self.selected_positions = [node]
                self.selected_nodes = [node]
                self.game_controls.move_button.config(state=tk.NORMAL)
    
        self.draw_graph()
    
    def update_ui_visibility(self):
        """Update UI section visibility based on game state"""
        # Clear selections that might be leftover from previous turns
        if not self.setup_mode and self.game.game_state:
            current_turn = getattr(self.game.game_state, 'turn', None)
            if current_turn and hasattr(current_turn, 'value'):
                # If it's not detectives' turn, clear detective selections
                if current_turn.value != 'detectives':
                    self.detective_selections = []
                    self.current_detective_index = 0
                # If it's not Mr. X's turn, clear Mr. X selections
                if current_turn.value != 'mrx':
                    self.game_controls.MrX_selections = []
        
        if self.setup_mode:
            # Setup mode - show only setup controls
            self.setup_section.pack(fill=tk.X, pady=(0, 10))
            self.controls_section.pack_forget()
            self.mrx_section.pack_forget()
            self.turn_display.pack_forget()
            self.moves_display.pack_forget()
            self.tickets_table_display.pack_forget()
            
        else:
            # Game mode - show all relevant controls
            self.setup_section.pack_forget()
            self.controls_section.pack(fill=tk.X, pady=(0, 10))
            self.turn_display.pack(fill=tk.X, pady=(0, 10))
            self.moves_display.pack(fill=tk.X, pady=(0, 10))
            
            # Setup AI agents if not already done
            if not hasattr(self, 'agents_initialized'):
                self.setup_ai_agents()
                self.agents_initialized = True
            
            # Show Mr. X controls only when it's Mr. X's turn in Shadow Chase
            is_shadow_chase = isinstance(self.game, ShadowChaseGame)
            is_mrx_turn = (self.game.game_state and 
                          self.game.game_state.turn == Player.MRX)
            
            if is_shadow_chase and is_mrx_turn and self.is_current_player_human():
                self.mrx_section.pack(fill=tk.X, pady=(0, 10))
            else:
                self.mrx_section.pack_forget()
            
            # Show only ticket table for Shadow Chase games
            if is_shadow_chase:
                if hasattr(self, 'tickets_table_display'):
                    self.tickets_table_display.pack(fill=tk.X, pady=(0, 10))
            else:
                if hasattr(self, 'tickets_table_display'):
                    self.tickets_table_display.pack_forget()
            
            # For AI players, automatically make selections but wait for continue button
            if self.is_current_player_ai():
                self.make_ai_selections()

    def setup_graph_display(self):
        """Setup matplotlib graph display"""
        super().setup_graph_display(self.graph_frame)
        
        # Mouse click handler
        self.canvas.mpl_connect('button_press_event', self.on_graph_click)
        
        self.draw_graph()

    def draw_graph(self):
        """Draw the game graph with parallel edges for multiple transport types"""
        self.ax.clear()
        
        # Draw board image first (as background)
        if hasattr(self, 'show_board_image') and self.show_board_image:
            self.draw_board_image()
        
        # Update available moves for highlighting
        if not self.setup_mode:
            self.update_available_moves()
        
        # Collect highlighted edges by transport type
        highlighted_by_transport = {}
        for edge_info in self.highlighted_edges:
            if len(edge_info) == 3:  # (from, to, transport)
                from_pos, to_pos, transport = edge_info
                if transport not in highlighted_by_transport:
                    highlighted_by_transport[transport] = []
                highlighted_by_transport[transport].append((from_pos, to_pos))
        
        # Determine whether to show edges based on board image overlay
        show_edges = not (hasattr(self, 'show_board_image') and self.show_board_image and not self.show_edges_with_image)
        
        # Use base class method for drawing edges with highlighting
        self.draw_edges_with_parallel_positioning(
            alpha=0.3, 
            highlighted_edges=highlighted_by_transport,
            show_edges=show_edges
        )
        
        # Draw nodes based on game state - use unified coloring system
        # Always get colors for all nodes to ensure consistency
        ui_state = {
            'selected_positions': getattr(self, 'selected_positions', []),
            'active_player_positions': getattr(self, 'active_player_positions', []),
            'detective_selections': getattr(self, 'detective_selections', [])
        }
        all_node_colors, all_node_sizes = self.get_node_colors_and_sizes(mode="live", ui_state=ui_state)
        
        if hasattr(self, 'show_board_image') and self.show_board_image:
            # Show only active nodes when board image is displayed
            active_nodes = self._get_active_nodes()
            filtered_pos = {node: pos for node, pos in self.pos.items() if node in active_nodes}
            
            if filtered_pos:
                # Filter colors and sizes for active nodes only, maintaining color consistency
                # Create the subgraph first to get the actual node order that NetworkX will use
                subgraph = self.game.graph.subgraph(active_nodes)
                subgraph_nodes_list = list(subgraph.nodes())  # This is the order NetworkX will actually use
                
                # Create mapping from original graph nodes to their indices
                all_nodes_list = list(self.game.graph.nodes())
                node_to_index = {node: i for i, node in enumerate(all_nodes_list)}
                
                # Get colors and sizes in the order that matches the subgraph node order
                filtered_colors = [all_node_colors[node_to_index[node]] for node in subgraph_nodes_list]
                filtered_sizes = [all_node_sizes[node_to_index[node]] for node in subgraph_nodes_list]
                
                # Create positions dict that matches the subgraph nodes
                ordered_filtered_pos = {node: filtered_pos[node] for node in subgraph_nodes_list if node in filtered_pos}
                
                # Draw nodes - now the colors/sizes align with NetworkX's internal node order
                nx.draw_networkx_nodes(subgraph, ordered_filtered_pos, ax=self.ax,
                                      node_color=filtered_colors, node_size=filtered_sizes)
                
                # Draw labels only for active nodes
                nx.draw_networkx_labels(subgraph, ordered_filtered_pos, ax=self.ax, 
                                      font_size=8)
        else:
            # Show all nodes when board image is not displayed - use the same coloring system
            nx.draw_networkx_nodes(self.game.graph, self.pos, ax=self.ax,
                                  node_color=all_node_colors, node_size=all_node_sizes)
            # Draw labels
            nx.draw_networkx_labels(self.game.graph, self.pos, ax=self.ax, font_size=8)
        
        # Draw heuristic shadows for possible Mr. X positions (only during detective turns)
        self._draw_heuristic_shadows()
        
        # Draw black dotted rings around selected nodes
        for node in self.selected_nodes:
            if node in self.pos:
                x, y = self.pos[node]
                circle = plt.Circle((x, y), 0.04, fill=False, color='black', 
                                  linewidth=2, linestyle='--', alpha=0.8)
                self.ax.add_patch(circle)
        
        # Draw circles around available move destinations
        if not self.setup_mode and self.active_player_positions:
            # Get current player's available destinations
            available_destinations = set()
            for source_pos in self.active_player_positions:
                if source_pos in self.current_player_moves:
                    available_destinations.update(self.current_player_moves[source_pos].keys())
            
            # Draw black circles around available destinations
            for node in available_destinations:
                if node in self.pos and node not in self.selected_nodes:  # Don't double-circle selected nodes
                    x, y = self.pos[node]
                    circle = plt.Circle((x, y), 0.038, fill=False, color='black', 
                                      linewidth=3, alpha=1)
                    self.ax.add_patch(circle)

        self.ax.set_title("Shadow Chase Game", fontsize=14, fontweight='bold')

        # Use base class method for legend - only show when edges are visible
        if show_edges:
            self.draw_transport_legend()
        
        self.ax.axis('off')
        self.canvas.draw()
        
        self.update_info()
        
    def _get_active_nodes(self):
        """Get nodes that should be visible when board image is shown"""
        active_nodes = set()
        
        if not self.game.game_state:
            # During setup, show selected positions
            active_nodes.update(self.selected_positions)
        else:
            # Add detective positions
            if hasattr(self.game.game_state, 'detective_positions'):
                active_nodes.update(self.game.game_state.detective_positions)
            
            # Add Mr. X position if visible
            if hasattr(self.game.game_state, 'MrX_position'):
                if hasattr(self.game.game_state, 'MrX_visible') and self.game.game_state.MrX_visible:
                    active_nodes.add(self.game.game_state.MrX_position)
            
            # Add possible Mr. X positions during detective turns (heuristic shadows)
            if (hasattr(self.game.game_state, 'current_turn') and 
                self.game.game_state.current_turn.value == 'detective' and 
                hasattr(self, 'heuristics') and self.heuristics):
                try:
                    possible_positions = self.heuristics.get_possible_positions()
                    if possible_positions:
                        active_nodes.update(possible_positions)
                except:
                    pass
            
            # Add nodes from current highlighted moves
            for edge_info in self.highlighted_edges:
                if len(edge_info) >= 2:
                    active_nodes.add(edge_info[0])  # from node
                    active_nodes.add(edge_info[1])  # to node
            
            # Add selected nodes
            active_nodes.update(self.selected_nodes)
        
        return active_nodes
    
    def toggle_board_image_overlay(self):
        """Toggle the board image overlay"""
        if hasattr(self, 'toggle_board_image'):
            is_shown = self.toggle_board_image()
            if is_shown:
                self.toggle_board_button.config(text="🖼️ Hide Board Image")
            else:
                self.toggle_board_button.config(text="🖼️ Show Board Image")
            self.draw_graph()
        else:
            messagebox.showinfo("Info", "Board image functionality not available")
    
    def open_calibrator(self):
        """Open the board overlay calibrator"""
        try:
            import subprocess
            import sys
            
            # Run the calibrator script
            subprocess.Popen([sys.executable, "calibrate_board_overlay.py"])
            messagebox.showinfo("Calibrator", "Board overlay calibrator opened in a new window")
        except Exception as e:
            messagebox.showerror("Error", f"Could not open calibrator: {e}")
    
    def refresh_calibration(self):
        """Refresh calibration parameters and redraw"""
        try:
            # Reload calibration parameters
            if hasattr(self, 'load_calibration_parameters'):
                self.load_calibration_parameters()
            
            # Recalculate positions
            if hasattr(self.game, 'node_positions') and self.game.node_positions:
                self.calculate_calibrated_positions()
            
            # Redraw graph
            self.draw_graph()
            messagebox.showinfo("Success", "Calibration refreshed successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Could not refresh calibration: {e}")
    
    def _draw_heuristic_shadows(self):
        """Draw subtle shadows on nodes where Mr. X could possibly be located"""
        # Only draw shadows if heuristics are enabled and available
        if not (self.setup_controls.get_heuristics_enabled() and 
                HEURISTICS_AVAILABLE and 
                isinstance(self.game, ShadowChaseGame) and 
                self.game.game_state and
                not self.setup_mode):
            return
            
        # Only show during detective turns (when Mr. X is hidden)
        if (self.game.game_state.turn != Player.DETECTIVES or 
            self.game.game_state.MrX_visible):
            return
            
        # Initialize or update heuristics
        if self.heuristics is None:
            self._initialize_heuristics()
        else:
            # Update heuristics with current game state
            self.heuristics.update_game_state(self.game)
        
        if self.heuristics is None:
            return
            
        try:
            # Get possible Mr. X positions
            possible_positions = self.heuristics.get_possible_MrX_positions()
            
            if not possible_positions:
                return
            
            # Draw shadows for each possible position
            for node in possible_positions:
                if node in self.pos:
                    x, y = self.pos[node]
                    
                    # Draw a more visible shadow circle behind the node for testing
                    shadow = plt.Circle((x, y), 0.042, 
                                      fill=True, color='red', alpha=0.4, zorder=0)
                    self.ax.add_patch(shadow)
                    
                    # Draw a more visible red ring around possible positions
                    ring = plt.Circle((x, y), 0.042, fill=False, color='darkred', 
                                    linewidth=2.5, alpha=0.8, linestyle=':')
                    self.ax.add_patch(ring)
                    
        except Exception as e:
            print(f"Error drawing heuristic shadows: {e}")
            import traceback
            traceback.print_exc()
            # Disable heuristics if there's an error
            self.heuristics = None

    def setup_ai_agents(self):
        """Initialize AI agents based on game mode and selected agent types"""
        from agents import get_agent_registry
        registry = get_agent_registry()
        
        if self.game_mode == "human_vs_human":
            self.detective_agent = None
            self.MrX_agent = None
        elif self.game_mode == "human_det_vs_ai_mrx":
            # Human plays as detectives, AI plays as Mr. X
            self.detective_agent = None
            self.MrX_agent = registry.create_MrX_agent(self.MrX_agent_type) if isinstance(self.game, ShadowChaseGame) else None
        elif self.game_mode == "ai_det_vs_human_mrx":
            # AI plays as detectives, Human plays as Mr. X
            if isinstance(self.game, ShadowChaseGame):
                self.detective_agent = registry.create_multi_detective_agent(self.detective_agent_type, self.game.num_detectives)
            else:
                self.detective_agent = None
            self.MrX_agent = None
        elif self.game_mode == "ai_vs_ai":
            # AI plays both sides
            if isinstance(self.game, ShadowChaseGame):
                self.detective_agent = registry.create_multi_detective_agent(self.detective_agent_type, self.game.num_detectives)
                self.MrX_agent = registry.create_MrX_agent(self.MrX_agent_type)
            else:
                self.detective_agent = None
                self.MrX_agent = None

    def is_current_player_ai(self):
        """Check if the current player is AI-controlled"""
        if not self.game.game_state:
            return False
        
        current_player = self.game.game_state.turn
        
        if current_player == Player.DETECTIVES:
            return self.detective_agent is not None
        else:  # Player.MRX
            return self.MrX_agent is not None

    def is_current_player_human(self):
        """Check if the current player is human-controlled"""
        return not self.is_current_player_ai()

    def make_ai_move(self):
        """Make an AI move for the current player"""
        if not self.game.game_state or self.game.is_game_over():
            return False
        
        current_player = self.game.game_state.turn
        
        try:
            if current_player == Player.DETECTIVES and self.detective_agent:
                # AI detectives move
                detective_moves = self.detective_agent.choose_all_moves(self.game)
                if detective_moves:
                    success = self.game.make_move(detective_moves=detective_moves)
                    return success
                else:
                    return False
            
            elif current_player == Player.MRX and self.MrX_agent:
                # AI Mr. X moves
                move_result = self.MrX_agent.choose_move(self.game)
                if move_result and len(move_result) == 3:
                    dest, transport, use_double = move_result
                    success = self.game.make_move(MrX_moves=[(dest, transport)], use_double_move=use_double)
                    return success
                else:
                    return False
            
        except Exception as e:
            print(f"AI move error: {e}")  # Log to console instead of popup
            return False
        
        return False

    def make_ai_selections(self):
        """Make AI selections for display but don't execute the move yet"""
        if not self.game.game_state or self.game.is_game_over():
            return
        
        current_player = self.game.game_state.turn
        
        try:
            if current_player == Player.DETECTIVES and self.detective_agent:
                # Get AI detective moves for display
                detective_moves = self.detective_agent.choose_all_moves(self.game)
                if detective_moves:
                    # Set up the selections for display
                    self.detective_selections = detective_moves
                    self.selected_nodes = [move[0] for move in detective_moves]
                    self.current_detective_index = len(detective_moves)  # All detectives selected
                    self.game_controls.move_button.config(state=tk.NORMAL)
                    
            elif current_player == Player.MRX and self.MrX_agent:
                # Get AI Mr. X move for display
                move_result = self.MrX_agent.choose_move(self.game)
                if move_result and len(move_result) == 3:
                    dest, transport, use_double = move_result
                    # Set up the selections for display
                    self.game_controls.MrX_selections = [(dest, transport)]
                    self.selected_nodes = [dest]
                    
                    # Set the double move checkbox if AI wants to use it
                    if hasattr(self.game_controls, 'double_move_var'):
                        self.game_controls.double_move_var.set(use_double)
                    
                    self.game_controls.move_button.config(state=tk.NORMAL)
            
            # Redraw graph to show AI selections
            self.draw_graph()
            
        except Exception as e:
            print(f"AI selection error: {e}")  # Log to console instead of popup

    def check_and_make_ai_move(self):
        """Remove this method - no longer needed"""
        pass

    def _execute_ai_move(self):
        """Remove this method - no longer needed"""
        pass
        """Remove this method - no longer needed"""
        pass

    def _execute_ai_move(self):
        """Remove this method - no longer needed"""
        pass

    def set_auto_positions(self, positions: list):
        """Set positions programmatically and optionally start the game"""
        if len(positions) != self.game.num_detectives + 1:
            raise ValueError(f"Expected {self.game.num_detectives + 1} positions, got {len(positions)}")
        
        self.selected_positions = positions
        
        # Enable start button
        detective_positions = self.selected_positions[:self.game.num_detectives]
        MrX_position = self.selected_positions[self.game.num_detectives]
        
        if MrX_position not in detective_positions:
            self.setup_controls.start_button.config(state=tk.NORMAL)
        
        self.draw_graph()

    def auto_start_game(self):
        """Automatically start the game with current selected positions"""
        if len(self.selected_positions) == self.game.num_detectives + 1:
            self.setup_controls.start_game()
        else:
            raise ValueError("Not enough positions selected for auto-start")
