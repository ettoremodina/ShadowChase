import tkinter as tk
from tkinter import ttk, messagebox
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from ..core.game import Game, Player, GameState, ScotlandYardGame, TicketType, TransportType
from ..solver.minimax_solver import MinimaxSolver
from ..storage.game_loader import GameLoader
from .ui_components import ScrollableFrame, StyledButton, InfoDisplay
from .setup_controls import SetupControls
from .game_controls import GameControls

class GameVisualizer:
    """Interactive GUI for Cops and Robbers game"""
    
    def __init__(self, game: Game, loader: 'GameLoader' = None):
        self.loader = loader or GameLoader()
        self.game = game
        self.solver = MinimaxSolver(game)
        self.root = tk.Tk()
        self.root.title("🎯 Cops and Robbers Game")
        self.root.geometry(f"{self.root.winfo_screenwidth()}x{self.root.winfo_screenheight()}")
        self.root.configure(bg="#f8f9fa")
        
        # Transport type colors and styles
        self.transport_styles = {
            1: {'color': 'yellow', 'width': 2, 'name': 'Taxi'},
            2: {'color': 'blue', 'width': 3, 'name': 'Bus'},
            3: {'color': 'red', 'width': 4, 'name': 'Underground'},
            4: {'color': 'green', 'width': 3, 'name': 'Ferry'}
        }
        
        # Game state
        self.selected_positions = []
        self.setup_mode = True
        self.auto_play = False
        self.solver_result = None
        self.current_player_moves = {}
        self.highlighted_edges = []
        self.active_player_positions = []
        self.current_cop_index = 0
        self.cop_selections = []
        self.selected_nodes = []
        
        # Mr. X special moves state
        self.use_black_ticket = tk.BooleanVar()
        self.double_move_requested = False
        self.mr_x_selections = []

        # UI Controllers
        self.setup_controls = SetupControls(self)
        self.game_controls = GameControls(self)
        
        # UI components
        self.setup_ui()
        self.setup_graph_display()
        
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
        
        self.tickets_display = self.game_controls.create_tickets_display(
            self.scrollable_controls.scrollable_frame)
        self.tickets_display.pack(fill=tk.X, pady=(0, 10))
        
        # Solver section
        self.create_solver_section()
        
        # Save/Load section
        self.create_save_load_section()
        
        # Game info section
        self.create_info_section()
        
        # Right panel for graph
        self.graph_frame = ttk.Frame(main_frame)
        self.graph_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Initial UI state
        self.update_ui_visibility()
    
    def create_solver_section(self):
        """Create the solver controls section"""
        self.solver_section = ttk.LabelFrame(self.scrollable_controls.scrollable_frame, 
                                           text="🧠 AI Solver")
        
        style = ttk.Style()
        style.configure("Solver.TLabelframe.Label", anchor="w", font=('Arial', 11, 'bold'))
        self.solver_section.configure(style="Solver.TLabelframe")
        
        button_frame = ttk.Frame(self.solver_section)
        button_frame.pack(fill=tk.X, padx=10, pady=8)
        
        self.solve_button = StyledButton(button_frame, "🧠 Solve Game", 
                                       command=self.solve_game, style_type="primary",
                                       state=tk.DISABLED)
        self.solve_button.pack(fill=tk.X, pady=3)
        
        self.strategy_button = StyledButton(button_frame, "📊 Show Strategy", 
                                          command=self.show_strategy, state=tk.DISABLED)
        self.strategy_button.pack(fill=tk.X, pady=3)
        
        self.solver_section.pack(fill=tk.X, pady=(0, 10))
    
    def create_save_load_section(self):
        """Create the save/load controls section"""
        self.save_load_section = ttk.LabelFrame(self.scrollable_controls.scrollable_frame, 
                                              text="💾 Save & Load")
        
        style = ttk.Style()
        style.configure("SaveLoad.TLabelframe.Label", anchor="w", font=('Arial', 11, 'bold'))
        self.save_load_section.configure(style="SaveLoad.TLabelframe")
        
        button_frame = ttk.Frame(self.save_load_section)
        button_frame.pack(fill=tk.X, padx=10, pady=8)
        
        self.save_button = StyledButton(button_frame, "💾 Save Game", 
                                      command=self.save_current_game, state=tk.DISABLED)
        self.save_button.pack(fill=tk.X, pady=3)
        
        self.load_button = StyledButton(button_frame, "📂 Load Game", 
                                      command=self.show_load_dialog)
        self.load_button.pack(fill=tk.X, pady=3)
        
        self.history_button = StyledButton(button_frame, "📈 Game History", 
                                         command=self.show_game_history)
        self.history_button.pack(fill=tk.X, pady=3)
        
        self.save_load_section.pack(fill=tk.X, pady=(0, 10))
    
    def create_info_section(self):
        """Create the game information section"""
        self.info_display = InfoDisplay(self.scrollable_controls.scrollable_frame, 
                                      "ℹ️ Game Information", height=8)
        self.info_display.pack(fill=tk.X, pady=(0, 10))
    
    def update_ui_visibility(self):
        """Update UI section visibility based on game state"""
        if self.setup_mode:
            # Setup mode - show only setup controls
            self.setup_section.pack(fill=tk.X, pady=(0, 10))
            self.controls_section.pack_forget()
            self.mrx_section.pack_forget()
            self.turn_display.pack_forget()
            self.moves_display.pack_forget()
            self.tickets_display.pack_forget()
            self.solver_section.pack_forget()
            
            # Disable save during setup
            self.save_button.config(state=tk.DISABLED)
        else:
            # Game mode - show all relevant controls
            self.setup_section.pack_forget()
            self.controls_section.pack(fill=tk.X, pady=(0, 10))
            self.turn_display.pack(fill=tk.X, pady=(0, 10))
            self.moves_display.pack(fill=tk.X, pady=(0, 10))
            
            # Show Mr. X controls only when it's Mr. X's turn in Scotland Yard
            is_scotland_yard = isinstance(self.game, ScotlandYardGame)
            is_mrx_turn = (self.game.game_state and 
                          self.game.game_state.turn == Player.ROBBER)
            
            if is_scotland_yard and is_mrx_turn:
                self.mrx_section.pack(fill=tk.X, pady=(0, 10))
            else:
                self.mrx_section.pack_forget()
            
            # Show tickets only for Scotland Yard games
            if is_scotland_yard:
                self.tickets_display.pack(fill=tk.X, pady=(0, 10))
            else:
                self.tickets_display.pack_forget()
            
            # Show solver section
            self.solver_section.pack(fill=tk.X, pady=(0, 10))
            
            # Enable save during game
            if self.game.game_state:
                self.save_button.config(state=tk.NORMAL)

    def setup_graph_display(self):
        """Setup matplotlib graph display"""
        self.fig = Figure(figsize=(10, 8), dpi=100, facecolor='#f8f9fa')
        self.ax = self.fig.add_subplot(111)
        
        self.canvas = FigureCanvasTkAgg(self.fig, self.graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Mouse click handler
        self.canvas.mpl_connect('button_press_event', self.on_graph_click)
        
        # Calculate graph layout once
        self.pos = nx.spring_layout(self.game.graph, seed=42, k=1, iterations=50)
        self.draw_graph()
    
    def draw_graph(self):
        """Draw the game graph with move highlighting"""
        self.ax.clear()
        
        # Update available moves for highlighting
        if not self.setup_mode:
            self.update_available_moves()
        
        # Track which transport types are actually used for legend
        legend_handles = []
        
        # Collect highlighted edges by transport type
        highlighted_by_transport = {}
        for edge_info in self.highlighted_edges:
            if len(edge_info) == 3:  # (from, to, transport)
                from_pos, to_pos, transport = edge_info
                if transport not in highlighted_by_transport:
                    highlighted_by_transport[transport] = []
                highlighted_by_transport[transport].append((from_pos, to_pos))
        
        # Draw all edges with reduced transparency for non-highlighted ones
        for transport_type, style in self.transport_styles.items():
            edges_for_type = []
            highlighted_edges_for_type = highlighted_by_transport.get(transport_type, [])
            
            for u, v, data in self.game.graph.edges(data=True):
                edge_transports = data.get('transports', [])
                edge_type = data.get('edge_type', None)
                
                if transport_type in edge_transports or edge_type == transport_type:
                    edges_for_type.append((u, v))
            
            if edges_for_type:
                # Draw non-highlighted edges with low transparency
                non_highlighted = [edge for edge in edges_for_type if edge not in highlighted_edges_for_type]
                if non_highlighted:
                    nx.draw_networkx_edges(
                        self.game.graph, self.pos, 
                        edgelist=non_highlighted,
                        ax=self.ax,
                        edge_color=style['color'],
                        width=style['width'],
                        alpha=0.3
                    )
                
                # Draw highlighted edges with full color and increased thickness
                if highlighted_edges_for_type:
                    nx.draw_networkx_edges(
                        self.game.graph, self.pos,
                        edgelist=highlighted_edges_for_type,
                        ax=self.ax,
                        edge_color=style['color'],
                        width=style['width'] + 2,
                        alpha=1
                    )
                
                import matplotlib.lines as mlines
                legend_handles.append(mlines.Line2D([], [], color=style['color'], 
                                                  linewidth=style['width'], 
                                                  label=style['name']))
        
        # Color nodes based on game state
        node_colors = []
        node_sizes = []
        
        for node in self.game.graph.nodes():
            if self.setup_mode:
                if node in self.selected_positions:
                    if len(self.selected_positions) <= self.game.num_cops:
                        node_colors.append('blue')
                    else:
                        node_colors.append('red')
                    node_sizes.append(600)
                else:
                    node_colors.append('lightgray')
                    node_sizes.append(300)
            else:
                if self.game.game_state:
                    if node in self.active_player_positions:
                        if node in self.game.game_state.cop_positions:
                            node_colors.append('cyan')
                        else:
                            node_colors.append('orange')
                        node_sizes.append(700)
                    elif node in self.cop_selections:
                        node_colors.append('purple')
                        node_sizes.append(600)
                    elif node in self.game.game_state.cop_positions:
                        node_colors.append('blue')
                        node_sizes.append(500)
                    elif node == self.game.game_state.robber_position:
                        is_scotland_yard = isinstance(self.game, ScotlandYardGame)
                        if not is_scotland_yard or self.game.game_state.mr_x_visible:
                            node_colors.append('red')
                            node_sizes.append(500)
                        else:
                            node_colors.append('lightgray')
                            node_sizes.append(300)
                    else:
                        node_colors.append('lightgray')
                        node_sizes.append(300)
                else:
                    node_colors.append('lightgray')
                    node_sizes.append(300)
        
        # Draw nodes
        nx.draw_networkx_nodes(self.game.graph, self.pos, ax=self.ax,
                              node_color=node_colors, node_size=node_sizes)
        
        # Draw black dotted rings around selected nodes
        for node in self.selected_nodes:
            if node in self.pos:
                x, y = self.pos[node]
                circle = plt.Circle((x, y), 0.04, fill=False, color='black', 
                                  linewidth=2, linestyle='--', alpha=0.8)
                self.ax.add_patch(circle)
        
        # Draw labels
        nx.draw_networkx_labels(self.game.graph, self.pos, ax=self.ax, font_size=8)
        
        self.ax.set_title("🎯 Cops and Robbers Game", fontsize=14, fontweight='bold')
        
        if legend_handles:
            self.ax.legend(handles=legend_handles, loc='upper right')
        
        self.ax.axis('off')
        self.canvas.draw()
        
        self.update_info()
    
    def update_info(self):
        """Update game information display"""
        if not self.info_display:
            return
            
        if self.setup_mode:
            info_text = "🎯 Setup Mode\n"
            info_text += f"📊 Need: {self.game.num_cops} cops + 1 robber\n"
            info_text += f"✅ Selected: {len(self.selected_positions)}\n"
        elif self.game.game_state:
            state_info = self.game.get_state_representation()
            info_text = f"🎮 Turn: {state_info['turn'].title()}\n"
            info_text += f"📊 Turn count: {state_info['turn_count']}\n"
            info_text += f"👮 Cop positions: {state_info['cop_positions']}\n"
            
            is_scotland_yard = isinstance(self.game, ScotlandYardGame)
            if is_scotland_yard:
                if self.game.game_state.mr_x_visible:
                    info_text += f"🕵️‍♂️ Mr. X position: {state_info['robber_position']} (VISIBLE)\n"
                else:
                    info_text += f"🕵️‍♂️ Mr. X position: HIDDEN\n"
            else:
                info_text += f"🏃 Robber position: {state_info['robber_position']}\n"
            
            if self.game.is_game_over():
                winner = self.game.get_winner()
                if winner:
                    info_text += f"🏆 Game Over - Winner: {winner.value.title()}\n"
            else:
                info_text += "⏳ Game in progress\n"
            
            if self.solver_result:
                info_text += f"\n🧠 Solver: Cops can win = {self.solver_result.cops_can_win}\n"
        else:
            info_text = "ℹ️ No game initialized"
        
        self.info_display.set_text(info_text)
        
        # Update other displays
        self.game_controls.update_turn_display()
        self.game_controls.update_moves_display()
        self.game_controls.update_tickets_display()
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
            if len(self.selected_positions) < self.game.num_cops + 1:
                self.selected_positions.append(node)
        
        # Enable start button when we have enough positions
        if len(self.selected_positions) == self.game.num_cops + 1:
            cop_positions = self.selected_positions[:self.game.num_cops]
            robber_position = self.selected_positions[self.game.num_cops]
            
            if robber_position not in cop_positions:
                self.setup_controls.start_button.config(state=tk.NORMAL)
            else:
                self.setup_controls.start_button.config(state=tk.DISABLED)
        else:
            self.setup_controls.start_button.config(state=tk.DISABLED)
        
        self.draw_graph()
    
    def start_game(self):
        """Start the game with selected positions"""
        if len(self.selected_positions) != self.game.num_cops + 1:
            messagebox.showerror("Error", f"Select {self.game.num_cops} positions and 1 robber position")
            return
        
        cop_positions = self.selected_positions[:self.game.num_cops]
        robber_position = self.selected_positions[self.game.num_cops]
        
        if robber_position in cop_positions:
            messagebox.showerror("Error", "Robber and cops cannot start in the same position")
            return
        
        # Reset game state
        self.game.game_state = None
        self.game.game_history = []
        
        try:
            # Use appropriate initialization method
            if isinstance(self.game, ScotlandYardGame):
                self.game.initialize_scotland_yard_game(cop_positions, robber_position)
            else:
                self.game.initialize_game(cop_positions, robber_position)
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            return
        
        self.setup_mode = False
        self.selected_positions = []
        
        # Update UI
        self.setup_controls.start_button.config(state=tk.DISABLED)
        self.game_controls.move_button.config(state=tk.DISABLED)
        self.game_controls.auto_button.config(state=tk.NORMAL)
        self.solve_button.config(state=tk.NORMAL)
        self.save_button.config(state=tk.NORMAL)
        
        self.draw_graph()
    
    def reset_setup(self):
        """Reset game setup"""
        self.selected_positions = []
        self.setup_mode = True
        self.auto_play = False
        self.solver_result = None
        self.current_player_moves = {}
        self.highlighted_edges = []
        self.active_player_positions = []
        self.current_cop_index = 0
        self.cop_selections = []
        self.selected_nodes = []
        
        self.setup_controls.start_button.config(state=tk.DISABLED)
        self.game_controls.move_button.config(state=tk.DISABLED)
        self.game_controls.auto_button.config(state=tk.NORMAL, text="Auto Play")
        self.solve_button.config(state=tk.DISABLED)
        self.strategy_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.DISABLED)
        
        # Clear game state
        self.game.game_state = None
        self.game.game_history = []
        self.draw_graph()
    
    def toggle_auto_play(self):
        """Toggle automatic play mode"""
        self.auto_play = not self.auto_play
        
        if self.auto_play:
            self.game_controls.auto_button.configure(text="⏹️ Stop Auto")
            self.root.after(1000, self.auto_play_step)
        else:
            self.game_controls.auto_button.configure(text="🤖 Auto Play")
    
    def auto_play_step(self):
        """Execute one step of automatic play"""
        if not self.auto_play or self.game.is_game_over():
            self.auto_play = False
            self.game_controls.auto_button.configure(text="🤖 Auto Play")
            return
        
        if self.solver_result and self.solver_result.cops_can_win:
            # Use solver strategy
            if self.game.game_state.turn == Player.COPS and self.solver_result.cop_strategy:
                move = self.solver_result.cop_strategy.get_move(self.game.game_state)
                if move:
                    self.game.make_move(new_positions=move)
            elif self.game.game_state.turn == Player.ROBBER and self.solver_result.robber_strategy:
                move = self.solver_result.robber_strategy.get_move(self.game.game_state)
                if move:
                    self.game.make_move(new_robber_pos=move[0])
        else:
            # Make random valid move
            self.game.make_random_move()
        
        self.draw_graph()
        
        if self.game.is_game_over():
            winner = self.game.get_winner()
            messagebox.showinfo("🏆 Game Over", f"{winner.value.title()} wins!")
            self.auto_play = False
            self.game_controls.auto_button.configure(text="🤖 Auto Play")
             
            # Save the game automatically when it ends
            try:
                game_id = self.game.save_game(self.loader)
                messagebox.showinfo("💾 Game Saved", f"Game saved as {game_id}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save game: {str(e)}")
        else:
            self.root.after(1000, self.auto_play_step)
    
    def solve_game(self):
        """Solve the current game"""
        if not self.game.game_state:
            messagebox.showerror("Error", "No game initialized")
            return
        
        try:
            self.info_display.insert("🧠 Solving game...\n")
            self.root.update()
            
            self.solver_result = self.solver.solve(
                self.game.game_history[0].cop_positions,
                self.game.game_history[0].robber_position
            )
            
            result_text = f"🧠 Solver Result:\n"
            result_text += f"✅ Cops can win: {self.solver_result.cops_can_win}\n"
            if self.solver_result.game_length:
                result_text += f"⏱️ Game length: {self.solver_result.game_length}\n"
            result_text += "\n"
            
            self.info_display.insert(result_text)
            self.strategy_button.config(state=tk.NORMAL)
            
        except Exception as e:
            messagebox.showerror("Error", f"Solver failed: {str(e)}")
    
    def show_strategy(self):
        """Show optimal strategy"""
        if not self.solver_result:
            messagebox.showerror("Error", "No solver result available")
            return
        
        strategy_window = tk.Toplevel(self.root)
        strategy_window.title("🧠 Game Strategy")
        strategy_window.geometry("900x700")
        
        text_widget = tk.Text(strategy_window, wrap=tk.WORD, font=('Consolas', 9))
        scrollbar = ttk.Scrollbar(strategy_window, orient="vertical", command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        strategy_text = f"🧠 Optimal Strategy:\n\n"
        strategy_text += f"✅ Cops can win: {self.solver_result.cops_can_win}\n\n"
        
        if self.solver_result.cops_can_win and self.solver_result.cop_strategy:
            strategy_text += "👮 Cop Strategy:\n"
            for state_key, move in self.solver_result.cop_strategy.moves.items():
                strategy_text += f"State {state_key}: Move to {move}\n"
        
        if not self.solver_result.cops_can_win and self.solver_result.robber_strategy:
            strategy_text += "🏃 Robber Strategy:\n"
            for state_key, move in self.solver_result.robber_strategy.moves.items():
                strategy_text += f"State {state_key}: Move to {move}\n"
        
        text_widget.insert(tk.END, strategy_text)
        text_widget.config(state=tk.DISABLED)
    
    def update_available_moves(self):
        """Update available moves for the current player"""
        self.current_player_moves = {}
        self.highlighted_edges = []
        self.active_player_positions = []
    
        if not self.game.game_state or self.game.is_game_over():
            return
    
        is_scotland_yard = isinstance(self.game, ScotlandYardGame)
        current_player = self.game.game_state.turn
    
        if current_player == Player.COPS:
            self.game_controls.double_move_requested = False # Reset on cops' turn
            self.game_controls.mr_x_selections = []
            if self.current_cop_index < self.game.num_cops:
                cop_pos = self.game.game_state.cop_positions[self.current_cop_index]
                self.active_player_positions = [cop_pos]
                
                valid_moves = self.game.get_valid_moves(Player.COPS, cop_pos, pending_moves=self.cop_selections)
                
                self.current_player_moves[cop_pos] = {}
                for move in valid_moves:
                    if is_scotland_yard:
                        dest, transport = move
                        if dest not in self.current_player_moves[cop_pos]:
                            self.current_player_moves[cop_pos][dest] = []
                        self.current_player_moves[cop_pos][dest].append(transport.value)
                        self.highlighted_edges.append((cop_pos, dest, transport.value))
                    else: # Standard Game
                        dest = move
                        self.current_player_moves[cop_pos][dest] = [1] # Generic transport
                        self.highlighted_edges.append((cop_pos, dest, 1))

        else:  # Robber's turn
            robber_pos = self.game.game_state.robber_position
            self.active_player_positions = [robber_pos]
            
            valid_moves = self.game.get_valid_moves(Player.ROBBER, robber_pos)
            
            self.current_player_moves[robber_pos] = {}
            for move in valid_moves:
                if is_scotland_yard:
                    dest, transport = move
                    mr_x_tickets = self.game.get_mr_x_tickets()
                    transports_for_move = []
                    
                    required_ticket = TicketType[transport.name]
                    if mr_x_tickets.get(required_ticket, 0) > 0:
                        transports_for_move.append(transport.value)
                    if mr_x_tickets.get(TicketType.BLACK, 0) > 0:
                        transports_for_move.append(TransportType.BLACK.value)
                    
                    if transports_for_move:
                        self.current_player_moves[robber_pos][dest] = transports_for_move
                        for t_val in transports_for_move:
                            self.highlighted_edges.append((robber_pos, dest, t_val))
                else: # Standard Game
                    dest = move
                    self.current_player_moves[robber_pos][dest] = [1]
                    self.highlighted_edges.append((robber_pos, dest, 1))

    def handle_game_click(self, node):
        """Handle a node click during the game by checking against available moves."""
        if not self.game.game_state or self.game.is_game_over() or not self.active_player_positions:
            return
    
        source_pos = self.active_player_positions[0]
        
        # Check if the clicked node is a valid destination from the active player's position.
        if source_pos not in self.current_player_moves or node not in self.current_player_moves[source_pos]:
            messagebox.showwarning("Invalid Move", f"Position {node} is not a valid move for the current player.")
            return
    
        is_scotland_yard = isinstance(self.game, ScotlandYardGame)
        current_player = self.game.game_state.turn
    
        if current_player == Player.COPS:
            if is_scotland_yard:
                # For detectives, there's only one transport type per edge.
                transport_value = self.current_player_moves[source_pos][node][0]
                transport = TransportType(transport_value)
                self.cop_selections.append((node, transport))
            else:
                self.cop_selections.append(node)
            
            self.selected_nodes.append(node)
            self.current_cop_index += 1
    
            if len(self.cop_selections) == self.game.num_cops:
                self.game_controls.move_button.config(state=tk.NORMAL)
        
        else:  # Robber's turn
            if is_scotland_yard:
                # Determine transport type for the move
                if self.game_controls.use_black_ticket.get():
                    transport = TransportType.BLACK
                else:
                    # Prefer specific ticket over black if not explicitly chosen
                    available_transports = self.current_player_moves[source_pos][node]
                    transport_value = min(available_transports) # Prefers non-black
                    transport = TransportType(transport_value)

                self.game_controls.mr_x_selections = [(node, transport)]
                self.selected_nodes = [node]
                self.game_controls.move_button.config(state=tk.NORMAL)
            else: # Standard game
                self.selected_positions = [node]
                self.selected_nodes = [node]
                self.game_controls.move_button.config(state=tk.NORMAL)
    
        self.draw_graph()

    def skip_turn(self):
        """Handles a detective skipping their turn when they have no moves."""
        if self.game.game_state.turn == Player.COPS and self.current_cop_index < self.game.num_cops:
            cop_pos = self.game.game_state.cop_positions[self.current_cop_index]
            
            # Verify there are no moves
            if cop_pos in self.current_player_moves and not self.current_player_moves[cop_pos]:
                self.cop_selections.append((cop_pos, None)) # Append a "stay" move
                self.current_cop_index += 1
                
                if len(self.cop_selections) == self.game.num_cops:
                    self.game_controls.move_button.config(state=tk.NORMAL)
                
                self.draw_graph()
            else:
                messagebox.showwarning("Invalid Action", "Skip turn is only for players with no available moves.")
        
        self.skip_button.config(state=tk.DISABLED)

    def make_manual_move(self):
        """Make a manual move by sending selected moves to the game object."""
        if (self.game.game_state.turn == Player.COPS and len(self.cop_selections) != self.game.num_cops) or \
           (self.game.game_state.turn == Player.ROBBER and not self.mr_x_selections and not self.selected_positions):
            messagebox.showwarning("Invalid Selection", "A move must be selected for all players.")
            return
    
        try:
            is_scotland_yard = isinstance(self.game, ScotlandYardGame)
            success = False
    
            if self.game.game_state.turn == Player.COPS:
                if is_scotland_yard:
                    success = self.game.make_move(detective_moves=self.cop_selections)
                else:
                    success = self.game.make_move(new_positions=self.cop_selections)
            else:  # Robber's turn
                if is_scotland_yard:
                    success = self.game.make_move(mr_x_moves=self.mr_x_selections, 
                                                use_double_move=self.game_controls.double_move_requested)
                else:
                    success = self.game.make_move(new_robber_pos=self.selected_positions[0])
    
            if not success:
                messagebox.showerror("Invalid Move", "The move was rejected by the game engine.")
    
            # Reset UI state after move attempt
            self.selected_positions = []
            self.cop_selections = []
            self.mr_x_selections = []
            self.current_cop_index = 0
            self.selected_nodes = []
            self.game_controls.move_button.config(state=tk.DISABLED)
            self.game_controls.double_move_requested = False
            self.game_controls.use_black_ticket.set(False)
            self.update_ui_visibility()
            self.draw_graph()
    
            # Check for game over
            if self.game.is_game_over():
                winner = self.game.get_winner()
                winner_name = winner.value.title() if winner else "No one"
                messagebox.showinfo("Game Over", f"{winner_name} wins!")
                self.game_controls.auto_button.config(state=tk.DISABLED)
    
        except Exception as e:
            messagebox.showerror("Move Error", f"An error occurred while making the move: {str(e)}")
            # Reset UI state on error
            self.selected_positions = []
            self.cop_selections = []
            self.mr_x_selections = []
            self.current_cop_index = 0
            self.selected_nodes = []
            self.game_controls.move_button.config(state=tk.DISABLED)
            self.game_controls.double_move_requested = False
            self.game_controls.use_black_ticket.set(False)
            self.draw_graph()

    def toggle_double_move(self):
        """Request double move for next Mr. X turn."""
        self.game_controls.double_move_requested = not self.game_controls.double_move_requested
        if self.game_controls.double_move_requested:
            self.game_controls.double_move_button.config(text="⚡ Cancel Double Move")
        else:
            self.game_controls.double_move_button.config(text="⚡ Use Double Move")
        """Toggles the double move state for Mr. X."""
        self.game.game_state.double_move_active = not self.game.game_state.double_move_active
        if self.game.game_state.double_move_active:
            self.game_controls.double_move_button.config(relief=tk.SUNKEN)
        else:
            self.game_controls.double_move_button.config(relief=tk.RAISED)
            # Reset selections if double move is cancelled
            self.game_controls.mr_x_selections = []
            self.selected_nodes = []
        self.draw_graph()

    def update_mrx_controls(self):
        """Updates the state of Mr. X's special move controls."""
        if self.game.game_state and self.game.game_state.turn == Player.ROBBER:
            mr_x_tickets = self.game.get_mr_x_tickets()
            
            if mr_x_tickets.get(TicketType.DOUBLE_MOVE, 0) > 0:
                self.game_controls.double_move_button.config(state=tk.NORMAL)
            else:
                self.game_controls.double_move_button.config(state=tk.DISABLED)
                self.game.game_state.double_move_active = False
                self.game_controls.double_move_button.config(relief=tk.RAISED)

            if mr_x_tickets.get(TicketType.BLACK, 0) > 0:
                self.game_controls.black_ticket_check.config(state=tk.NORMAL)
            else:
                self.game_controls.black_ticket_check.config(state=tk.DISABLED)
                self.game_controls.use_black_ticket.set(False)
        else:
            self.game_controls.double_move_button.config(state=tk.DISABLED)
            self.game_controls.black_ticket_check.config(state=tk.DISABLED)
            self.game.game_state.double_move_active = False
            self.game_controls.double_move_button.config(relief=tk.RAISED)
            self.game_controls.use_black_ticket.set(False)

    def run(self):
        """Start the GUI application"""
        self.root.mainloop()
    
    def save_current_game(self):
        """Save the current game state"""
        if not self.game.game_state:
            messagebox.showerror("Error", "No game to save")
            return
        
        try:
            game_id = self.game.save_game(self.loader)
            messagebox.showinfo("💾 Game Saved", f"Game saved as {game_id}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save game: {str(e)}")
    
    def show_load_dialog(self):
        """Show dialog to load a saved game"""
        try:
            games = self.loader.list_saved_games()
            if not games:
                messagebox.showinfo("No Games", "No saved games found")
                return
            
            # Create selection dialog
            dialog = tk.Toplevel(self.root)
            dialog.title("Load Game")
            dialog.geometry("400x300")
            
            tk.Label(dialog, text="Select a game to load:", font=('Arial', 12)).pack(pady=10)
            
            listbox = tk.Listbox(dialog, height=10)
            listbox.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
            
            for game in games:
                listbox.insert(tk.END, f"{game['id']} - {game['created_at']}")
            
            def load_selected():
                selection = listbox.curselection()
                if selection:
                    game_id = games[selection[0]]['id']
                    try:
                        loaded_game = self.loader.load_game(game_id)
                        self.game = loaded_game
                        self.setup_mode = False
                        dialog.destroy()
                        self.update_ui_visibility()
                        self.draw_graph()
                        messagebox.showinfo("Success", f"Game {game_id} loaded")
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to load game: {str(e)}")
            
            tk.Button(dialog, text="Load", command=load_selected).pack(pady=5)
            tk.Button(dialog, text="Cancel", command=dialog.destroy).pack(pady=5)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to list games: {str(e)}")
    
    def show_game_history(self):
        """Show game history in a new window"""
        pass

    def skip_turn(self):
        """Handles a detective skipping their turn when they have no moves."""
        if self.game.game_state.turn == Player.COPS and self.current_cop_index < self.game.num_cops:
            cop_pos = self.game.game_state.cop_positions[self.current_cop_index]
            
            # Verify there are no moves
            if cop_pos in self.current_player_moves and not self.current_player_moves[cop_pos]:
                self.cop_selections.append((cop_pos, None)) # Append a "stay" move
                self.current_cop_index += 1
                
                if len(self.cop_selections) == self.game.num_cops:
                    self.game_controls.move_button.config(state=tk.NORMAL)
                
                self.draw_graph()
            else:
                messagebox.showwarning("Invalid Action", "Skip turn is only for players with no available moves.")
        
        self.skip_button.config(state=tk.DISABLED)

    def make_manual_move(self):
        """Make a manual move by sending selected moves to the game object."""
        if (self.game.game_state.turn == Player.COPS and len(self.cop_selections) != self.game.num_cops) or \
           (self.game.game_state.turn == Player.ROBBER and not self.mr_x_selections and not self.selected_positions):
            messagebox.showwarning("Invalid Selection", "A move must be selected for all players.")
            return
    
        try:
            is_scotland_yard = isinstance(self.game, ScotlandYardGame)
            success = False
    
            if self.game.game_state.turn == Player.COPS:
                if is_scotland_yard:
                    success = self.game.make_move(detective_moves=self.cop_selections)
                else:
                    success = self.game.make_move(new_positions=self.cop_selections)
            else:  # Robber's turn
                if is_scotland_yard:
                    success = self.game.make_move(mr_x_moves=self.mr_x_selections)
                else:
                    success = self.game.make_move(new_robber_pos=self.selected_positions[0])
    
            if not success:
                messagebox.showerror("Invalid Move", "The move was rejected by the game engine. This may be due to an unexpected state change.")
    
            # Reset UI state after move attempt
            self.selected_positions = []
            self.cop_selections = []
            self.mr_x_selections = []
            self.current_cop_index = 0
            self.selected_nodes = []
            self.game_controls.move_button.config(state=tk.DISABLED)
            self.game.game_state.double_move_active = False
            self.game_controls.use_black_ticket.set(False)
            self.draw_graph()
    
            # Check for game over
            if self.game.is_game_over():
                winner = self.game.get_winner()
                winner_name = winner.value.title() if winner else "No one"
                messagebox.showinfo("Game Over", f"{winner_name} wins!")
                self.game_controls.auto_button.config(state=tk.DISABLED)
    
        except Exception as e:
            messagebox.showerror("Move Error", f"An error occurred while making the move: {str(e)}")
            # Reset UI state on error
            self.selected_positions = []
            self.cop_selections = []
            self.mr_x_selections = []
            self.current_cop_index = 0
            self.selected_nodes = []
            self.game_controls.move_button.config(state=tk.DISABLED)
            self.game.game_state.double_move_active = False
            self.game_controls.use_black_ticket.set(False)
            self.draw_graph()

    def toggle_double_move(self):
        """Toggles the double move state for Mr. X."""
        self.game.game_state.double_move_active = not self.game.game_state.double_move_active
        if self.game.game_state.double_move_active:
            self.game_controls.double_move_button.config(relief=tk.SUNKEN)
        else:
            self.game_controls.double_move_button.config(relief=tk.RAISED)
            # Reset selections if double move is cancelled
            self.game_controls.mr_x_selections = []
            self.selected_nodes = []
        self.draw_graph()

    def update_mrx_controls(self):
        """Updates the state of Mr. X's special move controls."""
        if self.game.game_state and self.game.game_state.turn == Player.ROBBER:
            mr_x_tickets = self.game.get_mr_x_tickets()
            
            if mr_x_tickets.get(TicketType.DOUBLE_MOVE, 0) > 0:
                self.game_controls.double_move_button.config(state=tk.NORMAL)
            else:
                self.game_controls.double_move_button.config(state=tk.DISABLED)
                self.game.game_state.double_move_active = False
                self.game_controls.double_move_button.config(relief=tk.RAISED)

            if mr_x_tickets.get(TicketType.BLACK, 0) > 0:
                self.game_controls.black_ticket_check.config(state=tk.NORMAL)
            else:
                self.game_controls.black_ticket_check.config(state=tk.DISABLED)
                self.game_controls.use_black_ticket.set(False)
        else:
            self.game_controls.double_move_button.config(state=tk.DISABLED)
            self.game_controls.black_ticket_check.config(state=tk.DISABLED)
            self.game.game_state.double_move_active = False
            self.game_controls.double_move_button.config(relief=tk.RAISED)
            self.game_controls.use_black_ticket.set(False)

    def run(self):
        """Start the GUI application"""
        self.root.mainloop()
    
    
    
    def update_available_moves(self):
        """Update available moves for the current player by querying the game object."""
        self.current_player_moves = {}
        self.highlighted_edges = []
        self.active_player_positions = []
    
        if not self.game.game_state or self.game.is_game_over():
            return
    
        is_scotland_yard = isinstance(self.game, ScotlandYardGame)
        current_player = self.game.game_state.turn
    
        if current_player == Player.COPS:
            self.game.game_state.double_move_active = False # Reset on cops' turn
            self.game_controls.mr_x_selections = []
            if self.current_cop_index < self.game.num_cops:
                cop_pos = self.game.game_state.cop_positions[self.current_cop_index]
                self.active_player_positions = [cop_pos]
                
                # The get_valid_moves method in ScotlandYardGame already filters by tickets and occupied positions.
                valid_moves = self.game.get_valid_moves(Player.COPS, cop_pos, pending_moves=self.cop_selections)
                
                self.current_player_moves[cop_pos] = {}
                for move in valid_moves:
                    if is_scotland_yard:
                        dest, transport = move
                        if dest not in self.current_player_moves[cop_pos]:
                            self.current_player_moves[cop_pos][dest] = []
                        self.current_player_moves[cop_pos][dest].append(transport.value)
                        self.highlighted_edges.append((cop_pos, dest, transport.value))
                    else: # Standard Game
                        dest = move
                        self.current_player_moves[cop_pos][dest] = [1] # Generic transport
                        self.highlighted_edges.append((cop_pos, dest, 1))

        else:  # Robber's turn
            if self.game.game_state.double_move_active and self.game_controls.mr_x_selections:
                # Second move of a double move starts from the first move's destination
                robber_pos = self.game_controls.mr_x_selections[0][0]
            else:
                robber_pos = self.game.game_state.robber_position
            self.active_player_positions = [robber_pos]
            
            valid_moves = self.game.get_valid_moves(Player.ROBBER, robber_pos)
            
            self.current_player_moves[robber_pos] = {}
            for move in valid_moves:
                if is_scotland_yard:
                    dest, transport = move
                    # Mr. X can use a specific ticket or a black ticket.
                    # The UI needs to know which options are available for a given route.
                    mr_x_tickets = self.game.get_mr_x_tickets()
                    transports_for_move = []
                    
                    required_ticket = TicketType[transport.name]
                    if mr_x_tickets.get(required_ticket, 0) > 0:
                        transports_for_move.append(transport.value)
                    if mr_x_tickets.get(TicketType.BLACK, 0) > 0:
                        transports_for_move.append(TransportType.BLACK.value)
                    
                    if transports_for_move:
                        self.current_player_moves[robber_pos][dest] = transports_for_move
                        for t_val in transports_for_move:
                            self.highlighted_edges.append((robber_pos, dest, t_val))
                else: # Standard Game
                    dest = move
                    self.current_player_moves[robber_pos][dest] = [1]
                    self.highlighted_edges.append((robber_pos, dest, 1))

    def handle_game_click(self, node):
        """Handle a node click during the game by checking against available moves."""
        if not self.game.game_state or self.game.is_game_over() or not self.active_player_positions:
            return
    
        source_pos = self.active_player_positions[0]
        
        # Check if the clicked node is a valid destination from the active player's position.
        if source_pos not in self.current_player_moves or node not in self.current_player_moves[source_pos]:
            messagebox.showwarning("Invalid Move", f"Position {node} is not a valid move for the current player.")
            return
    
        is_scotland_yard = isinstance(self.game, ScotlandYardGame)
        current_player = self.game.game_state.turn
    
        if current_player == Player.COPS:
            if is_scotland_yard:
                # For detectives, there's only one transport type per edge.
                transport_value = self.current_player_moves[source_pos][node][0]
                transport = TransportType(transport_value)
                self.cop_selections.append((node, transport))
            else:
                self.cop_selections.append(node)
            
            self.selected_nodes.append(node)
            self.current_cop_index += 1
    
            if len(self.cop_selections) == self.game.num_cops:
                self.game_controls.move_button.config(state=tk.NORMAL)
        
        else:  # Robber's turn
            if is_scotland_yard:
                # Determine transport type for the move
                if self.game_controls.use_black_ticket.get():
                    transport = TransportType.BLACK
                else:
                    # Prefer specific ticket over black if not explicitly chosen
                    available_transports = self.current_player_moves[source_pos][node]
                    transport_value = min(available_transports) # Prefers non-black
                    transport = TransportType(transport_value)

                if self.game.game_state.double_move_active:
                    self.game_controls.mr_x_selections.append((node, transport))
                    self.selected_nodes.append(node)
                    if len(self.game_controls.mr_x_selections) == 2:
                        self.game_controls.move_button.config(state=tk.NORMAL)
                else:
                    self.game_controls.mr_x_selections = [(node, transport)]
                    self.selected_nodes = [node]
                    self.game_controls.move_button.config(state=tk.NORMAL)
            else: # Standard game
                self.selected_positions = [node]
                self.selected_nodes = [node]
                self.game_controls.move_button.config(state=tk.NORMAL)
    
        self.draw_graph()

    def skip_turn(self):
        """Handles a detective skipping their turn when they have no moves."""
        if self.game.game_state.turn == Player.COPS and self.current_cop_index < self.game.num_cops:
            cop_pos = self.game.game_state.cop_positions[self.current_cop_index]
            
            # Verify there are no moves
            if cop_pos in self.current_player_moves and not self.current_player_moves[cop_pos]:
                self.cop_selections.append((cop_pos, None)) # Append a "stay" move
                self.current_cop_index += 1
                
                if len(self.cop_selections) == self.game.num_cops:
                    self.game_controls.move_button.config(state=tk.NORMAL)
                
                self.draw_graph()
            else:
                messagebox.showwarning("Invalid Action", "Skip turn is only for players with no available moves.")
        
        self.skip_button.config(state=tk.DISABLED)

    def make_manual_move(self):
        """Make a manual move by sending selected moves to the game object."""
        if (self.game.game_state.turn == Player.COPS and len(self.cop_selections) != self.game.num_cops) or \
           (self.game.game_state.turn == Player.ROBBER and not self.mr_x_selections and not self.selected_positions):
            messagebox.showwarning("Invalid Selection", "A move must be selected for all players.")
            return
    
        try:
            is_scotland_yard = isinstance(self.game, ScotlandYardGame)
            success = False
    
            if self.game.game_state.turn == Player.COPS:
                if is_scotland_yard:
                    success = self.game.make_move(detective_moves=self.cop_selections)
                else:
                    success = self.game.make_move(new_positions=self.cop_selections)
            else:  # Robber's turn
                if is_scotland_yard:
                    success = self.game.make_move(mr_x_moves=self.mr_x_selections)
                else:
                    success = self.game.make_move(new_robber_pos=self.selected_positions[0])
    
            if not success:
                messagebox.showerror("Invalid Move", "The move was rejected by the game engine. This may be due to an unexpected state change.")
    
            # Reset UI state after move attempt
            self.selected_positions = []
            self.cop_selections = []
            self.mr_x_selections = []
            self.current_cop_index = 0
            self.selected_nodes = []
            self.game_controls.move_button.config(state=tk.DISABLED)
            self.game.game_state.double_move_active = False
            self.game_controls.use_black_ticket.set(False)
            self.draw_graph()
    
            # Check for game over
            if self.game.is_game_over():
                winner = self.game.get_winner()
                winner_name = winner.value.title() if winner else "No one"
                messagebox.showinfo("Game Over", f"{winner_name} wins!")
                self.game_controls.auto_button.config(state=tk.DISABLED)
    
        except Exception as e:
            messagebox.showerror("Move Error", f"An error occurred while making the move: {str(e)}")
            # Reset UI state on error
            self.selected_positions = []
            self.cop_selections = []
            self.mr_x_selections = []
            self.current_cop_index = 0
            self.selected_nodes = []
            self.game_controls.move_button.config(state=tk.DISABLED)
            self.game.game_state.double_move_active = False
            self.game_controls.use_black_ticket.set(False)
            self.draw_graph()

    def toggle_double_move(self):
        """Toggles the double move state for Mr. X."""
        self.game.game_state.double_move_active = not self.game.game_state.double_move_active
        if self.game.game_state.double_move_active:
            self.game_controls.double_move_button.config(relief=tk.SUNKEN)
        else:
            self.game_controls.double_move_button.config(relief=tk.RAISED)
            # Reset selections if double move is cancelled
            self.game_controls.mr_x_selections = []
            self.selected_nodes = []
        self.draw_graph()

    def update_mrx_controls(self):
        """Updates the state of Mr. X's special move controls."""
        if self.game.game_state and self.game.game_state.turn == Player.ROBBER:
            mr_x_tickets = self.game.get_mr_x_tickets()
            
            if mr_x_tickets.get(TicketType.DOUBLE_MOVE, 0) > 0:
                self.game_controls.double_move_button.config(state=tk.NORMAL)
            else:
                self.game_controls.double_move_button.config(state=tk.DISABLED)
                self.game.game_state.double_move_active = False
                self.game_controls.double_move_button.config(relief=tk.RAISED)

            if mr_x_tickets.get(TicketType.BLACK, 0) > 0:
                self.game_controls.black_ticket_check.config(state=tk.NORMAL)
            else:
                self.game_controls.black_ticket_check.config(state=tk.DISABLED)
                self.game_controls.use_black_ticket.set(False)
        else:
            self.game_controls.double_move_button.config(state=tk.DISABLED)
            self.game_controls.black_ticket_check.config(state=tk.DISABLED)
            self.game.game_state.double_move_active = False
            self.game_controls.double_move_button.config(relief=tk.RAISED)
            self.game_controls.use_black_ticket.set(False)

