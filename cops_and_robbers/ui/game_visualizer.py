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

class GameVisualizer:
    """Interactive GUI for Cops and Robbers game"""
    
    def __init__(self, game: Game, loader: 'GameLoader' = None):
        self.loader = loader or GameLoader()
        self.game = game
        self.solver = MinimaxSolver(game)
        self.root = tk.Tk()
        self.root.title("Cops and Robbers Game")
        self.root.geometry(f"{self.root.winfo_screenwidth()}x{self.root.winfo_screenheight()}")
        
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
        self.current_player_moves = {}  # Maps positions to available moves
        self.highlighted_edges = []     # List of edges to highlight with transport types
        self.active_player_positions = []  # Positions of players that need to move
        self.current_cop_index = 0      # Which cop is currently being moved
        self.cop_selections = []        # Track cop selections during turn
        self.selected_nodes = []        # Nodes with visual selection feedback
        
        # UI components
        self.setup_ui()
        self.setup_graph_display()
        
    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel for controls
        self.control_frame = ttk.Frame(main_frame, width=350)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        self.control_frame.pack_propagate(False)
        
        # Game setup section
        setup_section = ttk.LabelFrame(self.control_frame, text="Game Setup")
        setup_section.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(setup_section, text="Select positions for setup").pack(pady=5)
        
        self.setup_button = ttk.Button(setup_section, text="Start Game", 
                                      command=self.start_game, state=tk.DISABLED)
        self.setup_button.pack(pady=5)
        
        self.reset_button = ttk.Button(setup_section, text="Reset Setup", 
                                      command=self.reset_setup)
        self.reset_button.pack(pady=5)
        
        # Game controls section
        controls_section = ttk.LabelFrame(self.control_frame, text="Game Controls")
        controls_section.pack(fill=tk.X, pady=(0, 10))
        
        self.move_button = ttk.Button(controls_section, text="Make Move", 
                                     command=self.make_manual_move, state=tk.DISABLED)
        self.move_button.pack(pady=5)
        
        self.auto_button = ttk.Button(controls_section, text="Auto Play", 
                                     command=self.toggle_auto_play, state=tk.DISABLED)
        self.auto_button.pack(pady=5)
        
        # Current turn section
        self.turn_section = ttk.LabelFrame(self.control_frame, text="Current Turn")
        self.turn_section.pack(fill=tk.X, pady=(0, 10))
        
        self.turn_text = tk.Text(self.turn_section, height=3, wrap=tk.WORD)
        self.turn_text.pack(fill=tk.BOTH, expand=True)
        
        # Available moves section
        self.moves_section = ttk.LabelFrame(self.control_frame, text="Available Moves")
        self.moves_section.pack(fill=tk.X, pady=(0, 10))
        
        self.moves_text = tk.Text(self.moves_section, height=6, wrap=tk.WORD)
        self.moves_text.pack(fill=tk.BOTH, expand=True)
        
        # Tickets section for Scotland Yard
        self.tickets_section = ttk.LabelFrame(self.control_frame, text="Tickets")
        self.tickets_section.pack(fill=tk.X, pady=(0, 10))
        
        self.tickets_text = tk.Text(self.tickets_section, height=6, wrap=tk.WORD)
        self.tickets_text.pack(fill=tk.BOTH, expand=True)
        
        # Solver section
        solver_section = ttk.LabelFrame(self.control_frame, text="Solver")
        solver_section.pack(fill=tk.X, pady=(0, 10))
        
        self.solve_button = ttk.Button(solver_section, text="Solve Game", 
                                      command=self.solve_game, state=tk.DISABLED)
        self.solve_button.pack(pady=5)
        
        self.strategy_button = ttk.Button(solver_section, text="Show Strategy", 
                                         command=self.show_strategy, state=tk.DISABLED)
        self.strategy_button.pack(pady=5)
        
        # Game info section
        info_section = ttk.LabelFrame(self.control_frame, text="Game Information")
        info_section.pack(fill=tk.X, pady=(0, 10))
        
        # Add save/load UI
        self.add_save_load_ui()
        
        self.info_text = tk.Text(info_section, height=8, wrap=tk.WORD)
        self.info_text.pack(fill=tk.BOTH, expand=True)
        
        # Right panel for graph
        self.graph_frame = ttk.Frame(main_frame)
        self.graph_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
    def setup_graph_display(self):
        """Setup matplotlib graph display"""
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        self.canvas = FigureCanvasTkAgg(self.fig, self.graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Mouse click handler - simplified node detection
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
                        alpha=0.15  # Very faded for non-available moves
                    )
                
                # Draw highlighted edges with full color and increased thickness
                if highlighted_edges_for_type:
                    nx.draw_networkx_edges(
                        self.game.graph, self.pos,
                        edgelist=highlighted_edges_for_type,
                        ax=self.ax,
                        edge_color=style['color'],
                        width=style['width'] + 2,  # Thicker for highlighted
                        alpha=0.9  # Nearly full opacity
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
                        node_colors.append('blue')  # Selected cop position
                    else:
                        node_colors.append('red')   # Selected robber position
                    node_sizes.append(600)
                else:
                    node_colors.append('lightgray')
                    node_sizes.append(300)
            else:
                # Game mode
                if self.game.game_state:
                    if node in self.active_player_positions:
                        # Highlight active player(s)
                        if node in self.game.game_state.cop_positions:
                            node_colors.append('cyan')  # Active detective
                        else:
                            node_colors.append('orange')  # Active Mr. X/robber
                        node_sizes.append(700)
                    elif node in self.cop_selections:
                        # Show already selected cop positions during turn
                        node_colors.append('purple')  # Selected cop destination
                        node_sizes.append(600)
                    elif node in self.game.game_state.cop_positions:
                        node_colors.append('blue')
                        node_sizes.append(500)
                    elif node == self.game.game_state.robber_position:
                        # Show Mr. X only if visible
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
                circle = plt.Circle((x, y), 0.08, fill=False, color='black', 
                                  linewidth=2, linestyle='--', alpha=0.8)
                self.ax.add_patch(circle)
        
        # Draw labels
        nx.draw_networkx_labels(self.game.graph, self.pos, ax=self.ax, font_size=8)
        
        self.ax.set_title("Cops and Robbers Game")
        
        # Only show legend if we have legend handles
        if legend_handles:
            self.ax.legend(handles=legend_handles, loc='upper right')
        
        self.ax.axis('off')
        self.canvas.draw()
        
        self.update_info()
    
    def on_graph_click(self, event):
        """Handle mouse clicks on graph - simplified node detection"""
        if event.inaxes != self.ax:
            return
        
        click_pos = (event.xdata, event.ydata)
        if click_pos[0] is None or click_pos[1] is None:
            return
        
        # Find closest node by checking all nodes
        closest_node = None
        min_dist = float('inf')
        threshold = 0.15  # Adjust based on graph size
        
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
                self.setup_button.config(state=tk.NORMAL)
            else:
                self.setup_button.config(state=tk.DISABLED)
        else:
            self.setup_button.config(state=tk.DISABLED)
        
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
        self.setup_button.config(state=tk.DISABLED)
        self.move_button.config(state=tk.DISABLED)
        self.auto_button.config(state=tk.NORMAL)
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
        
        self.setup_button.config(state=tk.DISABLED)
        self.move_button.config(state=tk.DISABLED)
        self.auto_button.config(state=tk.NORMAL, text="Auto Play")
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
            self.auto_button.config(text="Stop Auto")
            self.root.after(1000, self.auto_play_step)
        else:
            self.auto_button.config(text="Auto Play")
    
    def auto_play_step(self):
        """Execute one step of automatic play"""
        if not self.auto_play or self.game.is_game_over():
            self.auto_play = False
            self.auto_button.config(text="Auto Play")
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
            messagebox.showinfo("Game Over", f"{winner.value.title()} wins!")
            self.auto_play = False
            self.auto_button.config(text="Auto Play")
             
            # Save the game automatically when it ends
            try:
                game_id = self.game.save_game(self.loader)
                messagebox.showinfo("Game Saved", f"Game saved as {game_id}")
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
            self.info_text.insert(tk.END, "Solving game...\n")
            self.root.update()
            
            self.solver_result = self.solver.solve(
                self.game.game_history[0].cop_positions,
                self.game.game_history[0].robber_position
            )
            
            result_text = f"Solver Result:\n"
            result_text += f"Cops can win: {self.solver_result.cops_can_win}\n"
            if self.solver_result.game_length:
                result_text += f"Game length: {self.solver_result.game_length}\n"
            result_text += "\n"
            
            self.info_text.insert(tk.END, result_text)
            self.strategy_button.config(state=tk.NORMAL)
            
        except Exception as e:
            messagebox.showerror("Error", f"Solver failed: {str(e)}")
    
    def show_strategy(self):
        """Show optimal strategy"""
        if not self.solver_result:
            messagebox.showerror("Error", "No solver result available")
            return
        
        strategy_window = tk.Toplevel(self.root)
        strategy_window.title("Game Strategy")
        strategy_window.geometry(f"{self.root.winfo_screenwidth()}x{self.root.winfo_screenheight()}")
        
        text_widget = tk.Text(strategy_window, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(strategy_window, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        strategy_text = f"Optimal Strategy:\n\n"
        strategy_text += f"Cops can win: {self.solver_result.cops_can_win}\n\n"
        
        if self.solver_result.cops_can_win and self.solver_result.cop_strategy:
            strategy_text += "Cop Strategy:\n"
            for state_key, move in self.solver_result.cop_strategy.moves.items():
                strategy_text += f"State {state_key}: Move to {move}\n"
        
        if not self.solver_result.cops_can_win and self.solver_result.robber_strategy:
            strategy_text += "Robber Strategy:\n"
            for state_key, move in self.solver_result.robber_strategy.moves.items():
                strategy_text += f"State {state_key}: Move to {move}\n"
        
        text_widget.insert(tk.END, strategy_text)
        text_widget.config(state=tk.DISABLED)
    
    def update_info(self):
        """Update game information display"""
        self.info_text.delete(1.0, tk.END)
        
        if self.setup_mode:
            self.info_text.insert(tk.END, "Setup Mode\n")
            self.info_text.insert(tk.END, f"Need: {self.game.num_cops} cops + 1 robber\n")
            self.info_text.insert(tk.END, f"Selected: {len(self.selected_positions)}\n")
        elif self.game.game_state:
            state_info = self.game.get_state_representation()
            self.info_text.insert(tk.END, f"Turn: {state_info['turn'].title()}\n")
            self.info_text.insert(tk.END, f"Turn count: {state_info['turn_count']}\n")
            self.info_text.insert(tk.END, f"Cop positions: {state_info['cop_positions']}\n")
            
            # Show robber position based on game type
            is_scotland_yard = isinstance(self.game, ScotlandYardGame)
            if is_scotland_yard:
                if self.game.game_state.mr_x_visible:
                    self.info_text.insert(tk.END, f"Mr. X position: {state_info['robber_position']} (VISIBLE)\n")
                else:
                    self.info_text.insert(tk.END, f"Mr. X position: HIDDEN\n")
            else:
                self.info_text.insert(tk.END, f"Robber position: {state_info['robber_position']}\n")
            
            if self.game.is_game_over():
                winner = self.game.get_winner()
                if winner:
                    self.info_text.insert(tk.END, f"Game Over - Winner: {winner.value.title()}\n")
            else:
                self.info_text.insert(tk.END, "Game in progress\n")
            
            if self.solver_result:
                self.info_text.insert(tk.END, f"\nSolver: Cops can win = {self.solver_result.cops_can_win}\n")
        
        # Update other displays
        self.update_turn_display()
        self.update_moves_display()
        self.update_tickets_display()
    
    def update_tickets_display(self):
        """Update the tickets display for Scotland Yard games"""
        self.tickets_text.delete(1.0, tk.END)
        
        if not self.game.game_state:
            return
        
        is_scotland_yard = isinstance(self.game, ScotlandYardGame)
        if not is_scotland_yard:
            self.tickets_text.insert(tk.END, "Not a Scotland Yard game")
            return
        
        # Show detective tickets
        self.tickets_text.insert(tk.END, "DETECTIVE TICKETS:\n")
        for i in range(self.game.num_cops):
            tickets = self.game.get_detective_tickets(i)
            pos = self.game.game_state.cop_positions[i]
            self.tickets_text.insert(tk.END, f"Det. {i+1} (pos {pos}):\n")
            for ticket_type, count in tickets.items():
                self.tickets_text.insert(tk.END, f"  {ticket_type.value}: {count}\n")
        
        # Show Mr. X tickets
        mr_x_tickets = self.game.get_mr_x_tickets()
        self.tickets_text.insert(tk.END, "\nMR. X TICKETS:\n")
        for ticket_type, count in mr_x_tickets.items():
            self.tickets_text.insert(tk.END, f"  {ticket_type.value}: {count}\n")
    
    def add_save_load_ui(self):
        """Add save/load section to control panel"""
        # Add to setup_ui method in GameVisualizer
        
        # Save/Load section
        save_load_section = ttk.LabelFrame(self.control_frame, text="Save/Load")
        save_load_section.pack(fill=tk.X, pady=(0, 10))
        
        self.save_button = ttk.Button(save_load_section, text="Save Game", 
                                     command=self.save_current_game, state=tk.DISABLED)
        self.save_button.pack(pady=2)
        
        self.load_button = ttk.Button(save_load_section, text="Load Game", 
                                     command=self.show_load_dialog)
        self.load_button.pack(pady=2)
        
        self.history_button = ttk.Button(save_load_section, text="Game History", 
                                        command=self.show_game_history)
        self.history_button.pack(pady=2)
    
    def save_current_game(self):
        """Save the current game"""
        if not self.game.game_state:
            messagebox.showerror("Error", "No game to save")
            return
        
        try:
            game_id = self.game.save_game(self.loader)
            messagebox.showinfo("Success", f"Game saved as {game_id}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save game: {str(e)}")
    
    def show_load_dialog(self):
        """Show dialog to load saved games"""
        load_window = tk.Toplevel(self.root)
        load_window.title("Load Game")
        load_window.geometry(f"{self.root.winfo_screenwidth()}x{self.root.winfo_screenheight()}")
        
        # List of saved games
        games = self.loader.list_games()
        
        columns = ('ID', 'Date', 'Graph Type', 'Winner', 'Turns')
        tree = ttk.Treeview(load_window, columns=columns, show='headings')
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=100)
        
        for game in games:
            tree.insert('', tk.END, values=(
                game['game_id'],
                game['created_at'][:10],
                game['graph_type'],
                game.get('winner', 'Ongoing'),
                game.get('total_turns', 0)
            ))
        
        tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        def load_selected():
            selection = tree.selection()
            if selection:
                game_id = tree.item(selection[0])['values'][0]
                self.load_game(game_id)
                load_window.destroy()
        
        ttk.Button(load_window, text="Load Selected", 
                  command=load_selected).pack(pady=10)
    
    def load_game(self, game_id: str):
        """Load a specific game"""
        try:
            loaded_game = self.loader.load_game(game_id)
            if loaded_game:
                self.game = loaded_game
                self.setup_mode = False
                self.selected_positions = []
                self.draw_graph()
                messagebox.showinfo("Success", f"Game {game_id} loaded")
            else:
                messagebox.showerror("Error", f"Failed to load game {game_id}")
        except Exception as e:
            messagebox.showerror("Error", f"Error loading game: {str(e)}")
    
    def show_game_history(self):
        """Show game history and statistics"""
        history_window = tk.Toplevel(self.root)
        history_window.title("Game History & Statistics")
        history_window.geometry(f"{self.root.winfo_screenwidth()}x{self.root.winfo_screenheight()}")
        
        notebook = ttk.Notebook(history_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Games list tab
        games_frame = ttk.Frame(notebook)
        notebook.add(games_frame, text="Saved Games")
        
        games = self.loader.list_games()
        games_text = tk.Text(games_frame, wrap=tk.WORD)
        games_scrollbar = ttk.Scrollbar(games_frame, orient=tk.VERTICAL, 
                                       command=games_text.yview)
        games_text.configure(yscrollcommand=games_scrollbar.set)
        
        for game in games:
            games_text.insert(tk.END, f"ID: {game['game_id']}\n")
            games_text.insert(tk.END, f"Date: {game['created_at']}\n")
            games_text.insert(tk.END, f"Type: {game['graph_type']}\n")
            games_text.insert(tk.END, f"Winner: {game.get('winner', 'Ongoing')}\n")
            games_text.insert(tk.END, f"Turns: {game.get('total_turns', 0)}\n")
            games_text.insert(tk.END, "-" * 40 + "\n")
        
        games_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        games_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Statistics tab
        stats_frame = ttk.Frame(notebook)
        notebook.add(stats_frame, text="Statistics")
        
        stats = self.loader.generate_statistics()
        stats_text = tk.Text(stats_frame, wrap=tk.WORD)
        stats_text.insert(tk.END, f"Total Games: {stats.get('total_games', 0)}\n")
        stats_text.insert(tk.END, f"Completed Games: {stats.get('completed_games', 0)}\n")
        stats_text.insert(tk.END, f"Cops Wins: {stats.get('cops_wins', 0)}\n")
        stats_text.insert(tk.END, f"Robber Wins: {stats.get('robber_wins', 0)}\n")
        stats_text.insert(tk.END, f"Average Game Length: {stats.get('average_game_length', 0):.1f}\n")
        
        stats_text.pack(fill=tk.BOTH, expand=True)

    def update_turn_display(self):
        """Update current turn information"""
        self.turn_text.delete(1.0, tk.END)
        
        if self.setup_mode:
            self.turn_text.insert(tk.END, "Setup Phase - Click nodes to select positions")
            return
        
        if not self.game.game_state:
            return
        
        is_scotland_yard = isinstance(self.game, ScotlandYardGame)
        current_player = self.game.game_state.turn
        
        if current_player == Player.COPS:
            if is_scotland_yard:
                if self.current_cop_index < self.game.num_cops:
                    det_pos = self.game.game_state.cop_positions[self.current_cop_index]
                    self.turn_text.insert(tk.END, f"DETECTIVE {self.current_cop_index + 1}'S TURN\n")
                    self.turn_text.insert(tk.END, f"Moving from position {det_pos}\n")
                    self.turn_text.insert(tk.END, f"Progress: {len(self.cop_selections)}/{self.game.num_cops}")
                else:
                    self.turn_text.insert(tk.END, "All detectives selected - make move")
            else:
                if self.current_cop_index < self.game.num_cops:
                    cop_pos = self.game.game_state.cop_positions[self.current_cop_index]
                    self.turn_text.insert(tk.END, f"COP {self.current_cop_index + 1}'S TURN\n")
                    self.turn_text.insert(tk.END, f"Moving from position {cop_pos}\n")
                    self.turn_text.insert(tk.END, f"Progress: {len(self.cop_selections)}/{self.game.num_cops}")
                else:
                    self.turn_text.insert(tk.END, "All cops selected - make move")
        else:
            if is_scotland_yard:
                self.turn_text.insert(tk.END, "MR. X'S TURN\nSelect new position")
            else:
                self.turn_text.insert(tk.END, "ROBBER'S TURN\nSelect new position")
    
    def update_moves_display(self):
        """Update available moves display"""
        self.moves_text.delete(1.0, tk.END)
        
        if self.setup_mode or not self.game.game_state:
            return
        
        is_scotland_yard = isinstance(self.game, ScotlandYardGame)
        
        if not self.current_player_moves:
            self.moves_text.insert(tk.END, "No available moves")
            return
        
        # Show current cop's moves or robber/Mr. X moves
        if self.game.game_state.turn == Player.COPS and self.current_cop_index < self.game.num_cops:
            cop_pos = self.game.game_state.cop_positions[self.current_cop_index]
            if cop_pos in self.current_player_moves:
                moves = self.current_player_moves[cop_pos]
                player_name = f"Detective {self.current_cop_index + 1}" if is_scotland_yard else f"Cop {self.current_cop_index + 1}"
                self.moves_text.insert(tk.END, f"{player_name} from {cop_pos}:\n")
                for target_pos, transports in moves.items():
                    if is_scotland_yard:
                        transport_names = []
                        for t in transports:
                            if t == 1: transport_names.append("Taxi")
                            elif t == 2: transport_names.append("Bus") 
                            elif t == 3: transport_names.append("Underground")
                            elif t == 4: transport_names.append("Black")
                        self.moves_text.insert(tk.END, f"  → {target_pos} ({', '.join(transport_names)})\n")
                    else:
                        self.moves_text.insert(tk.END, f"  → {target_pos}\n")
        else:
            # Robber/Mr. X moves
            for source_pos, moves in self.current_player_moves.items():
                player_name = "Mr. X" if is_scotland_yard else "Robber"
                self.moves_text.insert(tk.END, f"{player_name} from {source_pos}:\n")
                for target_pos, transports in moves.items():
                    if is_scotland_yard:
                        transport_names = []
                        for t in transports:
                            if t == 1: transport_names.append("Taxi")
                            elif t == 2: transport_names.append("Bus") 
                            elif t == 3: transport_names.append("Underground")
                            elif t == 4: transport_names.append("Black")
                        self.moves_text.insert(tk.END, f"  → {target_pos} ({', '.join(transport_names)})\n")
                    else:
                        self.moves_text.insert(tk.END, f"  → {target_pos}\n")

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
            if self.current_cop_index < self.game.num_cops:
                cop_pos = self.game.game_state.cop_positions[self.current_cop_index]
                self.active_player_positions = [cop_pos]
                
                # The get_valid_moves method in ScotlandYardGame already filters by tickets and occupied positions.
                valid_moves = self.game.get_valid_moves(Player.COPS, cop_pos)
                
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
                self.move_button.config(state=tk.NORMAL)
        
        else:  # Robber's turn
            self.selected_nodes = [node]
            if is_scotland_yard:
                # For Mr. X, if multiple transport options exist, we must ask.
                # For simplicity, we'll prefer the specific ticket over the black ticket.
                available_transports = self.current_player_moves[source_pos][node]
                transport_value = min(available_transports) # Prefer Taxi > Bus > Underground > Black
                transport = TransportType(transport_value)
                self.selected_positions = [(node, transport)]
            else:
                self.selected_positions = [node]
            
            self.move_button.config(state=tk.NORMAL)
    
        self.draw_graph()

    def make_manual_move(self):
        """Make a manual move by sending selected moves to the game object."""
        if (self.game.game_state.turn == Player.COPS and len(self.cop_selections) != self.game.num_cops) or \
           (self.game.game_state.turn == Player.ROBBER and not self.selected_positions):
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
                    success = self.game.make_move(mr_x_move=self.selected_positions[0])
                else:
                    success = self.game.make_move(new_robber_pos=self.selected_positions[0])
    
            if not success:
                messagebox.showerror("Invalid Move", "The move was rejected by the game engine. This may be due to an unexpected state change.")
    
            # Reset UI state after move attempt
            self.selected_positions = []
            self.cop_selections = []
            self.current_cop_index = 0
            self.selected_nodes = []
            self.move_button.config(state=tk.DISABLED)
            self.draw_graph()
    
            # Check for game over
            if self.game.is_game_over():
                winner = self.game.get_winner()
                winner_name = winner.value.title() if winner else "No one"
                messagebox.showinfo("Game Over", f"{winner_name} wins!")
                self.auto_button.config(state=tk.DISABLED)
    
        except Exception as e:
            messagebox.showerror("Move Error", f"An error occurred while making the move: {str(e)}")
            # Reset UI state on error
            self.selected_positions = []
            self.cop_selections = []
            self.current_cop_index = 0
            self.selected_nodes = []
            self.move_button.config(state=tk.DISABLED)
            self.draw_graph()
    
    