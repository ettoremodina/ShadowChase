import tkinter as tk
from tkinter import ttk, messagebox
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from typing import Dict, List, Optional, Tuple
from ..core.game import Game, Player, GameState
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
            1: {'color': 'yellow', 'width': 2, 'name': 'Taxi'},      # Taxi - yellow, thin
            2: {'color': 'blue', 'width': 3, 'name': 'Bus'},        # Bus - blue, medium  
            3: {'color': 'red', 'width': 4, 'name': 'Underground'}, # Underground - red, thick
            4: {'color': 'green', 'width': 3, 'name': 'Ferry'}      # Ferry - green, medium
        }
        
        # Game state
        self.selected_positions = []
        self.setup_mode = True
        self.auto_play = False
        self.solver_result = None
        self.loader = loader or GameLoader()
        
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
        
        ttk.Label(setup_section, text="Select cop positions, then robber position").pack(pady=5)
        
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
        
        # Add tickets section for Scotland Yard
        self.tickets_section = ttk.LabelFrame(self.control_frame, text="Tickets")
        self.tickets_section.pack(fill=tk.X, pady=(0, 10))
        
        self.tickets_text = tk.Text(self.tickets_section, height=6, wrap=tk.WORD)
        self.tickets_text.pack(fill=tk.BOTH, expand=True)
        
        # Move selection help
        self.move_help_section = ttk.LabelFrame(self.control_frame, text="Move Instructions")
        self.move_help_section.pack(fill=tk.X, pady=(0, 10))
        
        self.move_help_text = tk.Text(self.move_help_section, height=4, wrap=tk.WORD)
        self.move_help_text.pack(fill=tk.BOTH, expand=True)
        
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
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        self.canvas = FigureCanvasTkAgg(self.fig, self.graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Mouse click handler
        self.canvas.mpl_connect('button_press_event', self.on_graph_click)
        
        # Calculate graph layout
        self.pos = nx.spring_layout(self.game.graph, seed=42)
        self.draw_graph()
        
    def draw_graph(self):
        """Draw the game graph"""
        self.ax.clear()
        
        # Track which transport types are actually used
        legend_handles = []
        
        # Draw edges by transport type
        for transport_type, style in self.transport_styles.items():
            # Get edges for this transport type
            edges_for_type = []
            for u, v, data in self.game.graph.edges(data=True):
                # Handle both 'edge_type' and 'transports' attributes
                edge_transports = data.get('transports', [])
                edge_type = data.get('edge_type', None)
                
                if transport_type in edge_transports or edge_type == transport_type:
                    edges_for_type.append((u, v))
            
            if edges_for_type:
                edge_collection = nx.draw_networkx_edges(
                    self.game.graph, self.pos, 
                    edgelist=edges_for_type,
                    ax=self.ax,
                    edge_color=style['color'],
                    width=style['width'],
                    alpha=0.7
                )
                # Create a custom legend entry
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
                    node_sizes.append(500)
                else:
                    node_colors.append('lightgray')
                    node_sizes.append(300)
            else:
                # Game mode
                if self.game.game_state:
                    if node in self.game.game_state.cop_positions:
                        node_colors.append('blue')
                        node_sizes.append(500)
                    elif node == self.game.game_state.robber_position:
                        node_colors.append('red')
                        node_sizes.append(500)
                    else:
                        node_colors.append('lightgray')
                        node_sizes.append(300)
                else:
                    node_colors.append('lightgray')
                    node_sizes.append(300)
        
        # Draw nodes
        nx.draw_networkx_nodes(self.game.graph, self.pos, ax=self.ax,
                              node_color=node_colors, node_size=node_sizes)
        
        # Draw labels
        nx.draw_networkx_labels(self.game.graph, self.pos, ax=self.ax)
        
        self.ax.set_title("Cops and Robbers Game")
        
        # Only show legend if we have legend handles
        if legend_handles:
            self.ax.legend(handles=legend_handles, loc='upper right')
        
        self.ax.axis('off')
        self.canvas.draw()
        
        self.update_info()
    
    def on_graph_click(self, event):
        """Handle mouse clicks on graph"""
        if event.inaxes != self.ax:
            return
        
        # Find closest node
        click_pos = (event.xdata, event.ydata)
        if click_pos[0] is None or click_pos[1] is None:
            return
        
        min_dist = float('inf')
        closest_node = None
        
        for node, pos in self.pos.items():
            dist = np.sqrt((pos[0] - click_pos[0])**2 + (pos[1] - click_pos[1])**2)
            if dist < min_dist:
                min_dist = dist
                closest_node = node
        
        if min_dist < 0.1:  # Threshold for node selection
            if self.setup_mode:
                self.handle_setup_click(closest_node)
            else:
                self.handle_game_click(closest_node)
    
    def handle_setup_click(self, node):
        """Handle node clicks during setup"""
        if node in self.selected_positions:
            self.selected_positions.remove(node)
        else:
            # Prevent selecting the same position for multiple entities
            if len(self.selected_positions) < self.game.num_cops + 1:
                self.selected_positions.append(node)
        
        # Enable start button when we have enough positions
        if len(self.selected_positions) == self.game.num_cops + 1:
            # Check for conflicts before enabling button
            cop_positions = self.selected_positions[:self.game.num_cops]
            robber_position = self.selected_positions[self.game.num_cops]
            
            if robber_position not in cop_positions:
                self.setup_button.config(state=tk.NORMAL)
            else:
                self.setup_button.config(state=tk.DISABLED)
        else:
            self.setup_button.config(state=tk.DISABLED)
        
        self.draw_graph()
    
    def handle_game_click(self, node):
        """Handle node clicks during game"""
        if not self.game.game_state or self.game.is_game_over():
            return
        
        # Check if this is a Scotland Yard game
        is_scotland_yard = hasattr(self.game, 'is_scotland_yard') and self.game.is_scotland_yard
        
        if self.game.game_state.turn == Player.COPS:
            # Select positions for cops/detectives
            if node in self.selected_positions:
                self.selected_positions.remove(node)
            else:
                # For Scotland Yard, validate the move more thoroughly
                if is_scotland_yard:
                    try:
                        # Check if this is a valid move for any detective
                        valid_move = False
                        available_detective = None
                        
                        for i, cop_pos in enumerate(self.game.game_state.cop_positions):
                            if cop_pos not in self.selected_positions:  # Detective not already moved
                                try:
                                    valid_moves = self.game.get_valid_moves_with_tickets(Player.COPS, i)
                                    if node in valid_moves:
                                        valid_move = True
                                        available_detective = i + 1
                                        break
                                except Exception:
                                    continue
                        
                        if not valid_move:
                            messagebox.showwarning("Invalid Move", 
                                                 f"No detective can move to position {node}.\n"
                                                 f"Check: tickets available, position not occupied, "
                                                 f"detective not already moved this turn.")
                            return
                    except Exception as e:
                        messagebox.showerror("Move Validation Error", 
                                           f"Error checking move validity: {str(e)}")
                        return
                
                self.selected_positions.append(node)
            
            if len(self.selected_positions) == self.game.num_cops:
                self.move_button.config(state=tk.NORMAL)
            else:
                self.move_button.config(state=tk.DISABLED)
        
        else:  # Robber's/Mr. X's turn
            if is_scotland_yard:
                try:
                    valid_moves = self.game.get_valid_moves_with_tickets(Player.MR_X)
                    if node not in valid_moves:
                        messagebox.showwarning("Invalid Move", 
                                             f"Mr. X cannot move to position {node}.\n"
                                             f"Check: tickets available, position not occupied by detective.")
                        return
                except Exception as e:
                    messagebox.showerror("Move Validation Error", 
                                       f"Error checking Mr. X move: {str(e)}")
                    return
            
            self.selected_positions = [node]
            self.move_button.config(state=tk.NORMAL)
        
        self.draw_graph()
    
    def start_game(self):
        """Start the game with selected positions"""
        if len(self.selected_positions) != self.game.num_cops + 1:
            messagebox.showerror("Error", f"Select {self.game.num_cops} cop positions and 1 robber position")
            return
        
        cop_positions = self.selected_positions[:self.game.num_cops]
        robber_position = self.selected_positions[self.game.num_cops]
        
        # Check for position conflicts
        if robber_position in cop_positions:
            messagebox.showerror("Error", "Robber and cops cannot start in the same position")
            return
        
        # Reset any existing game state before initializing
        self.game.game_state = None
        self.game.game_history = []
        
        try:
            self.game.initialize_game(cop_positions, robber_position)
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            return
        
        self.setup_mode = False
        self.selected_positions = []
        
        # Update UI - enable save button now that game is initialized
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
    
   
    def make_manual_move(self):
        """Make a manual move with comprehensive error handling"""
        if not self.selected_positions:
            messagebox.showwarning("No Selection", "Please select position(s) first.")
            return
        
        try:
            if self.game.game_state.turn == Player.COPS:
                if len(self.selected_positions) != self.game.num_cops:
                    messagebox.showerror("Invalid Selection", 
                                       f"Select {self.game.num_cops} positions for cops/detectives.\n"
                                       f"Currently selected: {len(self.selected_positions)}")
                    return
                
                # Validate move before attempting
                is_scotland_yard = hasattr(self.game, 'is_scotland_yard') and self.game.is_scotland_yard
                if is_scotland_yard:
                    # Check for duplicate positions
                    if len(set(self.selected_positions)) != len(self.selected_positions):
                        messagebox.showerror("Invalid Move", 
                                           "Detectives cannot occupy the same position.")
                        self.selected_positions = []
                        self.move_button.config(state=tk.DISABLED)
                        self.draw_graph()
                        return
                    
                    # Check if Mr. X position would be occupied
                    mr_x_pos = self.game.game_state.robber_position
                    if mr_x_pos in self.selected_positions:
                        messagebox.showerror("Invalid Move", 
                                           f"Detective cannot move to position {mr_x_pos} "
                                           f"(occupied by Mr. X).")
                        self.selected_positions = []
                        self.move_button.config(state=tk.DISABLED)
                        self.draw_graph()
                        return
                
                success = self.game.make_move(new_positions=self.selected_positions)
                
            else:  # Robber/Mr. X turn
                if len(self.selected_positions) != 1:
                    messagebox.showerror("Invalid Selection", 
                                       "Select 1 position for robber/Mr. X.")
                    return
                
                # Validate move before attempting
                is_scotland_yard = hasattr(self.game, 'is_scotland_yard') and self.game.is_scotland_yard
                if is_scotland_yard:
                    # Check if position is occupied by detective
                    if self.selected_positions[0] in self.game.game_state.cop_positions:
                        messagebox.showerror("Invalid Move", 
                                           f"Mr. X cannot move to position {self.selected_positions[0]} "
                                           f"(occupied by detective).")
                        self.selected_positions = []
                        self.move_button.config(state=tk.DISABLED)
                        self.draw_graph()
                        return
                
                success = self.game.make_move(new_robber_pos=self.selected_positions[0])
            
            if not success:
                # Generic move failure - could be due to game rules, connectivity, etc.
                player_name = "detectives" if self.game.game_state.turn == Player.COPS else "Mr. X" if hasattr(self.game, 'is_scotland_yard') else "robber"
                messagebox.showerror("Invalid Move", 
                                   f"Move not allowed for {player_name}.\n"
                                   f"Check: valid connections, sufficient tickets (if applicable), "
                                   f"position availability.")
                # Don't clear selections, let player try again
                return
            
            # Move successful - clear selections and update UI
            self.selected_positions = []
            self.move_button.config(state=tk.DISABLED)
            self.draw_graph()
            
            # Check for game over
            if self.game.is_game_over():
                winner = self.game.get_winner()
                winner_name = winner.value.title() if winner else "Unknown"
                messagebox.showinfo("Game Over", f"{winner_name} wins!")
                self.auto_play = False
                self.auto_button.config(text="Auto Play")
        
        except ValueError as e:
            # Specific game rule violations
            messagebox.showerror("Move Error", f"Invalid move: {str(e)}")
            # Don't clear selections, let player try again
            
        except AttributeError as e:
            # Missing method or attribute errors
            messagebox.showerror("Game Error", f"Game state error: {str(e)}")
            self.selected_positions = []
            self.move_button.config(state=tk.DISABLED)
            self.draw_graph()
            
        except Exception as e:
            # Any other unexpected errors
            messagebox.showerror("Unexpected Error", 
                               f"An unexpected error occurred: {str(e)}\n"
                               f"Please try a different move or restart the game.")
            self.selected_positions = []
            self.move_button.config(state=tk.DISABLED)
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
            self.make_random_move()
        
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
    
    def make_random_move(self):
        """Make a random valid move with error handling"""
        import random
        
        try:
            if self.game.game_state.turn == Player.COPS:
                # Random cop moves
                new_positions = []
                for i, cop_pos in enumerate(self.game.game_state.cop_positions):
                    try:
                        # Try to get valid moves with tickets for Scotland Yard
                        is_scotland_yard = hasattr(self.game, 'is_scotland_yard') and self.game.is_scotland_yard
                        if is_scotland_yard:
                            valid_moves = list(self.game.get_valid_moves_with_tickets(Player.COPS, i))
                        else:
                            valid_moves = list(self.game.get_valid_moves(Player.COPS, cop_pos))
                        
                        if valid_moves:
                            new_positions.append(random.choice(valid_moves))
                        else:
                            new_positions.append(cop_pos)  # Stay in place if no valid moves
                    except Exception:
                        new_positions.append(cop_pos)  # Stay in place on error
                
                self.game.make_move(new_positions=new_positions)
            else:
                # Random robber move
                try:
                    is_scotland_yard = hasattr(self.game, 'is_scotland_yard') and self.game.is_scotland_yard
                    if is_scotland_yard:
                        valid_moves = list(self.game.get_valid_moves_with_tickets(Player.MR_X))
                    else:
                        valid_moves = list(self.game.get_valid_moves(Player.ROBBER))
                    
                    if valid_moves:
                        new_pos = random.choice(valid_moves)
                        self.game.make_move(new_robber_pos=new_pos)
                except Exception:
                    # If random move fails, just pass the turn
                    pass
                    
        except Exception as e:
            # If random move completely fails, log but don't crash
            print(f"Random move failed: {e}")
            # Try to pass turn or handle gracefully
            pass
    
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
            self.info_text.insert(tk.END, f"Cops needed: {self.game.num_cops}\n")
            self.info_text.insert(tk.END, f"Selected: {len(self.selected_positions)}\n")
        elif self.game.game_state:
            state_info = self.game.get_state_representation()
            self.info_text.insert(tk.END, f"Turn: {state_info['turn'].title()}\n")
            self.info_text.insert(tk.END, f"Turn count: {state_info['turn_count']}\n")
            self.info_text.insert(tk.END, f"Cop positions: {state_info['cop_positions']}\n")
            
            # Show robber position based on game type
            is_scotland_yard = hasattr(self.game, 'is_scotland_yard') and self.game.is_scotland_yard
            if is_scotland_yard:
                if self.game.game_state.mr_x_visible:
                    self.info_text.insert(tk.END, f"Mr. X position: {state_info['robber_position']} (VISIBLE)\n")
                else:
                    self.info_text.insert(tk.END, f"Mr. X position: HIDDEN\n")
            else:
                self.info_text.insert(tk.END, f"Robber position: {state_info['robber_position']}\n")
            
            # Only show game over info if game is actually over
            if self.game.is_game_over():
                winner = self.game.get_winner()
                if winner:
                    self.info_text.insert(tk.END, f"Game Over - Winner: {winner.value.title()}\n")
            else:
                self.info_text.insert(tk.END, "Game in progress\n")
            
            if self.solver_result:
                self.info_text.insert(tk.END, f"\nSolver: Cops can win = {self.solver_result.cops_can_win}\n")
        
        # Update tickets display
        self.update_tickets_display()
        
        # Update move help
        self.update_move_help()
    
    def update_tickets_display(self):
        """Update the tickets display for Scotland Yard games"""
        self.tickets_text.delete(1.0, tk.END)
        
        if not self.game.game_state:
            return
        
        is_scotland_yard = hasattr(self.game, 'is_scotland_yard') and self.game.is_scotland_yard
        if not is_scotland_yard:
            self.tickets_text.insert(tk.END, "Not a Scotland Yard game")
            return
        
        # Show detective tickets
        self.tickets_text.insert(tk.END, "DETECTIVE TICKETS:\n")
        for i in range(self.game.num_cops):
            tickets = self.game.get_detective_tickets(i)
            pos = self.game.game_state.cop_positions[i]
            self.tickets_text.insert(tk.END, f"Detective {i+1} (pos {pos}):\n")
            for ticket_type, count in tickets.items():
                self.tickets_text.insert(tk.END, f"  {ticket_type.value}: {count}\n")
            self.tickets_text.insert(tk.END, "\n")
        
        # Show Mr. X tickets
        mr_x_tickets = self.game.get_mr_x_tickets()
        self.tickets_text.insert(tk.END, "MR. X TICKETS:\n")
        for ticket_type, count in mr_x_tickets.items():
            self.tickets_text.insert(tk.END, f"  {ticket_type.value}: {count}\n")
    
    def update_move_help(self):
        """Update move instruction text"""
        self.move_help_text.delete(1.0, tk.END)
        
        if self.setup_mode:
            self.move_help_text.insert(tk.END, "Click nodes to select starting positions")
            return
        
        if not self.game.game_state or self.game.is_game_over():
            return
        
        is_scotland_yard = hasattr(self.game, 'is_scotland_yard') and self.game.is_scotland_yard
        
        if self.game.game_state.turn == Player.COPS:
            if is_scotland_yard:
                self.move_help_text.insert(tk.END, f"Select {self.game.num_cops} new positions for detectives.\n")
                self.move_help_text.insert(tk.END, "Check tickets before moving!\n")
                self.move_help_text.insert(tk.END, f"Selected: {len(self.selected_positions)}/{self.game.num_cops}")
            else:
                self.move_help_text.insert(tk.END, f"Select {self.game.num_cops} new positions for cops.\n")
                self.move_help_text.insert(tk.END, f"Selected: {len(self.selected_positions)}/{self.game.num_cops}")
        else:
            if is_scotland_yard:
                self.move_help_text.insert(tk.END, "Select new position for Mr. X.\n")
                self.move_help_text.insert(tk.END, "Check tickets before moving!")
            else:
                self.move_help_text.insert(tk.END, "Select new position for robber.")
    
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

    
    def run(self):
        """Start the GUI application"""
        self.root.mainloop()
