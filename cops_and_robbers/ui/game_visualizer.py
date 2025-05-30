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
        self.root.geometry("1200x800")
        
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
        self.control_frame = ttk.Frame(main_frame, width=300)
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
        
        # Draw edges
        nx.draw_networkx_edges(self.game.graph, self.pos, ax=self.ax, 
                              edge_color='gray', alpha=0.5)
        
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
            self.selected_positions.append(node)
        
        # Enable start button when we have enough positions
        if len(self.selected_positions) == self.game.num_cops + 1:
            self.setup_button.config(state=tk.NORMAL)
        else:
            self.setup_button.config(state=tk.DISABLED)
        
        self.draw_graph()
    
    def handle_game_click(self, node):
        """Handle node clicks during game"""
        if not self.game.game_state or self.game.is_game_over():
            return
        
        if self.game.game_state.turn == Player.COPS:
            # Select positions for cops
            if node in self.selected_positions:
                self.selected_positions.remove(node)
            else:
                self.selected_positions.append(node)
            
            if len(self.selected_positions) == self.game.num_cops:
                self.move_button.config(state=tk.NORMAL)
            else:
                self.move_button.config(state=tk.DISABLED)
        
        else:  # Robber's turn
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
        
        self.game.initialize_game(cop_positions, robber_position)
        self.setup_mode = False
        self.selected_positions = []
        
        # Update UI
        self.setup_button.config(state=tk.DISABLED)
        self.move_button.config(state=tk.DISABLED)
        self.auto_button.config(state=tk.NORMAL)
        self.solve_button.config(state=tk.NORMAL)
        
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
        
        self.game.game_state = None
        self.draw_graph()
    
   
    def make_manual_move(self):
        """Make a manual move"""
        if not self.selected_positions:
            return
        
        if self.game.game_state.turn == Player.COPS:
            if len(self.selected_positions) != self.game.num_cops:
                messagebox.showerror("Error", f"Select {self.game.num_cops} positions for cops")
                return
            
            success = self.game.make_move(new_positions=self.selected_positions)
        else:
            if len(self.selected_positions) != 1:
                messagebox.showerror("Error", "Select 1 position for robber")
                return
            
            success = self.game.make_move(new_robber_pos=self.selected_positions[0])
        
        if not success:
            messagebox.showerror("Error", "Invalid move")
            return
        
        self.selected_positions = []
        self.move_button.config(state=tk.DISABLED)
        self.draw_graph()
        
        if self.game.is_game_over():
            winner = self.game.get_winner()
            messagebox.showinfo("Game Over", f"{winner.value.title()} wins!")
            self.auto_play = False
            self.auto_button.config(text="Auto Play")
    
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
        """Make a random valid move"""
        import random
        
        if self.game.game_state.turn == Player.COPS:
            # Random cop moves
            new_positions = []
            for cop_pos in self.game.game_state.cop_positions:
                valid_moves = list(self.game.get_valid_moves(Player.COPS, cop_pos))
                if valid_moves:
                    new_positions.append(random.choice(valid_moves))
                else:
                    new_positions.append(cop_pos)
            self.game.make_move(new_positions=new_positions)
        else:
            # Random robber move
            valid_moves = list(self.game.get_valid_moves(Player.ROBBER))
            if valid_moves:
                new_pos = random.choice(valid_moves)
                self.game.make_move(new_robber_pos=new_pos)
    
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
        strategy_window.geometry("600x400")
        
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
            self.info_text.insert(tk.END, f"Robber position: {state_info['robber_position']}\n")
            
            if state_info['game_over']:
                self.info_text.insert(tk.END, f"Winner: {state_info['winner'].title()}\n")
            
            if self.solver_result:
                self.info_text.insert(tk.END, f"\nSolver: Cops can win = {self.solver_result.cops_can_win}\n")
    
    
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
        load_window.geometry("600x400")
        
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
        history_window.geometry("800x600")
        
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
