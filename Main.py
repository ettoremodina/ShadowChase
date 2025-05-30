import tkinter as tk
from tkinter import ttk, messagebox
import networkx as nx
from cops_and_robbers import GameVisualizer, Game
from cops_and_robbers.examples.example_games import *

class GameLauncher:
    """Main application launcher with game selection"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Cops and Robbers - Game Launcher")
        self.root.geometry("400x300")
        self.setup_ui()
    
    def setup_ui(self):
        """Setup launcher UI"""
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        title_label = ttk.Label(main_frame, text="Cops and Robbers Game", 
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Game type selection
        game_frame = ttk.LabelFrame(main_frame, text="Select Graph Type", padding="10")
        game_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.game_var = tk.StringVar(value="path")
        
        games = [
            ("Path Graph", "path"),
            ("Cycle Graph", "cycle"), 
            ("Complete Graph", "complete"),
            ("Grid Graph", "grid"),
            ("Petersen Graph", "petersen"),
            ("Scotland Yard", "scotland_yard")
        ]
        
        for text, value in games:
            ttk.Radiobutton(game_frame, text=text, variable=self.game_var, 
                           value=value).pack(anchor=tk.W)
        
        # Parameters
        param_frame = ttk.LabelFrame(main_frame, text="Parameters", padding="10")
        param_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(param_frame, text="Graph Size:").pack(anchor=tk.W)
        self.size_var = tk.StringVar(value="5")
        ttk.Entry(param_frame, textvariable=self.size_var, width=10).pack(anchor=tk.W)
        
        ttk.Label(param_frame, text="Number of Cops:").pack(anchor=tk.W)
        self.cops_var = tk.StringVar(value="1")
        ttk.Entry(param_frame, textvariable=self.cops_var, width=10).pack(anchor=tk.W)
        
        # Launch button
        ttk.Button(main_frame, text="Launch Game", 
                  command=self.launch_game).pack(pady=20)
    
    def launch_game(self):
        """Launch selected game"""
        try:
            size = int(self.size_var.get())
            num_cops = int(self.cops_var.get())
            game_type = self.game_var.get()
            
            if game_type == "path":
                game = create_path_graph_game(size, num_cops)
            elif game_type == "cycle":
                game = create_cycle_graph_game(size, num_cops)
            elif game_type == "complete":
                game = create_complete_graph_game(size, num_cops)
            elif game_type == "grid":
                game = create_grid_graph_game(size, size, num_cops)
            elif game_type == "petersen":
                game = create_petersen_graph_game(num_cops)
            elif game_type == "scotland_yard":
                game = create_scotlandYard_game(num_cops)
            else:
                raise ValueError("Unknown game type")
            
            visualizer = GameVisualizer(game)
            visualizer.run()
            
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid parameters: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create game: {str(e)}")
    
    def run(self):
        """Start the launcher"""
        self.root.mainloop()

if __name__ == "__main__":
    launcher = GameLauncher()
    launcher.run()