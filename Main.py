import tkinter as tk
from tkinter import ttk, messagebox
import networkx as nx
from cops_and_robbers.ui.game_visualizer import GameVisualizer
from cops_and_robbers.examples.example_games import *

class GameLauncher:
    """Unified game launcher with all game types and modes"""
    
    def __init__(self, fullscreen: bool = False):
        self.root = tk.Tk()
        self.root.title("Cops and Robbers - Game Launcher")
        
        if fullscreen:
            self.root.attributes('-fullscreen', True)
            self.root.bind('<Escape>', lambda e: self.root.destroy())
        else:
            self.root.geometry("500x400")
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup launcher UI"""
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        title_label = ttk.Label(main_frame, text="Cops and Robbers Game", 
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Game type selection
        game_frame = ttk.LabelFrame(main_frame, text="Select Game Type", padding="10")
        game_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.game_var = tk.StringVar(value="scotland_yard")
        
        games = [
            ("Scotland Yard (Full Rules)", "scotland_yard"),
            ("Scotland Yard (Simplified)", "simple_scotland_yard"),
            ("Test Scotland Yard (Full Rules)", "test_scotland_yard"),
            ("Test Scotland Yard (Simplified)", "simple_test_scotland_yard"),
            ("Path Graph", "path"),
            ("Cycle Graph", "cycle"), 
            ("Complete Graph", "complete"),
            ("Grid Graph", "grid"),
            ("Petersen Graph", "petersen")
        ]
        
        for text, value in games:
            ttk.Radiobutton(game_frame, text=text, variable=self.game_var, 
                           value=value).pack(anchor=tk.W)
        
        # Parameters
        param_frame = ttk.LabelFrame(main_frame, text="Parameters", padding="10")
        param_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Graph size (for non-Scotland Yard games)
        ttk.Label(param_frame, text="Graph Size (for non-Scotland Yard):").pack(anchor=tk.W)
        self.size_var = tk.StringVar(value="5")
        ttk.Entry(param_frame, textvariable=self.size_var, width=10).pack(anchor=tk.W, pady=(0, 5))
        
        # Number of cops/detectives
        ttk.Label(param_frame, text="Number of Cops/Detectives:").pack(anchor=tk.W)
        self.cops_var = tk.StringVar(value="3")
        ttk.Entry(param_frame, textvariable=self.cops_var, width=10).pack(anchor=tk.W)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=20)
        
        ttk.Button(button_frame, text="Launch Game", 
                  command=self.launch_game).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(button_frame, text="Quick Test Mode", 
                  command=self.launch_test).pack(side=tk.LEFT, padx=(0, 10))
        
        if self.root.attributes('-fullscreen'):
            ttk.Button(button_frame, text="Exit", 
                      command=self.root.destroy).pack(side=tk.LEFT)
    
    def launch_game(self):
        """Launch selected game"""
        try:
            num_cops = int(self.cops_var.get())
            game_type = self.game_var.get()
            
            if game_type == "scotland_yard":
                game = create_scotlandYard_game(num_cops)
            elif game_type == "simple_scotland_yard":
                game = create_simple_scotland_yard_game(num_cops, show_robber=True, use_tickets=False)
            elif game_type == "test_scotland_yard":
                game = create_test_scotland_yard_game(2)
                game.initialize_scotland_yard_game([1, 3], 8) 
            elif game_type == "simple_test_scotland_yard":
                game = create_simple_test_scotland_yard_game(num_cops, show_robber=True, use_tickets=False)
            else:
                size = int(self.size_var.get())
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
                else:
                    raise ValueError("Unknown game type")
            
            visualizer = GameVisualizer(game)
            visualizer.run()
            
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid parameters: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create game: {str(e)}")
    
    def launch_test(self):
        """Launch test script"""
        import subprocess
        import sys
        try:
            subprocess.run([sys.executable, "test.py"], check=True)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch test: {str(e)}")
    
    def run(self):
        """Start the launcher"""
        self.root.mainloop()

if __name__ == "__main__":
    import sys
    fullscreen = "--fullscreen" in sys.argv or "-f" in sys.argv
    
    launcher = GameLauncher(fullscreen=fullscreen)
    launcher.run()