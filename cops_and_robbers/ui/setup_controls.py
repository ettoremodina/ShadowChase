import tkinter as tk
from tkinter import ttk, messagebox
from .ui_components import StyledButton, InfoDisplay
from ..core.game import ScotlandYardGame

class SetupControls:
    """Handles game setup UI and logic"""
    
    def __init__(self, visualizer):
        self.visualizer = visualizer
        self.setup_section = None
        
    def create_setup_section(self, parent):
        """Create the game setup section"""
        self.setup_section = ttk.LabelFrame(parent, text="🎯 Game Setup")
        
        # Configure label style for left alignment
        style = ttk.Style()
        style.configure("Setup.TLabelframe.Label", anchor="w", font=('Arial', 11, 'bold'))
        self.setup_section.configure(style="Setup.TLabelframe")
        
        # Instructions
        instruction_frame = ttk.Frame(self.setup_section)
        instruction_frame.pack(fill=tk.X, padx=10, pady=8)
        
        ttk.Label(instruction_frame, 
                 text="Click nodes on the graph to select starting positions",
                 font=('Arial', 9)).pack(anchor="w")
        
        # Status display
        self.status_display = InfoDisplay(self.setup_section, "Selection Status", height=3)
        self.status_display.pack(fill=tk.X, padx=5, pady=5)
        
        # Buttons frame
        button_frame = ttk.Frame(self.setup_section)
        button_frame.pack(fill=tk.X, padx=10, pady=8)
        
        self.start_button = StyledButton(button_frame, "🚀 Start Game", 
                                       command=self.start_game, style_type="success",
                                       state=tk.DISABLED)
        self.start_button.pack(fill=tk.X, pady=3)
        
        self.reset_button = StyledButton(button_frame, "🔄 Reset Setup", 
                                       command=self.reset_setup, style_type="warning")
        self.reset_button.pack(fill=tk.X, pady=3)
        
        return self.setup_section
    
    def update_status(self):
        """Update the setup status display"""
        if not self.status_display:
            return
            
        selected_count = len(self.visualizer.selected_positions)
        needed_total = self.visualizer.game.num_cops + 1
        
        status_text = f"Selected: {selected_count}/{needed_total}\n"
        status_text += f"Need: {self.visualizer.game.num_cops} detectives + 1 Mr. X\n"
        
        if selected_count < self.visualizer.game.num_cops:
            status_text += f"Select detective {selected_count + 1} position"
        elif selected_count == self.visualizer.game.num_cops:
            status_text += "Select Mr. X starting position"
        else:
            status_text += "Ready to start!"
        
        self.status_display.set_text(status_text)
        
        # Update button state
        if (selected_count == needed_total and 
            self.visualizer.selected_positions[-1] not in self.visualizer.selected_positions[:-1]):
            self.start_button.config(state=tk.NORMAL)
        else:
            self.start_button.config(state=tk.DISABLED)
    
    def start_game(self):
        """Start the game with selected positions"""
        if len(self.visualizer.selected_positions) != self.visualizer.game.num_cops + 1:
            messagebox.showerror("Error", f"Select {self.visualizer.game.num_cops} positions and 1 robber position")
            return
        
        cop_positions = self.visualizer.selected_positions[:self.visualizer.game.num_cops]
        robber_position = self.visualizer.selected_positions[self.visualizer.game.num_cops]
        
        if robber_position in cop_positions:
            messagebox.showerror("Error", "Mr. X and detectives cannot start in the same position")
            return
        
        # Reset game state
        self.visualizer.game.game_state = None
        self.visualizer.game.game_history = []
        
        try:
            # Use appropriate initialization method
            if isinstance(self.visualizer.game, ScotlandYardGame):
                self.visualizer.game.initialize_scotland_yard_game(cop_positions, robber_position)
            else:
                self.visualizer.game.initialize_game(cop_positions, robber_position)
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            return
        
        self.visualizer.setup_mode = False
        self.visualizer.selected_positions = []
        self.visualizer.current_cop_index = 0
        self.visualizer.cop_selections = []
        self.visualizer.selected_nodes = []
        
        # Update UI visibility and redraw
        self.visualizer.update_ui_visibility()
        self.visualizer.draw_graph()
        
        # messagebox.showinfo("🎉 Game Started", "Game has been initialized successfully!")

    def reset_setup(self):
        """Reset game setup"""
        self.visualizer.selected_positions = []
        self.visualizer.setup_mode = True
        self.visualizer.auto_play = False
        self.visualizer.solver_result = None
        self.visualizer.current_player_moves = {}
        self.visualizer.highlighted_edges = []
        self.visualizer.active_player_positions = []
        self.visualizer.current_cop_index = 0
        self.visualizer.cop_selections = []
        self.visualizer.selected_nodes = []
        
        # Clear game state
        self.visualizer.game.game_state = None
        self.visualizer.game.game_history = []
        
        # Update UI and redraw
        self.visualizer.update_ui_visibility()
        self.visualizer.draw_graph()
        self.update_status()
