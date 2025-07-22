import tkinter as tk
from tkinter import ttk, messagebox
from .ui_components import StyledButton, InfoDisplay
from ..core.game import ScotlandYardGame

class SetupControls:
    """Handles game setup UI and logic"""
    
    def __init__(self, visualizer):
        self.visualizer = visualizer
        self.setup_section = None
        self.game_mode = tk.StringVar(value="human_vs_human")
        self.heuristics_enabled = tk.BooleanVar(value=False)  # New heuristics toggle
        
        # Agent selection variables
        self.mr_x_agent_type = tk.StringVar(value="random")
        self.detective_agent_type = tk.StringVar(value="random")
        
    def create_setup_section(self, parent):
        """Create the game setup section"""
        self.setup_section = ttk.LabelFrame(parent, text="🎯 Game Setup")
        
        # Configure label style for left alignment
        style = ttk.Style()
        style.configure("Setup.TLabelframe.Label", anchor="w", font=('Arial', 11, 'bold'))
        self.setup_section.configure(style="Setup.TLabelframe")
        
        # Game mode selection
        mode_frame = ttk.LabelFrame(self.setup_section, text="🎮 Game Mode")
        mode_frame.pack(fill=tk.X, padx=10, pady=8)
        
        game_modes = [
            ("human_vs_human", "👥 Human vs Human"),
            ("human_det_vs_ai_mrx", "👮 Human Detectives vs 🤖 AI Mr. X"),
            ("ai_det_vs_human_mrx", "🤖 AI Detectives vs 👤 Human Mr. X"),
            ("ai_vs_ai", "🤖 AI vs AI")
        ]
        
        for mode_value, mode_text in game_modes:
            radio = ttk.Radiobutton(mode_frame, text=mode_text, 
                                  variable=self.game_mode, value=mode_value,
                                  command=self._on_game_mode_change)
            radio.pack(anchor="w", padx=10, pady=2)
        
        # Agent selection (only shown when AI is involved)
        self.agent_frame = ttk.LabelFrame(self.setup_section, text="🤖 AI Agent Selection")
        
        # Mr. X agent selection
        mr_x_subframe = ttk.Frame(self.agent_frame)
        mr_x_subframe.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(mr_x_subframe, text="Mr. X AI Agent:", font=('Arial', 9, 'bold')).pack(anchor="w")
        
        # Import locally to avoid circular import
        from agents import AgentSelector
        agent_choices = AgentSelector.get_agent_choices_for_ui()
        self.mr_x_agent_combo = ttk.Combobox(mr_x_subframe, textvariable=self.mr_x_agent_type,
                                           values=[choice[1] for choice in agent_choices],
                                           state="readonly")
        self.mr_x_agent_combo.pack(fill=tk.X, pady=(2, 0))
        
        # Detective agent selection
        det_subframe = ttk.Frame(self.agent_frame)
        det_subframe.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(det_subframe, text="Detective AI Agent:", font=('Arial', 9, 'bold')).pack(anchor="w")
        
        self.detective_agent_combo = ttk.Combobox(det_subframe, textvariable=self.detective_agent_type,
                                                values=[choice[1] for choice in agent_choices],
                                                state="readonly")
        self.detective_agent_combo.pack(fill=tk.X, pady=(2, 0))
        
        # Initially hide agent selection and update based on game mode
        self._on_game_mode_change()
        
        # Heuristics visualization toggle (only for Scotland Yard games)
        if isinstance(self.visualizer.game, ScotlandYardGame):
            heuristics_frame = ttk.LabelFrame(self.setup_section, text="🧠 Heuristics")
            heuristics_frame.pack(fill=tk.X, padx=10, pady=8)
            
            heuristics_checkbox = ttk.Checkbutton(
                heuristics_frame, 
                text="🔍 Show possible Mr. X positions (detective turns only)",
                variable=self.heuristics_enabled,
                command=self._on_heuristics_toggle
            )
            heuristics_checkbox.pack(anchor="w", padx=10, pady=5)
        
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
    
    def _on_heuristics_toggle(self):
        """Handle heuristics toggle change"""
        if self.heuristics_enabled.get():
            # Initialize heuristics when enabled
            self.visualizer._initialize_heuristics()
        # Redraw graph to show/hide heuristics
        if not self.visualizer.setup_mode:
            self.visualizer.draw_graph()
    
    def _on_game_mode_change(self):
        """Handle game mode change to show/hide agent selection"""
        game_mode = self.game_mode.get()
        
        # Determine which AI agents are needed
        needs_mr_x_ai = game_mode in ["human_det_vs_ai_mrx", "ai_vs_ai"]
        needs_detective_ai = game_mode in ["ai_det_vs_human_mrx", "ai_vs_ai"]
        
        if needs_mr_x_ai or needs_detective_ai:
            # Show agent selection frame
            self.agent_frame.pack(fill=tk.X, padx=10, pady=8)
            
            # Show/hide individual agent selections based on need
            if needs_mr_x_ai:
                self.mr_x_agent_combo.pack_info()  # Make sure it's visible
            else:
                self.mr_x_agent_combo.pack_forget()  # Hide it
            
            if needs_detective_ai:
                self.detective_agent_combo.pack_info()  # Make sure it's visible
            else:
                self.detective_agent_combo.pack_forget()  # Hide it
        else:
            # Hide entire agent selection frame
            self.agent_frame.pack_forget()
    
    def get_heuristics_enabled(self):
        """Get whether heuristics visualization is enabled"""
        return self.heuristics_enabled.get()
    
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
    
    def get_selected_mode(self):
        """Get the selected game mode"""
        return self.game_mode.get()
    
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
        
        # Set the game mode in the visualizer
        self.visualizer.game_mode = self.get_selected_mode()
        
        # Set selected agent types
        from agents import AgentType, AgentSelector
        try:
            # Map combobox values back to AgentType enums
            agent_choices = AgentSelector.get_agent_choices_for_ui()
            agent_value_map = {choice[1]: choice[0] for choice in agent_choices}
            
            mr_x_display_name = self.mr_x_agent_type.get()
            detective_display_name = self.detective_agent_type.get()
            
            if mr_x_display_name in agent_value_map:
                self.visualizer.mr_x_agent_type = AgentType(agent_value_map[mr_x_display_name])
            
            if detective_display_name in agent_value_map:
                self.visualizer.detective_agent_type = AgentType(agent_value_map[detective_display_name])
        
        except Exception as e:
            print(f"Warning: Failed to set agent types: {e}")
        
        # Initialize heuristics if enabled
        if self.heuristics_enabled.get():
            self.visualizer._initialize_heuristics()
        
        self.visualizer.setup_mode = False
        self.visualizer.selected_positions = []
        self.visualizer.current_cop_index = 0
        self.visualizer.cop_selections = []
        self.visualizer.selected_nodes = []
        
        # Update UI visibility and redraw
        self.visualizer.update_ui_visibility()
        self.visualizer.draw_graph()
        
        # Show game mode info
        mode_descriptions = {
            "human_vs_human": "Human vs Human",
            "human_det_vs_ai_mrx": "Human Detectives vs AI Mr. X",
            "ai_det_vs_human_mrx": "AI Detectives vs Human Mr. X", 
            "ai_vs_ai": "AI vs AI"
        }
        messagebox.showinfo("🎉 Game Started", 
                          f"Game initialized successfully!\nMode: {mode_descriptions[self.visualizer.game_mode]}")

    def reset_setup(self):
        """Reset game setup"""
        self.visualizer.selected_positions = []
        self.visualizer.setup_mode = True
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
