"""
Setup Controller Component

Handles the game setup phase with position selection and validation.
"""

import tkinter as tk
from tkinter import messagebox
from typing import Set, List, Optional, Callable
from ..core.game import Game, Player, ScotlandYardGame


class SetupController:
    """Controller for game setup phase"""
    
    def __init__(self, game: Game):
        """Initialize setup controller"""
        self.game = game
        self.is_scotland_yard = isinstance(game, ScotlandYardGame)
        
        # Selection state
        self.selected_cop_positions: List[int] = []
        self.selected_robber_position: Optional[int] = None
        self.current_selection_mode = "cops"  # "cops" or "robber"
        
        # Required positions
        self.required_cops = game.num_cops
        self.required_robber = 1
        
        # Callbacks
        self.on_selection_changed: Optional[Callable] = None
        self.on_setup_complete: Optional[Callable] = None
        self.on_status_update: Optional[Callable[[str], None]] = None
        
        # Initial state
        self.reset_selections()
    
    def handle_node_click(self, node: int) -> bool:
        """Handle node click during setup phase"""
        if self.current_selection_mode == "cops":
            return self._handle_cop_selection(node)
        elif self.current_selection_mode == "robber":
            return self._handle_robber_selection(node)
        return False
    
    def _handle_cop_selection(self, node: int) -> bool:
        """Handle cop position selection"""
        # Check if already selected
        if node in self.selected_cop_positions:
            # Deselect
            self.selected_cop_positions.remove(node)
            self._update_status()
            self._notify_selection_changed()
            return True
        
        # Check if robber is at this position
        if node == self.selected_robber_position:
            self._show_message("Cannot place detective at Mr. X's position!")
            return False
        
        # Check if we have room for more cops
        if len(self.selected_cop_positions) >= self.required_cops:
            # Replace last selection or show message
            if self.required_cops == 1:
                self.selected_cop_positions = [node]
            else:
                self._show_message(f"Already selected {self.required_cops} detectives. Click existing selection to deselect.")
                return False
        else:
            # Add new selection
            self.selected_cop_positions.append(node)
        
        self._update_status()
        self._notify_selection_changed()
        
        # Check if we should switch to robber selection
        if len(self.selected_cop_positions) == self.required_cops:
            if self.selected_robber_position is None:
                self.current_selection_mode = "robber"
                self._update_status()
        
        return True
    
    def _handle_robber_selection(self, node: int) -> bool:
        """Handle robber position selection"""
        # Check if cop is at this position
        if node in self.selected_cop_positions:
            self._show_message("Cannot place Mr. X at a detective's position!")
            return False
        
        # Set robber position
        self.selected_robber_position = node
        self._update_status()
        self._notify_selection_changed()
        
        # Check if setup is complete
        self._check_setup_complete()
        
        return True
    
    def get_selected_nodes(self) -> Set[int]:
        """Get all currently selected nodes"""
        selected = set(self.selected_cop_positions)
        if self.selected_robber_position is not None:
            selected.add(self.selected_robber_position)
        return selected
    
    def get_cop_positions(self) -> List[int]:
        """Get selected cop positions"""
        return self.selected_cop_positions.copy()
    
    def get_robber_position(self) -> Optional[int]:
        """Get selected robber position"""
        return self.selected_robber_position
    
    def is_setup_complete(self) -> bool:
        """Check if setup is complete"""
        cops_ready = len(self.selected_cop_positions) == self.required_cops
        robber_ready = self.selected_robber_position is not None
        return cops_ready and robber_ready
    
    def get_status_message(self) -> str:
        """Get current status message"""
        if self.current_selection_mode == "cops":
            selected = len(self.selected_cop_positions)
            if selected == 0:
                return f"Click {self.required_cops} nodes to place detectives"
            elif selected < self.required_cops:
                remaining = self.required_cops - selected
                return f"Selected {selected}/{self.required_cops} detectives. Select {remaining} more."
            else:
                return f"All {self.required_cops} detectives placed. Now select Mr. X position."
        
        elif self.current_selection_mode == "robber":
            if self.selected_robber_position is None:
                return "Click a node to place Mr. X"
            else:
                return "Setup complete! Click 'Start Game' to begin."
        
        return "Setup in progress..."
    
    def reset_selections(self):
        """Reset all selections"""
        self.selected_cop_positions.clear()
        self.selected_robber_position = None
        self.current_selection_mode = "cops"
        self._update_status()
        self._notify_selection_changed()
    
    def start_game(self) -> bool:
        """Start the game with selected positions"""
        if not self.is_setup_complete():
            self._show_message("Setup not complete. Please select all positions.")
            return False
        
        try:
            if self.is_scotland_yard:
                # Initialize Scotland Yard game
                self.game.initialize_scotland_yard_game(
                    self.selected_cop_positions,
                    self.selected_robber_position
                )
            else:
                # Initialize standard game
                self.game.initialize_game(
                    self.selected_cop_positions,
                    self.selected_robber_position
                )
            
            if self.on_setup_complete:
                self.on_setup_complete()
            
            return True
            
        except Exception as e:
            self._show_message(f"Failed to start game: {str(e)}")
            return False
    
    def validate_positions(self) -> bool:
        """Validate selected positions"""
        # Check all positions are different
        all_positions = set(self.selected_cop_positions)
        if self.selected_robber_position is not None:
            if self.selected_robber_position in all_positions:
                return False
            all_positions.add(self.selected_robber_position)
        
        # Check all positions are valid nodes
        for pos in all_positions:
            if pos not in self.game.graph.nodes():
                return False
        
        return True
    
    def get_available_nodes(self) -> Set[int]:
        """Get nodes available for selection"""
        available = set(self.game.graph.nodes())
        
        if self.current_selection_mode == "cops":
            # Remove robber position if selected
            if self.selected_robber_position is not None:
                available.discard(self.selected_robber_position)
        
        elif self.current_selection_mode == "robber":
            # Remove cop positions
            for pos in self.selected_cop_positions:
                available.discard(pos)
        
        return available
    
    def set_selection_mode(self, mode: str):
        """Set current selection mode"""
        if mode in ["cops", "robber"]:
            self.current_selection_mode = mode
            self._update_status()
    
    def _update_status(self):
        """Update status message"""
        if self.on_status_update:
            message = self.get_status_message()
            self.on_status_update(message)
    
    def _notify_selection_changed(self):
        """Notify that selection has changed"""
        if self.on_selection_changed:
            self.on_selection_changed()
    
    def _check_setup_complete(self):
        """Check if setup is complete and notify"""
        if self.is_setup_complete():
            self._update_status()
    
    def _show_message(self, message: str):
        """Show message to user"""
        messagebox.showwarning("Setup", message)
    
    def get_node_colors(self) -> dict:
        """Get node colors for visualization"""
        colors = {}
        
        # Cop positions - blue
        for pos in self.selected_cop_positions:
            colors[pos] = "blue"
        
        # Robber position - red
        if self.selected_robber_position is not None:
            colors[self.selected_robber_position] = "red"
        
        return colors
    
    def auto_setup_random(self):
        """Automatically setup with random positions"""
        import random
        
        available_nodes = list(self.game.graph.nodes())
        
        # Select random positions ensuring no conflicts
        selected_positions = random.sample(available_nodes, self.required_cops + 1)
        
        self.selected_cop_positions = selected_positions[:self.required_cops]
        self.selected_robber_position = selected_positions[self.required_cops]
        
        self.current_selection_mode = "robber"  # Set to final mode
        self._update_status()
        self._notify_selection_changed()
    
    def auto_setup_strategic(self):
        """Automatically setup with strategic positions"""
        # Place cops at nodes with high degree (central positions)
        # Place robber at node with good escape routes
        
        node_degrees = dict(self.game.graph.degree())
        sorted_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)
        
        # Place cops at high-degree nodes
        self.selected_cop_positions = [node for node, _ in sorted_nodes[:self.required_cops]]
        
        # Place robber at a node that's not too close to cops but has good connectivity
        available_for_robber = [node for node, _ in sorted_nodes if node not in self.selected_cop_positions]
        
        if available_for_robber:
            # Choose a node with good degree but not the highest
            mid_index = min(len(available_for_robber) // 3, len(available_for_robber) - 1)
            self.selected_robber_position = available_for_robber[mid_index]
        
        self.current_selection_mode = "robber"
        self._update_status()
        self._notify_selection_changed()
    
    # Callback setters
    def set_selection_changed_callback(self, callback: Callable):
        """Set callback for selection changes"""
        self.on_selection_changed = callback
    
    def set_setup_complete_callback(self, callback: Callable):
        """Set callback for setup completion"""
        self.on_setup_complete = callback
    
    def set_status_update_callback(self, callback: Callable[[str], None]):
        """Set callback for status updates"""
        self.on_status_update = callback
