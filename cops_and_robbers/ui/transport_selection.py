"""
Transport selection dialog for Scotland Yard game.
Allows users to choose between multiple transport types when moving between nodes.
"""

import tkinter as tk
from tkinter import ttk
from typing import List, Optional, Tuple
from ..core.game import TransportType


class TransportSelectionDialog:
    """Dialog for selecting transport type when multiple options are available"""
    
    def __init__(self, parent, source_pos: int, dest_pos: int, 
                 available_transports: List[TransportType], 
                 player_type: str = "Player",
                 can_use_black: bool = False):
        self.parent = parent
        self.source_pos = source_pos
        self.dest_pos = dest_pos
        self.available_transports = available_transports
        self.player_type = player_type
        self.can_use_black = can_use_black
        self.selected_transport = None
        
        self.dialog = None
        self.transport_var = tk.StringVar()
        
        # Transport icons
        self.transport_icons = {
            TransportType.TAXI: "🚕",
            TransportType.BUS: "🚌", 
            TransportType.UNDERGROUND: "🚇",
            TransportType.BLACK: "⚫",
            TransportType.FERRY: "🚢"
        }
        
    def show(self) -> Optional[TransportType]:
        """Show the dialog and return the selected transport type"""
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title("Select Transport")
        self.dialog.geometry("400x300")
        self.dialog.resizable(False, False)
        
        # Make dialog modal
        self.dialog.transient(self.parent)
        self.dialog.grab_set()
        
        # Center the dialog
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (self.dialog.winfo_width() // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (self.dialog.winfo_height() // 2)
        self.dialog.geometry(f"+{x}+{y}")
        
        self._create_widgets()
        
        # Wait for user choice
        self.parent.wait_window(self.dialog)
        
        return self.selected_transport
    
    def _create_widgets(self):
        """Create the dialog widgets"""
        # Title
        title_frame = ttk.Frame(self.dialog)
        title_frame.pack(fill=tk.X, padx=20, pady=10)
        
        title_label = ttk.Label(title_frame, 
                               text=f"🚀 {self.player_type} Transport Selection",
                               font=('Arial', 12, 'bold'))
        title_label.pack()
        
        # Move description
        move_frame = ttk.Frame(self.dialog)
        move_frame.pack(fill=tk.X, padx=20, pady=5)
        
        move_label = ttk.Label(move_frame, 
                              text=f"Moving from position {self.source_pos} to {self.dest_pos}",
                              font=('Arial', 10))
        move_label.pack()
        
        # Transport options
        options_frame = ttk.LabelFrame(self.dialog, text="Available Transport Types")
        options_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Add regular transport options
        for i, transport in enumerate(self.available_transports):
            transport_frame = ttk.Frame(options_frame)
            transport_frame.pack(fill=tk.X, padx=10, pady=5)
            
            icon = self.transport_icons.get(transport, "🎫")
            transport_name = transport.name.capitalize()
            
            radio_text = f"{icon} {transport_name}"
            
            # Add transport details
            if transport == TransportType.TAXI:
                radio_text += " (Fast, short distance)"
            elif transport == TransportType.BUS:
                radio_text += " (Medium speed, medium distance)"
            elif transport == TransportType.UNDERGROUND:
                radio_text += " (Fast, long distance)"
            elif transport == TransportType.BLACK:
                radio_text += " (Secret Mr. X ticket)"
            elif transport == TransportType.FERRY:
                radio_text += " (Water transport)"
            
            # Use transport.value as the radio button value for consistent comparison
            radio = ttk.Radiobutton(transport_frame, 
                                  text=radio_text,
                                  variable=self.transport_var,
                                  value=str(transport.value),  # Use string of enum value
                                #   font=('Arial', 10)
                                  )
            radio.pack(anchor=tk.W)
            
            # Select first option by default
            if i == 0:
                self.transport_var.set(str(transport.value))
        
        # Add black ticket option for Mr. X if available
        if self.can_use_black and self.player_type == "Mr. X":
            separator_frame = ttk.Frame(options_frame)
            separator_frame.pack(fill=tk.X, padx=10, pady=5)
            
            ttk.Separator(separator_frame, orient='horizontal').pack(fill=tk.X, pady=5)
            
            black_frame = ttk.Frame(options_frame)
            black_frame.pack(fill=tk.X, padx=10, pady=5)
            
            black_radio = ttk.Radiobutton(black_frame, 
                                        text="⚫ Use Black Ticket (Hide transport type)",
                                        variable=self.transport_var,
                                        value=str(TransportType.BLACK.value),
                                        # font=('Arial', 10, 'bold')
                                        )
            black_radio.pack(anchor=tk.W)
        
        # Buttons
        button_frame = ttk.Frame(self.dialog)
        button_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ok_button = ttk.Button(button_frame, text="✅ Confirm Move", 
                              command=self._on_ok)
        ok_button.pack(side=tk.RIGHT, padx=5)
        
        cancel_button = ttk.Button(button_frame, text="❌ Cancel", 
                                  command=self._on_cancel)
        cancel_button.pack(side=tk.RIGHT, padx=5)
        
        # Key bindings
        self.dialog.bind('<Return>', lambda e: self._on_ok())
        self.dialog.bind('<Escape>', lambda e: self._on_cancel())
        
        # Focus on OK button
        ok_button.focus()
    
    def _on_ok(self):
        """Handle OK button click"""
        selected_value = self.transport_var.get()
        if selected_value:
            try:
                # Convert string back to integer and find matching transport
                selected_int = int(selected_value)
                for transport in self.available_transports:
                    if transport.value == selected_int:
                        self.selected_transport = transport
                        break
                
                # Handle black ticket selection
                if selected_int == TransportType.BLACK.value:
                    self.selected_transport = TransportType.BLACK
                    
            except ValueError:
                self.selected_transport = None
        
        self.dialog.destroy()
    
    def _on_cancel(self):
        """Handle Cancel button click"""
        self.selected_transport = None
        self.dialog.destroy()


def select_transport(parent, source_pos: int, dest_pos: int, 
                    available_transports: List[TransportType],
                    player_type: str = "Player",
                    can_use_black: bool = False) -> Optional[TransportType]:
    """
    Convenience function to show transport selection dialog
    
    Args:
        parent: Parent widget
        source_pos: Source position
        dest_pos: Destination position
        available_transports: List of available transport types
        player_type: Type of player (for dialog title)
        can_use_black: Whether black ticket can be used (for Mr. X)
    
    Returns:
        Selected transport type or None if cancelled
    """
    if not available_transports:
        return None
    
    # Always show dialog if there are multiple options OR if Mr. X can use black ticket
    if len(available_transports) == 1 and not can_use_black:
        return available_transports[0]
    
    dialog = TransportSelectionDialog(parent, source_pos, dest_pos, 
                                     available_transports, player_type, can_use_black)
    dialog.show()
    return dialog.selected_transport
