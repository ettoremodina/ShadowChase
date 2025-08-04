import tkinter as tk
from tkinter import ttk
from typing import Optional, Dict, Any


class VisualTicketDisplay(ttk.Frame):
    """Enhanced visual ticket display with table-like structure and colored elements"""
    
    def __init__(self, parent, title="🎫 Ticket Information", **kwargs):
        super().__init__(parent, **kwargs)
        
        # Create the main container with title
        self.label_frame = ttk.LabelFrame(self, text=title)
        self.label_frame.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)
        
        # Configure label style
        style = ttk.Style()
        style.configure("TicketDisplay.TLabelframe.Label", anchor="w", font=('Arial', 10, 'bold'))
        self.label_frame.configure(style="TicketDisplay.TLabelframe")
        
        # Create scrollable container
        self.canvas = tk.Canvas(self.label_frame, bg="#f8f9fa", highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.label_frame, orient="vertical", command=self.canvas.yview)
        self.main_frame = tk.Frame(self.canvas, bg="#f8f9fa")
        
        # Configure scrolling
        self.main_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.main_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Pack canvas and scrollbar
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0), pady=8)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 8), pady=8)
        
        # Add mouse wheel scrolling support
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.main_frame.bind("<MouseWheel>", self._on_mousewheel)
        
        # Transport colors matching the game
        self.transport_colors = {
            'taxi': '#FFD700',      # Yellow
            'bus': '#4169E1',       # Blue  
            'underground': '#DC143C', # Red
            'black': '#2F2F2F',     # Dark gray
            'double_move': '#8A2BE2'  # Purple
        }
        
        # Store player rows for easy updating
        self.player_rows = {}
        
        self._create_header()
    
    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling"""
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    def _create_header(self):
        """Create the table header"""
        header_frame = tk.Frame(self.main_frame, bg="#e9ecef", relief="solid", bd=1)
        header_frame.pack(fill=tk.X, pady=(0, 2))
        
        # Configure grid weights
        header_frame.grid_columnconfigure(0, weight=2)  # Player name column wider
        for i in range(1, 6):
            header_frame.grid_columnconfigure(i, weight=1)
        
        # Header labels
        headers = ["Player", "🚕", "🚌", "🚇", "⚫", "⚡"]
        header_colors = ["#495057", "#FFD700", "#4169E1", "#DC143C", "#2F2F2F", "#8A2BE2"]
        
        for i, (header, color) in enumerate(zip(headers, header_colors)):
            label = tk.Label(header_frame, text=header, font=('Arial', 9, 'bold'), 
                           bg="#e9ecef", fg=color if i > 0 else "#495057", 
                           padx=8, pady=6)
            label.grid(row=0, column=i, sticky="ew")
    
    def _create_player_row(self, player_name: str, is_MrX: bool = False):
        """Create a row for a player"""
        # Background color based on player type
        bg_color = "#ffebee" if is_MrX else "#e3f2fd"
        
        row_frame = tk.Frame(self.main_frame, bg=bg_color, relief="solid", bd=1)
        row_frame.pack(fill=tk.X, pady=1)
        
        # Configure grid weights
        row_frame.grid_columnconfigure(0, weight=2)
        for i in range(1, 6):
            row_frame.grid_columnconfigure(i, weight=1)
        
        # Player name label
        name_color = "#c62828" if is_MrX else "#1565c0"
        name_label = tk.Label(row_frame, text=player_name, font=('Arial', 9, 'bold'),
                            bg=bg_color, fg=name_color, padx=8, pady=4)
        name_label.grid(row=0, column=0, sticky="ew")
        
        # Ticket count labels
        ticket_labels = []
        for i in range(5):
            label = tk.Label(row_frame, text="0", font=('Arial', 9),
                           bg=bg_color, fg="#495057", padx=8, pady=4)
            label.grid(row=0, column=i+1, sticky="ew")
            ticket_labels.append(label)
        
        return row_frame, ticket_labels
    
    def update_tickets(self, game_state, game):
        """Update the ticket display with current game state"""
        # Clear existing rows
        for row_frame, _ in self.player_rows.values():
            row_frame.destroy()
        self.player_rows.clear()
        
        if not game_state:
            return
        
        # Add detective rows
        for i in range(game.num_detectives):
            player_name = f"Det {i+1}"
            row_frame, ticket_labels = self._create_player_row(player_name, is_MrX=False)
            self.player_rows[f"detective_{i}"] = (row_frame, ticket_labels)
            
            # Get detective tickets
            tickets = {}
            if hasattr(game_state, 'detective_tickets'):
                detective_tickets = game_state.detective_tickets
                if isinstance(detective_tickets, dict) and i in detective_tickets:
                    tickets = detective_tickets[i]
                elif isinstance(detective_tickets, list) and i < len(detective_tickets):
                    tickets = detective_tickets[i]
            
            # Update ticket counts
            self._update_ticket_labels(ticket_labels, tickets, is_MrX=False)
        
        # Add Mr. X row
        MrX_row_frame, MrX_labels = self._create_player_row("Mr. X", is_MrX=True)
        self.player_rows["MrX"] = (MrX_row_frame, MrX_labels)
        
        # Get Mr. X tickets
        MrX_tickets = {}
        if hasattr(game_state, 'MrX_tickets'):
            MrX_tickets = game_state.MrX_tickets
        
        # Update Mr. X ticket counts
        self._update_ticket_labels(MrX_labels, MrX_tickets, is_MrX=True)
    
    def _update_ticket_labels(self, labels, tickets, is_MrX=False):
        """Update individual ticket count labels"""
        ticket_types = ['taxi', 'bus', 'underground', 'black', 'double_move']
        
        for i, ticket_type in enumerate(ticket_types):
            count = self._get_ticket_count(tickets, ticket_type)
            
            # Show "-" for detectives who don't have black/double tickets
            if not is_MrX and ticket_type in ['black', 'double_move']:
                labels[i].config(text="-", fg="#adb5bd")
            else:
                # Color the number based on availability
                color = "#28a745" if count > 0 else "#dc3545"
                labels[i].config(text=str(count), fg=color, font=('Arial', 9, 'bold'))
    
    def _get_ticket_count(self, tickets, ticket_name):
        """Helper to get ticket count handling different formats"""
        if not tickets:
            return 0
        
        # Try different possible key formats
        possible_keys = [
            ticket_name.lower(),
            ticket_name.upper(),
            f"TicketType.{ticket_name.upper()}",
        ]
        
        # Also try enum objects
        for key, value in tickets.items():
            if hasattr(key, 'value') and key.value.lower() == ticket_name.lower():
                return value
            elif hasattr(key, 'name') and key.name.lower() == ticket_name.lower():
                return value
            elif str(key).lower() == ticket_name.lower():
                return value
        
        # Try string keys
        for possible_key in possible_keys:
            if possible_key in tickets:
                return tickets[possible_key]
        
        return 0


class EnhancedTurnDisplay(ttk.Frame):
    """Enhanced turn display with progress indicators and status badges"""
    
    def __init__(self, parent, title="📋 Current Turn", **kwargs):
        super().__init__(parent, **kwargs)
        
        # Create the main container with title
        self.label_frame = ttk.LabelFrame(self, text=title)
        self.label_frame.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)
        
        # Configure label style
        style = ttk.Style()
        style.configure("TurnDisplay.TLabelframe.Label", anchor="w", font=('Arial', 10, 'bold'))
        self.label_frame.configure(style="TurnDisplay.TLabelframe")
        
        # Create main container
        self.main_frame = tk.Frame(self.label_frame, bg="#f8f9fa")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        # Player type colors
        self.player_colors = {
            'detective': {'bg': '#e3f2fd', 'fg': '#1565c0', 'accent': '#42a5f5'},
            'MrX': {'bg': '#ffebee', 'fg': '#c62828', 'accent': '#ef5350'}
        }
        
        # Transport colors and emojis
        self.transport_info = {
            1: {'emoji': '🚕', 'name': 'Taxi', 'color': '#FFD700'},
            2: {'emoji': '🚌', 'name': 'Bus', 'color': '#4169E1'},
            3: {'emoji': '🚇', 'name': 'Underground', 'color': '#DC143C'},
            4: {'emoji': '⚫', 'name': 'Black', 'color': '#2F2F2F'}
        }
        
        self._create_display_elements()
    
    def _create_display_elements(self):
        """Create the display elements"""
        # Player card frame
        self.player_card = tk.Frame(self.main_frame, relief="solid", bd=2, bg="#ffffff")
        self.player_card.pack(fill=tk.X, pady=(0, 8))
        
        # Player info frame
        self.player_info = tk.Frame(self.player_card, bg="#ffffff")
        self.player_info.pack(fill=tk.X, padx=10, pady=8)
        
        # Player name and type
        self.player_label = tk.Label(self.player_info, font=('Arial', 11, 'bold'),
                                   bg="#ffffff", fg="#333333")
        self.player_label.pack(anchor="w")
        
        # Status badge frame
        self.status_frame = tk.Frame(self.player_info, bg="#ffffff")
        self.status_frame.pack(fill=tk.X, pady=(4, 0))
        
        # Status badge
        self.status_badge = tk.Label(self.status_frame, font=('Arial', 8, 'bold'),
                                   padx=8, pady=2, relief="solid", bd=1)
        self.status_badge.pack(side=tk.LEFT)
        
        # Progress frame (for detectives)
        self.progress_frame = tk.Frame(self.main_frame, bg="#f8f9fa")
        
        # Progress label
        self.progress_label = tk.Label(self.progress_frame, font=('Arial', 9),
                                     bg="#f8f9fa", fg="#666666")
        self.progress_label.pack(anchor="w")
        
        # Progress bar frame
        self.progress_bar_frame = tk.Frame(self.progress_frame, bg="#f8f9fa")
        self.progress_bar_frame.pack(fill=tk.X, pady=(4, 0))
        
        # Transport selection frame (dynamic buttons)
        self.transport_frame = tk.Frame(self.main_frame, bg="#f8f9fa")
        self.transport_buttons = {}
        
        # Instructions
        self.instructions = tk.Label(self.main_frame, font=('Arial', 9),
                                   bg="#f8f9fa", fg="#666666", wraplength=200)
        self.instructions.pack(anchor="w", pady=(8, 0))
    
    def show_transport_selection(self, available_transports, selected_node, callback, can_use_black=False):
        """Show transport selection buttons for the selected destination"""
        # Clear existing buttons
        self.hide_transport_selection()
        
        if not available_transports and not can_use_black:
            return
        
        # Show transport frame
        self.transport_frame.pack(fill=tk.X, pady=(8, 0))
        
        # Add header label
        header_label = tk.Label(self.transport_frame, 
                               text=f"Select transport to node {selected_node}:",
                               font=('Arial', 9, 'bold'),
                               bg="#f8f9fa", fg="#333333")
        header_label.pack(anchor="w", pady=(0, 4))
        
        # Create button frame
        button_frame = tk.Frame(self.transport_frame, bg="#f8f9fa")
        button_frame.pack(fill=tk.X)
        
        # Create transport buttons
        black_ticket_included = False
        for i, transport in enumerate(available_transports):
            transport_value = transport.value if hasattr(transport, 'value') else transport
            
            # Check if this is already a black ticket
            if transport_value == 4:  # TicketType.BLACK.value
                black_ticket_included = True
            
            transport_info = self.transport_info.get(transport_value, {
                'emoji': '🎫', 'name': str(transport), 'color': '#666666'
            })
            
            # Create styled button
            btn = tk.Button(button_frame,
                          text=f"{transport_info['emoji']} {transport_info['name']}",
                          font=('Arial', 9, 'bold'),
                          bg=transport_info['color'],
                          fg='white' if transport_value != 1 else 'black',  # Taxi uses black text
                          relief="raised",
                          bd=2,
                          padx=12,
                          pady=6,
                          command=lambda t=transport: callback(t))
            
            btn.pack(side=tk.LEFT, padx=2, pady=2)
            self.transport_buttons[transport_value] = btn
        
        # Add black ticket button if available for Mr. X and not already included
        if can_use_black and not black_ticket_included:
            from ..core.game import TicketType
            black_btn = tk.Button(button_frame,
                                text="⚫ Black Ticket",
                                font=('Arial', 9, 'bold'),
                                bg='#2c2c2c',
                                fg='white',
                                relief="raised",
                                bd=2,
                                padx=12,
                                pady=6,
                                command=lambda: callback(TicketType.BLACK))
            
            black_btn.pack(side=tk.LEFT, padx=2, pady=2)
            self.transport_buttons['black'] = black_btn
    
    def hide_transport_selection(self):
        """Hide transport selection buttons"""
        # Clear all widgets in transport frame
        for widget in self.transport_frame.winfo_children():
            widget.destroy()
        self.transport_buttons.clear()
        self.transport_frame.pack_forget()
    
    def update_display(self, game_state, game, current_detective_index=0, detective_selections=None, 
                      is_ai_turn=False, double_move_requested=False):
        """Update the turn display with current game state"""
        # Hide transport selection when updating display
        self.hide_transport_selection()
        
        if not game_state:
            self._show_setup_mode()
            return
        
        detective_selections = detective_selections or []
        
        # Determine current player
        current_player = getattr(game_state, 'turn', None)
        if not current_player:
            return
        
        # Handle enum objects
        if hasattr(current_player, 'value'):
            player_name = current_player.value.lower()
        elif hasattr(current_player, 'name'):
            player_name = current_player.name.lower()
        else:
            player_name = str(current_player).lower()
        
        is_detective_turn = 'detective' in player_name or 'det' in player_name
        
        if is_detective_turn:
            self._update_detective_turn(game, current_detective_index, detective_selections, is_ai_turn)
        else:
            self._update_MrX_turn(game_state, is_ai_turn, double_move_requested)
    
    def _update_detective_turn(self, game, current_detective_index, detective_selections, is_ai_turn):
        """Update display for detective turn"""
        colors = self.player_colors['detective']
        
        # Update player card colors
        self.player_card.config(bg=colors['bg'], bd=2, highlightbackground=colors['accent'])
        self.player_info.config(bg=colors['bg'])
        
        # Player name
        if current_detective_index < game.num_detectives and not is_ai_turn:
            det_pos = getattr(game.game_state, 'detective_positions', [])
            if current_detective_index < len(det_pos):
                position = det_pos[current_detective_index]
                self.player_label.config(
                    text=f"🕵️ Detective {current_detective_index + 1} (Pos: {position})",
                    bg=colors['bg'], fg=colors['fg']
                )
            else:
                self.player_label.config(
                    text=f"🕵️ Detective {current_detective_index + 1}",
                    bg=colors['bg'], fg=colors['fg']
                )
        else:
            self.player_label.config(
                text="🕵️ Detectives",
                bg=colors['bg'], fg=colors['fg']
            )
        
        # Status badge
        if is_ai_turn:
            self._set_status_badge("🤖 AI", "#17a2b8", "#ffffff")
            self.instructions.config(text="Click 'Continue' to let AI make moves")
        else:
            self._set_status_badge("👤 Human", "#28a745", "#ffffff")
            if len(detective_selections) == game.num_detectives:
                self.instructions.config(text="✅ All detectives selected - make move")
            else:
                self.instructions.config(text="📍 Select detective moves")
        
        # Progress bar
        self._show_progress(len(detective_selections), game.num_detectives)
    
    def _update_MrX_turn(self, game_state, is_ai_turn, double_move_requested):
        """Update display for Mr. X turn"""
        colors = self.player_colors['MrX']
        
        # Update player card colors
        self.player_card.config(bg=colors['bg'], bd=2, highlightbackground=colors['accent'])
        self.player_info.config(bg=colors['bg'])
        
        # Player name with special move status
        double_status = ""
        if double_move_requested:
            double_status = " (DOUBLE MOVE)"
        elif getattr(game_state, 'double_move_active', False):
            double_status = " (SECOND MOVE)"
        
        self.player_label.config(
            text=f"🕵️‍♂️ Mr. X{double_status}",
            bg=colors['bg'], fg=colors['fg']
        )
        
        # Status badge
        if is_ai_turn:
            self._set_status_badge("🤖 AI", "#17a2b8", "#ffffff")
            self.instructions.config(text="Click 'Continue' to let AI make move")
        else:
            self._set_status_badge("👤 Human", "#28a745", "#ffffff")
            self.instructions.config(text="📍 Select new position")
        
        # Hide progress bar for Mr. X
        self.progress_frame.pack_forget()
    
    def _show_setup_mode(self):
        """Show setup mode display"""
        colors = {'bg': '#fff3cd', 'fg': '#856404'}
        
        self.player_card.config(bg=colors['bg'], bd=2, highlightbackground="#ffc107")
        self.player_info.config(bg=colors['bg'])
        
        self.player_label.config(
            text="⚙️ Setup Phase",
            bg=colors['bg'], fg=colors['fg']
        )
        
        self._set_status_badge("Setup", "#ffc107", "#000000")
        self.instructions.config(text="Click nodes to select starting positions")
        self.progress_frame.pack_forget()
    
    def _set_status_badge(self, text, bg_color, fg_color):
        """Set status badge appearance"""
        self.status_badge.config(text=text, bg=bg_color, fg=fg_color)
    
    def _show_progress(self, current, total):
        """Show progress bar for detective moves"""
        self.progress_frame.pack(fill=tk.X, pady=(8, 0))
        
        # Update progress label
        self.progress_label.config(text=f"Progress: {current}/{total} detectives")
        
        # Clear existing progress bars
        for widget in self.progress_bar_frame.winfo_children():
            widget.destroy()
        
        # Create progress visualization
        for i in range(total):
            if i < current:
                # Completed
                bar = tk.Label(self.progress_bar_frame, bg="#28a745", width=3, height=1)
            else:
                # Pending
                bar = tk.Label(self.progress_bar_frame, bg="#dee2e6", width=3, height=1)
            bar.pack(side=tk.LEFT, padx=1)


class EnhancedMovesDisplay(ttk.Frame):
    """Enhanced available moves display with better formatting"""
    
    def __init__(self, parent, title="🎯 Available Moves", **kwargs):
        super().__init__(parent, **kwargs)
        
        # Create the main container with title
        self.label_frame = ttk.LabelFrame(self, text=title)
        self.label_frame.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)
        
        # Configure label style
        style = ttk.Style()
        style.configure("MovesDisplay.TLabelframe.Label", anchor="w", font=('Arial', 10, 'bold'))
        self.label_frame.configure(style="MovesDisplay.TLabelframe")
        
        # Create scrollable text area with better styling
        self.text_frame = tk.Frame(self.label_frame)
        self.text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.text_widget = tk.Text(
            self.text_frame,
            height=6,
            wrap=tk.WORD,
            bg="#f8f9fa",
            fg="#333333",
            relief="solid",
            bd=1,
            font=('Consolas', 9),
            padx=8,
            pady=6,
            state=tk.DISABLED
        )
        
        # Scrollbar
        self.scrollbar = ttk.Scrollbar(self.text_frame, orient="vertical", command=self.text_widget.yview)
        self.text_widget.configure(yscrollcommand=self.scrollbar.set)
        
        self.text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Transport colors for highlighting
        self.transport_colors = {
            'taxi': '#FFD700',
            'bus': '#4169E1', 
            'underground': '#DC143C',
            'black': '#2F2F2F',
            'double_move': '#8A2BE2'
        }
        
        # Configure text tags for different transport types
        self._configure_text_tags()
    
    def _configure_text_tags(self):
        """Configure text tags for colored transport types"""
        for transport, color in self.transport_colors.items():
            self.text_widget.tag_configure(transport, foreground=color, font=('Consolas', 9, 'bold'))
    
    def update_moves(self, moves_text):
        """Update the moves display with formatted text"""
        self.text_widget.config(state=tk.NORMAL)
        self.text_widget.delete(1.0, tk.END)
        
        if not moves_text:
            self.text_widget.insert(tk.END, "No moves available")
        else:
            # Parse and format the moves text with colored transport types
            lines = moves_text.split('\n')
            for line in lines:
                if any(transport in line.lower() for transport in self.transport_colors.keys()):
                    # Format line with transport colors
                    self._insert_formatted_line(line)
                else:
                    self.text_widget.insert(tk.END, line + '\n')
        
        self.text_widget.config(state=tk.DISABLED)
    
    def _insert_formatted_line(self, line):
        """Insert a line with formatted transport types"""
        remaining = line
        
        while remaining:
            # Find the next transport type mention
            earliest_pos = len(remaining)
            found_transport = None
            
            for transport in self.transport_colors.keys():
                pos = remaining.lower().find(transport)
                if pos != -1 and pos < earliest_pos:
                    earliest_pos = pos
                    found_transport = transport
            
            if found_transport is None:
                # No more transport types found
                self.text_widget.insert(tk.END, remaining + '\n')
                break
            
            # Insert text before transport type
            if earliest_pos > 0:
                self.text_widget.insert(tk.END, remaining[:earliest_pos])
            
            # Insert transport type with color
            transport_end = earliest_pos + len(found_transport)
            self.text_widget.insert(tk.END, remaining[earliest_pos:transport_end], found_transport)
            
            # Continue with remaining text
            remaining = remaining[transport_end:]
