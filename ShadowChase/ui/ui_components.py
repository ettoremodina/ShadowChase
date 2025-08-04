import tkinter as tk
from tkinter import ttk

# Import enhanced components
from .enhanced_components import VisualTicketDisplay, EnhancedTurnDisplay, EnhancedMovesDisplay

class ScrollableFrame(ttk.Frame):
    """A scrollable frame widget"""
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        
        # Create a canvas and scrollbar
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0, bg="#f0f0f0")
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        # Configure canvas to work with scrollable frame
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        # Create a window within the canvas to hold the scrollable frame
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        
        # Configure canvas to resize with window
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        
        # Add mousewheel scrolling
        self.bind_all("<MouseWheel>", self._on_mousewheel)
        
        # Pack canvas and scrollbar
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # Connect scrollbar to canvas
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
    
    def on_canvas_configure(self, event):
        """Resize the scrollable frame when the canvas is resized"""
        self.canvas.itemconfig(self.canvas_window, width=event.width - 20)  # Account for scrollbar
    
    def _on_mousewheel(self, event):
        """Handle mousewheel scrolling"""
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

class StyledButton(ttk.Button):
    """Enhanced button with custom styling"""
    def __init__(self, parent, text, command=None, style_type="default", **kwargs):
        self.style_type = style_type
        super().__init__(parent, text=text, command=command, **kwargs)
        self.apply_style()
    
    def apply_style(self):
        """Apply custom styling based on button type"""
        style = ttk.Style()
        
        if self.style_type == "primary":
            style.configure("Primary.TButton", 
                          background="#007acc", foreground="white", 
                          padding=(10, 6), font=('Arial', 9, 'bold'))
            self.configure(style="Primary.TButton")
        elif self.style_type == "success":
            style.configure("Success.TButton", 
                          background="#28a745", foreground="white", 
                          padding=(10, 6), font=('Arial', 9))
            self.configure(style="Success.TButton")
        elif self.style_type == "warning":
            style.configure("Warning.TButton", 
                          background="#ffc107", foreground="black", 
                          padding=(10, 6), font=('Arial', 9))
            self.configure(style="Warning.TButton")
        elif self.style_type == "danger":
            style.configure("Danger.TButton", 
                          background="#dc3545", foreground="white", 
                          padding=(10, 6), font=('Arial', 9))
            self.configure(style="Danger.TButton")

class InfoDisplay(ttk.Frame):
    """Custom frame for displaying game information"""
    def __init__(self, parent, title, height=6, **kwargs):
        super().__init__(parent, **kwargs)
        
        # Create the label frame with left-aligned text
        self.label_frame = ttk.LabelFrame(self, text=title)
        self.label_frame.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)
        
        # Configure label style for left alignment
        style = ttk.Style()
        style.configure("LeftAlign.TLabelframe.Label", anchor="w", font=('Arial', 10, 'bold'))
        self.label_frame.configure(style="LeftAlign.TLabelframe")
        
        # Create text widget with improved styling
        self.text_widget = tk.Text(
            self.label_frame, 
            height=height, 
            wrap=tk.WORD,
            bg="#f8f9fa",
            fg="#333333", 
            relief="solid",
            bd=1,
            font=('Consolas', 9),
            padx=8,
            pady=6
        )
        
        # Add scrollbar for text widget
        self.scrollbar = ttk.Scrollbar(self.label_frame, orient="vertical", command=self.text_widget.yview)
        self.text_widget.configure(yscrollcommand=self.scrollbar.set)
        
        # Pack text widget and scrollbar
        self.text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=5)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 5), pady=5)
        
        # Add mouse wheel scrolling support
        self.text_widget.bind("<MouseWheel>", self._on_mousewheel)
    
    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling"""
        self.text_widget.yview_scroll(int(-1*(event.delta/120)), "units")
    
    def clear(self):
        """Clear the text widget"""
        self.text_widget.delete(1.0, tk.END)
    
    def insert(self, text):
        """Insert text into the widget"""
        self.text_widget.insert(tk.END, text)
    
    def set_text(self, text):
        """Set the complete text content"""
        self.clear()
        self.insert(text)
