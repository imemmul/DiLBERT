import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk, filedialog
import pandas as pd
import os
import re

class HateSpeechAnnotator:
    def __init__(self, root, csv_file):
        self.root = root
        self.csv_file = csv_file
        
        # Colors for modern UI
        self.colors = {
            "bg_dark": "#2c3e50",
            "bg_light": "#ecf0f1",
            "accent_blue": "#3498db",
            "accent_green": "#2ecc71",
            "accent_red": "#e74c3c",
            "accent_purple": "#9b59b6",
            "text_dark": "#34495e",
            "text_light": "#ecf0f1",
            "btn_hover": "#2980b9"
        }
        
        # Check if file exists
        if not os.path.exists(csv_file):
            messagebox.showerror("File Not Found", f"Could not find {csv_file}. Please make sure the file exists.")
            root.destroy()
            return
        
        try:
            # Load data
            self.data = pd.read_csv(csv_file)
            
            # Check if data has required columns
            if 'text' not in self.data.columns:
                messagebox.showerror("Invalid File", "The CSV file must have a 'text' column.")
                root.destroy()
                return
                
            if 'label' not in self.data.columns:
                self.data['label'] = ''
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV file: {str(e)}")
            root.destroy()
            return
        
        # Find first unannotated entry
        self.current_index = self.find_next_unannotated()
        self.completed_count = self.count_completed()
        self.version = self.get_next_version()
        self.unsaved_changes = False
        
        # Set up the main window
        self.root.title(f"Turkish Hate Speech Annotator - {os.path.basename(self.csv_file)}")
        self.root.geometry("800x600")
        self.root.configure(bg=self.colors["bg_dark"])
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Apply a modern theme if ttk is used
        self.style = ttk.Style()
        self.style.theme_use('clam')  # Use a modern theme as base
        self.style.configure("TProgressbar", thickness=25, background=self.colors["accent_blue"])
        
        # Create menu bar with File menu
        self.menu_bar = tk.Menu(self.root)
        self.root.config(menu=self.menu_bar)
        
        self.file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="Open...", command=self.open_file)
        self.file_menu.add_command(label="Save", command=self.save_annotations)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=self.on_closing)
        
        # Bind keyboard shortcuts
        self.root.bind('h', lambda event: self.annotate("hate"))
        self.root.bind('n', lambda event: self.annotate("not-hate"))
        self.root.bind('s', lambda event: self.skip_text())
        self.root.bind('p', lambda event: self.previous_text())
        self.root.bind('<Right>', lambda event: self.next_text())
        self.root.bind('<Left>', lambda event: self.previous_text())
        self.root.bind('<Control-s>', lambda event: self.save_annotations())
        
        # Create the UI components
        self.setup_ui()
        
        # Load the first text to annotate
        self.load_next_text()
    
    def find_next_unannotated(self):
        """Find the index of the first row without a label."""
        for idx, row in self.data.iterrows():
            if pd.isna(row['label']) or row['label'] == '':
                return idx
        return 0  # Default to first item if all are annotated
    
    def count_completed(self):
        """Count the number of rows that have been annotated."""
        return self.data['label'].notna().sum() - self.data['label'].eq('').sum()
    
    def get_next_version(self):
        # Extract the base name without extension
        base_name = os.path.splitext(self.csv_file)[0]
        
        # Find all existing versions
        pattern = re.compile(f"{re.escape(base_name)}_v(\\d+)\\.csv")
        versions = []
        
        for file in os.listdir():
            match = pattern.match(file)
            if match:
                versions.append(int(match.group(1)))
        
        # Return the next version number
        return max(versions) + 1 if versions else 1
    
    def setup_ui(self):
        # Create frames for better organization with modern styling
        self.top_frame = tk.Frame(self.root, bg=self.colors["bg_dark"], pady=10)
        self.top_frame.pack(fill="x", padx=20, pady=(20, 10))
        
        self.text_frame = tk.Frame(self.root, bg=self.colors["bg_dark"], padx=15, pady=15, 
                                  highlightbackground=self.colors["accent_blue"], highlightthickness=1)
        self.text_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        self.status_frame = tk.Frame(self.root, bg=self.colors["bg_dark"])
        self.status_frame.pack(fill="x", padx=20)
        
        self.button_frame = tk.Frame(self.root, bg=self.colors["bg_dark"])
        self.button_frame.pack(fill="x", padx=20, pady=20)
        
        # Progress label
        self.progress_label = tk.Label(
            self.top_frame, 
            text=f"Annotating text {self.current_index + 1} of {len(self.data)}",
            font=("Segoe UI", 12),
            bg=self.colors["bg_dark"],
            fg=self.colors["text_light"]
        )
        self.progress_label.pack(side="left")
        
        # Completed count label
        self.completed_label = tk.Label(
            self.top_frame, 
            text=f"Completed: {self.completed_count} / {len(self.data)}",
            font=("Segoe UI", 12),
            bg=self.colors["bg_dark"],
            fg=self.colors["text_light"]
        )
        self.completed_label.pack(side="right")
        
        # Text display area (scrollable) with modern styling
        self.text_display = scrolledtext.ScrolledText(
            self.text_frame,
            wrap=tk.WORD,
            width=80,
            height=20,
            font=("Segoe UI", 11),
            bg=self.colors["bg_light"],
            fg=self.colors["text_dark"],
            padx=10,
            pady=10,
            borderwidth=0
        )
        self.text_display.pack(fill="both", expand=True)
        
        # Progress bar with modern styling
        self.progress_bar = ttk.Progressbar(
            self.status_frame, 
            orient="horizontal", 
            length=760, 
            mode="determinate",
            style="TProgressbar"
        )
        self.progress_bar.pack(pady=10)
        self.progress_bar["maximum"] = len(self.data)
        self.progress_bar["value"] = self.completed_count
        
        # Current status label
        self.status_label = tk.Label(
            self.status_frame,
            text="Current annotation: None",
            font=("Segoe UI", 10),
            bg=self.colors["bg_dark"],
            fg=self.colors["text_light"]
        )
        self.status_label.pack()
        
        # Shortcut hints
        self.shortcuts_label = tk.Label(
            self.status_frame,
            text="Shortcuts: H = Hate, N = Not Hate, S = Skip, P = Previous, ← → = Navigate, Ctrl+S = Save",
            font=("Segoe UI", 9),
            bg=self.colors["bg_dark"],
            fg=self.colors["accent_blue"]
        )
        self.shortcuts_label.pack(pady=5)
        
        # Navigation button bar
        self.nav_frame = tk.Frame(self.root, bg=self.colors["bg_dark"])
        self.nav_frame.pack(fill="x", padx=20, pady=10)
        
        # Previous button
        self.prev_button = tk.Button(
            self.nav_frame,
            text="← PREVIOUS (P)",
            command=self.previous_text,
            bg=self.colors["bg_dark"],
            fg="#FFC107",  # Yellow text
            relief=tk.FLAT,
            borderwidth=0,
            font=("Segoe UI", 10, "bold"),
            padx=10
        )
        self.prev_button.pack(side="left")
        
        # Next button
        self.next_button = tk.Button(
            self.nav_frame,
            text="NEXT (→)",
            command=self.next_text,
            bg=self.colors["bg_dark"],
            fg="#FFC107",  # Yellow text
            relief=tk.FLAT,
            borderwidth=0,
            font=("Segoe UI", 10, "bold"),
            padx=10
        )
        self.next_button.pack(side="right")
        
        # Annotation buttons with modern styling
        button_font = ("Segoe UI", 11, "bold")
        
        self.hate_button = tk.Button(
            self.button_frame,
            text="HATE (H)",
            command=lambda: self.annotate("hate"),
            bg=self.colors["accent_red"],
            fg="#FFECEC",  # Light pink text
            width=15,
            height=2,
            font=button_font,
            relief=tk.FLAT,
            borderwidth=0
        )
        self.hate_button.pack(side="left", padx=(0, 10))
        
        self.not_hate_button = tk.Button(
            self.button_frame,
            text="NOT HATE (N)",
            command=lambda: self.annotate("not-hate"),
            bg=self.colors["accent_green"],
            fg="#E8FFEF",  # Light green text
            width=15,
            height=2,
            font=button_font,
            relief=tk.FLAT,
            borderwidth=0
        )
        self.not_hate_button.pack(side="left", padx=10)
        
        self.skip_button = tk.Button(
            self.button_frame,
            text="SKIP (S)",
            command=self.skip_text,
            bg=self.colors["accent_blue"],
            fg="#E3F2FD",  # Light blue text
            width=15,
            height=2,
            font=button_font,
            relief=tk.FLAT,
            borderwidth=0
        )
        self.skip_button.pack(side="left", padx=10)
        
        self.save_button = tk.Button(
            self.button_frame,
            text="SAVE (Ctrl+S)",
            command=self.save_annotations,
            bg=self.colors["accent_purple"],
            fg="#F3E5F5",  # Light purple text
            width=15,
            height=2,
            font=button_font,
            relief=tk.FLAT,
            borderwidth=0
        )
        self.save_button.pack(side="left", padx=10)
        
        # Add hover effects to buttons
        self.add_hover_effects()
    
    def add_hover_effects(self):
        """Add hover effects to buttons for better UX"""
        def on_enter(event, button, original_color):
            # Darkens the button color slightly on hover
            button['background'] = self.darken_color(original_color)
            
        def on_leave(event, button, original_color):
            # Restores original color when mouse leaves
            button['background'] = original_color
            
        # Add effects to each button
        buttons = [
            (self.hate_button, self.colors["accent_red"]),
            (self.not_hate_button, self.colors["accent_green"]),
            (self.skip_button, self.colors["accent_blue"]),
            (self.save_button, self.colors["accent_purple"]),
            (self.prev_button, self.colors["bg_dark"]),
            (self.next_button, self.colors["bg_dark"])
        ]
        
        for button, color in buttons:
            button.bind("<Enter>", lambda event, b=button, c=color: on_enter(event, b, c))
            button.bind("<Leave>", lambda event, b=button, c=color: on_leave(event, b, c))
    
    def darken_color(self, hex_color, factor=0.8):
        """Darken a hex color by multiplying RGB values by factor"""
        # Convert hex to RGB
        h = hex_color.lstrip('#')
        rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
        
        # Darken each component
        darkened = tuple(int(c * factor) for c in rgb)
        
        # Convert back to hex
        return f'#{darkened[0]:02x}{darkened[1]:02x}{darkened[2]:02x}'
    
    def load_next_text(self):
        # If we've gone through all texts, show completion message
        if self.current_index >= len(self.data):
            messagebox.showinfo("Completed", "You've annotated all texts!")
            self.save_annotations()
            return
        
        # Get the current text and clear any previous text
        current_text = self.data.loc[self.current_index, 'text']
        current_label = self.data.loc[self.current_index, 'label']
        
        self.text_display.delete(1.0, tk.END)
        
        # Insert the text and update progress
        self.text_display.insert(tk.END, current_text)
        self.progress_label.config(text=f"Annotating text {self.current_index + 1} of {len(self.data)}")
        
        # Update status label
        if pd.notna(current_label) and current_label != '':
            label_color = self.colors["accent_red"] if current_label.lower() == "hate" else self.colors["accent_green"]
            self.status_label.config(
                text=f"Current annotation: {current_label.upper()}", 
                fg=label_color
            )
        else:
            self.status_label.config(text="Current annotation: None", fg=self.colors["text_light"])
        
        # Scroll to the top
        self.text_display.see(1.0)
    
    def annotate(self, label):
        if self.current_index < len(self.data):
            # Get current label to check if we're changing an existing annotation
            current_label = self.data.loc[self.current_index, 'label']
            is_new_annotation = pd.isna(current_label) or current_label == ''
            
            # Update the label in the dataframe
            self.data.loc[self.current_index, 'label'] = label
            self.unsaved_changes = True
            
            # Increment completed count if this is a new annotation
            if is_new_annotation:
                self.completed_count += 1
                self.completed_label.config(text=f"Completed: {self.completed_count} / {len(self.data)}")
                self.progress_bar["value"] = self.completed_count
            
            # Move to the next text
            self.current_index += 1
            self.load_next_text()
    
    def skip_text(self):
        if self.current_index < len(self.data):
            # Move to the next text without updating the label
            self.current_index += 1
            self.load_next_text()
    
    def save_annotations(self):
        # Create output filename with version
        output_file = f"{os.path.splitext(self.csv_file)[0]}_v{self.version}.csv"
        
        try:
            # Save the dataframe to CSV
            self.data.to_csv(output_file, index=False)
            
            # Show message
            messagebox.showinfo("Saved", f"Annotations saved to {output_file}")
            
            # Increment version for next save
            self.version += 1
            self.unsaved_changes = False
            
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file: {str(e)}")
            return False

    def open_file(self):
        """Open a new CSV file for annotation."""
        if self.unsaved_changes:
            response = messagebox.askyesnocancel("Save Changes", "Do you want to save your changes before opening a new file?")
            if response is None:  # Cancel
                return
            elif response:  # Yes
                if not self.save_annotations():
                    return  # If save failed, don't proceed

        # Open file dialog
        file_types = [
            ('CSV files', '*.csv'),
            ('All files', '*.*')
        ]
        
        new_file = filedialog.askopenfilename(
            title="Select CSV file to annotate",
            filetypes=file_types,
            initialdir=os.path.dirname(self.csv_file)
        )
        
        if not new_file:  # User canceled
            return
            
        # Update the application with the new file
        self.csv_file = new_file
        
        try:
            # Load new data
            self.data = pd.read_csv(new_file)
            
            # Check if data has required columns
            if 'text' not in self.data.columns:
                messagebox.showerror("Invalid File", "The CSV file must have a 'text' column.")
                return
                
            if 'label' not in self.data.columns:
                self.data['label'] = ''
                
            # Reset tracking variables
            self.current_index = self.find_next_unannotated()
            self.completed_count = self.count_completed()
            self.version = self.get_next_version()
            self.unsaved_changes = False
            
            # Update window title
            self.root.title(f"Turkish Hate Speech Annotator - {os.path.basename(self.csv_file)}")
            
            # Update UI
            self.completed_label.config(text=f"Completed: {self.completed_count} / {len(self.data)}")
            self.progress_bar["maximum"] = len(self.data)
            self.progress_bar["value"] = self.completed_count
            
            # Load the first text
            self.load_next_text()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV file: {str(e)}")
    
    def on_closing(self):
        """Handle window close event."""
        if self.unsaved_changes:
            response = messagebox.askyesnocancel("Save Changes", "Do you want to save your changes before exiting?")
            if response is None:  # Cancel
                return
            elif response:  # Yes
                if self.save_annotations():
                    self.root.destroy()
            else:  # No
                self.root.destroy()
        else:
            self.root.destroy()
    def next_text(self):
        """Move to the next text without changing the current annotation."""
        if self.current_index < len(self.data) - 1:
            # Move to the next text
            self.current_index += 1
            self.load_next_text()
        else:
            # If at the end, show a message
            messagebox.showinfo("End of Dataset", "You've reached the last text in the dataset.")
    def previous_text(self):
        """Move to the previous text for review or modification."""
        if self.current_index > 0:
            # Move to the previous text
            self.current_index -= 1
            self.load_next_text()
        else:
            # If at the beginning, show a message
            messagebox.showinfo("Beginning of Dataset", "You are at the first text in the dataset.")


def main():
    # Create the root window
    root = tk.Tk()
    root.withdraw()  # Hide the root window initially
    
    # Open file dialog to select CSV file
    file_types = [
        ('CSV files', '*.csv'),
        ('All files', '*.*')
    ]
    
    csv_file = filedialog.askopenfilename(
        title="Select CSV file to annotate",
        filetypes=file_types,
        initialdir="."
    )
    
    if not csv_file:  # User canceled selection
        messagebox.showinfo("Canceled", "No file selected. Application will exit.")
        root.destroy()
        return
    
    # Show the root window now that we have a file
    root.deiconify()
    
    # Create the annotator
    annotator = HateSpeechAnnotator(root, csv_file)
    
    # Start the Tkinter event loop
    root.mainloop()


if __name__ == "__main__":
    main()