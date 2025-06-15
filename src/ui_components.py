#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UI Components Module
-------------------
User interface components and widgets for the Online-Translator application.
Includes floating subtitle window, controls, and settings interface.

Copyright (c) 2025 Kiko Cisneros
Licensed under the MIT License (see LICENSE file for details)
"""

import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
from . import config


class HistoryWindow:
    """Separate window for viewing transcription history"""
    
    def __init__(self, history_manager):
        self.history_manager = history_manager
        self.window = None
        self.last_entry_count = 0
        self.auto_refresh_active = False
        
    def show(self):
        """Show the history window"""
        if self.window and self.window.winfo_exists():
            self.window.lift()
            self.window.focus()
            return
            
        self.window = tk.Toplevel()
        self.window.title("Transcription History")
        self.window.geometry("1000x600")
        self.window.configure(bg='white')
        
        # Set window close handler
        self.window.protocol("WM_DELETE_WINDOW", self.on_window_close)
        
        # Toolbar
        toolbar = tk.Frame(self.window, bg='lightgray', relief=tk.RAISED, bd=1)
        toolbar.pack(fill=tk.X, pady=(0,5))
        
        # Export buttons
        self.create_modern_button(toolbar, "📄 Export TXT", self.export_txt, 
                                bg="#4CAF50", fg="black").pack(side=tk.LEFT, padx=2, pady=2)
        self.create_modern_button(toolbar, "📊 Export CSV", self.export_csv,
                                bg="#2196F3", fg="black").pack(side=tk.LEFT, padx=2, pady=2)
        self.create_modern_button(toolbar, "🔗 Export JSON", self.export_json,
                                bg="#FF9800", fg="black").pack(side=tk.LEFT, padx=2, pady=2)
        
        tk.Frame(toolbar, width=20, bg='lightgray').pack(side=tk.LEFT)  # Spacer
        
        self.create_modern_button(toolbar, "🔄 Refresh", self.refresh_history,
                                bg="#607D8B", fg="black").pack(side=tk.LEFT, padx=2, pady=2)
        self.create_modern_button(toolbar, "🗑️ Clear All", self.clear_history,
                                bg="#F44336", fg="black").pack(side=tk.LEFT, padx=2, pady=2)
        
        # Auto-refresh toggle
        self.auto_refresh_var = tk.BooleanVar(value=True)
        auto_refresh_check = tk.Checkbutton(toolbar, text="Auto-refresh", variable=self.auto_refresh_var,
                                          bg='lightgray', fg='black', font=('Arial', 9))
        auto_refresh_check.pack(side=tk.LEFT, padx=10)
        
        # Stats
        stats_frame = tk.Frame(toolbar, bg='lightgray')
        stats_frame.pack(side=tk.RIGHT, padx=10)
        
        self.stats_label = tk.Label(stats_frame, bg='lightgray', fg='black', font=('Arial', 9))
        self.stats_label.pack()
        
        # Create notebook for different views
        self.notebook = ttk.Notebook(self.window)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=2)
        
        # Tab 1: Two-column table view
        self.table_frame = tk.Frame(self.notebook)
        self.notebook.add(self.table_frame, text="📊 Table View")
        
        # Tab 2: Transcriptions only
        self.transcription_frame = tk.Frame(self.notebook)
        self.notebook.add(self.transcription_frame, text="🎤 Transcriptions")
        
        # Tab 3: Translations only  
        self.translation_frame = tk.Frame(self.notebook)
        self.notebook.add(self.translation_frame, text="🌐 Translations")
        
        self.setup_table_view()
        self.setup_transcription_view()
        self.setup_translation_view()
        
        self.refresh_history()
        self.start_auto_refresh()
    
    def setup_table_view(self):
        """Setup the two-column table view"""
        # Create Treeview for two-column display
        columns = ('time', 'languages', 'transcription', 'translation')
        self.tree = ttk.Treeview(self.table_frame, columns=columns, show='headings', height=20)
        
        # Define column headings
        self.tree.heading('time', text='Time')
        self.tree.heading('languages', text='Languages')
        self.tree.heading('transcription', text='Original Text')
        self.tree.heading('translation', text='Translation')
        
        # Configure column widths and alignment
        self.tree.column('time', width=70, minwidth=70, anchor='center')
        self.tree.column('languages', width=90, minwidth=80, anchor='center')
        self.tree.column('transcription', width=350, minwidth=250, anchor='w')
        self.tree.column('translation', width=350, minwidth=250, anchor='w')
        
        # Scrollbars for table
        v_scrollbar_table = ttk.Scrollbar(self.table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        h_scrollbar_table = ttk.Scrollbar(self.table_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=v_scrollbar_table.set, xscrollcommand=h_scrollbar_table.set)
        
        # Pack treeview and scrollbars
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scrollbar_table.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar_table.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Style the treeview with larger row height to accommodate longer text
        style = ttk.Style()
        style.configure("Treeview", font=('Arial', 10), rowheight=60)  # Increased row height
        style.configure("Treeview.Heading", font=('Arial', 10, 'bold'))
        
        # Alternating row colors
        self.tree.tag_configure('oddrow', background='#f0f0f0')
        self.tree.tag_configure('evenrow', background='white')
        
        # Add tooltip functionality for long text
        self.create_tooltip_bindings()
    
    def create_tooltip_bindings(self):
        """Add tooltip functionality to show full text on hover"""
        def show_tooltip(event):
            # Get the item under cursor
            item = self.tree.identify('item', event.x, event.y)
            if item:
                # Get the column
                column = self.tree.identify('column', event.x, event.y)
                if column in ['#3', '#4']:  # transcription or translation columns
                    values = self.tree.item(item, 'values')
                    if column == '#3' and len(values) > 2:
                        text = values[2]  # transcription
                    elif column == '#4' and len(values) > 3:
                        text = values[3]  # translation
                    else:
                        return
                    
                    # Show tooltip if text is long
                    if len(text) > 50:
                        self.create_tooltip(event.x_root, event.y_root, text)
        
        def hide_tooltip(event):
            if hasattr(self, 'tooltip_window') and self.tooltip_window:
                self.tooltip_window.destroy()
                self.tooltip_window = None
        
        self.tree.bind('<Motion>', show_tooltip)
        self.tree.bind('<Leave>', hide_tooltip)
        self.tooltip_window = None
    
    def create_tooltip(self, x, y, text):
        """Create a tooltip window with the full text"""
        if hasattr(self, 'tooltip_window') and self.tooltip_window:
            self.tooltip_window.destroy()
        
        self.tooltip_window = tk.Toplevel(self.window)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{x+10}+{y+10}")
        
        # Create wrapped text label
        label = tk.Label(self.tooltip_window, text=text, justify='left',
                        background='#ffffe0', relief='solid', borderwidth=1,
                        font=('Arial', 9), wraplength=400, padx=5, pady=3)
        label.pack()
        
        # Auto-hide after 3 seconds
        self.window.after(3000, lambda: self.hide_tooltip_delayed())
    
    def hide_tooltip_delayed(self):
        """Hide tooltip after delay"""
        if hasattr(self, 'tooltip_window') and self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None
    
    def setup_transcription_view(self):
        """Setup the transcriptions-only text view"""
        self.transcription_text = tk.Text(self.transcription_frame, wrap=tk.WORD, font=('Arial', 11))
        transcription_scrollbar = ttk.Scrollbar(self.transcription_frame, orient=tk.VERTICAL, 
                                              command=self.transcription_text.yview)
        self.transcription_text.configure(yscrollcommand=transcription_scrollbar.set)
        
        self.transcription_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        transcription_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Configure text tags for styling
        self.transcription_text.tag_configure("timestamp", foreground="#666666", font=('Arial', 9))
        self.transcription_text.tag_configure("transcription", foreground="#000000", font=('Arial', 11))
        self.transcription_text.tag_configure("separator", foreground="#CCCCCC")
    
    def setup_translation_view(self):
        """Setup the translations-only text view"""
        self.translation_text = tk.Text(self.translation_frame, wrap=tk.WORD, font=('Arial', 11))
        translation_scrollbar = ttk.Scrollbar(self.translation_frame, orient=tk.VERTICAL, 
                                            command=self.translation_text.yview)
        self.translation_text.configure(yscrollcommand=translation_scrollbar.set)
        
        self.translation_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        translation_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Configure text tags for styling
        self.translation_text.tag_configure("timestamp", foreground="#666666", font=('Arial', 9))
        self.translation_text.tag_configure("translation", foreground="#0066CC", font=('Arial', 11, 'bold'))
        self.translation_text.tag_configure("separator", foreground="#CCCCCC")
    
    def start_auto_refresh(self):
        """Start auto-refresh timer"""
        self.auto_refresh_active = True
        self.check_for_updates()
        
    def check_for_updates(self):
        """Check for new entries and refresh if needed"""
        if self.window and self.window.winfo_exists() and self.auto_refresh_active:
            current_count = len(self.history_manager.entries)
            if current_count != self.last_entry_count and self.auto_refresh_var.get():
                self.refresh_history()
                self.last_entry_count = current_count
            
            # Schedule next check
            self.window.after(1000, self.check_for_updates)  # Check every second
    
    def on_window_close(self):
        """Handle window close event"""
        self.auto_refresh_active = False
        self.window.destroy()
        
    def refresh_history(self):
        """Refresh the history display in all views"""
        # Clear existing entries from all views
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.transcription_text.delete(1.0, tk.END)
        self.translation_text.delete(1.0, tk.END)
        
        if not self.history_manager.entries:
            # Empty state for all views
            self.tree.insert('', 'end', values=("", "", "No transcriptions yet.", "Start recording to see your transcriptions here!"))
            self.transcription_text.insert(tk.END, "No transcriptions yet.\nStart recording to see your transcriptions here!")
            self.translation_text.insert(tk.END, "No translations yet.\nStart recording to see your translations here!")
            self.stats_label.config(text="0 entries")
            return
            
        # Update stats
        total_entries = len(self.history_manager.entries)
        total_words = sum(len(entry['original_text'].split()) + len(entry['translated_text'].split()) 
                         for entry in self.history_manager.entries)
        self.stats_label.config(text=f"{total_entries} entries • {total_words} words")
        
        # Populate all views
        for i, entry in enumerate(self.history_manager.entries, 1):
            timestamp = datetime.fromisoformat(entry['timestamp']).strftime('%H:%M:%S')
            lang_info = f"{entry['source_language']} → {entry['target_language']}"
            
            # Table view (now shows full text without truncation)
            original_text = entry['original_text']
            translated_text = entry['translated_text']
            
            # Apply text wrapping for better display in table cells
            original_text = self.wrap_text_for_table(original_text, 45)
            translated_text = self.wrap_text_for_table(translated_text, 45)
            
            tag = 'evenrow' if i % 2 == 0 else 'oddrow'
            self.tree.insert('', 'end', values=(timestamp, lang_info, original_text, translated_text), tags=(tag,))
            
            # Transcriptions-only view
            self.transcription_text.insert(tk.END, f"#{i} - {timestamp} ({lang_info})\n", "timestamp")
            self.transcription_text.insert(tk.END, f"{entry['original_text']}\n", "transcription")
            self.transcription_text.insert(tk.END, "─" * 60 + "\n\n", "separator")
            
            # Translations-only view
            self.translation_text.insert(tk.END, f"#{i} - {timestamp} ({lang_info})\n", "timestamp")
            self.translation_text.insert(tk.END, f"{entry['translated_text']}\n", "translation")
            self.translation_text.insert(tk.END, "─" * 60 + "\n\n", "separator")
        
        # Auto-scroll all views to bottom (most recent)
        if self.tree.get_children():
            last_item = self.tree.get_children()[-1]
            self.tree.see(last_item)
        
        self.transcription_text.see(tk.END)
        self.translation_text.see(tk.END)
        
        self.last_entry_count = len(self.history_manager.entries)
    
    def create_modern_button(self, parent, text, command, bg="#4CAF50", fg="black", width=None):
        """Create a modern-styled button"""
        button = tk.Button(
            parent, 
            text=text, 
            command=command,
            bg=bg,
            fg=fg,
            font=('Arial', 9, 'bold'),
            relief=tk.FLAT,
            cursor="hand2",
            pady=4,
            padx=8,
            borderwidth=0
        )
        if width:
            button.config(width=width)
        
        # Hover effects
        def on_enter(e):
            button.config(bg=self.lighten_color(bg))
        def on_leave(e):
            button.config(bg=bg)
            
        button.bind("<Enter>", on_enter)
        button.bind("<Leave>", on_leave)
        
        return button
    
    def lighten_color(self, color):
        """Lighten a hex color for hover effect"""
        if color.startswith('#'):
            color_map = {
                "#66BB6A": "#81C784",  # Light Green
                "#FF6B6B": "#FF8A80",  # Light Red
                "#4FC3F7": "#81D4FA",  # Light Blue
                "#BA68C8": "#CE93D8",  # Light Purple
                "#90A4AE": "#B0BEC5",  # Light Blue Grey
                "#B0BEC5": "#CFD8DC",  # Lighter Grey
                "#757575": "#9E9E9E",  # Light Grey
                "#4CAF50": "#66BB6A",  # Green (original)
                "#2196F3": "#42A5F5",  # Blue (original)
                "#FF9800": "#FFB74D",  # Orange (original)
                "#607D8B": "#78909C",  # Blue Grey (original)
                "#F44336": "#EF5350"   # Red (original)
            }
            return color_map.get(color, color)
        return color
        
    def export_txt(self):
        self.history_manager.export_txt()
        self.refresh_history()
        
    def export_csv(self):
        self.history_manager.export_csv()
        self.refresh_history()
        
    def export_json(self):
        self.history_manager.export_json()
        self.refresh_history()
        
    def clear_history(self):
        if messagebox.askyesno("Clear History", "Are you sure you want to clear all transcription history?"):
            self.history_manager.clear_history()
            self.refresh_history()
    
    def wrap_text_for_table(self, text, max_length):
        """Wrap text to fit within a specified maximum length"""
        words = text.split()
        wrapped_text = []
        current_line = ""
        
        for word in words:
            if len(current_line) + len(word) + 1 > max_length:
                wrapped_text.append(current_line.strip())
                current_line = word
            else:
                if current_line:
                    current_line += " " + word
                else:
                    current_line = word
        
        if current_line:
            wrapped_text.append(current_line.strip())
        
        return "\n".join(wrapped_text)


def create_modern_button(parent, text, command, bg="#4CAF50", fg="black", width=None):
    """Create a modern-styled button with hover effects"""
    
    def lighten_color(color):
        """Lighten a hex color"""
        color = color.lstrip('#')
        rgb = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
        lightened = tuple(min(255, int(c * 1.2)) for c in rgb)
        return f"#{lightened[0]:02x}{lightened[1]:02x}{lightened[2]:02x}"
    
    button = tk.Button(
        parent, 
        text=text, 
        command=command,
        bg=bg,
        fg=fg,
        activebackground=lighten_color(bg),
        activeforeground=fg,
        relief=tk.FLAT,
        font=('Arial', 9, 'bold'),
        cursor='hand2',
        bd=0,
        padx=10,
        pady=5
    )
    
    if width:
        button.config(width=width)
    
    def on_enter(e):
        button.config(bg=lighten_color(bg))
    
    def on_leave(e):
        button.config(bg=bg)
        
    button.bind("<Enter>", on_enter)
    button.bind("<Leave>", on_leave)
    
    return button 


def create_audio_device_selector(parent, audio_transcriber, on_device_change=None):
    """Create an audio device selection dropdown (compact version)"""
    frame = tk.Frame(parent, bg=parent.cget('bg'))
    
    # Compact label
    label = tk.Label(frame, text="🎤", 
                    bg=frame.cget('bg'), fg='black', 
                    font=('Arial', 9, 'bold'))
    label.pack(side=tk.LEFT, padx=(0, 2))
    
    # Get available devices
    devices = audio_transcriber.get_audio_devices()
    device_names = [device['name'] for device in devices]
    
    # Create device type indicators (more compact)
    device_display_names = []
    for device in devices:
        if device['is_microphone']:
            device_display_names.append(f"🎤 {device['name'][:15]}...")  # Truncate long names
        elif device['is_system_audio']:
            device_display_names.append(f"🔊 {device['name'][:15]}...")
        else:
            device_display_names.append(f"🎧 {device['name'][:15]}...")
    
    # Compact dropdown
    device_var = tk.StringVar()
    
    # Set current selection
    current_device = audio_transcriber.selected_device
    if current_device in device_names:
        idx = device_names.index(current_device)
        device_var.set(device_display_names[idx])
    elif device_display_names:
        device_var.set(device_display_names[0])
    
    device_dropdown = ttk.Combobox(frame, textvariable=device_var, 
                                  values=device_display_names,
                                  state="readonly", width=20,  # More compact width
                                  font=('Arial', 8))
    device_dropdown.pack(side=tk.LEFT, padx=(0, 5))
    
    def on_device_selected(event=None):
        """Handle device selection"""
        selected_display = device_var.get()
        
        # Find the actual device name
        for i, display_name in enumerate(device_display_names):
            if display_name == selected_display:
                actual_device_name = device_names[i]
                audio_transcriber.set_audio_device(actual_device_name)
                
                if on_device_change:
                    on_device_change(actual_device_name)
                break
    
    device_dropdown.bind('<<ComboboxSelected>>', on_device_selected)
    
    # Compact refresh button
    def refresh_devices():
        """Refresh the device list"""
        nonlocal devices, device_names, device_display_names
        
        devices = audio_transcriber.get_audio_devices()
        device_names = [device['name'] for device in devices]
        
        device_display_names = []
        for device in devices:
            if device['is_microphone']:
                device_display_names.append(f"🎤 {device['name'][:15]}...")
            elif device['is_system_audio']:
                device_display_names.append(f"🔊 {device['name'][:15]}...")
            else:
                device_display_names.append(f"🎧 {device['name'][:15]}...")
        
        device_dropdown['values'] = device_display_names
        
        # Try to maintain current selection or select first device
        current_device = audio_transcriber.selected_device
        if current_device in device_names:
            idx = device_names.index(current_device)
            device_var.set(device_display_names[idx])
        elif device_display_names:
            device_var.set(device_display_names[0])
            on_device_selected()  # Auto-select first device
    
    refresh_btn = tk.Button(frame, text="🔄", command=refresh_devices,
                           bg="#607D8B", fg="white", relief=tk.FLAT,
                           font=('Arial', 7, 'bold'), cursor='hand2',
                           bd=0, padx=3, pady=1)  # More compact
    refresh_btn.pack(side=tk.LEFT)
    
    return frame, device_var, refresh_devices 