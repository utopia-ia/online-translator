#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
History Manager Module
---------------------
Manages transcription and translation history for the Online-Translator application.
Handles storage, retrieval, and export of historical data.

Copyright (c) 2024 Kiko Cisneros
Licensed under the MIT License (see LICENSE file for details)
"""

import os
import json
import csv
from datetime import datetime
from tkinter import filedialog, messagebox
from . import config


class TranscriptionHistory:
    """Manages transcription history and export functionality"""
    
    def __init__(self):
        self.entries = []
        self.current_session_start = datetime.now()
        
    def add_entry(self, original_text, translated_text, source_lang, target_lang):
        """Add a new transcription entry"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'original_text': original_text,
            'translated_text': translated_text,
            'source_language': source_lang,
            'target_language': target_lang
        }
        self.entries.append(entry)
        
        # Auto-save every 10 entries
        if len(self.entries) % 10 == 0:
            self.auto_save()
    
    def auto_save(self):
        """Auto-save to a backup file (silent)"""
        try:
            # Create transcriptions_history folder if it doesn't exist
            backup_folder = "transcriptions_history"
            if not os.path.exists(backup_folder):
                os.makedirs(backup_folder)
            
            backup_filename = f"transcription_backup_{self.current_session_start.strftime('%Y%m%d_%H%M%S')}.json"
            backup_file = os.path.join(backup_folder, backup_filename)
            
            # Silent save without dialog
            export_data = {
                'export_date': datetime.now().isoformat(),
                'session_start': self.current_session_start.isoformat(),
                'total_entries': len(self.entries),
                'transcriptions': self.entries
            }
            
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
                
            if config.ENABLE_DEBUG_LOGGING:
                print(f"Auto-saved {len(self.entries)} entries to {backup_file}")
        except Exception as e:
            if config.ENABLE_DEBUG_LOGGING:
                print(f"Auto-save error: {e}")
    
    def export_txt(self, filename=None):
        """Export transcriptions as plain text"""
        if not filename:
            filename = filedialog.asksaveasfilename(
                title="Export as Text",
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(f"Transcription Export - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("=" * 60 + "\n\n")
                    
                    for i, entry in enumerate(self.entries, 1):
                        f.write(f"Entry {i} - {entry['timestamp']}\n")
                        f.write(f"Language: {entry['source_language']} → {entry['target_language']}\n")
                        f.write(f"Original: {entry['original_text']}\n")
                        f.write(f"Translation: {entry['translated_text']}\n")
                        f.write("-" * 40 + "\n\n")
                
                messagebox.showinfo("Export Complete", f"Exported {len(self.entries)} entries to {filename}")
                return True
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export: {e}")
                return False
    
    def export_json(self, filename=None):
        """Export transcriptions as JSON"""
        if not filename:
            filename = filedialog.asksaveasfilename(
                title="Export as JSON",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
        
        if filename:
            try:
                export_data = {
                    'export_date': datetime.now().isoformat(),
                    'session_start': self.current_session_start.isoformat(),
                    'total_entries': len(self.entries),
                    'transcriptions': self.entries
                }
                
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
                
                # Only show dialog for manual exports, not auto-save
                if filename.startswith('transcription_backup_'):
                    return True
                else:
                    messagebox.showinfo("Export Complete", f"Exported {len(self.entries)} entries to {filename}")
                    return True
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export: {e}")
                return False
    
    def export_csv(self, filename=None):
        """Export transcriptions as CSV"""
        if not filename:
            filename = filedialog.asksaveasfilename(
                title="Export as CSV",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
        
        if filename:
            try:
                with open(filename, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Timestamp', 'Source Language', 'Target Language', 'Original Text', 'Translated Text'])
                    
                    for entry in self.entries:
                        writer.writerow([
                            entry['timestamp'],
                            entry['source_language'],
                            entry['target_language'],
                            entry['original_text'],
                            entry['translated_text']
                        ])
                
                messagebox.showinfo("Export Complete", f"Exported {len(self.entries)} entries to {filename}")
                return True
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export: {e}")
                return False
    
    def clear_history(self):
        """Clear all transcription history"""
        self.entries.clear()
        self.current_session_start = datetime.now() 