#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Online-Translator Application Core
---------------------------------
Main application module that coordinates all components and handles the core
functionality of the Online-Translator application.

Copyright (c) 2024 Kiko Cisneros
Licensed under the MIT License (see LICENSE file for details)
"""

import threading
import queue
import time
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
import sys
import signal
import atexit
from collections import deque
import numpy as np

# Import configuration
try:
    from . import config
except ImportError:
    print("Warning: config.py not found, using default settings")
    sys.exit(1)

# Import our modules
from .history_manager import TranscriptionHistory
from .audio_transcriber import RealTimeTranscriber
from .translator import TextTranslator
from .ui_components import HistoryWindow, create_modern_button, create_audio_device_selector

# MLX imports
try:
    import mlx.core as mx
except ImportError as e:
    print(f"Error importing MLX: {e}")
    print("Please install MLX dependencies: pip install mlx mlx-whisper mlx-lm")
    sys.exit(1)


class FloatingSubtitles:
    """Main application class for Online-Translator floating subtitles"""
    
    def __init__(self):
        self.root = tk.Tk()
        
        # Language settings - Initialize before UI setup
        self.source_language = tk.StringVar(value=config.DEFAULT_SOURCE_LANGUAGE)
        self.target_language = tk.StringVar(value=config.DEFAULT_TARGET_LANGUAGE)
        
        print("🔧 Initializing Online-Translator components...")
        
        # Initialize components
        self.history_manager = TranscriptionHistory()
        print("✅ History manager initialized")
        
        self.history_window = HistoryWindow(self.history_manager)
        print("✅ History window initialized")
        
        self.translator = TextTranslator()
        print("✅ Translator initialized")
        
        # Initialize real-time transcriber with integrated LLM validation
        print("🎤 Initializing real-time transcriber with LLM validation...")
        print("⚡ Fast startup mode - LLM will load on first transcription")
        try:
            self.transcriber = RealTimeTranscriber(
                sentence_callback=self.on_sentence_complete,
                accumulated_text_callback=self.on_accumulated_text_update,
                preload_llm=False
            )
            print("✅ Real-time transcriber initialized - ready to start!")
        except Exception as e:
            print(f"❌ Error initializing real-time transcriber: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Variables para mantener historial reciente en pantalla
        self.recent_transcriptions = []  # Últimas 3-4 transcripciones
        self.recent_translations = []    # Últimas 3-4 traducciones
        self.max_recent_items = 3        # Máximo de elementos recientes a mostrar
        self.current_accumulated_text = ""  # Current text being accumulated
        
        # Workflow timing statistics
        self.workflow_times = deque(maxlen=20)  # End-to-end times
        self.workflow_start_times = {}  # Track start times for each text
        self.total_workflows = 0
        
        print("🖼️  Setting up window...")
        self.setup_window()
        print("✅ Window setup complete")
        
        print("🎨 Setting up UI...")
        self.setup_ui()
        print("✅ UI setup complete")
        
        # Transcription variables
        self.is_running = False
        
        print("🚀 Online-Translator initialization complete!")
        
        # Auto-start transcription after UI is ready
        self.root.after(100, self.start_transcription)
        
    def setup_window(self):
        """Configure the floating window"""
        self.root.title("Online-Translator - Live")
        self.root.attributes('-topmost', True)  # Always on top
        self.root.attributes('-alpha', config.WINDOW_OPACITY)  # Semi-transparent
        
        # Remove window decorations for a cleaner look
        self.root.overrideredirect(True)
        
        # Position at bottom of screen
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        window_width = int(screen_width * config.WINDOW_WIDTH_RATIO)
        window_height = config.WINDOW_HEIGHT
        x = (screen_width - window_width) // 2
        y = screen_height - window_height - config.WINDOW_POSITION_Y_OFFSET
        
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.root.configure(bg=config.BACKGROUND_COLOR)
        
    def setup_ui(self):
        """Setup the user interface"""
        print("  📱 Creating main frame...")
        # Main frame
        main_frame = tk.Frame(self.root, bg=config.BACKGROUND_COLOR)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        print("  📺 Creating subtitle frame...")
        # Subtitle display en la parte superior
        subtitle_frame = tk.Frame(main_frame, bg=config.BACKGROUND_COLOR)
        subtitle_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8,2))
        
        # Audio level meter (hidden by default)
        self.meter_frame = tk.Frame(subtitle_frame, bg=config.BACKGROUND_COLOR)
        self.audio_meter_visible = False  # Hidden by default
        
        tk.Label(self.meter_frame, text="Audio Level:", fg=config.LABEL_COLOR, bg=config.BACKGROUND_COLOR,
               font=config.LABEL_FONT).pack(side=tk.LEFT)
        
        # Volume meter canvas
        self.volume_canvas = tk.Canvas(self.meter_frame, height=20, bg='black', highlightthickness=1, 
                                     highlightbackground='gray')
        self.volume_canvas.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10,0))
        
        # Volume level text
        self.volume_text = tk.Label(self.meter_frame, text="0.0%", fg=config.LABEL_COLOR, bg=config.BACKGROUND_COLOR,
                                  font=config.LABEL_FONT, width=6)
        self.volume_text.pack(side=tk.LEFT, padx=(5,0))
        
        # Crear contenedor de dos columnas para subtítulos
        columns_frame = tk.Frame(subtitle_frame, bg=config.BACKGROUND_COLOR)
        columns_frame.pack(fill=tk.BOTH, expand=True, pady=(5,0))
        
        # Calcular ancho disponible para las columnas
        screen_width = self.root.winfo_screenwidth()
        window_width = int(screen_width * config.WINDOW_WIDTH_RATIO)
        available_width = window_width - 24
        column_width = available_width // 2
        
        # Columna izquierda - Solo texto transcrito
        left_column = tk.Frame(columns_frame, bg=config.BACKGROUND_COLOR, relief=tk.SOLID, bd=1, width=column_width)
        left_column.pack(side=tk.LEFT, fill=tk.Y, padx=(0,2))
        left_column.pack_propagate(False)
        
        # Texto transcrito actual
        self.original_text = tk.Label(left_column, text="", fg=config.ORIGINAL_TEXT_COLOR, bg=config.BACKGROUND_COLOR,
                                    font=config.ORIGINAL_TEXT_FONT, wraplength=column_width-16, justify=tk.LEFT, anchor='nw')
        self.original_text.pack(fill=tk.BOTH, expand=True, padx=2, pady=(0,2), anchor='w')
        
        # Columna derecha - Solo texto traducido
        right_column = tk.Frame(columns_frame, bg=config.BACKGROUND_COLOR, relief=tk.SOLID, bd=1, width=column_width)
        right_column.pack(side=tk.LEFT, fill=tk.Y, padx=(2,0))
        right_column.pack_propagate(False)
        
        # Texto traducido actual
        self.translated_text = tk.Label(right_column, text="", fg=config.TRANSLATED_TEXT_COLOR, bg=config.BACKGROUND_COLOR,
                                      font=config.TRANSLATED_TEXT_FONT, wraplength=column_width-16, justify=tk.LEFT, anchor='nw')
        self.translated_text.pack(fill=tk.BOTH, expand=True, padx=2, pady=(0,2), anchor='w')
        
        print("  🎛️  Creating footer control bar...")
        # Barra de footer más estrecha
        footer_frame = tk.Frame(main_frame, bg='#2b2b2b', relief=tk.RAISED, bd=1, height=35)
        footer_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=0, pady=0)
        footer_frame.pack_propagate(False)
        
        # Contenedor interior con padding reducido
        footer_content = tk.Frame(footer_frame, bg='#2b2b2b')
        footer_content.pack(fill=tk.X, padx=6, pady=3)
        
        print("  🎤 Creating audio device selector...")
        # Audio device selector
        try:
            self.audio_device_frame, self.device_var, self.refresh_devices = create_audio_device_selector(
                footer_content, self.transcriber, on_device_change=self.on_audio_device_change
            )
            self.audio_device_frame.pack(side=tk.LEFT, padx=(0, 8))
            print("  ✅ Audio device selector created")
        except Exception as e:
            print(f"  ❌ Error creating audio device selector: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        print("  🌐 Creating language selection...")
        # Language selection
        lang_frame = tk.Frame(footer_content, bg='#2b2b2b')
        lang_frame.pack(side=tk.LEFT, padx=(0, 8))
        
        # Crear listas de nombres de idiomas para mostrar en UI
        source_language_names = [config.LANGUAGE_NAMES.get(code, code) for code in config.SUPPORTED_SOURCE_LANGUAGES]
        target_language_names = ["Desactivado"] + [config.LANGUAGE_NAMES.get(code, code) for code in config.SUPPORTED_TARGET_LANGUAGES]
        
        tk.Label(lang_frame, text="🎤", fg=config.LABEL_COLOR, bg='#2b2b2b', font=config.LABEL_FONT).pack(side=tk.LEFT)
        self.source_combo = ttk.Combobox(lang_frame, values=source_language_names, width=8, state="readonly")
        initial_source_name = config.LANGUAGE_NAMES.get(config.DEFAULT_SOURCE_LANGUAGE, config.DEFAULT_SOURCE_LANGUAGE)
        self.source_combo.set(initial_source_name)
        self.source_combo.pack(side=tk.LEFT, padx=(2,5))
        
        tk.Label(lang_frame, text="🌐", fg=config.LABEL_COLOR, bg='#2b2b2b', font=config.LABEL_FONT).pack(side=tk.LEFT)
        self.target_combo = ttk.Combobox(lang_frame, values=target_language_names, width=8, state="readonly")
        initial_target_name = config.LANGUAGE_NAMES.get(config.DEFAULT_TARGET_LANGUAGE, config.DEFAULT_TARGET_LANGUAGE)
        self.target_combo.set(initial_target_name)
        self.target_combo.pack(side=tk.LEFT, padx=(2,0))
        
        # Bind de eventos para actualizar variables internas cuando cambian los combos
        self.source_combo.bind('<<ComboboxSelected>>', self.on_source_language_change)
        self.target_combo.bind('<<ComboboxSelected>>', self.on_target_language_change)
        
        # Spacer para centrar botones principales
        spacer = tk.Frame(footer_content, bg='#2b2b2b')
        spacer.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        print("  🎛️  Creating main control buttons...")
        # Botones principales centrados
        main_controls_frame = tk.Frame(footer_content, bg='#2b2b2b')
        main_controls_frame.pack(side=tk.LEFT)
        
        try:
            self.play_pause_button = create_modern_button(main_controls_frame, "⏸️", self.toggle_transcription, bg='#66BB6A', fg='black')
            self.play_pause_button.pack(side=tk.LEFT, padx=2)
            print("    ✅ Play/pause button created")
            
            create_modern_button(main_controls_frame, "📋", self.show_history, bg='#BA68C8', fg='white').pack(side=tk.LEFT, padx=2)
            print("    ✅ History button created")
            
            # Audio meter toggle button
            self.audio_meter_button = create_modern_button(main_controls_frame, "📊", self.toggle_audio_meter, bg='#90A4AE', fg='black')
            self.audio_meter_button.pack(side=tk.LEFT, padx=2)
            print("    ✅ Audio meter button created")
            
        except Exception as e:
            print(f"    ❌ Error creating main control buttons: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Spacer para empujar botones de utilidad a la derecha
        spacer2 = tk.Frame(footer_content, bg='#2b2b2b')
        spacer2.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        print("  🔧 Creating utility buttons...")
        # Botones de utilidad a la derecha
        utility_controls_frame = tk.Frame(footer_content, bg='#2b2b2b')
        utility_controls_frame.pack(side=tk.RIGHT)
        
        try:
            create_modern_button(utility_controls_frame, "👁️", self.toggle_controls, bg='#B0BEC5', fg='black').pack(side=tk.LEFT, padx=2)
            print("    ✅ Toggle controls button created")
            
            create_modern_button(utility_controls_frame, "❌", self.quit_app, bg='#757575', fg='white').pack(side=tk.LEFT, padx=2)
            print("    ✅ Quit button created")
            
        except Exception as e:
            print(f"    ❌ Error creating utility buttons: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Control frame reference para toggle
        self.control_frame = footer_frame
        self.controls_visible = True
        
        print("🖱️  Binding events...")
        # Bind events
        self.root.bind('<Double-Button-1>', lambda e: self.toggle_controls())
        self.root.bind('<Button-1>', self.start_drag)
        self.root.bind('<B1-Motion>', self.drag_window)
        
        print("  ✅ UI setup complete - all components created")
        
    def on_audio_device_change(self, device_name):
        """Handle audio device change"""
        # Restart recording if currently active
        if self.is_running:
            self.stop_transcription()
            self.root.after(500, self.start_transcription)  # Small delay before restart
        
        print(f"🔄 Audio device changed to: {device_name}")
        
    def toggle_audio_meter(self):
        """Toggle audio meter visibility"""
        if self.audio_meter_visible:
            self.meter_frame.pack_forget()
            self.audio_meter_button.config(text="📊")
        else:
            # Buscar el subtitle_frame para insertar el meter
            subtitle_frame = self.original_text.master.master.master  # columna -> columns_frame -> subtitle_frame
            self.meter_frame.pack(fill=tk.X, pady=(0,10), in_=subtitle_frame, before=subtitle_frame.winfo_children()[1])
            self.audio_meter_button.config(text="📊 Hide")
        self.audio_meter_visible = not self.audio_meter_visible
        
    def show_history(self):
        """Show the transcription history window"""
        self.history_window.show()
        
    def start_drag(self, event):
        """Start dragging the window"""
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        
    def drag_window(self, event):
        """Drag the window"""
        x = self.root.winfo_x() + event.x - self.drag_start_x
        y = self.root.winfo_y() + event.y - self.drag_start_y
        self.root.geometry(f"+{x}+{y}")
        
    def toggle_controls(self):
        """Toggle control panel visibility"""
        if self.controls_visible:
            self.control_frame.pack_forget()
            # Ajustar altura cuando se ocultan los controles (menos la altura del footer estrecho)
            self.root.geometry(f"{self.root.winfo_width()}x{config.WINDOW_HEIGHT - 35}")
        else:
            self.control_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=0, pady=0)
            # Restaurar altura completa
            self.root.geometry(f"{self.root.winfo_width()}x{config.WINDOW_HEIGHT}")
        self.controls_visible = not self.controls_visible
        
    def toggle_transcription(self):
        """Toggle between start and stop transcription"""
        if self.is_running:
            self.stop_transcription()
        else:
            self.start_transcription()
    
    def start_transcription(self):
        """Start real-time transcription with continuous buffer processing"""
        print("🎯 Starting real-time transcription with LLM validation...")
        
        if not self.is_running:
            self.is_running = True
            
            try:
                # Start new continuous transcription system
                print("🧵 Starting continuous transcription (LLM will load on first use)...")
                self.transcriber.start_continuous_transcription()
                print("✅ Continuous transcription started - ready to process audio!")
                
                # Start volume meter updates (using the new transcriber's audio buffer)
                self.root.after(100, self.update_volume_meter)
                
                # Update button to show pause icon
                self.play_pause_button.config(text="⏸️", bg='#FF6B6B')
                
                print("🎉 Real-time transcription active! First LLM validation may take a few seconds.")
                
            except Exception as e:
                print(f"❌ Error during transcription startup: {e}")
                import traceback
                traceback.print_exc()
                self.is_running = False
    
    def on_sentence_complete(self, complete_sentence):
        """Callback when a complete sentence is ready (after LLM validation)"""
        # Record workflow start time
        workflow_start_time = time.time()
        self.workflow_start_times[complete_sentence] = workflow_start_time
        
        print(f"✅ LLM-validated sentence received: '{complete_sentence[:30]}...'")
        
        # Clear the current accumulated text since it's now a complete sentence
        self.current_accumulated_text = ""
        
        # Add new transcription to recent history
        self.recent_transcriptions.append(complete_sentence)
        if len(self.recent_transcriptions) > self.max_recent_items:
            self.recent_transcriptions.pop(0)
        
        # Update transcription area (left)
        def update_original():
            try:
                if hasattr(self, 'original_text') and self.original_text.winfo_exists():
                    display_text = "\n\n".join(self.recent_transcriptions)
                    self.original_text.config(text=display_text)
            except tk.TclError:
                pass
        
        self.root.after(0, update_original)
        
        # Translation mode
        target_lang = self.target_language.get()
        
        if target_lang == "Desactivate":
            # Translation disabled
            processed_text = "[Translation disabled]"
            target_lang_code = "disabled"
        else:
            # Get previous message as context if available
            previous_transcription = None
            if len(self.recent_transcriptions) > 1:
                previous_transcription = self.recent_transcriptions[-2]
                print(f"🔗 Using context from previous message: '{previous_transcription[:20]}...'")
            
            # Perform translation with context
            processed_text = self.translator.translate_text(
                complete_sentence, 
                target_lang, 
                previous_text=previous_transcription
            )
            target_lang_code = target_lang
            
            print(f"✅ Translation: '{complete_sentence[:30]}...' -> '{processed_text[:30]}...'")
        
        # Save to history
        self.history_manager.add_entry(
            original_text=complete_sentence,
            translated_text=processed_text,
            source_lang=self.source_language.get(),
            target_lang=target_lang_code
        )
        
        # Add processed text to recent translations
        self.recent_translations.append(processed_text)
        if len(self.recent_translations) > self.max_recent_items:
            self.recent_translations.pop(0)
        
        # Update translation area
        def update_translation():
            try:
                if hasattr(self, 'translated_text') and self.translated_text.winfo_exists():
                    display_text = "\n\n".join(self.recent_translations)
                    self.translated_text.config(text=display_text)
            except tk.TclError:
                pass
        
        self.root.after(0, update_translation)
        
        # Calculate workflow timing
        if complete_sentence in self.workflow_start_times:
            workflow_time = time.time() - self.workflow_start_times[complete_sentence]
            self.workflow_times.append(workflow_time)
            self.total_workflows += 1
            avg_workflow_time = sum(self.workflow_times) / len(self.workflow_times)
            
            print(f"⏱️  Translation workflow time: {workflow_time:.2f}s | Avg: {avg_workflow_time:.2f}s | Total: {self.total_workflows}")
            
            # Clean up timing record
            del self.workflow_start_times[complete_sentence]
    
    def stop_transcription(self):
        """Stop transcription service - MEJORADO"""
        try:
            if config.ENABLE_DEBUG_LOGGING:
                print("Stopping transcription service...")
            
            # Stop the transcriber
            if hasattr(self, 'transcriber'):
                # Si estamos cerrando la app, usar método rápido
                if hasattr(self, '_quitting') and self._quitting:
                    print("🛑 Quick shutdown mode")
                    self.transcriber._shutdown_requested = True
                
                self.transcriber.stop_continuous_transcription()
                
            if config.ENABLE_DEBUG_LOGGING:
                print("Transcription service stopped successfully")
                
        except Exception as e:
            if config.ENABLE_DEBUG_LOGGING:
                print(f"Error stopping transcription: {e}")

    def run(self):
        """Run the application with proper cleanup handlers"""
        try:
            # Register signal handlers for clean shutdown
            signal.signal(signal.SIGINT, self.signal_handler)
            signal.signal(signal.SIGTERM, self.signal_handler)
            
            # Register cleanup function to run on exit
            atexit.register(self.cleanup_all_resources)
            
            # Set window close protocol
            try:
                self.root.protocol("WM_DELETE_WINDOW", self.quit_app)
            except tk.TclError:
                # For windows without decorations, bind to window events
                self.root.bind('<Destroy>', lambda e: self.quit_app() if e.widget == self.root else None)
            
            if config.ENABLE_DEBUG_LOGGING:
                print("Online-Translator is now running...")
            else:
                print("🚀 Online-Translator is ready!")
                print("💡 Double-click to show/hide controls")
                print("🖱️  Drag to move the window")
                print("🎯 Audio source will be auto-detected")
            
            # Start the main loop
            self.root.mainloop()
            
        except KeyboardInterrupt:
            if config.ENABLE_DEBUG_LOGGING:
                print("\nReceived keyboard interrupt")
            self.quit_app()
        except Exception as e:
            if config.ENABLE_DEBUG_LOGGING:
                print(f"Unexpected error in main loop: {e}")
            self.quit_app()
        finally:
            # Ensure cleanup happens
            try:
                self.cleanup_all_resources()
            except Exception as e:
                if config.ENABLE_DEBUG_LOGGING:
                    print(f"Error in final cleanup: {e}")
                    
    def signal_handler(self, signum, frame):
        """Handle system signals for clean shutdown"""
        try:
            if config.ENABLE_DEBUG_LOGGING:
                print(f"Received signal {signum}, shutting down...")
            self.quit_app()
        except Exception as e:
            if config.ENABLE_DEBUG_LOGGING:
                print(f"Error in signal handler: {e}")
        finally:
            sys.exit(0)

    def quit_app(self):
        """Quit the application with proper cleanup - MEJORADO"""
        # Prevenir múltiples llamadas
        if hasattr(self, '_quitting') and self._quitting:
            return
        self._quitting = True
        
        try:
            if config.ENABLE_DEBUG_LOGGING:
                print("Starting application shutdown...")
            else:
                print("👋 Goodbye from Online-Translator!")
            
            # Detener flag de ejecución
            self.is_running = False
            
            # Stop transcription and translation gracefully
            self.stop_transcription()
            
            # Clean up semaphores and queues
            self.cleanup_all_resources()
            
            if config.ENABLE_DEBUG_LOGGING:
                print("Application shutdown complete")
                
        except Exception as e:
            if config.ENABLE_DEBUG_LOGGING:
                print(f"Error during shutdown: {e}")
        finally:
            try:
                # Verificar si la ventana aún existe antes de destruir
                if hasattr(self, 'root') and self.root.winfo_exists():
                    self.root.quit()  # Salir del mainloop primero
                    self.root.destroy()  # Después destruir la ventana
            except (tk.TclError, AttributeError):
                pass  # Ignorar errores si la ventana ya fue destruida

    def cleanup_all_resources(self):
        """Comprehensive cleanup of all resources and semaphores"""
        try:
            # Clean up transcriber resources
            if hasattr(self, 'transcriber'):
                try:
                    self.transcriber.cleanup_all()
                except Exception as e:
                    if config.ENABLE_DEBUG_LOGGING:
                        print(f"Error cleaning transcriber: {e}")
                        
            # Clean up MLX cache
            try:
                mx.clear_cache()
            except Exception as e:
                if config.ENABLE_DEBUG_LOGGING:
                    print(f"Error clearing MLX cache: {e}")
                    
        except Exception as e:
            if config.ENABLE_DEBUG_LOGGING:
                print(f"Error in cleanup_all_resources: {e}")
                
    def update_volume_meter(self):
        """Update the visual volume meter"""
        try:
            if not hasattr(self, 'volume_canvas') or not self.volume_canvas.winfo_exists():
                return
                
            # Get current volume level from continuous buffer
            volume = 0.0
            if hasattr(self.transcriber, 'continuous_buffer') and self.transcriber.continuous_buffer:
                # Calculate RMS from the most recent audio in buffer
                with self.transcriber.continuous_buffer.lock:
                    if len(self.transcriber.continuous_buffer.buffer) > 0:
                        # Get the most recent 1 second of audio for volume calculation
                        recent_samples = min(self.transcriber.sample_rate, len(self.transcriber.continuous_buffer.buffer))
                        recent_audio = self.transcriber.continuous_buffer.buffer[-recent_samples:]
                        volume = float(np.sqrt(np.mean(recent_audio.astype(np.float64)**2))) if len(recent_audio) > 0 else 0.0
            
            # Convert to percentage (0-100)
            volume_percent = min(volume * 1000, 100)  # Scale factor for visibility
            
            # Update text
            if hasattr(self, 'volume_text') and self.volume_text.winfo_exists():
                self.volume_text.config(text=f"{volume_percent:.1f}%")
            
            # Only update visual meter if it's visible
            if self.audio_meter_visible:
                # Clear canvas
                self.volume_canvas.delete("all")
                
                # Get canvas dimensions
                canvas_width = self.volume_canvas.winfo_width()
                canvas_height = self.volume_canvas.winfo_height()
                
                if canvas_width <= 1:  # Canvas not ready yet
                    self.root.after(100, self.update_volume_meter)
                    return
                
                # Draw volume bar
                bar_width = int((volume_percent / 100) * canvas_width)
                
                # Color based on volume level
                if volume_percent < 10:
                    color = '#404040'  # Dark gray for low volume
                elif volume_percent < 30:
                    color = '#00ff00'  # Green for good volume
                elif volume_percent < 70:
                    color = '#ffff00'  # Yellow for high volume
                else:
                    color = '#ff0000'  # Red for very high volume
                
                # Draw the volume bar
                if bar_width > 0:
                    self.volume_canvas.create_rectangle(0, 0, bar_width, canvas_height, 
                                                      fill=color, outline=color)
                
                # Simplified volume visualization - removed complex history bars
                # The continuous buffer provides real-time data without needing separate history
            
            # Schedule next update
            if self.is_running:
                self.root.after(50, self.update_volume_meter)  # Update every 50ms
                
        except Exception as e:
            if config.ENABLE_DEBUG_LOGGING:
                print(f"Volume meter update error: {e}")
            # Retry after a short delay
            self.root.after(100, self.update_volume_meter)

    def on_source_language_change(self, event):
        """Handle change in source language"""
        selected_name = self.source_combo.get()
        if selected_name:
            # Buscar el código correspondiente al nombre seleccionado
            for code in config.SUPPORTED_SOURCE_LANGUAGES:
                if config.LANGUAGE_NAMES.get(code, code) == selected_name:
                    self.source_language.set(code)
                    print(f"🎤 Source language changed to: {selected_name} ({code})")
                    break

    def on_target_language_change(self, event):
        """Handle change in target language"""
        selected_name = self.target_combo.get()
        if selected_name:
            if selected_name == "Desactivado":
                self.target_language.set("Desactivate")
                print(f"🌐 Target language disabled")
            else:
                # Buscar el código correspondiente al nombre seleccionado
                for code in config.SUPPORTED_TARGET_LANGUAGES:
                    if config.LANGUAGE_NAMES.get(code, code) == selected_name:
                        self.target_language.set(code)
                        print(f"🌐 Target language changed to: {selected_name} ({code})")
                        break

    def on_accumulated_text_update(self, accumulated_text):
        """Callback for real-time accumulated text updates (before LLM validation)"""
        self.current_accumulated_text = accumulated_text
        
        # Update the UI to show the current accumulated text in real-time
        def update_live_text():
            try:
                if hasattr(self, 'original_text') and self.original_text.winfo_exists():
                    # Show recent completed transcriptions + current accumulated text
                    display_parts = []
                    
                    # Add recent completed transcriptions
                    if self.recent_transcriptions:
                        display_parts.extend(self.recent_transcriptions)
                    
                    # Add current accumulated text (in progress)
                    if accumulated_text:
                        display_parts.append(f"⏳ {accumulated_text}")
                    
                    # Join with double line breaks for better readability
                    display_text = "\n\n".join(display_parts) if display_parts else ""
                    self.original_text.config(text=display_text)
            except tk.TclError:
                pass  # Widget was destroyed
        
        self.root.after(0, update_live_text)


def main():
    """Main function to run Online-Translator with comprehensive error handling"""
    app = None
    try:
        print("🎬 Starting Online-Translator - Live Assistant")
        print("❤️  Created with love by Kiko Cisneros for his children")
        print("🎤 Audio transcription and translation powered by MLX + ffmpeg")
        print("=" * 60)
        
        # Create and run the application
        app = FloatingSubtitles()
        
        # Set up exception handler for uncaught exceptions
        def handle_exception(exc_type, exc_value, exc_traceback):
            try:
                if config.ENABLE_DEBUG_LOGGING:
                    print(f"Uncaught exception: {exc_type.__name__}: {exc_value}")
                if app:
                    app.cleanup_all_resources()
            except Exception as cleanup_error:
                if config.ENABLE_DEBUG_LOGGING:
                    print(f"Error during exception cleanup: {cleanup_error}")
            finally:
                sys.exit(1)
        
        sys.excepthook = handle_exception
        
        # Run the application
        app.run()
        
    except KeyboardInterrupt:
        if config.ENABLE_DEBUG_LOGGING:
            print("\nOnline-Translator interrupted by user")
        if app:
            try:
                app.quit_app()
            except:
                pass
    except Exception as e:
        if config.ENABLE_DEBUG_LOGGING:
            print(f"Unexpected error in Online-Translator main: {e}")
        if app:
            try:
                app.cleanup_all_resources()
            except:
                pass
    finally:
        # Final cleanup attempt
        if app:
            try:
                app.cleanup_all_resources()
            except Exception as e:
                if config.ENABLE_DEBUG_LOGGING:
                    print(f"Error in final cleanup: {e}")
        
        # Force cleanup of any remaining MLX resources
        try:
            mx.clear_cache()
        except:
            pass
            
        if config.ENABLE_DEBUG_LOGGING:
            print("Online-Translator terminated cleanly")
        print("👋 Goodbye from Online-Translator!")


if __name__ == "__main__":
    main() 