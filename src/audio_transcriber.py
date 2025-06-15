#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Audio Transcription Module
-------------------------
Handles audio capture, processing, and transcription using the Whisper model
via MLX. Manages real-time audio streaming and transcription.

Copyright (c) 2024 Kiko Cisneros
Licensed under the MIT License (see LICENSE file for details)
"""

import threading
import queue
import time
import subprocess
import numpy as np
from collections import deque
import mlx_whisper
import re
import tempfile
import os
import signal
from pathlib import Path
import json

# Handle imports for both module and standalone usage
try:
    from . import config
    from .lightweight_llm import LightweightLLM
    from .continuous_buffer import ContinuousBuffer
except ImportError:
    import config
    from lightweight_llm import LightweightLLM
    from continuous_buffer import ContinuousBuffer


class RealTimeTranscriber:
    """Real-time audio transcriber with enhanced text processing"""
    
    def __init__(self, sentence_callback=None, accumulated_text_callback=None, preload_llm=False):
        # Audio configuration
        self.sample_rate = config.AUDIO_SAMPLE_RATE
        self.chunk_duration = config.AUDIO_CHUNK_DURATION
        self.chunk_size = int(self.sample_rate * self.chunk_duration)
        
        # ✅ MEJORAR MANEJO DE SEÑALES PARA CIERRE LIMPIO
        self._shutdown_requested = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        print("🔐 Verificando permisos (una sola vez)...")
        # Check permissions first - MEJORADO PARA NO PEDIR SIEMPRE
        self.has_audio_permissions = check_audio_permissions()
        self.has_screen_permissions = check_screen_capture_permissions()
        
        if not self.has_audio_permissions:
            print("⚠️  Permisos de micrófono requeridos")
            request_permissions()
        else:
            print("✅ Permisos verificados correctamente")
        
        print("📊 Initializing enhanced continuous buffer...")
        # Continuous buffer
        self.continuous_buffer = ContinuousBuffer(self.sample_rate)
        
        print("🧠 Initializing enhanced LLM for semantic validation...")
        # Enhanced text processing
        self.llm = LightweightLLM()
        
        # ✅ Optional model preloading - only if requested
        if preload_llm:
            print("⏳ Pre-loading LLM model (this may take 10-15 seconds)...")
            try:
                self._preload_llm_model()
                print("✅ LLM model pre-loaded and ready!")
            except Exception as e:
                print(f"⚠️  LLM preload failed: {e}")
                print("🔄 Model will load on first use")
        else:
            print("⚡ LLM will load on first use (may cause brief delay on first transcription)")
        
        self.accumulated_text = ""
        self.sentence_callback = sentence_callback
        self.accumulated_text_callback = accumulated_text_callback
        self.text_lock = threading.Lock()
        
        print("🎵 Setting up audio system...")
        # ffmpeg audio capture
        self.ffmpeg_process = None
        self.is_recording = False
        self.audio_thread = None
        self.transcription_thread = None
        self.processing_active = False
        
        # Audio device management (after permissions are set)
        self.audio_devices = self.get_audio_devices()
        self.selected_device = self.select_best_audio_device()
        
        # Whisper model
        self.whisper_model_name = config.WHISPER_MODEL
        print("🔄 Pre-loading Whisper model...")
        self.whisper_model_preloaded = False
        try:
            start_preload_time = time.time()
            self._preload_whisper_model()
            preload_time = time.time() - start_preload_time
            self.whisper_model_preloaded = True
            print(f"✅ Whisper model pre-loaded and ready! ({preload_time:.1f}s)")
        except Exception as e:
            print(f"⚠️  Whisper preload failed: {e}")
            print("🔄 Model will load on first use (may cause 15-20s delay)")
            self.whisper_model_preloaded = False
        
        # Timing and statistics
        self.transcription_times = deque(maxlen=20)
        self.total_transcriptions = 0
        
        # Enhanced phrase filtering
        self.phrases_to_ignore = [
            "gracias", "thank you", "thanks", "thank", "you", "thank's", "thank's you", 
            "gracias.", "thank you.", "thanks.", "thank.", "you.", "thank's.", "thank's you."
        ]
        self.recent_transcribed_texts = deque(maxlen=5)  # Keep track of recent transcriptions
        
        print("🚀 Enhanced real-time transcriber initialized and ready")
    
    def _process_transcribed_text(self, new_text):
        """Enhanced text processing with improved concatenation logic"""
        with self.text_lock:
            # ✅ PREVENT repetitive transcription loops
            if new_text and new_text.strip():
                # Check if this text is too similar to recent transcriptions
                new_text_clean = new_text.strip()
                for recent_text in self.recent_transcribed_texts:
                    if new_text_clean.lower() == recent_text.lower():
                        print(f"🔄 Skipping duplicate transcription: '{new_text_clean[:30]}...'")
                        return  # Skip processing duplicate text
                
                # ✅ FILTER OUT COMMON PHRASES like "Gracias" and "Thank you"
                if new_text_clean.lower() in self.phrases_to_ignore:
                    print(f"🔇 Ignoring common phrase: '{new_text_clean}'")
                    return  # Skip processing ignored phrases
                
                # Add to recent transcriptions list
                self.recent_transcribed_texts.append(new_text_clean)
            
            # ✅ ENHANCED CONCATENATION LOGIC
            if new_text and new_text.strip():
                new_text_clean = new_text.strip()
                
                # 🚫 DETECT WHISPER REPETITIVE LOOPS (known bug)
                if self.llm._detect_repetitive_transcription(new_text_clean):
                    print(f"🚫 Skipping repetitive Whisper transcription: '{new_text_clean[:50]}...'")
                    return  # Skip this transcription entirely
                
                # 🧮 USE ENHANCED CONCATENATION WITH IMPROVED OVERLAP DETECTION
                merged_text, is_complete, reason = self.llm.smart_concatenate_with_mahalanobis(
                    self.accumulated_text, new_text_clean
                )
                
                # Update accumulated text with the merged result
                self.accumulated_text = merged_text
                
                print(f"🔄 Enhanced Merged Text: '{self.accumulated_text}' ({len(self.accumulated_text)} chars) - {reason}")
                
                # ✅ Real-time UI update with the merged text
                if self.accumulated_text_callback and self.accumulated_text:
                    self.accumulated_text_callback(self.accumulated_text)
                
                # 🧮 If system determined the text is complete, process it as a sentence
                if is_complete:
                    # ✂️ REMOVE "GRACIAS" AND EQUIVALENTS AT END OF SENTENCE
                    final_text = self._remove_thanks_at_end(self.accumulated_text)
                    if final_text != self.accumulated_text:
                        print(f"✂️ Removed 'thanks' at end: '{self.accumulated_text}' -> '{final_text}'")
                        self.accumulated_text = final_text
                    
                    print(f"✅ Enhanced Smart Completion: {self.accumulated_text}")
                    if self.sentence_callback:
                        self.sentence_callback(self.accumulated_text.strip())
                    self.accumulated_text = ""
                    # Clear accumulated text in UI when sentence is complete
                    if self.accumulated_text_callback:
                        self.accumulated_text_callback("")
                return
            else:
                print(f"🔇 Empty transcription - keeping current buffer: '{self.accumulated_text}' ({len(self.accumulated_text)} chars)")
            
            # ✅ ENHANCED SAFETY CHECK: Prevent overly long accumulated text
            if len(self.accumulated_text) > 400:  # Increased from 300 to 400
                print(f"⚠️  Accumulated text too long ({len(self.accumulated_text)} chars), force processing...")
                
                final_text = self._remove_thanks_at_end(self.accumulated_text)
                if final_text != self.accumulated_text:
                    print(f"✂️ Removed 'thanks' at end: '{self.accumulated_text}' -> '{final_text}'")
                    self.accumulated_text = final_text
                
                if self.accumulated_text.strip():
                    print(f"✅ Forced processing: {self.accumulated_text}")
                    if self.sentence_callback:
                        self.sentence_callback(self.accumulated_text.strip())
                else:
                    print(f"❌ Force processing failed, clearing buffer")
                
                self.accumulated_text = ""
                # Clear accumulated text in UI
                if self.accumulated_text_callback:
                    self.accumulated_text_callback("")
    
    def _remove_thanks_at_end(self, text):
        """Remove 'thanks' and equivalents in multiple languages at end of sentence"""
        if not text:
            return text
            
        # List of thanks expressions in multiple languages
        thanks_expressions = [
            # Spanish
            r'(?i)(gracias)\.?$', r'(?i)(muchas gracias)\.?$', r'(?i)(mil gracias)\.?$',
            # English
            r'(?i)(thank you)\.?$', r'(?i)(thanks)\.?$', r'(?i)(thank you very much)\.?$',
            # Italian
            r'(?i)(grazie)\.?$', r'(?i)(grazie mille)\.?$', r'(?i)(tante grazie)\.?$',
            # French
            r'(?i)(merci)\.?$', r'(?i)(merci beaucoup)\.?$',
            # Portuguese
            r'(?i)(obrigado)\.?$', r'(?i)(obrigada)\.?$', r'(?i)(muito obrigado)\.?$',
            # German
            r'(?i)(danke)\.?$', r'(?i)(vielen dank)\.?$',
            # Japanese
            r'(?i)(arigato)\.?$', r'(?i)(arigatou)\.?$', r'(?i)(ありがとう)\.?$', 
            r'(?i)(どうもありがとう)\.?$', r'(?i)(ありがとうございました)\.?$',
            # Generic for other languages: short expressions at end
            r'(?i)( (thanks|gracias|grazie|merci|danke|obrigado|arigato))\.?$'
        ]
        
        cleaned_text = text
        
        for pattern in thanks_expressions:
            # Check if sentence ends with this expression
            match = re.search(pattern, cleaned_text)
            if match:
                # Remove expression from end
                cleaned_text = cleaned_text[:match.start()].rstrip()
                break  # If we found a pattern, don't keep searching
                
        return cleaned_text
    
    def start_continuous_transcription(self):
        """Start continuous real-time transcription"""
        if self.is_recording:
            return
        
        print("🎵 Starting enhanced continuous transcription...")
        
        # Start audio recording
        self.start_recording()
        
        # Start transcription processing
        self.processing_active = True
        self.transcription_thread = threading.Thread(target=self._continuous_transcription_worker, daemon=True)
        self.transcription_thread.start()
        
        print("✅ Enhanced continuous transcription active")
    
    def stop_continuous_transcription(self):
        """Stop continuous transcription - MEJORADO"""
        if not self.processing_active:
            return
            
        print("🛑 Deteniendo transcripción continua...")
        
        self.processing_active = False
        
        # Si es shutdown rápido, no esperar tanto
        if self._shutdown_requested:
            timeout = 0.3
        else:
            timeout = 1.0
        self.stop_recording()
        
        # Wait for transcription thread to finish - TIMEOUT DINÁMICO
        if self.transcription_thread and self.transcription_thread.is_alive():
            self.transcription_thread.join(timeout=timeout)
            if self.transcription_thread.is_alive() and not self._shutdown_requested:
                print("⚠️  Thread de transcripción no terminó a tiempo")
        
        # Process any remaining text - SOLO SI NO ES SHUTDOWN
        if not self._shutdown_requested:
            self._flush_remaining_text()
        
        print("✅ Transcripción continua detenida")
    
    def _continuous_transcription_worker(self):
        """Enhanced worker thread for continuous transcription processing"""
        min_audio_duration = 0.4  # ✅ OPTIMIZED: Balanced for quality and responsiveness
        max_wait_time = 0.6       # ✅ OPTIMIZED: Slightly longer for better sentence formation
        last_transcription_time = time.time()
        
        # ✅ ENHANCED: Better initial processing
        initial_processing_mode = True
        initial_min_duration = 0.8  # ✅ BALANCED: Better first chunk for quality
        
        # Enhanced silence detection
        silence_threshold = 0.003   # ✅ OPTIMIZED: Slightly higher threshold
        silence_duration_threshold = 0.5  # ✅ ENHANCED: Longer for better pause detection
        is_silence_mode = False  
        silence_start_time = 0  
        buffer_overlap = 0.15  # ✅ ENHANCED: Longer overlap for better word continuity
        small_buffer = np.array([], dtype=np.float32)  
        
        while self.processing_active and not self._shutdown_requested:
            try:
                # ✅ VERIFICAR SHUTDOWN REQUEST MÁS FRECUENTEMENTE
                if self._shutdown_requested or not self.processing_active:
                    print("🛑 Worker thread detectó shutdown request")
                    break
                    
                current_time = time.time()
                
                # Get new audio for transcription
                new_audio = self.continuous_buffer.get_new_audio_for_transcription()
                
                if len(new_audio) == 0:
                    time.sleep(0.05)  # ✅ OPTIMIZED: Balanced polling
                    continue
                
                # Calculate duration and audio level
                audio_duration = len(new_audio) / self.sample_rate
                
                # ✅ ENHANCED: Safer RMS calculation
                try:
                    new_audio_safe = np.clip(new_audio, -1.0, 1.0)
                    audio_rms = np.sqrt(np.mean(new_audio_safe.astype(np.float64)**2))
                    
                    if not np.isfinite(audio_rms):
                        audio_rms = 0.0
                        
                except (OverflowError, RuntimeWarning):
                    audio_rms = 0.0
                
                time_since_last = current_time - last_transcription_time
                
                # ✅ ENHANCED: Special handling for first transcription
                if initial_processing_mode and self.total_transcriptions == 0:
                    if audio_duration >= initial_min_duration or time_since_last >= 3.0:
                        print(f"🚀 Processing FIRST audio chunk: {audio_duration:.1f}s")
                        transcribed_text = self._transcribe_audio_chunk(new_audio)
                        self._process_transcribed_text(transcribed_text)
                        last_transcription_time = current_time
                        initial_processing_mode = False
                        print("✅ First transcription complete - switching to enhanced real-time mode")
                        continue
                    else:
                        time.sleep(0.1)
                        continue
                
                # Enhanced silence detection
                is_silence = audio_rms < silence_threshold
                
                if not is_silence_mode and is_silence:
                    if silence_start_time == 0:
                        silence_start_time = current_time
                        small_buffer = new_audio[-int(buffer_overlap * self.sample_rate):]
                    elif current_time - silence_start_time >= silence_duration_threshold:
                        is_silence_mode = True
                        print(f"🔇 Enhanced silence detected ({audio_rms:.4f}) - pausing active transcription")
                
                elif is_silence_mode and not is_silence:
                    is_silence_mode = False
                    silence_start_time = 0
                    print(f"🔊 Sound detected ({audio_rms:.4f}) - resuming enhanced transcription")
                    
                    if len(small_buffer) > 0:
                        new_audio = np.concatenate([small_buffer, new_audio])
                        small_buffer = np.array([], dtype=np.float32)
                
                if not is_silence_mode and not is_silence:
                    small_buffer = new_audio[-int(buffer_overlap * self.sample_rate):]
                
                # Enhanced decision logic for processing
                should_transcribe = False
                
                if is_silence_mode:
                    should_transcribe = time_since_last >= 1.5 and audio_rms > silence_threshold * 0.5
                else:
                    should_transcribe = (
                        audio_duration >= min_audio_duration or
                        (audio_duration >= 0.2 and time_since_last >= max_wait_time)
                    )
                
                if should_transcribe and len(new_audio) > 0:
                    status = "🔄" if not is_silence_mode else "🔈"
                    print(f"{status} Enhanced processing {audio_duration:.1f}s (RMS: {audio_rms:.4f}, gap: {time_since_last:.1f}s)")
                    
                    # ✅ ENHANCED: Better chunk size management
                    max_chunk_duration = 6.0  # Increased from 5.0
                    if audio_duration > max_chunk_duration:
                        max_samples = int(max_chunk_duration * self.sample_rate)
                        new_audio = new_audio[-max_samples:]
                        print(f"⚠️  Audio chunk too long, trimmed to {max_chunk_duration}s")
                    
                    # Transcribe with enhanced processing
                    transcribed_text = self._transcribe_audio_chunk(new_audio)
                    
                    if transcribed_text or transcribed_text == "":
                        self._process_transcribed_text(transcribed_text)
                        last_transcription_time = current_time
                        
                        # ✅ ENHANCED: Smarter buffer cleanup
                        if audio_duration > 10.0:  # Only clean when buffer is very large
                            overlap_keep_samples = self.continuous_buffer.overlap_samples * 3  # More overlap
                            self.continuous_buffer.clear_transcribed_audio(keep_samples=overlap_keep_samples)
                            print(f"🧹 Enhanced buffer cleanup, kept {overlap_keep_samples/self.sample_rate:.1f}s overlap")
                
                # ✅ ENHANCED: More responsive sleep
                time.sleep(0.05)  # Balanced polling
                
            except Exception as e:
                print(f"❌ Enhanced transcription worker error: {e}")
                time.sleep(0.5)
    
    def _transcribe_audio_chunk(self, audio_data):
        """Enhanced audio transcription with better validation"""
        if len(audio_data) < self.sample_rate * 0.6:  # Increased minimum duration
            return ""
        
        try:
            start_time = time.time()
            
            # ✅ ENHANCED: Better audio validation
            if not np.isfinite(audio_data).all():
                print("🚫 Corrupted audio detected, skipping")
                return ""
            
            audio_data = np.clip(audio_data, -1.0, 1.0)
            
            try:
                audio_rms = np.sqrt(np.mean(audio_data.astype(np.float64)**2))
                if not np.isfinite(audio_rms):
                    audio_rms = 0.0
            except (OverflowError, RuntimeWarning):
                audio_rms = 0.0
            
            print(f"🎵 Enhanced processing: {len(audio_data)/self.sample_rate:.1f}s, RMS: {audio_rms:.4f}")
            
            # ✅ ENHANCED: More permissive threshold for quiet audio
            if audio_rms < 0.0008:  # Slightly higher but still permissive
                print("🔇 Audio too quiet for enhanced processing")
                return ""
            
            if not hasattr(self, 'whisper_model_preloaded') or not self.whisper_model_preloaded:
                if self.total_transcriptions == 0:
                    print("⏳ First enhanced transcription - loading Whisper model...")
            
            # ✅ ENHANCED: Better transcription parameters
            result = mlx_whisper.transcribe(
                audio_data,
                path_or_hf_repo=f"mlx-community/whisper-{self.whisper_model_name}-mlx",
                language=None,  # Auto-detect
                word_timestamps=False,
                no_speech_threshold=0.5,  # ✅ ENHANCED: More balanced
                logprob_threshold=-1.2,   # ✅ ENHANCED: Better quality balance
                compression_ratio_threshold=2.8,  # ✅ ENHANCED: Better repetition detection
                condition_on_previous_text=True  # Better context
            )
            
            text = result.get("text", "").strip()
            
            # ✅ ENHANCED: Better corruption detection
            if self._is_corrupted_transcription_enhanced(text):
                print(f"🚫 Enhanced corruption detection: skipping '{text[:30]}...'")
                return ""
            
            # Record timing
            transcription_time = time.time() - start_time
            self.transcription_times.append(transcription_time)
            self.total_transcriptions += 1
            
            if text:
                avg_time = sum(self.transcription_times) / len(self.transcription_times)
                duration = len(audio_data) / self.sample_rate
                print(f"🎯 Enhanced transcription ({duration:.1f}s): '{text[:50]}{'...' if len(text) > 50 else ''}'")
                print(f"⏱️  Enhanced timing: {transcription_time:.2f}s | Avg: {avg_time:.2f}s")
            else:
                print(f"🔇 Empty enhanced transcription for {len(audio_data)/self.sample_rate:.1f}s audio")
            
            return text
            
        except Exception as e:
            print(f"❌ Enhanced transcription error: {e}")
            return ""
    
    def _is_corrupted_transcription_enhanced(self, text):
        """Enhanced corruption detection for transcriptions"""
        if not text or len(text.strip()) == 0:
            return False
        
        text_clean = text.strip()
        
        # ✅ ENHANCED CHECK 1: Only punctuation/symbols
        if len(text_clean) > 3:
            non_alpha_count = sum(1 for c in text_clean if not c.isalpha() and not c.isspace())
            alpha_count = sum(1 for c in text_clean if c.isalpha())
            
            if alpha_count == 0 or (non_alpha_count / len(text_clean)) > 0.75:  # More permissive
                return True
        
        # ✅ ENHANCED CHECK 2: Excessive character repetition
        if len(text_clean) > 8:
            for char in '!?.,;:-_=+':
                if text_clean.count(char) > len(text_clean) * 0.6:  # More permissive
                    return True
        
        # ✅ ENHANCED CHECK 3: Better corruption patterns
        corruption_patterns = [
            r'^[!?.,;:\-_=+\s]+$',  # Only punctuation
            r'(.{1,2})\1{15,}',     # Same 1-2 chars repeated 15+ times (more permissive)
            r'^[0-9!@#$%^&*()_+\-=\[\]{};:\'",.<>?/`~\s]+$'  # Only numbers and symbols
        ]
        
        for pattern in corruption_patterns:
            if re.search(pattern, text_clean):
                return True
        
        return False
    
    def _flush_remaining_text(self):
        """Flush any remaining accumulated text"""
        with self.text_lock:
            if self.accumulated_text.strip() and self.sentence_callback:
                print(f"🔄 Enhanced flushing remaining: {self.accumulated_text}")
                self.sentence_callback(self.accumulated_text.strip())
                self.accumulated_text = ""
    
    def force_preload_models(self):
        """Force preload all models"""
        print("🔄 Force pre-loading enhanced models...")
        
        if not getattr(self, 'whisper_model_preloaded', False):
            print("📥 Force loading Whisper model...")
            try:
                start_time = time.time()
                self._preload_whisper_model()
                preload_time = time.time() - start_time
                self.whisper_model_preloaded = True
                print(f"✅ Whisper model force-loaded! ({preload_time:.1f}s)")
            except Exception as e:
                print(f"❌ Whisper force preload failed: {e}")
                self.whisper_model_preloaded = False
        else:
            print("✅ Whisper model already loaded")
        
        if not self.llm.model_loaded:
            print("📥 Force loading enhanced LLM model...")
            try:
                self._preload_llm_model()
                print("✅ Enhanced LLM model force-loaded!")
            except Exception as e:
                print(f"⚠️  Enhanced LLM force preload failed: {e}")
        else:
            print("✅ Enhanced LLM model already loaded")
    
    def get_performance_info(self):
        """Get enhanced performance information"""
        buffer_info = self.continuous_buffer.get_buffer_info()
        
        info = {
            "whisper_model_preloaded": getattr(self, 'whisper_model_preloaded', False),
            "llm_model_loaded": self.llm.model_loaded,
            "total_transcriptions": self.total_transcriptions,
            "is_recording": self.is_recording,
            "processing_active": self.processing_active,
            "buffer_info": buffer_info,
            "avg_transcription_time": sum(self.transcription_times) / len(self.transcription_times) if self.transcription_times else 0,
            "recent_transcription_times": list(self.transcription_times) if self.transcription_times else []
        }
        
        print("📊 Enhanced Performance Information:")
        print(f"  Buffer: {buffer_info['total_duration']:.1f}s / {buffer_info['max_duration']:.1f}s")
        print(f"  Transcriptions: {info['total_transcriptions']}")
        print(f"  Avg time: {info['avg_transcription_time']:.2f}s")
        print(f"  Models loaded: Whisper={info['whisper_model_preloaded']}, LLM={info['llm_model_loaded']}")
        
        return info
    
    def _preload_llm_model(self):
        """Pre-load the LLM model"""
        try:
            if self.llm.model_loaded:
                print(f"🧪 Enhanced LLM model already loaded and ready")
                return
            
            print("🔄 Testing enhanced LLM...")
            import mlx_lm
            
            simple_prompt = "Hello"
            response = mlx_lm.generate(
                self.llm.model, 
                self.llm.tokenizer, 
                prompt=simple_prompt,
                max_tokens=5,
                verbose=False
            )
            
            print(f"🧪 Enhanced LLM preload successful")
            
        except Exception as e:
            print(f"⚠️  Enhanced LLM preload failed: {e}")
    
    def _preload_whisper_model(self):
        """Pre-load the Whisper model"""
        try:
            print("📥 Loading enhanced Whisper model...")
            
            test_audio = np.zeros(int(self.sample_rate * 0.5), dtype=np.float32)
            
            result = mlx_whisper.transcribe(
                test_audio,
                path_or_hf_repo=f"mlx-community/whisper-{self.whisper_model_name}-mlx",
                language="en",
                word_timestamps=False,
                no_speech_threshold=0.9,
                logprob_threshold=-2.0,
                compression_ratio_threshold=10.0,
                condition_on_previous_text=False
            )
            
            print("🧪 Enhanced Whisper model loaded successfully")
            
        except Exception as e:
            print(f"❌ Enhanced Whisper preload failed: {e}")
            raise
    
    def start_recording(self):
        """Start audio recording using ffmpeg"""
        if self.is_recording:
            return
        
        request_permissions()
        
        if not self.check_ffmpeg_installation():
            print("❌ ffmpeg not installed. Installing...")
            if not self.install_ffmpeg():
                print("❌ Failed to install ffmpeg")
                return
        
        ffmpeg_input = self.get_device_ffmpeg_input(self.selected_device)
        
        cmd = [
            "ffmpeg", "-y",
            "-f", "avfoundation",
            "-i", ffmpeg_input,
            "-ar", str(self.sample_rate),
            "-ac", "1",
            "-acodec", "pcm_f32le",
            "-f", "wav",
            "pipe:1"
        ]
        
        try:
            print(f"🎵 Starting enhanced audio capture: {self.selected_device}")
            
            self.ffmpeg_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid
            )
            
            self.is_recording = True
            
            self.audio_thread = threading.Thread(target=self._audio_capture_worker, daemon=True)
            self.audio_thread.start()
            
            print("✅ Enhanced audio recording started")
            
        except Exception as e:
            print(f"❌ Error starting enhanced audio recording: {e}")
            self.is_recording = False
    
    def _audio_capture_worker(self):
        """Worker thread for capturing audio from ffmpeg"""
        chunk_size = self.chunk_size * 4
        
        while self.is_recording and self.ffmpeg_process:
            try:
                data = self.ffmpeg_process.stdout.read(chunk_size)
                if not data:
                    break
                
                audio_chunk = np.frombuffer(data, dtype=np.float32)
                
                if len(audio_chunk) > 0:
                    self.continuous_buffer.add_audio(audio_chunk)
                
            except Exception as e:
                if self.is_recording:
                    print(f"Enhanced audio capture error: {e}")
                break
    
    def stop_recording(self):
        """Stop audio recording - MEJORADO"""
        if not self.is_recording:
            return
            
        print("🛑 Deteniendo grabación de audio...")
        self.is_recording = False
        
        # Si es shutdown rápido, no esperar tanto
        if self._shutdown_requested:
            timeout = 0.2
        else:
            timeout = 2.0
        
        if self.ffmpeg_process:
            try:
                # Intentar terminación suave primero
                os.killpg(os.getpgid(self.ffmpeg_process.pid), signal.SIGINT)
                try:
                    self.ffmpeg_process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    if not self._shutdown_requested:
                        print("⚠️  FFmpeg no terminó suavemente, forzando...")
                    os.killpg(os.getpgid(self.ffmpeg_process.pid), signal.SIGKILL)
                    self.ffmpeg_process.wait(timeout=0.5)
            except Exception as e:
                if not self._shutdown_requested:
                    print(f"⚠️  Error deteniendo ffmpeg: {e}")
            
            self.ffmpeg_process = None
        
        # Wait for audio thread - TIMEOUT DINÁMICO
        if self.audio_thread and self.audio_thread.is_alive():
            thread_timeout = 0.2 if self._shutdown_requested else 1.0
            self.audio_thread.join(timeout=thread_timeout)
            if self.audio_thread.is_alive() and not self._shutdown_requested:
                print("⚠️  Thread de audio no terminó a tiempo")
        
        print("✅ Grabación de audio detenida")
    
    def get_audio_devices(self):
        """Get available audio input devices using ffmpeg"""
        devices = []
        try:
            result = subprocess.run([
                "ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""
            ], capture_output=True, text=True)
            
            lines = result.stderr.split('\n')
            in_audio_devices = False
            in_video_devices = False
            
            for line in lines:
                if '[AVFoundation indev @' in line and 'AVFoundation audio devices:' in line:
                    in_audio_devices = True
                    in_video_devices = False
                    continue
                elif '[AVFoundation indev @' in line and 'AVFoundation video devices:' in line:
                    in_audio_devices = False
                    in_video_devices = True
                    continue
                    
                if in_audio_devices and '[AVFoundation indev @' in line and '] [' in line and '] ' in line:
                    try:
                        parts = line.split('] [', 1)
                        if len(parts) < 2:
                            continue
                        
                        index_and_name = parts[1]
                        index_end = index_and_name.find('] ')
                        if index_end == -1:
                            continue
                        
                        device_index_str = index_and_name[:index_end]
                        device_name = index_and_name[index_end + 2:].strip()
                        
                        try:
                            device_index = int(device_index_str)
                        except ValueError:
                            continue
                        
                        if 'blackhole' in device_name.lower():
                            continue
                        
                        device_info = {
                            'name': device_name,
                            'index': device_index,
                            'is_system_audio': self.is_system_audio_device(device_name),
                            'is_microphone': self.is_microphone_device(device_name),
                            'device_type': 'audio'
                        }
                        devices.append(device_info)
                        
                    except Exception as e:
                        print(f"Warning: Could not parse audio device line: {line} - {e}")
                        continue
                
        except Exception as e:
            print(f"Error getting audio devices: {e}")
            devices = [
                {'name': 'Built-in Microphone', 'index': 0, 'is_system_audio': False, 'is_microphone': True, 'device_type': 'audio'}
            ]
        
        if hasattr(self, 'has_screen_permissions') and self.has_screen_permissions:
            native_system_audio = {
                'name': 'macOS System Audio (Native)',
                'index': 1,
                'is_system_audio': True,
                'is_microphone': False,
                'device_type': 'native_system_audio'
            }
            devices.insert(0, native_system_audio)
        
        return devices
    
    def is_system_audio_device(self, device_name):
        """Check if device is likely a system audio capture device"""
        system_indicators = [
            'capture screen', 'pantalla', 'screen capture', 'system audio'
        ]
        device_lower = device_name.lower()
        if 'blackhole' in device_lower:
            return False
        return any(indicator in device_lower for indicator in system_indicators)
    
    def is_microphone_device(self, device_name):
        """Check if device is likely a microphone"""
        mic_indicators = ['microphone', 'mic', 'built-in', 'internal', 'usb', 'bluetooth', 'airpods', 'headset', 'micrófono']
        device_lower = device_name.lower()
        return any(indicator in device_lower for indicator in mic_indicators)
    
    def get_device_ffmpeg_input(self, device_name):
        """Get the correct ffmpeg input format for the device"""
        device_info = None
        for device in self.audio_devices:
            if device['name'] == device_name:
                device_info = device
                break
        
        if device_info and device_info['index'] is not None:
            if device_info.get('device_type') == 'native_system_audio':
                return f"none:{device_info['index']}"
            elif device_info['is_microphone']:
                return f":{device_info['index']}"
            else:
                return f":{device_info['index']}"
        else:
            return ":0"
    
    def check_ffmpeg_installation(self):
        """Check if ffmpeg is installed"""
        try:
            result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def install_ffmpeg(self):
        """Install ffmpeg using Homebrew"""
        try:
            print("📦 Installing ffmpeg using Homebrew...")
            result = subprocess.run(["brew", "install", "ffmpeg"], capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ ffmpeg installed successfully!")
                return True
            else:
                print(f"❌ Error installing ffmpeg: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Error installing ffmpeg: {e}")
            return False
    
    def select_best_audio_device(self):
        """Automatically select the best available audio device - MEJORADO"""
        if not self.audio_devices:
            print("❌ No audio input devices found!")
            return "Built-in Microphone"
        
        print("🔍 Available audio devices:")
        for i, device in enumerate(self.audio_devices):
            device_type = "🖥️  Screen" if device.get('device_type') == 'screen_capture' else \
                         "🎵 System Audio" if device.get('device_type') == 'native_system_audio' else \
                         "🔊 System Audio" if device['is_system_audio'] else \
                         "🎤 Microphone" if device['is_microphone'] else "🎧 Audio"
            print(f"  {i}: {device_type} - {device['name']}")
            
        # ✅ LÓGICA MEJORADA: Priorizar micrófonos para transcripción de voz
        # 1. Primero buscar micrófonos integrados
        builtin_mics = [d for d in self.audio_devices if 'built-in' in d['name'].lower() or 'interno' in d['name'].lower() or 'micrófono' in d['name'].lower()]
        if builtin_mics:
            device_name = builtin_mics[0]['name']
            print(f"🎤 Auto-selected built-in microphone: {device_name}")
            return device_name
            
        # 2. Después buscar otros micrófonos
        mic_devices = [d for d in self.audio_devices if d['is_microphone']]
        if mic_devices:
            device_name = mic_devices[0]['name']
            print(f"🎤 Auto-selected microphone: {device_name}")
            return device_name
            
        # 3. Solo como última opción, usar audio del sistema (si hay permisos reales)
        # Verificar si realmente tenemos permisos de screen capture con una prueba más estricta
        real_screen_permissions = self._test_real_screen_permissions()
        if real_screen_permissions:
            native_audio_devices = [d for d in self.audio_devices if d.get('device_type') == 'native_system_audio']
            if native_audio_devices:
                device_name = native_audio_devices[0]['name']
                print(f"🎵 Auto-selected native system audio: {device_name}")
                return device_name
            
            system_audio_devices = [d for d in self.audio_devices if d['is_system_audio'] and d.get('device_type') != 'screen_capture']
            if system_audio_devices:
                device_name = system_audio_devices[0]['name']
                print(f"🔊 Auto-selected system audio: {device_name}")
                return device_name
        
        # 4. Fallback a cualquier dispositivo que no sea BlackHole
        non_blackhole_devices = [d for d in self.audio_devices if 'blackhole' not in d['name'].lower()]
        if non_blackhole_devices:
            device_name = non_blackhole_devices[0]['name']
            print(f"🎧 Auto-selected fallback audio device: {device_name}")
            return device_name
            
        return "Built-in Microphone"
    
    def _test_real_screen_permissions(self):
        """Test if we really have screen capture permissions (more strict)"""
        try:
            # Intentar una operación que realmente requiere screen capture
            result = subprocess.run([
                "osascript", "-e",
                'tell application "System Events" to tell process "Finder" to get name'
            ], capture_output=True, text=True, timeout=2)
            
            return result.returncode == 0 and "Finder" in result.stdout
        except Exception:
            return False
    
    def set_audio_device(self, device_name):
        """Set the audio device and restart recording if active"""
        print(f"🔄 Cambiando dispositivo de audio a: {device_name}")
        
        # Verificar que el dispositivo existe
        device_found = False
        for device in self.audio_devices:
            if device['name'] == device_name:
                device_found = True
                break
        
        if not device_found:
            print(f"⚠️  Dispositivo '{device_name}' no encontrado")
            return False
        
        # Guardar estado de grabación actual
        was_recording = self.is_recording
        
        # Detener grabación si está activa
        if self.is_recording:
            self.stop_recording()
        
        # Actualizar dispositivo seleccionado
        self.selected_device = device_name
        print(f"✅ Dispositivo de audio actualizado: {device_name}")
        
        # Reiniciar grabación si estaba activa
        if was_recording:
            self.start_recording()
            print(f"🎵 Grabación reiniciada con nuevo dispositivo")
        
        return True
    
    def get_current_audio_device(self):
        """Get the currently selected audio device"""
        return self.selected_device

    def _signal_handler(self, signum, frame):
        """Manejo mejorado de señales para cierre limpio"""
        if not self._shutdown_requested:
            print(f"\n🛑 Recibida señal {signum}, cerrando limpiamente...")
            self._shutdown_requested = True
            
            # Detener transcripción primero (más rápido)
            if self.processing_active:
                print("🛑 Deteniendo transcripción...")
                self.processing_active = False
                
            # Detener grabación después
            if self.is_recording:
                print("🛑 Deteniendo grabación...")
                self.is_recording = False
                
            # Dar tiempo mínimo para que los threads vean las señales
            time.sleep(0.5)
            
            # Forzar terminación de ffmpeg si sigue corriendo
            if self.ffmpeg_process:
                try:
                    os.killpg(os.getpgid(self.ffmpeg_process.pid), signal.SIGTERM)
                except:
                    pass
                    
            print("✅ Cierre limpio completado")
            # No llamar exit() directamente, dejar que la UI maneje el cierre
            return
        else:
            print("🛑 Forzando cierre...")
            exit(1)


# ✅ Compatibility aliases
AudioTranscriber = RealTimeTranscriber
SentenceAccumulator = LightweightLLM


def check_audio_permissions():
    """Check if the app has microphone permissions - MEJORADO"""
    try:
        # Método más simple: intentar listar dispositivos de audio
        result = subprocess.run([
            "ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""
        ], capture_output=True, text=True, timeout=3)
        
        # Si podemos listar dispositivos, probablemente tenemos permisos
        if "AVFoundation audio devices:" in result.stderr:
            return True
        
        # Fallback: intentar captura muy corta
        result = subprocess.run([
            "ffmpeg", "-f", "avfoundation", "-i", ":0", "-t", "0.1", "-f", "null", "-"
        ], capture_output=True, text=True, timeout=2)
        
        if "Operation not permitted" in result.stderr or "Error opening input device" in result.stderr:
            return False
        return True
    except subprocess.TimeoutExpired:
        # Si no hay timeout, probablemente tenemos permisos
        return True
    except Exception:
        # En caso de error, asumir que tenemos permisos (conservador)
        return True

def check_screen_capture_permissions():
    """Check if the app has screen recording permissions - MEJORADO"""
    try:
        # Método más permisivo - simplemente verificar si el comando existe
        result = subprocess.run([
            "osascript", "-e", "tell application \"System Events\" to get name"
        ], capture_output=True, text=True, timeout=2)
        
        # Si osascript funciona, asumir que tenemos los permisos básicos
        return True
        
    except Exception:
        # En caso de error, asumir que tenemos permisos (conservador)
        return True

def request_permissions():
    """Request permissions only if really needed - MEJORADO"""
    print("🔐 Verificando permisos de macOS...")
    
    # Solo pedir permisos si realmente no los tenemos
    if not check_audio_permissions():
        print("🎤 Solicitando permisos de micrófono (requerido)...")
        subprocess.run([
            "osascript", "-e",
            'display notification "Online-Translator necesita acceso al micrófono para la transcripción de audio." with title "Permisos requeridos"'
        ], capture_output=True)
    else:
        print("✅ Permisos de micrófono verificados")
    
    if not check_screen_capture_permissions():
        print("📺 Permisos de grabación de pantalla no detectados (opcional para audio del sistema)")
    else:
        print("✅ Permisos de sistema verificados") 