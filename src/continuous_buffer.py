#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Continuous Buffer Module
-----------------------
Manages continuous audio buffer processing for real-time transcription.
Handles audio data streaming and buffering for optimal performance.

Copyright (c) 2024 Kiko Cisneros
Licensed under the MIT License (see LICENSE file for details)
"""

import threading
import numpy as np


class ContinuousBuffer:
    """Manages continuous audio buffer for real-time transcription"""
    
    def __init__(self, sample_rate=16000, max_duration=30.0):
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)
        self.buffer = np.array([], dtype=np.float32)
        self.lock = threading.Lock()
        
        # Transcription overlap management
        self.overlap_duration = 0.4  # 1 second overlap
        self.overlap_samples = int(self.overlap_duration * sample_rate)
        self.last_transcribed_position = 0
        
        print(f"📊 Continuous buffer initialized: {max_duration}s max, {self.overlap_duration}s overlap")
        
    def add_audio(self, audio_chunk):
        """Add new audio to the continuous buffer"""
        with self.lock:
            # Append new audio
            self.buffer = np.concatenate([self.buffer, audio_chunk])
            
            # Keep buffer within max size (sliding window)
            if len(self.buffer) > self.max_samples:
                excess = len(self.buffer) - self.max_samples
                self.buffer = self.buffer[excess:]
                # Adjust transcribed position
                self.last_transcribed_position = max(0, self.last_transcribed_position - excess)
    
    def get_new_audio_for_transcription(self):
        """Get new audio that hasn't been transcribed yet (with overlap)"""
        with self.lock:
            if len(self.buffer) == 0:
                return np.array([])
            
            # Calculate start position with overlap
            start_pos = max(0, self.last_transcribed_position - self.overlap_samples)
            
            # Get new audio chunk
            new_audio = self.buffer[start_pos:].copy()
            
            # Update last transcribed position
            self.last_transcribed_position = len(self.buffer)
            
            return new_audio
    
    def clear_transcribed_audio(self, keep_samples=None):
        """Clear old transcribed audio, keeping recent samples for context"""
        with self.lock:
            if keep_samples is None:
                keep_samples = self.overlap_samples * 2
            
            if len(self.buffer) > keep_samples:
                self.buffer = self.buffer[-keep_samples:]
                self.last_transcribed_position = 0
    
    def get_total_duration(self):
        """Get total duration of audio in buffer"""
        with self.lock:
            return len(self.buffer) / self.sample_rate
    
    def get_buffer_info(self):
        """Get detailed buffer information for debugging"""
        with self.lock:
            return {
                'total_samples': len(self.buffer),
                'total_duration': len(self.buffer) / self.sample_rate,
                'max_samples': self.max_samples,
                'max_duration': self.max_samples / self.sample_rate,
                'last_transcribed_position': self.last_transcribed_position,
                'overlap_samples': self.overlap_samples,
                'overlap_duration': self.overlap_duration
            } 