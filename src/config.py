#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration Module
-------------------
Configuration settings and constants for the Online-Translator application.
Includes paths, model settings, and application parameters.

Copyright (c) 2024 Kiko Cisneros
Licensed under the MIT License (see LICENSE file for details)
"""

# Audio settings
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHUNK_DURATION = 0.5
AUDIO_BLOCKSIZE = 1024

# Model settings
WHISPER_MODEL = "large"
TRANSLATION_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# MLX Translation settings for GPU optimization
MLX_GENERATION_CONFIG = {
    "temperature": 0.1,          # Low temperature for consistent translations
    "repetition_penalty": 1.1,   # Prevent repetition
    "repetition_context_size": 20, # Context for repetition detection
    "top_p": 0.9,               # Nucleus sampling
    "seed": None,               # Random seed for reproducibility (None = random)
}

# Dynamic token limits based on input length
MIN_TRANSLATION_TOKENS = 10
MAX_TRANSLATION_TOKENS = 100
TRANSLATION_TOKEN_MULTIPLIER = 2  # Output tokens = input_words * multiplier + MIN_TOKENS

# UI settings
WINDOW_WIDTH_RATIO = 0.8
WINDOW_HEIGHT = 220
WINDOW_OPACITY = 0.9
WINDOW_POSITION_Y_OFFSET = 50

# Font settings
ORIGINAL_TEXT_FONT = ('Arial', 12)
TRANSLATED_TEXT_FONT = ('Arial', 12)
LABEL_FONT = ('Arial', 10)

# Color settings
BACKGROUND_COLOR = 'black'
ORIGINAL_TEXT_COLOR = 'white'
TRANSLATED_TEXT_COLOR = '#F5F5DC'
LABEL_COLOR = 'lightgray'

# Language settings
DEFAULT_SOURCE_LANGUAGE = "auto"
DEFAULT_TARGET_LANGUAGE = "es"

# Language display names for UI
LANGUAGE_NAMES = {
    "auto": "Auto",
    "en": "English",
    "es": "Español", 
    "fr": "Français",
    "de": "Deutsch",
    "it": "Italiano",
    "pt": "Português",
    "ru": "Русский",
    "zh": "中文",
    "ja": "日本語",
    "ko": "한국어",
    "ar": "العربية",
    "hi": "हिन्दी",
    "th": "ไทย",
    "vi": "Tiếng Việt",
    "tr": "Türkçe",
    "pl": "Polski",
    "nl": "Nederlands",
    "sv": "Svenska",
    "da": "Dansk",
    "no": "Norsk",
    "fi": "Suomi",
    "cs": "Čeština",
    "hu": "Magyar",
    "ro": "Română",
    "bg": "Български",
    "hr": "Hrvatski",
    "sk": "Slovenčina",
    "sl": "Slovenščina",
    "et": "Eesti",
    "lv": "Latviešu",
    "lt": "Lietuvių",
    "mt": "Malti",
    "ca": "Català",
    "eu": "Euskera",
    "gl": "Galego",
    "cy": "Cymraeg",
    "ga": "Gaeilge",
    "is": "Íslenska",
    "mk": "Македонски",
    "sq": "Shqip",
    "sr": "Српски",
    "uk": "Українська",
    "be": "Беларуская",
    "kk": "Қазақша",
    "ky": "Кыргызча",
    "uz": "Oʻzbekcha",
    "tg": "Тоҷикӣ",
    "mn": "Монгол",
    "bn": "বাংলা",
    "te": "తెలుగు",
    "ta": "தமிழ்",
    "ml": "മലയാളം",
    "kn": "ಕನ್ನಡ",
    "gu": "ગુજરાતી",
    "pa": "ਪੰਜਾਬੀ",
    "ne": "नेपाली",
    "si": "සිංහල",
    "my": "မြန်မာ",
    "km": "ខ្មែរ",
    "lo": "ລາວ",
    "ka": "ქართული",
    "am": "አማርኛ",
    "he": "עברית",
    "fa": "فارسی",
    "ur": "اردو"
}

# Expanded language support
SUPPORTED_SOURCE_LANGUAGES = [
    "auto", "en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko", 
    "ar", "hi", "th", "vi", "tr", "pl", "nl", "sv", "da", "no", "fi", 
    "cs", "hu", "ro", "bg", "hr", "sk", "sl", "et", "lv", "lt", "mt",
    "ca", "eu", "gl", "cy", "ga", "is", "mk", "sq", "sr", "uk", "be",
    "kk", "ky", "uz", "tg", "mn", "bn", "te", "ta", "ml", "kn", "gu",
    "pa", "ne", "si", "my", "km", "lo", "ka", "am", "he", "fa", "ur"
]

SUPPORTED_TARGET_LANGUAGES = [
    "es", "en", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko", 
    "ar", "hi", "th", "vi", "tr", "pl", "nl", "sv", "da", "no", "fi", 
    "cs", "hu", "ro", "bg", "hr", "sk", "sl", "et", "lv", "lt", "mt",
    "ca", "eu", "gl", "cy", "ga", "is", "mk", "sq", "sr", "uk", "be",
    "kk", "ky", "uz", "tg", "mn", "bn", "te", "ta", "ml", "kn", "gu",
    "pa", "ne", "si", "my", "km", "lo", "ka", "am", "he", "fa", "ur"
]

# Processing settings
MAX_AUDIO_CHUNKS = 6
PROCESSING_TIMEOUT = 2.0
TRANSLATION_TEMPERATURE = 0.1
SILENCE_THRESHOLD = 0.0005
SILENCE_DURATION = 1.5
OVERLAP_DURATION = 0.2

# Debug settings
ENABLE_DEBUG_LOGGING = False
SHOW_PROCESSING_TIME = False 