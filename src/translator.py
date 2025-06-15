#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Translation Module
-----------------
Handles text translation functionality for the Online-Translator application.
Provides translation services using MLX models and manages translation context.

Copyright (c) 2024 Kiko Cisneros
Licensed under the MIT License (see LICENSE file for details)
"""

import time
import re
from collections import deque
from mlx_lm import load, generate
import mlx.core as mx
from . import config


class TextTranslator:
    """Handles text translation using MLX models"""
    
    def __init__(self):
        self.translation_model = None
        self.translation_tokenizer = None
        
        # Translation history for context
        self.translation_history = deque(maxlen=10)  # Keep last 10 translations for context
        
        # Timing statistics
        self.translation_times = deque(maxlen=20)  # Keep last 20 translation times
        self.model_load_times = deque(maxlen=5)   # Keep last 5 model load times
        self.total_translations = 0
        self.total_characters_translated = 0
        
        # MLX device info
        self.device = mx.default_device()
        print(f"🔧 MLX Translation using device: {self.device}")
        
        self.load_translation_model()
        
    def load_translation_model(self):
        """Load MLX translation model (Qwen) with GPU optimization"""
        try:
            print(f"🔄 Loading translation model: {config.TRANSLATION_MODEL}")
            print(f"🔧 Target device: {self.device}")
            load_start_time = time.time()
            
            # Clear MLX cache before loading
            mx.clear_cache()
            
            self.translation_model, self.translation_tokenizer = load(config.TRANSLATION_MODEL)
            
            load_time = time.time() - load_start_time
            self.model_load_times.append(load_time)
            
            # Check memory usage after loading
            try:
                peak_memory = mx.get_peak_memory() / 1024 / 1024  # Convert to MB
                cache_memory = mx.metal.get_cache_memory() / 1024 / 1024 if hasattr(mx.metal, 'get_cache_memory') else 0
                print(f"✅ Translation model loaded successfully in {load_time:.2f}s")
                print(f"🔧 GPU Memory - Peak: {peak_memory:.1f} MB, Cache: {cache_memory:.1f} MB")
            except Exception as e:
                print(f"✅ Translation model loaded successfully in {load_time:.2f}s")
                if config.ENABLE_DEBUG_LOGGING:
                    print(f"Memory info not available: {e}")
            
            if config.ENABLE_DEBUG_LOGGING:
                print(f"Model details: {config.TRANSLATION_MODEL}")
                print(f"Model device: {self.device}")
                
        except Exception as e:
            print(f"❌ Error loading translation model: {e}")
            print("🔄 Falling back to simple translation...")
            
    def translate_text(self, text, target_language="es", previous_text=None):
        """Translate text using MLX model with GPU optimization and optional context"""
        if not text.strip():
            return ""
            
        try:
            if self.translation_model is None:
                print(f"⚠️  Translation model not loaded, using fallback for: '{text[:30]}...'")
                return f"[{target_language.upper()}] {text}"
                
            # Start timing
            start_time = time.time()
            
            # Log translation request with context info
            context_info = f" (with context)" if previous_text else ""
            print(f"🌐 Starting translation{context_info}: '{text[:50]}{'...' if len(text) > 50 else ''}' -> {target_language.upper()}")
            
            # Comprehensive language code mapping
            language_names = {
                # Major languages
                "es": "Spanish", "en": "English", "fr": "French", "de": "German", 
                "it": "Italian", "pt": "Portuguese", "ru": "Russian", "zh": "Chinese",
                "ja": "Japanese", "ko": "Korean", "ar": "Arabic", "hi": "Hindi",
                
                # European languages
                "nl": "Dutch", "sv": "Swedish", "da": "Danish", "no": "Norwegian",
                "fi": "Finnish", "pl": "Polish", "cs": "Czech", "hu": "Hungarian",
                "ro": "Romanian", "bg": "Bulgarian", "hr": "Croatian", "sk": "Slovak",
                "sl": "Slovenian", "et": "Estonian", "lv": "Latvian", "lt": "Lithuanian",
                "mt": "Maltese", "cy": "Welsh", "ga": "Irish", "is": "Icelandic",
                
                # Regional Spanish/European languages
                "ca": "Catalan", "eu": "Basque", "gl": "Galician",
                
                # Slavic languages
                "mk": "Macedonian", "sq": "Albanian", "sr": "Serbian", 
                "uk": "Ukrainian", "be": "Belarusian",
                
                # Central Asian languages
                "kk": "Kazakh", "ky": "Kyrgyz", "uz": "Uzbek", "tg": "Tajik", "mn": "Mongolian",
                
                # South Asian languages
                "bn": "Bengali", "te": "Telugu", "ta": "Tamil", "ml": "Malayalam",
                "kn": "Kannada", "gu": "Gujarati", "pa": "Punjabi", "ne": "Nepali",
                "si": "Sinhala",
                
                # Southeast Asian languages
                "th": "Thai", "vi": "Vietnamese", "my": "Burmese", "km": "Khmer",
                "lo": "Lao", "tr": "Turkish",
                
                # Middle Eastern languages
                "fa": "Persian", "he": "Hebrew", "ka": "Georgian", "am": "Amharic",
                
                "auto": "auto-detected language"
            }
            
            target_lang_name = language_names.get(target_language, target_language.capitalize())
            
            # Try to detect source language from text features (simple heuristic)
            # This will be enhanced in future with proper language detection
            def detect_source_language(text):
                # Simple character set based detection for common languages
                # This is a basic heuristic and should be replaced with proper language detection
                text = text.lower()
                
                # Check for specific character sets
                if re.search(r'[а-яА-Я]', text):  # Cyrillic
                    return "ru"  # Russian as default for Cyrillic
                elif re.search(r'[一-龯]', text) or re.search(r'[\u4e00-\u9fff]', text):  # Chinese characters
                    return "zh"
                elif re.search(r'[ぁ-んァ-ン]', text):  # Japanese kana
                    return "ja"
                elif re.search(r'[가-힣]', text):  # Korean
                    return "ko"
                elif re.search(r'[א-ת]', text):  # Hebrew
                    return "he"
                elif re.search(r'[ا-ي]', text):  # Arabic
                    return "ar"
                elif re.search(r'[ก-๙]', text):  # Thai
                    return "th"
                # European languages
                elif re.search(r'[áàâäãåāăąèéêëēėęîïíīįìôöòóœøōõùúûüūğçćčñńňşšßžźż]', text):
                    # Check for specific European language markers
                    if re.search(r'[ñ]', text) and re.search(r'[áéíóú]', text):
                        return "es"  # Spanish
                    elif re.search(r'[àâçéèêëîïôùûüÿ]', text):
                        return "fr"  # French
                    elif re.search(r'[äöüß]', text):
                        return "de"  # German
                    elif re.search(r'[àèéìíîòóù]', text):
                        return "it"  # Italian
                    elif re.search(r'[ãõáéíóúçà]', text):
                        return "pt"  # Portuguese
                    else:
                        return "en"  # Default to English for Latin script
                else:
                    return "en"  # Default to English
            
            # Get source language
            source_lang = detect_source_language(text)
            source_lang_name = language_names.get(source_lang, "unknown language")
            
            # Build context-aware prompts
            def build_context_prompt(source_lang, target_lang, text, previous_text=None):
                """Build explicit prompts for all languages to avoid AI explanations"""
                
                # ✅ CONTEXTO MEJORADO - Incluir mensaje anterior pero traducir solo el segundo
                context_section = ""
                if previous_text and previous_text.strip():
                    context_section = f"""Contexto (mensaje anterior): "{previous_text.strip()}"

IMPORTANTE: El contexto anterior es SOLO para ayudarte a entender mejor el tema. ÚNICAMENTE traduce el texto que aparece después de "Texto a traducir:".

"""
                
                # ✅ PROMPTS ESPECÍFICOS PARA EVITAR EXPLICACIONES DE IA
                # Definir instrucciones estrictas por idioma de destino
                strict_instructions = {
                    "es": "SOLO traduce el texto marcado. NO agregues explicaciones, NOTAS, comentarios, aclaraciones, o cualquier texto adicional. NUNCA pongas NOTAS. ÚNICAMENTE la traducción.",
                    "en": "ONLY translate the marked text. DO NOT add explanations, NOTES, comments, clarifications, or any additional text. NEVER add NOTES. ONLY the translation.",
                    "fr": "SEULEMENT traduisez le texte marqué. N'ajoutez PAS d'explications, de NOTES, de commentaires, de clarifications ou de texte supplémentaire. JAMAIS de NOTES. SEULEMENT la traduction.",
                    "de": "NUR den markierten Text übersetzen. Fügen Sie KEINE Erklärungen, NOTIZEN, Kommentare, Erläuterungen oder zusätzlichen Text hinzu. NIEMALS NOTIZEN. NUR die Übersetzung.",
                    "it": "SOLO traduci il testo contrassegnato. NON aggiungere spiegazioni, NOTE, commenti, chiarimenti o testo aggiuntivo. MAI NOTE. SOLO la traduzione.",
                    "pt": "APENAS traduza o texto marcado. NÃO adicione explicações, NOTAS, comentários, esclarecimentos ou texto adicional. NUNCA NOTAS. APENAS a tradução.",
                    "ru": "ТОЛЬКО переведите отмеченный текст. НЕ добавляйте объяснений, ПРИМЕЧАНИЙ, комментариев, разъяснений или дополнительного текста. НИКОГДА ПРИМЕЧАНИЙ. ТОЛЬКО перевод.",
                    "zh": "仅翻译标记的文本。不要添加解释、注释、评论、说明或任何额外文本。绝不添加注释。仅翻译。",
                    "ja": "マークされたテキストのみを翻訳してください。説明、注釈、コメント、解説、追加テキストは一切追加しないでください。絶対に注釈を追加しないでください。翻訳のみ。",
                    "ko": "표시된 텍스트만 번역하세요. 설명, 메모, 댓글, 해설 또는 추가 텍스트를 추가하지 마세요. 절대 메모를 추가하지 마세요. 번역만.",
                    "ar": "ترجم النص المحدد فقط. لا تضيف تفسيرات أو ملاحظات أو تعليقات أو توضيحات أو أي نص إضافي. لا تضيف ملاحظات أبداً. الترجمة فقط.",
                    "hi": "केवल चिह्नित पाठ का अनुवाद करें। कोई स्पष्टीकरण, नोट्स, टिप्पणी, स्पष्टीकरण या अतिरिक्त पाठ न जोड़ें। कभी भी नोट्स न जोड़ें। केवल अनुवाद।",
                    "nl": "Vertaal ALLEEN de gemarkeerde tekst. Voeg GEEN uitleg, NOTITIES, commentaar, toelichtingen of extra tekst toe. NOOIT NOTITIES. ALLEEN de vertaling.",
                    "sv": "Översätt ENDAST den markerade texten. Lägg INTE till förklaringar, ANTECKNINGAR, kommentarer, förtydliganden eller extra text. ALDRIG ANTECKNINGAR. ENDAST översättningen.",
                    "da": "Oversæt KUN den markerede tekst. Tilføj IKKE forklaringer, NOTER, kommentarer, præciseringer eller ekstra tekst. ALDRIG NOTER. KUN oversættelsen.",
                    "no": "Oversett KUN den merkede teksten. Ikke legg til forklaringer, NOTATER, kommentarer, avklaringer eller ekstra tekst. ALDRI NOTATER. KUN oversettelsen.",
                    "fi": "Käännä VAIN merkitty teksti. Älä lisää selityksiä, MUISTIINPANOJA, kommentteja, selvennyksiä tai ylimääräistä tekstiä. EI KOSKAAN MUISTIINPANOJA. VAIN käännös.",
                    "pl": "Przetłumacz TYLKO oznaczony tekst. NIE dodawaj wyjaśnień, NOTATEK, komentarzy, wyjaśnień ani dodatkowego tekstu. NIGDY NOTATEK. TYLKO tłumaczenie.",
                    "cs": "Přeložte POUZE označený text. NEPŘIDÁVEJTE vysvětlení, POZNÁMKY, komentáře, objasnění nebo další text. NIKDY POZNÁMKY. POUZE překlad.",
                    "hu": "CSAK a megjelölt szöveget fordítsd le. NE adj hozzá magyarázatokat, JEGYZETEKET, megjegyzéseket, magyarázatokat vagy további szöveget. SOHA JEGYZETEKET. CSAK a fordítás.",
                    "ro": "Traduceți DOAR textul marcat. NU adăugați explicații, NOTE, comentarii, clarificări sau text suplimentar. NICIODATĂ NOTE. DOAR traducerea.",
                    "tr": "SADECE işaretli metni çevirin. Açıklama, NOTLAR, yorumlar, açıklamalar veya ek metin eklemeyin. ASLA NOT EKLEMEYİN. SADECE çeviri."
                }
                
                # Obtener instrucción estricta
                instruction = strict_instructions.get(target_lang, "ONLY translate the marked text. DO NOT add explanations, notes, comments, clarifications, or any additional text. ONLY the translation.")
                
                # ✅ PROMPTS ESPECÍFICOS POR PAR DE IDIOMAS CON CONTEXTO MEJORADO
                # Inglés -> Español
                if source_lang == "en" and target_lang == "es":
                    return f"""{context_section}Instrucciones: {instruction}

Texto a traducir (inglés): "{text}"

Traducción al español:"""
                
                # Español -> Inglés
                elif source_lang == "es" and target_lang == "en":
                    return f"""{context_section}Instructions: {instruction}

Text to translate (Spanish): "{text}"

English translation:"""
                
                # Inglés -> Francés
                elif source_lang == "en" and target_lang == "fr":
                    return f"""{context_section}Instructions: {instruction}

Text to translate (English): "{text}"

French translation:"""
                
                # Francés -> Inglés
                elif source_lang == "fr" and target_lang == "en":
                    return f"""{context_section}Instructions: {instruction}

Texte à traduire (français): "{text}"

English translation:"""
                
                # Inglés -> Alemán
                elif source_lang == "en" and target_lang == "de":
                    return f"""{context_section}Instructions: {instruction}

Text to translate (English): "{text}"

German translation:"""
                
                # Alemán -> Inglés
                elif source_lang == "de" and target_lang == "en":
                    return f"""{context_section}Instructions: {instruction}

Zu übersetzender Text (Deutsch): "{text}"

English translation:"""
                
                # Inglés -> Italiano
                elif source_lang == "en" and target_lang == "it":
                    return f"""{context_section}Instructions: {instruction}

Text to translate (English): "{text}"

Italian translation:"""
                
                # Italiano -> Inglés
                elif source_lang == "it" and target_lang == "en":
                    return f"""{context_section}Instructions: {instruction}

Testo da tradurre (italiano): "{text}"

English translation:"""
                
                # Inglés -> Portugués
                elif source_lang == "en" and target_lang == "pt":
                    return f"""{context_section}Instructions: {instruction}

Text to translate (English): "{text}"

Portuguese translation:"""
                
                # Portugués -> Inglés
                elif source_lang == "pt" and target_lang == "en":
                    return f"""{context_section}Instructions: {instruction}

Texto para traduzir (português): "{text}"

English translation:"""
                
                # Inglés -> Ruso
                elif source_lang == "en" and target_lang == "ru":
                    return f"""{context_section}Instructions: {instruction}

Text to translate (English): "{text}"

Russian translation:"""
                
                # Ruso -> Inglés
                elif source_lang == "ru" and target_lang == "en":
                    return f"""{context_section}Instructions: {instruction}

Текст для перевода (русский): "{text}"

English translation:"""
                
                # Inglés -> Chino
                elif source_lang == "en" and target_lang == "zh":
                    return f"""{context_section}Instructions: {instruction}

Text to translate (English): "{text}"

Chinese translation:"""
                
                # Chino -> Inglés
                elif source_lang == "zh" and target_lang == "en":
                    return f"""{context_section}Instructions: {instruction}

要翻译的文本 (中文): "{text}"

English translation:"""
                
                # Inglés -> Japonés
                elif source_lang == "en" and target_lang == "ja":
                    return f"""{context_section}Instructions: {instruction}

Text to translate (English): "{text}"

Japanese translation:"""
                
                # Japonés -> Inglés
                elif source_lang == "ja" and target_lang == "en":
                    return f"""{context_section}Instructions: {instruction}

翻訳するテキスト (日本語): "{text}"

English translation:"""
                
                # Inglés -> Coreano
                elif source_lang == "en" and target_lang == "ko":
                    return f"""{context_section}Instructions: {instruction}

Text to translate (English): "{text}"

Korean translation:"""
                
                # Coreano -> Inglés
                elif source_lang == "ko" and target_lang == "en":
                    return f"""{context_section}Instructions: {instruction}

번역할 텍스트 (한국어): "{text}"

English translation:"""
                
                # Inglés -> Árabe
                elif source_lang == "en" and target_lang == "ar":
                    return f"""{context_section}Instructions: {instruction}

Text to translate (English): "{text}"

Arabic translation:"""
                
                # Árabe -> Inglés
                elif source_lang == "ar" and target_lang == "en":
                    return f"""{context_section}Instructions: {instruction}

النص المراد ترجمته (العربية): "{text}"

English translation:"""
                
                # Inglés -> Hindi
                elif source_lang == "en" and target_lang == "hi":
                    return f"""{context_section}Instructions: {instruction}

Text to translate (English): "{text}"

Hindi translation:"""
                
                # Hindi -> Inglés
                elif source_lang == "hi" and target_lang == "en":
                    return f"""{context_section}Instructions: {instruction}

अनुवाद करने के लिए पाठ (हिंदी): "{text}"

English translation:"""
                
                # Inglés -> Holandés
                elif source_lang == "en" and target_lang == "nl":
                    return f"""{context_section}Instructions: {instruction}

Text to translate (English): "{text}"

Dutch translation:"""
                
                # Holandés -> Inglés
                elif source_lang == "nl" and target_lang == "en":
                    return f"""{context_section}Instructions: {instruction}

Te vertalen tekst (Nederlands): "{text}"

English translation:"""
                
                # Inglés -> Sueco
                elif source_lang == "en" and target_lang == "sv":
                    return f"""{context_section}Instructions: {instruction}

Text to translate (English): "{text}"

Swedish translation:"""
                
                # Sueco -> Inglés
                elif source_lang == "sv" and target_lang == "en":
                    return f"""{context_section}Instructions: {instruction}

Text att översätta (svenska): "{text}"

English translation:"""
                
                # Inglés -> Turco
                elif source_lang == "en" and target_lang == "tr":
                    return f"""{context_section}Instructions: {instruction}

Text to translate (English): "{text}"

Turkish translation:"""
                
                # Turco -> Inglés
                elif source_lang == "tr" and target_lang == "en":
                    return f"""{context_section}Instructions: {instruction}

Çevrilecek metin (Türkçe): "{text}"

English translation:"""
                
                # Español -> otros idiomas
                elif source_lang == "es" and target_lang != "en":
                    return f"""{context_section}Instructions: {instruction}

Texto a traducir (español): "{text}"

Traducción al {target_lang_name}:"""
                
                # Otros idiomas -> Español
                elif target_lang == "es" and source_lang != "en":
                    return f"""{context_section}Instrucciones: {instruction}

Texto a traducir ({source_lang_name}): "{text}"

Traducción al español:"""
                
                # Caso genérico con instrucciones estrictas
                else:
                    return f"""{context_section}Instructions: {instruction}

Text to translate ({source_lang_name}): "{text}"

Translation to {target_lang_name}:"""
            
            # Build the prompt with context
            prompt = build_context_prompt(source_lang, target_language, text, previous_text)
            
            if config.ENABLE_DEBUG_LOGGING:
                print(f"Using context-aware prompt for {source_lang}->{target_language}")
                if previous_text:
                    print(f"Context: '{previous_text[:30]}...'")
            
            # Measure prompt preparation time
            prompt_time = time.time() - start_time
            
            # Optimized generation parameters for GPU performance
            try:
                # Start model generation timing
                generation_start_time = time.time()
                
                # Generate translation with conservative parameters
                response = generate(
                    self.translation_model,
                    self.translation_tokenizer,
                    prompt=prompt,
                    max_tokens=min(200, len(text.split()) * 8),  # ✅ Aumentado de 120 a 200 y de 6x a 8x
                    verbose=False
                )
                
                generation_time = time.time() - generation_start_time
                
                if config.ENABLE_DEBUG_LOGGING:
                    print(f"Translation raw response: '{response}'")
                    print(f"🔧 Generation time: {generation_time:.3f}s on {self.device}")
                
                # Start post-processing timing
                postprocess_start_time = time.time()
                
                # Enhanced cleanup for context-aware translations
                translated = response.strip()
                
                # Remove common end tokens and artifacts
                end_tokens = ["<|endoftext|>", "</s>", "<|end|>", "<|im_end|>", "[INST]", "[/INST]"]
                for token in end_tokens:
                    if token in translated:
                        translated = translated.split(token)[0].strip()
                
                # ✅ DETECTAR Y REMOVER EXPLICACIONES DE IA
                explanation_patterns = [
                    # Inglés
                    r"This translation conveys.*",
                    r"This conveys.*",
                    r"The translation.*",
                    r"This translates to.*",
                    r"This means.*",
                    r"In Spanish.*",
                    r"In English.*",
                    r"Note:.*",
                    r"Note that.*",
                    r"It should be noted.*",
                    r"Please note.*",
                    r"This phrase.*",
                    r"The phrase.*",
                    r"The meaning.*",
                    r"This expression.*",
                    r"In this context.*",
                    r"Here.*translation.*",
                    r"The above.*",
                    r"This captures.*",
                    r"This maintains.*",
                    r"This preserves.*",
                    r"This reflects.*",
                    
                    # Español
                    r"Esta traducción.*",
                    r"Esta frase.*",
                    r"El significado.*",
                    r"En español.*",
                    r"En inglés.*",
                    r"Nota:.*",
                    r"Cabe señalar.*",
                    r"Es importante.*",
                    r"La traducción.*",
                    r"Esto significa.*",
                    r"Esta expresión.*",
                    r"En este contexto.*",
                    r"La frase.*",
                    r"Esto transmite.*",
                    r"Esto mantiene.*",
                    r"Esto preserva.*",
                    r"Esto refleja.*",
                    
                    # Francés
                    r"Cette traduction.*",
                    r"Cette phrase.*",
                    r"Le sens.*",
                    r"En français.*",
                    r"En anglais.*",
                    r"Note:.*",
                    r"Il convient.*",
                    r"La traduction.*",
                    r"Cela signifie.*",
                    r"Cette expression.*",
                    
                    # Alemán
                    r"Diese Übersetzung.*",
                    r"Dieser Satz.*",
                    r"Die Bedeutung.*",
                    r"Auf Deutsch.*",
                    r"Auf Englisch.*",
                    r"Hinweis:.*",
                    r"Es ist wichtig.*",
                    r"Die Übersetzung.*",
                    
                    # Italiano
                    r"Questa traduzione.*",
                    r"Questa frase.*",
                    r"Il significato.*",
                    r"In italiano.*",
                    r"In inglese.*",
                    r"Nota:.*",
                    r"È importante.*",
                    r"La traduzione.*",
                    
                    # Portugués
                    r"Esta tradução.*",
                    r"Esta frase.*",
                    r"O significado.*",
                    r"Em português.*",
                    r"Em inglês.*",
                    r"Nota:.*",
                    r"É importante.*",
                    r"A tradução.*",
                    
                    # Patrones genéricos
                    r"\(.*explains.*\)",
                    r"\(.*explanation.*\)",
                    r"\(.*translation.*\)",
                    r"\(.*note.*\)",
                    r"\(.*nota.*\)",
                    r"\".*translation.*\"$",
                    r"\".*explains.*\"$",
                    r"\".*conveys.*\"$",
                    r"\".*transmite.*\"$",
                    r"\".*significa.*\"$",
                    r"^(This|Esta|Cette|Diese|Questa|Esta)\s+(translation|traducción|traduction|Übersetzung|traduzione|tradução).*",
                    
                    # Patrones de fin de explicación
                    r"\..*This is.*$",
                    r"\..*Esto es.*$",
                    r"\..*C'est.*$",
                    r"\..*Das ist.*$",
                    r"\..*Questo è.*$",
                    r"\..*Isto é.*$"
                ]
                
                for pattern in explanation_patterns:
                    # Buscar y remover explicaciones al final
                    match = re.search(pattern, translated, re.IGNORECASE | re.DOTALL)
                    if match:
                        translated = translated[:match.start()].strip()
                        print(f"🧹 Removed AI explanation: '{match.group()[:30]}...'")
                        break
                
                # ✅ DETECTAR Y LIMPIAR REPETICIONES EXCESIVAS MEJORADO
                # Detectar repeticiones de signos de puntuación
                punct_repetition = r'([!?.,;:\-_=+])\1{10,}'
                if re.search(punct_repetition, translated):
                    translated = re.sub(punct_repetition, r'\1', translated)
                    print(f"🧹 Removed excessive punctuation repetition")
                
                # Detectar repeticiones de palabras/frases
                word_repetition = r'\b(\w+(?:\s+\w+){0,2})\s+(?:\1\s+){3,}'
                if re.search(word_repetition, translated, re.IGNORECASE):
                    # Tomar solo la primera ocurrencia antes de la repetición
                    match = re.search(word_repetition, translated, re.IGNORECASE)
                    if match:
                        translated = translated[:match.start()].strip()
                        print(f"🧹 Removed excessive word repetition pattern")
                
                # Remove training artifacts and prompts that sometimes leak through
                cleanup_prefixes = [
                    "Human:", "Assistant:", "Translation:", "Traducción:", "Traduction:", 
                    "Übersetzung:", "Traduzione:", "Tradução:", "Перевод:", "翻译:", 
                    "To translate", "Para traducir", "Pour traduire", "Um zu übersetzen",
                    "I'll translate", "Voy a traducir", "Je vais traduire",
                    "The translation", "La traducción", "La traduction"
                ]
                
                for prefix in cleanup_prefixes:
                    if prefix in translated:
                        translated = translated.split(prefix)[0].strip()
                
                # Language-specific cleanup - remove prompt markers
                lang_markers = {
                    "Spanish:": "", "English:": "", "French:": "", "German:": "", 
                    "Italian:": "", "Portuguese:": "", "Russian:": "", "Chinese:": "",
                    "Japanese:": "", "Korean:": "", "Arabic:": "", "Hindi:": "",
                    "Español:": "", "Inglés:": "", "Francés:": "", "Alemán:": "",
                    "Italiano:": "", "Portugués:": "", "Ruso:": "", "Chino:": ""
                }
                
                for marker in lang_markers:
                    if translated.startswith(marker):
                        translated = translated[len(marker):].strip()
                        break
                    # Also check for lowercase versions
                    if translated.startswith(marker.lower()):
                        translated = translated[len(marker):].strip()
                        break
                
                # Remove quotes if the entire translation is wrapped in them
                if ((translated.startswith('"') and translated.endswith('"')) or 
                    (translated.startswith("'") and translated.endswith("'"))):
                    translated = translated[1:-1].strip()
                
                # ✅ MANEJO MEJORADO DE ORACIONES MÚLTIPLES - MENOS AGRESIVO
                # Solo cortar si hay evidencia clara de que es un error (muchas oraciones muy cortas)
                sentences = re.split(r'[.!?]+\s+', translated)
                if len(sentences) > 4:  # Más de 4 oraciones, revisar
                    # Contar oraciones sustanciales (más de 5 palabras)
                    substantial_sentences = [s for s in sentences if len(s.split()) > 5]
                    short_sentences = [s for s in sentences if len(s.split()) <= 5]
                    
                    # Si hay muchas oraciones cortas vs sustanciales, posible error
                    if len(short_sentences) > len(substantial_sentences) and len(short_sentences) > 2:
                        # Tomar solo las primeras 2-3 oraciones sustanciales
                        if substantial_sentences:
                            translated = '. '.join(substantial_sentences[:3])
                    if not translated.endswith(('.', '!', '?')):
                            translated += '.'
                            print(f"🧹 Trimmed excessive short sentences")
                
                # ✅ DETECTAR Y ELIMINAR "NOTA:" O "NOTE:" AL FINAL ESPECÍFICAMENTE
                # Eliminar patrones específicos de notas al final de la traducción
                note_patterns_at_end = [
                    r'\s*NOTA\s*[:.].*$',  # NOTA: o NOTA. al final
                    r'\s*NOTE\s*[:.].*$',  # NOTE: o NOTE. al final
                    r'\s*Nota\s*[:.].*$',  # Nota: o Nota. al final
                    r'\s*Note\s*[:.].*$',  # Note: o Note. al final
                    r'\s*NOTES\s*[:.].*$', # NOTES: o NOTES. al final
                    r'\s*Notes\s*[:.].*$', # Notes: o Notes. al final
                    r'\s*NOTAS\s*[:.].*$', # NOTAS: o NOTAS. al final
                    r'\s*Notas\s*[:.].*$', # Notas: o Notas. al final
                    r'\s*\(NOTA\).*$',     # (NOTA) al final
                    r'\s*\(NOTE\).*$',     # (NOTE) al final
                    r'\s*\(Nota\).*$',     # (Nota) al final
                    r'\s*\(Note\).*$',     # (Note) al final
                ]
                
                for pattern in note_patterns_at_end:
                    before_cleanup = translated
                    translated = re.sub(pattern, '', translated, flags=re.IGNORECASE).strip()
                    if before_cleanup != translated:
                        print(f"🧹 Removed NOTE pattern at end: '{before_cleanup[len(translated):].strip()}'")
                        break  # Solo eliminar el primer patrón encontrado
                
                # Final cleanup
                translated = translated.strip(' "\'`.,:-*()[]{}')
                
                # ✅ AGREGAR PUNTO FINAL SI ES NECESARIO
                if len(translated) > 10 and not translated.endswith(('.', '!', '?', ':')):
                    translated += '.'
                
                postprocess_time = time.time() - postprocess_start_time
                
                # Enhanced validation
                if not translated or len(translated) < 2:
                    total_time = time.time() - start_time
                    print(f"⚠️  Translation too short: '{translated}' (Total: {total_time:.2f}s)")
                    if config.ENABLE_DEBUG_LOGGING:
                        print(f"Translation too short: '{translated}'")
                    return f"[{target_language.upper()}] {text}"
                
                # ✅ VALIDACIÓN MEJORADA PARA DETECTAR TRADUCCIONES SIN SENTIDO
                if self._is_nonsensical_translation(translated, text):
                    total_time = time.time() - start_time
                    print(f"⚠️  Nonsensical translation detected: '{translated}' (Total: {total_time:.2f}s)")
                    return f"[{target_language.upper()}] {text}"
                
                # Don't return if it's exactly the same as input (unless it's a proper noun or very short)
                if (translated.lower().strip() == text.lower().strip() and 
                    len(text.split()) > 1 and 
                    not text[0].isupper()):  # Allow proper nouns to remain the same
                    total_time = time.time() - start_time
                    print(f"⚠️  Translation same as original: '{translated}' (Total: {total_time:.2f}s)")
                    if config.ENABLE_DEBUG_LOGGING:
                        print(f"Translation same as original: '{translated}'")
                    return f"[{target_language.upper()}] {text}"
                
                # Calculate final timing and statistics
                total_time = time.time() - start_time
                self.translation_times.append(total_time)
                self.total_translations += 1
                self.total_characters_translated += len(text)
                
                # Calculate statistics
                avg_time = sum(self.translation_times) / len(self.translation_times)
                chars_per_second = len(text) / total_time if total_time > 0 else 0
                
                # Enhanced timing logs with GPU info and context info
                context_marker = " (+context)" if previous_text else ""
                print(f"✅ Translation complete{context_marker}: '{translated[:50]}{'...' if len(translated) > 50 else ''}'")
                print(f"⏱️  Timing - Prompt: {prompt_time:.3f}s | Generation: {generation_time:.3f}s | Post-proc: {postprocess_time:.3f}s | Total: {total_time:.2f}s")
                print(f"🔧 Device: {self.device} | Performance: {chars_per_second:.1f} chars/sec")
                print(f"📈 Stats - Avg: {avg_time:.2f}s | Count: {self.total_translations} | Total chars: {self.total_characters_translated}")
                
                if config.ENABLE_DEBUG_LOGGING:
                    print(f"Final translation: '{text}' -> '{translated}'")
                
                return translated
                
            except Exception as gen_error:
                total_time = time.time() - start_time
                print(f"❌ Generation error after {total_time:.2f}s on {self.device}: {gen_error}")
                if config.ENABLE_DEBUG_LOGGING:
                    print(f"Generation error: {gen_error}")
                return f"[{target_language.upper()}] {text}"
                
        except Exception as e:
            total_time = time.time() - start_time if 'start_time' in locals() else 0
            print(f"❌ Translation error after {total_time:.2f}s: {e}")
            if config.ENABLE_DEBUG_LOGGING:
                print(f"Translation error: {e}")
            return f"[{target_language.upper()}] {text}"
        
        finally:
            # Optimize GPU memory after each translation if enabled
            if hasattr(config, 'MLX_CACHE_OPTIMIZATION') and config.MLX_CACHE_OPTIMIZATION:
                if self.total_translations % 10 == 0:  # Clear cache every 10 translations
                    self.optimize_gpu_memory()

    def translate_with_context(self, text, target_language="es", use_auto_context=False):
        """Translate text with automatic context from previous translations"""
        
        # Get previous translation for context if enabled
        previous_text = None
        if use_auto_context and self.translation_history:
            # Use the most recent translation as context
            previous_entry = self.translation_history[-1]
            previous_text = previous_entry.get('original_text', '')
            
            # ✅ SER MÁS CONSERVADOR CON EL CONTEXTO
            # Solo usar contexto si es muy relevante y no muy largo
            if len(previous_text.split()) > 2 and len(previous_text.split()) < 15:
                # Verificar que el contexto anterior sea del mismo idioma aproximadamente
                current_words = set(text.lower().split()[:5])  # Primeras 5 palabras
                previous_words = set(previous_text.lower().split()[:5])  # Primeras 5 palabras
                
                # Solo usar contexto si hay alguna palabra en común (mismo tema)
                if current_words & previous_words:  # Intersección no vacía
                    if config.ENABLE_DEBUG_LOGGING:
                        print(f"📝 Using automatic context: '{previous_text[:30]}...'")
                else:
                    previous_text = None  # No usar contexto si no hay relación
            else:
                previous_text = None  # No usar contexto si es muy corto o muy largo
        
        # Perform the translation
        translated = self.translate_text(text, target_language, previous_text)
        
        # Store in history
        history_entry = {
            'original_text': text,
            'translated_text': translated,
            'target_language': target_language,
            'timestamp': time.time(),
            'used_context': previous_text is not None
        }
        self.translation_history.append(history_entry)
        
        return translated
    
    def get_translation_history(self, limit=5):
        """Get recent translation history"""
        recent_history = list(self.translation_history)[-limit:]
        return recent_history
    
    def clear_translation_history(self):
        """Clear translation history"""
        self.translation_history.clear()
        print("🗑️  Translation history cleared")

    def get_gpu_performance_info(self):
        """Get current GPU performance and memory information"""
        try:
            device_info = {
                "device": str(self.device),
                "peak_memory_mb": 0,
                "cache_memory_mb": 0,
                "is_gpu": str(self.device).startswith("Device(gpu")
            }
            
            try:
                device_info["peak_memory_mb"] = mx.get_peak_memory() / 1024 / 1024
            except:
                pass
                
            try:
                # Use the non-deprecated function
                device_info["cache_memory_mb"] = mx.get_cache_memory() / 1024 / 1024
            except:
                try:
                    # Fallback to deprecated function if new one doesn't exist
                    device_info["cache_memory_mb"] = mx.metal.get_cache_memory() / 1024 / 1024
                except:
                    pass
                
            return device_info
        except Exception as e:
            if config.ENABLE_DEBUG_LOGGING:
                print(f"Error getting GPU info: {e}")
            return {"device": str(self.device), "peak_memory_mb": 0, "cache_memory_mb": 0, "is_gpu": True}
    
    def optimize_gpu_memory(self):
        """Optimize GPU memory usage by clearing caches"""
        try:
            mx.clear_cache()
            if config.ENABLE_DEBUG_LOGGING:
                print("🔧 MLX cache cleared for memory optimization")
        except Exception as e:
            if config.ENABLE_DEBUG_LOGGING:
                print(f"Error clearing MLX cache: {e}") 

    def _is_nonsensical_translation(self, translation, original):
        # ✅ VALIDACIÓN MEJORADA PARA DETECTAR TRADUCCIONES SIN SENTIDO
        
        # Detectar mezclas de idiomas obvias
        mixed_language_patterns = [
            r"c'è\s+la\s+",  # Italiano mezclado con otro idioma
            r"\w+\s+la\s+armi",  # Patrones específicos sin sentido
            r"[а-я]+\s+[a-z]+\s+[а-я]+",  # Cirílico mezclado
            r"[一-龯]\s+[a-z]+\s+[一-龯]",  # Chino mezclado
        ]
        
        for pattern in mixed_language_patterns:
            if re.search(pattern, translation.lower()):
                return True
        
        # ✅ VALIDACIÓN MEJORADA DE LONGITUD - ser menos estricto
        original_words = len(original.split())
        translated_words = len(translation.split())
        
        # Solo marcar como sin sentido si es MUY corta (menos del 15%) Y el original es largo
        if (translated_words < original_words * 0.15 and 
            original_words > 8 and 
            translated_words < 3):  # Muy restrictivo
            return True
        
        # Detectar si solo hay puntuación o símbolos
        clean_translation = re.sub(r'[^\w\s]', '', translation).strip()
        if len(clean_translation) < 2:
            return True
        
        return False 