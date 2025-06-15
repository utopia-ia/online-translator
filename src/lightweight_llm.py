#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Lightweight LLM Module
---------------------
Provides lightweight language model functionality for semantic validation
and text processing in the Online-Translator application.

Copyright (c) 2024 Kiko Cisneros
Licensed under the MIT License (see LICENSE file for details)
"""

import re
import numpy as np
from collections import Counter, deque


class LightweightLLM:
    """Sistema de concatenación simple basado en distancia de Mahalanobis palabra por palabra"""
    
    def __init__(self):
        self.model_loaded = False
        self.model = None
        self.tokenizer = None
        # ✅ SIMPLIFIED APPROACH - Word-by-word Mahalanobis distance
        self.min_text_length = 3
        self.word_history = []  # Store recent words for Mahalanobis calculation
        self.max_history = 50   # Keep last 50 words
        print("🧠 Initializing Mahalanobis word-distance system...")
        print("✅ Word-distance system ready")
    
    def smart_concatenate_with_mahalanobis(self, accumulated_text, new_text):
        """🧮 SIMPLE: Concatenación simple basada en solapamientos palabra por palabra"""
        if not new_text or not new_text.strip():
            return accumulated_text, False, "Empty new text"
        
        new_text_clean = new_text.strip()
        
        # ✅ 1. DETECTAR SOLAPAMIENTOS PALABRA POR PALABRA
        if accumulated_text:
            cleaned_text = self._find_and_merge_overlap(accumulated_text, new_text_clean)
        else:
            cleaned_text = new_text_clean
        
        # ✅ 2. SOLO COMPLETAR SI LA PUNTUACIÓN ESTÁ EN MEDIO DE LA SIGUIENTE FRASE
        # No al final, sino EN MEDIO
        should_complete = self._has_punctuation_in_middle(cleaned_text)
        
        # ✅ 3. SAFETY VALVE: Textos muy largos - MÁS PERMISIVO
        if len(cleaned_text.split()) > 80:  # Incrementado de 50 a 80
            print(f"⚠️ Forzando completado por texto muy largo ({len(cleaned_text.split())} palabras)")
            return cleaned_text, True, "Mahalanobis + complete"
        
        return cleaned_text, should_complete, f"Mahalanobis + {'complete' if should_complete else 'continue'}"
    
    def _find_and_merge_overlap(self, accumulated_text, new_text):
        """Encuentra solapamientos palabra por palabra y usa la palabra más larga"""
        acc_words = accumulated_text.split()
        new_words = new_text.split()
        
        if not acc_words or not new_words:
            return accumulated_text + " " + new_text
        
        # ✅ BUSCAR SOLAPAMIENTO EMPEZANDO DESDE EL FINAL DE ACCUMULATED
        best_overlap_size = 0
        best_overlap_pos = 0
        
        # Comprobar hasta 6 palabras de solapamiento máximo
        max_check = min(6, len(acc_words), len(new_words))
        
        for overlap_size in range(1, max_check + 1):
            acc_end = acc_words[-overlap_size:]  # Últimas palabras de accumulated
            new_start = new_words[:overlap_size]  # Primeras palabras de new
            
            # Comprobar si son iguales (o muy similares con Mahalanobis)
            if self._words_match_with_mahalanobis(acc_end, new_start):
                best_overlap_size = overlap_size
                best_overlap_pos = overlap_size
                print(f"🔗 Solapamiento detectado: {overlap_size} palabras: {' '.join(acc_end)}")
        
        # ✅ MERGE CON LA PALABRA MÁS LARGA
        if best_overlap_size > 0:
            # Palabras que no se solapan de accumulated
            non_overlap_acc = acc_words[:-best_overlap_size]
            
            # Para cada palabra en el solapamiento, elegir la más larga
            merged_overlap = []
            for i in range(best_overlap_size):
                acc_word = acc_words[-(best_overlap_size-i)]
                new_word = new_words[i]
                
                # Elegir la palabra más larga
                longer_word = new_word if len(new_word) > len(acc_word) else acc_word
                merged_overlap.append(longer_word)
                
                if len(new_word) != len(acc_word):
                    print(f"🔗 Palabra más larga elegida: '{acc_word}' vs '{new_word}' -> '{longer_word}'")
            
            # Palabras que no se solapan de new
            non_overlap_new = new_words[best_overlap_size:]
            
            # Combinar todo
            result_words = non_overlap_acc + merged_overlap + non_overlap_new
            result = ' '.join(result_words)
            
            print(f"🧹 Merged con solapamiento: '{result}'")
            return result
        else:
            # Sin solapamiento, concatenación simple
            print(f"🔗 Sin solapamiento detectado, concatenación simple")
            return accumulated_text + " " + new_text
    
    def _words_match_with_mahalanobis(self, words1, words2):
        """Comprueba si dos listas de palabras coinciden usando distancia de Mahalanobis"""
        if len(words1) != len(words2):
            return False
        
        for w1, w2 in zip(words1, words2):
            distance = self._calculate_word_mahalanobis_distance(w1.lower(), w2.lower())
            # Umbral más permisivo para detectar palabras similares
            if distance > 0.15:  # Si la distancia es mayor, no coinciden
                return False
        
        return True
    
    def _has_punctuation_in_middle(self, text):
        """Detecta si hay puntuación EN MEDIO de la frase, no al final - MÁS CONSERVADOR"""
        if not text:
            return False
        
        text = text.strip()
        
        # ✅ BÚSQUEDA MÁS ESTRICTA DE PUNTUACIÓN EN MEDIO
        # Solo buscar puntuación que claramente indica final de oración completa
        
        # Remover puntuación del final para buscar en el medio
        text_without_end_punct = text.rstrip('.!?;: ')
        
        # ✅ SOLO COMPLETAR SI HAY MÚLTIPLES ORACIONES CLARAS
        # Contar cuántas oraciones completas hay
        sentence_endings = text_without_end_punct.count('.') + text_without_end_punct.count('!') + text_without_end_punct.count('?')
        
        # Solo completar si hay al menos 2 oraciones claramente separadas
        if sentence_endings >= 2:
            print(f"✅ Múltiples oraciones detectadas ({sentence_endings}): completando")
            return True
        
        # ✅ FRASES DE DESPEDIDA EXPLÍCITAS (siempre completar)
        goodbye_phrases = [
            'thank you', 'thanks', 'goodbye', 'bye', 'see you later',
            'thanks for watching', 'thanks for listening', 'that\'s it', 'that\'s all'
        ]
        
        text_lower = text.lower()
        for phrase in goodbye_phrases:
            if phrase in text_lower:
                print(f"✅ Frase de despedida detectada: '{phrase}'")
                return True
        
        # ✅ MÁS CONSERVADOR: No completar en otros casos
        return False
    
    def _detect_repetitive_transcription(self, text):
        """Detecta si Whisper generó bucle repetitivo (bug conocido)"""
        if not text or len(text.strip()) < 20:
            return False
        
        words = text.lower().split()
        if len(words) < 10:
            return False
        
        # ✅ MÉTODO 1: Detectar palabras consecutivas repetidas
        consecutive_repeats = 0
        max_consecutive = 0
        for i in range(1, len(words)):
            if words[i] == words[i-1]:
                consecutive_repeats += 1
                max_consecutive = max(max_consecutive, consecutive_repeats)
            else:
                consecutive_repeats = 0
        
        if max_consecutive > 5:  # Más de 5 repeticiones consecutivas
            print(f"🚫 Bucle Whisper detectado: '{words[0]}' repetido {max_consecutive+1} veces")
            return True
        
        # ✅ MÉTODO 2: Detectar alta frecuencia de una sola palabra
        from collections import Counter
        word_counts = Counter(words)
        most_common_word, count = word_counts.most_common(1)[0]
        
        # Si una palabra aparece más del 70% del texto, es bucle
        if count > len(words) * 0.7 and len(words) > 15:
            print(f"🚫 Bucle Whisper detectado: '{most_common_word}' aparece {count}/{len(words)} veces")
            return True
        
        return False
    
    def _is_semantic_continuation(self, accumulated_text, new_text):
        """Check if texts are semantically related and should be continued"""
        acc_words = set(accumulated_text.lower().split())
        new_words = set(new_text.lower().split())
        
        # ✅ SEMANTIC FIELD DETECTION
        # Define semantic fields that commonly continue each other
        tech_words = {'ai', 'robot', 'developers', 'software', 'hardware', 'applications', 'build', 'test', 'source', 'open'}
        business_words = {'price', 'priced', 'cost', 'dollar', 'dollars', 'market', 'company', 'partnership'}
        action_words = {'designed', 'help', 'create', 'develop', 'make', 'use', 'work'}
        
        # Check if both texts contain words from the same semantic field
        acc_tech = len(acc_words & tech_words)
        new_tech = len(new_words & tech_words)
        acc_business = len(acc_words & business_words)
        new_business = len(new_words & business_words)
        acc_action = len(acc_words & action_words)
        new_action = len(new_words & action_words)
        
        if (acc_tech > 0 and new_tech > 0) or (acc_business > 0 and new_business > 0) or (acc_action > 0 and new_action > 0):
            print(f"🔗 Semantic field continuity: tech({acc_tech},{new_tech}) business({acc_business},{new_business}) action({acc_action},{new_action})")
            return True
            
        return False
    
    def _has_clear_sentence_ending(self, text):
        """Detect clear sentence endings that should force completion - VERY STRICT"""
        if not text:
            return False
        
        # ✅ ONLY COMPLETE ON VERY OBVIOUS SENTENCE ENDINGS
        
        # 1. Explicit conversation endings (always complete these)
        ending_phrases = [
            'thank you', 'thanks', 'goodbye', 'bye', 'see you', 'talk to you later',
            'that\'s it', 'that\'s all', 'the end', 'that\'s everything',
            'in conclusion', 'to summarize', 'to conclude', 'finally',
            'alright', 'okay', 'good', 'perfect', 'excellent', 'great'
        ]
        
        text_lower = text.lower().strip()
        for phrase in ending_phrases:
            if phrase in text_lower:
                print(f"✅ Clear ending phrase detected: '{phrase}'")
                return True
            
        # 2. Question format (questions are usually complete)
        if text.strip().endswith('?'):
            words = text.split()
            if len(words) >= 3:  # Minimum reasonable question length
                print(f"✅ Complete question detected")
                return True
            
        # 3. Exclamations (usually complete thoughts)
        if text.strip().endswith('!'):
            words = text.split()
            if len(words) >= 3:  # Minimum reasonable exclamation length
                print(f"✅ Complete exclamation detected")
                return True
            
        # 4. VERY SPECIFIC complete sentence patterns with periods
        # Only complete period-ending sentences if they match specific complete patterns
        if text.strip().endswith('.'):
            words = text.split()
            
            # Must be reasonably long to be considered complete
            if len(words) < 6:
                return False
            
            # Pattern: Complete subject-verb-object structures
            complete_sentence_patterns = [
                # Declarative statements that are clearly complete
                r'\b(this|that|it|he|she|they|we|you|i)\s+(is|are|was|were|will be|has been|have been)\b.*\.',
                r'\b(the|a|an)\s+\w+\s+(is|are|was|were|will be|has been|have been)\b.*\.',
                
                # Action statements that are clearly complete
                r'\b(he|she|they|we|you|i)\s+(do|does|did|will|can|could|should|would)\b.*\.',
                r'\b(he|she|they|we|you|i)\s+(have|has|had|will have)\b.*\.',
                
                # Statements with clear objects/completions
                r'\b.*\s+(designed|created|built|made|developed|launched|released)\s+.*\.',
                r'\b.*\s+(costs|priced|worth|valued|available)\s+.*\.',
                r'\b.*\s+(announced|revealed|showed|demonstrated|presented)\s+.*\.',
                
                # Complete informational statements
                r'\b(google|apple|microsoft|amazon|meta|tesla|nvidia|intel|amd)\s+.*\.',
                r'\b.*\s+(company|corporation|organization|startup|business)\s+.*\.',
                r'\b.*\s+(product|service|application|software|hardware|device)\s+.*\.',
                
                # Statements with clear temporal completions
                r'\b.*(today|yesterday|tomorrow|now|currently|recently|finally)\s*\.',
                r'\b.*(launched|released|announced|unveiled|introduced)\s+(today|yesterday|this week|this month|this year)\s*\.'
            ]
            
            text_lower = text.lower()
            for pattern in complete_sentence_patterns:
                if re.search(pattern, text_lower):
                    print(f"✅ Complete sentence pattern detected")
                    return True
            
            # If it doesn't match any complete patterns, it's likely a fragment
            print(f"⏸️ Period detected but doesn't match complete sentence patterns - likely fragment")
            return False
        
        # 5. Quotes (often complete thoughts)
        if text.strip().endswith('"') or text.strip().endswith("'"):
            words = text.split()
            if len(words) >= 4:
                print(f"✅ Complete quoted statement detected")
                return True
        
        return False
    
    def _merge_natural_continuation(self, accumulated_text, new_text):
        """Merge texts that are natural continuations with smart overlap removal and truncated word handling"""
        accumulated_words = accumulated_text.split()
        new_words = new_text.split()
        
        # ✅ HANDLE TRUNCATED WORDS FIRST
        # Check if last word of accumulated is truncated and first word of new is complete
        if accumulated_words and new_words:
            last_acc_word = accumulated_words[-1]
            first_new_word = new_words[0]
            
            # Check for truncated word patterns
            if self._is_truncated_word_pair(last_acc_word, first_new_word):
                # Replace truncated word with complete word
                longer_word = first_new_word if len(first_new_word) > len(last_acc_word) else last_acc_word
                print(f"🔗 Truncated word detected: '{last_acc_word}' -> '{first_new_word}' -> keeping '{longer_word}'")
                
                # Replace last word in accumulated with longer version and skip first word in new
                accumulated_words[-1] = longer_word
                new_words = new_words[1:]  # Skip the first word since we used it
        
        # ✅ FIND AND REMOVE STANDARD OVERLAP
        if new_words:  # Only if there are still words to process
            max_overlap_check = min(4, len(accumulated_words), len(new_words))
            best_overlap = 0
            
            for overlap_size in range(max_overlap_check, 0, -1):
                acc_ending = [w.lower() for w in accumulated_words[-overlap_size:]]
                new_beginning = [w.lower() for w in new_words[:overlap_size]]
                
                if acc_ending == new_beginning:
                    best_overlap = overlap_size
                    break
            
            if best_overlap > 0:
                # Remove overlapping words from new text
                merged_words = accumulated_words + new_words[best_overlap:]
                print(f"🧹 Removed {best_overlap}-word overlap")
            else:
                # No overlap, simple concatenation
                merged_words = accumulated_words + new_words
        else:
            # All new words were handled in truncation, just return accumulated
            merged_words = accumulated_words
        
        return ' '.join(merged_words)
    
    def _is_truncated_word_pair(self, word1, word2):
        """Check if word1 is a truncated version of word2 or vice versa"""
        if not word1 or not word2:
            return False
        
        # ✅ METHOD 1: Check for hyphen indicating truncation
        if word1.endswith('-'):
            truncated = word1[:-1].lower()  # Remove hyphen
            complete = word2.lower()
            if complete.startswith(truncated) and len(complete) > len(truncated):
                return True
        
        if word2.endswith('-'):
            truncated = word2[:-1].lower()  # Remove hyphen
            complete = word1.lower()
            if complete.startswith(truncated) and len(complete) > len(truncated):
                return True
        
        # ✅ METHOD 2: Check for partial word completion (no hyphen)
        word1_clean = word1.lower().strip('.,!?";:')
        word2_clean = word2.lower().strip('.,!?";:')
        
        # If one word is contained at the start of another and is significantly shorter
        if len(word1_clean) >= 3 and len(word2_clean) >= 3:
            if word2_clean.startswith(word1_clean) and len(word2_clean) > len(word1_clean) + 1:
                return True
            if word1_clean.startswith(word2_clean) and len(word1_clean) > len(word2_clean) + 1:
                return True
        
        # ✅ METHOD 3: Similar words with different lengths (likely transcription variants)
        if len(word1_clean) >= 4 and len(word2_clean) >= 4:
            # Calculate simple similarity ratio
            similarity = self._calculate_word_similarity(word1_clean, word2_clean)
            length_diff = abs(len(word1_clean) - len(word2_clean))
            
            # If words are very similar but different lengths, likely one is truncated
            if similarity > 0.7 and length_diff >= 2:
                return True
        
        return False
    
    def _calculate_word_similarity(self, word1, word2):
        """Calculate similarity ratio between two words"""
        if not word1 or not word2:
            return 0.0
        
        # Simple character-based similarity
        shorter = min(len(word1), len(word2))
        matching_chars = 0
        
        for i in range(shorter):
            if word1[i] == word2[i]:
                matching_chars += 1
            else:
                break  # Stop at first mismatch (prefix similarity)
        
        return matching_chars / max(len(word1), len(word2))
    
    def _remove_mahalanobis_duplicates(self, accumulated_text, new_text):
        """Remove duplicate words using Mahalanobis distance - EXACT MATCHES ONLY"""
        if not accumulated_text or not new_text:
            return new_text
        
        acc_words = accumulated_text.lower().split()
        new_words = new_text.lower().split()
        original_new_words = new_text.split()
        
        if not acc_words or not new_words:
            return new_text
        
        cleaned_words = []
        duplicates_removed = 0
        
        for i, (new_word, original_word) in enumerate(zip(new_words, original_new_words)):
            # Check if this exact word appeared recently in accumulated text
            is_duplicate = False
            
            # Check last few words for exact matches
            check_range = min(10, len(acc_words))
            for j in range(check_range):
                acc_word = acc_words[-(j+1)]
                distance = self._calculate_word_mahalanobis_distance(new_word, acc_word)
                
                if distance < 0.01:  # Exact match threshold
                    print(f"🧮 Exact duplicate detected: '{new_word}' ≈ '{acc_word}' (dist: {distance:.3f})")
                    is_duplicate = True
                    duplicates_removed += 1
                    break
            
            if not is_duplicate:
                cleaned_words.append(original_word)
        
        if duplicates_removed > 0:
            print(f"🧹 Removed {duplicates_removed} duplicate words via Mahalanobis")
        
        return ' '.join(cleaned_words) if cleaned_words else new_text
    
    def _calculate_word_mahalanobis_distance(self, word1, word2):
        """Calculate simple Mahalanobis-like distance between two words"""
        # ✅ CHARACTER-BASED FEATURE VECTOR
        # Create simple feature vectors based on character frequency
        def word_to_vector(word):
            # Simple feature: character frequency (a-z)
            vector = np.zeros(26)
            word_clean = ''.join(c.lower() for c in word if c.isalpha())
            
            for char in word_clean:
                if 'a' <= char <= 'z':
                    vector[ord(char) - ord('a')] += 1
            
            # Normalize by word length
            if len(word_clean) > 0:
                vector = vector / len(word_clean)
                
            return vector
        
        v1 = word_to_vector(word1)
        v2 = word_to_vector(word2)
        
        # ✅ SIMPLIFIED MAHALANOBIS DISTANCE
        # Use identity covariance matrix for simplicity (equivalent to Euclidean)
        diff = v1 - v2
        distance = np.sqrt(np.sum(diff ** 2))
        
        return distance
    
    def _are_same_word_forms(self, word1, word2):
        """Check if two words are different forms of the same word"""
        # Simple heuristic: if one word is contained in the other and they share significant characters
        if word1 in word2 or word2 in word1:
            return True
        
        # Check if they have the same root (first 3-4 characters)
        if len(word1) >= 4 and len(word2) >= 4:
            if word1[:3] == word2[:3]:
                return True
        
        return False
    
    def _update_word_history(self, text):
        """Update word history for future Mahalanobis calculations"""
        words = text.lower().split()
        self.word_history.extend(words)
        
        # Keep only recent words
        if len(self.word_history) > self.max_history:
            self.word_history = self.word_history[-self.max_history:]
    
    def _has_very_explicit_ending(self, text):
        """Detect VERY explicit sentence endings - extremely strict"""
        if not text:
            return False
        
        # ✅ ONLY COMPLETE ON EXTREMELY OBVIOUS ENDINGS
        
        # 1. Explicit conversation endings (always complete these)
        explicit_endings = [
            'thank you', 'thanks', 'goodbye', 'bye', 'see you later', 'talk to you later',
            "that's it", "that's all", 'the end', "that's everything",
            'in conclusion', 'to summarize', 'to conclude', 'finally',
            'alright then', 'okay then', 'good job', 'perfect', 'excellent work',
            'thanks for watching', 'thanks for listening'
        ]
        
        text_lower = text.lower().strip()
        for phrase in explicit_endings:
            if phrase in text_lower:
                print(f"✅ Clear ending phrase detected: '{phrase}'")
                return True
        
        # 2. Question endings with clear punctuation
        if text_lower.endswith('?') and len(text.split()) >= 4:
            print(f"✅ Clear question detected")
            return True
        
        # 3. Exclamation with clear context
        if text_lower.endswith('!') and len(text.split()) >= 4:
            # Check if it's a complete exclamation, not a fragment
            exclamation_starters = ['wow', 'amazing', 'incredible', 'fantastic', 'great', 'excellent', 'perfect']
            if any(text_lower.startswith(starter) for starter in exclamation_starters):
                print(f"✅ Clear exclamation detected")
                return True
        
        # 4. NEVER complete on simple periods - they are almost always fragments
        return False
    
    def _appears_to_be_fragment_aggressive(self, text):
        """MENOS AGRESIVO: Detectar fragmentos pero ser más permisivo"""
        if not text:
            return False
        
        words = text.lower().split()
        if not words:
            return False
        
        # ✅ MENOS AGRESIVO: Solo fragmentos muy cortos son obviamente incompletos
        # 1. Textos muy cortos son fragmentos
        if len(words) < 5:  # Reducido de 8 a 5
            print(f"🔗 Fragment detected: too short ({len(words)} words)")
            return True
        
        # 2. Solo fragmentos MUY obvios
        text_clean = text.lower().strip('.,!?;: ')
        
        # Patrones de fragmentos MÁS ESPECÍFICOS - solo los más obvios
        obvious_incomplete_patterns = [
            # Solo los finales más obviamente incompletos
            r'\b(the|a|an|and|or|but|so|for|to|of|in|on|at|with|by|from)$',
            r'\b(is|are|was|were|has|have|had|will|would|could|should|can|may|might)$',
            
            # Conectores que claramente indican continuación
            r'\b(because|since|although|while|whereas|unless|until|before|after)$',
            r'\b(which|that|who|where|when|how|why|what)$',
        ]
        
        for pattern in obvious_incomplete_patterns:
            if re.search(pattern, text_clean):
                print(f"🔗 Fragment detected: ends with incomplete word")
                return True
        
        return False
    
    def correct_and_validate_text(self, text):
        """🧮 SIMPLIFIED: Basic text validation with improved fragment detection"""
        if not text or not text.strip():
            return False, text, "Empty text"
        
        text = text.strip()
        
        # ✅ 1. BASIC LENGTH CHECK
        words = text.split()
        if len(words) < self.min_text_length:
            return False, text, "Too short"
        
        # ✅ 2. SIMPLE WORD DUPLICATE REMOVAL
        corrected_text = self._remove_immediate_duplicates(text)
        
        # ✅ 3. CHECK FOR COMPLETION USING NEW LOGIC
        # Only complete if it has clear ending AND is not a fragment
        has_clear_ending = self._has_clear_sentence_ending(corrected_text)
        appears_to_be_fragment = self._appears_to_be_fragment_aggressive(corrected_text)
        
        is_complete = has_clear_ending and not appears_to_be_fragment
        
        # ✅ 4. LENGTH-BASED FALLBACK (más permisivo)
        if not is_complete and len(words) >= 8:  # Reducido de 12 a 8
            # Solo usar longitud si no es obviamente un fragmento
            if not appears_to_be_fragment:
                print(f"✅ Completando por longitud ({len(words)} palabras)")
                is_complete = True
        
        validation_method = "Mahalanobis validation"
        
        return is_complete, corrected_text, validation_method
    
    def _remove_immediate_duplicates(self, text):
        """Remove immediate word duplicates (word word -> word)"""
        words = text.split()
        cleaned_words = []
        
        for i, word in enumerate(words):
            # Only remove if the exact same word appears immediately before
            if i > 0 and word.lower() == words[i-1].lower():
                print(f"🧹 Removing immediate duplicate: '{word}'")
                continue
            cleaned_words.append(word)
        
        return ' '.join(cleaned_words)
    
    # ✅ COMPATIBILITY METHODS
    def validate_text_coherence(self, text):
        """Compatibility method"""
        is_valid, corrected_text, reason = self.correct_and_validate_text(text)
        return is_valid, reason
    
    def is_sentence_complete(self, text):
        """Compatibility method"""
        is_valid, corrected_text, reason = self.correct_and_validate_text(text)
        return is_valid
    
    def extract_complete_sentences(self, text):
        """Extract sentences based on punctuation detection"""
        if not text or not text.strip():
            return [], ""
        
        sentences = []
        remaining = text
        
        # ✅ SPLIT BY SENTENCE PUNCTUATION
        parts = re.split(r'([.!?。！？])\s*', text)
        
        current_sentence = ""
        i = 0
        while i < len(parts):
            part = parts[i]
            
            if re.match(r'[.!?。！？]', part):
                current_sentence += part
                if current_sentence.strip():
                    is_valid, corrected_sentence, reason = self.correct_and_validate_text(current_sentence.strip())
                    if is_valid and corrected_sentence:
                        sentences.append(corrected_sentence)
                        print(f"✅ Extracted Sentence: '{corrected_sentence}' ({reason})")
                current_sentence = ""
            else:
                current_sentence += part
            
            i += 1
        
        remaining = current_sentence.strip()
        return sentences, remaining 