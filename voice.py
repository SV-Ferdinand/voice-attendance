import torch
import torchaudio
import numpy as np
import librosa
import soundfile as sf
from resemblyzer import VoiceEncoder, preprocess_wav
from speechbrain.pretrained import SpeakerRecognition
import os
import io
import tempfile
from typing import List, Tuple, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(_name_)

class VoiceRecognitionService:
    """
    Advanced voice recognition service using Resemblyzer and SpeechBrain
    for voice biometric authentication in attendance system
    """
    
    def _init_(self):
        # Initialize Resemblyzer voice encoder
        self.resemblyzer_encoder = VoiceEncoder()
        
        # Initialize SpeechBrain speaker recognition model
        try:
            self.speechbrain_model = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb"
            )
            logger.info("SpeechBrain model loaded successfully")
        except Exception as e:
            logger.warning(f"SpeechBrain model loading failed: {e}")
            self.speechbrain_model = None
        
        # Audio processing parameters
        self.sample_rate = 16000
        self.min_duration = 1.0  # Minimum audio duration in seconds
        self.max_duration = 10.0  # Maximum audio duration in seconds
        
        # Similarity thresholds
        self.resemblyzer_threshold = 0.75
        self.speechbrain_threshold = 0.80
        self.combined_threshold = 0.77
        
    def preprocess_audio(self, audio_data: bytes, source_format: str = 'wav') -> Optional[np.ndarray]:
        """
        Preprocess audio data for voice recognition
        
        Args:
            audio_data: Raw audio bytes
            source_format: Source audio format (wav, mp3, etc.)
            
        Returns:
            Preprocessed audio array or None if processing fails
        """
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=f'.{source_format}', delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_path = temp_file.name
            
            try:
                # Load audio with librosa
                audio, sr = librosa.load(temp_path, sr=self.sample_rate)
                
                # Audio quality checks
                duration = len(audio) / sr
                if duration < self.min_duration:
                    logger.warning(f"Audio too short: {duration:.2f}s (minimum: {self.min_duration}s)")
                    return None
                
                if duration > self.max_duration:
                    logger.warning(f"Audio too long: {duration:.2f}s (maximum: {self.max_duration}s)")
                    # Trim to max duration
                    audio = audio[:int(self.max_duration * sr)]
                
                # Normalize audio
                audio = librosa.util.normalize(audio)
                
                # Remove silence
                audio, _ = librosa.effects.trim(audio, top_db=20)
                
                # Check if audio is not empty after trimming
                if len(audio) < sr * 0.5:  # Less than 0.5 seconds
                    logger.warning("Audio too short after silence removal")
                    return None
                
                return audio
                
            finally:
                # Clean up temporary file
                os.unlink(temp_path)
                
        except Exception as e:
            logger.error(f"Error preprocessing audio: {e}")
            return None
    
    def extract_resemblyzer_embedding(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract voice embedding using Resemblyzer
        
        Args:
            audio: Preprocessed audio array
            
        Returns:
            Voice embedding or None if extraction fails
        """
        try:
            # Preprocess for Resemblyzer
            wav = preprocess_wav(audio, source_sr=self.sample_rate)
            
            # Extract embedding
            embedding = self.resemblyzer_encoder.embed_utterance(wav)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error extracting Resemblyzer embedding: {e}")
            return None
    
    def extract_speechbrain_embedding(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract voice embedding using SpeechBrain
        
        Args:
            audio: Preprocessed audio array
            
        Returns:
            Voice embedding or None if extraction fails
        """
        if self.speechbrain_model is None:
            return None
            
        try:
            # Convert to tensor
            audio_tensor = torch.FloatTensor(audio).unsqueeze(0)
            
            # Extract embedding
            with torch.no_grad():
                embeddings = self.speechbrain_model.encode_batch(audio_tensor)
                embedding = embeddings.squeeze().cpu().numpy()
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error extracting SpeechBrain embedding: {e}")
            return None
    
    def extract_combined_embedding(self, audio_data: bytes) -> Optional[dict]:
        """
        Extract voice embeddings using both Resemblyzer and SpeechBrain
        
        Args:
            audio_data: Raw audio bytes
            
        Returns:
            Dictionary containing both embeddings or None if extraction fails
        """
        # Preprocess audio
        audio = self.preprocess_audio(audio_data)
        if audio is None:
            return None
        
        embeddings = {}
        
        # Extract Resemblyzer embedding
        resemblyzer_emb = self.extract_resemblyzer_embedding(audio)
        if resemblyzer_emb is not None:
            embeddings['resemblyzer'] = resemblyzer_emb.tolist()
        
        # Extract SpeechBrain embedding
        speechbrain_emb = self.extract_speechbrain_embedding(audio)
        if speechbrain_emb is not None:
            embeddings['speechbrain'] = speechbrain_emb.tolist()
        
        # Ensure at least one embedding was extracted
        if not embeddings:
            logger.error("Failed to extract any voice embeddings")
            return None
        
        return embeddings
    
    def calculate_cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            emb1: First embedding
            emb2: Second embedding
            
        Returns:
            Cosine similarity score (0-1)
        """
        try:
            # Normalize vectors
            emb1_norm = emb1 / np.linalg.norm(emb1)
            emb2_norm = emb2 / np.linalg.norm(emb2)
            
            # Calculate cosine similarity
            similarity = np.dot(emb1_norm, emb2_norm)
            
            # Ensure similarity is between 0 and 1
            similarity = max(0.0, min(1.0, (similarity + 1) / 2))
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def verify_voice(self, input_audio: bytes, stored_embeddings: List[dict]) -> dict:
        """
        Verify voice against stored embeddings
        
        Args:
            input_audio: Raw audio bytes from user
            stored_embeddings: List of stored embedding dictionaries
            
        Returns:
            Verification result with confidence scores
        """
        # Extract embedding from input audio
        input_embeddings = self.extract_combined_embedding(input_audio)
        if input_embeddings is None:
            return {
                'success': False,
                'confidence': 0.0,
                'error': 'Failed to process input audio'
            }
        
        best_resemblyzer_score = 0.0
        best_speechbrain_score = 0.0
        resemblyzer_scores = []
        speechbrain_scores = []
        
        # Compare with each stored embedding
        for stored_emb in stored_embeddings:
            # Compare Resemblyzer embeddings
            if 'resemblyzer' in input_embeddings and 'resemblyzer' in stored_emb:
                input_resem = np.array(input_embeddings['resemblyzer'])
                stored_resem = np.array(stored_emb['resemblyzer'])
                
                similarity = self.calculate_cosine_similarity(input_resem, stored_resem)
                resemblyzer_scores.append(similarity)
                best_resemblyzer_score = max(best_resemblyzer_score, similarity)
            
            # Compare SpeechBrain embeddings
            if 'speechbrain' in input_embeddings and 'speechbrain' in stored_emb:
                input_sb = np.array(input_embeddings['speechbrain'])
                stored_sb = np.array(stored_emb['speechbrain'])
                
                similarity = self.calculate_cosine_similarity(input_sb, stored_sb)
                speechbrain_scores.append(similarity)
                best_speechbrain_score = max(best_speechbrain_score, similarity)
        
        # Calculate combined confidence score
        scores = []
        if resemblyzer_scores:
            avg_resemblyzer = np.mean(resemblyzer_scores)
            scores.append(avg_resemblyzer)
        
        if speechbrain_scores:
            avg_speechbrain = np.mean(speechbrain_scores)
            scores.append(avg_speechbrain)
        
        if not scores:
            return {
                'success': False,
                'confidence': 0.0,
                'error': 'No compatible embeddings found'
            }
        
        # Combined confidence score (weighted average)
        if len(scores) == 2:
            # Both models available - weighted combination
            combined_confidence = (scores[0] * 0.6 + scores[1] * 0.4)  # Favor Resemblyzer slightly
        else:
            # Only one model available
            combined_confidence = scores[0]
        
        # Determine if verification passes
        success = combined_confidence >= self.combined_threshold
        
        return {
            'success': success,
            'confidence': float(combined_confidence),
            'resemblyzer_score': float(best_resemblyzer_score) if resemblyzer_scores else None,
            'speechbrain_score': float(best_speechbrain_score) if speechbrain_scores else None,
            'threshold_used': self.combined_threshold,
            'num_comparisons': len(stored_embeddings)
        }
    
    def validate_speech_content(self, audio_data: bytes, expected_word: str = "HADIR") -> dict:
        """
        Validate that the spoken content matches expected word
        Using simple energy-based validation (can be enhanced with STT)
        
        Args:
            audio_data: Raw audio bytes
            expected_word: Expected spoken word
            
        Returns:
            Validation result
        """
        try:
            audio = self.preprocess_audio(audio_data)
            if audio is None:
                return {'valid': False, 'confidence': 0.0, 'error': str(e)}

    def batch_process_recordings(self, audio_files: List[bytes]) -> List[dict]:
        """
        Process multiple audio recordings in batch
        
        Args:
            audio_files: List of raw audio bytes
            
        Returns:
            List of embedding dictionaries
        """
        embeddings = []
        
        for i, audio_data in enumerate(audio_files):
            logger.info(f"Processing recording {i+1}/{len(audio_files)}")
            
            embedding = self.extract_combined_embedding(audio_data)
            if embedding is not None:
                embeddings.append(embedding)
            else:
                logger.warning(f"Failed to process recording {i+1}")
        
        return embeddings

    def get_voice_quality_metrics(self, audio_data: bytes) -> dict:
        """
        Analyze voice quality metrics
        
        Args:
            audio_data: Raw audio bytes
            
        Returns:
            Dictionary containing quality metrics
        """
        try:
            audio = self.preprocess_audio(audio_data)
            if audio is None:
                return {'error': 'Failed to preprocess audio'}
            
            # Calculate various quality metrics
            duration = len(audio) / self.sample_rate
            rms_energy = np.sqrt(np.mean(audio ** 2))
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio))
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)
            
            # Signal-to-noise ratio estimation
            # Simple method: compare energy in speech vs. non-speech segments
            audio_abs = np.abs(audio)
            threshold = np.percentile(audio_abs, 70)
            speech_segments = audio_abs > threshold
            
            if np.any(speech_segments) and np.any(~speech_segments):
                speech_energy = np.mean(audio_abs[speech_segments] ** 2)
                noise_energy = np.mean(audio_abs[~speech_segments] ** 2)
                snr = 10 * np.log10(speech_energy / (noise_energy + 1e-10))
            else:
                snr = 0.0
            
            # Overall quality score
            quality_score = 0.0
            
            # Duration check
            if 1.0 <= duration <= 5.0:
                quality_score += 0.25
            
            # Energy check
            if 0.01 <= rms_energy <= 0.5:
                quality_score += 0.25
            
            # SNR check
            if snr > 10:
                quality_score += 0.25
            
            # Spectral check
            if np.mean(spectral_centroids) > 1000:  # Has high-frequency content
                quality_score += 0.25
            
            return {
                'duration': float(duration),
                'rms_energy': float(rms_energy),
                'zero_crossing_rate': float(np.mean(zero_crossing_rate)),
                'spectral_centroid_mean': float(np.mean(spectral_centroids)),
                'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
                'estimated_snr': float(snr),
                'quality_score': float(quality_score),
                'quality_rating': self._get_quality_rating(quality_score)
            }
            
        except Exception as e:
            logger.error(f"Error calculating quality metrics: {e}")
            return {'error': str(e)}
    
    def _get_quality_rating(self, score: float) -> str:
        """Convert quality score to rating"""
        if score >= 0.8:
            return 'Excellent'
        elif score >= 0.6:
            return 'Good'
        elif score >= 0.4:
            return 'Fair'
        else:
            return 'Poor'


# Enhanced Voice Recognition API Integration
class VoiceRecognitionAPI:
    """
    API wrapper for voice recognition service
    """
    
    def _init_(self):
        self.voice_service = VoiceRecognitionService()
    
    def register_voice_sample(self, audio_data: bytes, student_id: str, recording_number: int) -> dict:
        """
        Register a voice sample for a student
        
        Args:
            audio_data: Raw audio bytes
            student_id: Student identifier
            recording_number: Recording number (1, 2, or 3)
            
        Returns:
            Registration result
        """
        try:
            # Validate speech content
            speech_validation = self.voice_service.validate_speech_content(audio_data)
            if not speech_validation['valid']:
                return {
                    'success': False,
                    'error': f"Speech validation failed: {speech_validation.get('error', 'Unknown error')}",
                    'validation_details': speech_validation
                }
            
            # Get quality metrics
            quality_metrics = self.voice_service.get_voice_quality_metrics(audio_data)
            if 'error' in quality_metrics:
                return {
                    'success': False,
                    'error': f"Quality analysis failed: {quality_metrics['error']}"
                }
            
            # Check if quality is sufficient
            if quality_metrics.get('quality_score', 0) < 0.3:
                return {
                    'success': False,
                    'error': f"Audio quality too low: {quality_metrics.get('quality_rating', 'Unknown')}",
                    'quality_metrics': quality_metrics
                }
            
            # Extract voice embeddings
            embeddings = self.voice_service.extract_combined_embedding(audio_data)
            if embeddings is None:
                return {
                    'success': False,
                    'error': 'Failed to extract voice embeddings'
                }
            
            return {
                'success': True,
                'embeddings': embeddings,
                'quality_metrics': quality_metrics,
                'speech_validation': speech_validation,
                'recording_number': recording_number,
                'student_id': student_id
            }
            
        except Exception as e:
            logger.error(f"Error in register_voice_sample: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def verify_voice_for_attendance(self, audio_data: bytes, stored_embeddings: List[dict]) -> dict:
        """
        Verify voice for attendance marking
        
        Args:
            audio_data: Raw audio bytes
            stored_embeddings: List of stored voice embeddings
            
        Returns:
            Verification result
        """
        try:
            # Validate speech content first
            speech_validation = self.voice_service.validate_speech_content(audio_data)
            if not speech_validation['valid']:
                return {
                    'success': False,
                    'confidence': 0.0,
                    'error': f"Speech validation failed: {speech_validation.get('error', 'Unknown error')}",
                    'validation_details': speech_validation
                }
            
            # Perform voice verification
            verification_result = self.voice_service.verify_voice(audio_data, stored_embeddings)
            
            # Add speech validation info to result
            verification_result['speech_validation'] = speech_validation
            
            # Get quality metrics for logging
            quality_metrics = self.voice_service.get_voice_quality_metrics(audio_data)
            verification_result['quality_metrics'] = quality_metrics
            
            return verification_result
            
        except Exception as e:
            logger.error(f"Error in verify_voice_for_attendance: {e}")
            return {
                'success': False,
                'confidence': 0.0,
                'error': str(e)
            }


# Usage example and testing functions
def test_voice_recognition_service():
    """
    Test function for voice recognition service
    """
    # Initialize service
    voice_api = VoiceRecognitionAPI()
    
    # Test with sample audio file (you would replace this with actual audio)
    try:
        # Load a test audio file
        test_audio_path = "test_audio.wav"
        if os.path.exists(test_audio_path):
            with open(test_audio_path, 'rb') as f:
                audio_data = f.read()
            
            print("Testing voice registration...")
            registration_result = voice_api.register_voice_sample(
                audio_data=audio_data,
                student_id="123456789",
                recording_number=1
            )
            
            print("Registration result:", registration_result)
            
            if registration_result['success']:
                print("\nTesting voice verification...")
                verification_result = voice_api.verify_voice_for_attendance(
                    audio_data=audio_data,
                    stored_embeddings=[registration_result['embeddings']]
                )
                
                print("Verification result:", verification_result)
        else:
            print(f"Test audio file {test_audio_path} not found")
            
    except Exception as e:
        print(f"Test failed: {e}")


if _name_ == "_main_":
    # Run tests if this file is executed directly
    test_voice_recognition_service()0.0, 'error': 'Audio preprocessing failed'}
            
            # Simple validation based on audio characteristics
            # In production, integrate with Speech-to-Text for actual word recognition
            
            # Check audio energy levels
            energy = np.mean(audio ** 2)
            
            # Check if there's sufficient speech activity
            if energy < 0.001:  # Too quiet
                return {'valid': False, 'confidence': 0.0, 'error': 'Audio too quiet'}
            
            # Check duration (should be appropriate for saying "HADIR")
            duration = len(audio) / self.sample_rate
            if duration < 0.5 or duration > 3.0:
                return {'valid': False, 'confidence': 0.5, 'error': f'Duration {duration:.1f}s not suitable for "{expected_word}"'}
            
            # Basic spectral analysis to detect speech
            mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
            mfcc_variance = np.var(mfccs, axis=1).mean()
            
            if mfcc_variance < 0.1:  # Too little variation, possibly noise
                return {'valid': False, 'confidence': 0.3, 'error': 'Audio lacks speech characteristics'}
            
            # If all checks pass, assume valid (enhance with actual STT in production)
            confidence = min(0.9, energy * 100 + mfcc_variance)
            
            return {
                'valid': True,
                'confidence': float(confidence),
                'duration': duration,
                'energy_level': float(energy),
                'note': f'Basic validation passed for "{expected_word}" - integrate STT for better accuracy'
            }
            
        except Exception as e:
            logger.error(f"Error validating speech content: {e}")
            return {'valid': False, 'confidence':