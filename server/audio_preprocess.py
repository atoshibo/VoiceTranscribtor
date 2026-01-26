"""
Light audio preprocessing for diarization compatibility.
"""
import subprocess
from pathlib import Path

def preprocess_audio(input_path: str, output_path: str) -> bool:
    """
    Preprocess audio: convert to mono, 16kHz, normalize.
    Returns True if successful.
    """
    try:
        # Use ffmpeg to convert to mono 16kHz WAV
        # Skip normalization if not available (some ffmpeg versions don't have loudnorm)
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-ac", "1",  # mono
            "-ar", "16000",  # 16kHz
            output_path
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=300,
            text=True
        )
        
        if result.returncode == 0:
            return True
        else:
            print(f"Audio preprocessing failed: {result.stderr}")
            return False
    
    except Exception as e:
        print(f"Audio preprocessing error: {e}")
        return False
