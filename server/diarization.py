"""
Speaker diarization module.
Uses pyannote.audio for speaker detection and segmentation.
"""
import os
import warnings
from typing import List, Dict, Tuple
from pathlib import Path

# Fix NumPy 2.0 compatibility issue with pyannote
import numpy as np
if not hasattr(np, 'NaN'):
    np.NaN = np.nan
if not hasattr(np, 'Inf'):
    np.Inf = np.inf

# Suppress deprecation warnings from pyannote
warnings.filterwarnings('ignore', category=DeprecationWarning)

def perform_diarization(
    audio_path: str,
    num_speakers: int = 2,
    min_speakers: int = 1,
    max_speakers: int = 3
) -> List[Dict]:
    """
    Perform speaker diarization on audio file.
    
    Returns:
        List of segments: [{"start": float, "end": float, "speaker": str}, ...]
    """
    try:
        from pyannote.audio import Pipeline
        
        # Load diarization pipeline
        # Use a lightweight model that doesn't require authentication
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=os.getenv("HUGGINGFACE_TOKEN")  # Optional
        )
        
        # Run diarization
        diarization = pipeline(
            audio_path,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            num_speakers=num_speakers if num_speakers > 1 else None
        )
        
        # Convert to list of dicts
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": round(turn.start, 2),
                "end": round(turn.end, 2),
                "speaker": speaker
            })
        
        return segments
    
    except Exception as e:
        print(f"Diarization error: {e}")
        print("Falling back to single speaker")
        # Fallback: return single segment covering entire audio
        # We'll estimate duration from file if possible
        try:
            import subprocess
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", 
                 "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
                capture_output=True, text=True, timeout=5
            )
            duration = float(result.stdout.strip()) if result.stdout.strip() else 0.0
        except:
            duration = 0.0
        
        if duration > 0:
            return [{"start": 0.0, "end": duration, "speaker": "SPEAKER_00"}]
        else:
            return []


def align_diarization_with_transcript(
    diarization_segments: List[Dict],
    transcript_segments: List[Dict]
) -> List[Dict]:
    """
    Align diarization segments with transcript segments using time overlap.
    
    Returns:
        List of merged segments: [{"start": float, "end": float, "speaker": str, "text": str}, ...]
    """
    if not diarization_segments:
        # No diarization, return transcript with default speaker
        return [
            {**seg, "speaker": "SPEAKER_00"}
            for seg in transcript_segments
        ]
    
    if not transcript_segments:
        return []
    
    aligned = []
    
    # For each transcript segment, find overlapping diarization segment
    for trans_seg in transcript_segments:
        trans_start = trans_seg["start"]
        trans_end = trans_seg["end"]
        trans_text = trans_seg.get("text", "")
        
        # Find diarization segment that overlaps most with this transcript segment
        best_speaker = "SPEAKER_00"
        max_overlap = 0
        
        for diar_seg in diarization_segments:
            diar_start = diar_seg["start"]
            diar_end = diar_seg["end"]
            
            # Calculate overlap
            overlap_start = max(trans_start, diar_start)
            overlap_end = min(trans_end, diar_end)
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > max_overlap:
                max_overlap = overlap
                best_speaker = diar_seg["speaker"]
        
        aligned.append({
            "start": trans_start,
            "end": trans_end,
            "speaker": best_speaker,
            "text": trans_text
        })
    
    return aligned


def group_by_speaker(aligned_segments: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Group aligned segments by speaker.
    
    Returns:
        {"SPEAKER_00": [segments...], "SPEAKER_01": [segments...], ...}
    """
    grouped = {}
    for seg in aligned_segments:
        speaker = seg.get("speaker", "SPEAKER_00")
        if speaker not in grouped:
            grouped[speaker] = []
        grouped[speaker].append(seg)
    
    return grouped
