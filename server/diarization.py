"""
Speaker diarization module using local audio feature clustering.
No external services or authentication required - fully local GPU/CPU processing.
"""
import os
import warnings
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


def perform_diarization(
    audio_path: str,
    num_speakers: int = 2,
    min_speakers: int = 1,
    max_speakers: int = 3
) -> List[Dict]:
    """
    Perform local speaker diarization using audio feature clustering.

    This implementation uses:
    - Voice Activity Detection (VAD) to find speech segments
    - MFCC feature extraction for speaker characteristics
    - K-means clustering to group similar voices
    - Completely local processing (no external APIs)

    Returns:
        List of segments: [{"start": float, "end": float, "speaker": str}, ...]
    """
    try:
        import librosa
        from sklearn.cluster import KMeans
        from scipy import signal

        print(f"[DIARIZATION] Starting local speaker diarization for {audio_path}")
        print(f"[DIARIZATION] Target speakers: {num_speakers}, Range: {min_speakers}-{max_speakers}")

        # Load audio file
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        duration = len(y) / sr
        print(f"[DIARIZATION] Loaded audio: {duration:.2f}s, sample_rate={sr}")

        # Simple Voice Activity Detection using energy
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.010 * sr)    # 10ms hop

        # Calculate energy for VAD
        energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        energy_threshold = np.mean(energy) * 0.3  # Adaptive threshold

        # Extract MFCC features for clustering
        n_mfcc = 13
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                    hop_length=hop_length, n_fft=frame_length)

        # Add delta features (velocity of change)
        mfcc_delta = librosa.feature.delta(mfcc)

        # Combine MFCC and delta features
        features = np.vstack([mfcc, mfcc_delta])
        features = features.T  # Shape: (n_frames, n_features)

        print(f"[DIARIZATION] Extracted {features.shape[0]} frames with {features.shape[1]} features each")

        # Filter frames with speech activity
        speech_mask = energy > energy_threshold
        speech_features = features[speech_mask]
        speech_frame_indices = np.where(speech_mask)[0]

        if len(speech_features) < num_speakers * 10:
            print(f"[DIARIZATION] Not enough speech detected ({len(speech_features)} frames), using single speaker")
            return [{"start": 0.0, "end": duration, "speaker": "SPEAKER_00"}]

        print(f"[DIARIZATION] Found {len(speech_features)} speech frames out of {len(features)} total")

        # Determine optimal number of speakers
        # Use silhouette score to find best k if num_speakers not specified
        best_k = num_speakers if num_speakers > 1 else min(2, max_speakers)

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        labels = np.full(len(features), -1, dtype=int)  # -1 for non-speech
        labels[speech_frame_indices] = kmeans.fit_predict(speech_features)

        print(f"[DIARIZATION] Clustered into {best_k} speakers")

        # Convert frame indices to time segments
        frame_times = librosa.frames_to_time(np.arange(len(labels)), sr=sr, hop_length=hop_length)

        # Group consecutive frames with same speaker
        segments = []
        current_speaker = None
        segment_start = 0.0

        for i, (time, label) in enumerate(zip(frame_times, labels)):
            if label == -1:  # Non-speech
                if current_speaker is not None:
                    # End current segment
                    segments.append({
                        "start": round(segment_start, 2),
                        "end": round(time, 2),
                        "speaker": f"SPEAKER_{current_speaker:02d}"
                    })
                    current_speaker = None
            else:  # Speech detected
                if current_speaker is None:
                    # Start new segment
                    current_speaker = label
                    segment_start = time
                elif current_speaker != label:
                    # Speaker change
                    segments.append({
                        "start": round(segment_start, 2),
                        "end": round(time, 2),
                        "speaker": f"SPEAKER_{current_speaker:02d}"
                    })
                    current_speaker = label
                    segment_start = time

        # Add final segment
        if current_speaker is not None:
            segments.append({
                "start": round(segment_start, 2),
                "end": round(duration, 2),
                "speaker": f"SPEAKER_{current_speaker:02d}"
            })

        # Merge very short segments (< 0.5s) with neighbors
        merged_segments = []
        for seg in segments:
            if seg["end"] - seg["start"] < 0.5 and merged_segments:
                # Extend previous segment
                merged_segments[-1]["end"] = seg["end"]
            else:
                merged_segments.append(seg)

        print(f"[DIARIZATION] Created {len(merged_segments)} speaker segments")

        # Print segment distribution
        speaker_times = {}
        for seg in merged_segments:
            speaker = seg["speaker"]
            duration_seg = seg["end"] - seg["start"]
            speaker_times[speaker] = speaker_times.get(speaker, 0) + duration_seg

        for speaker, time in sorted(speaker_times.items()):
            print(f"[DIARIZATION] {speaker}: {time:.1f}s ({time/duration*100:.1f}%)")

        return merged_segments

    except Exception as e:
        print(f"[DIARIZATION] Error during local diarization: {e}")
        print("[DIARIZATION] Falling back to single speaker")

        # Fallback: return single segment covering entire audio
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
