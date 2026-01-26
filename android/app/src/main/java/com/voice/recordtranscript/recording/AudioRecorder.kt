package com.voice.recordtranscript.recording

import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

class AudioRecorder(private val outputFile: File) {
    private var audioRecord: AudioRecord? = null
    private var isRecording = false
    private var recordingThread: Thread? = null
    
    companion object {
        private const val SAMPLE_RATE = 16000
        private const val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
        private const val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT
        private const val TAG = "AudioRecorder"
    }
    
    fun start() {
        if (isRecording) return
        
        val bufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT)
        if (bufferSize == AudioRecord.ERROR_BAD_VALUE || bufferSize == AudioRecord.ERROR) {
            Log.e(TAG, "Invalid buffer size")
            return
        }
        
        try {
            audioRecord = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                SAMPLE_RATE,
                CHANNEL_CONFIG,
                AUDIO_FORMAT,
                bufferSize * 2
            )
            
            if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
                Log.e(TAG, "AudioRecord initialization failed")
                return
            }
            
            audioRecord?.startRecording()
            isRecording = true
            
            recordingThread = Thread {
                writeAudioDataToFile()
            }
            recordingThread?.start()
            
        } catch (e: Exception) {
            Log.e(TAG, "Error starting recording", e)
        }
    }
    
    fun stop() {
        if (!isRecording) return
        
        isRecording = false
        audioRecord?.stop()
        audioRecord?.release()
        audioRecord = null
        
        recordingThread?.join()
        recordingThread = null
    }
    
    private fun writeAudioDataToFile() {
        val buffer = ShortArray(4096)
        var fos: FileOutputStream? = null
        
        try {
            fos = FileOutputStream(outputFile)
            writeWavHeader(fos, 0) // Placeholder, will rewrite at end
            
            while (isRecording && audioRecord != null) {
                val read = audioRecord?.read(buffer, 0, buffer.size) ?: 0
                if (read > 0) {
                    // Convert shorts to bytes (little-endian)
                    for (i in 0 until read) {
                        val sample = buffer[i]
                        fos.write(sample.toInt() and 0xFF)
                        fos.write((sample.toInt() shr 8) and 0xFF)
                    }
                }
            }
            
            // Rewrite WAV header with actual size
            val fileSize = outputFile.length().toInt()
            fos.close()
            rewriteWavHeader(outputFile, fileSize)
            
        } catch (e: IOException) {
            Log.e(TAG, "Error writing audio data", e)
        } finally {
            fos?.close()
        }
    }
    
    private fun writeWavHeader(fos: FileOutputStream, dataSize: Int) {
        val header = ByteArray(44)
        var offset = 0
        
        // RIFF header
        "RIFF".toByteArray().copyInto(header, offset)
        offset += 4
        writeInt(header, offset, 36 + dataSize)
        offset += 4
        "WAVE".toByteArray().copyInto(header, offset)
        offset += 4
        
        // fmt chunk
        "fmt ".toByteArray().copyInto(header, offset)
        offset += 4
        writeInt(header, offset, 16) // fmt chunk size
        offset += 4
        writeShort(header, offset, 1) // audio format (PCM)
        offset += 2
        writeShort(header, offset, 1) // num channels (mono)
        offset += 2
        writeInt(header, offset, SAMPLE_RATE) // sample rate
        offset += 4
        writeInt(header, offset, SAMPLE_RATE * 2) // byte rate
        offset += 4
        writeShort(header, offset, 2) // block align
        offset += 2
        writeShort(header, offset, 16) // bits per sample
        offset += 2
        
        // data chunk
        "data".toByteArray().copyInto(header, offset)
        offset += 4
        writeInt(header, offset, dataSize)
        
        fos.write(header)
    }
    
    private fun rewriteWavHeader(file: File, dataSize: Int) {
        val raf = java.io.RandomAccessFile(file, "rw")
        raf.seek(4)
        raf.writeInt(36 + dataSize)
        raf.seek(40)
        raf.writeInt(dataSize)
        raf.close()
    }
    
    private fun writeInt(buffer: ByteArray, offset: Int, value: Int) {
        buffer[offset] = (value and 0xFF).toByte()
        buffer[offset + 1] = ((value shr 8) and 0xFF).toByte()
        buffer[offset + 2] = ((value shr 16) and 0xFF).toByte()
        buffer[offset + 3] = ((value shr 24) and 0xFF).toByte()
    }
    
    private fun writeShort(buffer: ByteArray, offset: Int, value: Int) {
        buffer[offset] = (value and 0xFF).toByte()
        buffer[offset + 1] = ((value shr 8) and 0xFF).toByte()
    }
}

