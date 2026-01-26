package com.voice.recordtranscript.upload

import android.content.Context
import android.util.Log
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import com.google.gson.Gson
import com.voice.recordtranscript.data.Session
import com.voice.recordtranscript.data.SessionStatus
import com.voice.recordtranscript.storage.SessionStorage
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.asRequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.File
import java.io.IOException
import java.util.concurrent.TimeUnit

class UploadWorker(context: Context, params: WorkerParameters) : CoroutineWorker(context, params) {
    
    private val storage = SessionStorage(context)
    private val client = OkHttpClient.Builder()
        .connectTimeout(5, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .build()
    
    companion object {
        private const val TAG = "UploadWorker"
    }
    
    override suspend fun doWork(): Result = withContext(Dispatchers.IO) {
        val prefs = applicationContext.getSharedPreferences("settings", Context.MODE_PRIVATE)
        val serverUrl = prefs.getString("server_url", "") ?: ""
        
        if (serverUrl.isEmpty()) {
            Log.d(TAG, "No server URL configured")
            return@withContext Result.retry()
        }
        
        // Check server health
        if (!checkServerHealth(serverUrl)) {
            Log.d(TAG, "Server not reachable")
            return@withContext Result.retry()
        }
        
        val pendingSessions = storage.getPendingSessions()
        if (pendingSessions.isEmpty()) {
            Log.d(TAG, "No pending sessions")
            return@withContext Result.success()
        }
        
        var successCount = 0
        var failureCount = 0
        
        for (session in pendingSessions) {
            try {
                // Handle stuck UPLOADING status: check if server has it
                if (session.status == SessionStatus.UPLOADING) {
                    if (checkSessionExistsOnServer(serverUrl, session.sessionId)) {
                        // Server has it, continue from transcription step
                        val sessionId = session.sessionId
                        val results = waitForResults(serverUrl, sessionId, maxWait = 10)
                        if (results != null) {
                            val sessionFolder = session.getSessionFolder(storage.baseDir)
                            saveTranscriptFiles(session, sessionFolder, results)
                            session.status = SessionStatus.DONE
                            storage.saveSession(session)
                            successCount++
                            continue
                        }
                        // Still processing, try transcribe again
                        transcribe(serverUrl, sessionId)
                        val results2 = waitForResults(serverUrl, sessionId)
                        if (results2 != null) {
                            val sessionFolder = session.getSessionFolder(storage.baseDir)
                            saveTranscriptFiles(session, sessionFolder, results2)
                            session.status = SessionStatus.DONE
                            storage.saveSession(session)
                            successCount++
                            continue
                        }
                        // Still not done, reset to PENDING for next retry
                        session.status = SessionStatus.PENDING
                        storage.saveSession(session)
                    } else {
                        // Stuck in UPLOADING but not on server, reset
                        session.status = SessionStatus.PENDING
                        storage.saveSession(session)
                    }
                }
                
                session.status = SessionStatus.UPLOADING
                storage.saveSession(session)
                
                val sessionFolder = session.getSessionFolder(storage.baseDir)
                val audioFile = session.getAudioFile(sessionFolder)
                
                if (!audioFile.exists() || audioFile.length() == 0L) {
                    Log.w(TAG, "Audio file not found or empty for session ${session.sessionId}")
                    session.status = SessionStatus.ERROR
                    storage.saveSession(session)
                    failureCount++
                    continue
                }
                
                // Upload audio and metadata
                val uploadResult = uploadAudio(serverUrl, audioFile, session)
                val sessionId = session.sessionId
                
                // Handle idempotent response
                if (uploadResult["status"] == "already_exists") {
                    val existingStatus = uploadResult["existing_status"] as? String
                    if (existingStatus == "done") {
                        // Already done, download results
                        val results = waitForResults(serverUrl, sessionId, maxWait = 5)
                        if (results != null) {
                            saveTranscriptFiles(session, sessionFolder, results)
                            session.status = SessionStatus.DONE
                            storage.saveSession(session)
                            successCount++
                            continue
                        }
                    }
                    // If uploaded or processing, continue to transcribe
                }
                
                // Start transcription
                transcribe(serverUrl, sessionId)
                
                // Wait for transcription and download results
                val results = waitForResults(serverUrl, sessionId)
                
                if (results != null) {
                    // Save transcript files
                    saveTranscriptFiles(session, sessionFolder, results)
                    session.status = SessionStatus.DONE
                    storage.saveSession(session)
                    successCount++
                } else {
                    throw Exception("Failed to get transcription results")
                }
                
            } catch (e: Exception) {
                Log.e(TAG, "Error processing session ${session.sessionId}", e)
                session.status = SessionStatus.ERROR
                session.retries = maxOf(0, session.retries - 1)
                storage.saveSession(session)
                failureCount++
            }
        }
        
        if (successCount > 0) {
            Result.success()
        } else if (failureCount > 0) {
            Result.retry()
        } else {
            Result.success()
        }
    }
    
    private fun checkServerHealth(serverUrl: String): Boolean {
        return try {
            val request = Request.Builder()
                .url("$serverUrl/api/health")
                .get()
                .build()
            
            val response = client.newCall(request).execute()
            response.isSuccessful && response.body?.string()?.contains("ok") == true
        } catch (e: Exception) {
            false
        }
    }
    
    private fun uploadAudio(serverUrl: String, audioFile: File, session: Session): Map<String, Any> {
        val requestBody = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart("audio", "audio.wav", audioFile.asRequestBody("audio/wav".toMediaType()))
            .addFormDataPart("metadata", session.toJson().toRequestBody("application/json".toMediaType()))
            .build()
        
        val request = Request.Builder()
            .url("$serverUrl/api/upload")
            .post(requestBody)
            .build()
        
        val response = client.newCall(request).execute()
        if (!response.isSuccessful) {
            throw IOException("Upload failed: ${response.code}")
        }
        
        val responseBody = response.body?.string() ?: throw IOException("Empty response")
        val json = Gson().fromJson(responseBody, Map::class.java) as Map<String, Any>
        return json
    }
    
    private fun checkSessionExistsOnServer(serverUrl: String, sessionId: String): Boolean {
        return try {
            val request = Request.Builder()
                .url("$serverUrl/api/session/$sessionId")
                .get()
                .build()
            
            val response = client.newCall(request).execute()
            response.isSuccessful
        } catch (e: Exception) {
            false
        }
    }
    
    private fun transcribe(serverUrl: String, sessionId: String) {
        val request = Request.Builder()
            .url("$serverUrl/api/transcribe/$sessionId")
            .post(RequestBody.create(null, ByteArray(0)))
            .build()
        
        val response = client.newCall(request).execute()
        if (!response.isSuccessful) {
            throw IOException("Transcribe request failed: ${response.code}")
        }
    }
    
    private fun waitForResults(serverUrl: String, sessionId: String, maxWait: Int = 60): Map<String, Any>? {
        var waited = 0
        while (waited < maxWait) {
            val request = Request.Builder()
                .url("$serverUrl/api/session/$sessionId")
                .get()
                .build()
            
            val response = client.newCall(request).execute()
            if (response.isSuccessful) {
                val responseBody = response.body?.string()
                if (responseBody != null) {
                    val json = Gson().fromJson(responseBody, Map::class.java) as Map<String, Any>
                    val status = json["status"] as? String
                    if (status == "done") {
                        return json
                    }
                }
            }
            
            Thread.sleep(2000)
            waited += 2
        }
        return null
    }
    
    private fun saveTranscriptFiles(session: Session, sessionFolder: File, results: Map<String, Any>) {
        // Save transcript.txt
        val transcript = results["transcript"] as? String ?: ""
        session.getTranscriptFile(sessionFolder).writeText(transcript)
        
        // Save transcript_timestamps.json
        val timestamps = results["timestamps"] as? List<Map<String, Any>> ?: emptyList()
        session.getTimestampsFile(sessionFolder).writeText(Gson().toJson(timestamps))
        
        // Save summary.json
        val summary = results["summary"] as? Map<String, Any> ?: emptyMap()
        session.getSummaryFile(sessionFolder).writeText(Gson().toJson(summary))
        
        // Save analytics.json
        val analytics = results["analytics"] as? Map<String, Any> ?: emptyMap()
        session.getAnalyticsFile(sessionFolder).writeText(Gson().toJson(analytics))
    }
}

