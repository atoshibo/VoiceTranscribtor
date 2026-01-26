package com.voice.recordtranscript.storage

import android.content.Context
import android.os.Environment
import com.voice.recordtranscript.data.Session
import com.voice.recordtranscript.data.SessionStatus
import java.io.File
import java.util.UUID

class SessionStorage(private val context: Context) {
    val baseDir: File
        get() {
            val externalDir = context.getExternalFilesDir(null)
                ?: context.filesDir
            return File(externalDir, "Recordings").apply { mkdirs() }
        }
    
    fun createSession(language: String, diarizationEnabled: Boolean, speakerCount: Int): Session {
        val sessionId = UUID.randomUUID().toString()
        val session = Session(
            sessionId = sessionId,
            status = SessionStatus.PENDING,
            language = language,
            diarizationEnabled = diarizationEnabled,
            speakerCount = speakerCount
        )
        
        val sessionFolder = session.getSessionFolder(baseDir)
        sessionFolder.mkdirs()
        
        val metadataFile = session.getMetadataFile(sessionFolder)
        metadataFile.writeText(session.toJson())
        
        return session
    }
    
    fun saveSession(session: Session) {
        val sessionFolder = session.getSessionFolder(baseDir)
        val metadataFile = session.getMetadataFile(sessionFolder)
        metadataFile.writeText(session.toJson())
    }
    
    fun loadSession(sessionId: String): Session? {
        val sessions = getAllSessions()
        return sessions.find { it.sessionId == sessionId }
    }
    
    fun getAllSessions(): List<Session> {
        val sessions = mutableListOf<Session>()
        val conversationsDir = File(baseDir, "Conversations")
        if (!conversationsDir.exists()) return sessions
        
        conversationsDir.walkTopDown().forEach { dateDir ->
            if (dateDir.isDirectory && dateDir.name.matches(Regex("\\d{4}-\\d{2}-\\d{2}"))) {
                dateDir.listFiles()?.forEach { sessionDir ->
                    if (sessionDir.isDirectory) {
                        val metadataFile = File(sessionDir, "session.json")
                        if (metadataFile.exists()) {
                            Session.fromJson(metadataFile.readText())?.let { sessions.add(it) }
                        }
                    }
                }
            }
        }
        
        return sessions.sortedByDescending { it.createdAt }
    }
    
    fun getPendingSessions(): List<Session> {
        return getAllSessions().filter { 
            it.status == SessionStatus.PENDING || 
            it.status == SessionStatus.UPLOADING || // Include stuck UPLOADING to recover
            (it.status == SessionStatus.ERROR && it.retries > 0)
        }
    }
}

