package com.voice.recordtranscript.ui

import android.Manifest
import android.content.pm.PackageManager
import android.media.MediaRecorder
import android.os.Bundle
import android.view.MotionEvent
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.work.Constraints
import androidx.work.ExistingPeriodicWorkPolicy
import androidx.work.NetworkType
import androidx.work.PeriodicWorkRequestBuilder
import androidx.work.WorkManager
import com.voice.recordtranscript.R
import com.voice.recordtranscript.data.Session
import com.voice.recordtranscript.data.SessionStatus
import com.voice.recordtranscript.databinding.ActivityMainBinding
import com.voice.recordtranscript.recording.AudioRecorder
import com.voice.recordtranscript.storage.SessionStorage
import com.voice.recordtranscript.upload.UploadWorker
import android.widget.AdapterView
import kotlinx.coroutines.launch
import java.io.File
import java.util.concurrent.TimeUnit

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private lateinit var storage: SessionStorage
    private var audioRecorder: AudioRecorder? = null
    private var currentSession: Session? = null
    private var isRecording = false
    private var isDevMode = false
    private lateinit var sessionsAdapter: SessionsAdapter
    
    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (!isGranted) {
            Toast.makeText(this, "Microphone permission required", Toast.LENGTH_LONG).show()
        }
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        
        storage = SessionStorage(this)
        setupUI()
        setupWorkManager()
        loadSessions()
        checkPermissions()
    }
    
    private fun setupUI() {
        isDevMode = getSharedPreferences("settings", MODE_PRIVATE)
            .getBoolean("dev_mode", false)
        
        updateUIMode()
        
        binding.recordButton.setOnTouchListener { _, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> startRecording()
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> stopRecording()
            }
            true
        }
        
        binding.syncButton.setOnClickListener {
            triggerSync()
        }
        
        binding.settingsButton.setOnClickListener {
            showSettingsDialog()
        }
        
        // Swipe gesture for dev/prod mode
        binding.root.setOnTouchListener { _, event ->
            if (event.action == MotionEvent.ACTION_MOVE && event.pointerCount == 2) {
                isDevMode = !isDevMode
                getSharedPreferences("settings", MODE_PRIVATE).edit()
                    .putBoolean("dev_mode", isDevMode).apply()
                updateUIMode()
                true
            } else {
                false
            }
        }
        
        sessionsAdapter = SessionsAdapter(emptyList()) { session ->
            showSessionDetails(session)
        }
        binding.sessionsRecyclerView.layoutManager = LinearLayoutManager(this)
        binding.sessionsRecyclerView.adapter = sessionsAdapter
    }
    
    private fun updateUIMode() {
        if (isDevMode) {
            binding.devPanel.visibility = android.view.View.VISIBLE
            binding.devModeLabel.text = "DEV MODE"
            loadDevSettings()
        } else {
            binding.devPanel.visibility = android.view.View.GONE
            binding.devModeLabel.text = "PROD MODE"
        }
    }
    
    private fun loadDevSettings() {
        val prefs = getSharedPreferences("settings", MODE_PRIVATE)
        binding.languageSpinner.setSelection(
            when (prefs.getString("language", "en")) {
                "ru" -> 0
                "en" -> 1
                "fr" -> 2
                else -> 1
            }
        )
        binding.diarizationSwitch.isChecked = prefs.getBoolean("diarization_enabled", false)
        binding.speakerCountSpinner.setSelection(
            when (prefs.getInt("speaker_count", 1)) {
                1 -> 0
                2 -> 1
                3 -> 2
                else -> 0
            }
        )
        
        // Save settings when changed
        binding.languageSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: android.widget.AdapterView<*>?, view: android.view.View?, position: Int, id: Long) {
                val lang = when (position) {
                    0 -> "ru"
                    1 -> "en"
                    2 -> "fr"
                    else -> "en"
                }
                prefs.edit().putString("language", lang).apply()
            }
            override fun onNothingSelected(parent: android.widget.AdapterView<*>?) {}
        }
        
        binding.diarizationSwitch.setOnCheckedChangeListener { _, isChecked ->
            prefs.edit().putBoolean("diarization_enabled", isChecked).apply()
        }
        
        binding.speakerCountSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: android.widget.AdapterView<*>?, view: android.view.View?, position: Int, id: Long) {
                val count = when (position) {
                    0 -> 1
                    1 -> 2
                    2 -> 3
                    else -> 1
                }
                prefs.edit().putInt("speaker_count", count).apply()
            }
            override fun onNothingSelected(parent: android.widget.AdapterView<*>?) {}
        }
    }
    
    private fun checkPermissions() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED
        ) {
            requestPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
        }
    }
    
    private fun startRecording() {
        if (isRecording) return
        
        checkPermissions()
        
        val prefs = getSharedPreferences("settings", MODE_PRIVATE)
        val language = prefs.getString("language", "en") ?: "en"
        val diarizationEnabled = prefs.getBoolean("diarization_enabled", false)
        val speakerCount = prefs.getInt("speaker_count", 1)
        
        currentSession = storage.createSession(language, diarizationEnabled, speakerCount)
        val sessionFolder = currentSession!!.getSessionFolder(storage.baseDir)
        val audioFile = currentSession!!.getAudioFile(sessionFolder)
        
        audioRecorder = AudioRecorder(audioFile)
        audioRecorder?.start()
        
        isRecording = true
        binding.recordButton.text = getString(R.string.stop)
        binding.statusText.text = "Recording..."
    }
    
    private fun stopRecording() {
        if (!isRecording) return
        
        audioRecorder?.stop()
        audioRecorder = null
        
        isRecording = false
        binding.recordButton.text = getString(R.string.record)
        
        currentSession?.let { session ->
            val sessionFolder = session.getSessionFolder(storage.baseDir)
            val audioFile = session.getAudioFile(sessionFolder)
            
            // Verify WAV file exists and is not empty before marking ready
            if (!audioFile.exists() || audioFile.length() == 0L) {
                binding.statusText.text = "Error: Recording not saved"
                session.status = com.voice.recordtranscript.data.SessionStatus.ERROR
                storage.saveSession(session)
                currentSession = null
                loadSessions()
                return
            }
            
            binding.statusText.text = "Recording saved"
            storage.saveSession(session)
            loadSessions()
            triggerSync()
        }
        
        currentSession = null
    }
    
    private fun triggerSync() {
        val workRequest = androidx.work.OneTimeWorkRequestBuilder<UploadWorker>()
            .setConstraints(
                Constraints.Builder()
                    .setRequiredNetworkType(NetworkType.CONNECTED)
                    .build()
            )
            .build()
        
        WorkManager.getInstance(this).enqueue(workRequest)
        binding.statusText.text = "Syncing..."
        
        lifecycleScope.launch {
            kotlinx.coroutines.delay(2000)
            loadSessions()
            binding.statusText.text = "Sync triggered"
        }
    }
    
    private fun setupWorkManager() {
        val constraints = Constraints.Builder()
            .setRequiredNetworkType(NetworkType.CONNECTED)
            .build()
        
        val periodicWork = PeriodicWorkRequestBuilder<UploadWorker>(15, TimeUnit.MINUTES)
            .setConstraints(constraints)
            .build()
        
        WorkManager.getInstance(this).enqueueUniquePeriodicWork(
            "upload_work",
            ExistingPeriodicWorkPolicy.KEEP,
            periodicWork
        )
    }
    
    private fun loadSessions() {
        lifecycleScope.launch {
            val sessions = storage.getAllSessions()
            sessionsAdapter.updateSessions(sessions)
        }
    }
    
    private fun showSettingsDialog() {
        val prefs = getSharedPreferences("settings", MODE_PRIVATE)
        val dialog = android.app.AlertDialog.Builder(this)
            .setTitle("Settings")
            .setView(android.widget.EditText(this).apply {
                setText(prefs.getString("server_url", ""))
                hint = "http://192.168.1.100:8000"
            })
            .setPositiveButton("Save") { dialog, _ ->
                val serverUrl = (dialog as? android.app.AlertDialog)
                    ?.findViewById<android.widget.EditText>(android.R.id.text1)?.text?.toString() ?: ""
                prefs.edit().putString("server_url", serverUrl).apply()
                Toast.makeText(this, "Server URL saved", Toast.LENGTH_SHORT).show()
            }
            .setNegativeButton("Cancel", null)
            .create()
        dialog.show()
    }
    
    private fun showSessionDetails(session: Session) {
        val sessionFolder = session.getSessionFolder(storage.baseDir)
        val transcriptFile = session.getTranscriptFile(sessionFolder)
        
        val message = if (transcriptFile.exists()) {
            transcriptFile.readText().take(500)
        } else {
            "Status: ${session.status}\nRetries: ${session.retries}"
        }
        
        android.app.AlertDialog.Builder(this)
            .setTitle("Session ${session.sessionId.take(8)}")
            .setMessage(message)
            .setPositiveButton("OK", null)
            .show()
    }
    
    override fun onDestroy() {
        super.onDestroy()
        audioRecorder?.stop()
    }
}

