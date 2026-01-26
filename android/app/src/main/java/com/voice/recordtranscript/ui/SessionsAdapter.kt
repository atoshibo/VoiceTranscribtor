package com.voice.recordtranscript.ui

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import com.voice.recordtranscript.R
import com.voice.recordtranscript.data.Session
import java.text.SimpleDateFormat
import java.util.Locale

class SessionsAdapter(
    private var sessions: List<Session>,
    private val onSessionClick: (Session) -> Unit
) : RecyclerView.Adapter<SessionsAdapter.ViewHolder>() {
    
    class ViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val sessionIdText: TextView = view.findViewById(R.id.sessionIdText)
        val statusText: TextView = view.findViewById(R.id.statusText)
        val dateText: TextView = view.findViewById(R.id.dateText)
    }
    
    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_session, parent, false)
        return ViewHolder(view)
    }
    
    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        val session = sessions[position]
        holder.sessionIdText.text = "Session ${session.sessionId.take(8)}"
        holder.statusText.text = session.status.name
        holder.dateText.text = SimpleDateFormat("MMM dd, HH:mm", Locale.US)
            .format(java.util.Date(session.createdAt))
        
        holder.itemView.setOnClickListener {
            onSessionClick(session)
        }
    }
    
    override fun getItemCount() = sessions.size
    
    fun updateSessions(newSessions: List<Session>) {
        sessions = newSessions
        notifyDataSetChanged()
    }
}

