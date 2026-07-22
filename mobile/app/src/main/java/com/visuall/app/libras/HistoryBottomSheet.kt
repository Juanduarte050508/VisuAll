package com.visuall.app.libras

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.os.Parcelable
import android.view.Gravity
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.bottomsheet.BottomSheetDialogFragment
import com.visuall.app.R
import com.visuall.app.databinding.DialogHistoryBinding
import kotlinx.parcelize.Parcelize
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

// ── Model ──────────────────────────────────────────────────────────────────
@Parcelize
data class HistoryEntry(
    val word: String,
    val timestamp: Long = System.currentTimeMillis(),
    val source: String = "LIBRAS"
) : Parcelable {
    fun formattedTime(): String =
        SimpleDateFormat("HH:mm", Locale.getDefault()).format(Date(timestamp))
}

// ── Bottom Sheet ───────────────────────────────────────────────────────────
class HistoryBottomSheet : BottomSheetDialogFragment() {

    private var _binding: DialogHistoryBinding? = null
    private val binding get() = _binding!!
    var onClearConversation: (() -> Unit)? = null

    companion object {
        private const val ARG_WORDS = "words"

        fun newInstance(words: ArrayList<HistoryEntry>): HistoryBottomSheet {
            return HistoryBottomSheet().also { sheet ->
                sheet.arguments = Bundle().apply {
                    putParcelableArrayList(ARG_WORDS, words)
                }
            }
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = DialogHistoryBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        @Suppress("DEPRECATION")
        val words = arguments?.getParcelableArrayList<HistoryEntry>(ARG_WORDS) ?: arrayListOf()

        binding.rvHistory.layoutManager = LinearLayoutManager(requireContext())
        binding.rvHistory.adapter = HistoryAdapter(words)
        updateEmptyState(words.isEmpty())

        binding.btnCopyConversation.setOnClickListener {
            copiarConversa(words)
        }

        binding.btnShareConversation.setOnClickListener {
            compartilharConversa(words)
        }

        binding.btnClearConversation.setOnClickListener {
            if (words.isEmpty()) return@setOnClickListener
            onClearConversation?.invoke()
            words.clear()
            binding.rvHistory.adapter = HistoryAdapter(words)
            updateEmptyState(empty = true)
            Toast.makeText(requireContext(), "Conversa limpa", Toast.LENGTH_SHORT).show()
        }
    }

    private fun updateEmptyState(empty: Boolean) {
        binding.tvEmpty.visibility = if (empty) View.VISIBLE else View.GONE
        binding.rvHistory.visibility = if (empty) View.GONE else View.VISIBLE
        binding.shareActions.alpha = if (empty) 0.35f else 1f
        binding.btnCopyConversation.isEnabled = !empty
        binding.btnShareConversation.isEnabled = !empty
        binding.btnClearConversation.isEnabled = !empty
    }

    private fun copiarConversa(items: List<HistoryEntry>) {
        val texto = formatarConversa(items)
        if (texto.isBlank()) return

        val clipboard = requireContext()
            .getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        clipboard.setPrimaryClip(ClipData.newPlainText("Conversa VisuAll", texto))
        Toast.makeText(requireContext(), "Conversa copiada", Toast.LENGTH_SHORT).show()
    }

    private fun compartilharConversa(items: List<HistoryEntry>) {
        val texto = formatarConversa(items)
        if (texto.isBlank()) return

        val intent = Intent(Intent.ACTION_SEND).apply {
            type = "text/plain"
            putExtra(Intent.EXTRA_TEXT, texto)
        }
        startActivity(Intent.createChooser(intent, "Compartilhar conversa"))
    }

    private fun formatarConversa(items: List<HistoryEntry>): String {
        if (items.isEmpty()) return ""
        return buildString {
            appendLine("Conversa VisuAll")
            appendLine()
            items.forEach { item ->
                val origem = if (item.source == "RESPOSTA") "Resposta" else "Libras"
                appendLine("[${item.formattedTime()}] $origem: ${item.word}")
            }
        }.trim()
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}

// ── Adapter ────────────────────────────────────────────────────────────────
class HistoryAdapter(private val items: List<HistoryEntry>) :
    RecyclerView.Adapter<HistoryAdapter.VH>() {

    inner class VH(view: View) : RecyclerView.ViewHolder(view) {
        val root: LinearLayout = view.findViewById(R.id.history_item_root)
        val header: LinearLayout = view.findViewById(R.id.history_header)
        val tvWord: TextView = view.findViewById(R.id.tv_word)
        val tvTime: TextView = view.findViewById(R.id.tv_time)
        val tvSource: TextView = view.findViewById(R.id.tv_source)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): VH {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_history, parent, false)
        return VH(view)
    }

    override fun onBindViewHolder(holder: VH, position: Int) {
        // Mais recente primeiro
        val item = items[items.size - 1 - position]
        holder.tvWord.text = item.word
        holder.tvTime.text = item.formattedTime()
        holder.tvSource.text = if (item.source == "RESPOSTA") "VOZ/TEXTO" else "LIBRAS"

        if (item.source == "RESPOSTA") {
            holder.root.gravity = Gravity.END
            holder.header.gravity = Gravity.CENTER_VERTICAL or Gravity.END
            holder.tvSource.setBackgroundResource(R.drawable.vf_bg_mode_active)
            holder.tvSource.setTextColor(0xFF070707.toInt())
            holder.tvWord.setBackgroundResource(R.drawable.vf_bg_reply_text)
        } else {
            holder.root.gravity = Gravity.START
            holder.header.gravity = Gravity.CENTER_VERTICAL or Gravity.START
            holder.tvSource.setBackgroundResource(R.drawable.vf_bg_mode_inactive)
            holder.tvSource.setTextColor(0xFFF5F1E8.toInt())
            holder.tvWord.setBackgroundResource(R.drawable.vf_bg_phrase)
        }
    }

    override fun getItemCount() = items.size
}
