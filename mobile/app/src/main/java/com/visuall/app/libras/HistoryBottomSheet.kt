package com.visuall.app.libras

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.os.Parcelable
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
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
//
// Uma folha por metade da conversa: o que foi sinalizado e o que foi respondido
// sao consultados em momentos diferentes e por pessoas diferentes, entao cada
// uma abre da sua propria tela. O titulo diz qual esta aberta.
//
// A ordem dos elementos e deliberada: conteudo primeiro, acoes depois. Antes as
// tres acoes ficavam ACIMA da lista, e a destrutiva (limpar) dividia peso e
// aparencia com copiar, enquanto compartilhar era a unica dourada -- a enfase
// visual caia na acao menos usada e a perigosa ficava a um toque de distancia
// no meio das outras. Agora copiar e compartilhar sao dois botoes iguais no
// rodape e limpar e um icone separado, com confirmacao.
class HistoryBottomSheet : BottomSheetDialogFragment() {

    private var _binding: DialogHistoryBinding? = null
    private val binding get() = _binding!!
    var onClearConversation: (() -> Unit)? = null

    companion object {
        private const val ARG_WORDS = "words"
        private const val ARG_TITULO = "titulo"

        fun newInstance(words: List<HistoryEntry>, titulo: String): HistoryBottomSheet {
            return HistoryBottomSheet().also { sheet ->
                sheet.arguments = Bundle().apply {
                    // Copia defensiva: em processo unico, um Bundle guarda a
                    // MESMA referencia da lista (nao serializa de verdade ate
                    // cruzar um processo). Sem isso, letras reconhecidas
                    // enquanto o historico esta aberto mudam a lista por
                    // baixo do RecyclerView sem notifyDataSetChanged().
                    putParcelableArrayList(ARG_WORDS, ArrayList(words))
                    putString(ARG_TITULO, titulo)
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
        val titulo = arguments?.getString(ARG_TITULO).orEmpty()

        binding.tvHistoryTitle.text = titulo
        binding.rvHistory.layoutManager = LinearLayoutManager(requireContext())
        binding.rvHistory.adapter = HistoryAdapter(words)
        updateEmptyState(words)

        binding.btnCopyConversation.setOnClickListener { copiarConversa(words) }
        binding.btnShareConversation.setOnClickListener { compartilharConversa(words) }

        binding.btnClearConversation.setOnClickListener {
            if (words.isEmpty()) return@setOnClickListener
            // Apagar o historico nao tem desfazer, e o botao fica ao lado de
            // dois inofensivos. Uma pergunta e barata perto de perder a conversa.
            AlertDialog.Builder(requireContext())
                .setTitle("Apagar este historico?")
                .setMessage("As ${words.size} mensagens de \"$titulo\" serao removidas. Nao da pra desfazer.")
                .setNegativeButton("Cancelar", null)
                .setPositiveButton("Apagar") { _, _ ->
                    onClearConversation?.invoke()
                    words.clear()
                    binding.rvHistory.adapter = HistoryAdapter(words)
                    updateEmptyState(words)
                    Toast.makeText(requireContext(), "Historico apagado", Toast.LENGTH_SHORT).show()
                }
                .show()
        }
    }

    private fun updateEmptyState(words: List<HistoryEntry>) {
        val empty = words.isEmpty()
        binding.tvEmpty.visibility = if (empty) View.VISIBLE else View.GONE
        binding.rvHistory.visibility = if (empty) View.GONE else View.VISIBLE
        binding.tvHistoryCount.text = when (words.size) {
            0 -> ""
            1 -> "1 mensagem"
            else -> "${words.size} mensagens"
        }
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
        Toast.makeText(requireContext(), "Copiado", Toast.LENGTH_SHORT).show()
    }

    private fun compartilharConversa(items: List<HistoryEntry>) {
        val texto = formatarConversa(items)
        if (texto.isBlank()) return

        val intent = Intent(Intent.ACTION_SEND).apply {
            type = "text/plain"
            putExtra(Intent.EXTRA_TEXT, texto)
        }
        startActivity(Intent.createChooser(intent, "Compartilhar"))
    }

    private fun formatarConversa(items: List<HistoryEntry>): String {
        if (items.isEmpty()) return ""
        val titulo = arguments?.getString(ARG_TITULO).orEmpty()
        return buildString {
            appendLine("VisuAll — $titulo")
            appendLine()
            items.forEach { item ->
                appendLine("[${item.formattedTime()}] ${item.word}")
            }
        }.trim()
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}

// ── Adapter ────────────────────────────────────────────────────────────────
//
// A etiqueta de origem em cada item saiu: como cada folha mostra uma origem so,
// repeti-la linha a linha era ruido -- o titulo da folha ja diz de quem e.
class HistoryAdapter(private val items: List<HistoryEntry>) :
    RecyclerView.Adapter<HistoryAdapter.VH>() {

    inner class VH(view: View) : RecyclerView.ViewHolder(view) {
        val tvWord: TextView = view.findViewById(R.id.tv_word)
        val tvTime: TextView = view.findViewById(R.id.tv_time)
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
        holder.tvWord.setBackgroundResource(
            if (item.source == "RESPOSTA") R.drawable.vf_bg_reply_text
            else R.drawable.vf_bg_phrase
        )
    }

    override fun getItemCount() = items.size
}
