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
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ColorFilter
import androidx.compose.ui.platform.ComposeView
import androidx.compose.ui.platform.ViewCompositionStrategy
import androidx.compose.ui.res.colorResource
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.google.android.material.bottomsheet.BottomSheetDialogFragment
import com.visuall.app.R
import kotlinx.parcelize.Parcelize
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

// -- Model -----------------------------------------------------------------
@Parcelize
data class HistoryEntry(
    val word: String,
    val timestamp: Long = System.currentTimeMillis(),
    val source: String = "LIBRAS"
) : Parcelable {
    fun formattedTime(): String =
        SimpleDateFormat("HH:mm", Locale.getDefault()).format(Date(timestamp))
}

// -- Bottom Sheet ----------------------------------------------------------
//
// A folha do historico virou Compose porque e uma UI declarativa simples:
// titulo, estado vazio/lista e acoes. Isso remove XML + RecyclerView.Adapter
// sem encostar no ciclo pesado da camera e dos modelos.
class HistoryBottomSheet : BottomSheetDialogFragment() {

    var onClearConversation: (() -> Unit)? = null

    companion object {
        private const val ARG_WORDS = "words"
        private const val ARG_TITULO = "titulo"

        fun newInstance(words: List<HistoryEntry>, titulo: String): HistoryBottomSheet {
            return HistoryBottomSheet().also { sheet ->
                sheet.arguments = Bundle().apply {
                    // Copia defensiva: em processo unico, um Bundle guarda a
                    // MESMA referencia da lista ate precisar parcelar de fato.
                    putParcelableArrayList(ARG_WORDS, ArrayList(words))
                    putString(ARG_TITULO, titulo)
                }
            }
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        @Suppress("DEPRECATION")
        val initialWords = arguments
            ?.getParcelableArrayList<HistoryEntry>(ARG_WORDS)
            ?.toList()
            .orEmpty()
        val titulo = arguments?.getString(ARG_TITULO).orEmpty()

        return ComposeView(requireContext()).apply {
            setViewCompositionStrategy(ViewCompositionStrategy.DisposeOnDetachedFromWindow)
            setContent {
                var words by remember { mutableStateOf(initialWords) }

                HistorySheetContent(
                    title = titulo,
                    entries = words,
                    onCopy = { copiarConversa(words, titulo) },
                    onShare = { compartilharConversa(words, titulo) },
                    onClear = {
                        if (words.isEmpty()) return@HistorySheetContent
                        confirmarLimpeza(titulo, words.size) {
                            onClearConversation?.invoke()
                            words = emptyList()
                            Toast.makeText(
                                requireContext(),
                                "Historico apagado",
                                Toast.LENGTH_SHORT
                            ).show()
                        }
                    }
                )
            }
        }
    }

    private fun confirmarLimpeza(titulo: String, total: Int, onConfirm: () -> Unit) {
        val mensagem = if (total == 1) {
            "A mensagem de \"$titulo\" sera removida. Nao da pra desfazer."
        } else {
            "As $total mensagens de \"$titulo\" serao removidas. Nao da pra desfazer."
        }

        AlertDialog.Builder(requireContext())
            .setTitle("Apagar este historico?")
            .setMessage(mensagem)
            .setNegativeButton("Cancelar", null)
            .setPositiveButton("Apagar") { _, _ -> onConfirm() }
            .show()
    }

    private fun copiarConversa(items: List<HistoryEntry>, titulo: String) {
        val texto = formatarConversa(items, titulo)
        if (texto.isBlank()) return

        val clipboard = requireContext()
            .getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        clipboard.setPrimaryClip(ClipData.newPlainText("Conversa VisuAll", texto))
        Toast.makeText(requireContext(), "Copiado", Toast.LENGTH_SHORT).show()
    }

    private fun compartilharConversa(items: List<HistoryEntry>, titulo: String) {
        val texto = formatarConversa(items, titulo)
        if (texto.isBlank()) return

        val intent = Intent(Intent.ACTION_SEND).apply {
            type = "text/plain"
            putExtra(Intent.EXTRA_TEXT, texto)
        }
        startActivity(Intent.createChooser(intent, "Compartilhar"))
    }

    private fun formatarConversa(items: List<HistoryEntry>, titulo: String): String {
        if (items.isEmpty()) return ""
        return buildString {
            appendLine("VisuAll - $titulo")
            appendLine()
            items.forEach { item ->
                appendLine("[${item.formattedTime()}] ${item.word}")
            }
        }.trim()
    }
}

@Composable
private fun HistorySheetContent(
    title: String,
    entries: List<HistoryEntry>,
    onCopy: () -> Unit,
    onShare: () -> Unit,
    onClear: () -> Unit
) {
    val gold = colorResource(R.color.gold_primary)
    val surface = colorResource(R.color.surface)
    val textPrimary = colorResource(R.color.text_primary)
    val textMuted = colorResource(R.color.text_muted)
    val empty = entries.isEmpty()

    MaterialTheme(
        colorScheme = darkColorScheme(
            primary = gold,
            surface = surface,
            onSurface = textPrimary
        )
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(topStart = 20.dp, topEnd = 20.dp))
                .background(surface)
                .padding(top = 12.dp, bottom = 20.dp)
        ) {
            Box(
                modifier = Modifier
                    .size(width = 40.dp, height = 4.dp)
                    .clip(RoundedCornerShape(2.dp))
                    .background(Color(0xFF444444))
                    .align(Alignment.CenterHorizontally)
            )

            Spacer(Modifier.height(18.dp))

            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 22.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = title.ifBlank { "Historico" },
                    modifier = Modifier.weight(1f),
                    color = textPrimary,
                    fontSize = 19.sp,
                    fontWeight = FontWeight.Bold,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )

                Text(
                    text = historyCountText(entries.size),
                    color = textMuted,
                    fontSize = 12.sp,
                    maxLines = 1
                )
            }

            Spacer(Modifier.height(14.dp))

            if (empty) {
                Text(
                    text = "Nada registrado ainda.",
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 36.dp),
                    color = textMuted,
                    fontSize = 14.sp,
                    textAlign = TextAlign.Center
                )
            } else {
                LazyColumn(
                    modifier = Modifier
                        .fillMaxWidth()
                        .heightIn(max = 440.dp)
                        .padding(horizontal = 22.dp)
                ) {
                    items(
                        items = entries.asReversed(),
                        key = { entry -> "${entry.timestamp}:${entry.source}:${entry.word}" }
                    ) { entry ->
                        HistoryRow(entry)
                    }
                }
            }

            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(top = 14.dp, bottom = 14.dp)
                    .height(1.dp)
                    .background(Color(0xFF242424))
            )

            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 22.dp),
                horizontalArrangement = Arrangement.Center,
                verticalAlignment = Alignment.CenterVertically
            ) {
                HistoryActionButton(
                    text = "COPIAR",
                    enabled = !empty,
                    modifier = Modifier.weight(1f),
                    onClick = onCopy
                )
                Spacer(Modifier.size(8.dp))
                HistoryActionButton(
                    text = "COMPARTILHAR",
                    enabled = !empty,
                    modifier = Modifier.weight(1f),
                    onClick = onShare
                )
                Spacer(Modifier.size(20.dp))
                ClearHistoryButton(enabled = !empty, onClick = onClear)
            }
        }
    }
}

@Composable
private fun HistoryRow(entry: HistoryEntry) {
    val textPrimary = colorResource(R.color.text_primary)
    val textMuted = colorResource(R.color.text_muted)
    val bubbleShape = RoundedCornerShape(14.dp)
    val response = entry.source == "RESPOSTA"
    val bubbleColor = if (response) Color(0xE6111111) else Color(0xE0111111)
    val borderColor = if (response) Color(0x66E8A020) else Color(0x22FFFFFF)

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 7.dp)
    ) {
        Text(
            text = entry.formattedTime(),
            modifier = Modifier.padding(start = 4.dp, bottom = 4.dp),
            color = textMuted,
            fontSize = 11.sp
        )
        Text(
            text = entry.word,
            modifier = Modifier
                .clip(bubbleShape)
                .background(bubbleColor)
                .border(1.dp, borderColor, bubbleShape)
                .padding(horizontal = 16.dp, vertical = 11.dp),
            color = textPrimary,
            fontSize = 16.sp,
            lineHeight = 20.sp
        )
    }
}

@Composable
private fun HistoryActionButton(
    text: String,
    enabled: Boolean,
    modifier: Modifier = Modifier,
    onClick: () -> Unit
) {
    val textPrimary = colorResource(R.color.text_primary)
    val shape = RoundedCornerShape(18.dp)

    Box(
        modifier = modifier
            .height(48.dp)
            .alpha(if (enabled) 1f else 0.35f)
            .clip(shape)
            .background(Color(0xCC242424))
            .border(1.dp, Color(0x66E8A020), shape)
            .clickable(enabled = enabled, role = Role.Button, onClick = onClick),
        contentAlignment = Alignment.Center
    ) {
        Text(
            text = text,
            modifier = Modifier.padding(horizontal = 8.dp),
            color = textPrimary,
            fontSize = 13.sp,
            fontWeight = FontWeight.Bold,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis
        )
    }
}

@Composable
private fun ClearHistoryButton(enabled: Boolean, onClick: () -> Unit) {
    val textPrimary = colorResource(R.color.text_primary)
    val goldLight = colorResource(R.color.gold_light)

    Box(
        modifier = Modifier
            .size(48.dp)
            .alpha(if (enabled) 1f else 0.35f)
            .clip(CircleShape)
            .background(Color(0xE6111111))
            .border(2.dp, goldLight, CircleShape)
            .clickable(
                enabled = enabled,
                onClickLabel = "Apagar este historico",
                role = Role.Button,
                onClick = onClick
            ),
        contentAlignment = Alignment.Center
    ) {
        Image(
            painter = painterResource(R.drawable.ic_delete),
            contentDescription = null,
            modifier = Modifier.size(22.dp),
            colorFilter = ColorFilter.tint(textPrimary)
        )
    }
}

private fun historyCountText(total: Int): String {
    return when (total) {
        0 -> ""
        1 -> "1 mensagem"
        else -> "$total mensagens"
    }
}
