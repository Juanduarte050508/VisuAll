package com.visuall.app.ui.compose

import androidx.annotation.DrawableRes
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ColorFilter
import androidx.compose.ui.res.colorResource
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import com.visuall.app.R
import com.visuall.app.ui.ScanFrameView

@Composable
fun CameraLandscapeHud(
    aspectRatioLabel: String,
    onLibrasClick: () -> Unit,
    onFlipClick: () -> Unit,
    onAspectRatioClick: () -> Unit
) {
    val gold = colorResource(R.color.gold_primary)

    Box(modifier = Modifier.fillMaxSize()) {
        Column(
            modifier = Modifier
                .align(Alignment.CenterEnd)
                .padding(end = 14.dp)
                .width(56.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            HudIconButton(
                iconRes = R.drawable.ic_accessibility,
                contentDescription = "Entrar no modo Libras",
                tint = gold,
                onClick = onLibrasClick
            )
            HudGap()
            HudIconButton(
                iconRes = R.drawable.ic_flip,
                contentDescription = "Girar camera",
                onClick = onFlipClick
            )
            HudGap()
            AspectRatioButton(
                label = aspectRatioLabel,
                onClick = onAspectRatioClick
            )
        }
    }
}

@Composable
fun LibrasLandscapeHud(
    onExitClick: () -> Unit,
    onFlipClick: () -> Unit
) {
    val gold = colorResource(R.color.gold_primary)

    Box(modifier = Modifier.fillMaxSize()) {
        AndroidView(
            factory = { context -> ScanFrameView(context) },
            modifier = Modifier
                .align(Alignment.Center)
                .size(width = 430.dp, height = 250.dp)
        )

        Column(
            modifier = Modifier
                .align(Alignment.CenterStart)
                .padding(start = 14.dp)
                .width(56.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            BackHudButton(onClick = onExitClick, textColor = gold)
            HudGap()
            HudIconButton(
                iconRes = R.drawable.ic_flip,
                contentDescription = "Girar camera",
                onClick = onFlipClick
            )
        }
    }
}

@Composable
private fun HudIconButton(
    @DrawableRes iconRes: Int,
    contentDescription: String,
    tint: Color = colorResource(R.color.text_primary),
    onClick: () -> Unit
) {
    val goldLight = colorResource(R.color.gold_light)

    Box(
        modifier = Modifier
            .size(48.dp)
            .clip(CircleShape)
            .background(Color(0xE6111111))
            .border(2.dp, goldLight, CircleShape)
            .clickable(
                onClickLabel = contentDescription,
                role = Role.Button,
                onClick = onClick
            ),
        contentAlignment = Alignment.Center
    ) {
        Image(
            painter = painterResource(iconRes),
            contentDescription = contentDescription,
            modifier = Modifier.size(24.dp),
            colorFilter = ColorFilter.tint(tint)
        )
    }
}

@Composable
private fun BackHudButton(onClick: () -> Unit, textColor: Color) {
    val goldLight = colorResource(R.color.gold_light)

    Box(
        modifier = Modifier
            .size(48.dp)
            .clip(CircleShape)
            .background(Color(0xE6111111))
            .border(2.dp, goldLight, CircleShape)
            .clickable(
                onClickLabel = "Voltar para camera normal",
                role = Role.Button,
                onClick = onClick
            ),
        contentAlignment = Alignment.Center
    ) {
        Text(
            text = "<",
            color = textColor,
            fontSize = 24.sp,
            fontWeight = FontWeight.Bold
        )
    }
}

@Composable
private fun AspectRatioButton(label: String, onClick: () -> Unit) {
    val shape = RoundedCornerShape(22.dp)

    Box(
        modifier = Modifier
            .size(width = 48.dp, height = 38.dp)
            .clip(shape)
            .background(Color(0xD90B0B0B))
            .border(1.dp, Color(0x66E8A020), shape)
            .clickable(
                onClickLabel = "Proporcao da foto",
                role = Role.Button,
                onClick = onClick
            ),
        contentAlignment = Alignment.Center
    ) {
        Text(
            text = label,
            color = Color.White,
            fontSize = 11.sp,
            fontWeight = FontWeight.Bold
        )
    }
}

@Composable
private fun HudGap() {
    Spacer(Modifier.height(12.dp))
}
