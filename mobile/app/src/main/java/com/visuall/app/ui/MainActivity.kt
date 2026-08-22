package com.visuall.app.ui

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.navigation.fragment.NavHostFragment
import com.visuall.app.R
import com.visuall.app.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding

    private val permissions = mutableListOf(
        Manifest.permission.CAMERA,
        Manifest.permission.RECORD_AUDIO
    ).apply {
        if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P)
            add(Manifest.permission.WRITE_EXTERNAL_STORAGE)
    }

    private val permLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { results ->
        if (results[Manifest.permission.CAMERA] != true) showPermissionDialog()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        // Nada de resetToCamera() aqui: o activity_main.xml já declara
        // app:navGraph, então o gráfico é montado (e o CameraFragment criado)
        // durante o setContentView acima. Chamar setGraph de novo montava o
        // gráfico uma segunda vez e criava um SEGUNDO CameraFragment, com as
        // duas instâncias disputando a câmera.
        checkPermissions()
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        resetToCamera()
    }

    // Só no relançamento pelo ícone (onNewIntent): volta pra câmera desfazendo
    // a pilha, em vez de reconstruir o gráfico do zero.
    private fun resetToCamera() {
        val navHost = supportFragmentManager
            .findFragmentById(R.id.nav_host_fragment) as? NavHostFragment ?: return
        runCatching { navHost.navController.popBackStack(R.id.nav_camera, false) }
    }

    private fun checkPermissions() {
        val needed = permissions.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }
        if (needed.isNotEmpty()) permLauncher.launch(needed.toTypedArray())
    }

    private fun showPermissionDialog() {
        AlertDialog.Builder(this)
            .setTitle(getString(R.string.perm_camera_title))
            .setMessage(getString(R.string.perm_camera_msg))
            .setPositiveButton(getString(R.string.perm_grant)) { _, _ -> checkPermissions() }
            .setNegativeButton("Cancelar", null)
            .show()
    }
}
