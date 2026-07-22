# VisuAll

Aplicativo Android de acessibilidade para reconhecimento experimental de sinais de Libras usando câmera, MediaPipe, ONNX Runtime e TensorFlow Lite.

## Requisitos

- Android Studio recente
- JDK 17
- Android SDK 34
- Celular ou emulador com Android 8.0 ou superior

## Como usar

1. Abra esta pasta no Android Studio.
2. Aguarde a sincronizacao do Gradle.
3. Selecione a configuracao `app`.
4. Conecte um celular com depuracao USB ou inicie um emulador.
5. Clique em Run.

Tambem da para gerar o APK debug pelo terminal:

```powershell
.\gradlew.bat assembleDebug
```

O APK fica em:

```text
app/build/outputs/apk/debug/app-debug.apk
```

Para treinar ou substituir modelos, leia [TREINAMENTO_MODELOS.md](TREINAMENTO_MODELOS.md).

## Estrutura

```text
app/                 codigo Android
app/src/main/assets/ modelos e labels usados pelo app
gradle/              Gradle Wrapper
build.gradle         configuracao raiz
settings.gradle      modulos do projeto
```

## Observacao

O projeto e um prototipo academico. A qualidade do reconhecimento depende de iluminacao, enquadramento, camera do aparelho e quantidade de exemplos usados nos modelos.
