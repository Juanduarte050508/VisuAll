# VisuAll

Reconhecimento de **Libras** em tempo real: a câmera vê o sinal, o aparelho
entende a letra, e a frase aparece escrita — sem nuvem, sem internet, tudo
rodando no próprio dispositivo.

> **Os óculos são os olhos. O celular é o cérebro. O fone é a voz.**
>
> Quem ouve conversa olhando para a pessoa, e não para uma tela.

---

## O mapa do repositório

São **três** frentes de trabalho, cada uma com o seu próprio guia:

| Pasta | O que vive lá | Guia |
|---|---|---|
| `mobile/` | O aplicativo Android. É ele que reconhece, escreve e fala. Roda no aparelho com MediaPipe Tasks, ONNX Runtime e TFLite. | [mobile/README.md](mobile/README.md) |
| `computer/` | O lado PC: o reconhecedor de mesa em Python, a interface web, as ferramentas de treino, os datasets e os modelos. É aqui que os modelos nascem. | [computer/README.md](computer/README.md) |
| `oculos/` | A câmera dos óculos: um ESP32-S3-CAM preso na armação, o firmware dele, e um dublê que roda no PC para desenvolver **sem a placa**. | [oculos/README.md](oculos/README.md) |

O caminho de uma letra, de ponta a ponta:

```text
camera  ->  MediaPipe          ->  modelo         ->  frase na tela
            (pontos da mao)        (que letra e)      (e voz, se quiser)

a camera pode ser a do celular OU a dos oculos, por Wi-Fi
o modelo vem de computer/treino/ e e copiado pra dentro do app
```

---

## Por onde começar

### Só quero ver o app rodando

```powershell
cd mobile
.\gradlew.bat assembleDebug
```

O APK sai em `mobile\app\build\outputs\apk\debug\app-debug.apk`.

> **Na primeira vez, abra a pasta `mobile` no Android Studio e deixe o Gradle
> sincronizar.** O arquivo `local.properties`, que diz onde está o Android SDK,
> não vai no git — sem ele o build para com *"SDK location not found"*.

Passo a passo com telas: [mobile/README.md](mobile/README.md).

### Quero treinar ou trocar os modelos

```powershell
cd computer
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Use Python 3.11 quando der; o time de treino é testado de 3.10 a 3.12.

O reconhecedor de mesa:

```powershell
python linear\backend\app.py
python modular\app_backend_unificado.py
```

O treino é **duplo-clique**, sem digitar comando, nos `.bat` de
`computer\treino\`: `Gravar.bat`, `Reforcar.bat`, `Treinar.bat`,
`TreinarCorpo.bat`, `RestaurarModelo.bat`.

O que sai do treino é gravado direto em `mobile\app\src\main\assets\`, então
**recompile o app depois de treinar**.

Detalhes: [computer/README.md](computer/README.md) e
[computer/treino/README.md](computer/treino/README.md).

### Quero mexer nos óculos

Dá para fazer tudo **sem ter a placa**. O dublê roda no PC, usa a webcam e fala
o mesmo protocolo do firmware:

```powershell
python oculos\mock_esp32_cam.py
```

Ele mostra um endereço; é esse endereço que se digita no app, no botão do olho.

Passo a passo do zero, escrito para quem nunca mexeu com isso:
[oculos/README.md](oculos/README.md).

---

## O que fica na raiz, e por quê

| Arquivo | Para que serve |
|---|---|
| `README.md` | este mapa |
| `CHANGELOG.md` | o histórico das **constantes de calibração** do reconhecimento: cada limiar, o valor atual e por que ele é esse. Não é lista de novidades. |
| `.github/` | as duas esteiras de CI: **Android Build** e **Python Training** |
| `.gitignore` | as regras compartilhadas |
| `LICENSE` | a licença |
