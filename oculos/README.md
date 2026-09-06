# Óculos VisuAll — a câmera que vai no rosto

O VisuAll traduz Libras. Hoje ele usa a câmera do celular, o que obriga quem
quer entender a **apontar o celular para a pessoa e olhar para a tela** — e
quem olha para a tela não está olhando para quem sinaliza. Em Libras isso custa
caro: expressão facial é gramática, é ela que diz se a frase é pergunta ou
negação. Quem olha para o celular está perdendo metade da língua.

Esta pasta é a resposta para isso: uma câmera minúscula presa na armação de um
óculos, mandando imagem por Wi-Fi para o celular no bolso.

> **Os óculos são os olhos. O celular é o cérebro. O fone é a voz que só você
> ouve.** Você olha para a pessoa, e a tradução chega no seu ouvido.

---

## Sumário

1. [O que já funciona](#1-o-que-já-funciona)
2. [O que você precisa ter](#2-o-que-você-precisa-ter)
3. [Parte 1 — pegar o código](#3-parte-1--pegar-o-código)
4. [Parte 2 — o dublê dos óculos (sem placa nenhuma)](#4-parte-2--o-dublê-dos-óculos-sem-placa-nenhuma)
5. [Parte 3 — instalar o app no celular](#5-parte-3--instalar-o-app-no-celular)
6. [Parte 4 — ver a webcam do PC no celular](#6-parte-4--ver-a-webcam-do-pc-no-celular)
7. [Parte 5 — o modo de bolso](#7-parte-5--o-modo-de-bolso)
8. [Parte 6 — o firmware da placa](#8-parte-6--o-firmware-da-placa)
9. [Quando der errado](#9-quando-der-errado)
10. [Como conferir cada elo da corrente](#10-como-conferir-cada-elo-da-corrente)
11. [Decisões de projeto](#11-decisões-de-projeto)

---

## 1. O que já funciona

| # | Etapa | Precisa da placa? | Estado |
|---|---|---|---|
| 1 | Dublê da câmera rodando no PC | não | **pronto** |
| 2 | O app recebendo imagem pela rede | não | **pronto** |
| 3 | Forçar o Wi-Fi sem internet | não | **pronto** |
| 4 | Firmware compilando | não (só gravar precisa) | **pronto** |
| 5 | Entrar na rede dos óculos pelo app | não | **pronto** |
| 6 | Modo de bolso (tela travada) | não | **pronto** |
| — | Gravar e ligar a placa de verdade | **sim** | esperando o hardware |

Tudo que está marcado como pronto foi testado num celular de verdade, não só
compilado. O que falta é físico.

---

## 2. O que você precisa ter

### Para acompanhar tudo que já está pronto (sem a placa)

- Um **PC com Windows** (o projeto foi feito nele; em Linux ou Mac quase tudo
  funciona igual, mudam os comandos de firewall).
- Uma **webcam** — a do notebook serve. Sem webcam também dá: o dublê tem um
  gerador de imagem embutido, veja a Parte 2.
- Um **celular Android** com a mesma rede Wi-Fi do PC.
- Um **cabo USB** para ligar o celular no PC.

### Programas

| Programa | Para quê | Onde |
|---|---|---|
| **Git** | baixar o código | https://git-scm.com |
| **Python 3.9 ou mais novo** | rodar o dublê da câmera | https://python.org |
| **Android Studio** | compilar o app e falar com o celular | https://developer.android.com/studio |
| **Arduino IDE 2.x** | compilar o firmware da placa | https://arduino.cc/en/software |

O Android Studio já traz junto o `adb` (o programa que conversa com o celular)
e um Java próprio. A Arduino IDE já traz o `arduino-cli`. Você não precisa
instalar nada disso separado.

### O hardware dos óculos (quando chegar)

| Peça | Detalhe |
|---|---|
| ESP32-S3-CAM **N16R8** | 16 MB de memória de programa, 8 MB de PSRAM |
| Câmera **OV5640** | a que vem com a placa |
| Bateria LiPo 1000 mAh | modelo 503450 |
| Carregador **TP4056** | o que tem proteção, não o simples |
| Elevador de tensão **MT3608** | para tirar 5 V da bateria |
| Capacitor **330 µF / 50 V** | três deles |
| Cabo flat AWM20798 | 24 vias, passo 0,5 mm |

> ⚠️ **Duas coisas antes de ligar a placa pela primeira vez:**
>
> 1. **Calibre o MT3608 para exatamente 5,0 V com um multímetro, SEM o ESP32
>    conectado.** Ele sai de fábrica desregulado e pode entregar 20 V.
> 2. **O capacitor de 330 µF não é opcional.** Quando o Wi-Fi transmite, a
>    corrente dá um pico de milissegundos. Sem o capacitor para segurar esse
>    pico, a tensão cai, a placa reinicia sozinha, e parece defeito de software.

---

## 3. Parte 1 — pegar o código

Abra o **PowerShell** (tecla Windows, digite "powershell", Enter) e:

```powershell
git clone https://github.com/Juanduarte050508/VisuAll.git
cd VisuAll
git checkout esp32-camera-remota
```

A última linha é importante: este trabalho vive num **ramo** separado chamado
`esp32-camera-remota`, não no principal. Se você pular essa linha, a pasta
`oculos/` nem vai existir.

Para conferir que deu certo:

```powershell
git branch --show-current
```

Tem que responder `esp32-camera-remota`.

---

## 4. Parte 2 — o dublê dos óculos (sem placa nenhuma)

### Por que existe

O trabalho difícil deste projeto não é o firmware da placa: é o app aprender a
receber imagem **pela rede** em vez da câmera do celular. Esperar a placa chegar
para só então começar deixaria o app parado.

O `mock_esp32_cam.py` resolve isso. Ele pega a webcam do PC e serve a imagem
**exatamente no mesmo formato** que o firmware vai servir — mesmo protocolo,
mesma resolução, mesmos cabeçalhos. Do ponto de vista do app, é indistinguível.
Quando a placa chegar, muda só o endereço.

Chamamos ele de "dublê" porque é isso mesmo: um dublê faz a cena perigosa no
lugar do ator, e ninguém na plateia percebe.

### 2.1 — Instalar o que ele precisa

```powershell
pip install opencv-python numpy
```

Se o `pip` não for reconhecido, use o Python pelo caminho completo:

```powershell
py -m pip install opencv-python numpy
```

### 2.2 — Rodar

Da pasta `VisuAll`:

```powershell
python oculos\mock_esp32_cam.py
```

A saída certa é assim:

```
abrindo a webcam...
Mock ESP32-CAM  --  fonte: webcam
  320x240 a 15 quadros/s, JPEG qualidade 88

  no proprio PC:  http://localhost:8080/
  no celular:     http://192.168.15.10:8080/

Ctrl+C pra parar.
```

**Anote o endereço que aparece depois de "no celular"** — você vai usar ele no
app. O seu vai ser diferente do exemplo. Se aparecer mais de um, o que costuma
funcionar é o que começa com `192.168.`.

Deixe essa janela aberta. Fechando ela, o dublê para.

**Não tem webcam?** Use:

```powershell
python oculos\mock_esp32_cam.py --sintetico
```

Ele gera uma imagem colorida com uma barra andando e um contador. Não serve para
testar reconhecimento — não tem mão nenhuma ali —, mas serve para testar **o
cano**: se o app recebe, decodifica e mostra essa imagem, vai receber as da
placa do mesmo jeito. O contador na tela deixa óbvio se o vídeo travou.

### 2.3 — Conferir no navegador do PC

Abra `http://localhost:8080/` no navegador. Você deve ver a imagem da webcam.

Se ver, **o dublê está bom** e qualquer problema daqui para frente é de rede ou
do app. Já é uma informação valiosa.

### 2.4 — Liberar o firewall do Windows

O Windows bloqueia conexões vindas de fora por padrão, então o celular não
alcança o PC até você liberar. Abra o PowerShell **como administrador** (clique
com o botão direito no ícone → "Executar como administrador") e rode:

```powershell
New-NetFirewallRule -DisplayName "VisuAll mock ESP32" -Direction Inbound -Protocol TCP -LocalPort 8080 -Action Allow
```

Isso é uma vez só. Depois nunca mais.

### 2.5 — Conferir no celular

No navegador do **celular**, abra o endereço que o dublê mostrou (aquele com
`192.168.`). Se a imagem da webcam aparecer, a rede está boa e você pode ir para
a próxima parte.

Se não aparecer, veja a seção [Quando der errado](#9-quando-der-errado).

---

## 5. Parte 3 — instalar o app no celular

### 3.1 — Preparar o celular

Isso é padrão de Android, não é coisa nossa:

1. **Configurações → Sobre o telefone → Informações de software**
2. Toque **sete vezes** em "Número da versão". Vai aparecer "você agora é um
   desenvolvedor".
3. Volte para **Configurações → Opções do desenvolvedor**
4. Ligue **Depuração USB**
5. Ligue o celular no PC pelo cabo. Vai aparecer uma caixa no celular pedindo
   permissão — aceite, e marque "sempre permitir".

Para conferir, no PowerShell:

```powershell
adb devices
```

Se o `adb` não for reconhecido, ele está dentro do Android Studio:

```powershell
$adb = "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe"
& $adb devices
```

A resposta certa tem uma linha com o número do seu aparelho e a palavra
`device`. Se disser `unauthorized`, a caixa de permissão não foi aceita no
celular.

### 3.2 — Dizer onde está o SDK do Android

**Este passo é obrigatório num clone novo, e é onde quase todo mundo tropeça.**

O arquivo que aponta para o SDK (`mobile/local.properties`) **não vai para o
git** de propósito: o caminho é diferente em cada máquina. Sem ele o gradle
para com:

```
SDK location not found. Define a valid SDK location with an ANDROID_HOME
environment variable or by setting the sdk.dir path in your project's
local properties file
```

Há dois jeitos de resolver. **O mais fácil:** abra a pasta `mobile` no Android
Studio uma vez e espere ele terminar de carregar. Ele cria o arquivo sozinho, e
depois disso a linha de comando funciona para sempre.

**Pelo terminal**, se preferir:

```powershell
$env:ANDROID_HOME = "$env:LOCALAPPDATA\Android\Sdk"
```

### 3.3 — Compilar e instalar

```powershell
cd mobile
.\gradlew.bat :app:assembleDebug
```

A primeira vez demora bastante (ele baixa as ferramentas). Depois é rápido.

Se der **"JAVA_HOME is not set"**, aponte para o Java que vem com o Android
Studio:

```powershell
$env:JAVA_HOME = "C:\Program Files\Android\Android Studio\jbr"
```

E rode de novo.

> As duas linhas de `$env:` valem só para a janela do PowerShell que está
> aberta. Fechou e abriu outra, precisa repetir. (Ou abra no Android Studio,
> que não precisa de nenhuma delas.)

Depois de compilar:

```powershell
& $adb install -r app\build\outputs\apk\debug\app-debug.apk
```

Tem que responder `Success`. O APK do modo de depuração tem uns 316 MB — é
grande porque carrega os modelos de reconhecimento dentro dele.

> Também dá para fazer tudo isso abrindo a pasta `mobile` no Android Studio e
> clicando no botão de play. É a mesma coisa.

---

## 6. Parte 4 — ver a webcam do PC no celular

Com o dublê rodando e o app instalado:

1. Abra o **VisuAll** no celular
2. Entre no **modo Libras**
3. No alto da tela tem um **ícone de olho** (fica entre "Modo Libras" e o
   relógio de histórico). Toque nele. É o olho porque é ele que escolhe de
   onde vem a imagem: a câmera do celular ou a que está nos óculos.
4. Como ainda não há endereço salvo, abre uma caixa de texto. Digite o endereço
   que o dublê mostrou, **com `/stream` no fim**:

   ```
   http://192.168.15.10:8080/stream
   ```

   (troque pelo IP que apareceu no seu PC)

5. Toque em **Conectar**

A imagem da webcam do PC deve aparecer no celular, com o reconhecimento rodando
em cima dela. Faça um sinal na frente da webcam e a letra sai no celular.

Para voltar à câmera do celular, toque no mesmo ícone de olho.

> **Para mudar o endereço depois**, segure o ícone de olho (toque longo).
> O endereço fica no toque longo porque se digita uma vez e a troca acontece
> toda hora.

### O que a caixa de endereço decide

O app usa o endereço para escolher entre dois comportamentos bem diferentes:

- **`http://192.168.4.1/stream`** (o endereço da placa) → o app pede ao Android
  para **entrar na rede dos óculos sozinho**. O Android mostra uma caixa
  perguntando, você confirma, e pronto: a conexão vale só para o VisuAll, e o
  celular continua na rede de casa (ou nos dados móveis) para todo o resto.
- **qualquer outro endereço** → o app usa o Wi-Fi em que o celular já está. É o
  caso do dublê rodando no PC.

---

## 7. Parte 5 — o modo de bolso

No uso de verdade o celular vai para o bolso e os óculos continuam vendo. Isso
cria dois problemas:

- A tela **precisa continuar acesa**. Se ela apagar, o Android para o app: o
  reconhecimento morre e a conexão com os óculos cai junto.
- Mas uma tela acesa dentro do bolso aceita toque no tecido e gasta bateria.

Por isso existe o **cadeado**, que aparece no alto da tela **só quando o modo
óculos está ligado**. Tocando nele:

- A tela fica **preta de verdade** (numa tela OLED isso é pixel desligado) e o
  brilho vai para o mínimo
- Toques são **engolidos** — nem os botões respondem
- O **gesto de voltar é ignorado**, porque ele dispara sozinho contra o tecido
  do bolso e tiraria você do modo óculos no meio de uma conversa
- O app **para de desenhar** a imagem e as linhas, o que economiza uma cópia de
  ~300 KB por quadro, treze vezes por segundo
- **O reconhecimento continua rodando normalmente**

Para desbloquear: **arraste o dedo para cima**, de baixo para cima, uns 20%
da altura da tela. A tela mostra três setas grandes apontando para cima, e é
esse o gesto. Um toque não desbloqueia, de propósito — no bolso, toque acontece
sozinho o tempo todo; um movimento longo e deliberado não.

---

## 8. Parte 6 — o firmware da placa

Está tudo em [`firmware/`](firmware/), com README próprio. Resumo:

```powershell
cd oculos\firmware
.\compilar.ps1                  # só compila (não precisa da placa)
.\compilar.ps1 -Porta COM5      # compila e grava (precisa da placa)
```

Precisa do core do ESP32 instalado uma vez (~1 GB):

```powershell
arduino-cli core install esp32:esp32 --additional-urls https://espressif.github.io/arduino-esp32/package_esp32_index.json
```

Se o `arduino-cli` não for reconhecido, ele vem dentro da Arduino IDE:

```powershell
$cli = "$env:LOCALAPPDATA\Programs\Arduino IDE\resources\app\lib\backend\resources\arduino-cli.exe"
```

O `compilar.ps1` acha ele sozinho.

Quando ligada, a placa cria a rede:

```
rede...: VisuAll-Oculos     senha: visuall2026
no app.: http://192.168.4.1/stream
```

O [README do firmware](firmware/README.md) explica em detalhe a opção de
compilação que mais dá dor de cabeça (`PSRAM=opi`) e como descobrir se o
modelo de pinos da câmera é o certo para a sua placa.

---

## 9. Quando der errado

Estes são problemas que aconteceram **de verdade** durante o desenvolvimento,
com o sintoma exato e a causa.

### "a fonte nao entrega imagem"

```
[ WARN ] VIDEOIO(DSHOW): backend is generally available but can't be used to capture by index
abri 0 mas ela nao entrega imagem -- quase sempre e outro programa segurando a webcam
```

**Causa:** outro programa está com a webcam. Quase sempre é **outra cópia do
próprio dublê** ainda rodando de antes. Também pode ser Teams, Meet, ou o app
Câmera do Windows.

**Solução:** feche o outro programa. Para achar cópias esquecidas do dublê:

```powershell
Get-Process python* | Select-Object Id, StartTime
Get-NetTCPConnection -LocalPort 8080 | Select-Object State, OwningProcess
```

Se houver um processo segurando a porta 8080, é ele. `Stop-Process -Id <numero>
-Force`.

> Isso custou horas uma vez. Um dublê antigo continuou segurando a porta 8080
> depois de "parado", e **todo teste seguinte foi respondido pelo processo
> velho**. O código era corrigido, o sintoma continuava idêntico, porque o
> código corrigido nunca chegava a atender ninguém.

### O dublê parece travado ao abrir

Se você viu isso, é uma versão antiga. O dublê agora imprime `abrindo a
webcam...` antes de tentar.

**Causa:** no Windows, o modo padrão do OpenCV para abrir a webcam levou **98
segundos** nesta máquina. O modo DirectShow levou **1,5**. O código já usa o
DirectShow; se ele recusar, o dublê avisa que vai tentar o modo lento.

### O celular abre no navegador mas o app não

O app está pedindo o endereço errado. Confira se tem **`/stream` no fim**. Sem
ele, o app recebe a página HTML de teste em vez do vídeo, e avisa:

> Esse endereco responde, mas nao e video. Falta o /stream no fim?

### Tela preta no app, sem erro nenhum

Se isso acontecer, o problema é do app para dentro — não da rede. Use a seção
[Como conferir cada elo](#10-como-conferir-cada-elo-da-corrente) para provar
que o vídeo está chegando no celular.

Já aconteceu duas vezes, por causas totalmente diferentes:

1. **O Android bloqueando HTTP sem criptografia.** Desde o Android 9 ele
   recusa conexões sem criptografia por padrão. A conexão morria *dentro* do
   aparelho, então do lado do PC não chegava nada e a suspeita ia para o
   firewall. Resolvido em `mobile/app/src/main/res/xml/network_security_config.xml`.
2. **Um detalhe de layout.** A imagem dos óculos tem tamanho amarrado às bordas
   da câmera do celular, e esconder a câmera com `GONE` faz ela virar um ponto
   no Android — a imagem encolhia para 0×0. O vídeo chegava, a conexão estava
   aberta, o reconhecimento rodava, e a imagem era desenhada num retângulo de
   tamanho zero.

### O celular volta para os dados móveis sozinho

É o comportamento normal do Android com uma rede **sem internet** — que é
exatamente o que os óculos criam. O app já resolve isso (veja
`RedeDosOculos.kt`), mas se você mexer nessa parte, a seção seguinte explica
como testar.

### Os botões não respondem durante testes automatizados

A tela do celular bloqueou por inatividade. Enquanto testa:

```powershell
& $adb shell "svc power stayon usb"     # mantém acesa no cabo
& $adb shell "svc power stayon false"   # devolve ao normal
```

---

## 10. Como conferir cada elo da corrente

Quando a imagem não aparece, vale **medir em vez de adivinhar**. O celular tem
um programinha chamado `nc` que fala HTTP na unha, então dá para descobrir
exatamente até onde a coisa chega.

```powershell
# 1. o celular alcança o PC?
#    (ping NÃO serve: o Windows bloqueia ping por padrão, e isso não quer
#     dizer que a porta 8080 esteja bloqueada)
& $adb shell "printf 'GET / HTTP/1.0\r\n\r\n' | timeout 6 nc 192.168.15.10 8080 | head -c 220"

# 2. o /stream entrega bytes de verdade?
& $adb shell "printf 'GET /stream HTTP/1.0\r\n\r\n' | timeout 4 nc 192.168.15.10 8080 | wc -c"

# 3. o app está mesmo conectado?
#    (1F90 é 8080 escrito em hexadecimal; estado 01 quer dizer "conectado")
& $adb shell "cat /proc/net/tcp /proc/net/tcp6 | grep 1F90"

# 4. por onde o celular está tentando falar com o PC?
& $adb shell "ip route get 192.168.15.10"
```

Como ler o resultado:

- **(1) devolve HTML e (2) devolve um número grande** → a rede está boa. Se a
  tela continua preta, o problema é do app para dentro.
- **(1) devolve vazio** → o celular não alcança o PC. Firewall, ou os dois não
  estão na mesma rede.
- **(4) responde `dev wlan0`** → está indo pelo Wi-Fi, certo. Se responder
  `dev rmnet0`, está indo pelos **dados móveis** e nunca vai chegar.

---

## 11. Testar o Wi-Fi sem internet (sem derrubar a casa)

Os óculos criam uma rede que **não tem internet nenhuma para oferecer** — não há
de onde tirar. O Android percebe isso, decide que a rede não presta e volta
sozinho para os dados móveis. O Wi-Fi continua aparecendo conectado na tela do
celular, mas o pedido do app sai pela operadora, onde `192.168.4.1` não existe.

Para testar isso parecia ser preciso tirar o cabo de internet do roteador — o
que deixa a casa inteira offline. **Não é preciso.**

O Android decide se uma rede tem internet tentando alcançar um endereço de teste
assim que conecta. Apontando esse teste para um endereço que não existe, o
celular conclui que aquele Wi-Fi não presta — **só naquele aparelho**, sem tocar
no roteador. `192.0.2.1` é um endereço reservado para documentação: não existe
na internet, então nada pode responder.

```powershell
# fazer o celular achar que o Wi-Fi não tem internet
& $adb shell "settings put global captive_portal_http_url http://192.0.2.1/generate_204"
& $adb shell "settings put global captive_portal_https_url https://192.0.2.1/generate_204"
& $adb shell "settings put global captive_portal_fallback_url http://192.0.2.1/gen_204"
& $adb shell "settings put global captive_portal_other_fallback_urls http://192.0.2.1/gen_204"
& $adb shell "cmd wifi set-wifi-enabled disabled"
& $adb shell "cmd wifi set-wifi-enabled enabled"

# DESFAZER (não esqueça)
& $adb shell "settings delete global captive_portal_http_url"
& $adb shell "settings delete global captive_portal_https_url"
& $adb shell "settings delete global captive_portal_fallback_url"
& $adb shell "settings delete global captive_portal_other_fallback_urls"
& $adb shell "cmd wifi set-wifi-enabled disabled"
& $adb shell "cmd wifi set-wifi-enabled enabled"
& $adb shell "cmd wifi start-scan"      # às vezes precisa disto para reconectar
```

A internet continua existindo; o celular é que passa a acreditar que não. Para o
mecanismo em teste dá no mesmo: o que faz o Android fugir da rede é ela perder a
marca `VALIDATED`, e ela perde do mesmo jeito nos dois casos.

**O que se vê com o desvio aplicado:**

```
Active default network: 158                  ← 158 é MOBILE[LTE], não o Wi-Fi
192.168.15.10 via 100.75.48.1 dev rmnet0     ← rmnet0 são os dados móveis
I RedeDosOculos: usando a rede 161 para o stream   ← 161 é o Wi-Fi
```

O celular manda tudo pela operadora, e o app manda o vídeo pelo Wi-Fi mesmo
assim. **Isso é a etapa funcionando.**

---

## 12. Decisões de projeto

Para quem for ler ou mexer no código.

### Onde fica cada coisa

```
oculos/
  mock_esp32_cam.py            o dublê da câmera
  firmware/
    oculos_camera/*.ino        o que roda na placa
    compilar.ps1               compila e grava

mobile/app/src/main/java/com/visuall/app/oculos/
  MjpegReader.kt               entende o formato do vídeo
  MjpegClient.kt               conecta e reconecta sozinho
  UltimoQuadro.kt              caixa de um quadro só
  NetworkStreamSource.kt       junta tudo e entrega Bitmap
  RedeDosOculos.kt             força o Wi-Fi certo
  EnderecoDosOculos.kt         o que se sabe da placa
  MensagemDeErro.kt            traduz falha para português
```

### Descartar quadro em vez de enfileirar

Os óculos mandam imagem no ritmo deles e o reconhecimento demora o que demora.
Se o que sobra fosse enfileirado, num momento de lentidão a fila cresceria e a
pessoa veria o gesto de dois segundos atrás — **pior que perder quadro, porque
esse atraso nunca se recupera sozinho**. Quadro velho de câmera ao vivo não vale
nada. É a mesma escolha que o CameraX faz com `STRATEGY_KEEP_ONLY_LATEST`.

### Content-Length obrigatório

O leitor recusa uma parte do vídeo que venha sem `Content-Length`. Sem ele só
restaria varrer os bytes procurando a próxima marca de separação — e uma imagem
pode conter essa marca por acaso. Melhor falhar dizendo o que houve do que
decodificar um quadro cortado em silêncio.

### Nem espelho, nem rotação

Nos óculos prontos quem os usa é **o ouvinte**, e a câmera aponta para quem
sinaliza: ela vê outra pessoa de frente, que é a mesma geometria da câmera
traseira do celular. O app não espelha essa.

Testando com uma webcam apontada para **você**, a imagem parece invertida — mas
está certa, e o efeito some no dia em que a câmera apontar para outra pessoa.
Não existe opção de espelhar de propósito: seria uma chave que alguém deixaria
ligada sem querer, e qualquer resultado de reconhecimento medido com ela ligada
não valeria nada.

### Testes

```powershell
cd mobile
.\gradlew.bat :app:testDebugUnitTest
```

São 183, e os desta parte são 54. Todos rodam sem celular e sem placa.

Um hábito que vale copiar: **todo teste importante foi conferido quebrando o
código de propósito**. Um teste que passa não prova nada até você ver ele
falhar pelo motivo certo. Foi assim que se descobriu que uma das confirmações
"tinha passado" sem nunca ter rodado — a tela do celular havia bloqueado e o
toque foi para o vazio.
