# Óculos — ESP32-S3-CAM

Câmera remota para o VisuAll: em vez de o app usar a câmera do celular, ele
recebe a imagem por Wi-Fi de um ESP32-S3-CAM montado numa armação de óculos.

Hardware fechado: **ESP32-S3-CAM N16R8 + OV5640**, alimentado por LiPo 1000mAh
(503450) via TP4056 com proteção e step-up MT3608, com capacitor de 330µF/50V
no barramento de 5V.

> Antes de ligar a placa: calibrar o MT3608 para 5,0 V com multímetro, **sem** o
> ESP32 conectado. E o capacitor de 330µF não é opcional — sem ele o pico de
> corrente da transmissão Wi-Fi derruba a placa por subtensão (brownout).

## O que dá pra fazer antes da placa chegar

O gargalo do projeto não é o firmware, é o app aprender a receber imagem pela
rede. Esse trabalho não precisa da placa: precisa de **algo que fale o mesmo
protocolo**. É o que o mock faz.

| # | Etapa | Depende da placa? |
|---|---|---|
| 1 | Mock MJPEG no PC | não — **pronto** |
| 2 | `NetworkStreamSource` no app | não — **pronto** |
| 3 | Forçar o Wi-Fi sem internet | não — **pronto**, ver `RedeDosOculos` |
| 4 | Compilar o firmware | não — **pronto**, ver `firmware/` |
| 5 | ~~Suportes impressos em 3D~~ | descartado: a montagem vai ser feita à mão, sem impressão 3D |

O Wokwi não ajuda aqui: ele não simula câmera (nem OV2640 nem OV5640) e o modo
AP simulado não aceita conexão de um celular real — ou seja, não valida
justamente as duas coisas que importam.

## 1. Mock — o dublê dos óculos

`mock_esp32_cam.py` serve um stream MJPEG em 320×240 com exatamente o mesmo
formato multipart do firmware CameraWebServer. Para o app, é indistinguível dos
óculos; quando a placa chegar, muda só o endereço.

```
pip install opencv-python           # já instalado nesta máquina
python mock_esp32_cam.py            # webcam, se houver
python mock_esp32_cam.py video.mp4  # um vídeo em loop
python mock_esp32_cam.py --sintetico
```

Ele imprime os endereços a tentar. Abra `http://<IP>:8080/` no navegador —
primeiro no próprio PC, depois no celular.

**Se abrir no PC e não no celular**, quase sempre é o Firewall do Windows.
Num PowerShell **como administrador**:

```powershell
New-NetFirewallRule -DisplayName "VisuAll mock ESP32" -Direction Inbound `
  -Protocol TCP -LocalPort 8080 -Action Allow
```

Sem webcam, use `--sintetico`: os quadros gerados têm uma barra que atravessa a
tela e um contador. Não servem para testar reconhecimento — não há mão nenhuma
neles — mas provam que o cano funciona: se o app exibe esses quadros, ele vai
exibir os do ESP32 do mesmo jeito. Se a barra congela, o stream travou.

## 2. Por onde a imagem entra no app

`LibrasAnalyzer.analyze()` converte o quadro da câmera em `Bitmap` na primeira
linha e **todo o resto do pipeline trabalha só com o Bitmap**. A fonte de rede
não precisa tocar em nada do reconhecimento: basta entregar `Bitmap` no mesmo
ponto em que o CameraX entrega hoje.

## Duas armadilhas que custaram caro

As duas se manifestam do mesmo jeito — **tela preta, nenhum erro** — e as duas
estão longe de onde o sintoma aparece. Ficam registradas porque nenhuma delas
se descobre lendo o código.

**O Android bloqueia HTTP sem criptografia.** Desde o Android 9 (API 28), e o
app é targetSdk 34. A conexão morre dentro do aparelho, antes de virar pacote:
do lado do PC não chega nada, e a tentação é caçar firewall e cabo. Resolvido
em `mobile/app/src/main/res/xml/network_security_config.xml`, ligado pelo
`android:networkSecurityConfig` no manifesto. Um ESP32 não faz TLS a 15
quadros/s, então HTTP não é escolha nossa — o comentário do arquivo explica por
que isso não é preguiça.

**`isVisible = false` é `View.GONE`, e GONE colapsa no ConstraintLayout.** A
`iv_oculos` (a imagem dos óculos) e o desenho dos landmarks têm largura e
altura `0dp` presas às quatro bordas da `preview_view`. Ao ligar os óculos, a
preview era escondida com `isVisible = false` — que é GONE, não INVISIBLE — e
no ConstraintLayout um view GONE vira um ponto: quem está preso a ele encolhe
junto. Os dois viravam 0×0.

O sintoma foi cruel: o stream chegava, a conexão TCP estava aberta, o
reconhecimento rodava a 13 quadros/s (o mock manda 15), e a tela ficava preta.
Não havia erro porque não havia erro — a imagem era desenhada num retângulo de
tamanho zero. O que fechou o diagnóstico foi olhar `/proc/net/tcp` no celular e
achar a conexão ESTABLISHED: com dado chegando e nada na tela, o problema só
podia ser de layout.

## Como conferir cada elo, do celular

Quando não aparecer imagem, vale medir em vez de adivinhar. O celular tem
`nc`, então dá pra falar HTTP na mão e ver até onde se chega:

```bash
# o celular alcança o PC? (ping não serve: o Windows bloqueia ICMP)
adb shell "printf 'GET / HTTP/1.0\r\n\r\n' | timeout 6 nc 192.168.15.10 8080 | head -c 220"

# o /stream entrega bytes de verdade?
adb shell "printf 'GET /stream HTTP/1.0\r\n\r\n' | timeout 4 nc 192.168.15.10 8080 | wc -c"

# o app está mesmo conectado? (1F90 = porta 8080; estado 01 = ESTABLISHED)
adb shell "cat /proc/net/tcp /proc/net/tcp6 | grep 1F90"
```

Se o `nc` traz bytes e a tela continua preta, o problema é do app pra dentro —
não da rede.
