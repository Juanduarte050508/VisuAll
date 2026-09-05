# Firmware dos óculos

O que roda na placa: cria a própria rede Wi-Fi e serve a câmera em MJPEG, no
mesmo formato do `oculos/mock_esp32_cam.py`. Do lado do app não muda nada além
do endereço.

```
rede....: VisuAll-Oculos     senha: visuall2026
no app..: http://192.168.4.1/stream
no nav..: http://192.168.4.1/
```

A rede é criada **pela placa** (SoftAP). Não depende de roteador, de internet,
nem de nada instalado no lugar — que é o requisito do projeto. O celular entra
nessa rede e fala direto com a placa. Do lado do app isso já está resolvido:
ver `RedeDosOculos`, que impede o Android de fugir de uma rede sem internet.

## Compilar

```powershell
.\compilar.ps1                  # só compila
.\compilar.ps1 -Porta COM5      # compila e grava
.\compilar.ps1 -Limpo           # recompila do zero
```

Usa o `arduino-cli` embutido na Arduino IDE 2.x, se houver. Precisa do core
`esp32:esp32` (instalado uma vez, ~1 GB):

```powershell
arduino-cli core install esp32:esp32 --additional-urls https://espressif.github.io/arduino-esp32/package_esp32_index.json
```

Estado atual: **compila limpo**, sem avisos. Ocupa 977 KB (31% do espaço de
programa) e 56 KB de RAM (17%).

### A opção que mais dá dor de cabeça

A linha de opções da placa está dentro do script justamente para não ser
digitada:

```
esp32:esp32:esp32s3:PSRAM=opi,FlashSize=16M,FlashMode=qio,PartitionScheme=huge_app,CDCOnBoot=cdc
```

**`PSRAM=opi`** é a que importa. A N16R8 tem PSRAM *octal*, e o padrão da IDE é
PSRAM desligada. Compilando com o padrão o firmware **sobe normalmente** — não
dá erro nenhum — mas a câmera fica com um único buffer na RAM interna e o
stream engasga. O sintoma manda procurar na antena, na bateria, no cabo; em
tudo menos na opção de compilação. Por isso o firmware imprime um aviso no
monitor serial se subir sem PSRAM.

`PartitionScheme=huge_app` também não é opcional: câmera + Wi-Fi não cabem na
partição padrão de 1,2 MB.

## O que só a placa pode confirmar

**Os pinos da câmera.** O `camera_pins.h` é cópia literal do arquivo do próprio
core do ESP32 — nenhum número foi digitado à mão —, mas *qual* dos modelos é a
sua placa só dá pra saber com ela ligada. O sketch usa
`CAMERA_MODEL_ESP32S3_EYE`, que é o mapeamento das ESP32-S3-CAM mais comuns
(mesmo pinout da Freenove ESP32-S3-WROOM).

Se estiver errado, o monitor serial diz na primeira linha:

```
esp_camera_init falhou: 0x105 (ESP_ERR_NOT_FOUND)
```

Aí é trocar o `#define CAMERA_MODEL_...` no topo do `.ino` por outro do
`camera_pins.h` — `CAMERA_MODEL_ESP32S3_CAM_LCD` e `CAMERA_MODEL_XIAO_ESP32S3`
são os próximos candidatos — e recompilar.

Quando acertar, o serial mostra qual sensor respondeu:

```
sensor detectado: PID 0x56  (OV5640=0x56, OV2640=0x26, OV3660=0x36)
```

Isso separa dois problemas que parecem o mesmo: pino errado (a câmera nem
responde) de sensor diferente do esperado (responde, com outro PID).

## Formato do stream

Estas linhas do `.ino` são um contrato com o `MjpegReader` do app:

```c
#define FRONTEIRA "123456789000000000000987654321"
static const char *TIPO_STREAM = "multipart/x-mixed-replace;boundary=" FRONTEIRA;
static const char *CABECALHO_PARTE =
    "--" FRONTEIRA "\r\nContent-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";
```

Dois detalhes que não são estilo:

- **A fronteira vem antes de cada quadro.** O `CameraWebServer` de fábrica manda
  depois; navegador tolera, mas é o contrário do que diz o formato multipart. O
  app aguenta os dois (há teste para isso), só que perde o primeiro quadro no
  formato de fábrica, porque não há fronteira antes dele para se ancorar.
- **O `Content-Length` é obrigatório.** Sem ele não há como saber onde o JPEG
  termina: os bytes da imagem podem conter qualquer coisa, inclusive algo
  parecido com a fronteira. O leitor recusa a parte em vez de decodificar um
  quadro cortado em silêncio.

## Gravar

USB, e no primeiro flash quase sempre é preciso entrar em modo boot: segure
**BOOT**, toque em **RESET**, solte **BOOT**.

Depois:

```powershell
arduino-cli monitor -p COM5 -c baudrate=115200
```

Se o monitor ficar mudo, troque `CDCOnBoot=cdc` por `CDCOnBoot=default` no
`compilar.ps1`: depende de a placa expor a USB nativa do S3 ou um conversor
serial.

O boot deve imprimir, em ordem: o motivo do último boot, o PID do sensor, e o
endereço da rede. Se aparecer

```
motivo do boot: 12  <<< BROWNOUT: falta corrente.
```

não é firmware: é o pico de corrente do Wi-Fi derrubando a alimentação. A
correção é o capacitor de 330µF e os 5,0 V do MT3608 — ver o README de cima —,
não desligar o detector de brownout.
