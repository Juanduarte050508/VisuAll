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
| 2 | `NetworkStreamSource` no app | não |
| 3 | Forçar o Wi-Fi sem internet (`bindProcessToNetwork`) | não |
| 4 | Compilar o firmware | não (só gravar precisa) |
| 5 | Suportes impressos em 3D | não |

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
