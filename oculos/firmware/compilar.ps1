# Compila (e opcionalmente grava) o firmware dos oculos.
#
#   .\compilar.ps1                 so compila
#   .\compilar.ps1 -Porta COM5     compila e grava na placa
#   .\compilar.ps1 -Limpo          recompila do zero
#
# Existe por causa da linha de opcoes da placa. Ela e comprida e uma das
# opcoes -- PSRAM=opi -- e o erro classico desta placa: a N16R8 tem PSRAM
# octal, e compilar com o padrao (desligada) gera um firmware que SOBE, mas
# que so consegue um buffer de imagem na RAM interna. O sintoma nao e um erro
# de compilacao: e o stream engasgando, que manda procurar na antena, na
# bateria, no cabo -- em tudo, menos aqui.
#
# O firmware avisa no monitor serial se subir sem PSRAM.

param(
    [string]$Porta = "",
    [switch]$Limpo
)

$ErrorActionPreference = "Stop"

$sketch = Join-Path $PSScriptRoot "oculos_camera"

# A IDE 2.x traz o arduino-cli embutido; usar o dela evita pedir mais uma
# instalacao a quem so quer compilar.
$cli = Get-Command arduino-cli -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source
if (-not $cli) {
    $embutido = "$env:LOCALAPPDATA\Programs\Arduino IDE\resources\app\lib\backend\resources\arduino-cli.exe"
    if (Test-Path $embutido) { $cli = $embutido }
}
if (-not $cli) {
    Write-Error "arduino-cli nao encontrado. Instale a Arduino IDE 2.x ou o arduino-cli."
}

# ESP32-S3-CAM N16R8: 16MB de flash, 8MB de PSRAM octal.
$fqbn = "esp32:esp32:esp32s3:" + (@(
    "PSRAM=opi",                    # octal: e o que a N16R8 tem
    "FlashSize=16M",
    "FlashMode=qio",
    "PartitionScheme=huge_app",     # 3MB pro app; camera + Wi-Fi nao cabem no padrao de 1,2MB
    "CDCOnBoot=cdc"                 # serial pela USB nativa da placa
) -join ",")

Write-Host "sketch: $sketch"
Write-Host "placa.: $fqbn"
Write-Host ""

$argumentos = @("compile", "--fqbn", $fqbn, "--warnings", "all")
if ($Limpo) { $argumentos += "--clean" }
$argumentos += $sketch

& $cli @argumentos
if ($LASTEXITCODE -ne 0) { Write-Error "falhou a compilacao" }

if ($Porta -ne "") {
    Write-Host ""
    Write-Host "gravando em $Porta..."
    & $cli upload --fqbn $fqbn --port $Porta $sketch
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "Se nao gravou: segure BOOT, toque em RESET, solte BOOT e rode de novo."
        Write-Error "falhou a gravacao"
    }
    Write-Host ""
    Write-Host "Pra ver o monitor serial:  $cli monitor -p $Porta -c baudrate=115200"
}
