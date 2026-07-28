# Changelog — Reconhecimento Libras (mobile)

Este arquivo existe porque as constantes de calibração em
`mobile/app/src/main/java/com/visuall/app/libras/LibrasAnalyzer.kt`
(limiares de confiança, margem, movimento, estabilidade) já foram
mexidas por dois desenvolvedores em paralelo sem essa história ficar
registrada em um só lugar — só espalhada em comentários de código e no
histórico do git, difícil de rastrear quando alguém precisa entender
"por que esse valor é esse". Cada entrada abaixo é uma decisão de
threshold, não uma lista de features.

Formato: **Constante(s)** — valor atual — decisão e por quê — status.

## Não lançado

- **Pipeline de treino de gestos corporais (novo)** — não existia NENHUM
  treino publicado neste repositório pro `body_model.tflite` (ele vinha de
  um pipeline externo, ver docstring de `linear/backend/app.py`). Criados
  `linear/backend/data_extraction/extract_from_videos_corpo.py` (extração
  de 225 features/frame — pose+mãos, normalização por ombros — reconstruída
  a partir do consumidor real `BodyGestureEngine.kt`, não do treino
  original) e `linear/backend/training/train_body_model.py` (LSTM Keras →
  TFLite com Select TF Ops, arquitetura nova, não é port de nada). Pasta
  `treinamento/` com `Capturar.bat`/`Treinar.bat` dá um jeito fácil de
  gravar clipes e rodar os dois. Pendente: validar com dados reais — ainda
  não há nenhum modelo de corpo treinado a partir desse pipeline.

- **Delegate GPU com fallback pra CPU** (`HandLandmarker`,
  `PoseLandmarker`, `FaceLandmarker`) — os três detectores MediaPipe
  tentam `Delegate.GPU` primeiro (bem mais rápido que CPU nesses
  modelos, quando o driver do aparelho suporta) e caem pra `Delegate.CPU`
  automaticamente se a criação do grafo falhar — mesma filosofia
  defensiva que Pose/Face já tinham pra inicialização em geral. Testado
  no emulador Pixel_4 do time (`mobile/tools/abrir-emulador-px4.bat`):
  a GPU dele rejeita o grafo (`GL_INVALID_ENUM`) e o fallback pra CPU
  funciona sem crash — então aqui nunca há ganho de velocidade, só a
  confirmação de que o fallback funciona. Pendente: validar em celular
  real se a GPU é aceita (ganho de velocidade esperado) e se o fallback
  também funciona lá caso não seja.

- **`MOVIMENTO_SUSTENTADO_MS=130`, `ESTAB_MIN_DINAMICO_MS=130`,
  `ESTAB_MIN_ESTATICO_MS=500`** — as três antigas gates em CONTAGEM DE
  FRAMES (`MOVIMENTO_SUSTENTADO_FRAMES=3`, `ESTAB_MIN_DINAMICO=3`,
  `ESTAB_MIN_ESTATICO=8`) viraram tempo em milissegundos. Um celular
  que analisa menos frames por segundo (aparelho mais fraco, ou os
  três detectores MediaPipe competindo pelo mesmo frame) fazia "3
  frames" corresponder a um tempo de parede bem maior que o pretendido
  — a janela real de um gesto dinâmico (~300-500ms) terminava antes da
  histerese liberar o classificador, perdendo H/J/K/X/Z justamente nos
  aparelhos mais lentos, que era exatamente o problema reportado.
  Tempo fixo se comporta igual não importa a taxa de quadros real.
  Valores escolhidos como equivalentes aos antigos numa taxa de ~20fps.
  Pendente: validar em celular real, principalmente num aparelho lento.

- **`FACE_DETECT_STRIDE=5`** (era 3, era todo frame antes disso) —
  FaceLandmarker é o 3º modelo completo rodando por frame; cada frame
  que ele NÃO roda sobra mais orçamento pra mão+classificação, que é o
  que realmente precisa de taxa de quadros alta pra pegar gestos
  rápidos. A sobrancelha muda de estado bem mais devagar que isso.
  `964348f` → este commit. Pendente: medir o ganho real em celular.

- **Downscale sem filtro bilinear** (`prepararBitmap`, antes usava
  `filter=true`) — essa transform roda todo frame; a imagem gerada só
  alimenta o detector, não é exibida, então a suavização do bilinear é
  custo pago à toa. Pendente: validar que a qualidade de detecção não
  piora perceptivelmente num celular real.

- **`LIMIAR_SOBRANCELHA=0.38`, `JANELA_SOBR=5`, `IDX_BROW_*`,
  `IDX_EYE_*`** — porte 1:1 do `ler_marcador` do Python
  (`m01_visuall_config.py`/`app.py`). Índices são da topologia de 468
  pontos do FaceMesh, compartilhada com o FaceLandmarker da Tasks API
  — não precisou remapear. `d9a4ddc`. Pendente: validar em celular
  real se a frase vira "?" de forma confiável.

- **`LIMIAR_MOVIMENTO=0.30`** — conflito resolvido entre 0.30 (Rafael,
  igual ao Python — deixava o "J" disparar com qualquer tremida) e
  0.55 (eu — travava gestos reais de H/J/K/X/Z). A causa raiz era usar
  UMA variável pra duas coisas: magnitude do movimento e se ele é
  intencional. Solução: volta a 0.30 (não perde gesto real), mas só é
  confiado depois de sustentado por `MOVIMENTO_SUSTENTADO_MS` (ver
  acima — era em frames, virou tempo). `d9a4ddc`. Pendente: teste real
  comparando falso-J vs. H/J/K/X/Z perdidos.

- **`INPUT_SHORT_SIDE=300`** — meio-termo entre 360 (valor do Python)
  e 255 (Rafael, ganho de velocidade). 300 perde menos detalhe que 255
  mas ainda é ~31% mais rápido que 360. `a44e3ee`. Pendente: comparar
  acurácia nas letras difíceis (E, I, U, F, G, P, Q, T, V, W, Y) nos
  três valores num celular real antes de fixar.

- **`CONFIANCA_DINAMICA=0.92`, `MARGEM_DINAMICA_MINIMA=0.28`** —
  meio-termo entre o valor solto do Rafael (0.90/0.20) e o apertado que
  eu tinha deixado (0.95/0.35). `a44e3ee`. Pendente: validação real (a
  mudança principal contra o falso-J foi a histerese do
  `LIMIAR_MOVIMENTO`, não este valor).

## Anteriores (sem entrada detalhada, ver `git log` do arquivo)

- `9970b2b` "Body Gestures working" (Rafael) — trouxe
  `LIMIAR_MOVIMENTO` 0.55→0.30, `CONFIANCA_DINAMICA` 0.95→0.90,
  `MARGEM_DINAMICA_MINIMA` 0.35→0.20, `ESTAB_MIN_DINAMICO` 6→3,
  `COOLDOWN_DINAMICO` 350→250, `INPUT_SHORT_SIDE` 360→255 — a origem
  do conflito resolvido acima.
- `09bff0e` "Require a confidence margin before committing a letter" —
  introduziu a checagem de margem (1ª − 2ª opção) porque o MLP é
  superconfiante (~0.99 quase sempre) e a confiança sozinha filtrava
  muito pouco.
