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

- **`FACE_DETECT_STRIDE=3`** — FaceLandmarker (marcador de sobrancelha)
  passa a rodar 1 a cada 3 frames em vez de todo frame; a sobrancelha
  não muda de estado tão rápido quanto uma letra, e é o 3º modelo
  completo rodando por frame. `964348f`. Pendente: medir o ganho real
  de latência em celular.

- **`LIMIAR_SOBRANCELHA=0.38`, `JANELA_SOBR=5`, `IDX_BROW_*`,
  `IDX_EYE_*`** — porte 1:1 do `ler_marcador` do Python
  (`m01_visuall_config.py`/`app.py`). Índices são da topologia de 468
  pontos do FaceMesh, compartilhada com o FaceLandmarker da Tasks API
  — não precisou remapear. `d9a4ddc`. Pendente: validar em celular
  real se a frase vira "?" de forma confiável.

- **`LIMIAR_MOVIMENTO=0.30`, `MOVIMENTO_SUSTENTADO_FRAMES=3`** —
  conflito resolvido entre 0.30 (Rafael, igual ao Python — deixava o
  "J" disparar com qualquer tremida) e 0.55 (eu — travava gestos reais
  de H/J/K/X/Z). A causa raiz era usar UMA variável pra duas coisas:
  magnitude do movimento e se ele é intencional. Solução: volta a 0.30
  (não perde gesto real), mas só é confiado depois de sustentado por
  `MOVIMENTO_SUSTENTADO_FRAMES` frames seguidos (rejeita ruído de 1
  frame). `d9a4ddc`. Pendente: teste real comparando falso-J vs.
  H/J/K/X/Z perdidos; se ainda sair J fácil, subir
  `MOVIMENTO_SUSTENTADO_FRAMES` antes de mexer em `LIMIAR_MOVIMENTO`
  de novo.

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
