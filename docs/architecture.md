# Notas Técnicas — VisuAll

Documento complementar com decisões de design, trade-offs e detalhes de implementação que não cabem no README principal.

## 1. Por que MLP em vez de CNN?

O input do classificador não é uma imagem — são **landmarks já normalizados** (coordenadas relativas ao pulso). MediaPipe Hands resolve a parte difícil (detectar a mão e localizar 21 pontos) com um modelo CNN próprio bem otimizado.

Depois disso, o problema vira tabular: dado um vetor de coordenadas, qual letra é? Nesse cenário, MLP é:

- **Mais leve**: 1.7MB (dinâmico) + 566KB (estático), contra dezenas de MB típicos de CNNs.
- **Mais rápido para inferir**: latência abaixo de 5ms na CPU.
- **Suficiente em precisão**: as fronteiras de decisão entre letras de Libras no espaço de landmarks são bem separáveis.

Uma CNN só faria sentido se quiséssemos ignorar o MediaPipe e treinar fim-a-fim a partir do frame raw. Isso seria mais robusto a oclusões, mas custaria 100× mais em compute e dados.

## 2. Roteamento estático vs dinâmico

A decisão de qual modelo usar é tomada por uma heurística simples:

```python
mov = std(x_pulso) + std(y_pulso) + std(x_indicador) + std(y_indicador)
if mov > 0.30:
    usar modelo dinâmico
else:
    usar modelo estático
```

### Por que essa heurística?

Tentei alternativas:
- **Classificador binário separado** (estático vs dinâmico): adiciona latência e mais um modelo a manter; ganho marginal.
- **Modelo unificado com janela sempre de 10 frames**: força letras estáticas a "esperarem" 10 frames antes de classificar — UX ruim.
- **Sempre rodar os dois modelos em paralelo e pegar o mais confiante**: dobra o custo de inferência.

A heurística baseada em movimento é praticamente gratuita (4 desvios-padrão sobre 5 valores) e funciona bem porque o gesto humano de uma letra dinâmica é claramente distinguível em magnitude da estabilidade de uma letra estática.

### Limites da abordagem

- Tremor da mão pode falsamente acionar modo dinâmico → mitigado pelo limiar de confiança.
- Gestos dinâmicos lentos podem não acionar → mitigado por o limiar de movimento ser baixo (0.30 em coordenadas normalizadas).

## 3. Normalização de landmarks

```python
def normalize_landmarks(pontos):
    base_x, base_y = pontos[0]   # pulso
    norm = []
    for x, y in pontos:
        norm.append(x - base_x)
        norm.append(y - base_y)
    max_v = max(abs(v) for v in norm) or 1.0
    return [v / max_v for v in norm]
```

Duas operações:
1. **Translação**: subtrai a posição do pulso → invariante à posição da mão na tela.
2. **Escala uniforme**: divide pelo maior valor absoluto → invariante ao tamanho da mão (distância da câmera).

**Não normalizo por componente** (não divido x e y separadamente) porque isso destruiria a proporção entre dimensões — a forma da mão precisa ser preservada para o classificador.

## 4. Threading model

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  CaptureThread  │───>│  raw_frame (Lock) │<───│ ProcessThread   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
                                                        ↓
                                              ┌──────────────────┐
                                              │ camera_data(Lock)│
                                              └──────────────────┘
                                                        ↑
                                                        │ leitura
                                              ┌──────────────────┐
                                              │  asyncio (main)  │ ──> WebSocket
                                              └──────────────────┘
```

- **CaptureThread**: lê a webcam o mais rápido possível, sobrescreve `raw_frame`. Se o processamento for mais lento, frames antigos são descartados — comportamento desejável (latência > completude).
- **ProcessThread**: monitora timestamp de `raw_frame`, processa apenas se for novo, escreve em `camera_data`.
- **asyncio loop (main)**: lê `camera_data` a cada 50ms e envia pelo WebSocket. Independente do FPS de processamento.

Os locks são curtos (cópia de referência ou de dict pequeno), nunca cobrem operações de I/O ou CPU pesadas. Isso evita contenção.

## 5. Estabilidade temporal antes de adicionar à frase

Predição instantânea ≠ letra confirmada. O fluxo é:

```
predição instantânea → conta frames consecutivos com mesma predição
                    → se atingir limiar (12 estática / 2 dinâmica) → vira candidata
                    → respeita cooldown desde última adição (1s/0.3s)
                    → adiciona à frase
                    → bloqueia repetição imediata da mesma letra por 1s
```

Os números (12, 2, 1s, 0.3s) saíram de teste empírico. Letras dinâmicas precisam de menos frames porque o gesto em si já dura ~300ms — não dá pra esperar 12 frames.

## 6. Treinamento dos modelos

### Dataset

- **Estático**: ~500 amostras por letra, capturadas via fotos. Total ~10k amostras.
- **Dinâmico**: ~400 amostras por letra (cada amostra = janela de 10 frames). Total ~2k amostras.

### Balanceamento

Ambos os scripts limitam o número de amostras por classe (`MAX_POR_CLASSE`) para evitar viés. Letras com mais dados são amostradas aleatoriamente até o limite.

### Hiperparâmetros

```python
MLPClassifier(
    hidden_layer_sizes=(256, 128),
    activation="relu",
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
)
```

- `(256, 128)` empiricamente bateu `(128,)` e `(512, 256, 128)`.
- `early_stopping` evita overfitting nos datasets pequenos do modo dinâmico.
- 80/20 train/test split com `stratify` para manter proporção das classes.

### Métricas observadas

- Estático: ~98% accuracy no test set.
- Dinâmico: ~94% accuracy no test set (números menores por menos dados disponíveis).

## 7. Limitações conhecidas

1. **Uma mão só** (`max_num_hands=1`). Letras como "H" em algumas variantes regionais usam duas mãos.
2. **Sem suporte a palavras dinâmicas** (sinais inteiros, só alfabeto manual).
3. **Sensibilidade a iluminação ruim**: MediaPipe perde a mão em ambientes muito escuros.
4. **Dataset com baixa diversidade**: gravado por um número pequeno de pessoas. Risco de viés de tom de pele e tamanho de mão.
5. **Latência de ~50ms no WebSocket**: aceitável para uso interativo, mas não para tempo real estrito.
