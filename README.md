# VisuAll — Reconhecimento de Libras em Tempo Real

> Reconhecimento do alfabeto de Libras (Língua Brasileira de Sinais) em tempo real, usando visão computacional e dois modelos MLP especializados — um para letras estáticas e outro para letras com movimento.

![Stack](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands-green)
![scikit--learn](https://img.shields.io/badge/scikit--learn-MLP-orange?logo=scikitlearn&logoColor=white)
![WebSocket](https://img.shields.io/badge/WebSocket-asyncio-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## TL;DR

Webcam capta o gesto → MediaPipe extrai 21 landmarks da mão → o sistema decide automaticamente se é uma letra **estática** (A, B, C…) ou **dinâmica** (H, J, K, X, Z) com base no movimento detectado nos últimos frames → o MLP correspondente classifica → a letra é adicionada à frase quando se mantém estável. Tudo isso roda em ~20fps no frontend via WebSocket.

---

## Demo

> *(Em breve: gif/vídeo do sistema em uso. Por enquanto, rode localmente seguindo as instruções abaixo.)*

O frontend simula a interface de um smartphone (J0VI) e mostra:
- Stream da webcam com landmarks da mão sobrepostos
- Letra atual sendo reconhecida + nível de confiança
- Frase em construção
- Histórico das últimas frases reconhecidas
- Indicador do modo de detecção ativo (estático / dinâmico)

---

## Arquitetura

```
┌─────────────────┐    WebSocket (ws://localhost:8000)   ┌──────────────────┐
│   Frontend      │ <──────────────────────────────────> │   Backend        │
│   (HTML/CSS/JS) │   ↓ frame JPEG base64 + estado       │   (Python)       │
└─────────────────┘   ↑ comandos (limpar, espaço…)       └──────────────────┘
                                                                  │
                                                                  ↓
                                                         ┌──────────────────┐
                                                         │ Capture Thread   │  ← OpenCV / webcam
                                                         │ Process Thread   │  ← MediaPipe + MLP
                                                         │ asyncio loop     │  ← WebSocket I/O
                                                         └──────────────────┘
```

### Pipeline de classificação

1. **Captura** (thread dedicada): lê frames da webcam a 30fps e armazena no buffer compartilhado.
2. **Processamento** (thread dedicada): para cada frame novo:
   - Extrai 21 landmarks 2D da mão via **MediaPipe Hands**.
   - Normaliza os pontos em relação ao pulso (invariante à posição na tela).
   - Calcula a variação de movimento nos últimos 5 frames.
   - Se movimento > limiar → roteia para o **MLP dinâmico** (janela de 10 frames, 420 features).
   - Se mão parada → roteia para o **MLP estático** (frame único, 42 features).
   - Aplica filtro de estabilidade (N frames consecutivos com mesma predição) antes de adicionar à frase.
3. **Gesto especial**: mão totalmente aberta por 3s limpa a frase atual.
4. **WebSocket**: envia frame + estado a ~20fps; recebe comandos do frontend (espaço, apagar, limpar).

### Por que dois modelos MLP em vez de um modelo único?

Letras estáticas e dinâmicas têm estruturas de feature radicalmente diferentes. Tentar treinar um modelo único forçaria padding/zero-fill e degradaria ambos os casos. Separar permite:
- **Estático**: 42 features (21 pontos × 2 coords), alta precisão para A–G, I, L, M, N, O, P, Q, R, S, T, U, V, W, Y.
- **Dinâmico**: 420 features (10 frames × 42), captura trajetória para H, J, K, X, Z.
- **Roteador heurístico** (limiar de movimento) escolhe qual modelo usar — sem custo extra de inferência.

---

## Stack

| Camada | Tecnologias |
|---|---|
| Captura de vídeo | OpenCV |
| Detecção de mão | MediaPipe Hands |
| Modelos | MLP (scikit-learn) — 256→128 hidden, ReLU, early stopping |
| Comunicação | WebSocket (`websockets` + `asyncio`) |
| Concorrência | `threading` (captura + processamento) + asyncio (I/O) |
| Frontend | HTML5 + CSS3 + JavaScript vanilla |

---

## Estrutura do repositório

```
visuall/
├── backend/
│   ├── app.py                          # Servidor WebSocket + pipeline de inferência
│   ├── data_extraction/
│   │   ├── extract_from_images.py      # Gera dataset estático a partir de fotos
│   │   └── extract_from_videos.py      # Gera dataset dinâmico a partir de vídeos
│   └── training/
│       ├── train_static_model.py       # Treina MLP estático
│       └── train_dynamic_model.py      # Treina MLP dinâmico
├── frontend/
│   └── index.html                      # UI completa (simula smartphone J0VI)
├── models/
│   ├── static_model.pkl                # MLP estático treinado
│   ├── static_classes.pkl              # Mapeamento idx → letra
│   ├── dynamic_model.pkl               # MLP dinâmico treinado
│   └── dynamic_classes.pkl
├── docs/
│   └── architecture.md                 # Notas técnicas mais profundas
├── requirements.txt
└── README.md
```

---

## Como rodar

### Pré-requisitos
- Python 3.10+
- Webcam funcional
- (Recomendado) Ambiente virtual

### Setup

```bash
# 1. Clone e entre no diretório
git clone https://github.com/Juanduarte050508/visuall.git
cd visuall

# 2. Crie ambiente virtual
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows

# 3. Instale dependências
pip install -r requirements.txt
```

### Execução

```bash
# Terminal 1 — backend
cd backend
python app.py
# Aguarde: "✅ Servidor pronto! MLP dinâmico + MLP estático ativos"

# Terminal 2 — frontend
# Basta abrir frontend/index.html no navegador (Chrome/Edge recomendado).
# Ele conecta automaticamente em ws://localhost:8000
```

---

## Treinando seus próprios modelos

Os modelos pré-treinados estão em `models/`. Caso queira treinar do zero:

### 1. Coletar dados

```
data/
├── raw_images/        # para letras estáticas
│   ├── A/  foto1.jpg foto2.jpg ...
│   ├── B/  ...
│   └── ...
└── raw_videos/        # para letras dinâmicas
    ├── H/  v1.mp4 v2.mp4 ...
    ├── J/  ...
    └── ...
```

### 2. Extrair landmarks

```bash
cd backend/data_extraction
python extract_from_images.py    # gera data/dataset_static.npz
python extract_from_videos.py    # gera data/dataset_dynamic.npz
```

### 3. Treinar

```bash
cd backend/training
python train_static_model.py     # gera models/static_model.pkl
python train_dynamic_model.py    # gera models/dynamic_model.pkl
```

---

## Decisões técnicas relevantes

- **Threading + asyncio**: a captura da webcam é bloqueante e não combina com o loop assíncrono do WebSocket. A solução foi separar em duas threads de daemon (captura + processamento) sincronizadas via `Lock`, deixando o asyncio responsável apenas pelo I/O de rede.
- **Resize para 320×240 antes do MediaPipe**: o frame original (640×480) é mantido para exibição, mas a inferência usa uma versão reduzida — ganho de ~2× em FPS sem perda relevante de precisão na detecção de mão.
- **Janela de 10 frames para o modelo dinâmico**: testado com 8, 10 e 15. 10 oferece o melhor trade-off entre latência (~330ms a 30fps) e precisão na captura do gesto completo.
- **Limiar de confiança (0.90) + estabilidade temporal**: predição só vira "letra confirmada" se o modelo estiver com ≥90% de confiança E mantiver a mesma letra por N frames consecutivos (12 estáticas, 2 dinâmicas — gestos dinâmicos passam rápido).
- **Cooldown pós-letra**: 1s para estáticas, 0.3s para dinâmicas — evita repetições acidentais durante a transição entre gestos.

---

## Próximos passos

- [ ] Tornar o frontend hospedável (servir HTML pelo próprio backend via aiohttp)
- [ ] Adicionar suporte a duas mãos para letras como "H"
- [ ] Migrar para LSTM para melhor captura temporal de letras dinâmicas
- [ ] Empacotar inferência em ONNX para rodar embarcado em smartphone
- [ ] Criar dataset público com diversidade de tons de pele e iluminação

---

## Contexto do projeto

Este projeto faz parte do **Challenge FIAP 2026** — programa em que estudantes de Engenharia de Software desenvolvem soluções reais para empresas parceiras. A parceira do desafio é a **J0VI**, uma proposta de smartphone com foco em acessibilidade.

O reconhecimento de Libras aqui demonstrado é uma das features acessibilidade desenvolvidas. O escopo deste repositório cobre **a parte de visão computacional e ML** do produto, isolada como um sistema funcional independente.

**Equipe VisuAll** — desenvolvido em equipe por estudantes de Engenharia de Software da FIAP. Este repositório contém o código no qual atuei como desenvolvedor principal das partes de captura, modelagem e backend.

---

## Sobre o autor

**Juan Duarte Moura** — estudante de Engenharia de Software (FIAP, turma de 2030) e técnico em Mecatrônica (ETEC). Background em integração hardware/software (Arduino, Fusion 360, F1 in Schools). Interessado em visão computacional, IA aplicada e backend.

[LinkedIn](https://www.linkedin.com/in/) · [Outros projetos](https://github.com/Juanduarte050508)

---

## Licença

[MIT](LICENSE) — sinta-se livre para usar como referência. Uma menção é apreciada mas não obrigatória.
