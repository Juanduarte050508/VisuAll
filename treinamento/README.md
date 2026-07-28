# Treinamento — gravar e treinar letras/gestos

Essa pasta existe pra resolver um problema específico: letras com movimento
(H, K, X, Z — só o J reconhece bem) e gestos corporais (AJUDAR, COMPUTADOR,
CONVERSAR, PESSOA, SURDO) estão demorando ou falhando pra reconhecer no
celular. A causa mais provável **não é o código do app** — é falta de dados
de treino de verdade pra essas classes (os gestos corporais, em especial,
nunca tiveram um pipeline de treino publicado neste repositório: o modelo
atual veio de fora). Esta pasta dá um jeito fácil de gravar amostras suas e
gerar modelos novos a partir delas.

Só precisa de duas coisas: **duplo-clique** e **clicar em GRAVAR**.

> **Nota:** essa pasta também tem `abrir_treinamento.bat` / `interface_treinamento.py`
> / `treinar_visuall.py` / `COMO_USAR.md` — uma segunda ferramenta que o
> Rafael construiu em paralelo, com uma abordagem diferente (você organiza
> fotos/vídeos numa pasta à parte e importa, em vez de gravar direto pela
> câmera; e treina modelos individuais por letra/gesto além do geral). As
> duas ferramentas fazem parte do mesmo objetivo mas ainda não foram
> unificadas — ver conversa/CHANGELOG para o time decidir qual seguir ou
> como combinar as duas.

## O que cada arquivo faz (esta ferramenta)

| Arquivo | Pra que serve |
|---|---|
| `Capturar.bat` | Abre a câmera com um botão GRAVAR. Grava letras/gestos. |
| `Treinar.bat` | Pega tudo que você gravou e treina os modelos novos. |

Na primeira vez que você roda qualquer um dos dois `.bat`, ele instala
sozinho um Python isolado (dentro de `treinamento\.venv`, não mexe no resto
do seu PC) com tudo que precisa. Só exige ter **Python 3.10 ou mais novo**
instalado e marcado "Add python.exe to PATH" na instalação
(https://python.org/downloads/). Da segunda vez em diante abre na hora.

## Passo 1 — Gravar amostras (`Capturar.bat`)

Dê duplo-clique em `Capturar.bat`. Na janela que abrir:

1. Escolha a **Categoria**: letra parada, letra com movimento, ou gesto
   corporal.
2. Escolha o **Rótulo** (qual letra ou palavra você vai fazer).
3. Clique **GRAVAR**. Você tem 3 segundos pra se preparar (contagem na
   tela), depois grava sozinho por 3 segundos, e salva sozinho. Sem precisar
   segurar nem soltar nada.
4. Repita: pode gravar o mesmo rótulo várias vezes seguidas (o contador na
   tela mostra quantas amostras aquele rótulo já tem).

### Dicas por categoria

- **Letra parada (estática)** — segure a mão fazendo a letra durante os 3
  segundos, mas mexa um pouco (gire o pulso, mude a distância da câmera) pra
  gerar variedade — o programa tira várias fotos automaticamente desse
  clipe. Grave uns 3-5 clipes por letra, em posições/ângulos diferentes.
- **Letra com movimento (H, J, K, X, Z)** — faça o movimento completo da
  letra UMA vez dentro da janela de 3 segundos (nem muito rápido nem muito
  devagar — no ritmo normal que você sinaliza). Grave pelo menos **15-20
  clipes por letra**, principalmente H, K, X e Z (o J já reconhece bem hoje,
  mas não custa gravar mais alguns também).
- **Gesto corporal** — faça o sinal completo (as duas mãos, se for o caso)
  dentro dos 3 segundos, com o corpo todo visível pela câmera (não só a
  mão). Grave pelo menos **15-20 clipes por gesto**. **Importante:** grave
  também vários clipes de **NEUTRO** — você parado, sem fazer nenhum sinal,
  mas com a mão à mostra (o modelo só aprende com quadros em que te vê; se a
  mão sai do quadro no NEUTRO ele não aprende nada com aquele clipe). Sem
  amostras de NEUTRO o modelo tende a "ver sinais" em qualquer movimento
  parado.

Os arquivos vão pra `VisuAll\data\raw_images\<LETRA>\`,
`VisuAll\data\raw_videos\<LETRA>\` ou `VisuAll\data\raw_videos_corpo\<GESTO>\`
(criadas automaticamente). Essa pasta `data\` não vai pro git — fica só no
seu PC.

## Passo 2 — Treinar (`Treinar.bat`)

Depois de gravar bastante (pode fechar o Capturar e reabrir quantas vezes
quiser antes disso), dê duplo-clique em `Treinar.bat`. Ele:

1. Extrai os pontos da mão/corpo (landmarks) de cada foto/vídeo gravado.
2. Treina os modelos com o que encontrar (pula sozinho qualquer categoria
   sem amostras — não precisa ter gravado as três).
3. Salva os modelos novos direto em `mobile\app\src\main\assets\`:
   `letras_estaticas\geral\model.onnx`, `letras_dinamicas\geral\model.onnx`,
   `gestos\geral\model.tflite` (e os `labels.txt` correspondentes).

No fim, é só recompilar o app Android (`assembleDebug` ou rodar pelo Android
Studio) pra usar os modelos novos. Pode rodar `Treinar.bat` de novo sempre
que gravar mais amostras — ele reprocessa tudo que tiver em `data\`.

## Perguntas comuns

**Preciso gravar tudo de novo toda vez?** Não — `Capturar.bat` só
*acrescenta* clipes novos, nunca apaga os antigos. `Treinar.bat` sempre usa
tudo que já foi gravado até agora.

**Quebrei um clipe / gravei errado?** Vá em
`VisuAll\data\raw_videos\<LETRA>\` (ou a pasta equivalente) e apague o
arquivo com o nome/data errados — os nomes têm data e hora, então dá pra
identificar o mais recente.

**A câmera não abre.** Feche outros programas que possam estar usando a
webcam (Zoom, Teams, outro Capturar.bat aberto) e tente de novo.

**Quero forçar reinstalar as dependências Python.** Apague a pasta
`treinamento\.venv` e rode qualquer um dos `.bat` de novo.

## Estado atual (o que já existia vs. o que é novo)

- **Letras paradas e letras com movimento**: o pipeline de extração e treino
  já existia (`linear/backend/data_extraction/extract_from_{images,videos}.py`,
  `linear/backend/training/train_{static,dynamic}_model.py`) — `Capturar.bat`
  e `Treinar.bat` só dão uma forma fácil de alimentar ele.
- **Gestos corporais**: não existia NENHUM treino publicado neste
  repositório — o `body_model.tflite` atual veio de um pipeline externo (ver
  comentário em `linear/backend/app.py`). `extract_from_videos_corpo.py` e
  `train_body_model.py` são novos, escritos pra bater exatamente com o que
  `BodyGestureEngine.kt` espera (225 features por quadro, janela de 30
  quadros, mesma normalização por ombros) — mas a arquitetura do modelo
  (LSTM) é nova, criada pra este projeto, não é cópia de nada existente.
  **Ainda não validada com dados reais** — é o ponto de partida pra vocês
  gravarem e testarem.
