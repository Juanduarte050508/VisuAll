# Treinamento — gravar e treinar letras/gestos

Essa pasta existe pra resolver um problema específico: letras com movimento
(H, K, X, Z — só o J reconhece bem) e gestos corporais (AJUDAR, COMPUTADOR,
CONVERSAR, PESSOA, SURDO) estavam demorando ou falhando pra reconhecer no
celular. A causa mais provável **não é o código do app** — é falta de dados
de treino de verdade pra essas classes (os gestos corporais, em especial,
nunca tiveram um pipeline de treino publicado neste repositório: o modelo
atual veio de fora). Esta pasta dá um jeito fácil de gravar amostras suas e
gerar modelos novos a partir delas.

## O que cada arquivo faz

| Arquivo | Pra que serve |
|---|---|
| `Capturar.bat` | Abre a câmera com um botão GRAVAR. Grava letras/gestos direto. |
| `Treinar.bat` | Atalho rápido: pega tudo que foi gravado e treina os modelos, sem perguntar nada. |
| `abrir_treinamento.bat` | Interface completa: importar uma pasta externa de fotos/vídeos, treinar só uma categoria, ver o status do que já existe. |

Na primeira vez que você roda qualquer um dos três `.bat`, ele instala
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
   tela mostra quantos clipes aquele rótulo já tem).

### Dicas por categoria

- **Letra parada (estática)** — segure a mão fazendo a letra durante os 3
  segundos, mas mexa um pouco (gire o pulso, mude a distância da câmera) pra
  gerar variedade — na hora de treinar, o programa tira vários quadros
  automaticamente desse clipe. Grave uns 3-5 clipes por letra, em
  posições/ângulos diferentes.
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

Os clipes vão pra `treinamento\dados\raw_static_videos\<LETRA>\`,
`treinamento\dados\raw_videos\<LETRA>\` ou
`treinamento\dados\raw_body_videos\<GESTO>\` (pastas criadas
automaticamente). Essa pasta `dados\` não vai pro git — fica só no seu PC.

Já tem fotos/vídeos gravados de outro jeito (celular, outro programa)? Não
precisa passar pelo Capturar — abra `abrir_treinamento.bat`, escolha a
pasta com as subpastas por letra/gesto (ex.: `H/video1.mp4`,
`AJUDAR/gesto1.mp4`) e clique **Importar mídias** (ou **Importar + Extrair +
Treinar** pra fazer tudo de uma vez). Fotos soltas também funcionam pra
letra parada.

## Passo 2 — Treinar

Depois de gravar bastante (pode fechar o Capturar e reabrir quantas vezes
quiser antes disso), tem duas formas de treinar:

- **`Treinar.bat`** — duplo clique, sem perguntar nada: extrai os pontos da
  mão/corpo (landmarks) de tudo que existe em `treinamento\dados\` e treina
  as três categorias que tiverem amostras. Mais simples, pra quando você só
  quer atualizar tudo.
- **`abrir_treinamento.bat`** — abre uma janela com mais controle: treinar
  só uma categoria (`Tipo de treino`), importar uma pasta externa antes,
  ver o botão **Status** (quantos arquivos/datasets/modelos já existem).

Qualquer um dos dois salva os modelos novos direto em
`mobile\app\src\main\assets\`:

- `letras_estaticas\geral\model.onnx` — modelo geral, todas as 21 letras
  paradas.
- `letras_dinamicas\geral\model.onnx` — modelo geral, as 5 letras com
  movimento.
- `gestos\geral\model.tflite` — modelo geral, os gestos corporais.

Cada `geral\` vem com um `labels.txt` do lado. Sempre que você grava dados
novos de **todas** as letras de uma categoria, esse modelo geral é
atualizado. Se faltar alguma letra (ex.: gravou só H, J, K mas não X e Z),
ele treina e salva um **modelo parcial** só com o que existe — sem
sobrescrever o geral — e, no caso de letra com movimento, esse parcial
também é aplicado em `letras_dinamicas\parcial\` (o app tenta ele primeiro,
com o geral como reserva). Cada letra/gesto com dados também ganha um
**modelo individual** (`letras_estaticas\<LETRA>\`,
`letras_dinamicas\<LETRA>\`) — um classificador dedicado só pra aquela letra,
tentado antes de tudo. Isso deixa reforçar UMA letra problemática (H, por
exemplo) sem precisar ter dados balanceados de todas as outras ao mesmo
tempo.

No fim, é só recompilar o app Android (`assembleDebug` ou rodar pelo Android
Studio) pra usar os modelos novos.

## Perguntas comuns

**Preciso gravar tudo de novo toda vez?** Não — `Capturar.bat` só
*acrescenta* clipes novos, nunca apaga os antigos. Treinar sempre usa tudo
que já foi gravado/importado até agora.

**Quebrei um clipe / gravei errado?** Vá em
`treinamento\dados\raw_videos\<LETRA>\` (ou a pasta equivalente) e apague o
arquivo com o nome/data errados — os nomes têm data e hora, então dá pra
identificar o mais recente.

**A câmera não abre.** Feche outros programas que possam estar usando a
webcam (Zoom, Teams, outro Capturar.bat aberto) e tente de novo.

**Quero forçar reinstalar as dependências Python.** Apague a pasta
`treinamento\.venv` e rode qualquer um dos `.bat` de novo.

## Estado atual (o que já existia vs. o que é novo)

- **Letras paradas e letras com movimento**: já existia um pipeline de
  extração e treino "geral" fora desta pasta
  (`linear/backend/data_extraction/extract_from_{images,videos}.py`,
  `linear/backend/training/train_{static,dynamic}_model.py}`) — continua
  existindo e funcionando, é o que `letras_*\geral\` usa por baixo.
  `Capturar.bat` dá um jeito fácil de gerar dados novos pra ele, e
  `treinar_visuall.py` (motor por trás de `Treinar.bat` e
  `abrir_treinamento.bat`) acrescenta os modelos parciais/individuais por
  cima.
- **Gestos corporais**: não existia NENHUM treino publicado neste
  repositório antes — o `model.tflite` atual (`gestos/geral/`) veio de um
  pipeline externo (ver comentário em `linear/backend/app.py`). O suporte a
  treinar gestos corporais aqui (extração de 225 features por quadro —
  pose+mãos, normalização por ombros — e o modelo LSTM) é novo, escrito pra
  bater exatamente com o que `BodyGestureEngine.kt` espera. **Ainda não
  validado com dados reais** — é o ponto de partida pra gravarem e
  testarem.
