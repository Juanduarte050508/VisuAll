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

- **Repositorio dividido em `mobile/` e `computer/`** — o app Android ficou
  isolado em `mobile/`; backend de PC, scripts de treino, fixtures e modelos
  Python foram agrupados em `computer/`. Caminhos de CI, scripts e testes foram
  repontados para essa nova estrutura. Status: **ativo**.

- **APK/AAB gerados ficaram fora do Git** — `*.apk`, `*.aab` e builds Android
  locais agora sao ignorados. Isso nao congela o aplicativo: codigo, assets e
  modelos usados em `mobile/app/src/main` continuam versionados; o binario e
  recriado a partir da branch atual. Status: **ativo**.

- **`normaliza_corpo` fixado em float32 em toda operação** — a função misturava
  `np.float32` com float comum do Python (`/ 2.0`, `float(np.sqrt(...))`), e o
  resultado disso **depende da versão do numpy**: no 1.x um escalar `np.float32`
  dividido por float do Python sobe para float64; no 2.x (NEP 50) fica em
  float32. As duas contas diferem por alguns ULP — o bastante para mudar a 6ª
  casa decimal. Na prática: a mesma gravação, treinada em duas máquinas com
  numpy diferente, gerava features levemente diferentes, em silêncio. Apareceu
  porque a CI (numpy 2) e uma máquina de desenvolvimento (numpy 1.24) passaram a
  discordar sobre o contrato de landmarks. Agora todas as operações são float32
  explícito, que é o que o Kotlin faz (`Float` em tudo) e o que torna a conta
  reprodutível em qualquer máquina. As fixtures, regeradas com a versão
  corrigida, voltaram a bater com as originais valor por valor. Status:
  **corrigido**.

- **`onnxruntime` entrou no `requirements.txt`, e a verificação pós-export
  voltou para a CI** — o `Treinar.bat` já exigia `import onnxruntime` na
  checagem de dependências (linha 35) e, ao falhar, rodava
  `pip install -r requirements.txt` — que não listava o pacote. O `.bat`
  tentava instalar, o `pip` terminava com sucesso sem instalar nada, e o
  treino seguia imprimindo "onnxruntime nao instalado -- pulando a validacao".
  Ou seja: a validação do modelo exportado estava desligada na prática desde
  sempre, sem ninguém ver. Sem teto de versão de propósito — ao contrário de
  `onnx`/`mediapipe`/`tensorflow`, o `onnxruntime` declara `protobuf` sem
  faixa nenhuma, então não entra na briga que os outros tetos resolvem.
  `treino/tests/test_verificacao_modelo.py` cobre os quatro casos do contrato
  do app (entrada `landmarks_input`, forma `[N, features]`, saída `[1]` com as
  probabilidades) e o caso TFLite de janela errada. **Sem `skip` de propósito:**
  `valida` devolve `False` tanto para modelo errado quanto para "onnxruntime
  ausente", então um skip deixaria a suite verde sem verificar nada — o teste
  positivo exige `True` e falha alto se a dependência sumir. Status: **ativo**.

- **`treino/Treinar.bat` invocava `extrair_negativos.py` por um caminho com um
  TAB no meio** — a linha 80 tinha o byte `0x09` onde devia ter `	` de
  `%RAIZ%	reino\`, provavelmente de um escape mastigado na hora de escrever o
  arquivo. O passo de extrair os negativos (os clipes de "Nada", que são o que
  ensina o modelo a **não** responder a movimento qualquer) nunca rodou desde
  que o arquivo foi criado. Status: **corrigido**.

- **Ferramenta de treino passou a ser `treino/`; `treinamento/` foi removida** —
  o repositório ficou com duas ferramentas de treino em paralelo e era questão
  de tempo até alguém treinar por uma e conferir pela outra. Ficou a `treino/`.
  O gerador de `tests/fixtures/landmark_contract.json` passou a chamar o
  `normaliza_corpo` da `treino/`, que faz `sqrt(dx*dx + dy*dy + dz*dz)` e
  percorre ponto a ponto — **exatamente** o que `LibrasMath.kt` faz (linhas
  107-127), enquanto o `treinar_visuall.py` usava `np.linalg.norm` vetorizado.
  O conteúdo do contrato **não mudou**: os mesmos números, os mesmos índices
  de `resample`. (Numa primeira tentativa 8 valores oscilaram na 6ª casa; era
  bug de tipo no `normaliza_corpo`, corrigido na entrada acima.)
  `treinamento/tests/test_landmark_contract.py` virou
  `treino/tests/test_landmark_contract.py`, repontado para os gêmeos
  (`normalize_landmarks`, `normaliza_corpo`, `reamostra`).
  `test_verificacao_modelo.py` foi junto na remoção e **voltou em seguida**,
  reescrito para o `valida` da `treino/` — ver a entrada abaixo.
  Status: **ativo**.

- **`CONFIANCA_INDIVIDUAL_SEM_RIVAL` = 0.99 (novo)** — o portão de margem dos
  modelos individuais não filtrava nada quando existia **um único** modelo
  treinado. Com um só, o segundo colocado é 0, então a margem passa a valer o
  mesmo que a própria confiança e `margem >= 0.32` é satisfeito por qualquer
  resposta acima de `CONFIANCA_INDIVIDUAL`. E esse não é um caso de borda: é o
  primeiro que vai acontecer — treinar uma letra só, pra medir se gravar
  resolve, gera exatamente um modelo individual. Como o binário nunca viu "mão
  mexendo sem sinalizar" como negativo (mesmo motivo de `CONFIANCA_INDIVIDUAL`
  ser alta), ele responde alto com facilidade, e o resultado seria a letra
  recém-treinada aparecendo em qualquer movimento. Agora, sem rival, a
  exigência sobe pra 0.99 e a margem é reportada como 0 em vez de fingir uma
  folga que não foi medida. Status: **ativo, precisa validar em celular** — se
  a letra treinada não aparecer nunca, este é o primeiro valor a baixar.

- **`LIMITE_FRACAO_DESCARTE` = 0.05, e o filtro de outliers passou a medir
  distância ao vizinho** — o filtro novo (`filter_outlier_samples`) usava
  `mediana + 6*MAD` da distância até o centro da classe. Medido com dados no
  formato real, ele **apagava grupos legítimos inteiros**: a partir de 60/40
  entre dois enquadramentos, o grupo menor sumia 100% — porque com um grupo
  dominante a mediana e o MAD passam a descrever só ele, e o limite fecha em
  volta da maioria. Gravar de duas distâncias é justamente o que o README pede,
  então isso destruiria 20-40% de qualquer coleta desbalanceada, em silêncio.
  Trocar o MAD por percentis evitava a perda mas errava pro outro lado: com
  dois grupos a dispersão global inflava e amostra degenerada de verdade
  passava. A medida certa é a distância ao k-ésimo vizinho (k=3): amostra ruim
  está **sozinha**, amostra de outro enquadramento tem vizinhos colados. O
  `LIMITE_FRACAO_DESCARTE` ficou como trava final — se o filtro quiser levar
  mais de 5% de uma classe, ele desiste e avisa em vez de apagar calado.
  Coberto por `TestFiltroDeOutliers`, verificado reinjetando o algoritmo
  antigo (falha) e revertendo (passa). Status: **ativo, ainda não medido com
  dados de gravação real** — os cenários testados são sintéticos.

- **CI do Android voltou a rodar (nunca tinha rodado)** —
  `mobile/gradle.properties`, que é versionado, tinha
  `org.gradle.java.home=C:\Program Files\Android\Android Studio\jbr`: um
  caminho absoluto de uma máquina específica. O runner é Linux, então o
  Gradle abortava na primeira chamada ("Java home supplied is invalid").
  A linha está lá desde a criação do arquivo, ou seja, o workflow do Android
  **nunca passou** e ninguém tinha notado. Removida (com comentário
  explicando por que não voltar a pôr): o Android Studio usa o JDK das
  próprias configurações e a linha de comando usa `JAVA_HOME` — que é
  obrigatório de todo jeito, porque o `gradlew` precisa de Java pra iniciar
  antes de conseguir ler esse arquivo.

- **`requirements.txt` reescrito depois de testar instalação limpa** — a
  correção anterior (fixar `protobuf<4`) estava errada: ela foi deduzida da
  versão antiga de mediapipe instalada nesta máquina. Instalando do zero num
  venv limpo, o conjunto declarado resolvia pra mediapipe 1.0.0 + protobuf 6
  e **não funcionava**. Investigando as versões de verdade apareceu um
  conflito de três pontas: mediapipe 0.10.14-21 exige `protobuf<5`,
  tensorflow 2.20 exige `protobuf>=5.28`, e onnx 1.19 (puxado por skl2onnx
  recente) exige `protobuf>=5`. Agora os tetos são nas bibliotecas, não no
  protobuf (que é consequência), e a combinação foi verificada resolvendo
  sem conflito: mediapipe 0.10.21 + protobuf 4.25.9 + tensorflow 2.19.1 +
  onnx 1.16.2.

- **Aviso claro quando o Python é antigo demais** — descoberto no mesmo
  teste: mediapipe 0.10.13+ e tensorflow 2.16+ instalam em Python 3.9
  (existe wheel cp39) mas quebram no import com `TypeError: unhashable
  type: 'list'`, um erro que não diz nada sobre a causa. O projeto sempre
  documentou 3.10+, mas nada verificava — quem tivesse 3.9 esperava vários
  minutos de download pra receber esse erro no fim. `_ambiente.bat` agora
  checa a versão antes de montar o ambiente e explica o que fazer.

- **Salvar calibração deixa de travar a tela** — o botão de salvar amostra
  lia e regravava o CSV de treino inteiro (até ~5 MB no caso dinâmico
  cheio) na thread da interface. O app congelava a cada amostra salva, e
  piorava conforme o arquivo enchia — parecia que "foi ficando lento".
  Numa sessão de gravação são dezenas de salvamentos seguidos, então isso
  aparecia o tempo todo. Agora a escrita vai pra uma thread própria (uma
  só, pra as gravações não se atropelarem, e o `clear()` entra na mesma
  fila pra não apagar em cima de uma gravação pendente) e o arquivo é
  ANEXADO em vez de regravado: a poda das linhas antigas só acontece
  quando o teto é de fato ultrapassado, com a contagem guardada em prefs
  pra não precisar ler o arquivo só pra saber o tamanho.

- **`LetterCommitGate`: a regra que aceita uma letra virou testável** — os
  quatro portões (estabilidade mínima, letra válida, não repetir a última,
  cooldown) viviam soltos no meio do processamento de câmera do
  `LibrasAnalyzer`, onde só dava pra verificar com celular, câmera e modelo
  na mão. É justamente a regra mais ajustada por tentativa e erro neste
  arquivo. Extraída sem mudança de comportamento, agora com 11 testes que
  fixam o que se espera dela em vez de só os números: "um quadro isolado
  nunca vira letra", "mão parada não digita AAAA", "trocar de letra
  reinicia a contagem", "dinâmica estabiliza mais rápido que estática",
  "`dinamico_individual`/`dinamico_parcial` contam como dinâmica". Os
  testes leem as constantes reais, então mexer nos limiares os mantém
  válidos. Verificados injetando dois bugs de propósito (remover a trava de
  repetição e trocar o prefixo por igualdade exata) — os dois foram pegos.

- **Contrato travado entre a matemática do treino e a do app** — a mesma
  conta (normalização de mão, de corpo e reamostragem de janela) existia
  escrita duas vezes, em Python e em Kotlin, sem nada garantindo que
  batessem. Se uma mudasse sem a outra, nada quebrava e nenhum erro
  aparecia: o app só passaria a errar mais, porque foi treinado num formato
  e usado em outro. Agora `tests/fixtures/landmark_contract.json` congela
  entradas/saídas e dois testes (`LandmarkContractTest` no Kotlin,
  `test_landmark_contract.py` no Python) falham se um lado divergir; o CI
  ainda regera as fixtures e exige que não mudem. **Já pegou uma divergência
  real**: com os ombros quase colados (pose degenerada) o Python dividia por
  ~1e-5 e gerava features na casa dos milhares, enquanto o Kotlin já
  protegia e dividia por 1 — quadros assim envenenavam o treino com valores
  que o app nunca produz. Os dois agora usam `ESCALA_MINIMA_OMBROS=0.0001`.
  A verificação de que o teste realmente pega o erro foi feita injetando a
  divergência de propósito (e revelou que a primeira versão do teste NÃO
  pegava, por faltar um caso de cada lado do limiar — corrigido).

- **Verificação automática do modelo exportado** — depois de cada treino, o
  `.onnx`/`.tflite` gerado é aberto e conferido contra o que o app carrega
  (entrada `landmarks_input`, forma `[N, features]`, saída de
  probabilidades; no TFLite, `[1,30,225]` e execução real pra garantir o
  Select TF Ops). Antes, um export com nome ou formato errado só aparecia
  com o app já instalado, como "parou de reconhecer". 5 testes garantem que
  a verificação REJEITA modelo errado — checagem que só aprova não protege.

- **Falhas de carregamento de modelo deixam de ser silenciosas** — o modo
  corpo desativava-se sozinho e a tela ficava muda; modelos individuais e
  parciais eram engolidos por `runCatching{}.getOrNull()`, tornando "modelo
  corrompido" indistinguível de "modelo nunca treinado". Agora o motivo vai
  pro log e, no modo corpo, pra tela. Importante agora que os modelos passam
  a ser regerados com frequência.

- **Lista de letras unificada** — a tela de calibração tinha o alfabeto
  escrito à mão, separado dos `labels.txt` que acompanham os modelos. Se um
  treino mudasse as letras, a calibração seguiria oferecendo as antigas e a
  pessoa gravaria amostras para uma letra que o modelo não tem. Agora a UI
  deriva dos labels reais (com o alfabeto padrão só como rede de segurança,
  pra tela nunca ficar vazia).

- **Código de treino consolidado em `treinamento/`** — removidos 6 arquivos
  em `linear/backend/` (`extract_from_{images,videos}.py`, os dois aliases
  legados e `train_{static,dynamic}_model.py`) que nada mais chamava:
  `treinar_visuall.py` já fazia tudo o que eles faziam, e manter os dois
  convidava a mexer no arquivo errado. `training_common.py` veio junto.

- **Modelos `.pkl` fora do controle de versão** — eram versionados à força
  contra o próprio `.gitignore`, pesam ~2,3 MB, o app não os usa (ele lê
  `.onnx`/`.tflite`) e cada retreino guardaria uma cópia nova no histórico
  para sempre. Continuam sendo gerados localmente. Isso interrompe o
  crescimento; o histórico já existente não foi reescrito.

- **Exemplos negativos ("não é sinal nenhum") no treino** — ataca a causa
  raiz do reconhecimento fácil demais, que a subida de limiares abaixo só
  remendava. Os modelos individuais são binários ("é esta letra ou não?") e
  só viam OUTRAS LETRAS como exemplo negativo — nunca uma mão à toa —, então
  não tinham como aprender a rejeitar o que não é letra nenhuma. Agora o
  Capturar tem a categoria "Nada (não é sinal nenhum)", e
  `train_individual_mlp` mistura esses clipes no pool de negativos (com teto
  por fonte, pra não afogar os positivos). Quem não gravar nenhum continua
  treinando igual, só com um aviso. Pendente: gravar os clipes e medir com o
  `VALIDACAO.md` — se a Parte 2 (falso positivo) zerar, dá pra considerar
  baixar `CONFIANCA_INDIVIDUAL` de volta, já que ele existe como remendo do
  mesmo problema.

- **`requirements.txt`: pinos de `protobuf`/`onnx`/`skl2onnx`** — instalar o
  arquivo como estava, do zero, resolvia pra skl2onnx 1.20 + onnx 1.19 +
  protobuf 6, e nessa combinação o **mediapipe para de funcionar**
  (`MessageFactory object has no attribute GetPrototype` ao criar qualquer
  detector). Ou seja: quem clonasse o repo e rodasse `Capturar.bat` pela
  primeira vez pegava um ambiente quebrado. Encontrado ao instalar as
  dependências declaradas mas ausentes. Travado em `protobuf<4` +
  `onnx<1.17` + `skl2onnx<1.18`, combinação verificada rodando mediapipe,
  sklearn/ONNX e tensorflow juntos.

- **`CONFIANCA_MINIMA` 0.90→0.93, `MARGEM_ESTATICA_MINIMA` 0.25→0.30,
  `CONFIANCA_DINAMICA` 0.92→0.95, `MARGEM_DINAMICA_MINIMA` 0.28→0.32,
  `BODY_CONFIDENCE` 0.85→0.90, `CONFIANCA_INDIVIDUAL` novo em 0.97** —
  usuário relatou reconhecimento fácil demais depois dos modelos
  individuais por letra entrarem (ver entrada de unificação abaixo):
  letras/gestos sendo confirmados sem estar sendo feitos. Causa mais
  provável: os modelos individuais são classificadores binários ("é esta
  letra ou não"), treinados só contra OUTRAS LETRAS REAIS como exemplo
  negativo — nunca viram "mão se mexendo sem sinalizar nada" no treino, e
  por isso tendem a ficar confiantes demais (overconfident) em movimento
  que não é sinal nenhum. `CONFIANCA_INDIVIDUAL` é uma barra bem mais alta
  usada só por esse nível (o geral e o parcial continuam usando
  `CONFIANCA_MINIMA`/`CONFIANCA_DINAMICA`, só um pouco mais estritos que
  antes). Estabilidade/cooldown (`ESTAB_MIN_*`, `COOLDOWN_*`, ver entrada
  mais abaixo) não foram mexidos — pedido explícito de manter, é o que
  segura a mesma letra por mais tempo antes de comitar. Pendente: validar
  em celular real que ainda reconhece letras/gestos de verdade sem travar
  demais.

- **Roteiro de validação (`VALIDACAO.md`) e primeiros testes automáticos do
  lado Python** — quase toda entrada deste arquivo termina em "pendente:
  validar em celular real", e nenhuma jamais foi validada: os ajustes foram
  se empilhando sem ninguém medir se o anterior ajudou. `VALIDACAO.md` é um
  teste de ~10 min (alfabeto A→Z + 30s sem sinalizar pra contar falso
  positivo + gestos), feito sempre igual, pra comparar antes/depois. O teste
  de falso positivo é o que faltava: sem ele dá pra "melhorar" o alfabeto só
  afrouxando limiares e deixando o app chutar. Junto vieram 18 testes
  automáticos do motor de treino (`treinamento/tests/`) e um workflow que
  roda eles + verifica que o mediapipe realmente inicializa num ambiente
  recém-instalado — era o buraco que deixou passar tanto o conflito do
  protobuf acima quanto o modelo de corpo salvo no caminho errado.

- **`LibrasFragment.kt`: extraídos `WordSuggestionEngine` e
  `TrainingProgressCalculator`** — continuava sendo o maior arquivo do app
  (~1250 linhas). Saíram as duas partes que eram lógica pura, sem View: a
  pontuação/ranking das sugestões de palavra e as contas do painel de treino
  (percentual, letras fracas, próxima a treinar). Ganharam 20 testes
  unitários — antes, a única forma de conferir se "ban" sugeria "banheiro",
  ou se 400 amostras de uma letra só não marcavam 100% no painel, era abrir
  o app e soletrar na mão. Câmera, HUD e painéis continuam no Fragment.

- **Ferramentas de treino unificadas em `treinamento/`** — duas pessoas
  construíram, em paralelo e sem saber uma da outra, ferramentas pro mesmo
  problema (dados de treino insuficientes pra letras com movimento e
  gestos corporais). Fundidas em uma: `Capturar.bat` (câmera com botão
  GRAVAR, contagem de 3s, grava e salva sozinho) agora grava direto em
  `treinamento/dados/raw_*`, o layout que `treinar_visuall.py` já lia;
  `Treinar.bat` virou um atalho rápido pra esse motor (extrai+treina tudo
  sem perguntar nada); `abrir_treinamento.bat` continua a interface
  completa (importar pasta externa, treinar só uma categoria, ver status).
  Os modelos individuais/parciais/gerais (H/J/K/Z já commitados) e o
  suporte a gestos corporais (extração de 225 features/frame — pose+mãos,
  normalização por ombros — e treino LSTM→TFLite) são do
  `treinar_visuall.py`; a implementação equivalente que existia em
  `linear/backend/` (`extract_from_videos_corpo.py`, `train_body_model.py`)
  foi removida por ficar duplicada depois da fusão. Documentação
  consolidada em `treinamento/README.md` (era dois arquivos,
  `README.md` + `COMO_USAR.md`, com instruções diferentes).

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
