# TREINO — gravar vídeos e gerar o modelo do app

Como ensinar letras novas ao app do celular. Todo o trabalho pesado acontece no
PC; o celular só recebe, no fim, **dois arquivos pequenos**.

---

## O essencial em 30 segundos

```
   NO PC (quando quiser ensinar algo novo)      NO CELULAR (sempre)
   ──────────────────────────────────────       ───────────────────
   1. Gravar.bat   grava os vídeos
   2. Treinar.bat  vira model.onnx        ──→   o .onnx vai dentro do APK
   3. compilar o app                            funciona offline, sem PC
```

Os **vídeos ficam no PC para sempre** — nunca vão pro celular. O que vai é só o
`model.onnx` (uns 200–600 KB). Depois de instalado, o app não precisa de PC nem
de internet.

---

## Passo 1 — Gravar

Duplo-clique em **`treino\Gravar.bat`**.

Aparece a letra grande na tela. Aperta **espaço** → conta **3, 2, 1** → grava 3
segundos → salva sozinho → pula pra próxima letra.

| Tecla | O que faz |
|---|---|
| **ESPAÇO** | grava o clipe |
| **N** / **P** | próxima / anterior letra |
| **TAB** | alterna letras paradas ↔ com movimento |
| **Q** | sai |

**Meta:** 15–20 clipes por letra com movimento (H, J, K, X, Z), 3–5 por letra
parada — variando ângulo e distância.

**A regra que mais estraga treino:** a mão precisa ficar visível **do começo ao
fim** dos 3 segundos. Qualquer quadro sem mão faz o extractor jogar fora a
sequência inteira ([extract_from_videos.py:70-72](linear/backend/data_extraction/extract_from_videos.py#L70-L72)),
e ele precisa de 10 quadros seguidos com mão pra gerar 1 amostra.

### Gravou pelo celular?

Grave com a câmera normal, uma letra por sessão, e importe:

```bat
python treino\importar.py --adb                    :: cabo USB + adb
python treino\importar.py "D:\DCIM\Camera" --rotulo H
python treino\importar.py "D:\dataset"             :: já em subpastas por letra
```

Ele renomeia, converte formatos que o extractor não lê, e avisa quais clipes
ficaram curtos demais.

**Vale gravar pelo celular?** Um pouco, sim. O modelo não enxerga a imagem — ele
enxerga só as **posições dos dedos, já normalizadas** (o cálculo desconta onde a
mão está na tela e o tamanho dela). Por isso resolução e qualidade de câmera quase
não importam. O que muda de verdade é o **ângulo**: webcam fica na altura do
monitor, celular fica na mão. Então: grave a maior parte no PC (é muito mais
rápido) e complete com alguns clipes de celular pra cobrir o ângulo real.

---

## Passo 2 — Treinar

Duplo-clique em **`treino\Treinar.bat`**. Ele faz tudo:

1. lê as fotos e os vídeos, extrai as posições dos dedos com o MediaPipe;
2. treina;
3. **grava o `model.onnx` e o `labels.txt` direto na pasta do app.**

No fim ele confere o modelo gerado e imprime algo assim:

```
validado: entrada 'landmarks_input' [None, 42], saida[1] (1, 21), soma=1.000
```

Essa linha é importante: significa que o arquivo está no formato exato que o app
sabe ler. Se ela não aparecer, **não compile o app** — algo saiu errado.

Também sai uma tabela de acerto por letra. É onde se vê se melhorou.

---

## Passo 3 — Pôr no celular

```bat
cd mobile
.\gradlew.bat assembleDebug
```

O APK sai em `mobile\app\build\outputs\apk\debug\app-debug.apk`. Instale e pronto
— o modelo novo já está dentro dele.

Não existe "passar o modelo pro celular" separado: o `.onnx` é **embutido no APK**
na hora de compilar. Por isso o app funciona sem internet.

---

## Aprimorar UMA letra sem mexer nas outras (modo reforço)

**É o caso mais comum:** o alfabeto já funciona e você só quer melhorar o E, o
F e o G, sem risco de estragar o resto.

Para isso **não se treina o modelo geral** — treinam-se os **modelos
individuais**, que o app testa antes dele.

1. Grave mais amostras **só dessas letras** (`Gravar.bat`)
2. Grave também uns 10 clipes no modo **"nada"** ← importante, veja abaixo
3. Dois cliques em **`Reforcar.bat`**. Ele pergunta quais letras; você digita
   `E,F,G` e dá ENTER.

> Quem preferir linha de comando: `Treinar.bat --reforcar E,F,G` faz o mesmo.
> O `Reforcar.bat` é só uma casca que pergunta as letras e chama isso.

O modelo geral e o `labels.txt` **não são tocados**. As outras 18 letras
continuam exatamente como estavam.

### Por que gravar "Nada" muda tudo

Cada modelo individual é uma pergunta de sim/não: *"isto é um E?"*. Para
aprender a dizer **não**, ele precisa ver exemplos do que não é E.

Se os únicos "não" que ele vir forem **outras letras**, ele nunca aprendeu como
é uma mão que não está sinalizando — e no app responde "é E!" para você coçando
a cabeça. É exatamente a causa de "o app reconhece letra sem eu estar fazendo
letra".

No `Gravar.bat`, aperte **TAB** até o modo dizer `nada` e grave 10–15 clipes de
você mexendo a mão à toa: ajeitando o cabelo, gesticulando como quem fala,
coçando a cabeça. **A mão precisa aparecer** — ela só não pode estar fazendo uma
letra.

É a melhoria mais barata que existe neste pipeline.

---

## A pegadinha dos modelos individuais

O app guarda, além do modelo geral, **modelos separados por letra** em
`letras_dinamicas\H\`, `letras_dinamicas\J\` etc. E ele testa os individuais
**antes** do geral ([LetraEngine.kt:209-219](mobile/app/src/main/java/com/visuall/app/libras/LetraEngine.kt#L209-L219)).

Consequência prática: se você treinar o modelo geral e existir um individual
antigo do H, **o H vai continuar usando o antigo** — parece que o treino não fez
efeito.

Hoje existem individuais para: **H, J, K, Z**. Para o modelo novo valer nessas
letras:

```bat
python treino\exportar_onnx.py --limpar-individuais
```

O `Treinar.bat` repassa as opções, então `Treinar.bat --limpar-individuais`
também funciona. Sem a opção, ele só **avisa** que existem individuais e deixa
como está.

---

## Por que isto funciona (o contrato com o app)

O app espera um `.onnx` com quatro características exatas. O
[treino/exportar_onnx.py](treino/exportar_onnx.py) garante as quatro e **testa o
arquivo gerado** antes de dar por concluído:

| # | Exigência | Onde está escrito no app |
|---|---|---|
| 1 | entrada chamada `landmarks_input` | [LetraEngine.kt:237](mobile/app/src/main/java/com/visuall/app/libras/LetraEngine.kt#L237) |
| 2 | shape `[1,42]` (parada) / `[1,420]` (movimento), float32 | [LibrasAnalyzer.kt:124-125](mobile/app/src/main/java/com/visuall/app/libras/LibrasAnalyzer.kt#L124-L125) |
| 3 | exportado com `zipmap=False` — o app lê a saída de índice `[1]` | [LetraEngine.kt:230-241](mobile/app/src/main/java/com/visuall/app/libras/LetraEngine.kt#L230-L241) |
| 4 | `labels.txt` na mesma ordem das probabilidades | [LetraEngine.kt:25-33](mobile/app/src/main/java/com/visuall/app/libras/LetraEngine.kt#L25-L33) |

E a peça que faz o treino do PC servir pro celular: a conta que normaliza os
dedos é **a mesma nos dois lados** — `normalize_landmarks()` no Python e
`LibrasMath.normalizeLandmarks()` no Kotlin ([LibrasMath.kt:23-33](mobile/app/src/main/java/com/visuall/app/libras/LibrasMath.kt#L23-L33))
fazem exatamente o mesmo cálculo: põem o pulso na origem e dividem pelo maior
valor absoluto.

> **Cuidado ao mexer:** se alguém mudar essa conta em só um dos lados, **nada
> quebra e nenhum erro aparece** — o app só passa a errar mais, porque foi
> ensinado num formato e está sendo usado em outro. O próprio comentário no topo
> do `LibrasMath.kt` avisa sobre isso.

---

## Problemas comuns

**"Nenhuma amostra extraída"** — a mão saiu do quadro no meio dos clipes. Regrave
mantendo ela visível o tempo todo.

**Treinei mas o app não mudou** — três suspeitos, nesta ordem: (1) não recompilou
o APK; (2) existe modelo individual daquela letra (veja a seção acima); (3) a
linha `validado:` não apareceu no fim do treino.

**"só ha 1 classe com dados"** — o modelo geral precisa de pelo menos 2 letras pra
ter o que distinguir. Grave outra letra.

**A câmera não abre** — feche Zoom/Teams/OBS. Com mais de uma câmera, mude
`CAMERA_INDEX` no topo de [treino/gravar.py](treino/gravar.py).

**Gravei errado** — apague o arquivo em `data\raw_videos\<LETRA>\` (os nomes têm
data e hora).

**Preciso regravar tudo sempre?** Não. Gravar só acrescenta; treinar usa tudo que
já existe.

**Ao commitar:** os `.onnx` são arquivos grandes e o repo usa Git LFS pra eles
(`mobile/.gitattributes`). Já está configurado — só não converta em texto.

---

## Arquivos

**Os três de dois cliques:**

| Arquivo | Quando usar |
|---|---|
| [treino/Gravar.bat](treino/Gravar.bat) | **sempre** — gravar amostras |
| [treino/Reforcar.bat](treino/Reforcar.bat) | o alfabeto já funciona, quero **melhorar algumas letras** |
| [treino/Treinar.bat](treino/Treinar.bat) | gravei **todas** as letras, quero **refazer o modelo do zero** |

Os de dentro (não precisa abrir):

| Arquivo | O que é |
|---|---|
| [treino/gravar.py](treino/gravar.py) | a janela de gravação (contador de 3s) |
| [treino/exportar_onnx.py](treino/exportar_onnx.py) | o motor: treina e gera o `.onnx` no contrato do app |
| [treino/extrair_negativos.py](treino/extrair_negativos.py) | lê os clipes de "nada" |
| [treino/importar.py](treino/importar.py) | traz vídeos do celular pra pasta certa |

Os extractors reaproveitados ficam em
[linear/backend/data_extraction/](linear/backend/data_extraction/).

> **Não faz parte deste fluxo:** `linear/backend/app.py`, `modular/` e os `.pkl`
> em `models/` são a versão de PC (backend Python + página web), que usa outro
> formato de modelo e não tem relação com o app do celular.

---

# Gestos corporais (AJUDAR, COMPUTADOR, CONVERSAR, PESSOA, SURDO)

Funciona parecido, mas com **uma diferença que muda tudo**.

## A diferença: não existe modelo individual

Nas letras dá para reforçar só o E, porque cada letra tem seu modelinho próprio.
**Nos gestos não existe isso** — o app carrega um único
`gestos/geral/model.tflite` ([BodyGestureEngine.kt:87](mobile/app/src/main/java/com/visuall/app/libras/BodyGestureEngine.kt#L87)),
e o `labels.txt` dele é a lista completa.

Consequência: **treinar substitui todos os gestos de uma vez.** Não dá para
reforçar só o SURDO — se você gravar só ele, o app esquece os outros cinco.

Então, para reforçar, é preciso gravar **todos os 6**:

```
AJUDAR   COMPUTADOR   CONVERSAR   NEUTRO   PESSOA   SURDO
```

**O NEUTRO não é opcional.** Ele é o "estou parado, não estou sinalizando" —
sem ele o app vê sinal em qualquer movimento. Grave você parado na frente da
câmera, respirando normal, talvez ajeitando a roupa.

O script barra sozinho se faltar algum e diz quais são.

## Como gravar

1. `Gravar.bat` → **TAB** até o modo dizer `corpo`
2. **Corpo inteiro visível** — afaste-se da câmera. Ombros, braços e as duas
   mãos precisam aparecer (a conta de normalização usa a distância entre os
   ombros; sem eles o quadro é descartado)
3. **15 a 20 clipes de cada** um dos 6. A gravação aqui é de **4 segundos**
   (gesto de corpo começa e termina mais devagar que uma letra)

## Como treinar

Dois cliques em **`TreinarCorpo.bat`**.

Duas avisos honestos:

- **Na primeira vez ele baixa o TensorFlow (~600 MB).** É o único treino que
  precisa disso — as letras não usam.
- **Demora.** Cada quadro passa pelo detector de corpo *e* pelo de mãos. Com
  120 clipes, são vários minutos.

No fim procure a linha:

```
validado: entrada [1, 30, 225], saída [1, 6], soma=1.000
```

Depois: Android Studio → **Run**, como sempre.

## Como funciona por dentro

Cada quadro vira **225 números** = 75 pontos × (x, y, z):

| Pontos | O quê |
|---|---|
| 0–32 | corpo (ombros, cotovelos, quadril…) |
| 33–53 | mão **esquerda** |
| 54–74 | mão **direita** |

O gesto inteiro é reduzido a **30 quadros** e vai para uma rede LSTM, que
aprende a *sequência* do movimento — não uma pose isolada.

As duas contas que precisam bater com o app são
`normaliza_corpo()` ↔ `LibrasMath.normalizeBodyFrame` e
`reamostra()` ↔ `LibrasMath.resample`. Estão verificadas como idênticas; se
alguém mexer numa só, o app não dá erro — só passa a errar mais.
