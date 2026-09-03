# Treino do VisuAll — guia rápido

Tudo aqui é **duplo-clique**. Nenhum comando digitado.

---

## O ciclo, sempre igual

```
1. GRAVAR      Gravar.bat        você faz os sinais na webcam
2. TREINAR     (um dos 3 .bat)   o PC vira os vídeos em modelo
3. INSTALAR    Android Studio    o modelo vai DENTRO do app
```

**Sem o passo 3 nada muda no celular.** O modelo não é um arquivo que se
copia — ele é embutido no app quando você compila.

Depois de instalado, o celular funciona sozinho: sem PC, sem internet.

---

## Qual .bat usar?

| Quero… | Uso | Apaga algo? |
|---|---|---|
| Gravar amostras | **`Gravar.bat`** | não |
| Melhorar **algumas letras** | **`Reforcar.bat`** | **não** ✅ |
| Melhorar **gestos corporais** | **`TreinarCorpo.bat`** | substitui (com backup) |
| Voltar um modelo que piorou | **`RestaurarModelo.bat`** | — |
| Refazer o **alfabeto inteiro** do zero | `Treinar.bat` | ⚠️ sim |

> Na dúvida entre `Reforcar.bat` e `Treinar.bat`, use **`Reforcar.bat`**.
> Ele nunca apaga o que já funciona.

---

## 1. Gravar

Duplo-clique em **`Gravar.bat`**.

| Tecla | O que faz |
|---|---|
| **ESPAÇO** | grava (3s de contagem, depois grava sozinho) |
| **R** | apaga a última gravação — errou? aperta R e refaz |
| **N** / **P** | próxima / anterior letra |
| **TAB** | troca o modo |
| **Q** | sai |

Ele **fica na mesma letra**. Grave 15 seguidas, depois **N** para a próxima.

### Os 4 modos (TAB alterna)

| Modo | O que gravar | Salva |
|---|---|---|
| `estatica` | letras paradas: A B C D E F G I L M N O P Q R S T U V W Y | fotos |
| `dinamica` | letras com movimento: H J K X Z | vídeo |
| `corpo` | AJUDAR COMPUTADOR CONVERSAR NEUTRO PESSOA SURDO | vídeo (4s) |
| `nada` | mão à toa, coçando a cabeça, gesticulando | vídeo |

### Quanto gravar

| | Quantidade |
|---|---|
| Letra parada | 3 a 5 gravações |
| Letra com movimento | 15 a 20 |
| Gesto corporal | 15 a 20 |
| "nada" | 10 a 15 (uma vez só serve pra sempre) |

### A regra que mais estraga treino

**A mão não pode sair da tela durante a gravação.** Se sair, aquele clipe é
jogado fora inteiro — e você não é avisado na hora.

No modo `corpo`: **afaste-se** até aparecerem ombros, braços e as duas mãos.

### Pra que serve o modo "nada"

É o que ensina o app a **não** reconhecer letra quando você não está fazendo
letra. Se o app fica "vendo letras" enquanto você mexe a mão à toa, é disto
que ele precisa. Grave 10-15 e nunca mais precisa.

---

## 2. Treinar

### Letras → `Reforcar.bat`

Duplo-clique. Ele pergunta:

```
   Letras: _
```

Digite as que você gravou, separadas por vírgula — `H,J,K,X,Z` — e ENTER.

**Não apaga nada.** Cria um modelo extra para cada letra, que o app testa
antes do modelo geral. As outras letras continuam iguais.

### Gestos corporais → `TreinarCorpo.bat`

Duplo-clique, sem perguntar nada.

Você **não precisa** gravar os 6. Os que faltarem são preservados
automaticamente a partir do modelo que já está no app.

Só saiba: quanto mais gestos você gravar **de verdade**, melhor o resultado.
Os preservados são uma aproximação — servem pra não esquecer, não pra
melhorar.

Na primeira vez ele baixa o TensorFlow (~600 MB). Demora. É só nessa vez.

### Como saber se deu certo

Procure esta linha no fim:

```
validado: entrada [1, 30, 225], saída [1, 6], soma=1.000
```

**Se ela não aparecer, não compile o app.** Algo deu errado.

Também sai uma nota por letra/gesto. Nota baixa = precisa de mais gravações
daquele.

---

## 3. Instalar no celular

1. Abra o **Android Studio**
2. **File → Open** → escolha a pasta **`mobile`** (não a pasta de cima)
3. Espere carregar (a primeira vez demora bastante)
4. Ligue o celular no cabo, com **depuração USB** ativada
5. Clique em **▶ Run**

Pronto. Pode tirar o cabo.

---

## Deu ruim?

| Problema | O que fazer |
|---|---|
| Ficou pior que antes | **`RestaurarModelo.bat`** → escolhe o número → recompila |
| Piorou só uma letra | apague a pasta dela em `mobile\app\src\main\assets\letras_estaticas\<LETRA>\` |
| "Nenhuma amostra extraída" | a mão saiu da tela nos clipes. Regrave |
| Teclas não respondem | clique na janela do vídeo primeiro (ela precisa estar em foco) |
| Câmera não abre | feche Zoom / Teams / OBS |
| Treinei mas o app não mudou | você recompilou? (passo 3) |
| Gravei errado | apague o arquivo em `computer\data\raw_videos\<LETRA>\` (o nome tem data e hora) |

---

## Coisas boas de saber

**Nunca precisa regravar tudo.** Gravar só acrescenta. Cada treino usa tudo
que ja existe na pasta `computer\data\`, inclusive o que você gravou semanas atrás.

**Pode parar no meio.** Grave 3 letras hoje, 3 amanhã. Só rode o treino
quando quiser.

**Os vídeos ficam no PC.** Nunca vão pro celular. Só o modelo vai — e ele
tem poucos KB.

**Não apague a pasta `computer\data\`.** É o seu acervo. Sem ela, não dá pra
retreinar.

---

## Onde as coisas ficam

```
computer\data\raw_images\<LETRA>\        fotos das letras paradas
computer\data\raw_videos\<LETRA>\        videos das letras com movimento
computer\data\raw_body_videos\<GESTO>\   videos dos gestos corporais
computer\data\raw_negativos\NADA\        videos de "nao e sinal nenhum"

mobile\app\src\main\assets\     os modelos que o app usa
computer\treino\modelos_anteriores\      backups automaticos (o RestaurarModelo usa)
```

Detalhes técnicos (formatos, contrato com o app, como funciona por dentro)
estao em **`computer\TREINO.md`**.
