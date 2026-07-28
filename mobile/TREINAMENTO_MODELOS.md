# Treinamento dos modelos

> Pra gravar amostras novas com a webcam e treinar sem mexer em código, veja
> `treinamento/README.md` na raiz do repo. Este arquivo aqui documenta o
> formato/requisitos dos modelos em si.

Este app usa dois tipos de reconhecimento:

- **Estatico:** uma mao parada, com `42` features por amostra.
- **Dinamico:** movimento curto da mao, com janela de `10` frames e `420` features por amostra.

Os modelos ativos ficam em:

```text
app/src/main/assets/letras_estaticas/geral/model.onnx
app/src/main/assets/letras_estaticas/geral/labels.txt
app/src/main/assets/letras_dinamicas/geral/model.onnx
app/src/main/assets/letras_dinamicas/geral/labels.txt
```

(além do modelo "geral" acima, `LetraEngine.kt` também pode carregar modelos
individuais por letra em `letras_estaticas/<LETRA>/` e
`letras_dinamicas/<LETRA>/`, e um modelo parcial em
`letras_dinamicas/parcial/` — usados antes do geral quando existem, ver
`treinamento/COMO_USAR.md`.)

## 1. Coletar amostras no celular

1. Abra o app em um celular real.
2. Entre no modo Libras.
3. Toque em calibrar.
4. Para letras estaticas, segure o sinal firme ate completar a barra.
5. Para letras dinamicas (`H`, `J`, `K`, `X`, `Z`), faca o movimento inteiro dentro do quadro.
6. Salve a letra.
7. Repita ate cada letra ter pelo menos `96` amostras.
8. Toque em exportar dados.

O app exporta:

```text
visuall_libras_phone_dataset.csv
visuall_libras_dynamic_phone_dataset.csv
```

## 2. Organizar o dataset

Junte os CSVs exportados por celulares diferentes, mantendo o cabecalho uma unica vez.

Boa meta por classe:

```text
Minimo: 100 amostras por letra
Bom:    300 amostras por letra
Forte:  500+ amostras por letra, com pessoas, luzes e celulares diferentes
```

Distribua as coletas entre camera frontal/traseira, fundo claro/escuro, mao perto/longe e pequenas variacoes de angulo.

## 3. Treinar modelo estatico

Use o CSV estatico para treinar um classificador multiclasse com entrada `42`.

Requisitos do arquivo ONNX final:

```text
Entrada: landmarks_input, shape [1, 42], float32
Saida: probabilidades por letra, na mesma ordem de static_labels.txt
```

Antes de substituir o modelo no app, valide:

```text
Acuracia geral >= 90%
Recall por letra >= 85%
Pouca confusao entre E/I/U/F/G/P/Q/T/V/W/Y
```

Se uma letra ficar fraca, colete mais exemplos dela antes de reduzir limiares no codigo.

## 4. Treinar modelo dinamico

Use o CSV dinamico para treinar com entrada `420`.

Cada linha representa uma janela de `10` frames:

```text
10 frames x 42 features = 420 features
```

Requisitos do arquivo ONNX final:

```text
Entrada: landmarks_input, shape [1, 420], float32
Saida: probabilidades por letra, na mesma ordem de dynamic_labels.txt
```

Valide especialmente `H`, `J`, `K`, `X` e `Z`, porque elas dependem de movimento. O modelo dinamico so deve aceitar uma classe quando a confianca e a margem contra a segunda opcao forem boas.

## 5. Aplicar no app

1. Substitua os arquivos em `app/src/main/assets/`.
2. Confira se os labels estao na mesma ordem usada no treino.
3. Rode:

```powershell
.\gradlew.bat assembleDebug
```

4. Instale o APK gerado em:

```text
app/build/outputs/apk/debug/app-debug.apk
```

## 6. Teste final

Teste cada letra em tempo real e anote:

- letras que confundem com outras;
- letras que so funcionam em um angulo;
- letras dinamicas que precisam de movimento exagerado;
- falsos positivos quando a mao esta parada.

Priorize melhorar dados antes de alterar thresholds. Mais dados variados normalmente aumentam precisao sem deixar o app instavel.
