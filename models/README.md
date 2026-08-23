# Modelos pre-treinados

Esta pasta guarda os modelos em formato Python (`.pkl`), gerados como
subproduto do treino. **Eles nao vao mais para o Git** (ver `.gitignore`):
sao arquivos pesados que o app nao usa e cujo conteudo ja esta duplicado nos
`.onnx`/`.tflite` que vao para `mobile/app/src/main/assets/`. Antes eram
versionados a forca, e cada retreino guardava uma copia nova no historico
para sempre — o repositorio so crescia.

Se voce clonou o repositorio e esta pasta esta vazia, esta tudo certo: rode
o treino (`treino/Treinar.bat`) e os `.pkl` reaparecem localmente. O que
o app precisa para funcionar esta em `assets/`, e esses sim sao versionados.

| Arquivo (local, nao versionado) | Tamanho | Descricao |
|---|---|---|
| `static_model.pkl` | ~570KB | MLP para letras estaticas (A, B, C, D, E, F, G, I, L, M, N, O, P, Q, R, S, T, U, V, W, Y) |
| `static_classes.pkl` | <1KB | Mapeamento `idx → letra` do modelo estatico |
| `dynamic_model.pkl` | ~1.7MB | MLP para letras dinamicas / com movimento (H, J, K, X, Z) |
| `dynamic_classes.pkl` | <1KB | Mapeamento `idx → letra` do modelo dinamico |

Para treinar ou retreinar, veja `treino/README.md`.

Saidas Android geradas pelos treinos (estas sim versionadas, o app carrega
direto):

| Arquivo | Descricao |
|---|---|
| `mobile/app/src/main/assets/letras_estaticas/geral/model.onnx` | Modelo estatico com entrada `landmarks_input`, shape `[1, 42]` |
| `mobile/app/src/main/assets/letras_estaticas/geral/labels.txt` | Labels estaticos na mesma ordem da saida do ONNX |
| `mobile/app/src/main/assets/letras_dinamicas/geral/model.onnx` | Modelo dinamico com entrada `landmarks_input`, shape `[1, 420]` |
| `mobile/app/src/main/assets/letras_dinamicas/geral/labels.txt` | Labels dinamicos na mesma ordem da saida do ONNX |
| `mobile/app/src/main/assets/gestos/geral/model.tflite` | Modelo de gestos corporais, entrada `[1, 30, 225]` |
| `mobile/app/src/main/assets/gestos/geral/labels.txt` | Labels de gestos na mesma ordem da saida do TFLite |

Modelos individuais (um por letra, tentados antes do geral) ficam em:

- `mobile/app/src/main/assets/letras_dinamicas/<LETRA>/`
- `mobile/app/src/main/assets/letras_estaticas/<LETRA>/`

A pasta `letras_dinamicas/parcial/` ainda recebe copia do treino, mas **o app
nao le mais ela** — o fallback foi removido do `LetraEngine.kt` no commit
`1f33768`. Serve so pra inspecao manual.

O treino verifica sozinho cada modelo exportado (nome e formato da entrada)
antes de considerar o export valido — ver `valida` em
`treino/exportar_onnx.py` (letras) e `treino/treinar_corpo.py` (gestos). A
verificacao das letras roda o modelo de verdade, entao precisa do
`onnxruntime` instalado; sem ele o treino avisa e segue sem validar.

> **Nota:** modelos legados ou versoes antigas devem ir para uma subpasta
> `legacy/` (presente no `.gitignore`).
