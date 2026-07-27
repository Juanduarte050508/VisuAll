# Modelos pre-treinados

Esta pasta contem os modelos prontos para inferencia do backend Python. Os
mesmos scripts de treino tambem exportam os modelos usados pelo app Android em
`mobile/app/src/main/assets/`.

| Arquivo | Tamanho | Descrição |
|---|---|---|
| `static_model.pkl` | ~570KB | MLP para letras estáticas (A, B, C, D, E, F, G, I, L, M, N, O, P, Q, R, S, T, U, V, W, Y) |
| `static_classes.pkl` | <1KB | Mapeamento `idx → letra` do modelo estático |
| `dynamic_model.pkl` | ~1.7MB | MLP para letras dinâmicas / com movimento (H, J, K, X, Z) |
| `dynamic_classes.pkl` | <1KB | Mapeamento `idx → letra` do modelo dinâmico |

Para retreinar do zero, veja `linear/backend/training/`.

Saidas Android geradas pelos treinos:

| Arquivo | Descricao |
|---|---|
| `mobile/app/src/main/assets/static_model.onnx` | Modelo estatico com entrada `landmarks_input`, shape `[1, 42]` |
| `mobile/app/src/main/assets/static_labels.txt` | Labels estaticos na mesma ordem da saida do ONNX |
| `mobile/app/src/main/assets/dynamic_model.onnx` | Modelo dinamico com entrada `landmarks_input`, shape `[1, 420]` |
| `mobile/app/src/main/assets/dynamic_labels.txt` | Labels dinamicos na mesma ordem da saida do ONNX |

> **Nota:** modelos legados ou versões antigas devem ir para uma subpasta `legacy/` (presente no `.gitignore`).
