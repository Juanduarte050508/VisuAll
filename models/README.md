# Modelos pré-treinados

Esta pasta contém os modelos prontos para inferência. Ambos são `MLPClassifier` do scikit-learn.

| Arquivo | Tamanho | Descrição |
|---|---|---|
| `static_model.pkl` | ~570KB | MLP para letras estáticas (A, B, C, D, E, F, G, I, L, M, N, O, P, Q, R, S, T, U, V, W, Y) |
| `static_classes.pkl` | <1KB | Mapeamento `idx → letra` do modelo estático |
| `dynamic_model.pkl` | ~1.7MB | MLP para letras dinâmicas / com movimento (H, J, K, X, Z) |
| `dynamic_classes.pkl` | <1KB | Mapeamento `idx → letra` do modelo dinâmico |

Para retreinar do zero, veja `backend/training/`.

> **Nota:** modelos legados ou versões antigas devem ir para uma subpasta `legacy/` (presente no `.gitignore`).
