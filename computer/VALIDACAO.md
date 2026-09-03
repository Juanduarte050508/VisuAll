# Roteiro de validação em celular real

O `CHANGELOG.md` tem hoje quase dez ajustes de limiar marcados como
"pendente: validar em celular real". Cada um foi mexido por tentativa e
erro, em cima do anterior, sem ninguém confirmar se o de antes ajudou ou
atrapalhou. Isso é ruim de duas formas: não dá pra saber qual mudança
resolveu o quê, e é fácil "consertar" um problema reintroduzindo outro que
já tinha sido resolvido.

Este roteiro é curto de propósito (~10 minutos). O objetivo não é medir
acurácia científica — é ter **o mesmo teste, feito do mesmo jeito, antes e
depois de cada mudança**, pra a conversa deixar de ser "achei que melhorou"
e virar "de 26 letras, acertava 18 e agora acerta 22".

## Regras que fazem o número valer alguma coisa

1. **Mesma pessoa** faz o teste antes e depois. Sinalizadores diferentes têm
   mãos e ritmos diferentes; trocar de pessoa no meio invalida a comparação.
2. **Mesmo celular**, mesma iluminação, mesma distância da câmera (~50cm,
   tronco visível).
3. **Sempre o alfabeto inteiro, na ordem A→Z.** Não pule as letras difíceis
   — são justamente elas que dizem se a mudança prestou.
4. **Uma tentativa por letra.** Se errar, anota erro e segue. Repetir até
   acertar transforma o teste em treino e o número perde o sentido.
5. Anote o **commit** que estava instalado (`git rev-parse --short HEAD`).

## Parte 1 — Alfabeto (26 letras)

Faça cada letra e espere o app decidir. Marque:

- **OK** — apareceu a letra certa.
- **ERRO: X** — apareceu outra letra (anote qual).
- **NADA** — não reconheceu nada dentro de ~5 segundos.

| Letra | Resultado | Letra | Resultado |
|---|---|---|---|
| A |  | N |  |
| B |  | O |  |
| C |  | P |  |
| D |  | Q |  |
| E |  | R |  |
| F |  | S |  |
| G |  | T |  |
| H |  | U |  |
| I |  | V |  |
| J |  | W |  |
| K |  | X |  |
| L |  | Y |  |
| M |  | Z |  |

**Atenção especial** às letras que historicamente se confundem:
`E / I / U`, `F / G`, `P / Q`, `T / V / W / Y`. E às com movimento:
`H, J, K, X, Z` (só o J vinha funcionando bem).

## Parte 2 — Falso positivo (o teste que estava faltando)

Este é o que corresponde à reclamação de "reconhece fácil demais". Sem ele,
dá pra "melhorar" o alfabeto só afrouxando tudo e deixando o app chutar.

Durante **30 segundos**, fique na frente da câmera com a mão à mostra
**sem fazer nenhuma letra**: coce a cabeça, ajeite o cabelo, gesticule como
quem está falando, mexa a mão à toa.

- Quantas letras apareceram na frase? → **______**
- O ideal é **0**. Qualquer número acima de 2 significa que os limiares
  estão frouxos demais, mesmo que a Parte 1 tenha ido bem.

## Parte 3 — Gestos corporais

Faça cada sinal uma vez, do jeito normal:

| Gesto | Reconheceu? | Demorou quanto? |
|---|---|---|
| PESSOA |  |  |
| SURDO |  |  |
| CONVERSAR |  |  |
| COMPUTADOR |  |  |
| AJUDAR |  |  |

E o equivalente à Parte 2: fique **30 segundos parado, sem sinalizar**, e
anote quantos gestos apareceram sozinhos → **______** (ideal: 0).

## Como anotar o resultado

Some e registre no `CHANGELOG.md`, junto da entrada da mudança que você
está validando:

```
Validado em <data>, <celular>, commit <hash>, por <nome>:
  Alfabeto:        __/26 corretas
  Falso positivo:  __ letras em 30s parado
  Gestos:          __/5 reconhecidos
  Falso positivo (corpo): __ gestos em 30s parado
```

Se o resultado for pior que o teste anterior, **reverta a mudança** em vez
de empilhar mais um ajuste em cima. Foi exatamente essa pilha que criou a
situação atual, com valores que ninguém sabe mais se ajudam.

## Ordem sugerida quando algo vai mal

- **Alfabeto ruim E falso positivo alto** → o problema quase certamente é
  falta de dados de treino, não limiar. Grave mais amostras
  (`computer/treino/README.md`, a partir da raiz) antes de mexer em qualquer constante.
- **Alfabeto bom, falso positivo alto** → limiares frouxos. Suba
  `CONFIANCA_MINIMA` / `CONFIANCA_DINAMICA` / `CONFIANCA_INDIVIDUAL` em
  `LibrasAnalyzer.kt`, um por vez.
- **Falso positivo 0, alfabeto ruim** → limiares apertados demais. Desça os
  mesmos valores, um por vez.
- **Letras com movimento (H/K/X/Z) falhando, estáticas OK** → é o caminho
  do modelo dinâmico. Veja `LIMIAR_MOVIMENTO` e `MOVIMENTO_SUSTENTADO_MS`,
  e confira quantos clipes dessas letras existem em `data/raw_videos/`.
