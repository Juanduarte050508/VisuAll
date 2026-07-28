# Como usar o treinamento

## Abrir a interface

Dê dois cliques em:

```text
VisuAll/treinamento/abrir_treinamento.bat
```

Se voce ja estiver dentro da pasta `VisuAll`, rode:

```powershell
python .\treinamento\interface_treinamento.py
```

## Organizar os arquivos

Crie uma pasta com subpastas usando o nome da letra ou gesto:

```text
dataset_libras/
  A/
    foto1.jpg
  H/
    movimento1.mp4
  AJUDAR/
    gesto1.mp4
```

Use:

- fotos para letras paradas;
- videos para letras com movimento;
- videos para gestos corporais.

Ao treinar um grupo, todos os labels daquele grupo precisam ter amostras. Por
exemplo, letras dinamicas precisam de `H`, `J`, `K`, `X` e `Z`; se faltar uma,
o app nao e sobrescrito.

Mesmo assim, o que existir e treinado como modelo parcial e salvo em:

```text
treinamento/modelos_parciais/
```

O programa mantem os 5 parciais mais recentes de cada tipo e apaga os mais
antigos automaticamente.

Cada label com dados tambem gera um modelo individual separado em:

```text
treinamento/modelos_individuais/
```

Exemplo: se existem videos de `H`, `J`, `K` e `Z`, cada um ganha seu proprio
modelo. O app tenta esses modelos individuais primeiro e usa o modelo geral
como fallback.

## Na interface

1. Em `Letra ou gesto`, deixe `AUTO` se a pasta ja estiver separada por labels.
2. Clique em `Selecionar pasta`.
3. Escolha `Importar + Extrair + Treinar`.

Os modelos treinados entram automaticamente em:

```text
mobile/app/src/main/assets/letras_estaticas/
mobile/app/src/main/assets/letras_dinamicas/
mobile/app/src/main/assets/gestos/
```

Dentro de cada categoria, `geral` guarda o modelo completo, `parcial` guarda
um treino incompleto quando faltar algum label, e as pastas por letra/gesto
guardam modelos individuais quando existirem.

As fotos e videos importados ficam somente em:

```text
treinamento/dados/
```

Essa pasta de dados e os modelos temporarios ficam ignorados pelo Git para nao
subir videos e arquivos pesados no GitHub.

