VOCABULARIO = {
    "PESSOA": {"tipo": "subst", "genero": "f", "palavra": "pessoa", "artigo": "a"},
    "SURDO": {"tipo": "adj", "masc": "surdo", "fem": "surda"},
    "CONVERSAR": {"tipo": "verbo", "conj": "conversa", "inf": "conversar"},
    "COMPUTADOR": {"tipo": "subst", "genero": "m", "palavra": "computador", "artigo": "o"},
    "AJUDAR": {"tipo": "verbo", "conj": "ajuda", "inf": "ajudar"},
}


def traduzir_frase(palavras, interrogativo=False):
    partes, ult_gen, ult_tipo = [], None, None
    ja_houve_verbo = False
    for i, p in enumerate(palavras):
        p = p.upper()
        if p == "NEUTRO":
            continue
        if p not in VOCABULARIO:
            partes.append(p.capitalize())
            ult_gen = "m"
            ult_tipo = "subst"
            continue
        v = VOCABULARIO[p]
        t = v["tipo"]
        if t == "subst":
            art = v["artigo"].capitalize() if i == 0 else v["artigo"]
            partes.append(f"{art} {v['palavra']}")
            ult_gen = v["genero"]
            ult_tipo = "subst"
        elif t == "adj":
            partes.append(v["fem"] if ult_gen == "f" else v["masc"])
            ult_tipo = "adj"
        elif t == "verbo":
            # 1o verbo conjugado; do 2o em diante infinitivo com "a".
            # Olhar so o token anterior deixava "...surda conversa" em
            # COMPUTADOR AJUDAR PESSOA SURDO CONVERSAR.
            partes.append(f"a {v['inf']}" if ja_houve_verbo else v["conj"])
            ja_houve_verbo = True
            ult_tipo = "verbo"
    if not partes:
        return ""
    frase = " ".join(partes)
    frase = frase[0].upper() + frase[1:]
    return frase + ("?" if interrogativo else "")


def montar_exibicao(tokens, palavra_atual, sobr_ativo):
    base = traduzir_frase(list(tokens))
    if palavra_atual:
        base = (base + " " + palavra_atual).strip()
    if sobr_ativo and base.strip() and not base.rstrip().endswith("?"):
        base = base.rstrip() + "?"
    return base
