import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Esta parte define a raiz do projeto, fica basicamente assim:
# (sistemaOperacional.caminho.nomeDaPasta(sistemaOperacional.caminho.caminhoAbsoluto(__file__)))
# 
#   __file__ é uma variável do python que significa "Este arquivo que está aberto"
#      
#   abspath(X) é a função para encontrar o caminho absoluto, ou seja, desde o C:\ até o arquivo X
#   
#   Logo --> os.path.abspath(__file__) chama o caminho absoluto de C:\ até o arquivo aberto
#   
#   e como dirname() é a função que pega apenas a pasta do arquivo, a linha completa pega o nome
#   o caminho até a pasta do arquivo, sem selecionar ele.

def P(nome):
    return os.path.join(BASE_DIR, nome)
#   Já esta função faz o caminho contrário, ela pega o nome inserido em P(nome) e junta com o
#   Caminho da pasta, dando liberdade para selecionar qualquer arquivo dentro do endereço raiz

#################################################################################################

#TUDO ISTO AQUI É DEFINIÇÃO DE VARIÁVEIS QUE SERÃO USADAS EM OUTROS ARQUIVOS***
# Alfabeto
JANELA_MLP = 10 # Olha 10 frames, ou seja, captura movimento em 10 frames, nem mais nem menos
CONFIANCA_MINIMA = 0.90 # Aceita no mínimo 90% de similaridade com o movimento treinado (0%-100%)
LIMIAR_MOVIMENTO = 0.30 # compara movimento entre os pontos
TEMPO_PRA_LIMPAR = 3.0 # Tempo para que o sinal de apagar historico seja considerado (3s)

#///////////////////////////////////////////////////////////////////////////////////////////////#

# Corpo
N_POSE, N_HAND = 33, 21 # Pontos do media pipe, 33 pontos para o corpo e 21 para as mãos
N_FEATURES = (N_POSE + 2 * N_HAND) * 3 # cada ponto vale 3 (X,Y,Z), ele aceita 33 do corpo e
#--------------------------------------# 42 das mãos (21 cada), vezes 3 que da 225 (75x,75y,75z)
JANELA_MODELO = 30 # Modelo corporal precisa de mais frames, logo pede 30 por gesto
LIMIAR_INICIO = 0.050 # Acima deste nivel de movimento, começa a captura
LIMIAR_FIM = 0.030 #--# Abaixo deste, para, logo a zona morta é entre os dois, onde nada muda
FRAMES_INICIO = 3 # Para começar, precisa de 3 frames ACIMA do limiar de inicio
FRAMES_FIM = 5 #--# Para terminar, precisa de 5 frames ABAIXO do limiar de fim
MIN_FRAMES_GESTO = 10 # Gesto curto é ignorado (menos de 10 frames)
MAX_FRAMES_GESTO = 60 # Gesto longo também (Mais de 60 frames)
CONFIANCA_CORPO = 0.85 # No mínimo 85% de confiança, ja que os do corpo são mais dificeis
COOLDOWN_CORPO = 2.0 # Depois de um movimento, espera 2 segundos para poder captar outro
CLASSE_NEUTRO = "NEUTRO" # Entende como "Neutro" movimentos inuteis
JANELA_MOV = 5 # Calcula a movimentação dos ultimos 5 frames
TEMPO_LIMPAR_CORPO = 5.0 #5 segundos de mão aberta para limpar, mais tempo do que no alfabeto

#///////////////////////////////////////////////////////////////////////////////////////////////#

# Rosto
LIMIAR_SOBRANCELHA = 0.38 # Se a sombrancelha estiver mais que 38% levantada, vira pergunta
JANELA_SOBR = 5 # Precisa manter 5 frames com a sombrancelha levantada, para evitar erros
IDX_BROW_L, IDX_BROW_R = 105, 334        #--v
IDX_EYE_TOP_L, IDX_EYE_TOP_R = 159, 386     # Apenas pontos de captura do rosto
IDX_EYE_OUT_L, IDX_EYE_OUT_R = 33, 263   #--^

# TUDO ISTO AQUI É DEFINIÇÃO DE VARIÁVEIS QUE SERÃO USADAS EM OUTROS ARQUIVOS***