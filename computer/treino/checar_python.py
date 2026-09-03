import sys


versao = sys.version_info[:2]
if (3, 10) <= versao <= (3, 12):
    raise SystemExit(0)

print()
print("ERRO: o treino do VisuAll precisa de Python 3.10, 3.11 ou 3.12.")
print("A versao chamada por este atalho foi: %s.%s" % versao)
print()
print("Motivo: MediaPipe, TensorFlow e ONNX precisam resolver juntos;")
print("Python 3.13/3.14 ainda nao e a faixa testada por este projeto.")
print()
print("Instale Python 3.11 e deixe o comando 'python' apontar para ele,")
print("ou rode os scripts manualmente com o executavel correto.")
raise SystemExit(1)
