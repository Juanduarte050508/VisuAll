# Ferramentas de emulação — VisuAll mobile

## `abrir-emulador-px4.bat` (Windows)

Abre o emulador **Pixel_4** para testar o app. Na primeira vez, se a AVD
não existir, ele **cria sozinho**: baixa a imagem `x86` (32-bit), cria a
AVD `Pixel_4` e já liga a webcam na câmera frontal.

### Como usar
1. `git pull`
2. Abra o Android Studio pelo menos uma vez (para o SDK ficar pronto).
3. Duplo clique em `mobile/tools/abrir-emulador-px4.bat`.
4. Espere o Android bootar (~15s).
5. No Android Studio, aperte **Run** para instalar e abrir o VisuAll.

### Por que imagem x86 (32-bit) e não x86_64?
O MediaPipe não tem biblioteca `x86_64`. Numa AVD x86_64 o app roda em
`arm64` **traduzido** e fica lento. Na imagem `x86` de 32-bit tudo roda
**nativo** no processador Intel/AMD — rápido, sem travar.

### Requisitos
- Android Studio / Android SDK instalado.
- **Android SDK Command-line Tools** (Settings → Android SDK → SDK Tools)
  — necessário só se a AVD ainda não existir, para o script poder criá-la.
- Uma webcam (para o modo Libras enxergar as mãos).

> Se sua webcam estiver sendo usada por Discord/OBS/etc., feche esses
> programas antes: o emulador não compartilha a câmera e a imagem chega
> corrompida.
