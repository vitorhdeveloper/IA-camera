# IA-Camera-Python

Projeto de visão computacional que utiliza IA para detectar objetos em tempo real através da câmera selecionada.

Foi desenvolvido em **Python 3**, usando **OpenCV** para captura de vídeo e **YOLOv8 (Ultralytics)** para identificação dos objetos. O sistema processa cada frame da câmera e desenha na tela as detecções com nome.

Para encerrar a aplicação, pressione **ESC**.

## Instalação

Instale as dependências:

```bash id="d4e5f6"
pip install -r requirements.txt
```

## Como usar

Execute o arquivo:

```bash id="g7h8i9"
python main.py
```

## O que o projeto faz

* Acessa a webcam do computador
* Detecta objetos em tempo real
* Exibe caixas ao redor dos objetos
* Mostra o nome e a confiança da detecção

**Desenvolvido por Vitor Hugo**
