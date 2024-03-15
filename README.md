# psg-multilabel-classifier

Trabalho de Conclusão de Curso, para graduação em Sistemas de Informação pela EACH-USP. Este projeto tem como foco o desenvolvimento de classificadores multirrótulo inéditos.

## Setup

Este programa foi desenvolvido para a versão **3.9.7** do Python. **É possível que novas versões tenham conflitos com as dependências**, já que alguns pacotes usados aqui não possuem suporte para versões mais recentes do Python. Você pode tentar utilizar outras versões do Python, mas não há garantias de que tudo funcionará corretamente. Acesse o [site oficial](https://www.python.org/downloads/) para baixar a versão 3.9.7.

### Simples

Assumindo que você possui a versão esperada do Python, você pode simplesmente instalar as dependências com o seguinte comando:

```bash
python -m pip install -r requirements.txt
```

No entanto, para evitar conflitos com a sua instalação local do Python, **é recomendado que você utilize um ambiente virtual**. Basta seguir as instruções abaixo.

### Ambiente virtual

Essa aplicação usa o `virtualenv` (e não o `venv`) para gerir seu ambiente virtual. A opção pelo `virtualenv` se dá pela possibilidade de fixar uma versão do Python, permitindo reproduzir o código com mais facilidade.

Este programa utiliza a versão **3.9.7** do Python. Para começar, garanta que você possui esta versão do Python instalada. Os comandos a seguir assumem que você está utilizando Windows.

Comece instalando o `virtualenv`:

```bash
python -m pip install virtualenv
```

Crie o ambiente virtual, apontando para o local de instalação do Python 3.9.7. Para descobrir o local de instalação, utilize o seguinte comando (novamente, aplicável somente no Windows):

```bash
(gcm python).Path
```

O resultado deve ser algo como `C:\Users\<user>\AppData\Local\Programs\Python\Python39\python.exe`. Veja que podem existir diferentes versões do Python instalados no seu sistema (talvez você veja `Python310\python.exe` ou mesmo `Python38/python.exe`). Se este for o caso, escolha a versão correta.

Utilize este caminho para criar o ambiente virtual:

```bash
python -m virtualenv -p <caminho_para_python_39> venv
```

Ative o ambiente virtual:

```bash
.\venv\Scripts\activate
```

Para verificar se o ambiente virtual foi ativado, rode o seguinte comando:

```bash
python -c "import sys; print(sys.executable)"
```

O resultado deve ser algo como:

```bash
C:\Users\<user>\<...>\psg-multilabel-classifier\venv\Scripts\python.exe
```

Agora, instale as dependências:

```bash
python -m pip install -r requirements.txt
```

## Testes unitário

Tanto os modelos de base, quanto os modelos principais, possuem testes unitários. Por meio de um `random_state` fixo, eles testam o modelo com um conjunto de dados reduzido, obtendo resultados determinísticos.

Ainda assim, alguns testes podem levar um tempo para executar, devido à natureza mais intensiva de alguns modelos. O tempo total deve ficar em torno de 2 minutos e meio.

Para rodar os testes, utilize este command:

```sh
python -m pytest
```

Caso queria executar um teste específico, utilize uma opção especial do `pytest`:

```sh
python -m pytest -k <nome_do_test>
```

## Tarefas

- [ ] Adicionar tipagem correta para todas as classes.
- [ ] Resolver todos os "TODO"s deixados no código.
- [ ] Entender como deixar os modelos imunes aos dados serem passados como matriz esparsa ou matriz normal.
  - [ ] [Isto](https://stackoverflow.com/questions/7922487/how-to-transform-numpy-matrix-or-array-to-scipy-sparse-matrix) pode ser útil.