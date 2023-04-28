# psg-multilabel-classifier

Trabalho de Conclusão de Curso, com foco em classificadores multirrótulo.

## TODOs

- [ ] Ler página sobre [multiclass](https://scikit-learn.org/stable/modules/multiclass.html) do Scikit e verificar quais técnicas já são implementadas.
- [ ] Realizar testes simples com essas técnicas
  - [ ] Listar as técnicas aqui
  - [ ] ...
- [ ] Correlacionar as técnicas do Scikit com o que foi apresentado no artigo da professora.
- [ ] Pesquisar implementações de multirrótulo em outroas bibliotecas.

## Técnicas

- Uso do [MultiOutput Classifier](https://scikit-learn.org/stable/modules/multiclass.html#multioutputclassifier) ([documentação](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html#sklearn.multioutput.MultiOutputClassifier)) testa um classificador para cada rótulo, sendo uma extensão para classificadores que não realizam por padrão a classificação multirrótulo.

- [ClassifierChains](https://scikit-learn.org/stable/modules/multiclass.html#classifierchain) faz uma combinação de classificadores binários e é capaz de encontrar correlação 

## Dúvidas

- [ ] Multioutput, alguma relação com nosso projeto? (visto aqui: https://scikit-learn.org/stable/modules/multiclass.html)
  - [ ] Parece que tem tudo a ver, mas ainda não sei dizer precisamente qual a diferença entre _multioutput_ e _multilabel_.
- [ ] A ideia é implementar o que ainda não exite... Isso significa implementar _do zero_ o algoritmo?

## Links

https://scikit-learn.org/stable/modules/multiclass.html