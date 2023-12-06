# psg-multilabel-classifier

Trabalho de Conclusão de Curso, para graduação em Sistemas de Informação pela EACH-USP. Este projeto tem como foco o desenvolvimento de classificadores multirrótulo inéditos.

## Testes unitário

Tanto os modelos de base, quanto os principais, possuem testes unitários. Por meio de um `random_state` fixo, eles testam o modelo com um conjunto de dados reduzido, obtendo resultados determinísticos.

Ainda assim, alguns testes podem levar um tempo para executar, devido à natureza mais intensiva de alguns modelos. O tempo total fica em torno de 2 minutos e meio.

Para rodar os testes, utilize este command:

```sh
pytest
```

Caso queria executar um teste específico, utilize uma opção especial do `pytest`:

```sh
pytest -k <nome_do_test>
```
