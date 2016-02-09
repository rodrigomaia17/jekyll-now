---
layout: post
title: "Começando com Machine Learning em Python"
keywords: "python, tutorial, machine-learning, pandas, kaggle"
description: Um pequeno guia sobre machine learning utilizando python
---

Olá! Nesse post vamos explicar o básico de conceitos de machine learning para iniciantes (como eu), além de um passo a passo de como usar python em um problema bem legal proposto pelo kaggle, seu próximo site mais acessado ;).

### Conceitos Básicos

Uns conceitos básicos para alinharmos o vocábulário:

**Machine Learning:** É um método de análise de dados que te permite 'ensinar' o computador

---

### Show me the code!

Vamos começar a escrever o código. É importante que você tenha um ambiente de python preparado e redondinho. Recomendo utilizar a distribuição _Anaconda_  pois já vem com tudo que precisaremos em termos de bibliotecas de data science.

Crie uma pasta para nosso projeto, e nela copie os arquivos que voce baixou do site do kaggle. São eles:

* **train.csv**: contém as informações dos passageios **e** a informação se ele sobreviveu ou não. É esse arquivo que iremos alterar, tratar, normalizar as informações a fim de se tornarem as melhores possíveis para o computador "aprender".

* **test.csv**: contém apenas as informações dos passageios. O seu algoritmo será responsável por dizer se o passageiro sobreviveu ou não. Ao final do processo, esse será o arquivo que iremos submeter para o site do kaggle para que calcule nossa pontuação. 

Abra o console na pasta do seu projeto e inicie o interpretador do python:

```zsh
~/projects/kaggle-titanic  ᐅ python3
Python 3.5.0 |Anaconda 2.4.0 (x86_64)| (default, Oct 20 2015, 14:39:26)
[GCC 4.2.1 (Apple Inc. build 5577)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>
```

Sua versão pode ser diferente da minha, as bibliotecas que usarei são bastante consistentes e provavelmente não terá discrepâncias.

#### Lendo o arquivo .csv

Primeiro vamos carregar o arquivo train.csv. Para isso usaremos uma lib muito boa chamada **pandas**. A primeira coisa a se fazer é importar a lib:

```python
>>> import pandas as pd
```

Pandas é uma lib que provê estruturas de dados muito úteis para esse tipo de informação que estamos carregando. A estrutura que estaremos usando para representar as infomrações do arquivo train.csv é o **DataFrame**. DataFrame é uma estrutura de duas dimensões com colunas, bem parecido com a imagem mental que temos de uma tabela SQL. Assim que importarmos o .csv , poderemos ver que a estrututura resultante é um DataFrame:

```zsh
>>> train = pd.read_csv('train.csv')
```

Após ler o arquivo, vamos ver um pequeno resumo do arquivo e de suas colunas:

```zsh
>>> train.info()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 891 entries, 0 to 890
Data columns (total 12 columns):
PassengerId    891 non-null int64
Survived       891 non-null int64
Pclass         891 non-null int64
Name           891 non-null object
Sex            891 non-null object
Age            714 non-null float64
SibSp          891 non-null int64
Parch          891 non-null int64
Ticket         891 non-null object
Fare           891 non-null float64
Cabin          204 non-null object
Embarked       889 non-null object
dtypes: float64(2), int64(5), object(5)
memory usage: 90.5+ Kb
```

Bacana não é? Temos 891 passageiros nesse arquivo, e cada um possui esse tanto de informação. Vamos ver uma primeira linha de exemplo?

```zsh
>>> train.head(1)
   PassengerId  Survived  Pclass                     Name   Sex  Age  SibSp  \
0            1         0       3  Braund, Mr. Owen Harris  male   22      1

   Parch     Ticket  Fare Cabin Embarked
0      0  A/5 21171  7.25   NaN        S
```

_(Rolou uma quebra de linha aqui, está tudo bem)_

O comando head(n) mostra as primeiras n linhas do DataFrame. No site do Kaggle ele te dá uma legenda para esses nomes de coluna e seus significados. Por exemplo, a coluna Pclass fala qual classe de acomodação o passageiro estava (de primeira até terceira classe). 

O mais interessante é reparar na coluna Survived.
