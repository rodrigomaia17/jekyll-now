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

- **train.csv**: contém as informações dos passageios **e** a informação se ele sobreviveu ou não. É esse arquivo que iremos alterar, tratar, normalizar as informações a fim de se tornarem as melhores possíveis para o computador "aprender".

- **test.csv**: contém apenas as informações dos passageios. O seu algoritmo será responsável por dizer se o passageiro sobreviveu ou não. Ao final do processo, esse será o arquivo que iremos submeter para o site do kaggle para que calcule nossa pontuação. 

Abra o console na pasta do seu projeto e inicie o interpretador do python:

```python
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

```python
>>> train = pd.read_csv('train.csv')
```

Após ler o arquivo, vamos ver um pequeno resumo do arquivo e de suas colunas:

```python
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

```python
>>> train.head(1)
   PassengerId  Survived  Pclass                     Name   Sex  Age  SibSp  \
0            1         0       3  Braund, Mr. Owen Harris  male   22      1

   Parch     Ticket  Fare Cabin Embarked
0      0  A/5 21171  7.25   NaN        S
```

_(Rolou uma quebra de linha aqui,mas está tudo bem)_

O comando head(n) mostra as primeiras n linhas do DataFrame. No site do Kaggle ele te dá uma legenda para esses nomes de coluna e seus significados. Por exemplo, a coluna Pclass fala qual classe de acomodação o passageiro estava (de primeira até terceira classe). 

O mais interessante é reparar na coluna Survived. Como esse é o arquivo para 'treinar' nosso agoritmo, ele possui a informação se o passageiro sobreviveu ou não ao Titanic. No outro arquivo .csv, o test.csv, nós teremos outros passageiros com todas as informações **menos** a informação se sobreviveu ou não. Essa é a informação que nosso algoritmo terá de preencher.

Mas agora que conseguimos ler e entender um pouco do .csv, vamos pensar no dominio do nosso problema e ver o que podemos fazer:

### Desenvolvendo uma possível solução do problema

Aqui é aonde entra um ou dos pilares do machine learning, o __entendimento do domínio__. É muito importante que entendamos o que estamos analizando para informar ao algoritmo o que ele deve aprender para conseguir propor os resultados. Esse exercício que tentaremos fazer agora.  

Pensando em um desastre do tipo __Titanic__ em que temos que priorizar o salvamento de algumas pessoas em detrimento de outras, lembramos rapidamente da máxima __"Mulheres e crianças primeiro"__. Como essa ideia encaixa na nossa base de informação?

A informação do gênero (nesse caso apenas binária) está na coluna "Sex". Vendo a informação das linhas, percebemos que esses valores estão discriminados com o valor 'male' ou 'female' escritos como String. Caso queira conferir, basta usarmos o comando unique:

```python
>>> train.Sex.unique()
array(['male', 'female'], dtype=object)
```  

Isso é uma boa representação para os algoritmos de machine learning? Não. O ideal é usarmos apenas números para representar nossas informações. Então teremos que ter o trabalho de converter esses valores para representações numéricas, iremos cobrir isso na próxima seção.  

A outra informação que queremos saber é se o passageiro é criança ou não. Para isso temos a coluna de idade na nossa base de dados, e ela já é numérica. Trabalho pronto? Não. Dando uma olhada mais aprofundada percebemos que nem todos os passageiros possuem essa informação. Se nesse momento estamos baseando nossa solução apenas nas informações de idade e sexo, é ideal que elas sejam o mais 'normalizadas' possível.  

Vamos então tratar esses dados?

### Tratando a informação

Normalmente os algoritmos de Machine Learning trabalham apenas com números. Então é importante que forneçamos informações relevantes e bem tratadas. Uma feature que utilizaremos com certeza é a coluna Age. Vamos ver quais são os valores de Age que temos nessa base de dados:

```python
>>> train.Age.unique()
array([ 22.  ,  38.  ,  26.  ,  35.  ,    nan,  54.  ,   2.  ,  27.  ,
        14.  ,   4.  ,  58.  ,  20.  ,  39.  ,  55.  ,  31.  ,  34.  ,
        15.  ,  28.  ,   8.  ,  19.  ,  40.  ,  66.  ,  42.  ,  21.  ,
        18.  ,   3.  ,   7.  ,  49.  ,  29.  ,  65.  ,  28.5 ,   5.  ,
        11.  ,  45.  ,  17.  ,  32.  ,  16.  ,  25.  ,   0.83,  30.  ,
        33.  ,  23.  ,  24.  ,  46.  ,  59.  ,  71.  ,  37.  ,  47.  ,
        14.5 ,  70.5 ,  32.5 ,  12.  ,   9.  ,  36.5 ,  51.  ,  55.5 ,
        40.5 ,  44.  ,   1.  ,  61.  ,  56.  ,  50.  ,  36.  ,  45.5 ,
        20.5 ,  62.  ,  41.  ,  52.  ,  63.  ,  23.5 ,   0.92,  43.  ,
        60.  ,  10.  ,  64.  ,  13.  ,  48.  ,   0.75,  53.  ,  57.  ,
        80.  ,  70.  ,  24.5 ,   6.  ,   0.67,  30.5 ,   0.42,  34.5 ,  74.  ])
```

Entendendo esse comando que usei: 

* Primeiro chamei train.Age que me retorna todos os valores da coluna Age;
* Depois usei o método unique() que me retorna uma array com todos os valores distintos daquela coluna.


O importante aqui é que vermos que existem linhas em que a idade é nula. Uma outra forma de saber se temos valores nulos em uma coluna seria:


```python
>>> train.Age.isnull().any()
True
```
_Aqui pegamos todos os valores que são nulos, e depois uso só a função any() para saber se existe qualquer valor assim._

Pensando que essa informação nula poderia afetar negativamente o nosso algoritmo, temos que substituir isso por outro valor. A primeira ideia que podemos tentar é trocar todos os nulos pela média da idade geral dos passageiros. Utilizaremos a função **fillna(new_value)**. Essa função substitui todos os valores nulos pelo valor que você passar por parametro.   

```python
>>age_mean = train.Age.mean()
>>train.Age = train.Age.fillna(age_mean)
```

Podemos conferir se temos algum valor nulo agora na coluna Age:

```python
>> train.Age.isnull().any()
False
```

Excelente! Agora vamos trabalhar na informação de sexo do passageiro:

Como falamos antes, essa informação é representada com duas strings: 'female' e 'male'. Precisamos mapear isso para valores inteiros. Podemos mapear os valores 'female' em 1 e os valores 'male' em 2. Para isso escreveremos o seguinte código:

```python
>>> train.Sex = train.Sex.map({'male': 2, 'female': 1})
```

Podemos conferir que deu certo observando novamente o primeiro passageiro:

```python
>>> train.head(1)
   PassengerId  Survived  Pclass                     Name  Sex  Age  SibSp  \
0            1         0       3  Braund, Mr. Owen Harris    2   22      1

   Parch     Ticket  Fare Cabin Embarked
0      0  A/5 21171  7.25   NaN        S
```

Com as duas features tratadas podemos partir para a parte mais legal:

--- 

### Construindo uma solução para o problema 

Agora que já temos o dado tratado, iremos para a parte de:

  1. Escolher um algoritmo para nosso problema
  2. Treinar esse algoritmo com os dados do train.csv
  3. Testar a eficácia do nosso modelo usando os próprios dados do train.csv 
  3. Usar o algoritmo para propor uma solução para os dados do test.csv

#### Escolhendo um algoritmo

Essa decisão de qual algoritmo usar envolve muito mais coisa do que pretendo abordar nesse arquivo de introdução. Pela origem do nosso problema, precisamos de encontrar um algoritmo que seja capaz de fazer uma __classificação__. Isso é, dado as informações do passageiro, o algorimo tem que classificá-lo em "Sobreviveu" ou "Não sobreviveu". Nessa introdução resolvi usar um método de __Decision Tree__ por ser fácil de entender e de visualizar. 

__Decision Tree__ é uma técnica de algoritmos de Machine learning que visa criar árvores de decisão baseados nos valores da suas features. O algoritmo observa os valores e tenta chegar na melhor árvore possível que possa classificar corretamente a informação dada. Por exemplo, uma árvore de decisão para classificar se uma pessoa está no grupo de maior risco de doença cardíaca seria algo como:  


![tree-example](/images/machine-learning-tree-example.png)
_Apenas lembrando que não sei nada de doenças cardíacas e você não deve suspender seu acompanhamento por causa dessa imagem ;)_

{% comment %}
graph TD
    A[É fumante?]
    A-- Sim -->B
    B[Tem menos que 30 anos?]
    B-- Sim -->X1
    B-- Não -->Z1       
    A-- Não -->E
    E[Se alimenta bem?]
    E-- Sim -->X
    E-- Não -->Z
    X[Menor Risco]
    Z[Maior Risco]
    X1[Menor Risco]
    Z1[Maior Risco] 
{% endcomment %}


--- 

### Testando o algoritmo

>>> clf.fit(train[['Age','Sex']],train.Survived)
>>> test = pd.read_csv('test.csv')
>>> test.Sex = test.Sex.map({'male':2, 'female':1})
>>> test.Age = test.Age.fillna(test.Age.mean())
>>> clf.predict(test[['Age','Sex']])
>>> test['Survived'] = pd.Series(clf.predict(test[['Age','Sex']]))
>>> test[['PassengerId','Survived']].to_csv('final.csv', index=False)
--- 

### Enviando para o Kaggle

