---
layout: post
title: WILT#02 - Como adicionar JS-Beautify em sua aplicação Node.js
---

Após muito tempo participando de um projeto open-source bem legal ( [Tablero](https://github.com/twtablero/tablero) ) , resolvi enforçar (essa palavra existe?) alguns padrões de código importantes, para que não ficasse igual a uns outros projetos open-source por aí. 

Com isso, resolvi usar outra coisa além do sempre útil [jsHint](http://jshint.com/). Procurei, procurei, e deste então tenho o meu novo melhor amigo: o [js-beautify](https://github.com/beautify-web/js-beautify) .

O bacana dele é que podemos integrar ao npm, e executá-lo ~automagically~  sempre que rodar o teste (ou outra task que tu queiras). 

Para isso, tive de seguir alguns passos simples:

  - Instalar o js-beautify:
  {% highlight bash %}
    $ npm install js-beautify --save-dev
  {% endhighlight %}
  
  - Adicionar uma npm task no package.json chamando o js-beautify. Eu resolvi separar por destino de arquivos, mas mais simples também funciona.
  {% highlight js %}
    ...
    "scripts":{ 
      "pretest": "npm run beautify && npm run jshint",
      "hint": "./node_modules/jshint/bin/jshint .",
      "beautify": "npm run beautify:js && npm run beautify:html",
      "beautify:js": "git ls-files '**/*.js' | grep -vf .jshintignore | xargs ./node_modules/js-beautify/js/bin/js-beautify.js -s 2 -r -j --good-stuff",
      "beautify:css": "git ls-files '**/*.css' | grep -vf .jshintignore | xargs ./node_modules/js-beautify/js/bin/css-beautify.js -s 2 -r",
      "beautify:html": "git ls-files '**/*.html' | grep -vf .jshintignore | xargs ./node_modules/js-beautify/js/bin/html-beautify.js -s 2 -r"
      ...
    }
  {% endhighlight %}
  
  - Pronto. Depois disso, a cada vez que você rodar seus testes, o seu código será ~beautificado~.
  
Vocês podem ver [aqui](https://github.com/TWtablero/tablero/pull/226/files) o PR que fiz quando usei o js-beautify pela primeira vez no projeto que eu estava trabalhando. e [aqui](https://github.com/TWtablero/tablero/blob/b5b9b30bef06c889c5557d00bf3cf26d510851f8/package.json) está o meu package.json inteiro após essas mudanças.

No github do js-beautify você pode ver como é possível fazer customizações no resultado. Eu achei o padrão já bem bom!

E por hoje, isso é só. 

##### Conselho do dia

> Nunca pesquise no google qualquer sintoma que esteja sentindo, será sempre câncer.

##### GIF do dia

![XôRecalque!](http://gifs.gif-animado.com/eucontraorecalque1.gif)
