window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']],
    packages: {'[+]': ['ams']},
    macros: {
      bm: ['\\mathbf{#1}', 1],
      bold: ['\\mathbf{#1}', 1],
      argmax: ['\\mathop{\\operatorname{argmax}}', 0] ,   // 新增：支持 \argmax
      argmin: ['\\mathop{\\operatorname{argmin}}', 0] ,   // 新增：支持 \argmin
      empty: ['\\emptyset', 0]
    },
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: '.*|',
    processHtmlClass: 'arithmatex'
  }
};

document$.subscribe(() => {
  if (window.MathJax?.typesetPromise) {
    MathJax.typesetPromise();
  }
});