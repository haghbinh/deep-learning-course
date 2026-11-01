window.MathJax = {
  loader: { load: ["[tex]/boldsymbol"] },
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
    packages: { "[+]": ["boldsymbol"] }
  },
  // Persian inside \text{...} is drawn with the page font (Vazirmatn), so the
  // letters join correctly instead of being drawn one by one by MathJax.
  chtml: { mtextInheritFont: true },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

document$.subscribe(() => {
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});
