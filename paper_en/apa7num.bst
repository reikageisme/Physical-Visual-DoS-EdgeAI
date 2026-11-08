%% apa7num.bst -- BibTeX style: numbered reference list [n], entries formatted in APA 7th edition.
%% Reads standard fields (author, year, title, journal/booktitle, volume, number, pages, publisher, doi, url).
%% List sorted by author surnames then year (APA order). Use with:
%%   \usepackage[numbers,sort&compress]{natbib}  \bibliographystyle{apa7num}  \bibliography{references}
%% Titles are converted to sentence case ("t"); protect proper nouns/acronyms with braces in the .bib.

ENTRY
  { author booktitle doi journal number pages publisher title url volume year }
  {}
  { label }

INTEGERS { nameptr numnames }
STRINGS { s t }

FUNCTION {not} { { #0 } { #1 } if$ }
FUNCTION {and} { 'skip$ { pop$ #0 } if$ }
FUNCTION {field.or.null} { duplicate$ empty$ { pop$ "" } 'skip$ if$ }

FUNCTION {apa.name} { "{vv~}{ll}, {f.}{, jj}" format.name$ }

FUNCTION {format.authors}
{ author empty$
    { "" }
    { author 's :=
      s num.names$ 'numnames :=
      numnames #20 >
        { ""
          #1 'nameptr :=
          { nameptr #19 > not }
          { s nameptr apa.name * ", " *
            nameptr #1 + 'nameptr := }
          while$
          "\ldots{} " * s numnames apa.name *
        }
        { ""
          #1 'nameptr :=
          { nameptr numnames > not }
          { nameptr #1 >
              { numnames nameptr =
                  { ", \& " * }
                  { ", " * }
                if$ }
              'skip$
            if$
            s nameptr apa.name *
            nameptr #1 + 'nameptr := }
          while$
        }
      if$
    }
  if$
}

FUNCTION {format.title}
{ title empty$ { "" } { title "t" change.case$ add.period$ } if$ }

FUNCTION {format.link}
{ doi empty$
    { url empty$ { "" } { " \url{" url * "}" * } if$ }
    { " \url{https://doi.org/" doi * "}" * }
  if$
}

FUNCTION {head}
{ format.authors " (" * year * "). " * format.title * }

FUNCTION {output.bibitem}
{ newline$
  "\bibitem{" cite$ * "}" * write$ newline$
}

FUNCTION {article}
{ output.bibitem
  head " \textit{" * journal * "}" *
  volume empty$ 'skip$ { ", \textit{" * volume * "}" * } if$
  number empty$ 'skip$ { "(" * number * ")" * } if$
  pages empty$ 'skip$ { ", " * pages * } if$
  "." * format.link *
  write$ newline$
}

FUNCTION {inproceedings}
{ output.bibitem
  head " In \textit{" * booktitle * "}" *
  volume empty$
    { pages empty$ 'skip$ { " (pp. " * pages * ")" * } if$ }
    { pages empty$
        { " (Vol. " * volume * ")" * }
        { " (Vol. " * volume * ", pp. " * pages * ")" * }
      if$ }
  if$
  "." *
  publisher empty$ 'skip$ { " " * publisher * "." * } if$
  format.link *
  write$ newline$
}

FUNCTION {misc} { inproceedings }
FUNCTION {default.type} { inproceedings }

READ

FUNCTION {sort.format.names}
{ 's :=
  ""
  #1 'nameptr :=
  s num.names$ 'numnames :=
  { nameptr numnames > not }
  { s nameptr "{vv{ } }{ll{ }}{  f{ }}" format.name$ * "   " *
    nameptr #1 + 'nameptr := }
  while$
}

FUNCTION {presort}
{ author empty$ { title } { author sort.format.names } if$
  "    " * year field.or.null *
  purify$ "l" change.case$
  #1 entry.max$ substring$
  'sort.key$ :=
}

ITERATE {presort}

SORT

FUNCTION {begin.bib}
{ "\begin{thebibliography}{99}" write$ newline$ }

EXECUTE {begin.bib}

ITERATE {call.type$}

FUNCTION {end.bib}
{ newline$ "\end{thebibliography}" write$ newline$ }

EXECUTE {end.bib}
