\newcommand\classname{bioinfo}
\newcommand\versionnumber{0.1}

\newif\if@cropmarkson \@cropmarksontrue

\NeedsTexFormat{LaTeX2e}[2001/06/01]
\ProvidesClass{\classname}[\versionnumber]

\setlength{\paperheight}{11truein}
\setlength{\paperwidth}{8.5truein}

\newif\if@final

\DeclareOption{draft}{\PassOptionsToPackage{draft}{graphicx}}
\DeclareOption{a4paper}{\PassOptionsToPackage{a4}{crop}}
\DeclareOption{centre}{\PassOptionsToPackage{center}{crop}}
\DeclareOption{crop}{\PassOptionsToPackage{cam}{crop}\global\@cropmarksontrue}
\DeclareOption{nocrop}{\PassOptionsToPackage{off}{crop}\global\@cropmarksonfalse}
\DeclareOption{info}{\PassOptionsToPackage{info}{crop}}
\DeclareOption{noinfo}{\PassOptionsToPackage{noinfo}{crop}}
\DeclareOption{final}{\global\@finaltrue}

\ExecuteOptions{a4paper,nocrop,centre,info}

\ProcessOptions

\RequirePackage{}\RequirePackage{inputenc,crop,graphicx,amsmath,array,color,amssymb,flushend,stfloats,amsthm,chngpage,times}

%\RequirePackage[LY1]{fontenc}
%\RequirePackage[LY1,mtbold]{mathtime}
\def\authoraffliate{\fontfamily{phv}\selectfont}
\def\helvetica{\fontfamily{phv}\selectfont}
\def\helveticaitalic{\fontfamily{phv}\itshape\selectfont}
\def\helveticabold{\fontfamily{phv}\bfseries\selectfont}
\def\helveticabolditalic{\fontfamily{phv}\bfseries\itshape\selectfont}

\newcommand\@ptsize{0}

\@twosidetrue

\@mparswitchfalse

\reversemarginpar

\renewcommand\normalsize{%
    \@setfonsize\normalsize{9}{11}
    \abovedisplayskip 10\p@ \plus2\p@ \@minus5\p@
}