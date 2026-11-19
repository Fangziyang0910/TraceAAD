"""Build bilingual ICML 2027 manuscripts without refreshing frozen evidence."""
from pathlib import Path
import argparse
import hashlib
import re
import shutil
import subprocess

P = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--pandoc')
args = parser.parse_args()
pandoc = args.pandoc or shutil.which('pandoc')
if not pandoc:
    try:
        import pypandoc
        pandoc = pypandoc.get_pandoc_path()
    except ImportError:
        raise SystemExit('Install pandoc or pypandoc-binary, or pass --pandoc PATH.')


def latex(markdown):
    return subprocess.run(
        [str(pandoc), '--from=markdown+tex_math_dollars', '--to=latex',
         '--natbib', '--no-highlight', '--wrap=none', '--shift-heading-level=-1'],
        input=markdown, text=True, capture_output=True, check=True, cwd=P,
    ).stdout.strip()


def algorithm(language):
    pairs = [
        ('Task interface, evaluator, LLM, budget $B$, initial size $N_0$', '任务接口、评价器、LLM、预算 $B$、初始规模 $N_0$'),
        ('Initialize tree $T$ with virtual root $v_0$', '初始化仅含虚拟根 $v_0$ 的树 $T$'),
        ('budget remains and stopping rule is not met', '评价预算尚有剩余且未满足停止条件'),
        ('fewer than $N_0$ feasible initial heuristics or $\\mathcal P=\\varnothing$', '可行初始启发式不足 $N_0$ 个或 $\\mathcal P=\\varnothing$'),
        ('Generate an initial heuristic under $v_0$', '在虚拟根 $v_0$ 下生成初始启发式'),
        ('Sample parent $p\\in\\mathcal P$ and search operator', '从 $\\mathcal P$ 中抽样父代 $p$ 与搜索算子'),
        ('Build prompt context from $T$', '从树 $T$ 构造提示上下文'),
        ('Generate candidate and evaluate (at most one repair)', '生成候选程序并评价（至多修复一次）'),
        ('no candidate was evaluated', '没有候选程序进入评价'),
        ('Attach evaluated node $v$ under its parent and update counts', '将最终受评节点 $v$ 接入父代下并更新评价计数'),
        ('fewer than $N_0$ feasible initial heuristics collected', '已获得的可行初始启发式不足 $N_0$ 个'),
        ('Select representatives $R$ and skeleton $\\mathcal H$', '选出代表集 $R$ 并补齐骨架 $\\mathcal H$'),
        ('Partition $\\mathcal V\\setminus\\mathcal H$ into path and subtree regions', '将非骨架节点划分为路径区域与子树区域'),
        ('Reflect on path and non-root subtree regions with $|C|\\geq m$ using pre-update $T$',
         '基于更新前的树 $T$ 反思 $|C|\\geq m$ 的路径区与非根子树区'),
        ('Contract paths and prune subtrees; store $E_{u\\to w},E_u$', '收缩路径并剪除子树，保存边经验 $E_{u\\to w}$ 与节点经验 $E_u$'),
        ('best feasible heuristic evaluated during search', '搜索中评价过的最优可行启发式'),
    ]
    s = [a if language == 'en' else b for a, b in pairs]
    return r'''\begin{algorithm}[H]
\caption{TREES}\label{alg:trees}
\begin{algorithmic}[1]
\REQUIRE ''' + s[0] + r'''
\STATE ''' + s[1] + r'''
\WHILE{''' + s[2] + r'''}
  \IF{''' + s[3] + r'''}
    \STATE ''' + s[4] + r'''
  \ELSE
    \STATE ''' + s[5] + r'''
  \ENDIF
  \STATE ''' + s[6] + r'''
  \STATE ''' + s[7] + r'''
  \IF{''' + s[8] + r'''}
    \STATE \textbf{continue}
  \ENDIF
  \STATE ''' + s[9] + r'''
  \IF{''' + s[10] + r'''}
    \STATE \textbf{continue}
  \ENDIF
  \STATE ''' + s[11] + r'''
  \STATE ''' + s[12] + r'''
  \STATE ''' + s[13] + r'''
  \STATE ''' + s[14] + r'''
\ENDWHILE
\RETURN ''' + s[15] + r'''
\end{algorithmic}
\end{algorithm}'''


def convert_body(markdown, language):
    markdown = re.sub(r'^(#{2,4}) (?:\d+(?:\.\d+)*\.|[A-Z]\.\d+\.)\s+', r'\1 ', markdown, flags=re.M)
    markdown = re.sub(r'^## (?:Appendix|附录)\s+[A-Z]\.\s+', '## ', markdown, flags=re.M)
    markdown = re.sub(r'```text\n(?:Algorithm 1: |算法1：)TREES\n.*?```', 'EVOTRACEALGORITHM', markdown, flags=re.S)
    graphics = []
    def figure(match):
        path, lead, tail = match.groups()
        path = {'figures/early-search.png': 'figures/early-search.pdf',
                'figures/representative-search.png': 'figures/representative-search.pdf',
                'figures/inheritance-case.png': 'figures/inheritance-case.pdf'}.get(path, path)
        caption = latex(lead.strip() + ' ' + tail.strip())
        number = len(graphics) + 1
        placement = 'htbp' if Path(path).stem == 'inheritance-case' else 't'
        graphics.append(f'\\begin{{figure*}}[{placement}]\n\\centering\n'
                        + f'\\includegraphics[width=\\textwidth]{{{path}}}\n'
                        + f'\\caption{{{caption}}}\\label{{fig:trees-{Path(path).stem}}}\n'
                        + '\\end{figure*}')
        return f'EVOTRACEFIGURE{number}'
    markdown = re.sub(r'!\[[^\]]*\]\(([^)]+)\)\n\n\*\*(?:Figure \d+\.|图\d+｜)(.*?)\*\*\s*(.*?)(?=\n\n)', figure, markdown, flags=re.S)
    score_table = None
    def table(match):
        nonlocal score_table
        caption = latex(match[1].strip() + ' ' + match[2].strip())
        rows = []
        for line in match[3].strip().splitlines()[2:]:
            cells = [c.strip() for c in line.strip('|').split('|')]
            rows.append(' & '.join([cells[0], *cells[1].split(' / '), *cells[2:]]) + r' \\')
        headings = ('Task & Seed 0 & Seed 1 & Seed 2 & Mean & SD' if language == 'en'
                    else '任务 & 种子0 & 种子1 & 种子2 & 均值 & 标准差')
        score_table = ('\\begin{table*}[t]\n\\caption{' + caption + '}\\label{tab:training}\n'
                       '\\centering\n\\begin{tabular}{lrrrrr}\n\\toprule\n' + headings + r' \\'
                       + '\n\\midrule\n' + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n\\end{table*}')
        return 'EVOTRACESCORETABLE\n\n'
    markdown = re.sub(r'\*\*(?:Table 1\.|表1｜)(.*?)\*\*([^\n]*)\n\n((?:\|[^\n]+\n)+)', table, markdown)
    markdown = re.sub(r'\*\*(?:Table [2-9]\.|表[2-9]｜)(.*?)\*\*([^\n]*)\n\n(?=\|)',
                      lambda m: 'Table: ' + m[1].strip() + ' ' + m[2].strip() + '\n\n', markdown)
    # Declare the two wide results floats when the analysis starts, so they
    # appear beside its text rather than being delayed until the conclusion.
    if len(graphics) == 3 and score_table:
        heading = ('## Analysis of Experience Accumulation During Search' if language == 'en'
                   else '## 搜索中的经验积累分析')
        assert markdown.count(heading) == 1
        markdown = markdown.replace('EVOTRACEFIGURE3', '').replace('EVOTRACESCORETABLE', '')
        markdown = markdown.replace(heading, heading + '\n\nEVOTRACEFIGURE3\n\nEVOTRACESCORETABLE', 1)
    result = latex(markdown)
    # Keep each verbatim prompt together with its short label in the appendix.
    result = re.sub(
        r'^(?:\\text(?:bf|tt)\{[^\n]+\}\n\n)?\\begin\{verbatim\}\n.*?^\\end\{verbatim\}$',
        lambda match: r'\noindent\begin{minipage}{\linewidth}' + '\n'
        + match[0] + '\n' + r'\end{minipage}\par',
        result,
        flags=re.M | re.S,
    )
    for i, graphic in enumerate(graphics, 1):
        result = result.replace(f'EVOTRACEFIGURE{i}', graphic)
    if score_table:
        result = result.replace('EVOTRACESCORETABLE', score_table)
    return result.replace('EVOTRACEALGORITHM', algorithm(language))


for language, source, name in [('en', 'paper.md', 'main'), ('zh', 'paper-中文.md', 'main-zh')]:
    text = (P/source).read_text()
    title = latex(text.splitlines()[0][2:])
    display_title = title
    if language == 'en':
        # Balance the long title while retaining its unbroken PDF metadata.
        display_title = title.replace(' in Tree-Structured Algorithm Populations',
                                      r'\\ in Tree-Structured Algorithm Populations')
        display_title = r'\texorpdfstring{' + display_title + '}{' + title + '}'
    abstract_heading = '## Abstract\n\n' if language == 'en' else '## 摘要\n\n'
    abstract = latex(text.split(abstract_heading, 1)[1].split('\n## ', 1)[0])
    body = '## ' + text.split('\n## ', 2)[2]
    appendix_heading = '## Appendix A.' if language == 'en' else '## 附录 A.'
    main, appendix = body.split(appendix_heading, 1)
    appendix = appendix_heading + appendix
    main = main.split('## References')[0].split('## 参考文献')[0]
    main_tex, appendix_tex = convert_body(main, language), convert_body(appendix, language)
    cjk = (r'''\usepackage{xeCJK}
\setCJKmainfont{Noto Serif CJK SC}
\setCJKsansfont{Noto Sans CJK SC}
\setCJKmonofont{Noto Sans Mono CJK SC}
\renewcommand{\tablename}{表}
\renewcommand{\figurename}{图}
\renewcommand{\abstractname}{摘要}
\renewenvironment{abstract}
{\centerline{\large\bf 摘要}
 \vspace{-0.12in}\begin{quote}}
{\par\end{quote}\vskip0.12in}
\renewcommand{\refname}{参考文献}
\makeatletter
\renewcommand{\fnum@table}{表~\thetable}
\renewcommand{\fnum@figure}{图~\thefigure}
\makeatother
\floatname{algorithm}{算法}
''' if language == 'zh' else '')
    tex = r'''% Generated from the Markdown manuscript by paper/build.py.
\documentclass{article}
\makeatletter
\def\input@path{{vendor/icml2027/}}
\makeatother
\usepackage{microtype}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{hyperref}
\newcommand{\theHalgorithm}{\arabic{algorithm}}
\usepackage{icml2027}
\usepackage{amsmath,amssymb,mathtools}
\usepackage{fontspec}
\setmainfont{Times New Roman}
\setsansfont{TeX Gyre Heros}
\setmonofont{TeX Gyre Cursor}[Scale=MatchLowercase]
\input{style.tex}
''' + cjk + r'''
\icmltitlerunning{Reflective Elimination: Accumulating Design Experience}
\begin{document}
\twocolumn[
\icmltitle{''' + display_title + r'''}
\begin{icmlauthorlist}
\icmlauthor{Anonymous Authors}{anon}
\end{icmlauthorlist}
\icmlaffiliation{anon}{Anonymous Institution}
\icmlcorrespondingauthor{}{}
\icmlkeywords{Automated algorithm design, evolutionary search, reflection}
\vskip 0.3in
]
\printAffiliationsAndNotice{}
\begin{abstract}
''' + abstract + '\n\\end{abstract}\n\n' + main_tex + r'''

\FloatBarrier
\label{evotrace-main-end}
\bibliographystyle{vendor/icml2027/icml2027}
\bibliography{references}

\clearpage
\onecolumn
\raggedbottom
\appendix
''' + appendix_tex + '\n\\end{document}\n'
    tex = '% Markdown SHA256: ' + hashlib.sha256(text.encode()).hexdigest() + '\n' + tex
    (P/(name+'.tex')).write_text(tex)
    with (P/(name+'-build.log')).open('w') as log:
        subprocess.run(['latexmk', '-xelatex', '-interaction=nonstopmode', '-halt-on-error',
                        '-outdir=build', name+'.tex'], cwd=P, check=True, stdout=log, stderr=subprocess.STDOUT)
    shutil.copy2(P/'build'/(name+'.pdf'), P/('trees.pdf' if language == 'en' else 'trees-中文.pdf'))
    print(name, 'complete', flush=True)
