"""Build English and Chinese PDFs from the maintained Markdown manuscripts.

Requires pandoc, XeLaTeX, latexmk, and BibTeX. Uses the frozen evidence;
never refreshes results or runs candidate algorithms.
"""
from pathlib import Path
import argparse, json, os, re, shutil, subprocess, tempfile
P=Path(__file__).resolve().parent;ROOT=P.parent
parser=argparse.ArgumentParser();parser.add_argument('--pandoc');args=parser.parse_args()
pandoc=args.pandoc or shutil.which('pandoc')
if not pandoc:
    try:
        import pypandoc
        pandoc=pypandoc.get_pandoc_path()
    except ImportError:raise SystemExit('Install pandoc or pypandoc-binary, or pass --pandoc PATH.')
shutil.copy2(ROOT/'docs/paper-references.bib',P/'references.bib')
for language,source,name in [('en','paper.md','main'),('zh','paper-中文.md','main-zh')]:
    text=(ROOT/'docs'/source).read_text()
    title=text.splitlines()[0][2:]
    abstract=text.split('## Abstract\n\n' if language=='en' else '## 摘要\n\n',1)[1].split('\n## ',1)[0]
    body='## '+text.split('\n## ',2)[2]
    body=body.split('## References')[0].split('## 参考文献')[0]
    body=body.replace('figures/evotrace/07-framework.png','figures/framework.png').replace('figures/evotrace/08-reflective-simplification.png','figures/reflective-simplification.png').replace('../paper/figures/early-search.png','figures/early-search.pdf')
    # Figure captions stay with the image rather than becoming separate paragraphs.
    body=re.sub(r'!\[[^\]]*\]\(([^)]+)\)\n\n\*\*(?:Figure \d+\.|图\d+｜)(.*?)\*\*\s*(.*?)(?=\n\n)',lambda m:'!['+m[2].strip()+' '+m[3].strip()+']('+m[1]+')',body,flags=re.S)
    body=re.sub(r'\*\*(?:Table 1\.|表1｜)(.*?)\*\*\n\n((?:\|[^\n]+\n)+)', lambda m:m[2]+'\nTable: '+m[1].strip()+'\n\n', body)
    # Bibliography is generated once by BibTeX; preserve explicit manuscript section numbering.
    meta='---\n'+ '\n'.join(k+': '+json.dumps(v,ensure_ascii=False) for k,v in {'title':title,'author':'Anonymous authors' if language=='en' else '匿名作者','date':'','abstract':abstract,'bibliography':'references.bib','biblio-style':'plainnat','link-citations':True}.items())+'\n---\n\n'
    tmp=P/(name+'-build.md');tmp.write_text(meta+body)
    cmd=[str(pandoc),str(tmp),'--from=markdown+tex_math_dollars','--to=latex','--standalone','--natbib','--no-highlight',
         '-V','documentclass=article','-V','fontsize=11pt','-V','mainfont=TeX Gyre Termes','-V','mathfont=TeX Gyre Termes Math',
         '-V','geometry=margin=0.9in','-V','colorlinks=true','-V','urlcolor=blue','-V','linkcolor=black','-V','citecolor=black',
         '--include-in-header',str(P/'style.tex'),'-o',str(P/(name+'.tex'))]
    if language=='zh':cmd+=['-V','CJKmainfont=Noto Serif CJK SC']
    subprocess.run(cmd,cwd=P,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    tex=(P/(name+'.tex')).read_text()
    if language=='zh':
        tex=tex.replace('\\begin{document}', '\\setCJKmonofont{Noto Sans Mono CJK SC}\n\\renewcommand{\\tablename}{表}\n\\renewcommand{\\abstractname}{摘要}\n\\renewcommand{\\refname}{参考文献}\n\\renewcommand{\\figurename}{图}\n\\begin{document}')
    # Keep the four-task score table and the short search algorithm intact.
    tex=re.sub(r'(\\begin\{longtable\}.*?\\end\{longtable\})', lambda m:('\\Needspace{12\\baselineskip}\n' if '\\caption{' in m[0] else '')+m[0], tex, flags=re.S)
    tex=re.sub(r'(\\begin\{verbatim\}.*?\\end\{verbatim\})', lambda m:'\\par\\noindent\\begin{minipage}{\\linewidth}\n'+m[0]+'\n\\end{minipage}\n', tex, flags=re.S)
    (P/(name+'.tex')).write_text(tex)
    tmp.unlink()
    subprocess.run(['latexmk','-xelatex','-interaction=nonstopmode','-halt-on-error','-outdir=build',name+'.tex'],cwd=P,check=True,stdout=(P/(name+'-build.log')).open('w'),stderr=subprocess.STDOUT)
    shutil.copy2(P/'build'/(name+'.pdf'),P/('evotrace.pdf' if language=='en' else 'evotrace-中文.pdf'))
    print(name,'complete')
