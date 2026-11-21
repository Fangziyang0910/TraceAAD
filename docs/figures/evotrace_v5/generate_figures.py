"""Generate publication-style schematic figures for EvoTrace V5 design doc via headless Chrome SVG rendering."""

from pathlib import Path
import subprocess
import tempfile
from PIL import Image

OUT_DIR = Path(__file__).resolve().parent

BASE_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: #FFFFFF;
  font-family: "Noto Sans CJK SC", "Noto Sans", sans-serif;
  color: #1E293B;
  display: inline-block;
}
svg { display: block; background: #FFFFFF; }
text { font-family: "Noto Sans CJK SC", "Noto Sans", sans-serif; }
.mono { font-family: "Noto Sans Mono CJK SC", "Maple Mono NF CN", monospace; }
"""


def sub(base: str, subscript: str, size: float = 9.5) -> str:
    return f'{base}<tspan dy="3" font-size="{size}">{subscript}</tspan><tspan dy="-3">&#8203;</tspan>'


def render_svg(svg_body: str, width: int, height: int, out_png: Path) -> None:
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<style>{BASE_CSS}</style>
</head>
<body>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
{svg_body}
</svg>
</body>
</html>"""
    with tempfile.TemporaryDirectory() as tmp:
        html_path = Path(tmp) / "fig.html"
        raw_png = Path(tmp) / "raw.png"
        html_path.write_text(html, encoding="utf-8")
        subprocess.run(
            [
                "google-chrome",
                "--headless=new",
                "--disable-gpu",
                "--no-sandbox",
                "--hide-scrollbars",
                f"--window-size={width},{height}",
                "--force-device-scale-factor=2",
                f"--screenshot={raw_png}",
                html_path.as_uri(),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        img = Image.open(raw_png).convert("RGB")
        img = img.crop((0, 0, min(img.width, width * 2), min(img.height, height * 2)))
        img.save(out_png, format="PNG", dpi=(300, 300))


DEFS_ARROWS = """
<defs>
  <marker id="arr-dark" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">
    <path d="M 0 1.2 L 9 5 L 0 8.8 z" fill="#334155"/>
  </marker>
  <marker id="arr-teal" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">
    <path d="M 0 1.2 L 9 5 L 0 8.8 z" fill="#0F766E"/>
  </marker>
  <marker id="arr-blue" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M 0 1.2 L 9 5 L 0 8.8 z" fill="#2563EB"/>
  </marker>
  <marker id="arr-amber" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">
    <path d="M 0 1.2 L 9 5 L 0 8.8 z" fill="#EA580C"/>
  </marker>
  <marker id="arr-muted" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
    <path d="M 0 1.2 L 9 5 L 0 8.8 z" fill="#94A3B8"/>
  </marker>
</defs>
"""


def build_fig1() -> tuple[str, int, int]:
    """图 1：EvoTrace V5 整体机制流程示意"""
    w, h = 1360, 670
    mD = sub("m", "D")
    sE = sub("s", "E")
    dT = sub("d", "T", 9)
    pr = sub("p", "r", 9)
    sr = sub("s̃", "r", 9)
    nr = sub("n", "r", 9)
    Sr = sub("S̃", "r", 9)
    ur = sub("û", "r", 9)
    ptr = sub("p̃", "r", 9)
    ar = sub("a", "r", 9)

    svg = DEFS_ARROWS + f"""
    <text x="{w/2}" y="38" text-anchor="middle" font-size="23.5" font-weight="700" fill="#0F172A">EvoTrace V5 整体机制流程：树状种群、反思式淘汰、谱系上下文与分叉预算分配</text>
    <text x="{w/2}" y="62" text-anchor="middle" font-size="13.5" fill="#475569">纯黑盒目标评价 · 每次归约得到的树直接作为下一轮现役树 · 图中节点与分数均为示意示例</text>

    <!-- Panel A -->
    <rect x="24" y="84" width="424" height="472" rx="14" fill="#FFFFFF" stroke="#0D9488" stroke-width="2"/>
    <path d="M 24 98 Q 24 84 38 84 L 434 84 Q 448 84 448 98 L 448 128 L 24 128 Z" fill="#0F766E"/>
    <text x="236" y="112" text-anchor="middle" font-size="16" font-weight="700" fill="#FFFFFF">(a) 树状种群与局部反思式淘汰</text>

    <!-- Highlight regions (well inside panel borders) -->
    <rect x="94" y="250" width="76" height="106" rx="12" fill="#FFF7ED" stroke="#FDBA74" stroke-width="1.5" stroke-dasharray="5,4"/>
    <text x="64" y="302" text-anchor="middle" font-size="11.5" font-weight="700" fill="#C2410C">路径</text>
    <text x="64" y="318" text-anchor="middle" font-size="11.5" font-weight="700" fill="#C2410C">收缩</text>

    <rect x="258" y="318" width="168" height="80" rx="12" fill="#FEF3C7" stroke="#F59E0B" stroke-width="1.5" stroke-dasharray="5,4"/>
    <text x="342" y="416" text-anchor="middle" font-size="11.5" font-weight="700" fill="#B45309">分支剪枝（至少 2 弱叶）</text>

    <!-- Edges in Panel A -->
    <line x1="236" y1="168" x2="162" y2="205" stroke="#334155" stroke-width="2" marker-end="url(#arr-dark)"/>
    <line x1="236" y1="168" x2="338" y2="205" stroke="#334155" stroke-width="2" marker-end="url(#arr-dark)"/>
    <line x1="150" y1="236" x2="134" y2="262" stroke="#94A3B8" stroke-width="1.8" marker-end="url(#arr-muted)"/>
    <line x1="132" y1="296" x2="132" y2="318" stroke="#94A3B8" stroke-width="1.8" marker-end="url(#arr-muted)"/>
    <line x1="132" y1="354" x2="132" y2="386" stroke="#94A3B8" stroke-width="1.8" marker-end="url(#arr-muted)"/>
    <!-- Reconnected edge A -> D -->
    <path d="M 172 232 Q 214 316 158 402" fill="none" stroke="#0F766E" stroke-width="2.4" stroke-dasharray="6,3" marker-end="url(#arr-teal)"/>
    <!-- Branch edges E -> X, Y -->
    <line x1="328" y1="236" x2="304" y2="336" stroke="#94A3B8" stroke-width="1.8" marker-end="url(#arr-muted)"/>
    <line x1="350" y1="236" x2="378" y2="336" stroke="#94A3B8" stroke-width="1.8" marker-end="url(#arr-muted)"/>

    <!-- Virtual root -->
    <rect x="218" y="142" width="36" height="26" rx="6" fill="#F8FAFC" stroke="#334155" stroke-width="1.8"/>
    <text x="236" y="160" text-anchor="middle" font-size="14" font-weight="700" fill="#0F172A">v₀</text>

    <!-- Nodes -->
    <circle cx="158" cy="220" r="20" fill="#0F766E" stroke="#134E4A" stroke-width="2"/>
    <text x="158" y="225" text-anchor="middle" font-size="14" font-weight="700" fill="#FFFFFF">A</text>
    <circle cx="132" cy="278" r="17" fill="#E2E8F0" stroke="#475569" stroke-width="1.8"/>
    <text x="132" y="283" text-anchor="middle" font-size="13" font-weight="600" fill="#334155">B</text>
    <circle cx="132" cy="336" r="17" fill="#E2E8F0" stroke="#475569" stroke-width="1.8"/>
    <text x="132" y="341" text-anchor="middle" font-size="13" font-weight="600" fill="#334155">C</text>
    <circle cx="132" cy="408" r="20" fill="#0F766E" stroke="#134E4A" stroke-width="2"/>
    <text x="132" y="413" text-anchor="middle" font-size="14" font-weight="700" fill="#FFFFFF">D</text>

    <circle cx="338" cy="220" r="20" fill="#0F766E" stroke="#134E4A" stroke-width="2"/>
    <text x="338" y="225" text-anchor="middle" font-size="14" font-weight="700" fill="#FFFFFF">E</text>
    <circle cx="298" cy="358" r="17" fill="#E2E8F0" stroke="#475569" stroke-width="1.8"/>
    <text x="298" y="363" text-anchor="middle" font-size="13" font-weight="600" fill="#334155">X</text>
    <circle cx="384" cy="358" r="17" fill="#E2E8F0" stroke="#475569" stroke-width="1.8"/>
    <text x="384" y="363" text-anchor="middle" font-size="13" font-weight="600" fill="#334155">Y</text>

    <!-- Experience badges -->
    <rect x="190" y="302" width="48" height="26" rx="7" fill="#FFF7ED" stroke="#EA580C" stroke-width="1.8"/>
    <text x="214" y="320" text-anchor="middle" font-size="13" font-weight="700" fill="#9A3412">{mD}</text>
    <rect x="372" y="207" width="46" height="26" rx="7" fill="#FFF7ED" stroke="#EA580C" stroke-width="1.8"/>
    <text x="395" y="225" text-anchor="middle" font-size="13" font-weight="700" fill="#9A3412">{sE}</text>
    <line x1="358" y1="220" x2="372" y2="220" stroke="#EA580C" stroke-width="1.6"/>

    <!-- Bottom summary box inside Panel A -->
    <rect x="42" y="446" width="388" height="94" rx="10" fill="#F8FAFC" stroke="#CBD5E1" stroke-width="1.2"/>
    <text x="56" y="469" font-size="12.5" font-weight="700" fill="#0F172A">淘汰同时完成经验凝练（无状态、无静态容量上限）：</text>
    <text x="56" y="491" font-size="12" fill="#334155">• 路径收缩：D 严格胜过 B, C → 重连 A→D，写边经验 {mD}</text>
    <text x="56" y="511" font-size="12" fill="#334155">• 分支剪枝：E 严格胜过弱叶 X, Y → 收回至 E，写节点经验 {sE}</text>
    <text x="56" y="531" font-size="12" fill="#334155">• 后续再次归约时，已有经验随新对照原位修订</text>

    <!-- Arrow A -> B -->
    <line x1="448" y1="312" x2="486" y2="312" stroke="#0F766E" stroke-width="2.6" marker-end="url(#arr-teal)"/>
    <text x="467" y="298" text-anchor="middle" font-size="11.5" font-weight="600" fill="#0F766E">归约后</text>
    <text x="467" y="330" text-anchor="middle" font-size="11.5" font-weight="600" fill="#0F766E">活树</text>

    <!-- Panel B -->
    <rect x="490" y="84" width="374" height="472" rx="14" fill="#FFFFFF" stroke="#2563EB" stroke-width="2"/>
    <path d="M 490 98 Q 490 84 504 84 L 850 84 Q 864 84 864 98 L 864 128 L 490 128 Z" fill="#1D4ED8"/>
    <text x="677" y="112" text-anchor="middle" font-size="16" font-weight="700" fill="#FFFFFF">(b) 分叉局部非平稳后验预算分配</text>

    <!-- Routing diagram inside Panel B -->
    <rect x="659" y="146" width="36" height="26" rx="6" fill="#F8FAFC" stroke="#334155" stroke-width="1.8"/>
    <text x="677" y="164" text-anchor="middle" font-size="14" font-weight="700" fill="#0F172A">v₀</text>
    <line x1="665" y1="172" x2="606" y2="210" stroke="#2563EB" stroke-width="2.6" marker-end="url(#arr-blue)"/>
    <line x1="689" y1="172" x2="756" y2="210" stroke="#94A3B8" stroke-width="1.8" marker-end="url(#arr-muted)"/>

    <circle cx="598" cy="228" r="20" fill="#0F766E" stroke="#1D4ED8" stroke-width="3"/>
    <text x="598" y="233" text-anchor="middle" font-size="14" font-weight="700" fill="#FFFFFF">A</text>
    <circle cx="764" cy="228" r="20" fill="#0F766E" stroke="#134E4A" stroke-width="2"/>
    <text x="764" y="233" text-anchor="middle" font-size="14" font-weight="700" fill="#FFFFFF">E</text>
    <rect x="794" y="216" width="44" height="24" rx="6" fill="#FFF7ED" stroke="#EA580C" stroke-width="1.5"/>
    <text x="816" y="232" text-anchor="middle" font-size="12" font-weight="700" fill="#9A3412">{sE}</text>

    <path d="M 578 238 Q 532 260 546 300" fill="none" stroke="#94A3B8" stroke-width="1.8" stroke-dasharray="4,3" marker-end="url(#arr-muted)"/>
    <text x="540" y="318" text-anchor="middle" font-size="11.5" fill="#475569">直接改写 A</text>
    <line x1="606" y1="248" x2="640" y2="302" stroke="#2563EB" stroke-width="2.6" marker-end="url(#arr-blue)"/>
    <rect x="628" y="264" width="46" height="23" rx="6" fill="#FFF7ED" stroke="#EA580C" stroke-width="1.5"/>
    <text x="651" y="280" text-anchor="middle" font-size="12" font-weight="700" fill="#9A3412">{mD}</text>

    <circle cx="648" cy="324" r="21" fill="#2563EB" stroke="#1E3A8A" stroke-width="2.5"/>
    <text x="648" y="329" text-anchor="middle" font-size="14" font-weight="700" fill="#FFFFFF">D</text>
    <rect x="680" y="310" width="138" height="28" rx="7" fill="#EFF6FF" stroke="#2563EB" stroke-width="1.5"/>
    <text x="749" y="329" text-anchor="middle" font-size="12" font-weight="700" fill="#1E40AF">选中主父代 p = D</text>

    <!-- Formula card inside Panel B -->
    <rect x="506" y="366" width="342" height="174" rx="10" fill="#F8FAFC" stroke="#CBD5E1" stroke-width="1.2"/>
    <text x="520" y="390" font-size="12.8" font-weight="700" fill="#0F172A">每个分叉口结合“分叉最优标杆 × 历史步幅”：</text>
    <text x="520" y="414" font-size="12" fill="#334155">1. 按当前差距折算历史进步：x = max(0, g − δ)</text>
    <text x="520" y="436" font-size="12" fill="#334155">2. 遗忘因子 γ 仅衰减旧突破，总评数 n 不衰减：</text>
    <rect x="518" y="447" width="318" height="38" rx="7" fill="#EFF6FF" stroke="#BFDBFE" stroke-width="1.2"/>
    <text x="677" y="470" text-anchor="middle" font-size="12.5" font-weight="700" fill="#1E3A8A">{pr} ~ Beta({sr}+1, {nr}−{sr}+1),  {ur} = (d+{Sr})/({nr}+2)</text>
    <text x="520" y="507" font-size="12" fill="#334155">3. 汤普森抽样选 {ptr} · {ar} 最大方向继续向下路由</text>
    <text x="520" y="528" font-size="12" fill="#334155">• 路径收缩时，终点新边直接承接原方向的子路账本</text>

    <!-- Arrow B -> C -->
    <line x1="864" y1="312" x2="904" y2="312" stroke="#2563EB" stroke-width="2.6" marker-end="url(#arr-blue)"/>
    <text x="884" y="298" text-anchor="middle" font-size="11.5" font-weight="600" fill="#1D4ED8">主父代</text>
    <text x="884" y="330" text-anchor="middle" font-size="11.5" font-weight="600" fill="#1D4ED8">p = D</text>

    <!-- Panel C -->
    <rect x="908" y="84" width="428" height="472" rx="14" fill="#FFFFFF" stroke="#0D9488" stroke-width="2"/>
    <path d="M 908 98 Q 908 84 922 84 L 1322 84 Q 1336 84 1336 98 L 1336 128 L 908 128 Z" fill="#0F766E"/>
    <text x="1122" y="112" text-anchor="middle" font-size="16" font-weight="700" fill="#FFFFFF">(c) 演化脉络上下文与三算子生成</text>

    <rect x="928" y="146" width="388" height="62" rx="10" fill="#FFF7ED" stroke="#EA580C" stroke-width="1.5"/>
    <text x="1122" y="170" text-anchor="middle" font-size="13" font-weight="700" fill="#9A3412">围绕主父代 p 读取树上谱系、经验与参考</text>
    <text x="1122" y="192" text-anchor="middle" font-size="11.8" fill="#7C2D12">纵向形成路径 + 边/节点经验 + 树距离 {dT} 选出的跨分支参考</text>

    <rect x="928" y="226" width="122" height="106" rx="10" fill="#EFF6FF" stroke="#3B82F6" stroke-width="1.6"/>
    <text x="989" y="250" text-anchor="middle" font-size="13.5" font-weight="700" fill="#1E40AF">精化 Refine</text>
    <text x="989" y="272" text-anchor="middle" font-size="11.5" fill="#334155">读纵向来路</text>
    <text x="989" y="291" text-anchor="middle" font-size="11.5" fill="#334155">与本地试错</text>
    <text x="989" y="314" text-anchor="middle" font-size="11" font-weight="600" fill="#1D4ED8">先验 0.50</text>

    <rect x="1061" y="226" width="122" height="106" rx="10" fill="#EFF6FF" stroke="#3B82F6" stroke-width="1.6"/>
    <text x="1122" y="250" text-anchor="middle" font-size="13.5" font-weight="700" fill="#1E40AF">交叉 Crossover</text>
    <text x="1122" y="272" text-anchor="middle" font-size="11.5" fill="#334155">按 {dT} 选跨分支</text>
    <text x="1122" y="291" text-anchor="middle" font-size="11.5" fill="#334155">供体（可较弱）</text>
    <text x="1122" y="314" text-anchor="middle" font-size="11" font-weight="600" fill="#1D4ED8">先验 0.30</text>

    <rect x="1194" y="226" width="122" height="106" rx="10" fill="#EFF6FF" stroke="#3B82F6" stroke-width="1.6"/>
    <text x="1255" y="250" text-anchor="middle" font-size="13.5" font-weight="700" fill="#1E40AF">探索 Explore</text>
    <text x="1255" y="272" text-anchor="middle" font-size="11.5" fill="#334155">各分支代表简述</text>
    <text x="1255" y="291" text-anchor="middle" font-size="11.5" fill="#334155">隐藏参考源码</text>
    <text x="1255" y="314" text-anchor="middle" font-size="11" font-weight="600" fill="#1D4ED8">先验 0.20</text>

    <line x1="989" y1="208" x2="989" y2="224" stroke="#EA580C" stroke-width="1.6" marker-end="url(#arr-amber)"/>
    <line x1="1122" y1="208" x2="1122" y2="224" stroke="#EA580C" stroke-width="1.6" marker-end="url(#arr-amber)"/>
    <line x1="1255" y1="208" x2="1255" y2="224" stroke="#EA580C" stroke-width="1.6" marker-end="url(#arr-amber)"/>

    <line x1="989" y1="332" x2="989" y2="358" stroke="#334155" stroke-width="1.6"/>
    <line x1="1122" y1="332" x2="1122" y2="372" stroke="#334155" stroke-width="1.8" marker-end="url(#arr-dark)"/>
    <line x1="1255" y1="332" x2="1255" y2="358" stroke="#334155" stroke-width="1.6"/>
    <line x1="989" y1="358" x2="1255" y2="358" stroke="#334155" stroke-width="1.6"/>

    <rect x="948" y="376" width="348" height="58" rx="10" fill="#DBEAFE" stroke="#2563EB" stroke-width="1.8"/>
    <text x="1122" y="401" text-anchor="middle" font-size="14" font-weight="700" fill="#1E3A8A">LLM 生成候选算法 + 纯黑盒训练评价 f(v)</text>
    <text x="1122" y="422" text-anchor="middle" font-size="11.5" fill="#1E40AF">按实扣减评价预算，候选程序永久归档</text>

    <line x1="1122" y1="434" x2="1122" y2="466" stroke="#2563EB" stroke-width="2" marker-end="url(#arr-blue)"/>
    <circle cx="1044" cy="496" r="19" fill="#0F766E" stroke="#134E4A" stroke-width="2"/>
    <text x="1044" y="501" text-anchor="middle" font-size="13.5" font-weight="700" fill="#FFFFFF">D</text>
    <line x1="1063" y1="496" x2="1098" y2="496" stroke="#2563EB" stroke-width="2.2" marker-end="url(#arr-blue)"/>
    <circle cx="1122" cy="496" r="20" fill="#2563EB" stroke="#1E3A8A" stroke-width="2"/>
    <text x="1122" y="501" text-anchor="middle" font-size="14" font-weight="700" fill="#FFFFFF">v</text>
    <rect x="1156" y="476" width="150" height="40" rx="8" fill="#F8FAFC" stroke="#CBD5E1" stroke-width="1.4"/>
    <text x="1231" y="494" text-anchor="middle" font-size="11.8" font-weight="600" fill="#0F172A">新可行子代接在 p 下</text>
    <text x="1231" y="510" text-anchor="middle" font-size="11" fill="#475569">同步更新分叉账本</text>

    <!-- Bottom feedback loop arrow -->
    <path d="M 1122 556 L 1122 596 L 236 596 L 236 562" fill="none" stroke="#2563EB" stroke-width="3" marker-end="url(#arr-blue)"/>
    <rect x="498" y="580" width="364" height="30" rx="15" fill="#EFF6FF" stroke="#2563EB" stroke-width="1.6"/>
    <text x="680" y="600" text-anchor="middle" font-size="13" font-weight="700" fill="#1E40AF">新节点入树并更新账本 → 立即触发下一轮无状态归约与反思</text>

    <!-- Legend -->
    <circle cx="310" cy="642" r="10" fill="#0F766E" stroke="#134E4A" stroke-width="1.5"/>
    <text x="328" y="647" font-size="12.5" fill="#334155">现役保留算法</text>
    <circle cx="468" cy="642" r="10" fill="#E2E8F0" stroke="#475569" stroke-width="1.5"/>
    <text x="486" y="647" font-size="12.5" fill="#334155">归约移除算法（永久留档）</text>
    <rect x="676" y="631" width="34" height="21" rx="5" fill="#FFF7ED" stroke="#EA580C" stroke-width="1.6"/>
    <text x="693" y="645" text-anchor="middle" font-size="11" font-weight="700" fill="#9A3412">m/s</text>
    <text x="718" y="647" font-size="12.5" fill="#334155">原位凝练的边/节点经验</text>
    <circle cx="910" cy="642" r="10" fill="#2563EB" stroke="#1E3A8A" stroke-width="1.5"/>
    <text x="928" y="647" font-size="12.5" fill="#334155">本轮选中父代 / 新生子代</text>
    """
    return svg, w, h


def build_fig2() -> tuple[str, int, int]:
    """图 2：无状态局部归约与原位双形态反思示意"""
    w, h = 1360, 636
    sB = sub("s", "B")
    sP = sub("s", "P")
    sA = sub("s", "A")
    mD = sub("m", "D")

    svg = DEFS_ARROWS + f"""
    <text x="{w/2}" y="38" text-anchor="middle" font-size="24" font-weight="700" fill="#0F172A">无状态局部归约与原位双形态反思示意</text>
    <text x="{w/2}" y="62" text-anchor="middle" font-size="13.5" fill="#475569">归约规则仅依赖当前树拓扑与黑盒目标值（越小越好，图中均为示意数值）</text>

    <!-- Panel A: Branch Pruning -->
    <rect x="24" y="84" width="420" height="484" rx="14" fill="#FFFFFF" stroke="#0D9488" stroke-width="2"/>
    <path d="M 24 98 Q 24 84 38 84 L 430 84 Q 444 84 444 98 L 444 128 L 24 128 Z" fill="#0F766E"/>
    <text x="234" y="112" text-anchor="middle" font-size="15.5" font-weight="700" fill="#FFFFFF">(a) 分支剪枝：单弱叶保留，双弱叶收回</text>

    <!-- Case 1 -->
    <rect x="42" y="144" width="384" height="108" rx="10" fill="#F8FAFC" stroke="#CBD5E1" stroke-width="1.2"/>
    <text x="56" y="167" font-size="12.5" font-weight="700" fill="#0F172A">情形 1：单链连续退步 → 整支保留（|W| &lt; 2）</text>
    <line x1="112" y1="210" x2="174" y2="210" stroke="#334155" stroke-width="2" marker-end="url(#arr-dark)"/>
    <line x1="216" y1="210" x2="278" y2="210" stroke="#334155" stroke-width="2" marker-end="url(#arr-dark)"/>
    <circle cx="92" cy="210" r="20" fill="#0F766E" stroke="#134E4A" stroke-width="1.8"/>
    <text x="92" y="208" text-anchor="middle" font-size="12.5" font-weight="700" fill="#FFFFFF">P</text>
    <text x="92" y="222" text-anchor="middle" font-size="10.5" fill="#CCFBF1">7.2</text>
    <circle cx="196" cy="210" r="20" fill="#0F766E" stroke="#134E4A" stroke-width="1.8"/>
    <text x="196" y="208" text-anchor="middle" font-size="12.5" font-weight="700" fill="#FFFFFF">B</text>
    <text x="196" y="222" text-anchor="middle" font-size="10.5" fill="#CCFBF1">7.7</text>
    <circle cx="300" cy="210" r="20" fill="#0F766E" stroke="#134E4A" stroke-width="1.8"/>
    <text x="300" y="208" text-anchor="middle" font-size="12.5" font-weight="700" fill="#FFFFFF">C</text>
    <text x="300" y="222" text-anchor="middle" font-size="10.5" fill="#CCFBF1">8.1</text>
    <rect x="336" y="195" width="76" height="30" rx="7" fill="#ECFDF5" stroke="#059669" stroke-width="1.4"/>
    <text x="374" y="215" text-anchor="middle" font-size="12" font-weight="700" fill="#065F46">全部保留</text>

    <!-- Case 2 -->
    <rect x="42" y="268" width="384" height="280" rx="10" fill="#F8FAFC" stroke="#CBD5E1" stroke-width="1.2"/>
    <text x="56" y="291" font-size="12.5" font-weight="700" fill="#0F172A">情形 2：B 长出第二弱叶 D → 收回 {{C, D}}，留 B 续用 {sB}</text>

    <rect x="148" y="376" width="94" height="104" rx="10" fill="#FEF3C7" stroke="#F59E0B" stroke-width="1.4" stroke-dasharray="4,3"/>
    <line x1="94" y1="360" x2="122" y2="404" stroke="#334155" stroke-width="1.8" marker-end="url(#arr-dark)"/>
    <line x1="146" y1="416" x2="174" y2="404" stroke="#94A3B8" stroke-width="1.8" marker-end="url(#arr-muted)"/>
    <line x1="146" y1="432" x2="174" y2="450" stroke="#94A3B8" stroke-width="1.8" marker-end="url(#arr-muted)"/>
    <circle cx="84" cy="344" r="19" fill="#0F766E" stroke="#134E4A" stroke-width="1.8"/>
    <text x="84" y="342" text-anchor="middle" font-size="12" font-weight="700" fill="#FFFFFF">P</text>
    <text x="84" y="355" text-anchor="middle" font-size="10" fill="#CCFBF1">7.2</text>
    <circle cx="132" cy="422" r="19" fill="#0F766E" stroke="#134E4A" stroke-width="1.8"/>
    <text x="132" y="420" text-anchor="middle" font-size="12" font-weight="700" fill="#FFFFFF">B</text>
    <text x="132" y="433" text-anchor="middle" font-size="10" fill="#CCFBF1">7.7</text>
    <circle cx="196" cy="400" r="18" fill="#E2E8F0" stroke="#475569" stroke-width="1.6"/>
    <text x="196" y="398" text-anchor="middle" font-size="11.5" font-weight="600" fill="#334155">C</text>
    <text x="196" y="411" text-anchor="middle" font-size="9.5" fill="#475569">8.1</text>
    <circle cx="196" cy="454" r="18" fill="#E2E8F0" stroke="#475569" stroke-width="1.6"/>
    <text x="196" y="452" text-anchor="middle" font-size="11.5" font-weight="600" fill="#334155">D</text>
    <text x="196" y="465" text-anchor="middle" font-size="9.5" fill="#475569">8.3</text>

    <line x1="252" y1="422" x2="294" y2="422" stroke="#EA580C" stroke-width="2.2" marker-end="url(#arr-amber)"/>
    <text x="273" y="410" text-anchor="middle" font-size="11" font-weight="700" fill="#C2410C">剪枝</text>
    <text x="273" y="440" text-anchor="middle" font-size="10.5" fill="#C2410C">+反思</text>

    <line x1="330" y1="363" x2="330" y2="400" stroke="#334155" stroke-width="2" marker-end="url(#arr-dark)"/>
    <circle cx="330" cy="344" r="19" fill="#0F766E" stroke="#134E4A" stroke-width="1.8"/>
    <text x="330" y="342" text-anchor="middle" font-size="12" font-weight="700" fill="#FFFFFF">P</text>
    <text x="330" y="355" text-anchor="middle" font-size="10" fill="#CCFBF1">7.2</text>
    <circle cx="330" cy="422" r="19" fill="#0F766E" stroke="#134E4A" stroke-width="1.8"/>
    <text x="330" y="420" text-anchor="middle" font-size="12" font-weight="700" fill="#FFFFFF">B</text>
    <text x="330" y="433" text-anchor="middle" font-size="10" fill="#CCFBF1">7.7</text>
    <rect x="360" y="410" width="46" height="24" rx="6" fill="#FFF7ED" stroke="#EA580C" stroke-width="1.6"/>
    <text x="383" y="426" text-anchor="middle" font-size="12" font-weight="700" fill="#9A3412">{sB}</text>
    <line x1="349" y1="422" x2="360" y2="422" stroke="#EA580C" stroke-width="1.5"/>

    <text x="234" y="510" text-anchor="middle" font-size="11.5" fill="#475569">无跨步隐藏标记：B 作为 P 的单弱叶暂留，可借助 {sB} 尝试回升；</text>
    <text x="234" y="530" text-anchor="middle" font-size="11.5" fill="#475569">若 P 后续再长出第二弱叶 E，则 {{B, E}} 再一并收回并整合进 {sP}。</text>

    <!-- Panel B -->
    <rect x="468" y="84" width="464" height="484" rx="14" fill="#FFFFFF" stroke="#0D9488" stroke-width="2"/>
    <path d="M 468 98 Q 468 84 482 84 L 918 84 Q 932 84 932 98 L 932 128 L 468 128 Z" fill="#0F766E"/>
    <text x="700" y="112" text-anchor="middle" font-size="15.5" font-weight="700" fill="#FFFFFF">(b) 路径收缩与“收缩当场触发剪枝”的结构封闭性</text>

    <!-- Step 0: Before (generous vertical clearance between B and E) -->
    <rect x="486" y="144" width="428" height="132" rx="10" fill="#F8FAFC" stroke="#CBD5E1" stroke-width="1.2"/>
    <text x="500" y="165" font-size="12" font-weight="700" fill="#0F172A">归约前：B 有孩子 D，A 下仅 E 一个弱叶（暂不可剪枝）</text>
    <rect x="614" y="176" width="54" height="46" rx="10" fill="#FFF7ED" stroke="#F97316" stroke-width="1.4" stroke-dasharray="4,3"/>
    <line x1="554" y1="199" x2="618" y2="199" stroke="#334155" stroke-width="1.8" marker-end="url(#arr-dark)"/>
    <line x1="660" y1="199" x2="726" y2="199" stroke="#334155" stroke-width="1.8" marker-end="url(#arr-dark)"/>
    <line x1="550" y1="214" x2="620" y2="246" stroke="#334155" stroke-width="1.8" marker-end="url(#arr-dark)"/>
    <circle cx="534" cy="199" r="19" fill="#0F766E" stroke="#134E4A" stroke-width="1.8"/>
    <text x="534" y="197" text-anchor="middle" font-size="12" font-weight="700" fill="#FFFFFF">A</text>
    <text x="534" y="210" text-anchor="middle" font-size="10" fill="#CCFBF1">7.2</text>
    <circle cx="641" cy="199" r="17" fill="#E2E8F0" stroke="#475569" stroke-width="1.6"/>
    <text x="641" y="197" text-anchor="middle" font-size="11.5" font-weight="600" fill="#334155">B</text>
    <text x="641" y="209" text-anchor="middle" font-size="9.5" fill="#475569">8.1</text>
    <circle cx="748" cy="199" r="19" fill="#0F766E" stroke="#134E4A" stroke-width="1.8"/>
    <text x="748" y="197" text-anchor="middle" font-size="12" font-weight="700" fill="#FFFFFF">D</text>
    <text x="748" y="210" text-anchor="middle" font-size="10" fill="#CCFBF1">7.6</text>
    <circle cx="641" cy="252" r="17" fill="#0F766E" stroke="#134E4A" stroke-width="1.7"/>
    <text x="641" y="250" text-anchor="middle" font-size="11.5" font-weight="700" fill="#FFFFFF">E</text>
    <text x="641" y="262" text-anchor="middle" font-size="9.5" fill="#CCFBF1">7.5</text>
    <text x="840" y="204" text-anchor="middle" font-size="11.5" font-weight="600" fill="#C2410C">D(7.6) ≻ B(8.1)</text>
    <text x="840" y="222" text-anchor="middle" font-size="11" fill="#475569">单链满足收缩</text>

    <!-- Down arrow 1 -->
    <line x1="700" y1="276" x2="700" y2="299" stroke="#0F766E" stroke-width="2.2" marker-end="url(#arr-teal)"/>
    <text x="712" y="292" font-size="11" font-weight="700" fill="#0F766E">第 1 步：路径收缩 A→B→D，写边经验 {mD}</text>

    <!-- Step 1: After contraction -->
    <rect x="486" y="303" width="428" height="114" rx="10" fill="#F8FAFC" stroke="#CBD5E1" stroke-width="1.2"/>
    <text x="500" y="324" font-size="12" font-weight="700" fill="#0F172A">重连后即普通树：D 与 E 同成 A 的弱叶（|W(A)| = 2）</text>
    <rect x="652" y="335" width="126" height="72" rx="10" fill="#FEF3C7" stroke="#F59E0B" stroke-width="1.4" stroke-dasharray="4,3"/>
    <line x1="554" y1="368" x2="660" y2="354" stroke="#0F766E" stroke-width="2.2" marker-end="url(#arr-teal)"/>
    <line x1="554" y1="376" x2="660" y2="390" stroke="#334155" stroke-width="1.8" marker-end="url(#arr-dark)"/>
    <rect x="578" y="334" width="44" height="22" rx="6" fill="#FFF7ED" stroke="#EA580C" stroke-width="1.5"/>
    <text x="600" y="349" text-anchor="middle" font-size="11.5" font-weight="700" fill="#9A3412">{mD}</text>
    <circle cx="534" cy="372" r="19" fill="#0F766E" stroke="#134E4A" stroke-width="1.8"/>
    <text x="534" y="370" text-anchor="middle" font-size="12" font-weight="700" fill="#FFFFFF">A</text>
    <text x="534" y="383" text-anchor="middle" font-size="10" fill="#CCFBF1">7.2</text>
    <circle cx="682" cy="353" r="16.5" fill="#E2E8F0" stroke="#475569" stroke-width="1.6"/>
    <text x="682" y="351" text-anchor="middle" font-size="11" font-weight="600" fill="#334155">D</text>
    <text x="682" y="363" text-anchor="middle" font-size="9.5" fill="#475569">7.6</text>
    <circle cx="682" cy="391" r="16.5" fill="#E2E8F0" stroke="#475569" stroke-width="1.6"/>
    <text x="682" y="389" text-anchor="middle" font-size="11" font-weight="600" fill="#334155">E</text>
    <text x="682" y="401" text-anchor="middle" font-size="9.5" fill="#475569">7.5</text>
    <text x="844" y="365" text-anchor="middle" font-size="11.5" font-weight="600" fill="#B45309">A ≻ D 且 A ≻ E</text>
    <text x="844" y="383" text-anchor="middle" font-size="11" fill="#475569">当场满足剪枝</text>

    <!-- Down arrow 2 -->
    <line x1="700" y1="417" x2="700" y2="440" stroke="#EA580C" stroke-width="2.2" marker-end="url(#arr-amber)"/>
    <text x="712" y="433" font-size="11" font-weight="700" fill="#C2410C">第 2 步：立即剪枝 {{D, E}}，将 {mD} 融入 {sA}</text>

    <!-- Step 2 -->
    <rect x="486" y="444" width="428" height="104" rx="10" fill="#ECFDF5" stroke="#0D9488" stroke-width="1.5"/>
    <text x="500" y="466" font-size="12" font-weight="700" fill="#065F46">到达不动点：树保持紧凑，四份实测对照均已沉淀于 {sA}</text>
    <circle cx="574" cy="508" r="21" fill="#0F766E" stroke="#134E4A" stroke-width="2"/>
    <text x="574" y="506" text-anchor="middle" font-size="13" font-weight="700" fill="#FFFFFF">A</text>
    <text x="574" y="520" text-anchor="middle" font-size="10" fill="#CCFBF1">7.2</text>
    <line x1="595" y1="508" x2="626" y2="508" stroke="#EA580C" stroke-width="2"/>
    <rect x="626" y="486" width="268" height="44" rx="8" fill="#FFF7ED" stroke="#EA580C" stroke-width="1.6"/>
    <text x="760" y="505" text-anchor="middle" font-size="12" font-weight="700" fill="#9A3412">节点经验 {sA}（整合 A, B, D, E 及 {mD}）</text>
    <text x="760" y="522" text-anchor="middle" font-size="11" fill="#7C2D12">记录两条退步方向及 B→D 局部恢复部件</text>

    <!-- Panel C -->
    <rect x="956" y="84" width="380" height="484" rx="14" fill="#FFFFFF" stroke="#EA580C" stroke-width="2"/>
    <path d="M 956 98 Q 956 84 970 84 L 1322 84 Q 1336 84 1336 98 L 1336 128 L 956 128 Z" fill="#C2410C"/>
    <text x="1146" y="112" text-anchor="middle" font-size="15.5" font-weight="700" fill="#FFFFFF">(c) 原位双形态反思与多层修订</text>

    <rect x="976" y="146" width="340" height="86" rx="10" fill="#F8FAFC" stroke="#CBD5E1" stroke-width="1.4"/>
    <text x="992" y="170" font-size="13" font-weight="700" fill="#0F172A">反思输入（以源码与实测为准）</text>
    <text x="992" y="193" font-size="12" fill="#334155">• 归约区域内各程序源码、代码差异与实测分</text>
    <text x="992" y="214" font-size="12" fill="#334155">• 区域内原有边经验 / 节点经验（如存在）</text>

    <line x1="1064" y1="232" x2="1064" y2="262" stroke="#EA580C" stroke-width="2" marker-end="url(#arr-amber)"/>
    <line x1="1228" y1="232" x2="1228" y2="262" stroke="#EA580C" stroke-width="2" marker-end="url(#arr-amber)"/>

    <rect x="976" y="266" width="164" height="130" rx="10" fill="#FFF7ED" stroke="#EA580C" stroke-width="1.6"/>
    <text x="1058" y="290" text-anchor="middle" font-size="13" font-weight="700" fill="#9A3412">路径反思 → 边 m</text>
    <text x="988" y="314" font-size="11.5" fill="#334155">回答“如何形成”：</text>
    <text x="988" y="335" font-size="11.5" fill="#334155">• 哪些改动留在终点</text>
    <text x="988" y="356" font-size="11.5" fill="#334155">• 哪步退步、哪步恢复</text>
    <text x="988" y="377" font-size="11.5" fill="#334155">• 纳入中间旁支观察</text>

    <rect x="1152" y="266" width="164" height="130" rx="10" fill="#FFF7ED" stroke="#EA580C" stroke-width="1.6"/>
    <text x="1234" y="290" text-anchor="middle" font-size="13" font-weight="700" fill="#9A3412">剪枝反思 → 点 s</text>
    <text x="1164" y="314" font-size="11.5" fill="#334155">回答“试出过什么”：</text>
    <text x="1164" y="335" font-size="11.5" fill="#334155">• 各失败方向何处不同</text>
    <text x="1164" y="356" font-size="11.5" fill="#334155">• 在何条件下导致退步</text>
    <text x="1164" y="377" font-size="11.5" fill="#334155">• 弱支路可借用的部件</text>

    <line x1="1146" y1="396" x2="1146" y2="424" stroke="#0F766E" stroke-width="2" marker-end="url(#arr-teal)"/>
    <rect x="976" y="428" width="340" height="120" rx="10" fill="#ECFDF5" stroke="#0D9488" stroke-width="1.5"/>
    <text x="992" y="454" font-size="13" font-weight="700" fill="#065F46">输出一段自然语言经验 + 多层动态修订</text>
    <text x="992" y="478" font-size="11.8" fill="#334155">• 只记适用条件、计算改动与实测结果，不下硬命令</text>
    <text x="992" y="500" font-size="11.8" fill="#334155">• 承载经验的节点/边再次被归约时，旧经验随新对照</text>
    <text x="992" y="520" font-size="11.8" fill="#334155">  重新凝练；若新反例推翻旧结论，直接收窄或撤回</text>

    <!-- Legend -->
    <circle cx="340" cy="602" r="10" fill="#0F766E" stroke="#134E4A" stroke-width="1.5"/>
    <text x="358" y="607" font-size="12.5" fill="#334155">现役节点（保留）</text>
    <circle cx="514" cy="602" r="10" fill="#E2E8F0" stroke="#475569" stroke-width="1.5"/>
    <text x="532" y="607" font-size="12.5" fill="#334155">本轮归约移除节点</text>
    <rect x="692" y="591" width="34" height="21" rx="5" fill="#FFF7ED" stroke="#EA580C" stroke-width="1.6"/>
    <text x="709" y="605" text-anchor="middle" font-size="11" font-weight="700" fill="#9A3412">m/s</text>
    <text x="734" y="607" font-size="12.5" fill="#334155">写入边/节点的凝练经验</text>
    <rect x="920" y="591" width="24" height="21" rx="4" fill="#FEF3C7" stroke="#F59E0B" stroke-width="1.4" stroke-dasharray="3,2"/>
    <text x="952" y="607" font-size="12.5" fill="#334155">待剪枝弱叶组</text>
    """
    return svg, w, h


def build_fig3() -> tuple[str, int, int]:
    """图 3：树谱系驱动的三算子选材与演化上下文切片示意"""
    w, h = 1360, 620
    dT = sub("d", "T", 9.5)
    m2 = sub("m", "2", 9)
    s2 = sub("s", "2", 9)

    svg = DEFS_ARROWS + f"""
    <text x="{w/2}" y="38" text-anchor="middle" font-size="24" font-weight="700" fill="#0F172A">演化脉络引导的上下文组织与三类算子选材示意</text>
    <text x="{w/2}" y="62" text-anchor="middle" font-size="13.5" fill="#475569">同一棵三分支现役树下，选中主父代 p = #2 时，Refine / Crossover / Explore 读取的树上信息切片（示意示例）</text>

    <!-- Left Panel -->
    <rect x="24" y="84" width="540" height="486" rx="14" fill="#FFFFFF" stroke="#0D9488" stroke-width="2"/>
    <path d="M 24 98 Q 24 84 38 84 L 550 84 Q 564 84 564 98 L 564 128 L 24 128 Z" fill="#0F766E"/>
    <text x="294" y="112" text-anchor="middle" font-size="15.5" font-weight="700" fill="#FFFFFF">现役树状种群（3 条顶级分支，主父代 p = #2）</text>

    <!-- Highlight boxes placed cleanly BELOW v0 edges -->
    <rect x="40" y="228" width="154" height="174" rx="12" fill="#EFF6FF" stroke="#3B82F6" stroke-width="1.5" stroke-dasharray="5,3"/>
    <rect x="206" y="228" width="342" height="258" rx="12" fill="#F0FDF4" stroke="#16A34A" stroke-width="1.5" stroke-dasharray="5,3"/>

    <!-- Virtual root -->
    <rect x="276" y="140" width="36" height="26" rx="6" fill="#F8FAFC" stroke="#334155" stroke-width="1.8"/>
    <text x="294" y="158" text-anchor="middle" font-size="14" font-weight="700" fill="#0F172A">v₀</text>

    <!-- Edges from v0 (pass cleanly before box labels) -->
    <line x1="276" y1="162" x2="132" y2="256" stroke="#334155" stroke-width="2" marker-end="url(#arr-dark)"/>
    <line x1="294" y1="166" x2="294" y2="254" stroke="#334155" stroke-width="2" marker-end="url(#arr-dark)"/>
    <line x1="312" y1="162" x2="454" y2="256" stroke="#334155" stroke-width="2" marker-end="url(#arr-dark)"/>

    <!-- Opaque title pills on bottom/top edges of dashed regions so NO line ever crosses text -->
    <rect x="52" y="216" width="100" height="20" rx="10" fill="#DBEAFE" stroke="#3B82F6" stroke-width="1.2"/>
    <text x="102" y="230" text-anchor="middle" font-size="10.5" font-weight="700" fill="#1E40AF">主父代纵向来路</text>

    <rect x="336" y="216" width="200" height="20" rx="10" fill="#DCFCE7" stroke="#16A34A" stroke-width="1.2"/>
    <text x="436" y="230" text-anchor="middle" font-size="10.5" font-weight="700" fill="#15803D">跨分支候选区（非直系亲属）</text>

    <!-- Branch 1: 1 -> 2 -> 3 -->
    <line x1="126" y1="296" x2="126" y2="340" stroke="#2563EB" stroke-width="2.6" marker-end="url(#arr-blue)"/>
    <line x1="126" y1="386" x2="126" y2="428" stroke="#334155" stroke-width="2" marker-end="url(#arr-dark)"/>
    <rect x="74" y="308" width="40" height="20" rx="5" fill="#FFF7ED" stroke="#EA580C" stroke-width="1.4"/>
    <text x="94" y="322" text-anchor="middle" font-size="11" font-weight="700" fill="#9A3412">{m2}</text>
    <rect x="154" y="352" width="32" height="20" rx="5" fill="#FFF7ED" stroke="#EA580C" stroke-width="1.4"/>
    <text x="170" y="366" text-anchor="middle" font-size="11" font-weight="700" fill="#9A3412">{s2}</text>

    <!-- Branch 2: 4 -> 5, 4 -> 8 -> 9 -->
    <line x1="284" y1="294" x2="258" y2="344" stroke="#334155" stroke-width="2" marker-end="url(#arr-dark)"/>
    <line x1="306" y1="294" x2="344" y2="344" stroke="#334155" stroke-width="2" marker-end="url(#arr-dark)"/>
    <line x1="354" y1="384" x2="354" y2="428" stroke="#334155" stroke-width="2" marker-end="url(#arr-dark)"/>

    <!-- Branch 3: 6 -> 7 -->
    <line x1="454" y1="296" x2="454" y2="344" stroke="#334155" stroke-width="2" marker-end="url(#arr-dark)"/>

    <!-- Nodes -->
    <circle cx="126" cy="276" r="20" fill="#0F766E" stroke="#134E4A" stroke-width="1.8"/>
    <text x="126" y="274" text-anchor="middle" font-size="12" font-weight="700" fill="#FFFFFF">#1</text>
    <text x="126" y="287" text-anchor="middle" font-size="10" fill="#CCFBF1">7.5</text>

    <circle cx="126" cy="364" r="22" fill="#2563EB" stroke="#1E3A8A" stroke-width="3"/>
    <text x="126" y="362" text-anchor="middle" font-size="12.5" font-weight="700" fill="#FFFFFF">#2</text>
    <text x="126" y="376" text-anchor="middle" font-size="10.5" fill="#DBEAFE">7.2</text>
    <text x="98" y="368" text-anchor="end" font-size="11" font-weight="700" fill="#1E40AF">主父代 p</text>

    <circle cx="126" cy="450" r="20" fill="#0F766E" stroke="#134E4A" stroke-width="1.8"/>
    <text x="126" y="448" text-anchor="middle" font-size="12" font-weight="700" fill="#FFFFFF">#3</text>
    <text x="126" y="461" text-anchor="middle" font-size="10" fill="#CCFBF1">7.0</text>
    <text x="126" y="484" text-anchor="middle" font-size="10.5" fill="#475569">现役子尝试</text>

    <!-- Branch 2 nodes -->
    <circle cx="294" cy="276" r="20" fill="#0F766E" stroke="#134E4A" stroke-width="1.8"/>
    <text x="294" y="274" text-anchor="middle" font-size="12" font-weight="700" fill="#FFFFFF">#4</text>
    <text x="294" y="287" text-anchor="middle" font-size="10" fill="#CCFBF1">7.4</text>
    <text x="322" y="279" font-size="10.5" fill="#15803D">d=3</text>

    <circle cx="250" cy="366" r="20" fill="#0F766E" stroke="#D97706" stroke-width="3"/>
    <text x="250" y="364" text-anchor="middle" font-size="12" font-weight="700" fill="#FFFFFF">#5</text>
    <text x="250" y="377" text-anchor="middle" font-size="10" fill="#CCFBF1">7.1</text>
    <text x="254" y="401" text-anchor="middle" font-size="10" font-weight="700" fill="#B45309">分支最优 (d=4)</text>

    <circle cx="354" cy="366" r="20" fill="#0F766E" stroke="#134E4A" stroke-width="1.8"/>
    <text x="354" y="364" text-anchor="middle" font-size="12" font-weight="700" fill="#FFFFFF">#8</text>
    <text x="354" y="377" text-anchor="middle" font-size="10" fill="#CCFBF1">7.8</text>
    <text x="380" y="370" font-size="10.5" fill="#15803D">d=4</text>

    <circle cx="354" cy="450" r="20" fill="#0F766E" stroke="#D97706" stroke-width="3"/>
    <text x="354" y="448" text-anchor="middle" font-size="12" font-weight="700" fill="#FFFFFF">#9</text>
    <text x="354" y="461" text-anchor="middle" font-size="10" fill="#CCFBF1">7.7</text>
    <text x="382" y="454" font-size="10.5" font-weight="700" fill="#B45309">远端次支 (d=5)</text>

    <!-- Branch 3 nodes (#6 label placed cleanly to RIGHT of #6, zero arrow crossing!) -->
    <circle cx="454" cy="276" r="20" fill="#0F766E" stroke="#D97706" stroke-width="3"/>
    <text x="454" y="274" text-anchor="middle" font-size="12" font-weight="700" fill="#FFFFFF">#6</text>
    <text x="454" y="287" text-anchor="middle" font-size="10" fill="#CCFBF1">7.3</text>
    <text x="480" y="272" font-size="10" font-weight="700" fill="#B45309">分支最优</text>
    <text x="480" y="286" font-size="10" font-weight="700" fill="#B45309">(d=3)</text>

    <circle cx="454" cy="366" r="20" fill="#0F766E" stroke="#134E4A" stroke-width="1.8"/>
    <text x="454" y="364" text-anchor="middle" font-size="12" font-weight="700" fill="#FFFFFF">#7</text>
    <text x="454" y="377" text-anchor="middle" font-size="10" fill="#CCFBF1">7.6</text>
    <text x="470" y="401" text-anchor="middle" font-size="10" fill="#15803D">可作弱供体 (d=4)</text>

    <!-- Footer box inside Left Panel -->
    <rect x="40" y="504" width="508" height="50" rx="8" fill="#F8FAFC" stroke="#CBD5E1" stroke-width="1.2"/>
    <text x="294" y="524" text-anchor="middle" font-size="12" font-weight="700" fill="#0F172A">树上谱系距离 {dT}(u, v) = (|anc(u)| − lca) + (|anc(v)| − lca)</text>
    <text x="294" y="543" text-anchor="middle" font-size="11.2" fill="#475569">直接由现役树拓扑给出谱系远近，无需额外语法树距离或领域探针</text>

    <!-- Arrows to 3 operator cards -->
    <line x1="564" y1="168" x2="602" y2="168" stroke="#2563EB" stroke-width="2.2" marker-end="url(#arr-blue)"/>
    <line x1="564" y1="328" x2="602" y2="328" stroke="#0D9488" stroke-width="2.2" marker-end="url(#arr-teal)"/>
    <line x1="564" y1="488" x2="602" y2="488" stroke="#EA580C" stroke-width="2.2" marker-end="url(#arr-amber)"/>

    <!-- Card 1: Refine -->
    <rect x="606" y="84" width="730" height="150" rx="12" fill="#FFFFFF" stroke="#2563EB" stroke-width="2"/>
    <rect x="606" y="84" width="138" height="150" rx="12" fill="#EFF6FF"/>
    <rect x="732" y="84" width="12" height="150" fill="#EFF6FF"/>
    <line x1="744" y1="84" x2="744" y2="234" stroke="#BFDBFE" stroke-width="1.5"/>
    <text x="675" y="148" text-anchor="middle" font-size="16" font-weight="700" fill="#1E40AF">精化</text>
    <text x="675" y="170" text-anchor="middle" font-size="13" font-weight="700" fill="#1D4ED8">Refine</text>
    <text x="675" y="194" text-anchor="middle" font-size="11" fill="#475569">先验 0.50</text>
    <text x="764" y="114" font-size="13" font-weight="700" fill="#0F172A">读取范围：纵向形成路径 + 本地试错（不读其他谱系）</text>
    <text x="764" y="140" font-size="12" fill="#334155">1. 形成路径：#1 (7.5) → #2 (7.2) 的完整设计简述、代码差异与收缩边经验 {m2}</text>
    <text x="764" y="164" font-size="12" fill="#334155">2. 本地试错：#2 的剪枝节点经验 {s2}、现役子尝试 #3 (7.0) 及近期失败记录</text>
    <text x="764" y="188" font-size="12" fill="#334155">3. 改写起点：结尾附主父代 #2 源码（去注释）；省略祖先长篇生成自述</text>
    <text x="764" y="215" font-size="11.5" font-weight="600" fill="#1E40AF">目标：掌握当前实现如何演化而来以及本地已试错方向，定向改进薄弱计算环节。</text>

    <!-- Card 2: Crossover -->
    <rect x="606" y="252" width="730" height="150" rx="12" fill="#FFFFFF" stroke="#0D9488" stroke-width="2"/>
    <rect x="606" y="252" width="138" height="150" rx="12" fill="#ECFDF5"/>
    <rect x="732" y="252" width="12" height="150" fill="#ECFDF5"/>
    <line x1="744" y1="252" x2="744" y2="402" stroke="#A7F3D0" stroke-width="1.5"/>
    <text x="675" y="316" text-anchor="middle" font-size="16" font-weight="700" fill="#065F46">交叉</text>
    <text x="675" y="338" text-anchor="middle" font-size="13" font-weight="700" fill="#0F766E">Crossover</text>
    <text x="675" y="362" text-anchor="middle" font-size="11" fill="#475569">先验 0.30</text>
    <text x="764" y="282" font-size="13" font-weight="700" fill="#0F172A">选材规则：避开直系 {{#1, #3}}，从跨分支按 {dT} 加权抽供体</text>
    <text x="764" y="308" font-size="12" fill="#334155">1. 供体抽样：{{#4(d=3), #5(d=4), #6(d=3), #7(d=4), #8(d=4), #9(d=5)}} 中按距离加权</text>
    <text x="764" y="332" font-size="12" fill="#334155">2. 弱供体兼容：即使抽中较弱的 #7 (7.6)，仍提示从中提取可互补的局部计算部件</text>
    <text x="764" y="356" font-size="12" fill="#334155">3. 上下文：#2 的形成路径与试错 + 供体设计/特有祖先经验/源码 + #2 源码</text>
    <text x="764" y="383" font-size="11.5" font-weight="600" fill="#0F766E">目标：避免与直系近亲重复重组，将其他分支独立演化出的有效机制融入主父代。</text>

    <!-- Card 3: Explore -->
    <rect x="606" y="420" width="730" height="150" rx="12" fill="#FFFFFF" stroke="#EA580C" stroke-width="2"/>
    <rect x="606" y="420" width="138" height="150" rx="12" fill="#FFF7ED"/>
    <rect x="732" y="420" width="12" height="150" fill="#FFF7ED"/>
    <line x1="744" y1="420" x2="744" y2="570" stroke="#FED7AA" stroke-width="1.5"/>
    <text x="675" y="484" text-anchor="middle" font-size="16" font-weight="700" fill="#9A3412">探索</text>
    <text x="675" y="506" text-anchor="middle" font-size="13" font-weight="700" fill="#C2410C">Explore</text>
    <text x="675" y="530" text-anchor="middle" font-size="11" fill="#475569">先验 0.20</text>
    <text x="764" y="450" font-size="13" font-weight="700" fill="#0F172A">选材规则：先覆盖其他顶级分支最优代表，再按最大最小树距离补齐</text>
    <text x="764" y="476" font-size="12" fill="#334155">1. 分支覆盖：先取 #5 (7.1, 支 2 最优) 与 #6 (7.3, 支 3 最优)；再补远端次支 #9</text>
    <text x="764" y="500" font-size="12" fill="#334155">2. 防锚定呈现：仅主父代 #2 附源码；#5, #6, #9 只给成绩、完整设计简述与经验（隐藏源码）</text>
    <text x="764" y="524" font-size="12" fill="#334155">3. 对比生成：引导模型更换信息利用方式、问题表示或核心计算过程</text>
    <text x="764" y="551" font-size="11.5" font-weight="600" fill="#C2410C">目标：纵览种群已覆盖的主要方向而不被具体代码实现锚定，开辟新搜索方向。</text>

    <text x="{w/2}" y="602" text-anchor="middle" font-size="12" fill="#64748B">橙色描边节点（#5, #6, #9）示意 Explore 优先选取的分支代表与远端节点；绿色虚框为 Crossover 优先采样的跨分支候选池。</text>
    """
    return svg, w, h


def build_fig4() -> tuple[str, int, int]:
    """图 4：分叉局部非平稳后验预算分配与时间遗忘效应示意"""
    w, h = 1360, 560
    ur = sub("û", "r", 9.5)
    Sr = sub("S̃", "r", 9.5)
    nr = sub("n", "r", 9.5)
    uY = sub("û", "Y", 9.5)
    uX = sub("û", "X", 9.5)

    pts = []
    for n in range(0, 21):
        px = 768 + (n / 20.0) * 504
        val = 1.0 / (n + 2.0)
        py = 486 - (val / 0.55) * 116
        pts.append(f"{px:.1f},{py:.1f}")
    decay_polyline = " ".join(pts)

    # Carefully placed non-overlapping annotations on the decay curve
    dot_specs = [
        (0, "n=0: 0.50", 8, -8, "start"),
        (2, "n=2: 0.25", 12, -10, "start"),
        (10, "n=10: 0.083", 0, -12, "middle"),
        (20, "n=20: 0.045", -6, -12, "end"),
    ]
    dots_svg = ""
    for n, label, dx, dy, anchor in dot_specs:
        px = 768 + (n / 20.0) * 504
        val = 1.0 / (n + 2.0)
        py = 486 - (val / 0.55) * 116
        dots_svg += f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.5" fill="#2563EB"/>'
        dots_svg += f'<text x="{px+dx:.1f}" y="{py+dy:.1f}" text-anchor="{anchor}" font-size="11" font-weight="600" fill="#1E40AF">{label}</text>'

    svg = DEFS_ARROWS + f"""
    <text x="{w/2}" y="38" text-anchor="middle" font-size="24" font-weight="700" fill="#0F172A">分叉局部非平稳后验预算分配：数值推演与时间折扣效应</text>
    <text x="{w/2}" y="62" text-anchor="middle" font-size="13.5" fill="#475569">按当前差距 δ 折算历史步幅，并以遗忘因子 γ = 0.85 仅衰减过往正进展、不衰减总花费 n（示意算例）</text>

    <!-- Panel A -->
    <rect x="24" y="84" width="644" height="446" rx="14" fill="#FFFFFF" stroke="#2563EB" stroke-width="2"/>
    <path d="M 24 98 Q 24 84 38 84 L 654 84 Q 668 84 668 98 L 668 128 L 24 128 Z" fill="#1D4ED8"/>
    <text x="346" y="112" text-anchor="middle" font-size="15.5" font-weight="700" fill="#FFFFFF">(a) 分叉 A（5 评后，分叉最优 b* = 7.9，参考幅度 d = 0.15）</text>

    <rect x="44" y="144" width="604" height="140" rx="10" fill="#F8FAFC" stroke="#CBD5E1" stroke-width="1.2"/>
    <circle cx="104" cy="214" r="22" fill="#0F766E" stroke="#134E4A" stroke-width="2"/>
    <text x="104" y="212" text-anchor="middle" font-size="13.5" font-weight="700" fill="#FFFFFF">A</text>
    <text x="104" y="226" text-anchor="middle" font-size="10.5" fill="#CCFBF1">8.0</text>

    <line x1="126" y1="202" x2="210" y2="174" stroke="#64748B" stroke-width="2" stroke-dasharray="4,3" marker-end="url(#arr-dark)"/>
    <line x1="126" y1="214" x2="210" y2="214" stroke="#2563EB" stroke-width="2.6" marker-end="url(#arr-blue)"/>
    <line x1="126" y1="226" x2="210" y2="254" stroke="#94A3B8" stroke-width="2" marker-end="url(#arr-muted)"/>

    <rect x="216" y="156" width="414" height="32" rx="6" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.2"/>
    <text x="228" y="177" font-size="12" fill="#334155"><tspan font-weight="700" fill="#0F172A">直接改写 A</tspan>：最优 7.9（δ=0），步幅 [0, 0.1] → 折算 [0, 0.1]，n=2</text>

    <rect x="216" y="198" width="414" height="32" rx="6" fill="#EFF6FF" stroke="#2563EB" stroke-width="1.8"/>
    <text x="228" y="219" font-size="12" fill="#1E3A8A"><tspan font-weight="700">进入 B 子路</tspan>：8.5→8.1（δ=0.2），步幅 [0.4] → 折算 [0.2]，n=1</text>

    <rect x="216" y="240" width="414" height="32" rx="6" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.2"/>
    <text x="228" y="261" font-size="12" fill="#64748B"><tspan font-weight="700" fill="#334155">进入 E 子路</tspan>：最优 7.9（δ=0），2 次重复 [0, 0] → 折算 [0, 0]，n=2</text>

    <text x="52" y="314" font-size="13.5" font-weight="700" fill="#0F172A">各方向预期每评推进量 {ur} = (d + {Sr}) / ({nr} + 2)</text>

    <line x1="178" y1="328" x2="178" y2="466" stroke="#334155" stroke-width="1.5"/>
    <line x1="178" y1="466" x2="620" y2="466" stroke="#334155" stroke-width="1.5"/>
    <line x1="287" y1="328" x2="287" y2="466" stroke="#E2E8F0" stroke-width="1"/>
    <line x1="395" y1="328" x2="395" y2="466" stroke="#E2E8F0" stroke-width="1"/>
    <line x1="504" y1="328" x2="504" y2="466" stroke="#E2E8F0" stroke-width="1"/>
    <text x="178" y="484" text-anchor="middle" font-size="11" fill="#64748B">0</text>
    <text x="287" y="484" text-anchor="middle" font-size="11" fill="#64748B">0.04</text>
    <text x="395" y="484" text-anchor="middle" font-size="11" fill="#64748B">0.08</text>
    <text x="504" y="484" text-anchor="middle" font-size="11" fill="#64748B">0.12</text>

    <text x="168" y="362" text-anchor="end" font-size="12.5" font-weight="700" fill="#1D4ED8">进入 B 子路</text>
    <rect x="178" y="340" width="317" height="32" rx="5" fill="#2563EB"/>
    <text x="505" y="361" font-size="12.5" font-weight="700" fill="#1E3A8A">0.1167（最高）</text>

    <text x="168" y="406" text-anchor="end" font-size="12.5" font-weight="600" fill="#0F766E">直接改写 A</text>
    <rect x="178" y="384" width="170" height="32" rx="5" fill="#0D9488"/>
    <text x="358" y="405" font-size="12.5" font-weight="600" fill="#0F766E">0.0625</text>

    <text x="168" y="450" text-anchor="end" font-size="12.5" fill="#64748B">进入 E 子路</text>
    <rect x="178" y="428" width="102" height="32" rx="5" fill="#94A3B8"/>
    <text x="290" y="449" font-size="12.5" fill="#475569">0.0375（停滞降权）</text>

    <text x="346" y="514" text-anchor="middle" font-size="11.8" fill="#334155">B 虽落后 0.2，但单步走出 0.4 的有效回升，预期每评推进高于原地踏步的 E</text>

    <!-- Panel B -->
    <rect x="692" y="84" width="644" height="446" rx="14" fill="#FFFFFF" stroke="#0D9488" stroke-width="2"/>
    <path d="M 692 98 Q 692 84 706 84 L 1322 84 Q 1336 84 1336 98 L 1336 128 L 692 128 Z" fill="#0F766E"/>
    <text x="1014" y="112" text-anchor="middle" font-size="15.5" font-weight="700" fill="#FFFFFF">(b) 非平稳时间折扣（γ = 0.85）：区分近期突破与早期停滞</text>

    <rect x="712" y="144" width="604" height="170" rx="10" fill="#F8FAFC" stroke="#CBD5E1" stroke-width="1.2"/>
    <text x="728" y="168" font-size="13" font-weight="700" fill="#0F172A">同为 n = 9、最优 8.1（δ = 0.2，d = 0.15），但突破发生时机不同：</text>

    <text x="728" y="198" font-size="12" font-weight="700" fill="#0F766E">近期突破支路 Y</text>
    <text x="844" y="198" font-size="11.2" class="mono" fill="#475569">[0, 0, …, 0, 0.4]  (权重 γ⁰ = 1.00)</text>
    <rect x="728" y="208" width="380" height="26" rx="5" fill="#0D9488"/>
    <text x="1118" y="226" font-size="12.5" font-weight="700" fill="#065F46">{uY} = 0.0318（约 1.7×）</text>

    <text x="728" y="260" font-size="12" font-weight="700" fill="#C2410C">早期改进后停滞支路 X</text>
    <text x="876" y="260" font-size="11.2" class="mono" fill="#475569">[0.4, 0, …, 0, 0]  (权重 γ⁸ = 0.27)</text>
    <rect x="728" y="270" width="222" height="26" rx="5" fill="#F59E0B"/>
    <text x="960" y="288" font-size="12.5" font-weight="700" fill="#9A3412">{uX} = 0.0186（旧突破自然衰减）</text>

    <rect x="712" y="328" width="604" height="186" rx="10" fill="#F8FAFC" stroke="#CBD5E1" stroke-width="1.2"/>
    <text x="728" y="352" font-size="13" font-weight="700" fill="#0F172A">持续无进展支路（s̃ = 0）的期望成功率 E[p] = 1 / (n + 2)</text>

    <line x1="768" y1="368" x2="768" y2="486" stroke="#334155" stroke-width="1.4"/>
    <line x1="768" y1="486" x2="1282" y2="486" stroke="#334155" stroke-width="1.4"/>
    <line x1="768" y1="433" x2="1282" y2="433" stroke="#E2E8F0" stroke-width="1"/>
    <line x1="768" y1="380" x2="1282" y2="380" stroke="#E2E8F0" stroke-width="1"/>
    <text x="760" y="490" text-anchor="end" font-size="10.5" fill="#64748B">0</text>
    <text x="760" y="437" text-anchor="end" font-size="10.5" fill="#64748B">0.25</text>
    <text x="760" y="384" text-anchor="end" font-size="10.5" fill="#64748B">0.50</text>
    <text x="1025" y="505" text-anchor="middle" font-size="11" fill="#475569">连续无有效进展的已付评价次数 n（总评数不衰减，随 n 持续趋近 0 而不饱和）</text>

    <polyline points="{decay_polyline}" fill="none" stroke="#2563EB" stroke-width="2.8"/>
    {dots_svg}
    """
    return svg, w, h


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    specs = [
        ("fig1-v5-framework.png", build_fig1),
        ("fig2-v5-reduction-reflection.png", build_fig2),
        ("fig3-v5-lineage-operators.png", build_fig3),
        ("fig4-v5-fork-allocation.png", build_fig4),
    ]
    for name, builder in specs:
        svg, width, height = builder()
        out_path = OUT_DIR / name
        render_svg(svg, width, height, out_path)
        print(f"Rendered {out_path} ({width*2}x{height*2})")


if __name__ == "__main__":
    main()
