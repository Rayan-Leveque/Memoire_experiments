"""
build_pdf.py — Générateur PDF style LaTeX classique
Usage : python3 build_pdf.py
Output : scoring_rayan/docs/build/dossier_scoring_confiance_IDeXtend.pdf
"""

import re
import sys
from pathlib import Path
from datetime import datetime

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib.colors import black, white, Color
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, HRFlowable,
    )
    from reportlab.pdfgen import canvas
except ImportError:
    print("pip install --user --break-system-packages reportlab")
    sys.exit(1)

# ── Chemins ───────────────────────────────────────────────────────────────────
SCRIPT_DIR    = Path(__file__).parent
COMPOSANTS_DIR = SCRIPT_DIR / "composants"
BUILD_DIR     = SCRIPT_DIR / "build"
OUTPUT_FILE   = BUILD_DIR / "dossier_scoring_confiance_IDeXtend.pdf"
BUILD_DIR.mkdir(exist_ok=True)

COMPOSANTS = [
    "01_ocr.md",
    "02_asr_whisper.md",
    "03_embeddings.md",
    "04_reranker.md",
    "05_llm_qwen.md",
    "06_translation_nllb.md",
    "07_detection_langue.md",
]

# ── Constantes typographiques ─────────────────────────────────────────────────
W, H    = A4
LM = RM = 2.5 * cm
TM = BM = 2.5 * cm
TW = W - LM - RM          # text width

FONT_BODY  = "Times-Roman"
FONT_BOLD  = "Times-Bold"
FONT_ITAL  = "Times-Italic"
FONT_MONO  = "Courier"

SIZE_BODY  = 11
SIZE_SMALL = 9
SIZE_H1    = 17
SIZE_H2    = 13
SIZE_H3    = 11
SIZE_CAPTION = 9

GRAY_LIGHT = Color(0.95, 0.95, 0.95)
GRAY_MED   = Color(0.6,  0.6,  0.6)
GRAY_RULE  = Color(0.3,  0.3,  0.3)

# ── Styles ────────────────────────────────────────────────────────────────────
def build_styles():
    getSampleStyleSheet()  # initialise internal registry

    # Chapter title  (## level-1 heading → new chapter)
    h1 = ParagraphStyle("h1",
        fontName=FONT_BOLD, fontSize=SIZE_H1,
        leading=22, spaceBefore=0, spaceAfter=10,
        textColor=black, alignment=TA_LEFT,
    )
    # Section (## in body)
    h2 = ParagraphStyle("h2",
        fontName=FONT_BOLD, fontSize=SIZE_H2,
        leading=18, spaceBefore=14, spaceAfter=4,
        textColor=black,
    )
    # Subsection (###)
    h3 = ParagraphStyle("h3",
        fontName=FONT_BOLD, fontSize=SIZE_H3,
        leading=15, spaceBefore=10, spaceAfter=3,
        textColor=black,
    )
    # Subsubsection (####)
    h4 = ParagraphStyle("h4",
        fontName=FONT_BOLD + "" , fontSize=SIZE_BODY,
        leading=14, spaceBefore=8, spaceAfter=2,
        textColor=black,
    )

    body = ParagraphStyle("body",
        fontName=FONT_BODY, fontSize=SIZE_BODY,
        leading=16, spaceAfter=5,
        alignment=TA_JUSTIFY,
    )
    bullet = ParagraphStyle("bullet",
        fontName=FONT_BODY, fontSize=SIZE_BODY,
        leading=15, spaceAfter=3,
        leftIndent=18, firstLineIndent=-10,
    )
    num_item = ParagraphStyle("num_item",
        fontName=FONT_BODY, fontSize=SIZE_BODY,
        leading=15, spaceAfter=3,
        leftIndent=22, firstLineIndent=-14,
    )
    code = ParagraphStyle("code",
        fontName=FONT_MONO, fontSize=8,
        leading=11, spaceAfter=6, spaceBefore=4,
        leftIndent=12, rightIndent=12,
        backColor=GRAY_LIGHT,
        borderColor=GRAY_MED, borderWidth=0.5, borderPad=6,
    )
    formula = ParagraphStyle("formula",
        fontName=FONT_MONO, fontSize=9,
        leading=14, spaceAfter=8, spaceBefore=6,
        alignment=TA_CENTER,
        borderColor=GRAY_MED, borderWidth=0.5, borderPad=8,
    )
    # Blockquote / note (frame with left bar — emulated as bordered para)
    note = ParagraphStyle("note",
        fontName=FONT_ITAL, fontSize=SIZE_SMALL,
        leading=14, spaceAfter=8, spaceBefore=4,
        leftIndent=14, rightIndent=0,
        textColor=Color(0.2, 0.2, 0.2),
        borderColor=GRAY_MED, borderWidth=0, borderPad=0,
    )
    caption = ParagraphStyle("caption",
        fontName=FONT_ITAL, fontSize=SIZE_CAPTION,
        leading=12, spaceAfter=10, spaceBefore=2,
        alignment=TA_CENTER, textColor=GRAY_MED,
    )
    toc_ch = ParagraphStyle("toc_ch",
        fontName=FONT_BOLD, fontSize=SIZE_BODY,
        leading=16, spaceAfter=2,
    )
    toc_sec = ParagraphStyle("toc_sec",
        fontName=FONT_BODY, fontSize=SIZE_SMALL,
        leading=14, spaceAfter=1,
        leftIndent=16,
    )
    small = ParagraphStyle("small",
        fontName=FONT_BODY, fontSize=SIZE_SMALL,
        leading=13, spaceAfter=4,
    )

    return dict(h1=h1, h2=h2, h3=h3, h4=h4, body=body, bullet=bullet,
                num_item=num_item, code=code, formula=formula, note=note,
                caption=caption, toc_ch=toc_ch, toc_sec=toc_sec, small=small)

# ── Numérotation des chapitres ────────────────────────────────────────────────
class Counter:
    def __init__(self): self.ch = 0; self.sec = 0; self.subsec = 0
    def chapter(self):
        self.ch += 1; self.sec = 0; self.subsec = 0
        return str(self.ch)
    def section(self):
        self.sec += 1; self.subsec = 0
        return f"{self.ch}.{self.sec}"
    def subsection(self):
        self.subsec += 1
        return f"{self.ch}.{self.sec}.{self.subsec}"

CTR = Counter()

# ── En-tête/pied de page ──────────────────────────────────────────────────────
class LatexPageCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        canvas.Canvas.__init__(self, *args, **kwargs)
        self._saved = []

    def showPage(self):
        self._saved.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        n = len(self._saved)
        for state in self._saved:
            self.__dict__.update(state)
            self._draw_footer(n)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def _draw_footer(self, _total):
        p = self._pageNumber
        # Skip title page
        if p <= 1:
            return
        self.saveState()
        self.setFont(FONT_BODY, 9)
        self.setFillColor(GRAY_MED)
        # Thin rule above footer
        self.setStrokeColor(GRAY_MED)
        self.setLineWidth(0.4)
        self.line(LM, BM - 4, W - RM, BM - 4)
        # Left: short title
        self.drawString(LM, BM - 14,
            "IDeXtend — Scoring de confiance — AI Act Annexe III §6")
        # Right: page number
        self.drawRightString(W - RM, BM - 14, str(p))
        self.restoreState()

# ── Inline markdown → ReportLab XML ──────────────────────────────────────────
def md_inline(text):
    # 1. Extract code spans → placeholders (preserves content verbatim)
    spans = {}
    def save_code(m):
        k = f"\x00C{len(spans)}\x00"
        inner = m.group(1).replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
        spans[k] = f'<font name="{FONT_MONO}">{inner}</font>'
        return k
    text = re.sub(r'`([^`]+)`', save_code, text)

    # 2. Escape ALL remaining XML special chars before injecting any tags
    text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

    # 3. Bold (**…**) then italic (*…*) — safe now because < > are already escaped
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<i>\1</i>', text)

    # 4. Restore code spans
    for k, v in spans.items():
        text = text.replace(k, v)
    return text

# ── Formules ──────────────────────────────────────────────────────────────────
def clean_formula(src):
    f = src.strip().strip('$').strip()
    subs = [
        (r'\\bar\{([^}]+)\}',    r'<overline>\1</overline>'),
        (r'\\hat\{([^}]+)\}',    r'\1̂'),
        (r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)'),
        (r'\\left[\(\[]',  '('), (r'\\right[\)\]]', ')'),
        (r'\\sum',  '∑'),  (r'\\prod', '∏'),
        (r'\\min',  'min'),(r'\\max',  'max'),
        (r'\\exp',  'exp'),(r'\\log',  'log'),
        (r'\\cdot', '·'),  (r'\\times','×'),
        (r'\\leq',  '≤'),  (r'\\geq', '≥'), (r'\\neq','≠'),
        (r'\\in',   '∈'),  (r'\\notin','∉'),
        (r'\\alpha','α'), (r'\\beta','β'), (r'\\sigma','σ'),
        (r'\\rho',  'ρ'), (r'\\tau', 'τ'), (r'\\phi','φ'),
        (r'\\delta','δ'), (r'\\Delta','Δ'),
        (r'\\mathbf\{([^}]+)\}', r'\1'),
        (r'\\mathcal\{([^}]+)\}',r'\1'),
        (r'\\text\{([^}]+)\}',   r'\1'),
        (r'\\boxed\{([^}]+)\}',  r'[ \1 ]'),
        (r'\\label\{[^}]+\}',    ''),
        (r'\\quad', '   '), (r'\\;',' '), (r'\\,',' '),
        (r'\\[a-zA-Z]+',         ''),
        (r'\{', ''), (r'\}', ''),
    ]
    for pat, rep in subs:
        f = re.sub(pat, rep, f)
    return f.strip()

# ── Tableaux ──────────────────────────────────────────────────────────────────
def make_table(rows, styles_dict):
    """rows: list[list[str]]. Row 0 = header."""
    cell_s = ParagraphStyle("tc", fontName=FONT_BODY,  fontSize=8, leading=11)
    head_s = ParagraphStyle("th", fontName=FONT_BOLD,  fontSize=8, leading=11, textColor=white)

    data = []
    for r_i, row in enumerate(rows):
        is_hdr = (r_i == 0)
        data.append([
            Paragraph(md_inline(str(c)), head_s if is_hdr else cell_s)
            for c in row
        ])

    n = len(rows[0])
    cw = [TW / n] * n

    ts = TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  Color(0.15, 0.15, 0.15)),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  white),
        ("FONTNAME",      (0, 0), (-1, 0),  FONT_BOLD),
        ("FONTSIZE",      (0, 0), (-1,-1),  8),
        ("ROWBACKGROUNDS",(0, 1), (-1,-1),  [white, GRAY_LIGHT]),
        ("GRID",          (0, 0), (-1,-1),  0.3, GRAY_MED),
        ("TOPPADDING",    (0, 0), (-1,-1),  3),
        ("BOTTOMPADDING", (0, 0), (-1,-1),  3),
        ("LEFTPADDING",   (0, 0), (-1,-1),  5),
        ("RIGHTPADDING",  (0, 0), (-1,-1),  5),
        ("VALIGN",        (0, 0), (-1,-1),  "TOP"),
    ])
    t = Table(data, colWidths=cw, repeatRows=1)
    t.setStyle(ts)
    return t

# ── Parser Markdown → Flowables ───────────────────────────────────────────────
def parse_md(md_text, styles, ctr):
    story = []
    lines = md_text.split('\n')
    i = 0
    table_rows = []
    code_lines = []
    in_code = False
    in_formula = False
    formula_lines = []

    def flush_table():
        if table_rows:
            story.append(Spacer(1, 4))
            story.append(make_table(table_rows, styles))
            story.append(Spacer(1, 4))
        table_rows.clear()

    while i < len(lines):
        raw = lines[i]
        line = raw.strip()

        # ── Code fence ────────────────────────────────────────────────────────
        if line.startswith('```'):
            flush_table()
            if not in_code:
                in_code = True; code_lines = []
            else:
                in_code = False
                txt = '\n'.join(code_lines)
                txt = txt.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;').replace('\n','<br/>')
                story.append(Paragraph(txt, styles['code']))
                code_lines = []
            i += 1; continue
        if in_code:
            code_lines.append(raw); i += 1; continue

        # ── Formula block $$ ──────────────────────────────────────────────────
        if line == '$$':
            flush_table()
            if not in_formula:
                in_formula = True; formula_lines = []
            else:
                in_formula = False
                formula_text = ' '.join(
                    clean_formula(l) for l in formula_lines if l.strip()
                )
                story.append(Paragraph(formula_text, styles['formula']))
                formula_lines = []
            i += 1; continue
        if in_formula:
            formula_lines.append(raw); i += 1; continue

        # ── Markdown table ─────────────────────────────────────────────────────
        if line.startswith('|') and '|' in line[1:]:
            if re.match(r'^\|[\s:|-]+\|$', line):   # separator row
                i += 1; continue
            cells = [c.strip() for c in line.strip('|').split('|')]
            table_rows.append(cells)
            i += 1; continue
        else:
            flush_table()

        # ── Blank line ─────────────────────────────────────────────────────────
        if not line:
            story.append(Spacer(1, 5))
            i += 1; continue

        # ── HR ─────────────────────────────────────────────────────────────────
        if re.match(r'^-{3,}$', line) or re.match(r'^\*{3,}$', line):
            i += 1; continue   # skip horizontal rules (they're just section separators in MD)

        # ── Headings ───────────────────────────────────────────────────────────
        m = re.match(r'^(#{1,4})\s+(.*)', line)
        if m:
            lvl = len(m.group(1))
            title_raw = m.group(2)
            # Strip trailing badges like [IMPLÉMENTÉ] etc.
            title_raw = re.sub(r'\s*\[.*?\]\s*$', '', title_raw)
            title = md_inline(title_raw)

            if lvl == 1:
                # Chapter — page break + number
                num = ctr.chapter()
                story.append(PageBreak())
                story.append(Paragraph(f"Chapitre {num}", ParagraphStyle(
                    "chnum", fontName=FONT_BODY, fontSize=10, textColor=GRAY_MED,
                    leading=14, spaceAfter=2,
                )))
                story.append(HRFlowable(width="100%", thickness=1.5, color=black, spaceAfter=6))
                story.append(Paragraph(title, styles['h1']))
                story.append(HRFlowable(width="100%", thickness=0.5, color=GRAY_MED, spaceAfter=10))
            elif lvl == 2:
                num = ctr.section()
                story.append(Spacer(1, 6))
                story.append(Paragraph(f"{num}  {title}", styles['h2']))
                story.append(HRFlowable(width="100%", thickness=0.3, color=GRAY_MED, spaceAfter=4))
            elif lvl == 3:
                num = ctr.subsection()
                story.append(Paragraph(f"{num}  {title}", styles['h3']))
            else:
                story.append(Paragraph(f"<b>{title}</b>", styles['body']))
            i += 1; continue

        # ── Blockquote ────────────────────────────────────────────────────────
        if line.startswith('>'):
            content = line.lstrip('>').strip()
            story.append(Paragraph(md_inline(content), styles['note']))
            i += 1; continue

        # ── Bullet list ───────────────────────────────────────────────────────
        m = re.match(r'^[-*+]\s+(.*)', line)
        if m:
            story.append(Paragraph('&#x2022;&#x2002;' + md_inline(m.group(1)), styles['bullet']))
            i += 1; continue

        # ── Numbered list ─────────────────────────────────────────────────────
        m = re.match(r'^(\d+)\.\s+(.*)', line)
        if m:
            story.append(Paragraph(f"{m.group(1)}.&#x2002;{md_inline(m.group(2))}", styles['num_item']))
            i += 1; continue

        # ── Normal paragraph ──────────────────────────────────────────────────
        story.append(Paragraph(md_inline(line), styles['body']))
        i += 1

    flush_table()
    return story

# ── Page de titre ─────────────────────────────────────────────────────────────
def make_title_page(styles):
    s = []

    def cp(text, fname, fsize, align=TA_CENTER, space=0, color=black):
        return Paragraph(text, ParagraphStyle("_tp",
            fontName=fname, fontSize=fsize, leading=fsize*1.4,
            alignment=align, textColor=color, spaceAfter=space,
        ))

    s.append(Spacer(1, 4*cm))
    s.append(cp("GENDARMERIE NATIONALE", FONT_BODY, 11, space=2))
    s.append(cp("Centre de Formation et d'Investigation Avancée (CFIA)", FONT_ITAL, 10, space=20))
    s.append(HRFlowable(width="60%", thickness=1, color=black, hAlign="CENTER", spaceAfter=20))
    s.append(cp("Dossier technique", FONT_BODY, 14, space=8))
    s.append(cp("Scoring de confiance des composants IA", FONT_BOLD, 22, space=6))
    s.append(cp("Système IDeXtend", FONT_ITAL, 16, space=20))
    s.append(HRFlowable(width="60%", thickness=1, color=black, hAlign="CENTER", spaceAfter=20))
    s.append(cp("Conformité AI Act — Annexe III §6", FONT_ITAL, 11, space=4))
    s.append(cp("Système à haut risque — Application de la loi", FONT_ITAL, 11, space=40))

    meta = [
        ["Auteur",         "Leveque Rayan"],
        ["Date",           datetime.now().strftime("%d %B %Y")],
        ["Version",        "0.1 — Brouillon"],
        ["Branche git",    "feature/evaluation-modele"],
    ]
    lbl = ParagraphStyle("_l", fontName=FONT_BOLD, fontSize=10, leading=14)
    val = ParagraphStyle("_v", fontName=FONT_BODY, fontSize=10, leading=14)
    tdata = [[Paragraph(r[0], lbl), Paragraph(r[1], val)] for r in meta]
    t = Table(tdata, colWidths=[4*cm, 9*cm], hAlign="CENTER")
    t.setStyle(TableStyle([
        ("LINEBELOW",    (0,0),(-1,-2), 0.3, GRAY_MED),
        ("TOPPADDING",   (0,0),(-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
        ("LEFTPADDING",  (0,0),(-1,-1), 0),
    ]))
    s.append(t)
    s.append(Spacer(1, 3*cm))
    s.append(HRFlowable(width="100%", thickness=0.5, color=GRAY_MED, spaceAfter=6))
    s.append(cp("Document à usage interne — Diffusion restreinte CFIA",
                FONT_ITAL, 8, color=GRAY_MED))
    s.append(PageBreak())
    return s

# ── Résumé exécutif ───────────────────────────────────────────────────────────
def make_summary(styles, ctr):
    s = []
    ctr.chapter()   # count as chapter 0 (not printed)

    s.append(Paragraph("Résumé exécutif", styles['h1']))
    s.append(HRFlowable(width="100%", thickness=0.5, color=GRAY_MED, spaceAfter=8))
    s.append(Paragraph(
        "IDeXtend est un système RAG (<i>Retrieval-Augmented Generation</i>) d'aide à l'enquête "
        "judiciaire déployé à la CFIA Gendarmerie nationale. Classifié <b>système à haut risque</b> "
        "au titre de l'Annexe III §6 de l'AI Act (Règlement UE 2024/1689), il est soumis à des "
        "exigences strictes de transparence, de robustesse et de surveillance humaine (Art. 9, 13, 15). "
        "Ce dossier documente scientifiquement le <b>système de scoring de confiance</b> de chaque "
        "composant du pipeline.",
        styles['body']
    ))
    s.append(Spacer(1, 8))
    s.append(Paragraph("État d'implémentation au 11 mai 2026", styles['h2']))
    s.append(HRFlowable(width="100%", thickness=0.3, color=GRAY_MED, spaceAfter=4))

    rows = [
        ["Composant", "Modèle", "compute_confidence()", "Validé"],
        ["OCR",              "QwenVL 2.5-7B",          "A implémenter", "Non"],
        ["ASR (Whisper)",    "whisper-large-v3",        "A implémenter", "Non"],
        ["Embeddings",       "BAAI/bge-m3",             "A implémenter", "Non"],
        ["Reranker",         "BAAI/bge-reranker-v2-m3", "Implémenté",    "Non"],
        ["LLM (Qwen3)",      "Qwen3-4B",                "A définir",     "Non"],
        ["Traduction",       "NLLB-600M",               "Implémenté",    "Non"],
        ["Détection langue", "FastText LID",            "Implémenté",    "Non"],
    ]
    s.append(make_table(rows, styles))
    s.append(PageBreak())
    return s

# ── Build principal ───────────────────────────────────────────────────────────
def build():
    styles = build_styles()
    ctr    = Counter()

    doc = SimpleDocTemplate(
        str(OUTPUT_FILE),
        pagesize=A4,
        leftMargin=LM, rightMargin=RM,
        topMargin=TM,  bottomMargin=BM,
        title="Dossier scoring de confiance IDeXtend",
        author="Leveque Rayan",
        subject="AI Act Annexe III §6",
    )

    story = []
    story.extend(make_title_page(styles))
    story.extend(make_summary(styles, ctr))

    for filename in COMPOSANTS:
        path = COMPOSANTS_DIR / filename
        if not path.exists():
            print(f"  ⚠  Manquant : {path}")
            continue
        print(f"  → {filename}")
        md = path.read_text(encoding="utf-8")
        story.extend(parse_md(md, styles, ctr))

    print(f"\nGénération : {OUTPUT_FILE}")
    doc.build(story, canvasmaker=LatexPageCanvas)
    kb = OUTPUT_FILE.stat().st_size // 1024
    print(f"✓  {kb} Ko  →  {OUTPUT_FILE}")

if __name__ == "__main__":
    build()
