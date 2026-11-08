#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""md2docx.py：Markdown → DOCX（中文排版规范：1.3 倍行距、首行缩进 2 字符、宋体/Times 混排、
表格 Table Grid + 跨页表头 + 行不跨页、标题层级、页码页脚）。零第三方依赖（python-docx）。
用法: python md2docx.py <input.md> <output.docx>
"""
import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

CJK_BODY = "宋体"
CJK_HEAD = "黑体"
LATIN = "Times New Roman"


def set_run_font(run, size=None, bold=None, italic=None, cjk=CJK_BODY, latin=LATIN, color=None):
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    if italic is not None:
        run.font.italic = italic
    run.font.name = latin
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), cjk)
    rfonts.set(qn("w:ascii"), latin)
    rfonts.set(qn("w:hAnsi"), latin)
    if color is not None:
        run.font.color.rgb = color


def style_font(style, size, bold=False, cjk=CJK_BODY, latin=LATIN, color=RGBColor(0, 0, 0)):
    style.font.name = latin
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.color.rgb = color
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), cjk)
    rfonts.set(qn("w:ascii"), latin)
    rfonts.set(qn("w:hAnsi"), latin)


INLINE = re.compile(r"(\*\*.+?\*\*|`[^`]+`)")
LINK_PAT = re.compile(r"!??\[([^\]]*)\]\(([^)]+)\)")


def strip_links(text):
    """[text](url) → text（url）；图片语法整体删除。"""
    def rep(m):
        return f"{m.group(1)}（{m.group(2)}）" if m.group(1) else ""
    return LINK_PAT.sub(rep, text)


def add_inline(par, text, size=10.5, cjk=CJK_BODY, base_bold=False):
    text = strip_links(text)
    for tok in INLINE.split(text):
        if not tok:
            continue
        if tok.startswith("**") and tok.endswith("**") and len(tok) > 4:
            r = par.add_run(tok[2:-2])
            set_run_font(r, size=size, bold=True, cjk=cjk)
        elif tok.startswith("`") and tok.endswith("`") and len(tok) > 2:
            r = par.add_run(tok[1:-1])
            set_run_font(r, size=size - 0.5, bold=base_bold, cjk="Courier New", latin="Courier New")
        else:
            r = par.add_run(tok)
            set_run_font(r, size=size, bold=base_bold, cjk=cjk)


def shade(cell, hex_fill="EDEDED"):
    tcpr = cell._element.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tcpr.append(shd)


def row_no_split(row, header=False):
    trpr = row._tr.get_or_add_trPr()
    cs = OxmlElement("w:cantSplit")
    trpr.append(cs)
    if header:
        th = OxmlElement("w:tblHeader")
        trpr.append(th)


def add_table(doc, rows):
    header, body = rows[0], rows[1:]
    ncol = len(header)
    tbl = doc.add_table(rows=1 + len(body), cols=ncol)
    tbl.style = "Table Grid"
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = True
    hdr = tbl.rows[0]
    for j, txt in enumerate(header):
        cell = hdr.cells[j]
        cell.text = ""
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.line_spacing = 1.3
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(1)
        add_inline(p, txt.strip(), size=9)
        shade(cell)
    row_no_split(hdr, header=True)
    for i, r in enumerate(body, start=1):
        cells = tbl.rows[i].cells
        for j in range(ncol):
            txt = r[j].strip() if j < len(r) else ""
            cell = cells[j]
            cell.text = ""
            p = cell.paragraphs[0]
            p.paragraph_format.line_spacing = 1.3
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            if re.fullmatch(r"[\d.±\-–—+eE%\[\],\s]+", txt):
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_inline(p, txt, size=9)
        row_no_split(tbl.rows[i])
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def add_page_number_footer(doc):
    sec = doc.sections[0]
    p = sec.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("— ")
    set_run_font(r, size=9, color=RGBColor(0x66, 0x66, 0x66))
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE \\* arabic \\* MERGEFORMAT")
    rn = OxmlElement("w:r")
    fld.append(rn)
    p._p.append(fld)
    r2 = p.add_run(" —")
    set_run_font(r2, size=9, color=RGBColor(0x66, 0x66, 0x66))


def parse_pipe(line):
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c for c in s.split("|")]


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    lines = src.read_text(encoding="utf-8").splitlines()
    doc = Document()
    sec = doc.sections[0]
    sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
    sec.top_margin = sec.bottom_margin = Cm(2.54)
    sec.left_margin = sec.right_margin = Cm(2.8)

    style_font(doc.styles["Normal"], 10.5)
    doc.styles["Normal"].paragraph_format.line_spacing = 1.3
    doc.styles["Normal"].paragraph_format.space_after = Pt(4)
    for name, size in [("Heading 1", 15), ("Heading 2", 13), ("Heading 3", 11.5)]:
        try:
            style_font(doc.styles[name], size, bold=True, cjk=CJK_HEAD)
            doc.styles[name].paragraph_format.space_before = Pt(10)
            doc.styles[name].paragraph_format.space_after = Pt(5)
            doc.styles[name].paragraph_format.line_spacing = 1.25
        except KeyError:
            pass

    i = 0
    n_tbl = 0
    while i < len(lines):
        line = lines[i].rstrip()
        s = line.strip()
        if not s:
            i += 1
            continue
        if s in ("---", "***", "___"):
            i += 1
            continue
        # 表格
        if s.startswith("|") and i + 1 < len(lines) and re.match(r"^\|?[\s:\-|]+\|?$", lines[i + 1].strip()):
            rows = [parse_pipe(s)]
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(parse_pipe(lines[i]))
                i += 1
            add_table(doc, rows)
            n_tbl += 1
            continue
        # 围栏代码块
        if s.startswith("```"):
            i += 1
            code_lines = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i].rstrip())
                i += 1
            i += 1
            p = doc.add_paragraph()
            p.paragraph_format.line_spacing = 1.15
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(6)
            p.paragraph_format.left_indent = Cm(0.4)
            ppr = p._p.get_or_add_pPr()
            shd = OxmlElement("w:shd")
            shd.set(qn("w:val"), "clear")
            shd.set(qn("w:color"), "auto")
            shd.set(qn("w:fill"), "F5F5F5")
            ppr.append(shd)
            for k, cl in enumerate(code_lines):
                r = p.add_run(cl)
                set_run_font(r, size=8.5, cjk="Courier New", latin="Courier New")
                if k < len(code_lines) - 1:
                    r.add_break()
            continue
        # 标题
        m = re.match(r"^(#{1,6})\s+(.*)$", s)
        if m:
            level, text = len(m.group(1)), m.group(2).strip()
            if level == 1:
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_before = Pt(6)
                p.paragraph_format.space_after = Pt(10)
                r = p.add_run(text)
                set_run_font(r, size=18, bold=True, cjk=CJK_HEAD)
            else:
                p = doc.add_paragraph(style=f"Heading {min(level - 1, 3)}")
                add_inline(p, text, size={2: 15, 3: 13, 4: 11.5}.get(level, 11.5), cjk=CJK_HEAD, base_bold=True)
            i += 1
            continue
        # 引用
        if s.startswith(">"):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.75)
            p.paragraph_format.line_spacing = 1.25
            add_inline(p, s.lstrip("> ").strip(), size=9.5)
            for r in p.runs:
                r.font.color.rgb = RGBColor(0x44, 0x44, 0x44)
            i += 1
            continue
        # 无序列表
        if re.match(r"^[-*+]\s+", s):
            p = doc.add_paragraph(style="List Bullet")
            p.paragraph_format.line_spacing = 1.3
            p.paragraph_format.space_after = Pt(2)
            add_inline(p, re.sub(r"^[-*+]\s+", "", s))
            i += 1
            continue
        # 有序列表
        if re.match(r"^\d+[.)]\s+", s):
            p = doc.add_paragraph(style="List Number")
            p.paragraph_format.line_spacing = 1.3
            p.paragraph_format.space_after = Pt(2)
            add_inline(p, re.sub(r"^\d+[.)]\s+", "", s))
            i += 1
            continue
        # 普通段落（CJK 首行缩进 2 字符）
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        if re.search(r"[\u4e00-\u9fff]", s):
            p.paragraph_format.first_line_indent = Pt(21)
        add_inline(p, s)
        i += 1

    # 行距一致性收口（全文显式 1.3 倍）
    for par in doc.paragraphs:
        par.paragraph_format.line_spacing = 1.3
    for tb in doc.tables:
        for rw in tb.rows:
            for cl in rw.cells:
                for par in cl.paragraphs:
                    par.paragraph_format.line_spacing = 1.3

    # 送审行号（连续，每行计数）——可选 --linenum
    if "--linenum" in sys.argv:
        sectpr = doc.sections[0]._sectPr
        ln = OxmlElement("w:lnNumType")
        ln.set(qn("w:countBy"), "1")
        ln.set(qn("w:restart"), "continuous")
        cols = sectpr.find(qn("w:cols"))
        if cols is not None:
            cols.addprevious(ln)
        else:
            sectpr.append(ln)

    add_page_number_footer(doc)
    dst.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(dst))
    print(f"OK {dst}  paragraphs={len(doc.paragraphs)} tables={n_tbl}")


if __name__ == "__main__":
    main()