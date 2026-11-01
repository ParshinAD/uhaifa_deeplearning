"""Post-process the pandoc-built proposal docx: 11pt body font, justified text.

Pandoc's default reference docx sets the body font via docDefaults to 24 half-points
(12pt), left-aligns everything, and leaves the page geometry to Word's locale
default. This script patches the built .docx in place:

  * styles.xml   — docDefaults w:sz/w:szCs 24 -> 22 (11pt); Author/Date styles too.
                   Normal gets <w:jc w:val="both"/> (justified), which every body
                   style inherits; SourceCode gets an explicit left alignment,
                   because Word stretches soft-break lines in justified paragraphs
                   and that would mangle a code block. Title/Author/Date carry
                   their own centering and are unaffected. Paragraph spacing is
                   tightened to Word's usual rhythm (BodyText 9pt+9pt -> 0+8pt,
                   Heading1 18pt -> 12pt before), which keeps the Methodology
                   section on one page.
  * document.xml — pandoc's <w:sectPr> holds only a footnotePr, so insert an
                   explicit A4 page with 1in top/bottom + 0.8in side margins
                   (so the 6.5in figure fits), right after </w:footnotePr>.

Run after every pandoc build:
  pandoc thesis_proposal_parshin.md -o Thesis_Proposal_Aleksei_Parshin.docx
  python3 patch_docx.py Thesis_Proposal_Aleksei_Parshin.docx
"""
import re
import shutil
import sys
import zipfile
from pathlib import Path

PAGE_GEOMETRY = (
    '<w:pgSz w:w="11906" w:h="16838"/>'                      # A4
    '<w:pgMar w:top="1152" w:right="1152" w:bottom="1152" w:left="1152" '
    'w:header="720" w:footer="720" w:gutter="0"/>'           # 0.8in all around
)


def patch_styles(xml: str) -> str:
    """11pt body (docDefaults + Author/Date); justify Normal; keep code left."""
    m = re.search(r"<w:docDefaults>.*?</w:docDefaults>", xml, re.S)
    assert m, "docDefaults not found"
    patched = re.sub(r'(<w:szC?s? w:val=")24(")', r"\g<1>22\g<2>", m.group(0))
    xml = xml[: m.start()] + patched + xml[m.end():]
    for style_id in ("Author", "Date"):
        m = re.search(
            rf'<w:style [^>]*w:styleId="{style_id}".*?</w:style>', xml, re.S)
        if m:
            patched = re.sub(r'(<w:szC?s? w:val=")24(")', r"\g<1>22\g<2>", m.group(0))
            xml = xml[: m.start()] + patched + xml[m.end():]

    # Justify body text: Normal (which everything inherits from) gets jc=both.
    m = re.search(r'<w:style w:default="1" w:styleId="Normal"[^>]*>.*?</w:style>',
                  xml, re.S)
    assert m and "<w:jc" not in m.group(0), "Normal style missing or already aligned"
    patched = m.group(0).replace(
        "</w:style>", '<w:pPr><w:jc w:val="both"/></w:pPr></w:style>')
    xml = xml[: m.start()] + patched + xml[m.end():]

    # Word stretches lines ending in a soft break inside justified paragraphs,
    # which would mangle a code block — pin SourceCode to the left.
    m = re.search(r'<w:style [^>]*w:styleId="SourceCode".*?</w:style>', xml, re.S)
    assert m and "<w:wordWrap" in m.group(0), "SourceCode style not as expected"
    patched = m.group(0).replace(
        '<w:wordWrap w:val="off" />',
        '<w:wordWrap w:val="off" /><w:jc w:val="left"/>')
    xml = xml[: m.start()] + patched + xml[m.end():]

    # Tighten vertical rhythm: pandoc pads body paragraphs 9pt on BOTH sides;
    # use 0 before / 4pt after, and smaller gaps around the headings.
    assert '<w:spacing w:after="180" w:before="180" />' in xml, "BodyText spacing changed"
    xml = xml.replace('<w:spacing w:after="180" w:before="180" />',
                      '<w:spacing w:after="80" w:before="0" />')
    # Heading 2 at 14pt instead of pandoc's 16pt.
    xml = xml.replace('<w:sz w:val="32" />', '<w:sz w:val="28" />')
    assert '<w:spacing w:after="80" w:before="360" />' in xml, "Heading1 spacing changed"
    xml = xml.replace('<w:spacing w:after="80" w:before="360" />',
                      '<w:spacing w:after="60" w:before="160" />')
    assert '<w:spacing w:after="80" w:before="160" />' in xml, "Heading2/3 spacing changed"
    xml = xml.replace('<w:spacing w:after="80" w:before="160" />',
                      '<w:spacing w:after="40" w:before="100" />')
    return xml


FIG_WIDTH_EMU = 5943600            # 6.5 in — the width= the markdown asks for,
                                   # which pandoc caps below its own default page


def patch_document(xml: str) -> str:
    """Insert page size and margins; fix the figure width."""
    assert "<w:sectPr" in xml, "sectPr not found"
    assert "<w:pgSz" not in xml, "document already sets a page size"
    out, n = re.subn(r"(</w:footnotePr>)", r"\1" + PAGE_GEOMETRY, xml)
    assert n >= 1, "footnotePr not found inside sectPr"

    def resize(m: re.Match) -> str:
        cx, cy = int(m.group(2)), int(m.group(3))
        return (f'<{m.group(1)} cx="{FIG_WIDTH_EMU}" '
                f'cy="{round(cy * FIG_WIDTH_EMU / cx)}" />')

    out, n = re.subn(r'<(wp:extent|a:ext) cx="(\d+)" cy="(\d+)"\s*/>', resize, out)
    assert n == 2, f"expected exactly one image (2 extent tags), found {n}"
    return out


def main(path: str) -> None:
    src = Path(path)
    tmp = src.with_suffix(".tmp.docx")
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(
            tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/styles.xml":
                data = patch_styles(data.decode("utf-8")).encode("utf-8")
            elif item.filename == "word/document.xml":
                data = patch_document(data.decode("utf-8")).encode("utf-8")
            zout.writestr(item, data)
    shutil.move(tmp, src)
    print(f"patched {src}: body 11pt, justified, A4 geometry")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Thesis_Proposal_Aleksei_Parshin.docx")
