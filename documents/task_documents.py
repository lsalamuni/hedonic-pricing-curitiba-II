"""Task for compiling the paper."""

import shutil
import subprocess
from pathlib import Path

from hedonic_analysis.config import BLD, BLD_ANALYSIS, BLD_DATA, BLD_IMAGES, DOCUMENTS

_TABLE_MARKERS = {
    "% {{ADEQUACY_TABLE}}": "adequacy_tests.tex",
    "% {{FIRST_STAGE_TABLE}}": "first_stage_paper.tex",
    "% {{SECOND_STAGE_TABLE}}": "second_stage_paper.tex",
}


def task_compile_paper(
    paper_md: Path = DOCUMENTS / "paper.md",
    myst_yml: Path = DOCUMENTS / "myst.yml",
    refs: Path = DOCUMENTS / "refs.bib",
    classification: Path = BLD_DATA / "neighborhood_classification.parquet",
    regression: Path = BLD_DATA / "first_stage_results.parquet",
    adequacy_tex: Path = BLD_ANALYSIS / "adequacy_tests.tex",
    first_stage_tex: Path = BLD_ANALYSIS / "first_stage_paper.tex",
    second_stage_tex: Path = BLD_ANALYSIS / "second_stage_paper.tex",
    produces: Path = BLD / "paper.pdf",
) -> None:
    """Compile the paper from MyST Markdown using Jupyter Book 2.0."""
    # Copy pipeline images to documents/public/ for MyST resolution
    public = DOCUMENTS / "public"
    public.mkdir(parents=True, exist_ok=True)
    for img in sorted(BLD_IMAGES.glob("*.png")):
        shutil.copy2(img, public / img.name)

    # Substitute table placeholders with generated LaTeX content
    original_text = paper_md.read_text(encoding="utf-8")
    modified_text = original_text
    for marker, tex_name in _TABLE_MARKERS.items():
        tex_path = BLD_ANALYSIS / tex_name
        if tex_path.exists():
            modified_text = modified_text.replace(
                marker, tex_path.read_text(encoding="utf-8")
            )

    try:
        paper_md.write_text(modified_text, encoding="utf-8")
        subprocess.run(
            ("jupyter", "book", "build", "--pdf"),
            cwd=DOCUMENTS.absolute(),
            check=False,
        )
        build_pdf = DOCUMENTS / "_build" / "exports" / "paper.pdf"
        if not build_pdf.exists():
            msg = f"PDF not produced at {build_pdf}"
            raise FileNotFoundError(msg)
        shutil.copy(build_pdf, produces)
    finally:
        paper_md.write_text(original_text, encoding="utf-8")
