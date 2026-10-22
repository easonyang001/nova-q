"""Run the QUBO pipeline on a single benchmark folder.

Usage:
    # Existing cases already in cases/*.json:
    python scripts/run_single.py legacy/benchmarks/max_cut

    # Provide test cases as a text file — built after formulation, before coding:
    python scripts/run_single.py legacy/benchmarks/max_cut --input my_cases.txt

The benchmark folder must contain:
    prompt.txt      — natural-language problem statement
    cases/*.json    — pre-computed test cases (or provide --input to generate them)
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from qubo_auto.pipeline import Pipeline, load_config
from qubo_auto.verification.test_cases import load_prompt, load_test_cases


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run QUBO pipeline on one benchmark folder")
    parser.add_argument("benchmark_dir", help="Path to benchmark folder (containing prompt.txt)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--input", "-i",
        default=None,
        help="Text file with test cases in any human-readable format. "
             "Cases are built after formulation so the LLM has full QUBO context.",
    )
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    config = load_config(root / args.config)

    benchmark_dir = Path(args.benchmark_dir)
    if not benchmark_dir.is_absolute():
        benchmark_dir = root / benchmark_dir
    if not benchmark_dir.is_dir():
        print(f"[error] {_rel(benchmark_dir, root)} is not a directory")
        sys.exit(1)

    benchmark_name = benchmark_dir.name
    nl_problem = load_prompt(benchmark_dir)

    # Resolve raw test input if provided
    raw_test_input: str | None = None
    if args.input:
        input_path = Path(args.input)
        if not input_path.is_absolute():
            input_path = Path.cwd() / input_path
        if not input_path.exists():
            print(f"[error] Input file not found: {_rel(input_path, root)}")
            sys.exit(1)
        raw_test_input = input_path.read_text(encoding="utf-8")
        print(f"Benchmark    : {benchmark_name}")
        print(f"Test input   : {_rel(input_path, root)} (cases built after formulation)")
        print(f"Seed         : {args.seed}")
        test_cases = []
    else:
        test_cases = load_test_cases(benchmark_dir)
        print(f"Benchmark    : {benchmark_name}")
        print(f"Test cases   : {len(test_cases)} ({', '.join(c.name for c in test_cases)})")
        print(f"Variable sizes: {[c.n_variables for c in test_cases]}")
        print(f"Seed         : {args.seed}")

    pipeline = Pipeline(
        config=config,
        transcript_root=root / "transcripts",
        results_path=root / "results" / "runs.jsonl",
    )

    result = pipeline.run(
        nl_problem=nl_problem,
        benchmark_name=benchmark_name,
        seed=args.seed,
        test_cases=test_cases,
        benchmark_dir=benchmark_dir,
        raw_test_input=raw_test_input,
    )

    print(f"\nStatus:       {result.status}")
    print(f"Iterations:   {result.n_iterations}")
    print(f"Wall time:    {result.wall_time_s:.1f}s")
    print(f"Tokens used:  {result.tokens_used}")
    print(f"Verification: {result.verification}")
    if result.failure_modes:
        print(f"Failure modes: {result.failure_modes}")
    if result.error_trace:
        print(f"\nError trace:\n{result.error_trace[:1000]}")

    if result.status == "success":
        _generate_latex_deliverable(
            root=root,
            config=config,
            result=result,
            nl_problem=nl_problem,
            benchmark_name=benchmark_name,
        )

    print(f"\nFull result:\n{result.model_dump_json(indent=2)}")


def _generate_latex_deliverable(root, config, result, nl_problem, benchmark_name):
    import json
    from pathlib import Path

    transcript_dir = root / "transcripts" / result.run_id
    spec_path = _latest_file(transcript_dir, "restructure")
    formulate_path = _latest_file(transcript_dir, "formulate")
    if not spec_path or not formulate_path:
        print("\n[latex] Could not find transcripts to generate LaTeX — skipping.")
        return

    from qubo_auto.agents.writer_agent import generate_latex
    from qubo_auto.llm_client import LLMClient
    from qubo_auto.schemas import QUBOFormulation, StructuredSpec

    spec_raw = json.loads(Path(spec_path).read_text(encoding="utf-8"))["response"]
    formulation_raw = json.loads(Path(formulate_path).read_text(encoding="utf-8"))["response"]

    try:
        spec = StructuredSpec.model_validate(json.loads(spec_raw))
        formulation = QUBOFormulation.model_validate(json.loads(formulation_raw))
    except Exception as e:
        print(f"\n[latex] Could not parse spec/formulation for LaTeX: {e} — skipping.")
        return

    client = LLMClient(
        transcript_dir=root / "transcripts",
        run_id=result.run_id + "_latex",
        max_tokens=50_000,
    )
    models = config["models"]
    model = models.get("formulizer_agent", models.get("planner_agent"))

    print(f"\n[latex] Generating LaTeX deliverable (model={model}) ...")
    try:
        latex = generate_latex(
            nl_problem=nl_problem,
            spec=spec,
            formulation=formulation,
            client=client,
            model=model,
        )
    except Exception as e:
        print(f"[latex] Generation failed: {e}")
        return

    out_dir = root / "legacy" / "deliverables" / benchmark_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{benchmark_name}_{result.seed}.tex"
    out_path.write_text(latex, encoding="utf-8")
    print(f"[latex] Saved to {_rel(out_path, root)}")
    print(f"\n{'='*60}")
    print(latex)
    print(f"{'='*60}\n")

    _compile_pdf(out_path)


def _extract_body(full_doc: str) -> str:
    import re
    m = re.search(r"\\begin\{document\}(.*?)\\end\{document\}", full_doc, re.DOTALL)
    return m.group(1).strip() if m else full_doc


def _compile_pdf(tex_path: Path) -> None:
    import subprocess
    import shutil
    import tempfile

    full_doc = tex_path.read_text(encoding="utf-8")
    fragment = _extract_body(full_doc)

    # HTML is the primary output — always generate and open it
    html_path = _generate_html(tex_path, fragment)
    if html_path:
        _open_file(html_path)

    # PDF is bonus — compile only if pdflatex is available
    if not shutil.which("pdflatex"):
        print("[pdf] pdflatex not found — skipping PDF. Install MiKTeX or TeX Live to enable.")
        return

    print(f"[pdf] Compiling {tex_path.name} ...")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_tex = Path(tmpdir) / tex_path.name
        tmp_tex.write_text(full_doc, encoding="utf-8")
        cmd = [
            "pdflatex",
            "--disable-installer",   # suppress MiKTeX on-the-fly package install prompts
            "-interaction=nonstopmode",
            f"-output-directory={tmpdir}",
            str(tmp_tex),
        ]
        for pass_num in (1, 2):
            proc = subprocess.run(cmd, capture_output=True, text=True, cwd=tmpdir)
            if proc.returncode != 0:
                print(f"[pdf] pdflatex pass {pass_num} failed — HTML viewer is still available.")
                return

        tmp_pdf = Path(tmpdir) / tex_path.with_suffix(".pdf").name
        if tmp_pdf.exists():
            import shutil as sh
            sh.copy2(tmp_pdf, tex_path.with_suffix(".pdf"))
            print(f"[pdf] PDF also saved to {tex_path.parent.name}/{tex_path.with_suffix('.pdf').name}")


def _generate_html(tex_path: Path, fragment: str) -> Path | None:
    import html as html_lib
    import re

    # Extract LLM-suggested title from \title{...} in the full .tex file
    full_doc = tex_path.read_text(encoding="utf-8")
    title_match = re.search(r"\\title\{([^}]+)\}", full_doc)
    page_title = title_match.group(1) if title_match else tex_path.stem.replace("_", " ").title()

    body_html = fragment
    # Convert ANY \paragraph{...} to <h3>
    body_html = re.sub(r"\\paragraph\{([^}]+)\}", lambda m: f"<h3>{m.group(1).rstrip('.')}</h3>", body_html)
    # Convert \subsection* to <h2> so it renders like the PDF section header
    body_html = re.sub(r"\\subsection\*?\{([^}]+)\}", lambda m: f"<h2 class='subsection'>{m.group(1)}</h2>", body_html)
    # Convert \textbf{...} to <strong>
    body_html = re.sub(r"\\textbf\{([^}]+)\}", lambda m: f"<strong>{m.group(1)}</strong>", body_html)
    # Convert \textit{...} and \emph{...} to <em>
    body_html = re.sub(r"\\(?:textit|emph)\{([^}]+)\}", lambda m: f"<em>{m.group(1)}</em>", body_html)
    # Convert list environments to HTML
    body_html = re.sub(r"\\begin\{itemize\}(.*?)\\end\{itemize\}", lambda m: f"<ul>{m.group(1)}</ul>", body_html, flags=re.DOTALL)
    body_html = re.sub(r"\\begin\{enumerate\}(.*?)\\end\{enumerate\}", lambda m: f"<ol>{m.group(1)}</ol>", body_html, flags=re.DOTALL)
    body_html = re.sub(r"\\item\b", "<li>", body_html)
    # Remove \maketitle — title is rendered as the page <h1>
    body_html = re.sub(r"\\maketitle\b", "", body_html).strip()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{html_lib.escape(page_title)}</title>
  <script>
    MathJax = {{ tex: {{
      inlineMath: [['$','$']],
      displayMath: [['\\\\[','\\\\]']],
      tags: 'ams'
    }},
    options: {{ skipHtmlTags: ['script','noscript','style','textarea','pre'] }} }};
  </script>
  <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
  <style>
    body       {{ font-family: Georgia, serif; max-width: 960px; margin: 60px auto;
                  padding: 0 24px; line-height: 1.8; color: #222; background: #fafafa; }}
    h1         {{ border-bottom: 2px solid #333; padding-bottom: 6px; font-size: 1.55em; margin-bottom: 4px; }}
    h2.subsection {{ font-size: 1.15em; color: #333; margin-top: 0.4em; margin-bottom: 0.2em;
                     font-weight: bold; }}
    h3         {{ margin-top: 1.6em; color: #333; }}
    pre        {{ background: #f4f4f4; padding: 14px; border-radius: 6px; position: relative;
                  overflow-x: auto; font-size: 0.82em; border: 1px solid #ddd; }}
    .rendered  {{ background: #fff; border: 1px solid #ddd; border-radius: 6px;
                  padding: 28px; margin-top: 12px; overflow-x: auto; }}
    mjx-container[display="true"] {{ overflow-x: auto; max-width: 100%; }}
    .src-header {{ display: flex; align-items: center; justify-content: space-between;
                   margin-top: 2em; margin-bottom: 0; }}
    .src-header h3 {{ margin: 0; }}
    .copy-btn  {{ padding: 4px 14px; font-size: 0.8em; cursor: pointer; font-family: sans-serif;
                  background: #333; color: #fff; border: none; border-radius: 4px; }}
    .copy-btn:hover {{ background: #555; }}
    footer     {{ margin-top: 60px; font-size: 0.8em; color: #999;
                  border-top: 1px solid #ddd; padding-top: 12px; }}
  </style>
</head>
<body>
  <h1>{html_lib.escape(page_title)}</h1>
  <p><em>Generated by the QUBO Auto-Formulation Pipeline</em></p>

  <div class="rendered">
    {body_html}
  </div>

  <div class="src-header">
    <h3>LaTeX Source</h3>
    <button class="copy-btn" onclick="copyLatex(this)">Copy</button>
  </div>
  <pre id="latex-src">{html_lib.escape(fragment)}</pre>

  <script>
    function copyLatex(btn) {{
      var text = document.getElementById('latex-src').textContent;
      navigator.clipboard.writeText(text).then(function() {{
        btn.textContent = 'Copied!';
        setTimeout(function() {{ btn.textContent = 'Copy'; }}, 2000);
      }}).catch(function() {{
        btn.textContent = 'Error';
        setTimeout(function() {{ btn.textContent = 'Copy'; }}, 2000);
      }});
    }}
  </script>

  <footer>
    Save <code>{tex_path.name}</code> and compile with
    <a href="https://miktex.org">MiKTeX</a> or TeX Live for a PDF.
  </footer>
</body>
</html>"""

    html_path = tex_path.with_suffix(".html")
    html_path.write_text(html, encoding="utf-8")
    print(f"[html] Saved to {html_path.parent.name}/{html_path.name}")
    return html_path


def _open_file(path: Path) -> None:
    import subprocess
    import sys

    print(f"[view] Opening {path.name} ...")
    try:
        if sys.platform == "win32":
            subprocess.Popen(["start", "", str(path)], shell=True)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception as e:
        print(f"[view] Could not open automatically: {e}")


def _latest_file(transcript_dir: Path, step_name: str):
    if not transcript_dir.exists():
        return None
    matches = sorted(
        transcript_dir.glob(f"{step_name}_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


if __name__ == "__main__":
    main()
