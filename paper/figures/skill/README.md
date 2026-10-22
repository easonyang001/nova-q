# Skill-guided workflow redesign

All figures included in the long manuscript and QCNC draft now use this workflow.
The long manuscript includes four concept diagrams; QCNC includes the workflow,
verification and measured evidence figures. Archived supplementary figures are not
included in either main manuscript and are retained as historical artifacts.

## Sources and exports

- `workflow-editable.drawio`: uncompressed native diagrams.net objects, text and edges.
- `workflow-editable.svg`: editable text and vector primitives.
- `workflow-editable.pdf`: PDF included by the long manuscript; copied into QCNC.
- `workflow-editable.png`: preview of the authored vector diagram.
- `workflow-reference.png`: AI-generated composition study, excluded from the manuscripts.
- `build_workflow.py`: shared native Draw.io/SVG construction; CairoSVG exports SVG.

Run `python paper/figures/skill/build_workflow.py` from the repository root.
Requires CairoSVG. Draw.io CLI is unavailable here: native Draw.io CLI export and
visual inspection of its rendering were not performed. PDF/PNG are SVG exports,
not claimed to be tested Draw.io exports. Both editable sources use the same primitives.

## Workflow applied

- https://github.com/pengqianhan/codex-paper-figure-skill/blob/main/codex-paper-figure-skill/SKILL.md
- https://github.com/CuiMuxuan/academic-codex-skills/blob/main/skills/academic-figure-workflow/SKILL.md

Built-in image generation explored composition; native objects were then authored
and inspected as SVG/PDF. No third-party icons, traced raster text or raster background
are present in the final figure. The reference's inaccurate coverage expression and
overstated checker descriptions were corrected rather than reproduced.

## Figure brief and generation prompt

Scientific message: a confirmed request becomes a typed model, public-data instance,
QUBO candidate and original-problem delivery; evidence can route bounded repairs to
data or modelling. Local LLM roles remain distinct from deterministic execution.

Composition prompt used the built-in image-generation tool: a wide scientific journal
method figure on white, restrained navy/teal/amber, five left-to-right stages (Clarify,
Model, Prepare, Solve, Deliver), with a request document, binary facility-selection
model, public source tables, candidate/demand map, sparse QUBO graph and final selected
sites. Modelling must define the data schema before acquisition. Include optional
modelling-pattern retrieval and an amber lower validation-feedback lane returning to
data and model roles. Prefer miniature scientific objects over identical boxes.
Avoid robots, brains, gradients, marketing graphics and performance claims. Maps and
numbers are a labelled illustrative example, not measured real-data results.

## Evidence trace and interpretation

- `QuantumQUBO Agent/qubo_auto/annealing/clarify.py`: confirmed specification and assumptions.
- `QuantumQUBO Agent/qubo_auto/annealing/modeler.py`: generated typed model and schema.
- `QuantumQUBO Agent/qubo_auto/annealing/opendata.py` and `datakit.py`: acquisition/preparation.
- `QuantumQUBO Agent/qubo_auto/annealing/datacheck.py`: separate count/total computation.
- `QuantumQUBO Agent/qubo_auto/annealing/reqcheck.py`: specification/schema-based evaluator
  and decision comparisons; it does not receive the formulation.
- `QuantumQUBO Agent/qubo_auto/annealing/workflow.py`: preflight, bounded recovery and delivery.
- Manuscript `sections/system.tex`; QCNC `sections/method.tex`: architecture and limitations.

The illustrative model maximizes covered demand using binary coverage variables,
so overlapping sites do not count demand twice. Compilation negates maximization
and introduces encoding auxiliaries; the schematic graph is not the exact Q matrix
of the drawn map. No geography, coverage radius or optimality result is asserted.
Runtime checker agreement remains evidence, not a correctness guarantee.

## Validation

SVG/PDF preview and manuscript-page rendering inspected. PDF has embedded TrueType
fonts and selectable labels; no extracted text blocks fall outside its page.
Native Draw.io XML parses, has root cells and unique object IDs. Its GUI rendering
remains unverified. Both manuscripts rebuilt without overfull boxes or undefined
references; page counts are recorded in the final response and QCNC validation file.

## Remaining diagrams and results

`build_remaining.py` builds verification-layers, recovery-routes and acceptance-design
as native Draw.io and SVG plus PDF/PNG exports. Their generated composition studies
are saved as `verification-reference.png`, `recovery-reference.png` and
`acceptance-reference.png`; these are not embedded in the manuscripts. The studies
were inspected and the final native designs were revised for manuscript fidelity:
AED is balanced grouping, YouBike is a tour, retrieval is modelling-pattern context,
and checkers/judges have distinct input visibility. Unsupported reference claims and
fake results are excluded. Native Draw.io GUI rendering remains unverified.

`build_evidence.py` uses academic-figure-workflow's quantitative-plot branch.
Image generation does not create measured plots. It reads original benchmark counts,
backend/hybrid statistics and all three actual `result.json` gaps per displayed
condition. Assertions match those points to saved summary means. Hollow points show
repetitions and short vertical marks show means; no confidence intervals are invented.
Matplotlib sources and editable SVG are provided rather than a Draw.io copy of a plot.

All replacements were inspected as exported images and in the manuscripts. Native
Draw.io XML parses with unique object IDs, SVG retains real text, PDF embeds fonts,
and extracted text blocks remain within the figure boundaries. QCNC's source package
and validation record are rebuilt after this revision. Experimental values and
formal evidence limitations remain unchanged.

## User-selected illustration update

The user identified `verification-reference.png` as the desired paper figure. It is now
embedded directly in both manuscripts, replacing the simplified verification vector
reconstruction. The caption marks values and pass symbols as illustrative, clarifies
profile-based agent visibility and preserves checker limitations. Pixel hashes of the
embedded PDF image match the selected PNG. The retained Draw.io file is the earlier
simplified alternative and is not an exact editable reconstruction of the selected raster.

## Selected workflow illustration

`workflow-illustrated.png` is now the actual workflow figure in both manuscripts.
The built-in image editor preserved the user-selected detailed composition while
correcting coverage double-counting, the QUBO variable domain and checker scope.
The old Draw.io/SVG workflow remains an editable simplified alternative, not the
source of the raster now embedded. PDF image pixels were matched to the final PNG.
Only `paper/main.pdf` and `paper/qcnc2027/main.pdf` are current manuscript outputs.
