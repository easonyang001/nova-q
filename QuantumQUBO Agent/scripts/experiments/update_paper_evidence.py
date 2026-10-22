"""Generate a LaTeX section from audited saved results; never alter historical experiment tables."""
import json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];PAPER=ROOT.parent/'paper';E=ROOT/'results/paper_evidence_20261011'
pen=json.loads((E/'penalty_statistics.json').read_text());back=json.loads((E/'backend_statistics.json').read_text());enc=json.loads((E/'encoding_statistics.json').read_text());route=json.loads((E/'routing_statistics.json').read_text())
(PAPER/'figures').mkdir(exist_ok=True)
for name in ['penalty_comparison','backend_comparison','encoding_comparison','routing_comparison']:
 if (E/(name+'.pdf')).exists():shutil.copyfile(E/(name+'.pdf'),PAPER/'figures'/(name+'.pdf'))
escape=lambda s:s.replace('_',r'\_')
lines=[r'\subsection{Additional backend, penalty and encoding experiments}',r'\label{sec:additionalexperiments}',
'We froze the models before comparing policies and distinguish these experiments from live model generation.',
'The comparison uses a common solver-call ceiling within each backend; actual calls can differ through early stopping.',
'Compute seconds across simulated annealing and a GPU service are not equivalent resources.',
r'\paragraph{Extended penalty comparison.}',
'The eight synthetic test instances were repeated with ten solver seeds under eight policies, giving 640 runs.',
'Each run allowed at most six calls, with 60 SA reads and 300 sweeps per call. Every run returned a feasible result.',
'Table~\\ref{tab:extendedpenalty} reports mean gaps and hierarchical bootstrap intervals (10,000 draws, resampling',
'instances and then seeds within instances). Paired Wilcoxon tests use the eight per-instance seed-averaged',
'differences, rather than treating 80 runs as independent problems. Holm correction is applied to the four',
'penalty-versus-default tests. All four penalty policies had higher mean gaps than default (adjusted',
'$p=0.03125$). The result concerns this small fixed test set and does not establish a ranking across applications.',
r'\begin{table}[t]\centering\small',r'\caption{Extended local SA comparison. Intervals are 95\% hierarchical bootstrap intervals; all policies have 80 runs.}',r'\label{tab:extendedpenalty}',r'\begin{tabular}{@{}lrrr@{}}\toprule Policy & Mean gap & 95\% interval & Calls \\\midrule']
for row in pen['summaries']:lines.append(f"{escape(row['policy'])} & {row['mean_gap']:.4f} & [{row['ci95'][0]:.4f}, {row['ci95'][1]:.4f}] & {row['mean_calls']:.2f} \\")
lines += [r'\bottomrule\end{tabular}\end{table}',r'\begin{figure}[t]\centering\includegraphics[width=\linewidth]{figures/penalty_comparison.pdf}',r'\caption{Penalty comparison on eight synthetic instances and ten seeds, separate from the earlier agent comparison.}\end{figure}',r'\paragraph{Actual backend calls.}',
'We evaluated the frozen red-wine model and generated knapsack (60 items) and conflict selection (80 items)',
'models under default, penalty TPE and single-agent policies, with three repetitions per condition on SA and',
'CGA. All 54 conditions produced feasible solutions without external-call errors. Each condition allowed',
'at most three calls; CGA calls used a three-second search limit, while SA used 60 reads and 300 sweeps.',
'External repetitions are repeated submissions, since the GPU service exposes no RNG seed.',
'On CGA, the red-wine mean gaps were 0.5495 for default, 0.0740 for TPE and 0.0000 for the agent.',
'On the other two CGA models, defaults already attained the exact reference. Agent mean end-to-end times',
'on CGA were 73.9--97.0 seconds per condition, versus 19.5--23.7 seconds for defaults, including network',
'and local inference overhead. Quality gains therefore do not imply lower wall time. Three repetitions',
'are insufficient to establish a general hardware advantage.',r'\begin{figure}[t]\centering\includegraphics[width=\linewidth]{figures/backend_comparison.pdf}',r'\caption{Fixed-model SA and real CGA comparison, three repetitions per condition. The references are exact for these three models.}\end{figure}',
'An actual Fujitsu DA submission was rejected with HTTP 400 and the service message',
"``Monthly usage exceeds specified metering limit.'' Further submissions were withheld. This is an",
'external account limitation, not evidence of model infeasibility; no simulated stand-in is counted as a DA result.',
r'\paragraph{Encoding ablation.}',
'Two fixed knapsack instances (12 and 30 items) were compiled with binary slack, unary slack and unbalanced',
'encoding, keeping original objectives, constraints, default weights and call ceilings fixed. There were',
'60 SA conditions (ten seeds each) and 18 actual CGA conditions (three submissions each). For 30 items,',
'binary, unary and unbalanced encodings used 38, 163 and 30 variables, respectively; SA mean gaps were',
'0.1661, 0.3250 and 0.1188. CGA attained the reference in all three encodings for that instance, but',
'unbalanced encoding missed the optimum on the 12-item instance (gap 0.0387). Fewer variables can help',
'search, but an approximate encoding does not guarantee preservation of the original optimum. These',
'are two instance-specific ablations, not a broad evaluation of automatic encoding selection.',
r'\begin{figure}[t]\centering\includegraphics[width=\linewidth]{figures/encoding_comparison.pdf}',r'\caption{Encoding ablation with original-problem feasibility evaluation and exact knapsack references.}\end{figure}']
if route['completed']==18:
 lines += [r'\paragraph{A larger real-coordinate routing model.}',
'A fixed Taipei YouBike snapshot supplied 40 active station coordinates. A hand-written position-encoding',
'TSP model has 1,600 binary decision variables and was checked against direct tour evaluation on 12',
'valid tours. This experiment measures solving and tuning, not autonomous LLM formulation. Distances',
'are haversine kilometres, not road travel distances. A 20-restart 2-opt heuristic gave a feasible',
f"{route['baseline_km']:.6f}-km tour; it is not a certified optimum. Each policy/backend condition was",
'repeated three times with a three-call ceiling (30 reads and 300 sweeps for SA, three-second CGA calls).',
r'\begin{table}[t]\centering\small',r'\caption{40-station routing. Mean distances and relative differences use feasible runs only; feasibility counts report all runs. The comparison is to a feasible heuristic, not an exact optimum.}',r'\label{tab:largerouting}',r'\begin{tabular}{@{}llrrr@{}}\toprule Backend & Policy & Feasible & Mean km & Heur. gap \\\midrule']
 for r in route['groups']:
  mean=f"{r['mean_km']:.2f}" if r['mean_km'] is not None else '--';gap=f"{r['mean_gap']:.3f}" if r['mean_gap'] is not None else '--'
  lines.append(f"{r['backend']} & {escape(r['policy'])} & {r['feasible']}/{r['n']} & {mean} & {gap} \\")
 lines += [r'\bottomrule\end{tabular}\end{table}',r'\begin{figure}[t]\centering\includegraphics[width=\linewidth]{figures/routing_comparison.pdf}',r'\caption{Large routing quality. Missing feasible outputs are recorded separately rather than assigned an arbitrary gap.}\end{figure}']
lines += [r'\paragraph{Additional live-data acceptance and checker corrections.}',
'Three fresh wine model/adapter generations were retained. The initial acceptance checker incorrectly',
'rejected equivalent minimization of the negative objective, required an unnecessary input field',
'even when the confirmed setting was embedded in the model, and loaded unrelated unreadable candidate',
'files. After correcting the checker, all three saved outputs passed the full 2,048-assignment audit',
r'and delivered feasible solutions. The re-audit reused artifacts; it is not a prospective 100\% success',
'rate for a fixed autonomous acceptance protocol. Initial errors and corrected evaluations are both retained.',
r'The Wholesale Customers case used the public UCI data\footnote{\url{https://archive.ics.uci.edu/dataset/292/wholesale+customers}}, a fixed sample of 24 rows and six spending',
'features. Independent checks reproduced the sample, log transform, standardization and distance matrix.',
'The recovered run formed three groups of eight, with objective 856.8319 versus a feasible baseline',
r'of 1021.7818 (16.1\% lower). The run reused its generated model and obtained data and adapter code anew.',
'Failures exposed fixed-size requirement maps and an unsupported generated import of the preprocessing',
'module; the implementation now supports specific family-prefix mappings and tracked imports.']
karate_path=ROOT/'results/acceptance/karate_live_20261011_v4/acceptance.json'
if karate_path.exists():
 k=json.loads(karate_path.read_text())
 if k['success']:
  lines+=[r'A recovered Zachary karate-club case used the university-published 34-node, 78-edge GML archive\footnote{\url{https://websites.umich.edu/~mejn/netdata/}}.',
'An independent source check rejected an adapter that sorted adjacency columns lexicographically,',
'permuting columns without rows. Numeric ordering restored the source matrix. The delivered split',
'has 17 nodes per group and 17 cut edges, using the feasible conventional baseline because annealing',
'returned a worse best result (22 cut edges). This is attributed baseline delivery, not an annealer',
'improvement. Declared descriptive notes were discharged by the case-specific independent source audit.',
'Recovered examples do not establish general LLM diagnosis or repair success rates.']
lines += ['The suite now passes 120 automated tests, and UI checks retain feasible/failure delivery and',
'cumulative checkpoint resume. Raw instances, configurations, source hashes, transcripts, error records',
'and standalone figure artifacts accompany the report. General requirement-semantic correctness,',
'a held-out backend-transfer experiment and execution of the entire ten-case catalog remain open.']
# Ensure row endings contain two backslashes (generated ordinary f-strings above use one).
lines=[line+'\\' if line.endswith('\\') and not line.endswith('\\\\') else line for line in lines]
(PAPER/'sections/additional_experiments.tex').write_text('\n'.join(lines)+'\n')
print(PAPER/'sections/additional_experiments.tex')
