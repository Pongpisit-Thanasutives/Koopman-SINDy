# Independent final consistency validation

The numerical evidence is checked independently by `validate_package.py` and
its four local helper modules. The validator does not import the benchmark or
reporting modules and does not rerun discovery experiments. Run it from any
directory:

```bash
python /path/to/code_repository/results/final_validation/validate_package.py
```

For the new weak-form evidence alone:

```bash
python /path/to/code_repository/results/final_validation/validate_weak_outputs.py
```

The accompanying `validation_report.json` records the checked dataset hashes and
individual outcomes. This validates consistency of the delivered evidence; it
does not turn the conditional benchmark findings into a universal performance
claim.

Final result: **14,800 of 14,800 package checks pass**, with no unresolved
failures. The retained weak-evidence audit passes **1,832 of 1,832 checks**.

The current pass reorganizes files for submission. `latex_source` contains only
clean-manuscript compilation dependencies; compiled PDFs and the cover letter are
in `submission`, while build scripts, marked/response sources and unused assets
are in `local_checking`. The scientific manuscript and all experimental evidence
are unchanged. Historical preservation records retain their original paths and
hashes; an explicit relocation map resolves those files at their new locations.
Original build-script and instruction-file copies remain under
`local_checking/legacy_build` for the historical hash checks. No validation checks were removed and no experiments
were rerun. The isolated source archive contains exactly 33 compilation files.
All three rebuilt PDFs retain the same 31/32/7 page counts and are identical to
V7 in extracted text and rendered page images. Marked-source regeneration also
reproduces the stored marked TeX and complete diff byte for byte.

The preceding title-only pass shortens only Appendix D's title to **Weak-form preprocessing
comparison**. Exact source comparison confirms this single heading replacement;
322 other source and evidence files remain byte-identical to delivered V6.
All three PDFs were rebuilt. Their text is unchanged apart from the heading
(with marked-copy underline glyphs ignored during text comparison). Only page 28
changes visually in the clean and marked manuscripts; every other page and all
seven response pages are pixel-identical to V6. The affected pages and their
neighboring pages were inspected visually. Document and appendix fingerprints
were refreshed, historical preservation records were retained, and no experiments
were rerun.

The preceding response-only pass incorporated the author-supplied opening verbatim
and identifies TV as an established SINDy comparator. All 15 reviewer quotations
and all numerical statements in the responses remained unchanged. A local spacing
adjustment in the final correction list retains the seven-page response with
11-point text and 25-mm margins. At that pass, the clean and marked manuscripts, their sources,
and all code and experimental evidence remained byte-identical to delivered V5:
**327 protected files** are recorded in response_preservation_v6.json.
The updated response was compiled twice, every manuscript reference resolves,
and all seven pages were inspected visually. No experiments were rerun.


The preceding manuscript editorial and layout pass preserves **253 protected files** from
the previous delivered package: all 172 CSVs, 38 figure assets, 19 numerical and
execution JSON records, 21 numerical/helper Python sources, and three execution
shell scripts are byte-identical. All 14 tables retain every numeric token; all
nine figures retain their contents and captions. The Table 2 heading now reads
`#Seeds`, and float placement has been adjusted to reduce unused page space.
Four reporting sources changed solely to reproduce these float placements.
Exact source comparisons after narrowly removing the documented presentation
controls verify these changes; the earlier AST proof also continues to protect
the numerical implementations and configurations. The previous boldface edits
to Tables 5, 6, 8 and 14 and the requested caption changes are retained.

`editorial_preservation_v5.json` records the V4-to-V5 preserved fingerprints,
original table numeric tokens, figure content fingerprints, and exact permitted
presentation edits. `presentation_preservation.json` retains the earlier V4
reporting proof, its historical fingerprints, and the current presentation
fingerprints. Original execution manifests remain unchanged. Current reporting
manifests identify the updated presentation sources separately. No experiments,
numerical outputs, or figure assets were regenerated.

The standard-error definition added to Section 4 agrees with the implemented
intercept-only seed-cluster CR1 formula. It retains the record-weighted mean and
accommodates the two unbalanced optDMD summaries; on a balanced grid it equals
the sample standard deviation of seed means divided by the square root of the
number of seeds. The strengthened response to Reviewer 2 reports the existing
lower-noise Van der Pol advantage; its quoted errors, Scores, and support count
were checked directly against the saved records.

Validated items include:

- Record counts, unique case identifiers, finite successful results, metric
  ranges, and the per-record score definition.
- Main, sensitivity, strategy, classical-comparison, high-noise TV, and non-oracle
  summary statistics recalculated directly from raw records.
- Record-weighted means, seed-cluster standard errors, paired improvement counts,
  and their valid denominators.
- The full 720-row TV comparison, both displayed noise bands, every displayed
  Table 7 value, and all 520 saved matched contrasts and their grouped summaries.
  The 20% Van der Pol statements are checked at each sampling factor, including
  all five paired fine-grid error/Score improvements and the coarse-grid
  reversal in mean Score.
- Main and appendix coefficient examples against their raw-record metrics;
  high-noise and non-oracle TV coefficient vectors against reported F1,
  coefficient error, and exact-support outcomes.
- Non-oracle EBIC arithmetic and the agreement of all 40 shared equation records
  between the original non-oracle experiment and its matched TV extension.
- Retained main comparator records, the shared main/Appendix C ODE results, and
  the synchronized standalone Appendix C POD records.
- All 300 weak-form method evaluations on 60 datasets: complete frozen coverage,
  identical noisy-input hashes across paired methods, 3,840 coefficient entries,
  coefficient-derived F1/error/exact-support metrics, fixed physical grids and
  test counts, every condition/pool/paired summary, and all ten displayed table
  rows. Pooled uncertainty uses five independent seed averages rather than
  treating overlapping test windows as independent observations.
- Weak-form TV tuning: all 2,430 validation fits and 90 final fits converge at the
  stated tolerance; selected penalties minimize the saved observation-only
  validation scores. All 19 analytic identity/quadrature and clean-resolution
  checks also pass in the recorded library validation.
- Source/output fingerprints in the complex-rerun, TV, non-oracle TV, weak-form,
  and reporting manifests; copied manuscript figures and all current manuscript
  input/figure paths. Original execution manifests remain unchanged. The current
  TV reporting wrapper and other presentation-only changes are checked against
  separate reporting manifests that also protect the original numerical inputs.

The two retained main optDMD failures are explicitly represented in the raw
records and reported valid counts. They are excluded from successful-record
metrics. Large finite errors remain included. Archived/intermediate outputs are
identified separately from the reported corrected results.

The mathematical description was also read against the implementation: complex
evolution precedes real physical readout, the spectral branch and regularization
are specified, unequal-lag terminal pairs are excluded from map fitting, and the
strategy ablation uses complete uniform intervals. The fixed-rank POD control
and both TV comparisons support the manuscript's qualified interpretation.

The endpoint clarification states the implemented derivative stencil
explicitly: `(y[i+1] - y[i-1]) / (t[i+1] - t[i-1])`. It is second order on a
uniform grid. If the terminal interval is shorter, its truncation error at the
final interior sample is generally first order. This describes the existing
computation; no derivative formula, numerical result, or experiment was changed.

The new weak-form equations were independently read against the implementation.
The compact tests, signs from integration by parts, physical scaling, periodic
spatial windows, and exact integration of nodal-feature interpolants are
consistent. All arms use the same physical test supports. The PDE nuisance term
`u*u_xx` is explicitly replaced by `(u^2)_xx`, so comparisons are within the
shared weak library and are not described as an otherwise identical strong-form
comparison. This is a fixed-test diagnostic, not a reproduction of a complete
WSINDy package or its adaptive model-selection procedure.

The weak results are reported conditionally. For Van der Pol, EDMD has a lower
pooled median coefficient error but a higher mean coefficient error and lower
pooled support/Score than raw or TV states. Its low-noise gains and the stronger-
noise reversals are both stated. For Burgers, raw weak regression is already
accurate and EDMD does not improve pooled recovery. All planned conditions remain
in the saved records and aggregate table; no outcomes were discarded.

The baseline labels in Tables 3, 4, 9, 12, and 13 were traced to code: they use
raw noisy observations with temporal central differences, no temporal insertion,
no smoothing and no POD. PDE spatial derivatives are Fourier spectral
derivatives. The PDE sensitivity control at `q=1` is separately identified as
POD-only; the weak-form raw comparator does not use finite differences.

The final clean manuscript (31 pages), reviewer response (7 pages), and marked
manuscript (32 pages) pass the independent PDF text and final-log scan: no blank
pages, replacement characters, unresolved references/citations, duplicate labels,
horizontal overflows, or LaTeX errors were found. The 19 active manuscript source
files resolve all inputs, figures, equation/table/section references and 32 cited
bibliography keys. `document_validation.json` records the final PDF/source
fingerprints and confirms that all 15 reviewer quotations remain verbatim and
all response references to the manuscript resolve. Inherited title-page
vertical-box/font warnings were inspected
visually during production. All-page visual inspection was performed separately;
the saved-data validator does not replace that inspection. Build auxiliaries,
logs, caches and the obsolete prior-revision QA image directory are not required
by this script.
