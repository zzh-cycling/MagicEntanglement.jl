# Repository Guidelines

## Project Structure & Module Organization

`src/MagicEntanglement.jl` defines the module, includes implementation files, and exports the public API. `src/Logical.jl` implements logical Pauli operations; `src/Entanglement.jl` provides GF(2) helpers and entanglement calculations. Matching suites live in `test/Logical.jl` and `test/Entanglement.jl`, loaded by `test/runtests.jl`.

`exm/` contains experimental physics and MIPT scripts, including standalone checks outside the package test suite. Documentation lives in `docs/src/`, with `docs/make.jl` as the Documenter entry point. GitHub Actions workflows are in `.github/workflows/`.

## Build, Test, and Development Commands

Run commands from the repository root. Prefer Julia 1.11 or 1.12, which CI tests alongside prerelease Julia.

- `julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'` installs dependencies and precompiles the package.
- `julia --project=. -e 'using Pkg; Pkg.test()'` runs the complete package test suite.
- `julia --project=. -e 'using MagicEntanglement; include("test/Entanglement.jl")'` runs the entanglement suite directly.
- `julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'` prepares the documentation environment.
- `julia --project=docs docs/make.jl` builds documentation into `docs/build/`; the script also configures deployment for CI.

## Coding Style & Naming Conventions

Use four-space indentation and descriptive `snake_case` function names, matching existing code. Retain mathematical Unicode identifiers where they clarify formulas. Document public functions with Julia docstrings, signatures, assumptions, and small examples. Add public exports in `src/MagicEntanglement.jl`. No formatter or linter configuration is present; follow surrounding style.

## Testing Guidelines

Use Julia's `Test` standard library and descriptive `@testset` names grouped by function or physical scenario. Add regression tests to the matching module suite. Compare numerical results with `≈` and explicit tolerances where needed; use small known states or brute-force reference calculations. CI collects coverage for Codecov, but no minimum threshold is configured.

## Commit & Pull Request Guidelines

History uses short, action-oriented subjects such as `fix MIPT, add git ignore`; no strict commit format is established. Write specific subjects describing the affected behavior. PRs should explain the change, link relevant issues, and report validation commands and results. For mathematical changes, state assumptions and reference checks; update API documentation when behavior changes.
