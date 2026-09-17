# Development Principles and SOLID Prompts (Relax Project)

## Development Principles

| Principle | What to investigate | Boundary |
|-----------|---------------------|----------|
| Simple and correct | Branches, wrappers, configuration, or dependencies that add no current behavior or clarity | Prefer the smallest clear design that satisfies the requirement, not the fewest lines |
| Handle realistic failures | Fallbacks for states excluded by established internal contracts, or defaults that conceal broken invariants | Validate external input at boundaries; preserve justified timeout, retry, and cleanup behavior for real I/O and worker failures |
| Small, cohesive functions | Mixed responsibilities, unclear inputs/outputs, or dependencies that make a change hard to reason about | Split by responsibility; function length alone is not a finding |
| Prefer pure functions and immutable data | Hidden mutation of caller-owned data, implicit dependencies, or shared state with unclear ownership | Keep side effects at explicit boundaries; owned in-place tensor operations can be appropriate when contracts, autograd, and performance justify them |
| Refactor around real concepts or existing commonality | Repeated policy branches drifting apart, patch layers, or generic frameworks for hypothetical future needs | Refactor within the required scope; a second use or a count of duplicates is not an automatic mandate for abstraction |
| One authoritative source per fact | Derived values stored and updated independently, with inconsistent update or invalidation paths | Caches and snapshots need an explicit owner, lifetime, and consistency contract; justified caching is not inherently duplication |

### Calibration Examples

- **Report:** two independently updated fields encode the same rollout state, and a cancellation path updates only one. **Do not report:** a snapshot deliberately captures state at a documented version boundary.
- **Report:** a fallback converts a violated batch contract into apparently valid training data. **Do not report:** boundary validation rejects malformed external input, or a bounded retry handles a transient worker failure.
- **Report:** repeated backend policy branches already disagree for a supported mode. **Do not report:** a short dispatch over a closed set of modes, or a cohesive function solely because it exceeds a line count.

## SOLID Quick Reference

| Principle | Key Question | Red Flag |
|-----------|-------------|----------|
| **SRP** | "What is the single reason this module would change?" | File mixes unrelated concerns (e.g., data loading + loss computation + logging) |
| **OCP** | "Do supported variants duplicate policy that already needs coordinated edits?" | Repeated dispatch branches drift in behavior; a single explicit dispatch may be appropriate |
| **LSP** | "Does this implementation satisfy the contract its callers rely on?" | A supported operation fails or changes semantics for one implementation |
| **ISP** | "Do all implementers use all methods?" | ABC with many abstract methods, most left as stubs by implementers |
| **DIP** | "Are dependencies and side effects explicit where they affect behavior or testing?" | Hidden I/O or global state prevents isolation; concrete dependencies alone are not a defect |

______________________________________________________________________

## Common Code Smells

| Smell | Signs |
|-------|-------|
| **Mixed responsibilities** | Unrelated behavior and dependencies make changes or validation difficult |
| **Feature envy** | Method uses more data from another class than its own |
| **Data clumps** | Same group of parameters passed together repeatedly |
| **Unclear data contract** | Callers disagree on required fields, types, or ownership; choose a representation that clarifies the contract |
| **Shotgun surgery** | One change requires edits across many files |
| **Dead code** | Unreachable or never-called code |
| **Magic numbers** | Hardcoded values without named constants |

______________________________________________________________________

## Refactor Heuristics

1. Split by responsibility, not by size
2. Introduce abstraction for a demonstrated domain concept or shared behavior
3. Keep refactors incremental — isolate behavior before moving
4. Preserve behavior using relevant existing checks; add tests when a material behavior lacks coverage
5. Prefer composition over inheritance (AGENTS.md: hierarchy ≤ 2)
6. Choose dicts, dataclasses, protocols, or ABCs according to the actual contract and repository conventions; do not require a representation change solely as a stylistic preference
