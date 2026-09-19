# Design Review Checklist (Relax Project)

## Development Principles

| Principle | What to investigate | Boundary |
|-----------|---------------------|----------|
| Simple and correct | Branches, wrappers, configuration, or dependencies that add no current behavior or clarity | Prefer the smallest clear design that satisfies the requirement, not the fewest lines |
| Handle realistic failures | Fallbacks for states excluded by established internal contracts, or defaults that conceal broken invariants | Validate external input at boundaries; preserve justified timeout, retry, and cleanup behavior for real I/O and worker failures |
| Cohesive functions and interfaces | Mixed responsibilities, unclear inputs/outputs, or interfaces that force unsupported operations | Split by responsibility; function length alone is not a finding |
| Preserve caller contracts | A supported implementation changes promised inputs, outputs, errors, or side effects | Check actual callers and approved behavior changes; a representation or inheritance choice alone is not a defect |
| Prefer pure functions and explicit ownership | Hidden mutation of caller-owned data, implicit I/O or dependencies, or shared state with unclear ownership | Keep side effects explicit; owned in-place tensor operations can be appropriate when contracts, autograd, and performance justify them |
| Refactor around real concepts or existing commonality | Repeated policy branches drifting apart, patch layers, or generic frameworks for hypothetical future needs | Refactor within the required scope; a second use or a count of duplicates is not an automatic mandate for abstraction |
| One authoritative source per fact | Derived values stored and updated independently, with inconsistent update or invalidation paths | Caches and snapshots need an explicit owner, lifetime, and consistency contract; justified caching is not inherently duplication |

### Calibration Examples

- **Report:** two independently updated fields encode the same rollout state, and a cancellation path updates only one. **Do not report:** a snapshot deliberately captures state at a documented version boundary.
- **Report:** a fallback converts a violated batch contract into apparently valid training data. **Do not report:** boundary validation rejects malformed external input, or a bounded retry handles a transient worker failure.
- **Report:** repeated backend policy branches already disagree for a supported mode. **Do not report:** a short dispatch over a closed set of modes, or a cohesive function solely because it exceeds a line count.
