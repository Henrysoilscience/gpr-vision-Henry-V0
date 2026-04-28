# Annotation Manual v2.0

## Overview
- Annotate subsurface targets as contiguous masks using the target class.
- Label voids or structural layers as background to avoid false alerts.
- Record unclear regions in the review queue for senior confirmation.

## Boundary Criteria
- Draw tight masks around hyperbolic reflections and maintain smooth edges.
- Split overlapping targets into separate instances when separable.
- Ignore noise streaks unless corroborated by multiple adjacent scans.

## Negative Examples
- Moisture streaks without coherent hyperbolas.
- Cable shadows lacking consistent curvature.
- Equipment reflections outside the survey transect.

## Review Workflow
1. Annotator submits a batch of 20 scenes per review cycle.
2. Senior reviewer samples 10% of scenes for spot checks.
3. Corrections are logged in `docs/changelog.md` with brief rationale.

## Version Notes
- v2.0 clarifies negative examples and refines overlap handling.
