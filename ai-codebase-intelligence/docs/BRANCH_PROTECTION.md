# Branch Protection Recommendations

Apply these rules on the main branch:

- Require pull request before merge
- Require at least one approving review
- Require status checks to pass before merge
- Require branches to be up to date before merge
- Dismiss stale approvals when new commits are pushed
- Restrict force pushes and branch deletion

Suggested required checks:

- backend-test
- backend-lint
- frontend-test
- docker-build (for push)
- evaluation-smoke (for pull requests)
- integration-test

Rationale:

These checks ensure code quality, runtime confidence, and reproducible deployment behavior before merges.
