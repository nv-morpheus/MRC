## :snowflake: Code freeze for `branch-${CURRENT_VERSION}` and `v${CURRENT_VERSION}` release

### What does this mean?
Only critical/hotfix level issues should be merged into `branch-${CURRENT_VERSION}` until release (merging of this PR).

All other development PRs should be retargeted towards the next release branch: `branch-${NEXT_VERSION}`.

### What is the purpose of this PR?
- Update documentation
- Allow testing for the new release
- Enable a means to merge `branch-${CURRENT_VERSION}` into `main` for the release
