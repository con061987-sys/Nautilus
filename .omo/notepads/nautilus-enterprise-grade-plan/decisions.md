# Decisions

## Wave 0 Execution Strategy
- 0.1, 0.2, 0.3: parallel (independent fixes)
- 0.4, 0.5, 0.6, 0.7: parallel (independent bug fixes)
- 0.8: depends on 0.5 (key fix needed before submodule build)

## Fix Approach
- setup.py: add `setup()` call with `cmdclass=`
- Move imports inside function bodies for testability
- Remove --ignore patterns; add @pytest.mark.requires_deps and conftest auto-skip
