# Assumptions
Commands are run from arara.proj folder.

# Main results
See adhoc.ps1 (for Windows power shell)

# Local test/debug
python -m projects.ezmed.main --mode e2e --label medi0000-100 
python -m projects.ezmed.main --mode e2e --label medi0000-200 --C 0.4

# Data demo
python -m src.data.__init__ --mode toyexample_demo 
python -m src.data.__init__ --mode toyexample_demo --C 0.4 --D1 0.5 --D2 100

