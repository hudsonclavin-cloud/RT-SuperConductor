import warnings

from .cli import main

warnings.filterwarnings("ignore", message="No Pauling electronegativity")
raise SystemExit(main())
