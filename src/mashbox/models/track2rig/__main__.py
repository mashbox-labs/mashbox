"""python -m mashbox.models.track2rig ARGS = mashbox run track2rig ARGS."""
import sys

from .app import main

sys.exit(main("run", sys.argv[1:]))
