import os, sys
from os.path import join as opj
workspace = os.environ["WORKSPACE"]
sys.path.append(
    opj(workspace, 'code/GeDML/src')
)
import gedml

# debug code