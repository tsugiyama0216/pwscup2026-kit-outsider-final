"""Second predeclared batch: preserve Cox while changing tail quantile phase."""
from research import run
GRID=[dict(phase=p,strength=1.,steps=250,joint=True,rare_weight=10.) for p in [0.,.25,.75,1.]]
if __name__=='__main__': run(GRID,'joint')
