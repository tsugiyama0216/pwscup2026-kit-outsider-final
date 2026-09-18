"""Second fixed batch: preserve tail classes and calibrate categorical counts."""
from research import run
GRID=[dict(steps=250,marginal_weight=m,rare_weight=r,correlation_weight=300.,information_weight=100.,movement_weight=.02,trust_radius=1.,membership_weight=w,preserve_membership=True,balance_prefecture=True)
      for m,r,w in [(4.,4.,10000.),(10.,4.,10000.),(4.,10.,10000.),(4.,10.,100000.)]]
if __name__=='__main__': run(GRID,'constraints')
