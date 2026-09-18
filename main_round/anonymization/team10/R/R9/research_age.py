"""Third fixed batch: conditional age synthesis to close the remaining tail gap."""
from research import run
GRID=[dict(steps=300,marginal_weight=m,rare_weight=r,correlation_weight=300.,information_weight=100.,movement_weight=.02,trust_radius=1.,membership_weight=w,preserve_membership=True,balance_prefecture=True,balance_age=True)
      for m,r,w in [(4.,4.,10000.),(10.,4.,10000.),(4.,10.,10000.),(4.,10.,100000.)]]
if __name__=='__main__': run(GRID,'age')
