"""Third batch: freeze improved synthetic tail while correcting majority Cox."""
from research import run
GRID=[dict(phase=.75,strength=s,steps=250,joint=False,rare_weight=10.,lock_tail=True) for s in [.5,1.]]
if __name__=='__main__': run(GRID,'locked')
