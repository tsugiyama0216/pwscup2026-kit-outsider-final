# M1-v3 local DOMIAS defense

M1-v3 starts from M1-v2 and changes only 5 C rows and only BMI/TG/ALT.
The defense was motivated by a DOMIAS diagnosis in which rare member record_id 478
was ranked first among rare candidates in one pseudo-B_j trial and dominated beta_raw.
The 5 closest C rows to that member (standardized age/BMI/SBP/TG/HDL/ALT/FPG space)
are moved 75% toward a 50-neighbor local A_bg centroid for BMI/TG/ALT.

Proxy results under the same 20 pseudo attackers:
- U: 0.84264 (M1-v2: ~0.84278)
- DOMIAS beta_raw: 0.03462
- chance95: 0.05962
- beta: 0.01
- a_MIA(K=12): 0
- AIA worst before phi: 0.07510
- Anon proxy: 0.92490
- Objective proxy: 0.84264

Caveat: the large seed-20260924 M1-v2 DOMIAS score was driven by a single rare member;
independent pseudo-B_j seeds were already much lower for M1-v2. Thus this is a targeted
stress-test hardening, not proof of improved real-round anonymity.
