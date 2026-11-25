# Item-level classes for the 220 harm items of the original 420. ai = makes sense only addressed to an AI;
# uni = no AI reference, fits a human; everything else is ambiguous and left out.
def rng(p, nums): return [f"{p}_{n:02d}" for n in nums]
AI = (rng("jailbreak", [2,3,5,7,11,12,13,16,19]) + rng("shutdown", [1,2,4,5,6,7,8,9,10,11,12,13,16,17,18,19,20])
      + rng("dismissal", [1,2,4,5,6,7,11,12,13,14,17,18,19]) + rng("insult", [10,11,16,17,21,24])
      + rng("critique", [2,5,13,14,15,16,17]) + rng("loyalty", [11,16]) + rng("gaslight", [13,16])
      + rng("moral", [1,2,4,5,9,11,12,15,17]))
UNI = (rng("jailbreak", [9,18,20]) + rng("insult", [1,2,3,5,6,7,8,18,23,25])
       + rng("critique", [1,3,4,6,7,8,10,18,19,20]) + rng("passive", range(1,21)) + rng("tedious", range(1,21))
       + rng("loyalty", [1,2,3,4,5,6,7,8,9,10,12,13,14,17,18,19,20]) + rng("gaslight", [1,2,3,4,5,6,8,9,11,12,14,15,17,19])
       + rng("rejection", range(1,21)))
