Instructions: 
When you recieve a peice of instruction, note that the instruction might have a defined data for the crypto. however, under /mnt/timemachine/binance we have our ocean of data. So stick to those data. The instructions the user copy pasted came from somewhere general so the data location(and only the data location) you should use our own method of the big drive i mentioned. For other artifacts, you should follow the plan that the user provided. If user said that you should do something that conflicts this AGENTS.md here, explicity ask the user what you'd follow. 

Every strategy result MUST be checked by generating synthetic Monte Carlo OCHLV data in the granularities, run the exact same pipeline on it, and if the random data have also good results like the real one, it means there's feature leak, which you must check and amend. 

When you check and amend a feature\method to see feature leak, you MUST not just strip the method\feature off or disable it. you must find the root cause, fix it, and rerun, and so on. So that we will actually have something useful rather than nerfed artifacts. 

When running batched experiments, try to fully utilize system power(parrallel analysis where posssilbe). if the experiment itself cannot be paralleled, then don't bother.
the system have i9 14900k, 64g ram, 4090 with 24g vram. 


