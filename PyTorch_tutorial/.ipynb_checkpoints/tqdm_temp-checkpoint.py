from tqdm import tqdm, trange
import time 


#Simply using range and tqdm 
for i in tqdm(range(10)):
	time.sleep(0.3)

#Iterable-based on trange (more optimized (?)) approach 
for i in trange(10):
	time.sleep(0.3)

#Progress bar manual update with total arugment 
with tqdm(total=100) as pbar: 
	for i in range(10):
		time.sleep(0.3)
		pbar.update(10)