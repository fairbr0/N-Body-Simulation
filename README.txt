cs257.c has been altered to no longer include usleep, which capped the frame limit.
This was done to speed up automated testing.


To run automated testing, in terminal run 'python runHarness.py'. This will run
tests for a range of stars and time steps, which can be altered in the script.
Results are saves to results.txt