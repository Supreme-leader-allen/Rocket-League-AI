1. What research question do you want to answer?
2. How many parameters does the program have? Classify them into a hierarchical tree structure.
3. Based on the research question and the parameters, construct a data table.
4. Put all experiment settings and commands in a single bash script. Expose the scale as a parameter to that script, so the experiment can be launched with a single command, e.g. ROUND=10 ./exp.sh.
5. Generate as many plots as possible.

1. Use requirement.txt to make sure the environment is the same.
2. Use one single bash file to run the experiment, on both local and remote machine.
3. Use git to synchronize remote and local code.
4. Before running the experiment on the remote machine, check locally if the result is reasonable.
5. Check the first few results to make sure the experiment is reasonable.
6. Don't forget to shut down the machine.
