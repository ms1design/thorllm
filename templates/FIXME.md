- CLI does not have the autocomplete with TAB
- add CLI `thorllm model select` with a interactive selection
- we need to add CLI arg to set port: `thorllm start --port 666` or `thorllm start -p 666`
- when we start CLI `thorllm start` we should immediatelly start following logs until the `vllm` loads (we can check the `/v1/health` endpoint), then we can exit with the summarisation.
- Can we make colour more appealing? Maybe instead of the pink, lets choose the NVIDIA Green ;)
- Can we also fetch latest pytorch version and have the same version selector as for vllm?
- We need to have a small footer in the bottom middle of terminal with: project name / version / made by ms1design
- on the Configuration summary we need display all options selected by the user, include also vllm version, pytroch version, etc..
- we need to add more verbosity if we ask user to enter the sudo password, add a short explanation why we ask for it
- we need to add the `thorllm kill` command
- We need the `thorllm` ASCII logo on top middle part (not too big):
```
▗ ▌     ▜ ▜    
▜▘▛▌▛▌▛▘▐ ▐ ▛▛▌
▐▖▌▌▙▌▌ ▐▖▐▖▌▌▌
               
```

- after we installed vllm we are on the Installation complete screen, where the user is not completelly sure what to do next, be more helpful here and add short explanation what tu run for what effect:
```
── Installation complete 

  Activate:    source /home/narandill/thorllm/activate_vllm.sh
  Start:       thorllm start
  Watch logs:  thorllm logs -f
```