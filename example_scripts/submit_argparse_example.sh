#!/bin/bash

cd ./GPLVM_Shaista/example_scripts

script_call='python ./example_scripts/argparse_example.py --help'
team_name=team205

echo $script_call | \
    bsub -G $team_name \
    -o bsub_run.out \ ## file saving job standard output
    -e bsub_run.err \ ## file saving job standard error
    -R"select[mem>3500] rusage[mem=3500]" -M3500 \ ## memory requirements