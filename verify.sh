#!/bin/bash


# Set the number of iterations
gpu='false'

while getopts 'g' flag; do
  case "${flag}" in
    g) gpu='true' ;;
  esac
done


echo GPU: $gpu
num_iterations=10
num_user_wins=0
# Loop through the numbers 1 through num_iterations
for i in $(seq 1 $num_iterations); do
    echo iteration $i
    if [[ "$gpu" == "true" ]]; then
        # job_id=$(sbatch run.sh | grep -e "[0-9]*" -o)$
        # file_name=slurm-$job_id.out
        sbatch --wait run.sh
    else
        ./mcts > out
    fi
    win=$(grep -o "WIN" out | wc -w)
    echo Random challenger win: $win
    num_user_wins=$((num_user_wins+win))
done

echo random challenger total win: $num_user_wins