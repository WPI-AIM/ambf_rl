if [ -f .logs/training_out.txt ]; then
  tail -f .logs/training_out.txt
else
  echo "training_out.txt has not been generated yet, trying again in some time."
fi
