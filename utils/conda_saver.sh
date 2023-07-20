
echo "\n==================== pre-push hook ===================="

# Export conda environment to yaml file
$MAMBA_EXE  env export -n aitools > env.yml || $CONDA_EXE  env export -n aitools > env.yml

# Check if new environment file is different from original 
git diff --exit-code --quiet env.yml 

# If new environment file is different, commit it
if [[ $? -eq 0 ]]; then
    echo "Conda environment not changed. No additional commit."
else
    echo "Conda environment changed. Commiting new env.yml"
    git add env.yml
    git commit -m "Updating conda environment"
    echo 'You need to push again to push additional "Updating conda environment" commit.'
    exit 1
fi