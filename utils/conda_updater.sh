echo "\n==================== post-merge hook ===================="

changed_files="$(git diff-tree -r --name-only --no-commit-id ORIG_HEAD HEAD)"

check_run() {
    echo "$changed_files" | grep --quiet "$1" && eval "$2"
}

echo "Have to update the conda environment"
check_run env.yml "echo 'conda enviroment changed'"
check_run env.yml "exit 1"