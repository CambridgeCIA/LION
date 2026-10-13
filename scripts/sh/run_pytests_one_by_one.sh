# `-vv` prints each test name. Remove if you want to reduce the verbosity of the output.
# `-n 0` disables xdist and runs the tests in the main process (no multi-processing). Remove if you want to enable multi-processing for faster test execution.

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
# export CUDA_VISIBLE_DEVICES="0"
export CUDA_VISIBLE_DEVICES="-1"

# Run only tests that do not require CUDA.
pytest pytests/ -m "not cuda and not tomosipo" -vv -n 0

# Run only tests that require CUDA.
pytest pytests/ -m "cuda" -vv -n 0
