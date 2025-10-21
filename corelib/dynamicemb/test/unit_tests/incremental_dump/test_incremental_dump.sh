
set -e

pytest test/unit_tests/incremental_dump/test_batched_dynamicemb_tables.py -s
torchrun --nproc_per_node=1 -m pytest test/unit_tests/incremental_dump/test_distributed_dynamicemb.py -s
torchrun --nproc_per_node=4 -m pytest test/unit_tests/incremental_dump/test_distributed_dynamicemb.py -s
pytest test/unit_tests/incremental_dump/test_dynamicemb_extensions.py -s
