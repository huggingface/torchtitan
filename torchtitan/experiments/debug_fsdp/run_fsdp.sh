torchrun --nproc_per_node=2 minimal_fsdp.py --job.config_file configs/hf_transformers_debug.toml

torchrun --nproc_per_node=2 minimal_fsdp.py --job.config_file configs/native_llama3_debug.toml