CIFAR10_CONFIG_DIR := configs/CIFAR10
CIFAR10_CONFIG_FILES := $(shell find $(CIFAR10_CONFIG_DIR) -type f -name '*.yml')
MAIN_SCRIPT := src/main.py
COMMAND = uv run $(MAIN_SCRIPT) --config=$(1)

all: print_configs run_configs

print_configs:
	@echo "Configuration files found in $(CIFAR10_CONFIG_DIR):"
	@for file in $(CIFAR10_CONFIG_FILES); do \
		echo $$file; \
	done

run_configs:
	@for file in $(CIFAR10_CONFIG_FILES); do \
		echo "Running configuration: $$file"; \
		$(call COMMAND,$$file); \
		echo "Finished configuration: $$file"; \
	done

.PHONY: all print_configs run_configs
