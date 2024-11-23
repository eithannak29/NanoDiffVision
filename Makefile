MNIST_CONFIG_DIR := configs/MNIST
FASHIONMNIST_CONFIG_DIR := configs/FashionMNIST
MNIST_CONFIG_FILES := $(shell find $(MNIST_CONFIG_DIR) -type f -name '*.yml')
FASHIONMNIST_CONFIG_FILES := $(shell find $(FASHIONMNIST_CONFIG_DIR) -type f -name '*.yml')
MAIN_SCRIPT := src/main.py

define RUN_COMMAND
	uv run $(MAIN_SCRIPT) --config=$(1)
endef

.PHONY: all mnist specific_config

all: mnist fashionmnist

mnist:
	@echo "Running all MNIST configurations:"
	@for file in $(MNIST_CONFIG_FILES); do \
		echo "Running configuration: $$file"; \
		$(call RUN_COMMAND,$$file); \
		echo "Finished configuration: $$file"; \
	done

fashionmnist:
	@echo "Running all FashionMNIST configurations:"
	@for file in $(FASHIONMNIST_CONFIG_FILES); do \
		echo "Running configuration: $$file"; \
		$(call RUN_COMMAND,$$file); \
		echo "Finished configuration: $$file"; \
	done

config:
	@if [ -z "$(CONFIG_FILE)" ]; then \
		echo "Error: Please CONFIG_FILE, e.g., 'make config CONFIG_FILE=configs/MNIST/lite_config.yml'"; \
		exit 1; \
	fi
	@echo "Running configuration: $(CONFIG_FILE)"
	$(call RUN_COMMAND,$(CONFIG_FILE))
	@echo "Finished configuration: $(CONFIG_FILE)"
