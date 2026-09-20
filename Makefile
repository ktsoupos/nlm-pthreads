CC      := cc
CFLAGS  := -O3 -march=native -mavx2 -mfma -Wall -Wextra -Iinclude -Iexternal/stb
LDFLAGS := -lpthread -lm

SRC_DIR   := src
BUILD_DIR := build
TARGET    := nlm

# main.c currently lives at repo root; src/*.c will join it as variants land.
SRCS := main.c $(wildcard $(SRC_DIR)/*.c)
OBJS := $(addprefix $(BUILD_DIR)/,$(SRCS:.c=.o))

.PHONY: all cluster clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

# No -march=native: the login node's CPU may differ from the compute nodes',
# so a native build here can crash (illegal instruction) or silently
# under-optimize there.
cluster: CFLAGS := -O3 -mavx2 -mfma -Wall -Wextra -Iinclude -Iexternal/stb
cluster: clean $(TARGET)

clean:
	rm -rf $(BUILD_DIR) $(TARGET)
