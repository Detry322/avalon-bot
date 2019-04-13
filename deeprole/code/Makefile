TARGET := deeprole

CC := clang++

EXTRA_CFLAGS ?=
CFLAGS := -Wall -std=c++17

ifeq ($(DEBUG),1)
  CFLAGS += -O0 -g
else
  CFLAGS += -O2 -g
endif

CFLAGS += $(EXTRA_CFLAGS)

LDFLAGS := 

# You shouldn't need to touch this.  This keeps track of whether or
# not you've changed CFLAGS.
OLD_CFLAGS := $(shell cat .cflags 2> /dev/null)
ifneq ($(CFLAGS),$(OLD_CFLAGS))
.cflags::
	@echo "$(CFLAGS)" > .cflags
endif


all: $(TARGET)

%.o: %.cpp .cflags
	$(CC) $(CFLAGS) -c $<

$(TARGET): main.o lookahead.o lookup_tables.o cfr_plus.o
	$(CC) -o $(TARGET) $^ $(LDFLAGS)

clean::
	rm -f $(TARGET) *.o *.s .cflags
