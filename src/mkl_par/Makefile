# tool marcros
CC := icpc # FILL: the compiler
#CCFLAG := -g -mkl=sequential #-Dprecond -Dprofile# FILL: compile flags
CCFLAG := -O3 -mkl=parallel -lufget  
# CC := g++ # FILL: the compiler
# CCFLAG :=  -Wall -m64 -I${MKLROOT}/include # FILL: compile flags
DBGFLAG := -g
CCOBJFLAG := $(CCFLAG) -c

# path marcros
BIN_PATH := ./bin/
OBJ_PATH := ./obj/
SRC_PATH := ./
DBG_PATH := debug
RESULT_PATH := ./results/
LINKER_LINE :=  -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -liomp5 -lpthread -lm -ldl -lufget

# compile marcros
TARGET_NAME := MKL_par # FILL: target name
ifeq ($(OS),Windows_NT)
	TARGET_NAME := $(addsuffix .exe,$(TARGET_NAME))
endif
TARGET := $(BIN_PATH)/$(TARGET_NAME)
TARGET_DEBUG := $(DBG_PATH)/$(TARGET_NAME)
TARGET_GNU := $(BIN_PATH)/$(TARGET_NAME)
MAIN_SRC := main.cpp # FILL: src file contains `main()`

# src files & obj files
SRC := $(foreach x, $(SRC_PATH), $(wildcard $(addprefix $(x)/*,.c*)))
OBJ := $(addprefix $(OBJ_PATH)/, $(addsuffix .o, $(notdir $(basename $(SRC)))))
OBJ_DEBUG := $(addprefix $(DBG_PATH)/, $(addsuffix .o, $(notdir $(basename $(SRC)))))

# clean files list
DISTCLEAN_LIST := $(OBJ) \
                  $(OBJ_DEBUG)
CLEAN_LIST := $(TARGET) \
			  $(TARGET_DEBUG) \
			  $(DISTCLEAN_LIST)

# default rule
default: all

# non-phony targets
$(TARGET): $(OBJ)
	@mkdir -p $(RESULT_PATH)
	@mkdir -p $(@D)
	$(CC) $(CCFLAG) -o $@ $? 

# $(TARGET): $(OBJ)
# 	$(CC) $(CCFLAGS) -o $@ $?  $(LINKER_LINE)


$(OBJ_PATH)/%.o: $(SRC_PATH)/%.c*
	@mkdir -p $(@D)
	$(CC) $(CCOBJFLAG) -o $@ $<

$(DBG_PATH)/%.o: $(SRC_PATH)/%.c*
	$(CC) $(CCOBJFLAG) $(DBGFLAG) -o $@ $<

$(TARGET_DEBUG): $(OBJ_DEBUG)
	$(CC) $(CCFLAG) $(DBGFLAG) $? -o $@

# phony rules
.PHONY: all
all: $(TARGET)

.PHONY: debug
debug: $(TARGET_DEBUG)

.PHONY: gnu
gnu: $(TARGET_GNU)

.PHONY: clean
clean:
	@echo CLEAN $(CLEAN_LIST)
	@rm -f $(CLEAN_LIST)

.PHONY: distclean
distclean:
	@echo CLEAN $(CLEAN_LIST)
	@rm -f $(DISTCLEAN_LIST)
