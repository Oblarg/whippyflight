CXX = g++
CXXFLAGS = -std=c++20 -Wall -Wextra -O2 -I. -Iinclude
EIGEN_INCLUDE = -I/opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3

# Source files
SRC_DIR = src
INCLUDE_DIR = include
TEST_FILE = test_lqr_solver.cpp

# Object files
OBJ_FILES = $(SRC_DIR)/gain_generation.o

# Target executable
TARGET = test_lqr_solver

# Default target
all: $(TARGET)

# Compile the main test executable
$(TARGET): $(TEST_FILE) $(OBJ_FILES)
	$(CXX) $(CXXFLAGS) $(EIGEN_INCLUDE) -o $@ $^

# Compile source files
$(SRC_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(EIGEN_INCLUDE) -c -o $@ $<

# Clean build artifacts
clean:
	rm -f $(TARGET) $(OBJ_FILES)

# Run the test
test: $(TARGET)
	./$(TARGET)

.PHONY: all clean test 