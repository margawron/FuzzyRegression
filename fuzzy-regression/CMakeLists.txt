add_executable(FuzzyRegression src/main.cpp
        src/LinearRegressionDataGenerator.cpp src/TupleWriter.cpp include/TupleWriter.hpp)

target_include_directories(FuzzyRegression PRIVATE include)

target_link_libraries(FuzzyRegression PRIVATE NeuroFuzzyLib)

file(COPY data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
