add_executable(FuzzyRegression
        src/main.cpp
        src/datageneration/LinearRegressionDataGenerator.cpp
        src/streams/TupleWriter.cpp
        src/regression/FuzzyRegression.cpp)

set_property(TARGET FuzzyRegression PROPERTY CXX_STANDARD 17)
if (MSVC)
    # warning level 4 and all warnings as errors
    add_compile_options(/W4 /WX)
else()
    # lots of warnings and all warnings as errors
    add_compile_options(-Wall -Wextra -pedantic -Werror)
endif()

target_include_directories(FuzzyRegression PRIVATE include)

target_link_libraries(FuzzyRegression PRIVATE NeuroFuzzyLib)

file(COPY data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
