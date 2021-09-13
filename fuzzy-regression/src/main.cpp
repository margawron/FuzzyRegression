#include <iostream>
#include <vector>
#include <unordered_set>
#include <filesystem>
#include <optional>

#include <LinearRegressionDataGenerator.hpp>
#include <TupleWriter.hpp>

std::optional<std::unordered_set<std::string>> generateArgumentsSet(int argument_count, char *const *argument_variables);
void runMainProgram(const std::unordered_set<std::string>& arguments);
void generateTestData();
void processTestData();

int main(int argument_count, char** argument_variables){
    std::optional<std::unordered_set<std::string>> argumentsOptional = generateArgumentsSet(argument_count, argument_variables);
    if (!argumentsOptional.has_value()){
        return 0;
    } else {
        auto arguments = argumentsOptional.value();
        runMainProgram(arguments);
    }
    return 0;
}

std::optional<std::unordered_set<std::string>> generateArgumentsSet(int argument_count, char *const *argument_variables) {
    if (argument_count <= 1){
        std::cout << "Not correct mode chosen\n"
                  << "Provide correct for program to run\n"
                  << "To generate test data add -generate argument\n"
                  << "To run FCM algorithm and regression add -process argument\n"
                  << "Modes can be combined\n"
                  << "Data files should be contained in \"data\" folder relative to program location\n";
        return {};
    }
    std::unordered_set<std::string> argumentSet;
    for(int i = 1; i < argument_count; ++i){
        auto argument = argument_variables[i];
        argumentSet.insert(argument);
    }
    return argumentSet;
}

void runMainProgram(const std::unordered_set<std::string>& arguments) {
    if (auto generationArgumentIterator = arguments.find("-generate"); generationArgumentIterator != arguments.end()) {
        generateTestData();
    }
    if (auto processArgumentIterator = arguments.find("-process"); processArgumentIterator != arguments.end()){
        processTestData();
    }
}

void generateTestData() {
    auto generatorForFiveDimensionalData = LinearRegressionDataGenerator(
            std::vector<double>(std::initializer_list<double> {5.0, 3.0, 4.0, 5.0, 3.0}),
            std::vector<double>(std::initializer_list<double> {0.1, 0.3, 0.4, 0.5, 0.3}),
            2.0,
            1000.0);

    auto tupleWriter = TupleWriter(generatorForFiveDimensionalData, 1000);
    tupleWriter.generateTuplesAndSaveWithFilename(
            std::filesystem::path("data/test1.txt")
    );
    tupleWriter.generateTuplesAndSaveWithFilename(
            std::filesystem::path("data/test2.txt")
    );
    tupleWriter.generateTuplesAndSaveWithFilename(
            std::filesystem::path("data/test3.txt")
    );
}

void processTestData(){
    auto dataPath = std::filesystem::path("data");
    auto resultsPath = std::filesystem::path("results");
}

