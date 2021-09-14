#include <iostream>
#include <vector>
#include <unordered_set>
#include <filesystem>
#include <optional>
#include <ctime>

#include <readers/reader-complete.h>

#include <LinearRegressionDataGenerator.hpp>
#include <TupleWriter.hpp>
#include <FuzzyRegression.hpp>
#include <cstring>


std::optional<std::unordered_set<std::string>> generateArgumentsSet(int argument_count, char *const *argument_variables);
void runMainProgram(const std::unordered_set<std::string>& arguments);
void generateTestData();
void processData();
std::string printTime(const std::tm* timeStruct);
void processSingleFile(const std::filesystem::directory_entry& testDataDirectoryEntry,
                       const std::filesystem::path& resultPath);

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
        processData();
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

void processData(){
    auto dataPath = std::filesystem::path("data");
    auto resultsRootPath = std::filesystem::path("results");

    std::time_t currentTime = std::time(nullptr);
    const std::tm* currentTimeStruct = std::localtime(&currentTime);
    auto processDataFunctionEntryDateTime = printTime(currentTimeStruct);

    auto resultsPath = resultsRootPath / processDataFunctionEntryDateTime;
    if(!std::filesystem::exists(resultsPath)){
        std::filesystem::create_directories(resultsPath);
    }
    auto dataDirectoryIterator = std::filesystem::directory_iterator(dataPath);
    for(auto const& directoryEntry: dataDirectoryIterator){
        if (!directoryEntry.is_directory()){
            processSingleFile(directoryEntry, resultsPath);
        }
    }
}

std::string printTime(const std::tm* timeStruct) {
    std::ostringstream outputStringStream;
    outputStringStream << std::setfill('0')
                       << 1900 + timeStruct->tm_year << "-"
                       << std::setw(2) << 1 + timeStruct->tm_mon << "-"
                       << std::setw(2) << timeStruct->tm_mday
                       << "_"
                       << std::setw(2) << timeStruct->tm_hour << "-"
                       << std::setw(2) << timeStruct->tm_min << "-"
                       << std::setw(2) << timeStruct->tm_sec;
    return outputStringStream.str();
}

void processSingleFile(const std::filesystem::directory_entry& testDataDirectoryEntry,
                       const std::filesystem::path& resultPath) {
    ksi::reader_complete input;
    auto dataset = input.read(testDataDirectoryEntry.path());
    size_t numberOfData = dataset.getNumberOfData();
    for (int clusterSizeIteration = 0; clusterSizeIteration < numberOfData; clusterSizeIteration += 5) {
        auto fuzzyRegression = FuzzyRegression(dataset, 2);
        auto regressionResults = fuzzyRegression.processDataset();
        const std::string filename = testDataDirectoryEntry.path().stem().string() + "_" + std::to_string(clusterSizeIteration) + ".txt";

        std::fstream regressionResultOutputFile((resultPath / filename).string(), std::ios::out);
        if (!regressionResultOutputFile.is_open()){
            std::cout << "Could not create file " << filename << "\n";
            std::cout << strerror(errno) << "\n";
            return;
        }
        regressionResultOutputFile << "Got following results for regression with " << clusterSizeIteration << " clusters\n";
        for (int i = 0; i < regressionResults.size(); ++i) {
            regressionResultOutputFile << "x" << i+1 << " = " << regressionResults[i] << " ";
        }
        regressionResultOutputFile.close();
    }
}
