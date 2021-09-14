#ifndef FUZZY_REGRESSION_PROGRAM_HPP
#define FUZZY_REGRESSION_PROGRAM_HPP

#include <iostream>
#include <filesystem>
#include <string>
#include <vector>
#include <unordered_set>

#include "regression/FuzzyRegression.hpp"

class Program {
private:
    std::unordered_set<std::string> arguments;
    const std::_Setprecision MAX_PRECISION_COUT;
public:
    Program(int argument_count, char ** argument_values);
    int run();
private:
    std::unordered_set<std::string> generateArgumentSet(int argumentCount, char ** argumentValues);
    int runMainProgram();
    void generateTestData();
    void processData();
    std::string printTime(const std::tm* timeStruct);
    void processSingleFile(const std::filesystem::directory_entry& testDataDirectoryEntry,
                           const std::filesystem::path& resultPath);

    void prettyPrintRegressionData(std::fstream& outputFile,
                                   int clusterSizeForIteration,
                                   const RegressionResult& regressionResults) const;
    void printRegressionData(std::fstream& outputFile,
                             int clusterSizeForIteration,
                             const RegressionResult& regressionResults) const;

    void printRegressionDataHeader(std::fstream& outputFile, unsigned long numberOfAttributes) const;

    void printPerformanceFileHeader(std::fstream& performanceFile);
};

#endif