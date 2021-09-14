#ifndef FUZZY_REGRESSION_PROGRAM_HPP
#define FUZZY_REGRESSION_PROGRAM_HPP

#include <iostream>
#include <filesystem>
#include <string>
#include <vector>
#include <unordered_set>

class Program {
private:
    std::unordered_set<std::string> arguments;
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
};

#endif