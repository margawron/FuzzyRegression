#include <cstring>

#include "streams/TupleWriter.hpp"
#include "readers/reader-complete.h"
#include "regression/FuzzyRegression.hpp"
#include "helper/Program.hpp"
#include "datageneration/LinearRegressionDataGenerator.hpp"

Program::Program(int argument_count, char** argument_values)
: arguments(generateArgumentSet(argument_count, argument_values)),
MAX_PRECISION_COUT(std::setprecision(std::numeric_limits<long double>::digits10 + 1)){}

int Program::run() {
    if (!arguments.empty()){
        runMainProgram();
    }
    return 0;
}

std::unordered_set<std::string> Program::generateArgumentSet(int argumentCount, char** argumentValues) {
    if (argumentCount <= 1){
        std::cout << "Not correct mode chosen\n"
                  << "Provide correct for program to run\n"
                  << "To generate test data add -generate argument\n"
                  << "To run FCM algorithm and regression add -process argument\n"
                  << "Modes can be combined\n"
                  << "Data files should be contained in \"data\" folder relative to program location\n";
        return {};
    }
    std::unordered_set<std::string> argumentSet;
    for(int i = 1; i < argumentCount; ++i){
        auto argument = argumentValues[i];
        argumentSet.insert(argument);
    }
    return argumentSet;
}

int Program::runMainProgram() {
    if (auto generationArgumentIterator = arguments.find("-generate"); generationArgumentIterator != arguments.end()) {
        generateTestData();
    }
    if (auto processArgumentIterator = arguments.find("-process"); processArgumentIterator != arguments.end()){
        processData();
    }
    return 0;
}

void Program::generateTestData() {
    std::filesystem::path dataPath("data");
    if (!std::filesystem::exists(dataPath)) {
        std::filesystem::create_directories(dataPath);
    }
    {
        auto generatorForFiveDescriptionVariables = LinearRegressionDataGenerator(
                std::vector<double>(std::initializer_list<double>{5.0, 3.0, 4.0, 5.0, 3.0}),
                std::vector<double>(std::initializer_list<double>{0.001, 0.003, 0.004, 0.005, 0.003}),
                2.0,
                1000.0);

        auto tupleWriter = TupleWriter(generatorForFiveDescriptionVariables, 1000);
        tupleWriter.generateTuplesAndSaveWithFilename(
                dataPath / "fiveDescriptionVariables.txt"
        );
    }
    {
        auto generatorForTenDescriptionVariables = LinearRegressionDataGenerator(
                std::vector<double>(std::initializer_list<double>{5.0, -23.0, 4.0, -8.0, 3.0, 53.1, 0.2, -1337.3, 90.0, 123.0}),
                std::vector<double>(std::initializer_list<double>{0.001, 0.003, 0.004, 0.005, 0.003, 0.001, 0.001, 0.001, 0.001, 0.001}),
                2.0,
                1000.0);

        auto tupleWriter = TupleWriter(generatorForTenDescriptionVariables, 1000);
        tupleWriter.generateTuplesAndSaveWithFilename(
                dataPath/"tenDescriptionVariables.txt"
        );
    }
    {
        auto generatorForTwoDescriptionVariables = LinearRegressionDataGenerator(
                std::vector<double>(std::initializer_list<double>{0.2, -1337.3}),
                std::vector<double>(std::initializer_list<double>{0.001, 0.003}),
                2.0,
                1000.0);

        auto tupleWriter = TupleWriter(generatorForTwoDescriptionVariables, 400);
        tupleWriter.generateTuplesAndSaveWithFilename(
                dataPath/"twoDescriptionVariables.txt"
        );
    }
    {
        auto generatorForFourDescriptionVariables = LinearRegressionDataGenerator(
                std::vector<double>(std::initializer_list<double>{3.0, 53.1, 90.0, 123.0}),
                std::vector<double>(std::initializer_list<double>{0.001, 0.003, 0.004, 0.005}),
                2.0,
                1000.0);

        auto tupleWriter = TupleWriter(generatorForFourDescriptionVariables, 300);
        tupleWriter.generateTuplesAndSaveWithFilename(
                dataPath/"fourDescriptionVariables.txt"
        );
    }

}

void Program::processData() {
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

std::string Program::printTime(const std::tm* timeStruct) {
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

void Program::processSingleFile(const std::filesystem::directory_entry& testDataDirectoryEntry,
                                const std::filesystem::path& resultPath) {
    const std::string& testFileStem = testDataDirectoryEntry.path().stem().string();
    const std::string performanceFileName = testFileStem + "_perf.txt";
    std::fstream performanceFile(resultPath / performanceFileName, std::ios::out);
    if (!performanceFile.is_open()){
        std::cout << "Could not create performance file for " << testDataDirectoryEntry.path().string() << "\n";
        std::cout << "Aborting processing of this file\n";
        std::cout << strerror(errno) << "\n";
        return;
    }
    printPerformanceFileHeader(performanceFile);
    ksi::reader_complete input;
    auto dataset = input.read(testDataDirectoryEntry.path());
    // for 1000 datums with 5 describing values this took processing from 1 to about 70 clusters with 5 increment
    // took about 1-2 minutes, the bottleneck is FCM algorithm.doPartition(dataset)
    const int MAX_CLUSTERS_AMOUNT = 50;

    const std::string filename = testFileStem + "_results.txt";
    std::fstream regressionResultOutputFile((resultPath / filename).string(), std::ios::out);
    if (!regressionResultOutputFile.is_open()){
        std::cout << "Could not create file " << filename << "\n";
        std::cout << strerror(errno) << "\n";
        return;
    }
    printRegressionDataHeader(regressionResultOutputFile, dataset.getNumberOfAttributes());

    unsigned long maxNumberOfClusters = dataset.getNumberOfData() < MAX_CLUSTERS_AMOUNT ? dataset.getNumberOfData() : MAX_CLUSTERS_AMOUNT;
    for (int clusterSizeForIteration = 2; clusterSizeForIteration < maxNumberOfClusters; ++clusterSizeForIteration) {

        auto fuzzyRegression = FuzzyRegression(dataset, clusterSizeForIteration);
        auto regressionResults = fuzzyRegression.processDataset(performanceFile);

        printRegressionData(regressionResultOutputFile, clusterSizeForIteration, regressionResults);
    }
    regressionResultOutputFile.close();
    performanceFile.close();
}

void Program::prettyPrintRegressionData(std::fstream& outputFile,
                                        int clusterSizeForIteration,
                                        const RegressionResult& regressionResults) const {
    outputFile << "Got following results for regression with " << clusterSizeForIteration << " clusters\n";
    auto regressionCoefficients = regressionResults.regressionDescribingParameters;
    for (int i = 0; i < regressionCoefficients.size(); ++i) {
        outputFile << "x" << i + 1 << " = " << MAX_PRECISION_COUT << regressionCoefficients[i] << " ";
    }
    outputFile << "\n";
    outputFile << "R-squared error: " << MAX_PRECISION_COUT << regressionResults.coefficientOfDetermination;
}

void Program::printRegressionDataHeader(std::fstream& outputFile, unsigned long numberOfAttributes) const {
    outputFile << "Cluster size" << ";";
    for (int i = 0; i < numberOfAttributes-1; ++i) {
        outputFile << "X" << i+1 << ";";
    }
    outputFile << "Regression Error" << "\n";
}

void Program::printRegressionData(std::fstream& outputFile,
                                  int clusterSizeForIteration,
                                  const RegressionResult& regressionResults) const {
    outputFile << clusterSizeForIteration << ";";
    auto regressionCoefficients = regressionResults.regressionDescribingParameters;
    for (double regressionCoefficient : regressionCoefficients) {
        outputFile << MAX_PRECISION_COUT << regressionCoefficient << ";";
    }
    outputFile << MAX_PRECISION_COUT << regressionResults.coefficientOfDetermination << std::endl;
}

void Program::printPerformanceFileHeader(std::fstream& performanceFile) {
    performanceFile << "Number of Data;" << "Number of attributes;" << "Number of clusters;";
    performanceFile << "FCM duration;" << "Data preparation duration;" << "Regression duration;" << "Error calculation duration" << "\n";

}
