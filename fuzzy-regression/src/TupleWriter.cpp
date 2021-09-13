#include <TupleWriter.hpp>
#include <LinearRegressionDataGenerator.hpp>

#include <iostream>

TupleWriter::TupleWriter(LinearRegressionDataGenerator& dataGenerator,
                         const unsigned int numberOfValuesToGenerate) noexcept
                         : dataGenerator(dataGenerator),
                         numberOfValuesToGenerate(numberOfValuesToGenerate){}

void TupleWriter::generateTuplesAndSaveWithFilename(const std::filesystem::path&& outputPath) {
    std::ofstream outputFile(outputPath);
    if (!outputFile.is_open()){
        std::cout << "File was not opened correctly";
        throw "File was not opened correctly";
    }
    for (int i = 0; i < numberOfValuesToGenerate; ++i) {
        auto tuple = dataGenerator.generateTupleValue();
        for (const auto &item : tuple){
            outputFile << item << " ";
        }
        outputFile << "\n";
    }
    outputFile.close();
}
