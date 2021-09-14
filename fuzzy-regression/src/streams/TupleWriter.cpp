#include <streams/TupleWriter.hpp>
#include <datageneration/LinearRegressionDataGenerator.hpp>

#include <iostream>

TupleWriter::TupleWriter(LinearRegressionDataGenerator& dataGenerator,
                         const unsigned int numberOfTuplesToGenerate) noexcept
                         : dataGenerator(dataGenerator),
                           numberOfTuplesToGenerate(numberOfTuplesToGenerate){}

void TupleWriter::generateTuplesAndSaveWithFilename(const std::filesystem::path&& outputPath) {
    std::ofstream outputFile(outputPath);
    if (!outputFile.is_open()){
        std::cout << "File was not opened correctly";
        throw "File was not opened correctly";
    }
    for (int i = 0; i < numberOfTuplesToGenerate; ++i) {
        auto tuple = dataGenerator.generateTupleValue();
        for (const auto &item : tuple){
            outputFile << item << " ";
        }
        outputFile << "\n";
    }
    outputFile.close();
}
