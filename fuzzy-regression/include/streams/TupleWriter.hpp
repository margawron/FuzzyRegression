#ifndef FUZZY_REGRESSION_TUPLEWRITER_HPP
#define FUZZY_REGRESSION_TUPLEWRITER_HPP

#include "datageneration/LinearRegressionDataGenerator.hpp"

#include <fstream>
#include <filesystem>

class TupleWriter{
private:
    LinearRegressionDataGenerator& dataGenerator;
    const unsigned int numberOfTuplesToGenerate;
public:
    TupleWriter(LinearRegressionDataGenerator& dataGenerator,
                unsigned int numberOfTuplesToGenerate) noexcept;

    void generateTuplesAndSaveWithFilename(const std::filesystem::path&& outputPath);
};


#endif // FUZZY_REGRESSION_TUPLEWRITER_HPP