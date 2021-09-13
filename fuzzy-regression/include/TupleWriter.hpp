#ifndef FUZZY_REGRESSION_TUPLEWRITER_HPP
#define FUZZY_REGRESSION_TUPLEWRITER_HPP

#include "LinearRegressionDataGenerator.hpp"

#include <fstream>

class TupleWriter{
private:
    LinearRegressionDataGenerator& dataGenerator;
    const unsigned int numberOfValuesToGenerate;
public:
    TupleWriter(LinearRegressionDataGenerator& dataGenerator,
                unsigned int numberOfValuesToGenerate) noexcept;

    void generateTuplesAndSaveWithFilename(const std::string&& outputFilename);
};


#endif // FUZZY_REGRESSION_TUPLEWRITER_HPP