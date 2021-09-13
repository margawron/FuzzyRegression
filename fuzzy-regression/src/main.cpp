#include <vector>

#include <LinearRegressionDataGenerator.hpp>
#include <TupleWriter.hpp>

int main(int argument_count, char** argument_variables){
    auto generator = LinearRegressionDataGenerator(
            std::vector<double>(std::initializer_list<double> {5.0, 3.0, 4.0, 5.0, 3.0}),
            std::vector<double>(std::initializer_list<double> {0.1, 0.3, 0.4, 0.5, 0.3}),
            2.0,
            1000.0);

    auto tupleWriter = TupleWriter(generator,1000);
    tupleWriter.generateTuplesAndSaveWithFilename("test1.txt");
    tupleWriter.generateTuplesAndSaveWithFilename("test2.txt");
    tupleWriter.generateTuplesAndSaveWithFilename("test3.txt");

    return 0;
}