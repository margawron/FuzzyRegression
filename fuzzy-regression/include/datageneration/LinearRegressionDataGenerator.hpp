#ifndef FUZZY_REGRESSION_LINEARREGRESSIONDATAGENERATOR_HPP
#define FUZZY_REGRESSION_LINEARREGRESSIONDATAGENERATOR_HPP

#include <vector>
#include <random>

class LinearRegressionDataGenerator {
private:
    std::mt19937 mersenneTwisterGenerator;
    const std::vector<std::uniform_real_distribution<double>> uniformRealDistributionsForDeviation;
    std::uniform_real_distribution<double> uniformRealDistributionForIntercept;
    std::uniform_real_distribution<double> uniformRealDistributionForDomain;

    const std::vector<double> slopeParameters;

    static std::vector<std::uniform_real_distribution<double>>
    generateDistributionsForSlopeDeviations(const std::vector<double> &slopeDeviationsVector);
public:
    LinearRegressionDataGenerator(const std::vector<double>&& slopeParameters,
                                  const std::vector<double>&& slopeDeviations,
                                  double interceptDeviation,
                                  double domainMaxValue);

    std::vector<double> generateTupleValue();
};


#endif //FUZZY_REGRESSION_LINEARREGRESSIONDATAGENERATOR_HPP
