#include "LinearRegressionDataGenerator.hpp"

LinearRegressionDataGenerator::LinearRegressionDataGenerator(const std::vector<double> &&slopeParameters,
                                                             const std::vector<double> &&slopeDeviations,
                                                             double interceptDeviation,
                                                             double domainMaxValue)
        : slopeParameters(slopeParameters),
          mersenneTwisterGenerator(std::mt19937(std::random_device()())),
          uniformRealDistributionForIntercept(std::uniform_real_distribution<double>(-interceptDeviation, interceptDeviation)),
          uniformRealDistributionsForDeviation(generateDistributionsForSlopeDeviations(slopeDeviations)),
          uniformRealDistributionForDomain(std::uniform_real_distribution<double>(0, domainMaxValue)){
    if(slopeParameters.size() != slopeDeviations.size()){
        throw "Incompatible number of parameters and deviations";
    }
}

std::vector<std::uniform_real_distribution<double>>
LinearRegressionDataGenerator::generateDistributionsForSlopeDeviations(const std::vector<double>& slopeDeviationsVector){
    std::vector<std::uniform_real_distribution<double>> distributions(slopeDeviationsVector.size());
    for (const auto &deviation : slopeDeviationsVector){
        distributions.emplace_back(-deviation, deviation);
    }
    return distributions;
}

std::vector<double> LinearRegressionDataGenerator::generateTupleValue() {
    unsigned long describingValueAmount = slopeParameters.size();
    std::vector<double> describingAndDescribedParameters(describingValueAmount + 1);
    double describedValue = 0;
    for (int i = 0; i < describingValueAmount; ++i) {
        double x = uniformRealDistributionForDomain(mersenneTwisterGenerator);
        auto distribution = uniformRealDistributionsForDeviation[i];
        double ithDescribingParameterValue = (slopeParameters[i] * x) + distribution(mersenneTwisterGenerator);
        describedValue += ithDescribingParameterValue;
        describingAndDescribedParameters[i] = x;
    }
    describedValue += uniformRealDistributionForIntercept(mersenneTwisterGenerator);
    describingAndDescribedParameters[describingValueAmount] = describedValue;
    return describingAndDescribedParameters;
}
