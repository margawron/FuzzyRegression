#include "LinearRegressionDataGenerator.hpp"

LinearRegressionDataGenerator::LinearRegressionDataGenerator(const std::vector<double> &&slopeParameters,
                                                             const std::vector<double> &&slopeDeviations,
                                                             double interceptDeviation,
                                                             double domainMaxValue)
        : slopeParameters(slopeParameters),
          mersenneTwisterGenerator(std::mt19937()),
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
    std::vector<double> returnValue(slopeParameters.size());
    double domainValue = uniformRealDistributionForDomain(mersenneTwisterGenerator);
    for(int i = 0; i < slopeParameters.size(); ++i){
        auto distribution = uniformRealDistributionsForDeviation[i];
        returnValue[i] = (slopeParameters[i] * domainValue) + distribution(mersenneTwisterGenerator);
        returnValue[i] += uniformRealDistributionForIntercept(mersenneTwisterGenerator);
    }
    return returnValue;
}
