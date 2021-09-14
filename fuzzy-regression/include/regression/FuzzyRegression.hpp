#ifndef FUZZY_REGRESSION_FUZZYREGRESSION_HPP
#define FUZZY_REGRESSION_FUZZYREGRESSION_HPP

#include <filesystem>
#include <fstream>

#include <auxiliary/least-error-squares-regression.h>
#include <common/dataset.h>
#include <partitions/fcm.h>

struct AssociationStruct{
    int clusterNumber;
    double clusterAssociationValue;
};

struct RegressionResult{
    const std::vector<double> regressionDescribingParameters;
    const double coefficientOfDetermination;
};

class FuzzyRegression{
    constexpr static const double EPSILON_DEFAULT_VALUE = 1e-8;
private:
    const ksi::dataset& dataset;
    const int numberOfClusters;
    const double epsilon = epsilon;
public:
    FuzzyRegression(const ksi::dataset& dataset,
                    int numberOfClusters,
                    double epsilon = EPSILON_DEFAULT_VALUE);

    RegressionResult processDataset(std::ostream& performanceLoggingStream);

private:
    [[nodiscard]]
    std::vector<AssociationStruct>
    getPartitionAssociativityData(const std::vector<std::vector<double>>& partitionMatrix) const;

    [[nodiscard]]
    std::vector<double>
    getClusterWeightsFromPartition(const ksi::partition& partition) const;

    [[nodiscard]]
    std::vector<double>
    getClustersDescribedValue(const std::vector<std::vector<double>>& clusterCenters) const;

    [[nodiscard]]
    std::vector<std::vector<double>>
    getClustersDescribingValues(const std::vector<std::vector<double>>& clusterCenters) const;

    [[nodiscard]]
    double
    calculateRSquaredError(const std::vector<std::vector<double>>& rowsWithDescribingValues,
                           const std::vector<double>& describedValues,
                           const std::vector<double>& regressionCoefficients);
};


#endif // FUZZY_REGRESSION_FUZZYREGRESSION_HPP