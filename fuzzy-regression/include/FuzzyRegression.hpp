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

class FuzzyRegression{
    constexpr static const double EPSILON_DEFAULT_VALUE = 1e-8;
private:
    const ksi::dataset& dataset;
    const int numberOfClusters;
    const double epsilon = epsilon;
public:
    explicit FuzzyRegression(const ksi::dataset& dataset,
                             int numberOfClusters,
                             double epsilon = EPSILON_DEFAULT_VALUE);
    std::vector<double> processDataset();

private:
    [[nodiscard]]
    std::vector<AssociationStruct> getPartitionAssociativityData(const std::vector<std::vector<double>>& partitionMatrix) const;

    [[nodiscard]]
    std::vector<double> getClusterWeightsFromPartition(const ksi::partition& partition) const;

    std::vector<double> getClustersDescribedValue(const std::vector<std::vector<double>>& clusterCenters) const;

    std::vector<std::vector<double>>
    getClustersDescribingValues(const std::vector<std::vector<double>>& clusterCenters) const;
};


#endif // FUZZY_REGRESSION_FUZZYREGRESSION_HPP