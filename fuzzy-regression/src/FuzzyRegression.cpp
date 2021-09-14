#include "FuzzyRegression.hpp"

FuzzyRegression::FuzzyRegression(const ksi::dataset& dataset, int numberOfClusters, double epsilon)
: dataset(dataset),
numberOfClusters(numberOfClusters),
epsilon(epsilon){}

std::vector<double>
FuzzyRegression::processDataset() {
    ksi::fcm algorithm;
    algorithm.setEpsilonForFrobeniusNorm(epsilon);
    algorithm.setNumberOfClusters(numberOfClusters);
    auto partition = algorithm.doPartition(dataset);

    const std::vector<std::vector<double>> clusterCenters = partition.getClusterCentres();
    std::vector<double> clusterDescribedValues = getClustersDescribedValue(clusterCenters);
    std::vector<std::vector<double>> clusterDescribingValues = getClustersDescribingValues(clusterCenters);
    const std::vector<double> clusterWeights = getClusterWeightsFromPartition(partition);


    const std::vector<double>& fuzzyWeightedLinearRegressionResults = ksi::least_square_error_regression::weighted_linear_regression(
            clusterDescribingValues, clusterDescribedValues, clusterWeights);
    return fuzzyWeightedLinearRegressionResults;
}

std::vector<std::vector<double>>
FuzzyRegression::getClustersDescribingValues(const std::vector<std::vector<double>>& clusterCenters) const {
    std::vector<std::vector<double>> clusterDescribingValues;
    clusterDescribingValues.reserve(clusterCenters.size());
    for (const auto & singleClusterParameters : clusterCenters) {
        std::vector<double> singleClusterDescribingValues;
        unsigned long numberOfDescribingValuesInTuple = singleClusterParameters.size() - 1;
        singleClusterDescribingValues.reserve(numberOfDescribingValuesInTuple);
        for (int i = 0; i < numberOfDescribingValuesInTuple; ++i) {
            singleClusterDescribingValues.push_back(singleClusterParameters[i]);
        }
        clusterDescribingValues.push_back(singleClusterDescribingValues);
    }
    return clusterDescribingValues;
}

std::vector<double>
FuzzyRegression::getClustersDescribedValue(const std::vector<std::vector<double>>& clusterCenters) const {
    std::vector<double> clusterDescribedValues(numberOfClusters);
    for (int i = 0; i< numberOfClusters; ++i){
        double describedValue = clusterCenters[i].back();
        clusterDescribedValues[i] = describedValue;
    }
    return clusterDescribedValues;
}

std::vector<double>
FuzzyRegression::getClusterWeightsFromPartition(const ksi::partition& partition) const {
    std::vector<int> clusterAmounts(numberOfClusters);
    {
        std::vector<AssociationStruct> datumAssociationInfo =
                getPartitionAssociativityData(partition.getPartitionMatrix());
        for (const auto& associationInfo: datumAssociationInfo) {
            clusterAmounts[associationInfo.clusterNumber]++;
        }
    }
    const size_t datumElementsNumber = partition.getPartitionMatrix()[0].size();
    std::vector<double> clusterWeights;
    auto calculateProportionOfDatumsBelongingToCluster = [datumElementsNumber] (int numberOfAssociatedDatums) {
        return numberOfAssociatedDatums/(double) datumElementsNumber;
    };
    std::transform(clusterAmounts.begin(), clusterAmounts.end(), std::back_inserter(clusterWeights),
                   calculateProportionOfDatumsBelongingToCluster);
    return clusterWeights;
}

std::vector<AssociationStruct>
FuzzyRegression::getPartitionAssociativityData(const std::vector<std::vector<double>>& partitionMatrix) const {
    size_t elementCount = dataset.getNumberOfData();
    std::vector<AssociationStruct> datumClusterAssociationInfo(elementCount);

    const auto& firstClusterAssociationData = partitionMatrix[0];
    for (int i = 0; i < elementCount; ++i) {
        double firstClusterAssociationValue = firstClusterAssociationData[i];
        auto& initialPartitionAssociationValue = datumClusterAssociationInfo[i];
        initialPartitionAssociationValue.clusterAssociationValue = firstClusterAssociationValue;
    }

    for (int clusterNumber = 1; clusterNumber < numberOfClusters; ++clusterNumber) {
        auto clusterAssociationValues = partitionMatrix[clusterNumber];
        for (int elementNumber = 0; elementNumber < elementCount; ++elementNumber) {
            double associationValue = clusterAssociationValues[elementNumber];
            auto& datumAssociationInfo = datumClusterAssociationInfo[elementNumber];
            if (datumAssociationInfo.clusterAssociationValue < associationValue){
                datumAssociationInfo.clusterAssociationValue = associationValue;
                datumAssociationInfo.clusterNumber = clusterNumber;
            }
        }
    }
    return datumClusterAssociationInfo;
}
