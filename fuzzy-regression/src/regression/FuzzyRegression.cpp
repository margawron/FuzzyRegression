#include <chrono>

#include "regression/FuzzyRegression.hpp"

FuzzyRegression::FuzzyRegression(const ksi::dataset& dataset, int numberOfClusters, double epsilon)
: dataset(dataset),
numberOfClusters(numberOfClusters),
epsilon(epsilon){}

RegressionResult
FuzzyRegression::processDataset(std::ostream& performanceLoggingStream) {
    performanceLoggingStream << dataset.getNumberOfData() << ";"
                             << dataset.getNumberOfAttributes() << ";"
                             << numberOfClusters << ";";
    ksi::fcm algorithm;
    algorithm.setEpsilonForFrobeniusNorm(epsilon);
    algorithm.setNumberOfClusters(numberOfClusters);
    auto fcmStart = std::chrono::steady_clock::now();
    auto partition = algorithm.doPartition(dataset);
    auto fcmEnd = std::chrono::steady_clock::now();
    long fcmNanosecondDuration = std::chrono::duration_cast<std::chrono::nanoseconds>(fcmEnd - fcmStart).count();
    performanceLoggingStream << fcmNanosecondDuration << ";";

    auto dataPrepStart = std::chrono::steady_clock::now();
    const std::vector<std::vector<double>> clusterCenters = partition.getClusterCentres();
    std::vector<double> clusterDescribedValues = getClustersDescribedValue(clusterCenters);
    std::vector<std::vector<double>> clusterDescribingValues = getClustersDescribingValues(clusterCenters);
    const std::vector<double> clusterWeights = getClusterWeightsFromPartition(partition);
    auto dataPrepEnd = std::chrono::steady_clock::now();

    long dataPrepNanosecondDuration = std::chrono::duration_cast<std::chrono::nanoseconds>(dataPrepEnd - dataPrepStart).count();
    performanceLoggingStream << dataPrepNanosecondDuration << ";";

    auto fuzzyRegressionStart = std::chrono::steady_clock::now();
    const std::vector<double>& fuzzyRegressionCoefficients = ksi::least_square_error_regression::weighted_linear_regression(
            clusterDescribingValues, clusterDescribedValues, clusterWeights);
    auto fuzzyRegressionEnd = std::chrono::steady_clock::now();
    long fuzzyRegressionNanosecondDuration = std::chrono::duration_cast<std::chrono::nanoseconds>(fuzzyRegressionEnd- fuzzyRegressionStart).count();
    performanceLoggingStream << fuzzyRegressionNanosecondDuration << ";";

    auto rSquaredErrorStart = std::chrono::steady_clock::now();
    double regressionError = calculateRegressionError(clusterDescribingValues, clusterDescribedValues,
                                                      fuzzyRegressionCoefficients);
    auto rSquaredErrorEnd = std::chrono::steady_clock::now();
    long rSquaredNanosecondDuration = std::chrono::duration_cast<std::chrono::nanoseconds>(fuzzyRegressionEnd - fuzzyRegressionStart).count();
    performanceLoggingStream << rSquaredNanosecondDuration << "\n";

    return RegressionResult{fuzzyRegressionCoefficients, regressionError};
}

std::vector<std::vector<double>>
FuzzyRegression::getClustersDescribingValues(const std::vector<std::vector<double>>& clusterCenters) const {
    std::vector<std::vector<double>> clusterDescribingValues;
    clusterDescribingValues.reserve(numberOfClusters);
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

double FuzzyRegression::calculateRegressionError(const std::vector<std::vector<double>>& rowsWithDescribingValues,
                                                 const std::vector<double>& describedValues,
                                                 const std::vector<double>& regressionCoefficients) {
//    double averageOfDescribedValues = std::accumulate(describedValues.begin(), describedValues.end(), 0.0)/(double) describedValues.size();
    double sumOfLineErrors = 0;
//    double sumOfAvgErrors = 0;
    for (int i = 0; i < describedValues.size(); ++i) {
        const auto& describingValues = rowsWithDescribingValues[i];
        double predictedValue = 0;
        for (int j = 0; j < describingValues.size(); ++j) {
            predictedValue += regressionCoefficients[j] * describingValues[j];
        }
        double sqrdLineError = (describedValues[i] - predictedValue) *
                               (describedValues[i] - predictedValue);
//        double sqrdAvgError = (describedValues[i] - averageOfDescribedValues) *
//                              (describedValues[i] - averageOfDescribedValues);
        sumOfLineErrors += sqrdLineError;
//        sumOfAvgErrors += sqrdAvgError;
    }
//    double d = sumOfLineErrors / sumOfAvgErrors ;
//    return 1 - d;
    return sumOfLineErrors / (double)describedValues.size();
}