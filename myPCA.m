%
% myPCA( data, numberOfFeatures )
%
% Arguments: data: is an MxN matrix, where M is the number of examples and
%            n is the number of digits
%            featuresToExtract: number of features to be extracted from the
%            data
%
% Returns:  pcaData: the output data from the PCA algrorithm 
%           eigVec: the eigenvectors from PCA
%           eigVal the eigenvalues from PCA
%

function [pcaData,projectionVectors,eigVal] = myPCA( data, featuresToExtract)
    

    % Check the arguments
    if ~exist('data', 'var')
        error('Data argument required.');
    end
    % Convert data to double
    data = double(data);
    
    % Get the number of features and examples of the data
    [numberOfExamples,numberOfFeatures] = size(data);
    
    if exist('featuresToExtract', 'var')
        if( featuresToExtract > numberOfFeatures)
            error('Number of features to extract is bigger than the features the data has');
        end
    else 
        featuresToExtract = round(numberOfFeatures / 2);
    end
    
    % Step 1: normalize the data
    
    % Get the mean for each feature on the data
    dataMean = mean(data,1);
    
    % Allocate the space for normalizedData
    normalizedData = zeros(numberOfExamples,numberOfFeatures);
    
    % For each example subtract the dataMean
    for i = 1 : numberOfExamples
        normalizedData(i,:) = data(i,:) - dataMean;
    end
    
    
    % Step 2: Find the covariance matrix of the normalized data 
    covarianceMatrix = cov(normalizedData);
    
    % Step 3: Calculate the eigenvectors and eigenvalues 
    [eigVec, eigVal] = eig(covarianceMatrix);
    % Get the eigenvalues 
    eigVal = diag(eigVal);
    % Find the best eigenvalues
    bestEigVal = sortrows(eigVal,-1);
    
    for i = 1 : featuresToExtract 
        projectionVectors(:,i) = eigVec(:,eigVal == bestEigVal(i));
    end
    
    eigVal = bestEigVal;
    % Step 4: Get the new data 
    pcaData = normalizedData * projectionVectors;
    
end
   